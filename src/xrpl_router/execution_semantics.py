from __future__ import annotations

import os
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from .amm import AMM, amount_add_ripple_number, amount_sub_ripple_number
from .book_offers import BookOfferSegment, _mul_amount_by_rate
from .book_step import (
    BookStep,
    RouterQuoteView,
    _apply_rate,
    _iteration_violates_limit_quality,
    _limit_out_by_quality_function,
    _normalise_rows,
    _sum_amounts_sorted,
)
from .core import (
    Amount,
    INT64_MAX,
    IOUAmount,
    IOU_EXP_MAX,
    IOU_EXP_MIN,
    IOU_MANTISSA_MAX,
    IOU_MANTISSA_MIN,
    Quality,
    XRPAmount,
)
from .core.datatypes import Segment
from .flow import PaymentSandbox
from .tx_intent import TxIntent

XRP = "XRP"
MAX_FLOW_ITERS = 1000
MAX_AMM_FLOW_ITERS = 30


def _is_xrp(cur: str) -> bool:
    return str(cur) == XRP


def _amt_zero(cur: str) -> Amount:
    return XRPAmount(0) if _is_xrp(cur) else IOUAmount(0, 0)


def _amt_add(left: Amount, right: Amount) -> Amount:
    return amount_add_ripple_number(left, right)


def _amt_sub(left: Amount, right: Amount) -> Amount:
    return amount_sub_ripple_number(left, right)


def amount_to_decimal(amount: Amount) -> Decimal:
    if isinstance(amount, XRPAmount):
        return Decimal(amount.drops) / Decimal(1_000_000)
    if amount.is_zero():
        return Decimal("0")
    return Decimal(amount.mantissa) * (Decimal(10) ** amount.exponent)


def _debug_amount(amount: Amount | None) -> str | None:
    if amount is None:
        return None
    return format(amount_to_decimal(amount), "f")


def _debug_quality(quality: Quality | None) -> str | None:
    if quality is None:
        return None
    rate = quality.rate()
    return format(Decimal(rate.mantissa) * (Decimal(10) ** rate.exponent), "f")


def _debug_quote_slices(quote: Dict[str, Any], *, in_cur: str, out_cur: str) -> List[Dict[str, Any]]:
    del in_cur, out_cur
    rows: List[Dict[str, Any]] = []
    for s in quote.get("slices", []) or []:
        if not isinstance(s, dict):
            continue
        rows.append(
            {
                "src": s.get("src"),
                "source_id": s.get("source_id"),
                "out": _debug_amount(s.get("out_take")),
                "in": _debug_amount(s.get("in_take")),
                "quality": _debug_quality(s.get("avg_quality")),
            }
        )
    return rows


def _amt_units_floor(a: Amount) -> int:
    if isinstance(a, XRPAmount):
        return a.drops if a.drops > 0 else 0
    m = a.mantissa
    e = a.exponent
    if m <= 0:
        return 0
    k = e + 15
    if k >= 0:
        return m * (10 ** k)
    div = 10 ** (-k)
    return m // div


def _amt_from_units(cur: str, units: int) -> Amount:
    if units <= 0:
        return _amt_zero(cur)
    if _is_xrp(cur):
        return XRPAmount(int(units))
    mantissa = int(units)
    exponent = -15
    while mantissa > INT64_MAX:
        mantissa //= 10
        exponent += 1
    return IOUAmount(mantissa, exponent)


def _amount_is_positive(amount: Amount) -> bool:
    if isinstance(amount, XRPAmount):
        return amount.drops > 0
    return amount.mantissa > 0


def _amount_lt(left: Amount, right: Amount) -> bool:
    if type(left) is not type(right):
        raise RuntimeError("amount domain mismatch")
    return left < right


def _cap_input_budget(requested: Amount, source_cap: Optional[Amount]) -> Amount:
    if source_cap is None:
        return requested
    return source_cap if _amount_lt(source_cap, requested) else requested


def _same_account(left: Optional[str], right: Optional[str]) -> bool:
    return bool(left) and bool(right) and left == right


def _offer_input_rate_for_bookstep(intent: TxIntent, rate: Optional[IOUAmount]) -> Optional[IOUAmount]:
    if rate is None or _is_xrp(intent.in_cur):
        return None
    # BookStep::rate(asset, strandDst) returns parity when the book input issuer
    # is the strand destination. In default OfferCreate crossing, strandDst is
    # the transaction account.
    if _same_account(intent.source_account, intent.offer_taker_gets_issuer):
        return None
    return rate


def _offer_output_owner_fee_applies(intent: TxIntent, rate: Optional[IOUAmount]) -> bool:
    if rate is None or _is_xrp(intent.out_cur):
        return False
    # Same BookStep::rate parity rule for the output side when ownerPaysTransferFee
    # is true: no owner fee if the output issuer is the strand destination.
    return not _same_account(intent.source_account, intent.offer_taker_pays_issuer)


def _offer_gateway_sendmax(requested: Amount, input_rate: Optional[IOUAmount]) -> Amount:
    # OfferCreate::flowCross inflates sendMax by the issuer transfer rate before
    # calling flow(); after crossing it divides the actual input back down when
    # computing the posted remainder.
    if input_rate is None:
        return requested
    return _mul_amount_by_rate(
        requested,
        input_rate,
        result_is_xrp=isinstance(requested, XRPAmount),
        round_up=True,
    )


def _payment_output_endpoint_rate(intent: TxIntent, rate: Optional[IOUAmount]) -> Optional[IOUAmount]:
    if rate is None or _is_xrp(intent.out_cur):
        return None
    # DirectStep::rate(asset, dstAccount) returns parity when the destination is
    # the issuer. Otherwise a final IOU endpoint step needs gross input from the
    # previous BookStep/AMM so the destination receives the requested net amount.
    if _same_account(intent.payment_destination_account, intent.payment_delivermax_issuer):
        return None
    return rate


def _payment_book_output_target(requested: Amount, output_rate: Optional[IOUAmount]) -> Amount:
    # DirectStep::rev() computes the amount required from the previous step as
    # output * transferRate, rounding up. Metadata on the book/AMM side exposes
    # this gross amount, not the destination's net delivered amount.
    return _apply_rate(requested, output_rate, round_up=True)


def _strictly_better_quality_floor(q: Quality) -> Quality:
    """Return the next strictly better quality in XRPL ordering."""
    r = q.rate()
    if r.is_zero():
        return Quality.from_rate(IOUAmount(IOU_MANTISSA_MIN, IOU_EXP_MIN))
    if r.mantissa > IOU_MANTISSA_MIN:
        return Quality.from_rate(IOUAmount(r.mantissa - 1, r.exponent))
    if r.exponent <= IOU_EXP_MIN:
        return Quality.from_rate(r)
    return Quality.from_rate(IOUAmount(IOU_MANTISSA_MAX, r.exponent - 1))


def _amt_from_units_like(proto: Amount, units: int) -> Amount:
    if units <= 0:
        return XRPAmount(0) if isinstance(proto, XRPAmount) else IOUAmount(0, 0)
    if isinstance(proto, XRPAmount):
        return XRPAmount(int(units))
    mantissa = int(units)
    exponent = -15
    while mantissa > INT64_MAX:
        mantissa //= 10
        exponent += 1
    return IOUAmount(mantissa, exponent)


def _sell_deliver_ceiling(out_cur: str) -> Amount:
    if _is_xrp(out_cur):
        return XRPAmount(INT64_MAX)
    # Match rippled OfferCreate::flowCross(tfSell): for IOU output, it does not
    # cap deliver to the nominal TakerPays. It opens the output side to a large
    # canonical ceiling and then lets flow()/limitQuality determine the actual
    # crossing, with the post-cross remainder reconstructed afterwards.
    return IOUAmount(IOU_MANTISSA_MAX // 2, IOU_EXP_MAX)


def _min_in_for_one_out_quantum_from_segment(seg: Segment) -> Optional[Amount]:
    out_u = _amt_units_floor(seg.out_max)
    in_u = _amt_units_floor(seg.in_at_out_max)
    if out_u <= 0 or in_u <= 0:
        return None
    # Minimum IN needed to obtain one minimum OUT quantum on this segment.
    in_need_u = (in_u + out_u - 1) // out_u
    if in_need_u <= 0:
        return None
    return _amt_from_units_like(seg.in_at_out_max, in_need_u)


def _min_executable_input_quantum(
    *,
    segs: List[Segment],
    amm: Optional[AMM],
    out_cur: str,
    limit_quality: Optional[Quality],
) -> Optional[Amount]:
    best: Optional[Amount] = None

    def _consider(cand: Optional[Amount]) -> None:
        nonlocal best
        if cand is None or _amt_units_floor(cand) <= 0:
            return
        if best is None or _amount_lt(cand, best):
            best = cand

    for s in segs:
        if limit_quality is not None and s.quality.rate() > limit_quality.rate():
            continue
        _consider(_min_in_for_one_out_quantum_from_segment(s))

    if amm is not None:
        one_out = XRPAmount(1) if _is_xrp(out_cur) else IOUAmount(1, -15)
        try:
            in_need = amm.swap_in_given_out_st(one_out)
        except Exception:
            in_need = None
        if in_need is not None and _amt_units_floor(in_need) > 0:
            try:
                q_slice = Quality.from_amounts(one_out, in_need)
            except Exception:
                q_slice = None
            if q_slice is not None and (
                limit_quality is None or q_slice.rate() <= limit_quality.rate()
            ):
                _consider(in_need)

    return best


def _normalise_src(src: str) -> str:
    s = (src or "").upper()
    if "AMM" in s:
        return "AMM"
    if "CLOB" in s or "BOOK" in s or "ORDER" in s:
        return "CLOB"
    return "?"


def _visible_clob_tier_count(
    segs: List[Segment],
    *,
    limit_quality: Optional[Quality],
) -> int:
    tiers = set()
    for seg in segs:
        if _normalise_src(getattr(seg, "src", "")) != "CLOB":
            continue
        if _amt_units_floor(seg.out_max) <= 0 or _amt_units_floor(seg.in_at_out_max) <= 0:
            continue
        if limit_quality is not None and seg.quality.rate() > limit_quality.rate():
            continue
        rate = seg.quality.rate()
        tiers.add((rate.mantissa, rate.exponent))
    return len(tiers)


def _offercreate_hybrid_requires_tier_anchor(
    *,
    segs: List[Segment],
    in_cur: str,
    limit_quality: Optional[Quality],
) -> bool:
    """Retain BookStep tier anchoring only in the hybrid regime that drifts without it."""
    if not _is_xrp(in_cur):
        return False
    return _visible_clob_tier_count(segs, limit_quality=limit_quality) > 1


def _amount_eq_or_one_quantum_short(
    actual: Amount,
    requested: Amount,
) -> bool:
    actual_units = _amt_units_floor(actual)
    requested_units = _amt_units_floor(requested)
    return actual_units >= requested_units or requested_units - actual_units <= 1


def _require_quote_totals(quote: Dict[str, Any]) -> Tuple[Dict[str, Any], Amount, Amount]:
    summary = quote.get("summary", {}) or {}
    total_out = summary.get("total_out")
    total_in = summary.get("total_in")
    if total_out is None or total_in is None:
        raise RuntimeError("invalid quote summary: missing total_out/total_in")
    return summary, total_out, total_in


def _offer_mode(base_mode: str, intent: TxIntent) -> str:
    mode = base_mode
    if intent.offer_is_ioc:
        mode = f"{mode}+ioc"
    if intent.offer_is_fok:
        mode = f"{mode}+fok"
    if intent.offer_is_sell:
        mode = f"{mode}+sell"
    if intent.offer_is_passive:
        mode = f"{mode}+passive"
    return mode


def _offer_posts_remainder(*, intent: TxIntent, is_partial_offer: bool) -> bool:
    return bool(is_partial_offer and (not intent.offer_is_ioc) and (not intent.offer_is_fok))


def _finalise_offer_crossing_result(
    quote: Dict[str, Any],
    *,
    summary: Dict[str, Any],
    intent: TxIntent,
    base_mode: str,
    is_partial_offer: bool,
    flow_meta: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    summary["is_partial"] = bool(is_partial_offer)
    quote["summary"] = summary

    if intent.offer_is_fok and is_partial_offer:
        raise RuntimeError("OfferCreate tfFillOrKill not fully filled")

    meta = {
        "mode": _offer_mode(base_mode, intent),
        "post_to_book_remainder": _offer_posts_remainder(
            intent=intent,
            is_partial_offer=is_partial_offer,
        ),
    }
    if flow_meta is not None:
        meta["rounds"] = flow_meta.get("rounds")
        meta["sendmax_binding"] = flow_meta.get("sendmax_binding")
        if flow_meta.get("debug_rounds") is not None:
            meta["debug_rounds"] = flow_meta.get("debug_rounds")
    return quote, meta


def _quote_from_forward_trace(
    *,
    trace: List[Dict[str, Any]],
    total_out: Amount,
    total_in: Amount,
    requested_out: Amount,
) -> Dict[str, Any]:
    slices: List[Dict[str, Any]] = []
    for t in trace or []:
        if not isinstance(t, dict):
            continue
        out_take = t.get("take_out")
        if out_take is None:
            out_take = t.get("out_take")
        in_take = t.get("take_in")
        if in_take is None:
            in_take = t.get("in_take")
        avg_q = t.get("quality")
        if avg_q is None:
            avg_q = t.get("avg_quality")
        slices.append(
            {
                "src": t.get("src"),
                "source_id": t.get("source_id"),
                "out_take": out_take,
                "in_take": in_take,
                "owner_out_take": t.get("owner_out_take"),
                "avg_quality": avg_q,
            }
        )
    avg_quality = None
    if (total_out is not None) and (total_in is not None) and (not total_out.is_zero()) and (not total_in.is_zero()):
        avg_quality = Quality.from_amounts(total_out, total_in)
    is_partial = not _amount_eq_or_one_quantum_short(total_out, requested_out)
    fill_ratio = None
    try:
        req_dec = amount_to_decimal(requested_out)
        out_dec = amount_to_decimal(total_out)
        fill_ratio = float(out_dec / req_dec) if req_dec != 0 else 0.0
    except Exception:
        fill_ratio = None
    return {
        "slices": slices,
        "anchors": [],
        "summary": {
            "total_out": total_out,
            "total_in": total_in,
            "avg_quality": avg_quality,
            "requested_out": requested_out,
            "is_partial": is_partial,
            "fill_ratio": fill_ratio,
        },
    }


def _segments_to_mutable_tiers(segs: List[Segment]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, s in enumerate(segs):
        if isinstance(s, BookOfferSegment):
            src = _normalise_src(getattr(s.segment, "src", ""))
        else:
            src = _normalise_src(getattr(s, "src", ""))
        if src != "CLOB":
            continue
        if isinstance(s, BookOfferSegment):
            sid = str(s.offer_id or f"__anon_clob_{idx}")
            out.append(
                {
                    "source_id": sid,
                    "out_amt": s.out_max,
                    "in_amt": s.in_at_out_max,
                    "owner": s.owner,
                    "owner_is_out_issuer": s.owner_is_out_issuer,
                    "owner_funds": s.owner_funds,
                    "in_transfer_rate": s.in_transfer_rate,
                    "out_transfer_rate": s.out_transfer_rate,
                    "quoted_quality": s.quoted_quality,
                    "non_fill_deleted": s.non_fill_deleted,
                    "book_visible_only": s.book_visible_only,
                }
            )
            continue
        sid = str(getattr(s, "source_id", "") or f"__anon_clob_{idx}")
        out.append(
            {
                "source_id": sid,
                "out_amt": s.out_max,
                "in_amt": s.in_at_out_max,
            }
        )
    return out


def _mutable_tiers_to_segments(
    mtiers: List[Dict[str, Any]],
    in_cur: str,
    out_cur: str,
    src: str = "CLOB",
) -> List[Segment | BookOfferSegment]:
    segs: List[Segment | BookOfferSegment] = []
    for t in mtiers:
        out_amt = t.get("out_amt")
        in_amt = t.get("in_amt")
        if out_amt is None or in_amt is None:
            continue
        if not _amount_is_positive(out_amt) or not _amount_is_positive(in_amt):
            continue
        q = t.get("quoted_quality")
        if not isinstance(q, Quality):
            q = Quality.from_amounts(out_amt, in_amt)
        segment = Segment(src=src, quality=q, out_max=out_amt, in_at_out_max=in_amt)
        owner = t.get("owner")
        owner_funds = t.get("owner_funds")
        in_transfer_rate = t.get("in_transfer_rate")
        out_transfer_rate = t.get("out_transfer_rate")
        if (
            owner is None
            and owner_funds is None
            and in_transfer_rate is None
            and out_transfer_rate is None
            and "owner_is_out_issuer" not in t
        ):
            segs.append(segment)
            continue
        segs.append(
            BookOfferSegment(
                offer_id=str(t.get("source_id")),
                quoted_quality=q,
                segment=segment,
                owner=None if owner is None else str(owner),
                owner_is_out_issuer=bool(t.get("owner_is_out_issuer")),
                owner_funds=owner_funds,
                in_transfer_rate=in_transfer_rate,
                out_transfer_rate=out_transfer_rate,
                non_fill_deleted=bool(t.get("non_fill_deleted")),
                book_visible_only=bool(t.get("book_visible_only")),
            )
        )
    return segs


def _consume_clob_slices_inplace(
    mtiers: List[Dict[str, Any]],
    quote: Dict[str, Any],
    *,
    in_cur: str,
    out_cur: str,
    owner_pays_transfer_fee: bool,
) -> None:
    def base_offer_id(value: Any) -> str:
        return str(value or "").split("#", 1)[0]

    def matching_tiers(sid: str) -> List[Dict[str, Any]]:
        base_sid = base_offer_id(sid)
        return [t for t in mtiers if base_offer_id(t.get("source_id")) == base_sid]

    def clear_visible_residuals(candidates: List[Dict[str, Any]]) -> None:
        for residual in candidates:
            if bool(residual.get("book_visible_only")):
                residual["out_amt"] = _amt_zero(out_cur)
                residual["in_amt"] = _amt_zero(in_cur)

    def clear_visible_residuals_if_executable_slice_depleted(sid: str) -> None:
        candidates = matching_tiers(sid)
        if not any(
            not bool(t.get("book_visible_only"))
            and _amt_units_floor(t.get("out_amt")) <= 0
            and _amt_units_floor(t.get("in_amt")) <= 0
            for t in candidates
        ):
            return
        clear_visible_residuals(candidates)

    for s in quote.get("slices", []) or []:
        if not isinstance(s, dict):
            continue
        src = str(s.get("src") or s.get("kind") or "")
        if src == "CLOB_DELETE":
            sid = str(s.get("source_id") or "")
            for t in matching_tiers(sid):
                t["out_amt"] = _amt_zero(out_cur)
                t["in_amt"] = _amt_zero(in_cur)
            continue
        if _normalise_src(src) != "CLOB":
            continue
        sid = str(s.get("source_id") or "")
        if not sid:
            continue
        out_take = s.get("out_take")
        if out_take is None:
            continue
        out_take_u = _amt_units_floor(out_take)
        if out_take_u <= 0:
            continue
        candidates = matching_tiers(sid)
        t = next(
            (
                cand
                for cand in candidates
                if not bool(cand.get("book_visible_only"))
                and _amt_units_floor(cand.get("out_amt")) > 0
                and _amt_units_floor(cand.get("in_amt")) > 0
            ),
            None,
        )
        if t is None:
            t = next(
                (
                    cand
                    for cand in candidates
                    if _amt_units_floor(cand.get("out_amt")) > 0
                    and _amt_units_floor(cand.get("in_amt")) > 0
                ),
                None,
            )
        if t is None:
            continue
        cur_out_amt = t.get("out_amt")
        cur_in_amt = t.get("in_amt")
        cur_out_u = _amt_units_floor(cur_out_amt)
        cur_in_u = _amt_units_floor(cur_in_amt)
        if cur_out_u <= 0 or cur_in_u <= 0:
            continue
        if out_take_u >= cur_out_u:
            t["out_amt"] = _amt_zero(out_cur)
            t["in_amt"] = _amt_zero(in_cur)
            if not bool(t.get("book_visible_only")):
                clear_visible_residuals(candidates)
            owner_funds = t.get("owner_funds")
            if owner_pays_transfer_fee and owner_funds is not None and not bool(t.get("owner_is_out_issuer")):
                owner_pays = _apply_rate(out_take, t.get("out_transfer_rate"), round_up=False)
                if type(owner_pays) is type(owner_funds):
                    t["owner_funds"] = (
                        _amt_zero(out_cur) if owner_pays >= owner_funds else owner_funds - owner_pays
                    )
                    if owner_pays >= owner_funds:
                        clear_visible_residuals(candidates)
            continue

        rem_out_u = cur_out_u - out_take_u
        rem_in_u = (rem_out_u * cur_in_u + cur_out_u - 1) // cur_out_u

        if rem_out_u <= 0 or rem_in_u <= 0:
            t["out_amt"] = _amt_zero(out_cur)
            t["in_amt"] = _amt_zero(in_cur)
            if not bool(t.get("book_visible_only")):
                clear_visible_residuals(candidates)
        else:
            t["out_amt"] = _amt_from_units(out_cur, rem_out_u)
            t["in_amt"] = _amt_from_units(in_cur, rem_in_u)
        owner_funds = t.get("owner_funds")
        if owner_pays_transfer_fee and owner_funds is not None and not bool(t.get("owner_is_out_issuer")):
            owner_pays = _apply_rate(out_take, t.get("out_transfer_rate"), round_up=False)
            if type(owner_pays) is type(owner_funds):
                t["owner_funds"] = (
                    _amt_zero(out_cur) if owner_pays >= owner_funds else _amt_sub(owner_funds, owner_pays)
                )
                if owner_pays >= owner_funds:
                    clear_visible_residuals(candidates)
        clear_visible_residuals_if_executable_slice_depleted(sid)


def _sum_amm_slice_amounts(
    quote: Dict[str, Any],
    *,
    in_cur: str,
    out_cur: str,
) -> Tuple[Amount, Amount]:
    total_in = _amt_zero(in_cur)
    total_out = _amt_zero(out_cur)
    for s in quote.get("slices", []) or []:
        if not isinstance(s, dict):
            continue
        src = str(s.get("src") or s.get("kind") or "")
        if _normalise_src(src) != "AMM":
            continue
        in_take = s.get("in_take")
        out_take = s.get("out_take")
        if in_take is not None and _amt_units_floor(in_take) > 0:
            total_in = _amt_add(total_in, in_take)
        if out_take is not None and _amt_units_floor(out_take) > 0:
            total_out = _amt_add(total_out, out_take)
    return total_in, total_out


def _rebuild_quote_summary(
    *,
    quote: Dict[str, Any],
    in_cur: str,
    out_cur: str,
) -> Dict[str, Any]:
    total_in = _amt_zero(in_cur)
    total_out = _amt_zero(out_cur)
    for s in quote.get("slices", []) or []:
        if not isinstance(s, dict):
            continue
        in_take = s.get("in_take")
        out_take = s.get("out_take")
        if in_take is not None and _amt_units_floor(in_take) > 0:
            total_in = _amt_add(total_in, in_take)
        if out_take is not None and _amt_units_floor(out_take) > 0:
            total_out = _amt_add(total_out, out_take)

    summary = dict(quote.get("summary", {}) or {})
    requested_out = summary.get("requested_out")
    summary["total_in"] = total_in
    summary["total_out"] = total_out
    summary["avg_quality"] = (
        Quality.from_amounts(total_out, total_in)
        if _amt_units_floor(total_in) > 0 and _amt_units_floor(total_out) > 0
        else None
    )
    if requested_out is not None:
        is_partial = not _amount_eq_or_one_quantum_short(total_out, requested_out)
        summary["is_partial"] = is_partial
        try:
            req_dec = amount_to_decimal(requested_out)
            out_dec = amount_to_decimal(total_out)
            summary["fill_ratio"] = float(out_dec / req_dec) if req_dec != 0 else 0.0
        except Exception:
            summary["fill_ratio"] = None
    quote["summary"] = summary
    return quote


def _collapse_tx_amm_settlement(
    *,
    quote: Dict[str, Any],
    amm: Optional[AMM],
    in_cur: str,
    out_cur: str,
) -> Dict[str, Any]:
    """Represent all AMM usage in one transaction as a single comparison leg.

    Hybrid replay may alternate AMM and CLOB multiple times. Rippled applies
    each AMM offer to the sandbox as it is consumed, so per-slice rounding is
    part of the ledger result. Metadata exposes the net AMM balance delta as
    one state change, so we merge the trace for comparison, but preserve the
    summed per-slice in/out amounts instead of repricing aggregate input once
    from transaction pre-state.
    """
    if amm is None:
        return quote

    slices = [s for s in (quote.get("slices") or []) if isinstance(s, dict)]
    amm_slices = [
        s
        for s in slices
        if _normalise_src(str(s.get("src") or s.get("kind") or "")) == "AMM"
        and s.get("in_take") is not None
        and _amt_units_floor(s["in_take"]) > 0
    ]
    if len(amm_slices) <= 1:
        return quote

    total_amm_in = _amt_zero(in_cur)
    total_amm_out = _amt_zero(out_cur)
    for s in amm_slices:
        total_amm_in = _amt_add(total_amm_in, s["in_take"])
        total_amm_out = _amt_add(total_amm_out, s["out_take"])
    if _amt_units_floor(total_amm_in) <= 0:
        return quote

    merged_amm = {
        "src": "AMM",
        "source_id": None,
        "out_take": total_amm_out,
        "in_take": total_amm_in,
        "avg_quality": (
            Quality.from_amounts(total_amm_out, total_amm_in)
            if _amt_units_floor(total_amm_out) > 0
            else None
        ),
    }

    clob_slices = [
        s
        for s in slices
        if _normalise_src(str(s.get("src") or s.get("kind") or "")) != "AMM"
    ]
    quote["slices"] = clob_slices + [merged_amm]
    return _rebuild_quote_summary(quote=quote, in_cur=in_cur, out_cur=out_cur)


def run_payment_constrained_quote(
    *,
    amm: Optional[AMM],
    segs: List[Segment],
    in_cur: str,
    out_cur: str,
    deliver_max_out: Amount,
    send_max_in: Optional[Amount],
    deliver_min_out: Optional[Amount],
    limit_quality: Optional[Quality] = None,
    source_account: str | None = None,
    execution_mode: str = "payment",
    book_in_transfer_rate: Optional[IOUAmount] = None,
    book_out_transfer_rate: Optional[IOUAmount] = None,
    charge_input_transfer_fee: bool = True,
    owner_pays_transfer_fee: bool = False,
    enable_self_cross_filter: bool = False,
    fix_amm_v1_1_enabled: bool = True,
    fix_reduced_offers_v2_enabled: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Single-path payment execution in reverse-first iterative mode.

    Mirrors the whitepaper flow for one path:
    1) reverse on remaining requested OUT,
    2) re-run forward if SendMax binds,
    3) commit round-level CLOB/AMM consumption on shadow state,
    4) iterate until request is fulfilled or liquidity/budget is exhausted.
    """
    mtiers = _segments_to_mutable_tiers(segs)
    working_amm = amm.clone() if amm is not None else None
    remaining_out = deliver_max_out
    remaining_in = send_max_in

    total_in = _amt_zero(in_cur)
    total_out = _amt_zero(out_cur)
    saved_ins: List[Amount] = []
    saved_outs: List[Amount] = []
    all_trace: List[Dict[str, Any]] = []

    sendmax_bound_any = False
    amm_iters = 0
    rounds = 0
    debug_rounds: List[Dict[str, Any]] | None = [] if os.environ.get("XRPL_ROUTER_DEBUG_FLOW") else None

    def _debug_break(reason: str, **fields: Any) -> None:
        if debug_rounds is None:
            return
        row: Dict[str, Any] = {
            "round": rounds,
            "break_reason": reason,
            "remaining_out_before": _debug_amount(remaining_out),
            "remaining_in_before": _debug_amount(remaining_in),
            "limit_quality": _debug_quality(limit_quality),
        }
        row.update(fields)
        debug_rounds.append(row)

    def _save_round_totals(round_in: Amount, round_out: Amount) -> None:
        nonlocal total_in, total_out
        # Match rippled StrandFlow: accumulate pass results from smallest to
        # largest instead of preserving execution order as a running sum.
        saved_ins.append(round_in)
        saved_outs.append(round_out)
        total_in = _sum_amounts_sorted(
            saved_ins,
            zero_like=send_max_in if send_max_in is not None else _amt_zero(in_cur),
        )
        total_out = _sum_amounts_sorted(saved_outs, zero_like=deliver_max_out)

    # Match rippled StrandFlow's outer liquidity-pass bound. A lower local cap
    # can prematurely stop AMM/CLOB alternation before the input budget is dry.
    while rounds < MAX_FLOW_ITERS and _amt_units_floor(remaining_out) > 0:
        if remaining_in is not None and _amt_units_floor(remaining_in) <= 0:
            _debug_break("remaining_input_exhausted")
            break

        cur_segs = _mutable_tiers_to_segments(mtiers, in_cur, out_cur, src="CLOB")
        round_amm = working_amm if amm_iters < MAX_AMM_FLOW_ITERS else None
        if not cur_segs and round_amm is None:
            _debug_break("no_clob_or_amm_liquidity")
            break

        if execution_mode == "offer_crossing" and limit_quality is not None:
            step = BookStep.from_static(
                cur_segs,
                amm=round_amm,
                limit_quality=limit_quality,
                mode=execution_mode,
                source_account=source_account,
                book_in_transfer_rate=book_in_transfer_rate,
                book_out_transfer_rate=book_out_transfer_rate,
                charge_input_transfer_fee=charge_input_transfer_fee,
                owner_pays_transfer_fee=owner_pays_transfer_fee,
                enable_self_cross_filter=enable_self_cross_filter,
                fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
                fix_reduced_offers_v2_enabled=fix_reduced_offers_v2_enabled,
            )
            quality_upper_bound_before = step.quality_upper_bound()
            if quality_upper_bound_before < limit_quality:
                _debug_break(
                    "quality_upper_bound_below_limit",
                    quality_upper_bound_before=_debug_quality(quality_upper_bound_before),
                )
                break
        else:
            quality_upper_bound_before = None

        limit_remaining_out = remaining_out
        adjusted_remaining_out = False
        if execution_mode == "offer_crossing" and limit_quality is not None:
            limit_remaining_out, adjusted_remaining_out = _limit_out_by_quality_function(
                _normalise_rows(cur_segs),
                amm=round_amm,
                remaining_out=remaining_out,
                limit_quality=limit_quality,
                mode=execution_mode,
                source_account=source_account,
                book_in_transfer_rate=book_in_transfer_rate,
                book_out_transfer_rate=book_out_transfer_rate,
                charge_input_transfer_fee=charge_input_transfer_fee,
                owner_pays_transfer_fee=owner_pays_transfer_fee,
                enable_self_cross_filter=enable_self_cross_filter,
                fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
            )

        view = RouterQuoteView(
            lambda cur=cur_segs: cur,
            amm=round_amm,
            limit_quality=limit_quality,
            mode=execution_mode,
            source_account=source_account,
            book_in_transfer_rate=book_in_transfer_rate,
            book_out_transfer_rate=book_out_transfer_rate,
            charge_input_transfer_fee=charge_input_transfer_fee,
            owner_pays_transfer_fee=owner_pays_transfer_fee,
            enable_self_cross_filter=enable_self_cross_filter,
            fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
            fix_reduced_offers_v2_enabled=fix_reduced_offers_v2_enabled,
            apply_quality_limit_out=False,
            enforce_limit_quality_result=False,
        )
        reverse_quote = view.preview_out(limit_remaining_out)
        reverse_summary = reverse_quote.get("summary", {}) or {}
        required_in = reverse_summary.get("total_in")
        reverse_out = reverse_summary.get("total_out")
        if required_in is None or reverse_out is None:
            raise RuntimeError("invalid reverse quote summary for payment-constrained execution")
        if _amt_units_floor(reverse_out) <= 0:
            if remaining_in is None:
                _debug_break(
                    "reverse_dry_without_input_budget",
                    quality_upper_bound_before=_debug_quality(quality_upper_bound_before),
                )
                break
            step = BookStep.from_static(
                cur_segs,
                amm=round_amm,
                limit_quality=limit_quality,
                mode=execution_mode,
                source_account=source_account,
                book_in_transfer_rate=book_in_transfer_rate,
                book_out_transfer_rate=book_out_transfer_rate,
                charge_input_transfer_fee=charge_input_transfer_fee,
                owner_pays_transfer_fee=owner_pays_transfer_fee,
                enable_self_cross_filter=enable_self_cross_filter,
                fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
                fix_reduced_offers_v2_enabled=fix_reduced_offers_v2_enabled,
                apply_quality_limit_out=False,
                enforce_limit_quality_result=False,
            )
            sb = PaymentSandbox()
            fwd_out, fwd_in, fwd_trace = step.fwd(sb, remaining_in, return_trace=True)
            if _amt_units_floor(fwd_out) <= 0 or _amt_units_floor(fwd_in) <= 0:
                _debug_break(
                    "forward_fallback_dry",
                    quality_upper_bound_before=_debug_quality(quality_upper_bound_before),
                    fwd_out=_debug_amount(fwd_out),
                    fwd_in=_debug_amount(fwd_in),
                )
                break
            if _amount_lt(remaining_out, fwd_out):
                _debug_break(
                    "forward_fallback_exceeds_remaining_out",
                    quality_upper_bound_before=_debug_quality(quality_upper_bound_before),
                    fwd_out=_debug_amount(fwd_out),
                    fwd_in=_debug_amount(fwd_in),
                )
                break
            round_quote = _quote_from_forward_trace(
                trace=fwd_trace,
                total_out=fwd_out,
                total_in=fwd_in,
                requested_out=remaining_out,
            )
            sendmax_bound_any = True
            round_summary = round_quote.get("summary", {}) or {}
            round_out = round_summary.get("total_out")
            round_in = round_summary.get("total_in")
            if round_out is None or round_in is None:
                raise RuntimeError("invalid forward fallback summary for payment-constrained execution")
            if _amt_units_floor(round_out) <= 0 or _amt_units_floor(round_in) <= 0:
                _debug_break(
                    "forward_fallback_round_empty",
                    quality_upper_bound_before=_debug_quality(quality_upper_bound_before),
                    round_out=_debug_amount(round_out),
                    round_in=_debug_amount(round_in),
                )
                break
            all_trace.extend(round_quote.get("slices", []) or [])
            _save_round_totals(round_in, round_out)

            _consume_clob_slices_inplace(
                mtiers,
                round_quote,
                in_cur=in_cur,
                out_cur=out_cur,
                owner_pays_transfer_fee=owner_pays_transfer_fee,
            )

            if round_amm is not None:
                amm_in_round, amm_out_round = _sum_amm_slice_amounts(round_quote, in_cur=in_cur, out_cur=out_cur)
                if _amt_units_floor(amm_in_round) > 0 and _amt_units_floor(amm_out_round) > 0:
                    round_amm.apply_fill_st(amm_in_round, amm_out_round)
                    amm_iters += 1

            if _amount_eq_or_one_quantum_short(total_out, deliver_max_out):
                remaining_out = _amt_zero(out_cur)
            else:
                remaining_out = _amt_sub(deliver_max_out, total_out)

            if send_max_in is not None:
                if not _amount_lt(total_in, send_max_in):
                    remaining_in = _amt_zero(in_cur)
                else:
                    remaining_in = _amt_sub(send_max_in, total_in)

            rounds += 1
            continue

        sendmax_binding_round = bool(remaining_in is not None and _amount_lt(remaining_in, required_in))
        sendmax_bound_any = sendmax_bound_any or sendmax_binding_round

        if not sendmax_binding_round:
            round_quote = reverse_quote
        else:
            step = BookStep.from_static(
                cur_segs,
                amm=round_amm,
                limit_quality=limit_quality,
                mode=execution_mode,
                source_account=source_account,
                book_in_transfer_rate=book_in_transfer_rate,
                book_out_transfer_rate=book_out_transfer_rate,
                charge_input_transfer_fee=charge_input_transfer_fee,
                owner_pays_transfer_fee=owner_pays_transfer_fee,
                enable_self_cross_filter=enable_self_cross_filter,
                fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
                fix_reduced_offers_v2_enabled=fix_reduced_offers_v2_enabled,
                apply_quality_limit_out=False,
                enforce_limit_quality_result=False,
            )
            # Rippled re-executes the same limiting BookStep with fwd(maxIn)
            # after rev(remainingOut) has populated the step cache. The forward
            # pass is therefore anchored to the reverse result, not the full
            # remaining payment target.
            step._cached_out = reverse_out
            sb = PaymentSandbox()
            fwd_out, fwd_in, fwd_trace = step.fwd(sb, remaining_in, return_trace=True)
            round_quote = _quote_from_forward_trace(
                trace=fwd_trace,
                total_out=fwd_out,
                total_in=fwd_in,
                requested_out=reverse_out,
            )

        round_summary = round_quote.get("summary", {}) or {}
        round_out = round_summary.get("total_out")
        round_in = round_summary.get("total_in")
        if round_out is None or round_in is None:
            raise RuntimeError("invalid round quote summary for payment-constrained execution")
        if _amt_units_floor(round_out) <= 0 or _amt_units_floor(round_in) <= 0:
            _debug_break(
                "round_empty",
                quality_upper_bound_before=_debug_quality(quality_upper_bound_before),
                reverse_out=_debug_amount(reverse_out),
                reverse_required_in=_debug_amount(required_in),
                round_out=_debug_amount(round_out),
                round_in=_debug_amount(round_in),
            )
            break
        round_quality = (
            Quality.from_amounts(round_out, round_in)
            if limit_quality is not None
            else None
        )
        if round_quality is not None and _iteration_violates_limit_quality(
            mode=execution_mode,
            limit_quality=limit_quality,
            itr_quality=round_quality,
            adjusted_out=adjusted_remaining_out,
            amm_in=_amt_zero(in_cur),
            amm_out=_amt_zero(out_cur),
        ):
            _debug_break(
                "round_quality_below_limit",
                quality_upper_bound_before=_debug_quality(quality_upper_bound_before),
                limit_remaining_out=_debug_amount(limit_remaining_out),
                adjusted_remaining_out=adjusted_remaining_out,
                reverse_out=_debug_amount(reverse_out),
                reverse_required_in=_debug_amount(required_in),
                round_out=_debug_amount(round_out),
                round_in=_debug_amount(round_in),
                round_quality=_debug_quality(round_quality),
            )
            break
        if debug_rounds is not None:
            debug_rounds.append(
                {
                    "round": rounds,
                    "remaining_out_before": _debug_amount(remaining_out),
                    "remaining_in_before": _debug_amount(remaining_in),
                    "limit_remaining_out": _debug_amount(limit_remaining_out),
                    "adjusted_remaining_out": adjusted_remaining_out,
                    "reverse_out": _debug_amount(reverse_out),
                    "reverse_required_in": _debug_amount(required_in),
                    "sendmax_binding": sendmax_binding_round,
                    "round_out": _debug_amount(round_out),
                    "round_in": _debug_amount(round_in),
                    "round_quality": _debug_quality(round_quality),
                    "quality_upper_bound_before": _debug_quality(quality_upper_bound_before),
                    "limit_quality": _debug_quality(limit_quality),
                    "slices": _debug_quote_slices(round_quote, in_cur=in_cur, out_cur=out_cur),
                }
            )

        all_trace.extend(round_quote.get("slices", []) or [])
        _save_round_totals(round_in, round_out)

        _consume_clob_slices_inplace(
            mtiers,
            round_quote,
            in_cur=in_cur,
            out_cur=out_cur,
            owner_pays_transfer_fee=owner_pays_transfer_fee,
        )

        if round_amm is not None:
            amm_in_round, amm_out_round = _sum_amm_slice_amounts(round_quote, in_cur=in_cur, out_cur=out_cur)
            if _amt_units_floor(amm_in_round) > 0 and _amt_units_floor(amm_out_round) > 0:
                round_amm.apply_fill_st(amm_in_round, amm_out_round)
                amm_iters += 1

        if _amount_eq_or_one_quantum_short(total_out, deliver_max_out):
            remaining_out = _amt_zero(out_cur)
        else:
            remaining_out = _amt_sub(deliver_max_out, total_out)

        if remaining_in is not None and send_max_in is not None:
            # Align with rippled StrandFlow outer-loop budgeting: keep iterating
            # while there is still strictly positive remaining input budget.
            # Collapsing a 1-quantum remainder to zero here can incorrectly
            # suppress a final 1-drop payment tail on the next round.
            if not _amount_lt(total_in, send_max_in):
                remaining_in = _amt_zero(in_cur)
            else:
                remaining_in = _amt_sub(send_max_in, total_in)

        rounds += 1

    if deliver_min_out is not None and not _amount_eq_or_one_quantum_short(total_out, deliver_min_out):
        if sendmax_bound_any:
            raise RuntimeError("forward quote under SendMax is below DeliverMin")
        raise RuntimeError("reverse quote cannot satisfy DeliverMin")

    execution_total_in = total_in
    execution_total_out = total_out
    quote = _quote_from_forward_trace(
        trace=all_trace,
        total_out=execution_total_out,
        total_in=execution_total_in,
        requested_out=deliver_max_out,
    )
    quote["endpoint_slices"] = list(quote.get("slices") or [])
    quote = _collapse_tx_amm_settlement(
        quote=quote,
        amm=amm.clone() if amm is not None else None,
        in_cur=in_cur,
        out_cur=out_cur,
    )
    return quote, {
        "mode": execution_mode,
        "sendmax_binding": sendmax_bound_any,
        "required_in": None,
        "deliver_max_out": deliver_max_out,
        "deliver_min_out": deliver_min_out,
        "rounds": rounds,
        "execution_total_in": execution_total_in,
        "execution_total_out": execution_total_out,
        "debug_rounds": debug_rounds,
    }


def execute_single_path_intent(
    *,
    intent: TxIntent,
    amm: Optional[AMM],
    segs: List[Segment],
    in_cur: str,
    out_cur: str,
    book_in_transfer_rate: Optional[IOUAmount] = None,
    book_out_transfer_rate: Optional[IOUAmount] = None,
    fix_amm_v1_1_enabled: bool = True,
    fix_reduced_offers_v2_enabled: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute metadata intent under one-path whitepaper semantics.

    Payment and OfferCreate are dispatched with their own constraints, rather
    than forcing both through one Payment-only rule set.
    """
    tx_type = str(intent.transaction_type or "").upper()
    if tx_type == "PAYMENT":
        if intent.payment_sendmax_amt is None or intent.payment_delivermax_amt is None:
            raise RuntimeError("Payment intent missing SendMax/DeliverMax")
        if bool(intent.payment_no_ripple_direct):
            if not bool(intent.payment_paths_present):
                raise RuntimeError("Payment tfNoRippleDirect requires explicit Paths in single-path mode")
            if bool(intent.payment_paths_has_external_asset):
                raise RuntimeError("Payment tfNoRippleDirect with external path assets is out of single-path scope")
        effective_sendmax = _cap_input_budget(intent.payment_sendmax_amt, intent.source_funds_cap_amt)
        endpoint_out_rate = _payment_output_endpoint_rate(intent, book_out_transfer_rate)
        book_deliver_max_out = _payment_book_output_target(
            intent.payment_delivermax_amt,
            endpoint_out_rate,
        )
        book_deliver_min_out = (
            _payment_book_output_target(intent.payment_delivermin_amt, endpoint_out_rate)
            if intent.payment_delivermin_amt is not None
            else None
        )
        quote, meta = run_payment_constrained_quote(
            amm=amm,
            segs=segs,
            in_cur=in_cur,
            out_cur=out_cur,
            deliver_max_out=book_deliver_max_out,
            send_max_in=effective_sendmax,
            deliver_min_out=book_deliver_min_out,
            limit_quality=intent.payment_limit_quality,
            source_account=intent.source_account,
            execution_mode="payment",
            book_in_transfer_rate=book_in_transfer_rate,
            book_out_transfer_rate=book_out_transfer_rate,
            owner_pays_transfer_fee=False,
            enable_self_cross_filter=False,
            fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
            fix_reduced_offers_v2_enabled=fix_reduced_offers_v2_enabled,
        )
        summary, total_out_amt, total_in_amt = _require_quote_totals(quote)
        if _amt_units_floor(total_out_amt) <= 0 or _amt_units_floor(total_in_amt) <= 0:
            raise RuntimeError("model returned empty execution")
        if bool(summary.get("is_partial")):
            if not bool(intent.payment_partial_allowed):
                raise RuntimeError("Payment partial fill without tfPartialPayment")
            if book_deliver_min_out is not None:
                if _amount_lt(total_out_amt, book_deliver_min_out):
                    raise RuntimeError("Model partial fill below DeliverMin")
        payment_meta: Dict[str, Any] = {"mode": str(meta.get("mode") or "payment")}
        for key in (
            "rounds",
            "sendmax_binding",
            "debug_rounds",
        ):
            if meta.get(key) is not None:
                payment_meta[key] = meta.get(key)
        return quote, payment_meta

    if tx_type == "OFFERCREATE":
        if intent.offer_taker_gets_amt is None or intent.offer_taker_pays_amt is None:
            raise RuntimeError("OfferCreate intent missing TakerGets/TakerPays")
        if intent.offer_limit_quality is None:
            raise RuntimeError("OfferCreate intent missing limit quality")
        offer_input_rate = _offer_input_rate_for_bookstep(intent, book_in_transfer_rate)
        offer_owner_pays_transfer_fee = _offer_output_owner_fee_applies(intent, book_out_transfer_rate)
        gateway_sendmax = _offer_gateway_sendmax(intent.offer_taker_gets_amt, offer_input_rate)
        effective_in_cap = _cap_input_budget(gateway_sendmax, intent.source_funds_cap_amt)
        base_limit_quality = Quality.from_amounts(intent.offer_taker_pays_amt, gateway_sendmax)
        effective_limit_quality = (
            _strictly_better_quality_floor(base_limit_quality)
            if bool(intent.offer_is_passive)
            else base_limit_quality
        )
        # OfferCreate does not inherit Payment's post-SPQ anchor globally.
        # Keep the stricter tier anchor only for XRP->IOU hybrid books with
        # multiple visible CLOB tiers, where a relaxed AMM-first slice can
        # skip ahead of future tiers and poison rolling tx-prebook replay.
        if intent.offer_is_sell:
            quote, meta = run_payment_constrained_quote(
                amm=amm,
                segs=segs,
                in_cur=in_cur,
                out_cur=out_cur,
                deliver_max_out=_sell_deliver_ceiling(out_cur),
                send_max_in=effective_in_cap,
                deliver_min_out=None,
                limit_quality=effective_limit_quality,
                source_account=intent.source_account,
                execution_mode="offer_crossing",
                book_in_transfer_rate=offer_input_rate,
                book_out_transfer_rate=book_out_transfer_rate,
                charge_input_transfer_fee=offer_input_rate is not None,
                owner_pays_transfer_fee=offer_owner_pays_transfer_fee,
                enable_self_cross_filter=True,
                fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
                fix_reduced_offers_v2_enabled=fix_reduced_offers_v2_enabled,
            )
            mode = "offer-crossing-sell"
        else:
            # Non-sell OfferCreate crosses under TakerGets input budget toward
            # TakerPays output target with limit quality applied.
            quote, meta = run_payment_constrained_quote(
                amm=amm,
                segs=segs,
                in_cur=in_cur,
                out_cur=out_cur,
                deliver_max_out=intent.offer_taker_pays_amt,
                send_max_in=effective_in_cap,
                deliver_min_out=None,
                limit_quality=effective_limit_quality,
                source_account=intent.source_account,
                execution_mode="offer_crossing",
                book_in_transfer_rate=offer_input_rate,
                book_out_transfer_rate=book_out_transfer_rate,
                charge_input_transfer_fee=offer_input_rate is not None,
                owner_pays_transfer_fee=offer_owner_pays_transfer_fee,
                enable_self_cross_filter=True,
                fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
                fix_reduced_offers_v2_enabled=fix_reduced_offers_v2_enabled,
            )
            mode = "offer-crossing-buy"

        summary, total_out_amt, total_in_amt = _require_quote_totals(quote)
        out_units = _amt_units_floor(total_out_amt)
        in_units = _amt_units_floor(total_in_amt)
        if (out_units <= 0) != (in_units <= 0):
            raise RuntimeError("invalid quote summary: inconsistent empty execution")
        if out_units <= 0:
            # OfferCreate can validly have zero crossing and still succeed:
            # - IOC: no post
            # - non-IOC/non-FOK: remainder posts to book
            # - FOK: reject (not fully filled)
            return _finalise_offer_crossing_result(
                quote,
                summary=summary,
                intent=intent,
                base_mode=mode,
                is_partial_offer=True,
                flow_meta=meta,
            )

        # OfferCreate completion semantics differ by tfSell:
        # - tfSell: order is complete when TakerGets (sell-side cap) is fully consumed.
        # - non-sell: order is complete only when the full TakerPays target is acquired.
        if intent.offer_is_sell:
            execution_total_in_amt = meta.get("execution_total_in")
            if execution_total_in_amt is None:
                execution_total_in_amt = total_in_amt
            if _amount_eq_or_one_quantum_short(execution_total_in_amt, effective_in_cap):
                residual_in_amt = _amt_zero(in_cur)
            else:
                residual_in_amt = _amt_sub(effective_in_cap, execution_total_in_amt)
            min_exec_in = _min_executable_input_quantum(
                segs=segs,
                amm=amm,
                out_cur=out_cur,
                limit_quality=effective_limit_quality,
            )
            exhausted_budget = _amt_units_floor(residual_in_amt) <= 0
            if not exhausted_budget:
                if min_exec_in is None:
                    exhausted_budget = True
                else:
                    exhausted_budget = (
                        _amt_units_floor(residual_in_amt) < _amt_units_floor(min_exec_in)
                    )
            is_partial_offer = not exhausted_budget
        else:
            reached_target = _amount_eq_or_one_quantum_short(
                total_out_amt,
                intent.offer_taker_pays_amt,
            )
            is_partial_offer = not reached_target

        # Normalise summary to OfferCreate semantics (instead of Payment-style DeliverMax partial).
        return _finalise_offer_crossing_result(
            quote,
            summary=summary,
            intent=intent,
            base_mode=mode,
            is_partial_offer=is_partial_offer,
            flow_meta=meta,
        )

    raise RuntimeError(f"unsupported transaction type in execution: {tx_type or 'unknown'}")


__all__ = [
    "execute_single_path_intent",
    "run_payment_constrained_quote",
    "run_offer_crossing_sell_quote",
]
