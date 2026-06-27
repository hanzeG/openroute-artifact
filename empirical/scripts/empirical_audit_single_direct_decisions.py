#!/usr/bin/env python3
"""Emit compact per-round decision traces for selected isolated direct samples.

This is intentionally a small-tx audit tool. It reuses the fit pipeline inputs
but only for explicitly requested hashes, so it does not scan or materialize
large result sets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import empirical.scripts.empirical_fit_single_direct as fit
import xrpl_router.execution_semantics as ex
from xrpl_router.amm import AMM
from xrpl_router.book_step import BookStep
from xrpl_router.core import Amount, IOUAmount, Quality, XRPAmount
from xrpl_router.flow import PaymentSandbox
from xrpl_router.tx_intent import TxIntent, apply_offer_tick_size_to_intent


DEFAULT_DATASET = (
    ROOT
    / "artifacts/fit_inputs/rlusd_xrp/ledger_100725835_101035981/strict_direct_targets/two_week_isolated"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-ndjson", default=str(DEFAULT_DATASET / "tx_metadata_full_merged.ndjson"))
    parser.add_argument("--tx-prebook-snapshots", default=str(DEFAULT_DATASET / "tx_prebook_snapshots.ndjson"))
    parser.add_argument("--amm-swaps", default=str(DEFAULT_DATASET / "amm_swaps_two_week.parquet"))
    parser.add_argument("--amm-fees", default=str(DEFAULT_DATASET / "amm_fees_two_week.parquet"))
    parser.add_argument("--base-currency", default=fit.TARGET_IOU_CURRENCY)
    parser.add_argument("--base-label", default=fit.TARGET_IOU_LABEL)
    parser.add_argument("--reports-dir", default=None, help="Optional existing fit reports dir for model/real comparison legs.")
    parser.add_argument("--tx-hash", action="append", default=[], help="Transaction hash to audit. May be repeated.")
    parser.add_argument("--tx-file", default=None, help="Plain text file of transaction hashes.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--top-n", type=int, default=8, help="Number of top CLOB rows to include per round.")
    parser.add_argument("--max-rounds", type=int, default=32, help="Safety cap for audited model rounds.")
    return parser.parse_args()


def _amount_decimal(amount: Amount | None) -> str | None:
    if amount is None:
        return None
    return str(ex.amount_to_decimal(amount))


def _amount_payload(amount: Amount | None) -> dict[str, Any] | None:
    if amount is None:
        return None
    if isinstance(amount, XRPAmount):
        return {"kind": "XRP", "drops": amount.drops, "value": _amount_decimal(amount)}
    if isinstance(amount, IOUAmount):
        return {
            "kind": "IOU",
            "mantissa": amount.mantissa,
            "exponent": amount.exponent,
            "value": _amount_decimal(amount),
        }
    return {"kind": type(amount).__name__, "value": str(amount)}


def _amount_from_report(raw: dict[str, Any] | None, *, field_name: str) -> Amount | None:
    if raw is None:
        return None
    kind = str(raw.get("kind") or raw.get("currency") or "")
    if kind == "XRP":
        return XRPAmount(int(raw["drops"]))
    if kind == "IOU" or "mantissa" in raw:
        return IOUAmount(int(raw["mantissa"]), int(raw["exponent"]))
    value = raw.get("value")
    currency = str(raw.get("currency") or "")
    if currency == "XRP":
        return fit.parse_xrp_decimal(value, field_name=field_name)
    return fit.parse_iou_decimal(value, field_name=field_name)


def _quality_from_report_offer_amounts(intent_payload: dict[str, Any]) -> Quality | None:
    in_amt = _amount_from_report(intent_payload.get("taker_gets"), field_name="report.taker_gets")
    out_amt = _amount_from_report(intent_payload.get("taker_pays"), field_name="report.taker_pays")
    if in_amt is None or out_amt is None:
        return None
    return Quality.from_amounts(out_amt, in_amt)


def _intent_from_report(payload: dict[str, Any]) -> TxIntent:
    tx_type = str(payload.get("transaction_type") or "")
    source_funds_cap = _amount_from_report(payload.get("source_funds_cap"), field_name="report.source_funds_cap")
    if tx_type.upper() == "OFFERCREATE":
        return TxIntent(
            transaction_type=tx_type,
            flags=int(payload.get("flags") or 0),
            source_account=payload.get("source_account"),
            in_cur=str(payload["in_cur"]),
            out_cur=str(payload["out_cur"]),
            source_funds_cap_amt=source_funds_cap,
            offer_taker_gets_amt=_amount_from_report(payload.get("taker_gets"), field_name="report.taker_gets"),
            offer_taker_gets_currency=str(payload["in_cur"]),
            offer_taker_gets_issuer=None,
            offer_taker_pays_amt=_amount_from_report(payload.get("taker_pays"), field_name="report.taker_pays"),
            offer_taker_pays_currency=str(payload["out_cur"]),
            offer_taker_pays_issuer=None,
            offer_limit_quality=_quality_from_report_offer_amounts(payload),
            offer_is_sell=bool(payload.get("is_sell")),
            offer_is_ioc=bool(payload.get("is_ioc")),
            offer_is_fok=bool(payload.get("is_fok")),
            offer_is_passive=bool(payload.get("is_passive")),
            offer_cancel_sequence=payload.get("cancel_sequence"),
        )
    if tx_type.upper() == "PAYMENT":
        return TxIntent(
            transaction_type=tx_type,
            flags=int(payload.get("flags") or 0),
            source_account=payload.get("source_account"),
            in_cur=str(payload["in_cur"]),
            out_cur=str(payload["out_cur"]),
            source_funds_cap_amt=source_funds_cap,
            payment_sendmax_amt=_amount_from_report(payload.get("sendmax"), field_name="report.sendmax"),
            payment_sendmax_currency=str(payload["in_cur"]),
            payment_sendmax_issuer=None,
            payment_delivermax_amt=_amount_from_report(payload.get("delivermax"), field_name="report.delivermax"),
            payment_delivermax_currency=str(payload["out_cur"]),
            payment_delivermax_issuer=None,
            payment_delivermin_amt=_amount_from_report(payload.get("delivermin"), field_name="report.delivermin"),
            payment_partial_allowed=bool(payload.get("partial_allowed")),
            payment_no_ripple_direct=bool(payload.get("no_ripple_direct")),
            payment_limit_quality=None,
            payment_paths_present=bool(payload.get("paths_present")),
            payment_paths_count=int(payload.get("paths_count") or 0),
            payment_paths_has_external_asset=bool(payload.get("paths_has_external_asset")),
        )
    raise RuntimeError(f"unsupported report tx type: {tx_type}")


def _parse_report_reserve(value: str, *, currency: str, field_name: str) -> Amount:
    if currency == "XRP":
        return fit.parse_xrp_decimal(value, field_name=field_name)
    return fit.parse_iou_decimal(value, field_name=field_name)


def _amm_from_report_state(
    *,
    amm_state: dict[str, Any] | None,
    in_cur: str,
    out_cur: str,
) -> tuple[AMM | None, dict[str, Any] | None]:
    if not amm_state:
        return None, None
    asset_a = str(amm_state["amm_asset_currency"])
    asset_b = str(amm_state["amm_asset2_currency"])
    reserve_a = _parse_report_reserve(
        str(amm_state["reserve_a_before_used"]),
        currency=asset_a,
        field_name="report.reserve_a_before_used",
    )
    reserve_b = _parse_report_reserve(
        str(amm_state["reserve_b_before_used"]),
        currency=asset_b,
        field_name="report.reserve_b_before_used",
    )
    if in_cur == asset_a and out_cur == asset_b:
        x_reserve, y_reserve = reserve_a, reserve_b
        x_is_xrp, y_is_xrp = asset_a == "XRP", asset_b == "XRP"
    elif in_cur == asset_b and out_cur == asset_a:
        x_reserve, y_reserve = reserve_b, reserve_a
        x_is_xrp, y_is_xrp = asset_b == "XRP", asset_a == "XRP"
    else:
        raise RuntimeError(f"report AMM pool does not match intent: pool={asset_a}/{asset_b}, intent={in_cur}->{out_cur}")
    return (
        AMM(
            x_reserve=x_reserve,
            y_reserve=y_reserve,
            fee=str(amm_state["effective_fee"]),
            x_is_xrp=x_is_xrp,
            y_is_xrp=y_is_xrp,
        ),
        amm_state,
    )


def _quality_payload(quality: Quality | None, *, limit_quality: Quality | None = None) -> dict[str, Any] | None:
    if quality is None:
        return None
    payload: dict[str, Any] = {
        "raw": quality.value,
        "rate": _amount_payload(quality.rate()),
    }
    if limit_quality is not None:
        payload["raw_minus_limit"] = quality.value - limit_quality.value
        payload["gte_limit"] = quality >= limit_quality
        payload["lt_limit"] = quality < limit_quality
    return payload


def _safe_amount_lt(left: Amount | None, right: Amount | None) -> bool | None:
    if left is None or right is None:
        return None
    try:
        return ex._amount_lt(left, right)
    except RuntimeError:
        return None


def _slice_payload(row: dict[str, Any], *, limit_quality: Quality | None) -> dict[str, Any]:
    quality = row.get("quality") or row.get("avg_quality")
    return {
        "src": row.get("src"),
        "source_id": row.get("source_id"),
        "in": _amount_payload(row.get("in_take")),
        "out": _amount_payload(row.get("out_take")),
        "quality": _quality_payload(quality, limit_quality=limit_quality) if isinstance(quality, Quality) else None,
    }


def _top_rows_payload(
    rows: list[Any],
    *,
    intent: TxIntent,
    non_fill_deleted_offer_ids: set[str],
    limit_quality: Quality | None,
    top_n: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows[:top_n]:
        offer_id = str(getattr(row, "offer_id", "") or "")
        owner = str(getattr(row, "owner", "") or "")
        out.append(
            {
                "offer_id": offer_id,
                "owner": owner,
                "is_source_account": bool(owner and owner == str(intent.source_account or "")),
                "is_non_fill_deleted": offer_id in non_fill_deleted_offer_ids,
                "in": _amount_payload(getattr(row, "in_at_out_max", None)),
                "out": _amount_payload(getattr(row, "out_max", None)),
                "quality": _quality_payload(getattr(row, "quality", None), limit_quality=limit_quality),
                "quoted_quality": _quality_payload(getattr(row, "quoted_quality", None), limit_quality=limit_quality),
                "owner_funds": _amount_payload(getattr(row, "owner_funds", None)),
            }
        )
    return out


def _report_leg_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    return {
        "comparison": report.get("comparison"),
        "model_legs": [
            {
                "src": leg.get("src"),
                "id": leg.get("source_id") or leg.get("offer_id") or leg.get("amm_account"),
                "in": leg.get("in"),
                "out": leg.get("out"),
                "quality": leg.get("quality"),
            }
            for leg in report.get("model", {}).get("legs", [])
        ],
        "real_legs": [
            {
                "src": leg.get("src"),
                "id": leg.get("source_id") or leg.get("offer_id") or leg.get("amm_account"),
                "in": leg.get("in"),
                "out": leg.get("out"),
                "quality": leg.get("quality"),
            }
            for leg in report.get("real", {}).get("legs", [])
        ],
    }


def _load_hashes(args: argparse.Namespace) -> list[str]:
    hashes = [fit._normalise_tx_hash(v) for v in args.tx_hash]
    if args.tx_file:
        for line in Path(args.tx_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                hashes.append(fit._normalise_tx_hash(line))
    deduped: list[str] = []
    seen: set[str] = set()
    for h in hashes:
        if h and h not in seen:
            deduped.append(h)
            seen.add(h)
    if not deduped:
        raise RuntimeError("no tx hashes supplied")
    return deduped


def _build_sample_context(
    *,
    tx_hashes: list[str],
    metadata_path: Path,
    snapshots_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    samples = fit._load_metadata_samples(metadata_path=metadata_path, target_tx_hashes=set(tx_hashes))
    ordered_targets = [
        (tx_hash, fit._prebook_primary_side_for_output_currency(samples[tx_hash].intent.out_cur))
        for tx_hash in tx_hashes
    ]
    snapshots = {
        tx_hash: payload
        for tx_hash, payload in fit._iter_snapshot_rows_in_sample_order(
            snapshots_path=snapshots_path,
            ordered_targets=ordered_targets,
        )
    }
    return samples, snapshots


def _build_report_context(
    *,
    tx_hashes: list[str],
    snapshots_path: Path,
    reports: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    samples: dict[str, Any] = {}
    ordered_targets: list[tuple[str, str]] = []
    for tx_hash in tx_hashes:
        report = reports.get(tx_hash)
        if report is None:
            raise RuntimeError(f"report-driven audit missing report for tx={tx_hash}")
        intent = _intent_from_report(report["intent"])
        samples[tx_hash] = SimpleNamespace(
            tx_hash=tx_hash,
            ledger_index=int(report["ledger_index"]),
            transaction_index=int(report["transaction_index"]),
            result={},
            intent=intent,
        )
        side = str((report.get("snapshot") or {}).get("selected_side") or fit._prebook_primary_side_for_output_currency(intent.out_cur))
        ordered_targets.append((tx_hash, side))
    snapshots = {
        tx_hash: payload
        for tx_hash, payload in fit._iter_snapshot_rows_in_sample_order(
            snapshots_path=snapshots_path,
            ordered_targets=ordered_targets,
        )
    }
    return samples, snapshots


def _audit_tx(
    *,
    tx_hash: str,
    sample: Any,
    snapshot_payload: dict[str, Any],
    tick_size_snapshots: list[tuple[int, dict[str, int | None]]],
    amm_rows: dict[str, list[dict[str, Any]]],
    fee_rows: dict[str, list[dict[str, Any]]],
    report: dict[str, Any] | None,
    report_driven: bool,
    top_n: int,
    max_rounds: int,
) -> dict[str, Any]:
    tick_size = fit._effective_offer_tick_size_for_intent(
        intent=sample.intent,
        ledger_index=sample.ledger_index,
        tick_size_snapshots=tick_size_snapshots,
    )
    intent = apply_offer_tick_size_to_intent(sample.intent, tick_size)
    if report_driven and report is not None:
        owner_funds_overlay = {}
        non_fill_deleted_offer_ids = set((report.get("snapshot") or {}).get("non_fill_deleted_offer_ids") or [])
    else:
        owner_funds_overlay = fit._extract_overlay_owner_funds_from_metadata_result(sample.result)
        non_fill_deleted_offer_ids = fit._metadata_deleted_offer_without_fill_ids(sample.result)

    book_offers = fit._materialize_snapshot_book_offers(
        side_payload=snapshot_payload,
        derived_owner_funds_by_offer_id=owner_funds_overlay,
        funded_caps_by_offer_id={},
        strip_owner_funds_accounts=set(),
    )
    materialized_before_non_fill = len(book_offers)
    book_offers = fit._filter_non_fill_deleted_offers(
        book_offers=book_offers,
        offer_ids=non_fill_deleted_offer_ids,
        keep_owner=str(intent.source_account or "")
        if str(intent.transaction_type or "").upper() == "OFFERCREATE"
        else None,
    )
    after_non_fill = len(book_offers)
    book_offers = fit._apply_offercreate_cancel_sequence(book_offers=book_offers, intent=intent)
    after_cancel = len(book_offers)

    clob_segments = fit.build_clob_segments_from_offers(
        book_offers,
        in_cur=intent.in_cur,
        out_cur=intent.out_cur,
        iou_in_transfer_rate=fit.parse_iou_decimal("1", field_name="iou_in_transfer_rate"),
        iou_out_transfer_rate=fit.parse_iou_decimal("1", field_name="iou_out_transfer_rate"),
        exclude_account=None,
    )
    if report_driven and report is not None:
        amm, amm_state_report = _amm_from_report_state(
            amm_state=report.get("amm_state"),
            in_cur=intent.in_cur,
            out_cur=intent.out_cur,
        )
    else:
        amm, amm_state_report = fit._build_model_amm(
            tx_hash=tx_hash,
            metadata_result=sample.result,
            amm_rows=amm_rows.get(tx_hash, []),
            fee_rows=fee_rows.get(tx_hash, []),
            in_cur=intent.in_cur,
            out_cur=intent.out_cur,
        )

    if str(intent.transaction_type or "").upper() == "PAYMENT":
        deliver_max_out = intent.payment_delivermax_amt
        send_max_in = ex._cap_input_budget(intent.payment_sendmax_amt, intent.source_funds_cap_amt)
        limit_quality = intent.payment_limit_quality
        execution_mode = "payment"
        enable_self_cross_filter = False
    elif str(intent.transaction_type or "").upper() == "OFFERCREATE":
        send_max_in = ex._cap_input_budget(intent.offer_taker_gets_amt, intent.source_funds_cap_amt)
        limit_quality = (
            ex._strictly_better_quality_floor(intent.offer_limit_quality)
            if bool(intent.offer_is_passive)
            else intent.offer_limit_quality
        )
        deliver_max_out = (
            ex._sell_deliver_ceiling(intent.out_cur)
            if intent.offer_is_sell
            else intent.offer_taker_pays_amt
        )
        execution_mode = "offer_crossing"
        enable_self_cross_filter = True
    else:
        raise RuntimeError(f"unsupported tx type for audit: {intent.transaction_type}")

    mtiers = ex._segments_to_mutable_tiers(clob_segments)
    working_amm = amm.clone() if amm is not None else None
    remaining_out = deliver_max_out
    remaining_in = send_max_in
    total_in = ex._amt_zero(intent.in_cur)
    total_out = ex._amt_zero(intent.out_cur)
    rounds: list[dict[str, Any]] = []
    sendmax_bound_any = False

    for round_index in range(max_rounds):
        if ex._amt_units_floor(remaining_out) <= 0:
            break
        if remaining_in is not None and ex._amt_units_floor(remaining_in) <= 0:
            break

        cur_segs = ex._mutable_tiers_to_segments(mtiers, intent.in_cur, intent.out_cur, src="CLOB")
        if not cur_segs and working_amm is None:
            break

        top_rows = _top_rows_payload(
            cur_segs,
            intent=intent,
            non_fill_deleted_offer_ids=non_fill_deleted_offer_ids,
            limit_quality=limit_quality,
            top_n=top_n,
        )
        round_payload: dict[str, Any] = {
            "round": round_index,
            "remaining_in": _amount_payload(remaining_in),
            "remaining_out": _amount_payload(remaining_out),
            "top_clob": top_rows,
            "amm_spq": _quality_payload(working_amm.spq_quality_int(), limit_quality=limit_quality)
            if working_amm is not None
            else None,
        }

        view = ex.RouterQuoteView(
            lambda cur=cur_segs: cur,
            amm=working_amm,
            limit_quality=limit_quality,
            mode=execution_mode,
            source_account=intent.source_account,
            enable_self_cross_filter=enable_self_cross_filter,
        )
        reverse_quote = view.preview_out(remaining_out)
        reverse_summary = reverse_quote.get("summary", {}) or {}
        required_in = reverse_summary.get("total_in")
        reverse_out = reverse_summary.get("total_out")
        round_payload["reverse"] = {
            "total_in": _amount_payload(required_in),
            "total_out": _amount_payload(reverse_out),
            "avg_quality": _quality_payload(reverse_summary.get("avg_quality"), limit_quality=limit_quality),
            "slices": [_slice_payload(s, limit_quality=limit_quality) for s in reverse_quote.get("slices", [])],
        }

        if required_in is None or reverse_out is None:
            round_payload["stop_reason"] = "invalid_reverse_summary"
            rounds.append(round_payload)
            break

        if ex._amt_units_floor(reverse_out) <= 0:
            round_payload["stop_reason"] = "reverse_zero"
            if remaining_in is None:
                rounds.append(round_payload)
                break
            step = BookStep.from_static(
                cur_segs,
                amm=working_amm,
                limit_quality=limit_quality,
                mode=execution_mode,
                source_account=intent.source_account,
                owner_pays_transfer_fee=False,
                enable_self_cross_filter=enable_self_cross_filter,
            )
            sb = PaymentSandbox()
            fwd_out, fwd_in, fwd_trace = step.fwd(sb, remaining_in, return_trace=True)
            round_payload["forward_zero_fallback"] = {
                "total_in": _amount_payload(fwd_in),
                "total_out": _amount_payload(fwd_out),
                "would_overfill_remaining_out": _safe_amount_lt(remaining_out, fwd_out),
                "slices": [_slice_payload(s, limit_quality=limit_quality) for s in fwd_trace],
            }
            rounds.append(round_payload)
            break

        sendmax_binding_round = bool(remaining_in is not None and ex._amount_lt(remaining_in, required_in))
        sendmax_bound_any = sendmax_bound_any or sendmax_binding_round
        round_payload["sendmax_binding_round"] = sendmax_binding_round

        if not sendmax_binding_round:
            round_quote = reverse_quote
        else:
            step = BookStep.from_static(
                cur_segs,
                amm=working_amm,
                limit_quality=limit_quality,
                mode=execution_mode,
                source_account=intent.source_account,
                owner_pays_transfer_fee=False,
                enable_self_cross_filter=enable_self_cross_filter,
            )
            step._cached_out = remaining_out
            sb = PaymentSandbox()
            fwd_out, fwd_in, fwd_trace = step.fwd(sb, remaining_in, return_trace=True)
            round_payload["forward"] = {
                "total_in": _amount_payload(fwd_in),
                "total_out": _amount_payload(fwd_out),
                "slices": [_slice_payload(s, limit_quality=limit_quality) for s in fwd_trace],
            }
            round_quote = ex._quote_from_forward_trace(
                trace=fwd_trace,
                total_out=fwd_out,
                total_in=fwd_in,
                requested_out=reverse_out,
            )

        round_summary = round_quote.get("summary", {}) or {}
        round_out = round_summary.get("total_out")
        round_in = round_summary.get("total_in")
        round_payload["chosen"] = {
            "total_in": _amount_payload(round_in),
            "total_out": _amount_payload(round_out),
            "avg_quality": _quality_payload(round_summary.get("avg_quality"), limit_quality=limit_quality),
            "slices": [_slice_payload(s, limit_quality=limit_quality) for s in round_quote.get("slices", [])],
        }
        if round_out is None or round_in is None:
            round_payload["stop_reason"] = "invalid_round_summary"
            rounds.append(round_payload)
            break
        if ex._amt_units_floor(round_out) <= 0 or ex._amt_units_floor(round_in) <= 0:
            round_payload["stop_reason"] = "empty_round"
            rounds.append(round_payload)
            break

        ex._consume_clob_slices_inplace(
            mtiers,
            round_quote,
            in_cur=intent.in_cur,
            out_cur=intent.out_cur,
            owner_pays_transfer_fee=False,
        )
        if working_amm is not None:
            amm_in_round, amm_out_round = ex._sum_amm_slice_amounts(
                round_quote,
                in_cur=intent.in_cur,
                out_cur=intent.out_cur,
            )
            round_payload["amm_applied"] = {
                "in": _amount_payload(amm_in_round),
                "out": _amount_payload(amm_out_round),
            }
            if ex._amt_units_floor(amm_in_round) > 0 and ex._amt_units_floor(amm_out_round) > 0:
                working_amm.apply_fill_st(amm_in_round, amm_out_round)

        total_out = total_out + round_out
        total_in = total_in + round_in
        if ex._amount_eq_or_one_quantum_short(total_out, deliver_max_out):
            remaining_out = ex._amt_zero(intent.out_cur)
        else:
            remaining_out = deliver_max_out - total_out
        if remaining_in is not None and send_max_in is not None:
            remaining_in = ex._amt_zero(intent.in_cur) if not ex._amount_lt(total_in, send_max_in) else send_max_in - total_in

        round_payload["totals_after"] = {
            "total_in": _amount_payload(total_in),
            "total_out": _amount_payload(total_out),
            "remaining_in": _amount_payload(remaining_in),
            "remaining_out": _amount_payload(remaining_out),
        }
        rounds.append(round_payload)

    return {
        "tx_hash": tx_hash,
        "ledger_index": sample.ledger_index,
        "transaction_index": sample.transaction_index,
        "intent": fit._intent_report(intent),
        "effective_limit_quality": _quality_payload(limit_quality),
        "deliver_max_out": _amount_payload(deliver_max_out),
        "send_max_in": _amount_payload(send_max_in),
        "snapshot": {
            "selected_side": fit._prebook_primary_side_for_output_currency(intent.out_cur),
            "used_prebook_ledger": snapshot_payload.get("used_prebook_ledger"),
            "offers_count_visible": snapshot_payload.get("offers_count_visible"),
            "materialized_before_non_fill_deleted_filter": materialized_before_non_fill,
            "materialized_after_non_fill_deleted_filter": after_non_fill,
            "materialized_after_cancel_sequence": after_cancel,
            "non_fill_deleted_offer_count": len(non_fill_deleted_offer_ids),
            "non_fill_deleted_offer_ids": sorted(non_fill_deleted_offer_ids),
        },
        "amm_state": amm_state_report,
        "rounds": rounds,
        "model_total": {"in": _amount_payload(total_in), "out": _amount_payload(total_out)},
        "sendmax_bound_any": sendmax_bound_any,
        "existing_report": _report_leg_summary(report),
    }


def main() -> None:
    args = _parse_args()
    fit.TARGET_IOU_CURRENCY = str(args.base_currency)
    fit.TARGET_IOU_LABEL = str(args.base_label)
    tx_hashes = _load_hashes(args)
    metadata_path = Path(args.metadata_ndjson)
    snapshots_path = Path(args.tx_prebook_snapshots)

    reports_dir = Path(args.reports_dir) if args.reports_dir else None
    reports: dict[str, dict[str, Any]] = {}
    if reports_dir is not None:
        for tx_hash in tx_hashes:
            path = reports_dir / f"{tx_hash}.json"
            if path.exists():
                reports[tx_hash] = json.loads(path.read_text(encoding="utf-8"))
    report_driven = reports_dir is not None and all(tx_hash in reports for tx_hash in tx_hashes)

    if report_driven:
        samples, snapshots = _build_report_context(
            tx_hashes=tx_hashes,
            snapshots_path=snapshots_path,
            reports=reports,
        )
        tick_size_snapshots = fit._load_offer_tick_size_snapshots(metadata_path)
        amm_rows: dict[str, list[dict[str, Any]]] = {}
        fee_rows: dict[str, list[dict[str, Any]]] = {}
    else:
        samples, snapshots = _build_sample_context(
            tx_hashes=tx_hashes,
            metadata_path=metadata_path,
            snapshots_path=snapshots_path,
        )
        tick_size_snapshots = fit._load_offer_tick_size_snapshots(metadata_path)
        amm_rows = fit._load_amm_rows(Path(args.amm_swaps), tx_column="transaction_hash", target_tx_hashes=set(tx_hashes))
        fee_rows = fit._load_amm_rows(Path(args.amm_fees), tx_column="transaction_hash", target_tx_hashes=set(tx_hashes))

    audits = [
        _audit_tx(
            tx_hash=tx_hash,
            sample=samples[tx_hash],
            snapshot_payload=snapshots[tx_hash],
            tick_size_snapshots=tick_size_snapshots,
            amm_rows=amm_rows,
            fee_rows=fee_rows,
            report=reports.get(tx_hash),
            report_driven=report_driven,
            top_n=args.top_n,
            max_rounds=args.max_rounds,
        )
        for tx_hash in tx_hashes
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"audits": audits}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(audits)} audit traces to {output}")


if __name__ == "__main__":
    main()
