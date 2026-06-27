"""Strict single-path BookStep built on rebuilt core, book_offers, and AMM modules."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Callable, Iterable, Literal

from .amm import (
    AMM,
    AMMOverAsk,
    ROUND_TO_NEAREST,
    ROUND_UPWARD,
    _RippleNumber,
    _fee_multiplier_number,
    _number_div,
    _number_mul,
    _number_one,
    _number_sub,
    _quality_rate_number,
    _ripple_number_mantissa_scale_guard,
    amount_add_ripple_number,
    amount_sub_ripple_number,
)
from .book_offers import (
    BookOfferSegment,
    _divide_amount_by_quality_rate_strict,
    _mul_amount_by_quality_rate_strict,
)
from .core import (
    Amount,
    INT64_MAX,
    IOUAmount,
    IOU_EXP_MIN,
    IOU_MANTISSA_MIN,
    Quality,
    Segment,
    XRPAmount,
    ceil_div,
)
from .steps import Step

if TYPE_CHECKING:
    from .flow import PaymentSandbox


MAX_BOOK_STEP_ITERS = 256
RELATIVE_DISTANCE = Fraction(1, 10_000_000)
LIMIT_OUT_RELATIVE_DISTANCE = Fraction(1, 1_000_000_000)
QUALITY_ONE = 1_000_000_000
_INT64_MAX_LOG10_FLOOR = len(str(INT64_MAX)) - 1


@dataclass(frozen=True, slots=True)
class _StepRow:
    source_id: str | None
    quoted_quality: Quality
    segment: Segment
    book_offer: BookOfferSegment | None = None


@dataclass(frozen=True, slots=True)
class _PreparedRow:
    base_row: _StepRow
    raw_segment: Segment
    step_out: Amount
    step_in: Amount
    owner_gives: Amount
    in_rate: IOUAmount | None
    out_rate: IOUAmount | None


@dataclass(slots=True)
class _AMMAggregateExecution:
    routing_amm: AMM
    cumulative_in: Amount
    cumulative_out: Amount

    @classmethod
    def from_amm(cls, amm: AMM) -> "_AMMAggregateExecution":
        return cls(
            routing_amm=amm.clone(),
            cumulative_in=_zero_like(amm._x_amount),
            cumulative_out=_zero_like(amm._y_amount),
        )

    def has_fill(self) -> bool:
        return _is_positive(self.cumulative_in) and _is_positive(self.cumulative_out)

    def shadow_amm(self) -> AMM:
        return self.routing_amm.clone()

    def apply_exact_in(
        self,
        *,
        raw_in: Amount,
        raw_out: Amount,
    ) -> tuple[Amount, Amount, AMM]:
        self.routing_amm.apply_fill_st(raw_in, raw_out)
        self.cumulative_in = self.cumulative_in + raw_in
        self.cumulative_out = self.cumulative_out + raw_out
        if not _is_positive(raw_out):
            raise RuntimeError("aggregate AMM exact-in produced non-positive delta output")
        return raw_out, raw_in, self.shadow_amm()

    def apply_exact_out(
        self,
        *,
        raw_in: Amount,
        raw_out: Amount,
    ) -> tuple[Amount, Amount, AMM]:
        self.routing_amm.apply_fill_st(raw_in, raw_out)
        self.cumulative_in = self.cumulative_in + raw_in
        self.cumulative_out = self.cumulative_out + raw_out
        if not _is_positive(raw_in):
            raise RuntimeError("aggregate AMM exact-out produced non-positive delta input")
        return raw_out, raw_in, self.shadow_amm()

    def merged_trace_row(self) -> dict[str, object]:
        if not self.has_fill():
            raise RuntimeError("cannot build merged AMM trace without positive fill")
        return {
            "src": "AMM",
            "source_id": None,
            "out_take": self.cumulative_out,
            "in_take": self.cumulative_in,
            "quality": Quality.from_amounts(self.cumulative_out, self.cumulative_in),
        }


def _zero_like(amount: Amount) -> Amount:
    if isinstance(amount, XRPAmount):
        return XRPAmount(0)
    return IOUAmount(0, 0)


def _amount_add(left: Amount, right: Amount) -> Amount:
    return amount_add_ripple_number(left, right)


def _amount_sub(left: Amount, right: Amount) -> Amount:
    return amount_sub_ripple_number(left, right)


def _sum_amounts_sorted(values: list[Amount], *, zero_like: Amount) -> Amount:
    if not values:
        return _zero_like(zero_like)
    ordered = sorted(values)
    total = ordered[0]
    for value in ordered[1:]:
        total = _amount_add(total, value)
    return total


def _is_positive(amount: Amount) -> bool:
    if isinstance(amount, XRPAmount):
        return amount.drops > 0
    return amount.mantissa > 0


def _same_amount_type(left: Amount, right: Amount) -> bool:
    return type(left) is type(right)


def _min_amount(left: Amount, right: Amount) -> Amount:
    if not _same_amount_type(left, right):
        raise RuntimeError("amount domain mismatch")
    return left if left <= right else right


def _amount_units_floor(amount: Amount) -> int:
    if isinstance(amount, XRPAmount):
        return max(0, amount.drops)
    if amount.mantissa <= 0:
        return 0
    shift = amount.exponent + 15
    if shift >= 0:
        return amount.mantissa * (10**shift)
    return amount.mantissa // (10 ** (-shift))


def _amount_from_units_like(proto: Amount, units: int) -> Amount:
    if units <= 0:
        return _zero_like(proto)
    if isinstance(proto, XRPAmount):
        return XRPAmount(units)
    mantissa = units
    exponent = -15
    while mantissa > INT64_MAX:
        mantissa //= 10
        exponent += 1
    return IOUAmount(mantissa, exponent)


def _amount_to_fraction(amount: Amount) -> Fraction:
    return amount.as_fraction()


def _floor_log10_fraction(value: Fraction) -> int:
    if value <= 0:
        raise RuntimeError("log10 requires a strictly positive value")
    numerator = value.numerator
    denominator = value.denominator
    digits = len(str(numerator)) - len(str(denominator))
    if digits >= 0:
        if numerator < denominator * (10**digits):
            digits -= 1
        return digits
    if numerator * (10 ** (-digits)) < denominator:
        digits -= 1
    return digits


def _fraction_to_iou_amount(value: Fraction, *, round_up: bool) -> IOUAmount:
    if value <= 0:
        return IOUAmount(0, 0)
    exponent = _floor_log10_fraction(value) - 15
    scale = 10 ** (-exponent)
    scaled_num = value.numerator * scale
    mantissa = ceil_div(scaled_num, value.denominator) if round_up else (
        scaled_num // value.denominator
    )
    if mantissa <= 0:
        return IOUAmount(0, 0)
    return IOUAmount(mantissa, exponent)


def _fraction_to_amount_like(proto: Amount, value: Fraction, *, round_up: bool) -> Amount:
    if value <= 0:
        return _zero_like(proto)
    if isinstance(proto, XRPAmount):
        if round_up:
            return XRPAmount(ceil_div(value.numerator, value.denominator))
        return XRPAmount(value.numerator // value.denominator)
    return _fraction_to_iou_amount(value, round_up=round_up)


def _amount_relative_distance(left: Amount, right: Amount) -> Fraction:
    left_value = _amount_to_fraction(left)
    right_value = _amount_to_fraction(right)
    if left_value == right_value:
        return Fraction(0, 1)
    high = max(left_value, right_value)
    if high <= 0:
        return Fraction(0, 1)
    return (high - min(left_value, right_value)) / high


def _quality_rate_fraction(quality: Quality) -> Fraction:
    return quality.rate().as_fraction()


def _within_relative_distance(left: Quality, right: Quality) -> bool:
    if left == right:
        return True
    left_rate = _quality_rate_fraction(left)
    right_rate = _quality_rate_fraction(right)
    high = max(left_rate, right_rate)
    low = min(left_rate, right_rate)
    if high <= 0:
        return False
    return (high - low) / high < RELATIVE_DISTANCE


def _offer_stream_rows(
    rows: list[_StepRow],
    *,
    mode: Literal["payment", "offer_crossing"],
    source_account: str | None = None,
    book_in_transfer_rate: IOUAmount | None = None,
    book_out_transfer_rate: IOUAmount | None = None,
    charge_input_transfer_fee: bool = True,
    owner_pays_transfer_fee: bool = False,
    enable_self_cross_filter: bool = False,
) -> list[_StepRow]:
    # Match rippled OfferStream::step(): offers with no executable owner funds
    # are skipped for execution. BookTip visibility is handled separately.
    out: list[_StepRow] = []
    for row in _sorted_rows(rows):
        prepared = _prepare_row(
            row,
            mode=mode,
            source_account=source_account,
            book_in_transfer_rate=book_in_transfer_rate,
            book_out_transfer_rate=book_out_transfer_rate,
            charge_input_transfer_fee=charge_input_transfer_fee,
            owner_pays_transfer_fee=owner_pays_transfer_fee,
            enable_self_cross_filter=enable_self_cross_filter,
        )
        if prepared is not None:
            out.append(row)
    return out


def _lob_tip_quality(
    rows: list[_StepRow],
    *,
    mode: Literal["payment", "offer_crossing"],
    source_account: str | None = None,
    book_in_transfer_rate: IOUAmount | None = None,
    book_out_transfer_rate: IOUAmount | None = None,
    charge_input_transfer_fee: bool = True,
    owner_pays_transfer_fee: bool = False,
    enable_self_cross_filter: bool = False,
) -> Quality | None:
    del mode, source_account, book_in_transfer_rate, book_out_transfer_rate, charge_input_transfer_fee, owner_pays_transfer_fee, enable_self_cross_filter
    sorted_rows = _sorted_rows(rows)
    return sorted_rows[0].quoted_quality if sorted_rows else None


def _offer_crossing_amm_quality_threshold(
    *,
    lob_quality: Quality | None,
    limit_quality: Quality | None,
    fix_amm_v1_1_enabled: bool = True,
) -> Quality | None:
    if lob_quality is None:
        return None
    if fix_amm_v1_1_enabled and limit_quality is not None and limit_quality > lob_quality:
        return None
    return lob_quality


def _offer_crossing_boundary_lob_quality(
    *,
    lob_quality: Quality,
    limit_quality: Quality | None,
) -> bool:
    if limit_quality is None or limit_quality.value >= (2**64) - 1:
        return False
    return lob_quality.value == limit_quality.value + 1


def _amm_offer_for_clob_quality(
    amm: AMM,
    top_quality: Quality,
    *,
    max_out_cap: Amount | None,
    require_post_spq_floor: bool,
) -> Segment | None:
    segment = amm.synthetic_segment_for_quality(
        top_quality,
        max_out_cap=max_out_cap,
        require_post_spq_floor=require_post_spq_floor,
    )
    if segment is not None:
        return segment

    # AMMLiquidity::getOffer() has a fixAMMv1_2 fallback: if
    # changeSpotPriceQuality() cannot build a synthetic offer at the CLOB
    # quality, maxOffer() is still valid when its amount quality beats CLOB.
    max_offer = amm.max_offer_segment(max_out_cap=max_out_cap)
    if max_offer is None:
        return None
    amount_quality = Quality.from_amounts(max_offer.out_max, max_offer.in_at_out_max)
    return max_offer if amount_quality > top_quality else None


def _amm_is_tip_for_quality_function(
    *,
    amm: AMM | None,
    lob_quality: Quality | None,
    mode: Literal["payment", "offer_crossing"],
    limit_quality: Quality | None,
    fix_amm_v1_1_enabled: bool = True,
) -> bool:
    if amm is None:
        return False
    if lob_quality is None:
        return True
    if mode == "offer_crossing":
        threshold = _offer_crossing_amm_quality_threshold(
            lob_quality=lob_quality,
            limit_quality=limit_quality,
            fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
        )
        if threshold is None:
            amm_offer = amm.max_offer_segment(max_out_cap=amm.max_out_amount())
        else:
            amm_offer = _amm_offer_for_clob_quality(
                amm,
                threshold,
                max_out_cap=amm.max_out_amount(),
                require_post_spq_floor=True,
            )
        # Match BookStep::tip()/tipOfferQualityF(): limitOut uses the same AMM
        # quality function selected by the current BookTip comparison.
        return amm_offer is not None and amm_offer.quality > lob_quality

    spq = amm.spq_quality_int()
    if not (spq > lob_quality):
        return False
    if _within_relative_distance(spq, lob_quality):
        return False
    synthetic = _amm_offer_for_clob_quality(
        amm,
        lob_quality,
        max_out_cap=amm.max_out_amount(),
        require_post_spq_floor=True,
    )
    return synthetic is not None and synthetic.quality >= lob_quality


def _limit_out_by_quality_function(
    rows: list[_StepRow],
    *,
    amm: AMM | None,
    remaining_out: Amount,
    limit_quality: Quality | None,
    mode: Literal["payment", "offer_crossing"],
    source_account: str | None = None,
    book_in_transfer_rate: IOUAmount | None = None,
    book_out_transfer_rate: IOUAmount | None = None,
    charge_input_transfer_fee: bool = True,
    owner_pays_transfer_fee: bool = False,
    enable_self_cross_filter: bool = False,
    fix_amm_v1_1_enabled: bool = True,
) -> tuple[Amount, bool]:
    if limit_quality is None or amm is None:
        return remaining_out, False

    lob_quality = _lob_tip_quality(
        rows,
        mode=mode,
        source_account=source_account,
        book_in_transfer_rate=book_in_transfer_rate,
        book_out_transfer_rate=book_out_transfer_rate,
        charge_input_transfer_fee=charge_input_transfer_fee,
        owner_pays_transfer_fee=owner_pays_transfer_fee,
        enable_self_cross_filter=enable_self_cross_filter,
    )
    if not _amm_is_tip_for_quality_function(
        amm=amm,
        lob_quality=lob_quality,
        mode=mode,
        limit_quality=limit_quality,
        fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
    ):
        return remaining_out, False

    # Match rippled's transaction-rules guard: the AMM quality function uses the
    # same Number mantissa scale as the AMM swap math for this ledger.
    with _ripple_number_mantissa_scale_guard(amm.number_mantissa_scale):
        rate = _quality_rate_number(limit_quality)
        if rate.is_zero():
            return remaining_out, False

        fee_multiplier = _fee_multiplier_number(100_000 - amm.trading_fee)
        if fee_multiplier.signum() <= 0:
            return remaining_out, False

        pool_in = _RippleNumber.from_amount(amm._x_amount)
        pool_out = _RippleNumber.from_amount(amm._y_amount)
        slope = -_number_div(
            fee_multiplier,
            pool_in,
            rounding=ROUND_TO_NEAREST,
        )
        intercept = _number_div(
            _number_mul(pool_out, fee_multiplier, rounding=ROUND_TO_NEAREST),
            pool_in,
            rounding=ROUND_TO_NEAREST,
        )
        active_in_rate = book_in_transfer_rate if charge_input_transfer_fee else None
        if active_in_rate is not None:
            # rippled BookOfferCrossingStep::adjustQualityWithFees() composes
            # single-path AMM quality functions with the transfer-in rate.
            # The AMM pool math is raw input; the path quality threshold is
            # measured against gross input after issuer transfer fees.
            rate_scale = _number_div(
                _number_one(),
                _RippleNumber.from_amount(active_in_rate),
                rounding=ROUND_TO_NEAREST,
            )
            slope = _number_mul(slope, rate_scale, rounding=ROUND_TO_NEAREST)
            intercept = _number_mul(intercept, rate_scale, rounding=ROUND_TO_NEAREST)
        limited_out = _number_div(
            _number_sub(_number_div(_number_one(), rate, rounding=ROUND_UPWARD), intercept, rounding=ROUND_UPWARD),
            slope,
            rounding=ROUND_UPWARD,
        )
        if limited_out.signum() <= 0:
            return remaining_out, False

        capped = limited_out.to_amount(
            is_xrp=isinstance(remaining_out, XRPAmount),
            rounding=ROUND_TO_NEAREST,
        )
    if capped >= remaining_out:
        return remaining_out, False
    if _amount_relative_distance(capped, remaining_out) < LIMIT_OUT_RELATIVE_DISTANCE:
        return remaining_out, False
    return capped, True


def _normalise_rows(items: Iterable[Segment | BookOfferSegment]) -> list[_StepRow]:
    rows: list[_StepRow] = []
    for item in items:
        if isinstance(item, BookOfferSegment):
            rows.append(
                _StepRow(
                    source_id=item.offer_id,
                    quoted_quality=item.quoted_quality,
                    segment=item.segment,
                    book_offer=item,
                )
            )
            continue
        if isinstance(item, Segment):
            rows.append(
                _StepRow(
                    source_id=None,
                    quoted_quality=item.quality,
                    segment=item,
                    book_offer=None,
                )
            )
            continue
        raise RuntimeError(f"unsupported book step segment type: {type(item)!r}")
    return rows


def _sorted_rows(rows: list[_StepRow]) -> list[_StepRow]:
    # Quality.__lt__ mirrors rippled's "lower encoded value is better" ordering.
    # reverse=True therefore produces the whitepaper/rippled best-first book order.
    return sorted(rows, key=lambda row: row.quoted_quality, reverse=True)


def _filtered_rows(rows: list[_StepRow], limit_quality: Quality | None) -> list[_StepRow]:
    if limit_quality is None:
        return rows
    return [row for row in rows if row.quoted_quality >= limit_quality]


def _offer_crossing_candidate_floor(limit_quality: Quality | None) -> Quality | None:
    if limit_quality is None:
        return None
    if limit_quality.value >= (2**64) - 1:
        return limit_quality
    return Quality(limit_quality.value + 1)


def _clob_out_capacity(rows: list[_StepRow], proto: Amount) -> Amount:
    total = _zero_like(proto)
    for row in rows:
        if row.book_offer is not None and row.book_offer.book_visible_only:
            continue
        total = total + row.segment.out_max
    return total


def _trace_row(row: _StepRow, *, out_take: Amount, in_take: Amount) -> dict[str, object]:
    return {
        "src": row.segment.src,
        "source_id": row.source_id,
        "out_take": out_take,
        "in_take": in_take,
        "quality": Quality.from_amounts(out_take, in_take),
    }


def _trace_row_with_gross_input(
    *,
    src: str,
    source_id: str | None,
    out_take: Amount,
    net_in_take: Amount,
    gross_in_take: Amount,
    owner_out_take: Amount | None = None,
) -> dict[str, object]:
    row = {
        "src": src,
        "source_id": source_id,
        "out_take": out_take,
        # Metadata-visible offer/AMM balance delta.
        "in_take": net_in_take,
        "quality": Quality.from_amounts(out_take, net_in_take),
    }
    if gross_in_take != net_in_take:
        row["gross_in_take"] = gross_in_take
    if owner_out_take is not None and owner_out_take != out_take:
        row["owner_out_take"] = owner_out_take
    return row


def _trace_delete_row(row: _StepRow) -> dict[str, object]:
    return {
        "src": "CLOB_DELETE",
        "source_id": row.source_id,
        "reason": "self_cross",
    }


def _without_became_unfunded_visible_residuals(
    rows: list[_StepRow],
    offer_ids: set[str],
) -> list[_StepRow]:
    # In rippled, an offer that becomes unfunded during the current flow is not
    # executable, but it is not perm-removed from the book. Preserve its visible
    # residual so later BookTip/quality checks still see the ledger tip.
    del offer_ids
    return rows


def _mark_became_unfunded_after_consume(
    prepared: _PreparedRow,
    became_unfunded_offer_ids: set[str],
) -> None:
    offer = prepared.base_row.book_offer
    if offer is None or offer.book_visible_only or offer.owner_is_out_issuer or offer.owner_funds is None:
        return
    remaining_funds = offer.owner_funds - prepared.owner_gives
    if not _is_positive(remaining_funds):
        became_unfunded_offer_ids.add(offer.offer_id)


def _iteration_violates_limit_quality(
    *,
    mode: Literal["payment", "offer_crossing"],
    limit_quality: Quality | None,
    itr_quality: Quality,
    adjusted_out: bool,
    amm_in: Amount,
    amm_out: Amount,
) -> bool:
    if limit_quality is None:
        return False
    return itr_quality < limit_quality and (
        not adjusted_out or not _within_relative_distance(itr_quality, limit_quality)
    )


def _offer_crossing_offer_rejected_by_threshold(
    *,
    mode: Literal["payment", "offer_crossing"],
    limit_quality: Quality | None,
    offer_quality: Quality,
) -> bool:
    if mode != "offer_crossing" or limit_quality is None:
        return False
    # Match BookOfferCrossingStep::checkQualityThreshold(): this per-offer gate
    # is strict. StrandFlow's adjusted limitOut tolerance is applied later to the
    # whole flow result and does not let a below-threshold AMM offer enter the
    # BookStep callback.
    return offer_quality < limit_quality


def _log10_ceil_positive(value: int) -> int:
    if value <= 0:
        return 0
    digits = len(str(value))
    return digits - 1 if value == 10 ** (digits - 1) else digits


def _transfer_rate_raw(rate: IOUAmount) -> int:
    raw = rate.as_fraction() * QUALITY_ONE
    if raw.denominator != 1:
        raise RuntimeError("transfer rate is not exact on QUALITY_ONE grid")
    value = raw.numerator
    if value < 0 or value > 2**32 - 1:
        raise RuntimeError("transfer rate raw value out of uint32 range")
    return value


def _mul_ratio_amount(
    amount: Amount,
    *,
    numerator: int,
    denominator: int,
    round_up: bool,
) -> Amount:
    if denominator <= 0:
        raise RuntimeError("mulRatio denominator must be positive")
    if numerator < 0:
        raise RuntimeError("mulRatio numerator must be non-negative")
    if isinstance(amount, XRPAmount):
        product = amount.drops * numerator
        quotient, remainder = divmod(product, denominator)
        if remainder:
            if amount.drops >= 0 and round_up:
                quotient += 1
            elif amount.drops < 0 and not round_up:
                quotient -= 1
        return XRPAmount(quotient)

    negative = amount.mantissa < 0
    product = abs(amount.mantissa) * numerator
    low, remainder = divmod(product, denominator)
    exponent = amount.exponent

    if remainder:
        room_to_grow = _INT64_MAX_LOG10_FLOOR - _log10_ceil_positive(low)
        if room_to_grow > 0:
            scale = 10**room_to_grow
            exponent -= room_to_grow
            low *= scale
            remainder *= scale
        add_remainder, remainder = divmod(remainder, denominator)
        low += add_remainder

    has_remainder = bool(remainder)
    must_shrink = _log10_ceil_positive(low) - _INT64_MAX_LOG10_FLOOR
    if must_shrink > 0:
        scale = 10**must_shrink
        saved = low
        exponent += must_shrink
        low //= scale
        if not has_remainder:
            has_remainder = saved != low * scale

    mantissa = -low if negative else low
    result = IOUAmount(mantissa, exponent)
    if has_remainder:
        if round_up and not negative:
            if result.is_zero():
                return IOUAmount(IOU_MANTISSA_MIN, IOU_EXP_MIN)
            return IOUAmount(result.mantissa + 1, result.exponent)
        if not round_up and negative:
            if result.is_zero():
                return IOUAmount(-IOU_MANTISSA_MIN, IOU_EXP_MIN)
            return IOUAmount(result.mantissa - 1, result.exponent)
    return result


def _apply_rate(amount: Amount, rate: IOUAmount | None, *, round_up: bool) -> Amount:
    if rate is None:
        return amount
    return _mul_ratio_amount(
        amount,
        numerator=_transfer_rate_raw(rate),
        denominator=QUALITY_ONE,
        round_up=round_up,
    )


def _remove_rate(amount: Amount, rate: IOUAmount | None, *, round_up: bool) -> Amount:
    if rate is None:
        return amount
    return _mul_ratio_amount(
        amount,
        numerator=QUALITY_ONE,
        denominator=_transfer_rate_raw(rate),
        round_up=round_up,
    )


def _limit_segment_out(segment: Segment, limit: Amount, *, round_up: bool) -> Segment:
    if not _same_amount_type(segment.out_max, limit):
        raise RuntimeError("limit out amount domain mismatch")
    if limit >= segment.out_max:
        return segment
    limited_in = _mul_amount_by_quality_rate_strict(
        limit,
        segment.quality.rate(),
        result_is_xrp=isinstance(segment.in_at_out_max, XRPAmount),
        round_up=round_up,
    )
    if limited_in > segment.in_at_out_max:
        limited_in = segment.in_at_out_max
    return Segment(
        src="CLOB",
        quality=segment.quality,
        out_max=limit,
        in_at_out_max=limited_in,
    )


def _limit_segment_in(segment: Segment, limit: Amount, *, round_up: bool) -> Segment | None:
    if not _same_amount_type(segment.in_at_out_max, limit):
        raise RuntimeError("limit in amount domain mismatch")
    if not _is_positive(limit):
        return None
    if limit >= segment.in_at_out_max:
        return segment
    limited_out = _divide_amount_by_quality_rate_strict(
        limit,
        segment.quality.rate(),
        result_is_xrp=isinstance(segment.out_max, XRPAmount),
        round_up=round_up,
    )
    if limited_out > segment.out_max:
        limited_out = segment.out_max
    if not _is_positive(limited_out):
        return None
    return Segment(
        src="CLOB",
        quality=segment.quality,
        out_max=limited_out,
        in_at_out_max=limit,
    )


def _prepare_row(
    row: _StepRow,
    *,
    mode: Literal["payment", "offer_crossing"],
    source_account: str | None,
    book_in_transfer_rate: IOUAmount | None = None,
    book_out_transfer_rate: IOUAmount | None = None,
    charge_input_transfer_fee: bool = True,
    owner_pays_transfer_fee: bool = False,
    enable_self_cross_filter: bool = False,
) -> _PreparedRow | None:
    raw_segment = row.segment
    if row.book_offer is None:
        if not _is_positive(raw_segment.out_max) or not _is_positive(raw_segment.in_at_out_max):
            return None
        return _PreparedRow(
            base_row=row,
            raw_segment=raw_segment,
            step_out=raw_segment.out_max,
            step_in=raw_segment.in_at_out_max,
            owner_gives=raw_segment.out_max,
            in_rate=None,
            out_rate=None,
        )

    if row.book_offer.book_visible_only:
        return None

    owner = row.book_offer.owner or ""
    in_rate = (row.book_offer.in_transfer_rate or book_in_transfer_rate) if charge_input_transfer_fee else None
    if (mode == "offer_crossing" or enable_self_cross_filter) and source_account and owner == source_account:
        in_rate = None
    out_rate = (row.book_offer.out_transfer_rate or book_out_transfer_rate) if owner_pays_transfer_fee else None

    step_out = raw_segment.out_max
    step_in = _apply_rate(raw_segment.in_at_out_max, in_rate, round_up=True)
    owner_gives = _apply_rate(raw_segment.out_max, out_rate, round_up=False)

    owner_funds = row.book_offer.owner_funds
    if not row.book_offer.owner_is_out_issuer and owner_funds is not None and owner_funds < owner_gives:
        step_out = _remove_rate(owner_funds, out_rate, round_up=False)
        if not _is_positive(step_out):
            return None
        raw_segment = _limit_segment_out(raw_segment, step_out, round_up=False)
        step_out = raw_segment.out_max
        step_in = _apply_rate(raw_segment.in_at_out_max, in_rate, round_up=True)
        owner_gives = owner_funds

    if not _is_positive(step_out) or not _is_positive(step_in) or not _is_positive(owner_gives):
        return None
    return _PreparedRow(
        base_row=row,
        raw_segment=raw_segment,
        step_out=step_out,
        step_in=step_in,
        owner_gives=owner_gives,
        in_rate=in_rate,
        out_rate=out_rate,
    )


def _limit_prepared_out(prepared: _PreparedRow, limit: Amount) -> _PreparedRow:
    if not _same_amount_type(prepared.step_out, limit):
        raise RuntimeError("limit out amount domain mismatch")
    if limit >= prepared.step_out:
        return prepared
    owner_gives = _apply_rate(limit, prepared.out_rate, round_up=False)
    raw_segment = _limit_segment_out(prepared.raw_segment, limit, round_up=True)
    step_out = raw_segment.out_max
    step_in = _apply_rate(raw_segment.in_at_out_max, prepared.in_rate, round_up=True)
    return _PreparedRow(
        base_row=prepared.base_row,
        raw_segment=raw_segment,
        step_out=step_out,
        step_in=step_in,
        owner_gives=owner_gives,
        in_rate=prepared.in_rate,
        out_rate=prepared.out_rate,
    )


def _limit_prepared_in(
    prepared: _PreparedRow,
    limit: Amount,
    *,
    fix_reduced_offers_v2_enabled: bool,
) -> _PreparedRow | None:
    if not _same_amount_type(prepared.step_in, limit):
        raise RuntimeError("limit in amount domain mismatch")
    if not _is_positive(limit):
        return None
    if limit >= prepared.step_in:
        return prepared
    raw_in_limit = _remove_rate(limit, prepared.in_rate, round_up=False)
    if not _is_positive(raw_in_limit):
        return None
    # Offer::limitIn() only uses ceil_in_strict(roundUp=false) after
    # fixReducedOffersV2. Before that amendment it ignores roundUp and uses
    # legacy ceil_in(), which rounds the paired output up.
    raw_segment = _limit_segment_in(
        prepared.raw_segment,
        raw_in_limit,
        round_up=not fix_reduced_offers_v2_enabled,
    )
    if raw_segment is None:
        return None
    step_out = raw_segment.out_max
    step_in = _apply_rate(raw_segment.in_at_out_max, prepared.in_rate, round_up=True)
    owner_gives = _apply_rate(step_out, prepared.out_rate, round_up=False)
    if not _is_positive(step_out) or not _is_positive(step_in) or not _is_positive(owner_gives):
        return None
    return _PreparedRow(
        base_row=prepared.base_row,
        raw_segment=raw_segment,
        step_out=step_out,
        step_in=step_in,
        owner_gives=owner_gives,
        in_rate=prepared.in_rate,
        out_rate=prepared.out_rate,
    )


def _next_row_after_consume(prepared: _PreparedRow) -> _StepRow | None:
    base_row = prepared.base_row
    rem_out = _amount_sub(base_row.segment.out_max, prepared.raw_segment.out_max)
    rem_in = _amount_sub(base_row.segment.in_at_out_max, prepared.raw_segment.in_at_out_max)
    if not _is_positive(rem_out) or not _is_positive(rem_in):
        return None

    next_segment = Segment(
        src="CLOB",
        quality=base_row.segment.quality,
        out_max=rem_out,
        in_at_out_max=rem_in,
    )
    if base_row.book_offer is None:
        return _StepRow(
            source_id=base_row.source_id,
            quoted_quality=base_row.quoted_quality,
            segment=next_segment,
            book_offer=None,
        )

    next_funds = base_row.book_offer.owner_funds
    if not base_row.book_offer.owner_is_out_issuer and next_funds is not None:
        next_funds = _amount_sub(next_funds, prepared.owner_gives)
        if not _is_positive(next_funds):
            return None

    next_offer = BookOfferSegment(
        offer_id=base_row.book_offer.offer_id,
        quoted_quality=base_row.book_offer.quoted_quality,
        segment=next_segment,
        owner=base_row.book_offer.owner,
        owner_is_out_issuer=base_row.book_offer.owner_is_out_issuer,
        owner_funds=next_funds,
        in_transfer_rate=base_row.book_offer.in_transfer_rate,
        out_transfer_rate=base_row.book_offer.out_transfer_rate,
        non_fill_deleted=base_row.book_offer.non_fill_deleted,
        book_visible_only=base_row.book_offer.book_visible_only,
    )
    return _StepRow(
        source_id=next_offer.offer_id,
        quoted_quality=next_offer.quoted_quality,
        segment=next_offer.segment,
        book_offer=next_offer,
    )


def _should_delete_self_cross_offer(
    row: _StepRow,
    *,
    mode: Literal["payment", "offer_crossing"],
    source_account: str | None,
    limit_quality: Quality | None,
    enable_self_cross_filter: bool = False,
) -> bool:
    if (mode != "offer_crossing" and not enable_self_cross_filter) or limit_quality is None or row.book_offer is None:
        return False
    owner = row.book_offer.owner or ""
    if not source_account or owner != source_account:
        return False
    if row.book_offer.non_fill_deleted:
        return True
    return row.quoted_quality >= limit_quality


def _consume_clob_exact_out(
    rows: list[_StepRow],
    *,
    out_cap: Amount,
    limit_quality: Quality | None,
    mode: Literal["payment", "offer_crossing"],
    source_account: str | None,
    book_in_transfer_rate: IOUAmount | None = None,
    book_out_transfer_rate: IOUAmount | None = None,
    charge_input_transfer_fee: bool = True,
    owner_pays_transfer_fee: bool = False,
    enable_self_cross_filter: bool = False,
    fix_reduced_offers_v2_enabled: bool = True,
) -> tuple[Amount, Amount, list[_StepRow], list[dict[str, object]]]:
    if not rows:
        zero_out = _zero_like(out_cap)
        return zero_out, zero_out, [], []

    current_quality: Quality | None = None
    offer_attempted = False
    remaining_out = out_cap
    total_out = _zero_like(out_cap)
    total_in = _zero_like(rows[0].segment.in_at_out_max)
    saved_ins: list[Amount] = []
    saved_outs: list[Amount] = []
    next_rows: list[_StepRow] = []
    trace: list[dict[str, object]] = []
    became_unfunded_offer_ids: set[str] = set()

    for index, row in enumerate(rows):
        if remaining_out.is_zero():
            next_rows.extend(_without_became_unfunded_visible_residuals(rows[index:], became_unfunded_offer_ids))
            break
        if current_quality is None:
            current_quality = row.quoted_quality
        elif row.quoted_quality != current_quality:
            if not offer_attempted:
                current_quality = row.quoted_quality
            else:
                next_rows.extend(
                    _without_became_unfunded_visible_residuals(
                        rows[index:],
                        became_unfunded_offer_ids,
                    )
                )
                break

        if _should_delete_self_cross_offer(
            row,
            mode=mode,
            source_account=source_account,
            limit_quality=limit_quality,
            enable_self_cross_filter=enable_self_cross_filter,
        ):
            trace.append(_trace_delete_row(row))
            if not offer_attempted:
                current_quality = None
            continue

        if row.quoted_quality != current_quality:
            next_rows.extend(
                _without_became_unfunded_visible_residuals(
                    rows[index:],
                    became_unfunded_offer_ids,
                )
            )
            break

        prepared = _prepare_row(
            row,
            mode=mode,
            source_account=source_account,
            book_in_transfer_rate=book_in_transfer_rate,
            book_out_transfer_rate=book_out_transfer_rate,
            charge_input_transfer_fee=charge_input_transfer_fee,
            owner_pays_transfer_fee=owner_pays_transfer_fee,
            enable_self_cross_filter=enable_self_cross_filter,
        )
        if prepared is None:
            continue

        if prepared.step_out > remaining_out:
            prepared = _limit_prepared_out(prepared, remaining_out)

        if not _is_positive(prepared.step_out) or not _is_positive(prepared.step_in):
            next_rows.extend(
                _without_became_unfunded_visible_residuals(
                    rows[index + 1 :],
                    became_unfunded_offer_ids,
                )
            )
            break

        saved_ins.append(prepared.step_in)
        saved_outs.append(prepared.step_out)
        total_out = _sum_amounts_sorted(saved_outs, zero_like=out_cap)
        total_in = _sum_amounts_sorted(saved_ins, zero_like=rows[0].segment.in_at_out_max)
        remaining_out = _amount_sub(out_cap, total_out)
        trace.append(
            _trace_row_with_gross_input(
                src=row.segment.src,
                source_id=row.source_id,
                out_take=prepared.step_out,
                net_in_take=prepared.raw_segment.in_at_out_max,
                gross_in_take=prepared.step_in,
                owner_out_take=prepared.owner_gives,
            )
        )
        offer_attempted = True
        _mark_became_unfunded_after_consume(prepared, became_unfunded_offer_ids)

        next_row = _next_row_after_consume(prepared)
        if next_row is None and row.book_offer is not None and not row.book_offer.book_visible_only:
            became_unfunded_offer_ids.add(row.book_offer.offer_id)
        if next_row is not None:
            next_rows.append(next_row)
        if remaining_out.is_zero():
            next_rows.extend(
                _without_became_unfunded_visible_residuals(
                    rows[index + 1 :],
                    became_unfunded_offer_ids,
                )
            )
            break

    return total_out, total_in, _without_became_unfunded_visible_residuals(next_rows, became_unfunded_offer_ids), trace


def _consume_clob_exact_in(
    rows: list[_StepRow],
    *,
    in_budget: Amount,
    out_cap: Amount,
    limit_quality: Quality | None,
    mode: Literal["payment", "offer_crossing"],
    source_account: str | None,
    book_in_transfer_rate: IOUAmount | None = None,
    book_out_transfer_rate: IOUAmount | None = None,
    charge_input_transfer_fee: bool = True,
    owner_pays_transfer_fee: bool = False,
    enable_self_cross_filter: bool = False,
    fix_reduced_offers_v2_enabled: bool = True,
) -> tuple[Amount, Amount, list[_StepRow], list[dict[str, object]]]:
    if not rows:
        zero_out = _zero_like(out_cap)
        return zero_out, _zero_like(in_budget), [], []

    current_quality: Quality | None = None
    offer_attempted = False
    remaining_in = in_budget
    total_out = _zero_like(out_cap)
    total_in = _zero_like(in_budget)
    saved_ins: list[Amount] = []
    saved_outs: list[Amount] = []
    next_rows: list[_StepRow] = []
    trace: list[dict[str, object]] = []
    became_unfunded_offer_ids: set[str] = set()

    for index, row in enumerate(rows):
        if remaining_in.is_zero():
            next_rows.extend(_without_became_unfunded_visible_residuals(rows[index:], became_unfunded_offer_ids))
            break
        if current_quality is None:
            current_quality = row.quoted_quality
        elif row.quoted_quality != current_quality:
            if not offer_attempted:
                current_quality = row.quoted_quality
            else:
                next_rows.extend(
                    _without_became_unfunded_visible_residuals(
                        rows[index:],
                        became_unfunded_offer_ids,
                    )
                )
                break

        if _should_delete_self_cross_offer(
            row,
            mode=mode,
            source_account=source_account,
            limit_quality=limit_quality,
            enable_self_cross_filter=enable_self_cross_filter,
        ):
            trace.append(_trace_delete_row(row))
            if not offer_attempted:
                current_quality = None
            continue

        if row.quoted_quality != current_quality:
            next_rows.extend(
                _without_became_unfunded_visible_residuals(
                    rows[index:],
                    became_unfunded_offer_ids,
                )
            )
            break

        prepared = _prepare_row(
            row,
            mode=mode,
            source_account=source_account,
            book_in_transfer_rate=book_in_transfer_rate,
            book_out_transfer_rate=book_out_transfer_rate,
            charge_input_transfer_fee=charge_input_transfer_fee,
            owner_pays_transfer_fee=owner_pays_transfer_fee,
            enable_self_cross_filter=enable_self_cross_filter,
        )
        if prepared is None:
            continue

        original_prepared = prepared
        limited_by_input = prepared.step_in > remaining_in
        if limited_by_input:
            prepared = _limit_prepared_in(
                prepared,
                remaining_in,
                fix_reduced_offers_v2_enabled=fix_reduced_offers_v2_enabled,
            )
            if prepared is None:
                continue

        if not _is_positive(prepared.step_out) or not _is_positive(prepared.step_in):
            next_rows.extend(
                _without_became_unfunded_visible_residuals(
                    rows[index + 1 :],
                    became_unfunded_offer_ids,
                )
            )
            break

        saved_outs.append(prepared.step_out)
        if limited_by_input:
            saved_ins.append(remaining_in)
            total_out = _sum_amounts_sorted(saved_outs, zero_like=out_cap)
            total_in = in_budget
        else:
            saved_ins.append(prepared.step_in)
            total_out = _sum_amounts_sorted(saved_outs, zero_like=out_cap)
            total_in = _sum_amounts_sorted(saved_ins, zero_like=in_budget)

        if total_out > out_cap and total_in <= in_budget:
            # Mirror rippled BookStep::fwdImp(): forward execution is driven by
            # input first, then clamped back to the reverse-pass cache only when
            # limitStepOut() requires exactly the remaining input.
            last_out = saved_outs.pop()
            previous_out = _sum_amounts_sorted(saved_outs, zero_like=out_cap)
            remaining_out_for_cache = _amount_sub(out_cap, previous_out)
            if _is_positive(remaining_out_for_cache):
                corrected = _limit_prepared_out(original_prepared, remaining_out_for_cache)
                if corrected.step_in == remaining_in:
                    prepared = corrected
                    total_in = in_budget
                    total_out = out_cap
                    saved_ins = [total_in]
                    saved_outs = [total_out]
                else:
                    saved_outs.append(last_out)
            else:
                saved_outs.append(last_out)

        remaining_in = _amount_sub(in_budget, total_in)
        trace.append(
            _trace_row_with_gross_input(
                src=row.segment.src,
                source_id=row.source_id,
                out_take=prepared.step_out,
                net_in_take=prepared.raw_segment.in_at_out_max,
                gross_in_take=prepared.step_in,
                owner_out_take=prepared.owner_gives,
            )
        )
        offer_attempted = True
        _mark_became_unfunded_after_consume(prepared, became_unfunded_offer_ids)

        next_row = _next_row_after_consume(prepared)
        if next_row is None and row.book_offer is not None and not row.book_offer.book_visible_only:
            became_unfunded_offer_ids.add(row.book_offer.offer_id)
        if next_row is not None:
            next_rows.append(next_row)
        if remaining_in.is_zero():
            next_rows.extend(
                _without_became_unfunded_visible_residuals(
                    rows[index + 1 :],
                    became_unfunded_offer_ids,
                )
            )
            break

    return total_out, total_in, _without_became_unfunded_visible_residuals(next_rows, became_unfunded_offer_ids), trace


def _drop_leading_self_cross_deletes(
    rows: list[_StepRow],
    *,
    limit_quality: Quality | None,
    mode: Literal["payment", "offer_crossing"],
    source_account: str | None,
    enable_self_cross_filter: bool = False,
) -> tuple[list[_StepRow], list[dict[str, object]]]:
    if not rows:
        return rows, []
    current_quality: Quality | None = None
    index = 0
    trace: list[dict[str, object]] = []
    while index < len(rows):
        row = rows[index]
        if current_quality is None:
            current_quality = row.quoted_quality
        elif row.quoted_quality != current_quality:
            break
        if not _should_delete_self_cross_offer(
            row,
            mode=mode,
            source_account=source_account,
            limit_quality=limit_quality,
            enable_self_cross_filter=enable_self_cross_filter,
        ):
            break
        trace.append(_trace_delete_row(row))
        index += 1
    return (rows[index:] if index else rows), trace


def _competing_amm_segment(
    amm: AMM | None,
    *,
    top_quality: Quality,
    out_cap: Amount | None,
    mode: Literal["payment", "offer_crossing"],
    limit_quality: Quality | None,
    require_post_spq_floor: bool,
    fix_amm_v1_1_enabled: bool = True,
) -> Segment | None:
    if amm is None:
        return None

    if mode == "offer_crossing":
        threshold = _offer_crossing_amm_quality_threshold(
            lob_quality=top_quality,
            limit_quality=limit_quality,
            fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
        )
        if threshold is None:
            return amm.max_offer_segment(max_out_cap=out_cap)
        top_quality = threshold
    # Match rippled BookPaymentStep::tip()/getAMMOffer(): for payments, the
    # synthetic AMM offer competes against the current CLOB tip based on the
    # generated offer quality, not on raw SPQ alone. A payment-side AMM slice
    # can beat the next CLOB tier even when SPQ itself is worse than the tier.
    if mode == "payment":
        segment = _amm_offer_for_clob_quality(
            amm,
            top_quality,
            max_out_cap=None,
            require_post_spq_floor=require_post_spq_floor,
        )
        if segment is None:
            return None
        return segment if segment.quality >= top_quality else None

    spq = amm.spq_quality_int()
    if not (spq > top_quality):
        return None
    if _within_relative_distance(spq, top_quality):
        return None
    segment = _amm_offer_for_clob_quality(
        amm,
        top_quality,
        max_out_cap=None if mode == "offer_crossing" else out_cap,
        require_post_spq_floor=require_post_spq_floor,
    )
    if segment is None:
        return None
    if not (segment.quality >= top_quality):
        return None
    return segment


def _amm_take_exact_out(
    amm: AMM,
    out_req: Amount,
    *,
    use_actual_out: bool = False,
) -> tuple[Amount, Amount]:
    try:
        in_take = amm.dx_for_out_st(out_req)
        if use_actual_out:
            # Payment-side AMM exact-out rounds input up to the executable input
            # quantum. The ledger-visible pool delta is the output produced by
            # that rounded input.
            return amm.swap_out_given_in_st(in_take), in_take
        return out_req, in_take
    except AMMOverAsk as over_ask:
        return over_ask.max_out, over_ask.dx_for_max


def _amm_take_under_budget(
    amm: AMM,
    *,
    out_cap: Amount,
    in_budget: Amount,
    input_transfer_rate: IOUAmount | None = None,
) -> tuple[Amount, Amount, Amount]:
    if not _same_amount_type(out_cap, amm.max_out_amount()):
        raise RuntimeError("AMM output cap domain mismatch")
    raw_in_budget = _remove_rate(in_budget, input_transfer_rate, round_up=False)
    if _amount_units_floor(raw_in_budget) <= 0:
        return _zero_like(out_cap), _zero_like(in_budget), _zero_like(raw_in_budget)
    exact_in_for_cap = amm.swap_in_given_out_st(out_cap)
    if _amount_units_floor(exact_in_for_cap) <= 0:
        return _zero_like(out_cap), _zero_like(in_budget), _zero_like(raw_in_budget)
    if _amount_units_floor(raw_in_budget) >= _amount_units_floor(exact_in_for_cap):
        return out_cap, _apply_rate(exact_in_for_cap, input_transfer_rate, round_up=True), exact_in_for_cap
    out_take = amm.swap_out_given_in_st(raw_in_budget)
    if _amount_units_floor(out_take) <= 0:
        return _zero_like(out_cap), _zero_like(in_budget), _zero_like(raw_in_budget)
    # Match BookStep::limitStepIn(): when the step input limit binds, stpAmt.in
    # is set to the gross limit, then raw offer input is derived with
    # mulRatio(gross, QUALITY_ONE, transferRate, roundDown). Re-multiplying the
    # raw amount and rejecting on a 1-unit round-up overshoot incorrectly drops
    # valid AMM tails.
    return out_take, in_budget, raw_in_budget


def _amm_raw_in_from_gross(
    gross_in: Amount,
    *,
    input_transfer_rate: IOUAmount | None,
) -> Amount:
    return _remove_rate(gross_in, input_transfer_rate, round_up=False)


def _amm_take_exact_out_with_rate(
    amm: AMM,
    out_req: Amount,
    *,
    input_transfer_rate: IOUAmount | None,
    use_actual_out: bool = False,
) -> tuple[Amount, Amount, Amount]:
    out_take, raw_in = _amm_take_exact_out(
        amm,
        out_req,
        use_actual_out=use_actual_out,
    )
    gross_in = _apply_rate(raw_in, input_transfer_rate, round_up=True)
    return out_take, gross_in, raw_in


def _amm_trace_row(
    *,
    out_take: Amount,
    raw_in_take: Amount,
    gross_in_take: Amount,
    quality: Quality | None = None,
) -> dict[str, object]:
    row = _trace_row_with_gross_input(
        src="AMM",
        source_id=None,
        out_take=out_take,
        net_in_take=raw_in_take,
        gross_in_take=gross_in_take,
    )
    if quality is not None:
        row["quality"] = quality
    return row


def _offer_crossing_amm_only_take(
    *,
    amm: AMM,
    target_out: Amount,
    send_max: Amount | None,
    limit_quality: Quality,
    input_transfer_rate: IOUAmount | None = None,
) -> tuple[Amount, Amount, Amount, Quality] | None:
    max_offer = amm.max_offer_segment(max_out_cap=target_out)
    if max_offer is None:
        return None
    # In offer crossing, when no CLOB row remains crossable, rippled still
    # applies StrandFlow::limitOut() to the single-path AMM quality function.
    # Use that limitOut amount as the AMM maxOffer cap; generating a synthetic
    # offer directly at limitQuality is too conservative after AMM rounding.
    limited_out, _ = _limit_out_by_quality_function(
        [],
        amm=amm,
        remaining_out=target_out,
        limit_quality=limit_quality,
        mode="offer_crossing",
        source_account=None,
    )
    max_offer = amm.max_offer_segment(max_out_cap=limited_out)
    if max_offer is None:
        return None
    if send_max is None:
        out_take = max_offer.out_max
        raw_in_take = max_offer.in_at_out_max
        gross_in_take = _apply_rate(raw_in_take, input_transfer_rate, round_up=True)
    else:
        out_take, gross_in_take, raw_in_take = _amm_take_under_budget(
            amm,
            out_cap=max_offer.out_max,
            in_budget=send_max,
            input_transfer_rate=input_transfer_rate,
        )
    if not _is_positive(out_take) or not _is_positive(gross_in_take) or not _is_positive(raw_in_take):
        return None
    return out_take, gross_in_take, raw_in_take, max_offer.quality


def _quality_upper_bound_for_state(
    rows: list[_StepRow],
    *,
    amm: AMM | None,
    mode: Literal["payment", "offer_crossing"],
    limit_quality: Quality | None,
    fix_amm_v1_1_enabled: bool = True,
) -> Quality | None:
    # Match rippled BookStep::qualityUpperBound(): it uses BookTip, so ledger
    # visible CLOB offers can constrain the strand before OfferStream skips
    # unfunded rows during actual execution.
    sorted_rows = _sorted_rows(rows)
    lob_quality = sorted_rows[0].quoted_quality if sorted_rows else None
    if amm is None:
        return lob_quality

    if mode == "offer_crossing":
        if lob_quality is not None:
            threshold = _offer_crossing_amm_quality_threshold(
                lob_quality=lob_quality,
                limit_quality=limit_quality,
                fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
            )
            if threshold is None:
                amm_offer = amm.max_offer_segment(max_out_cap=amm.max_out_amount())
            else:
                amm_offer = _amm_offer_for_clob_quality(
                    amm,
                    threshold,
                    max_out_cap=amm.max_out_amount(),
                    require_post_spq_floor=True,
                )
            if amm_offer is not None and amm_offer.quality > lob_quality:
                return amm_offer.quality
            return lob_quality

        amm_offer = amm.max_offer_segment(max_out_cap=amm.max_out_amount())
        return amm_offer.quality if amm_offer is not None else None

    best = lob_quality if lob_quality is not None else Quality(0)
    amm_spq = amm.spq_quality_int()
    return amm_spq if amm_spq > best else best


def _execute_one_iteration(
    rows: list[_StepRow],
    *,
    target_out: Amount,
    send_max: Amount | None,
    limit_quality: Quality | None,
    mode: Literal["payment", "offer_crossing"],
    amm: AMM | None,
    source_account: str | None = None,
    book_in_transfer_rate: IOUAmount | None = None,
    book_out_transfer_rate: IOUAmount | None = None,
    charge_input_transfer_fee: bool = True,
    owner_pays_transfer_fee: bool = False,
    enable_self_cross_filter: bool = False,
    fix_amm_v1_1_enabled: bool = True,
    fix_reduced_offers_v2_enabled: bool = True,
) -> tuple[Amount, Amount, list[_StepRow], Amount, Amount, list[dict[str, object]]]:
    active_book_in_transfer_rate = book_in_transfer_rate if charge_input_transfer_fee else None
    visible_rows = _sorted_rows(rows)
    sorted_rows = _offer_stream_rows(
        visible_rows,
        mode=mode,
        source_account=source_account,
        book_in_transfer_rate=book_in_transfer_rate,
        book_out_transfer_rate=book_out_transfer_rate,
        charge_input_transfer_fee=charge_input_transfer_fee,
        owner_pays_transfer_fee=owner_pays_transfer_fee,
        enable_self_cross_filter=enable_self_cross_filter,
    )
    executable_tip_quality = sorted_rows[0].quoted_quality if sorted_rows else None
    if mode == "offer_crossing":
        rows = _filtered_rows(sorted_rows, _offer_crossing_candidate_floor(limit_quality))
    else:
        rows = sorted_rows
    zero_out = _zero_like(target_out)
    zero_in = _zero_like(send_max) if send_max is not None else (zero_out if not rows else _zero_like(rows[0].segment.in_at_out_max))

    if not rows:
        if amm is None:
            return zero_out, zero_in, [], zero_in, zero_out, []
        if mode == "offer_crossing" and executable_tip_quality is not None:
            amm_segment = _competing_amm_segment(
                amm,
                top_quality=executable_tip_quality,
                out_cap=target_out,
                mode=mode,
                limit_quality=limit_quality,
                require_post_spq_floor=True,
                fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
            )
            if amm_segment is None:
                return zero_out, zero_in, [], _zero_like(amm._x_amount), _zero_like(amm._y_amount), []
            if send_max is None:
                out_take = _min_amount(amm_segment.out_max, target_out)
                raw_in_take = (
                    amm_segment.in_at_out_max
                    if out_take == amm_segment.out_max
                    else amm.swap_in_given_out_st(out_take)
                )
                gross_in_take = _apply_rate(raw_in_take, active_book_in_transfer_rate, round_up=True)
            else:
                out_cap = _min_amount(amm_segment.out_max, target_out)
                out_take, gross_in_take, raw_in_take = _amm_take_under_budget(
                    amm,
                    out_cap=out_cap,
                    in_budget=send_max,
                    input_transfer_rate=active_book_in_transfer_rate,
                )
            if not _is_positive(out_take) or not _is_positive(gross_in_take) or not _is_positive(raw_in_take):
                return zero_out, zero_in, [], _zero_like(amm._x_amount), _zero_like(amm._y_amount), []
            if _offer_crossing_offer_rejected_by_threshold(
                mode=mode,
                limit_quality=limit_quality,
                offer_quality=amm_segment.quality,
            ):
                return zero_out, zero_in, [], _zero_like(amm._x_amount), _zero_like(amm._y_amount), []
            return (
                out_take,
                gross_in_take,
                [],
                raw_in_take,
                out_take,
                [_amm_trace_row(
                    out_take=out_take,
                    raw_in_take=raw_in_take,
                    gross_in_take=gross_in_take,
                    quality=amm_segment.quality,
                )],
            )
        if mode == "offer_crossing":
            if limit_quality is not None:
                limited_take = _offer_crossing_amm_only_take(
                    amm=amm,
                    target_out=target_out,
                    send_max=send_max,
                    limit_quality=limit_quality,
                    input_transfer_rate=active_book_in_transfer_rate,
                )
                if limited_take is None:
                    return zero_out, zero_in, [], _zero_like(amm._x_amount), _zero_like(amm._y_amount), []
                out_take, gross_in_take, raw_in_take, offer_quality = limited_take
                if _offer_crossing_offer_rejected_by_threshold(
                    mode=mode,
                    limit_quality=limit_quality,
                    offer_quality=offer_quality,
                ):
                    return zero_out, zero_in, [], _zero_like(amm._x_amount), _zero_like(amm._y_amount), []
                return (
                    out_take,
                    gross_in_take,
                    [],
                    raw_in_take,
                    out_take,
                    [_amm_trace_row(
                        out_take=out_take,
                        raw_in_take=raw_in_take,
                        gross_in_take=gross_in_take,
                        quality=offer_quality,
                    )],
                )
            max_offer = amm.max_offer_segment(max_out_cap=target_out)
            if max_offer is None:
                return zero_out, zero_in, [], _zero_like(amm._x_amount), _zero_like(amm._y_amount), []
            out_cap = max_offer.out_max
            amm_trace_quality = max_offer.quality
        elif limit_quality is not None:
            segment = amm.synthetic_segment_for_quality(
                limit_quality,
                max_out_cap=target_out,
                require_post_spq_floor=True,
            )
            if segment is None:
                return zero_out, zero_in, [], _zero_like(amm._x_amount), _zero_like(amm._y_amount), []
            out_cap = _min_amount(segment.out_max, target_out)
            amm_trace_quality = segment.quality
        else:
            max_offer = amm.max_offer_segment(max_out_cap=target_out)
            if max_offer is None:
                return zero_out, zero_in, [], _zero_like(amm._x_amount), _zero_like(amm._y_amount), []
            out_cap = max_offer.out_max
            amm_trace_quality = max_offer.quality

        if send_max is None:
            out_take, gross_in_take, raw_in_take = _amm_take_exact_out_with_rate(
                amm,
                out_cap,
                input_transfer_rate=active_book_in_transfer_rate,
                use_actual_out=mode == "payment",
            )
        else:
            out_take, gross_in_take, raw_in_take = _amm_take_under_budget(
                amm,
                out_cap=out_cap,
                in_budget=send_max,
                input_transfer_rate=active_book_in_transfer_rate,
            )

        if not _is_positive(out_take) or not _is_positive(gross_in_take) or not _is_positive(raw_in_take):
            return zero_out, zero_in, [], _zero_like(amm._x_amount), _zero_like(amm._y_amount), []
        if _offer_crossing_offer_rejected_by_threshold(
            mode=mode,
            limit_quality=limit_quality,
            offer_quality=amm_trace_quality,
        ):
            return zero_out, zero_in, [], _zero_like(amm._x_amount), _zero_like(amm._y_amount), []
        return (
            out_take,
            gross_in_take,
            [],
            raw_in_take,
            out_take,
            [_amm_trace_row(
                out_take=out_take,
                raw_in_take=raw_in_take,
                gross_in_take=gross_in_take,
                quality=amm_trace_quality,
            )],
        )

    if mode == "offer_crossing" and executable_tip_quality is not None and rows != sorted_rows:
        # Rippled calls tryAMM(offers.tip().quality()) before
        # checkQualityThreshold() can stop CLOB crossing. The tip here is the
        # OfferStream executable tip, not the raw BookTip used by limitOut().
        amm_segment = _competing_amm_segment(
            amm,
            top_quality=executable_tip_quality,
            out_cap=target_out,
            mode=mode,
            limit_quality=limit_quality,
            require_post_spq_floor=True,
            fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
        )
        if amm_segment is not None and amm is not None:
            if send_max is None:
                out_take = _min_amount(amm_segment.out_max, target_out)
                raw_in_take = (
                    amm_segment.in_at_out_max
                    if out_take == amm_segment.out_max
                    else amm.swap_in_given_out_st(out_take)
                )
                gross_in_take = _apply_rate(raw_in_take, active_book_in_transfer_rate, round_up=True)
            else:
                out_cap = _min_amount(amm_segment.out_max, target_out)
                out_take, gross_in_take, raw_in_take = _amm_take_under_budget(
                    amm,
                    out_cap=out_cap,
                    in_budget=send_max,
                    input_transfer_rate=active_book_in_transfer_rate,
                )

            if _is_positive(out_take) and _is_positive(gross_in_take) and _is_positive(raw_in_take):
                if _offer_crossing_offer_rejected_by_threshold(
                    mode=mode,
                    limit_quality=limit_quality,
                    offer_quality=amm_segment.quality,
                ):
                    return zero_out, zero_in, rows, _zero_like(amm._x_amount), _zero_like(amm._y_amount), []
                return (
                    out_take,
                    gross_in_take,
                    rows,
                    raw_in_take,
                    out_take,
                    [_amm_trace_row(
                        out_take=out_take,
                        raw_in_take=raw_in_take,
                        gross_in_take=gross_in_take,
                        quality=amm_segment.quality,
                    )],
                )

        if not rows:
            return zero_out, zero_in, [], _zero_like(sorted_rows[0].segment.in_at_out_max), _zero_like(target_out), []

    top_quality = rows[0].quoted_quality
    amm_segment = _competing_amm_segment(
        amm,
        top_quality=top_quality,
        out_cap=target_out,
        mode=mode,
        limit_quality=limit_quality,
        require_post_spq_floor=True,
        fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
    )

    if amm_segment is not None and amm is not None:
        if send_max is None:
            out_take = _min_amount(amm_segment.out_max, target_out)
            raw_in_take = (
                amm_segment.in_at_out_max
                if out_take == amm_segment.out_max
                else amm.swap_in_given_out_st(out_take)
            )
            gross_in_take = _apply_rate(raw_in_take, active_book_in_transfer_rate, round_up=True)
        else:
            out_cap = _min_amount(amm_segment.out_max, target_out)
            out_take, gross_in_take, raw_in_take = _amm_take_under_budget(
                amm,
                out_cap=out_cap,
                in_budget=send_max,
                input_transfer_rate=active_book_in_transfer_rate,
            )

        if _is_positive(out_take) and _is_positive(gross_in_take) and _is_positive(raw_in_take):
            if _offer_crossing_offer_rejected_by_threshold(
                mode=mode,
                limit_quality=limit_quality,
                offer_quality=amm_segment.quality,
            ):
                return zero_out, zero_in, rows, _zero_like(amm._x_amount), _zero_like(amm._y_amount), []
            return (
                out_take,
                gross_in_take,
                rows,
                raw_in_take,
                out_take,
                [_amm_trace_row(
                    out_take=out_take,
                    raw_in_take=raw_in_take,
                    gross_in_take=gross_in_take,
                    quality=amm_segment.quality,
                )],
            )

    pruned_rows, delete_trace = _drop_leading_self_cross_deletes(
        rows,
        limit_quality=limit_quality,
        mode=mode,
        source_account=source_account,
        enable_self_cross_filter=enable_self_cross_filter,
    )
    if len(pruned_rows) != len(rows):
        # Rippled calls tryAMM() once before execOffer() can delete a
        # self-crossing tip. After such deletes, the same forEachOffer pass
        # continues through CLOB offers only; it does not generate a new AMM
        # synthetic offer for the new tip in that pass.
        if (
            mode == "offer_crossing"
            and limit_quality is not None
            and pruned_rows
            and pruned_rows[0].quoted_quality < limit_quality
        ):
            return zero_out, zero_in, pruned_rows, _zero_like(rows[0].segment.in_at_out_max), _zero_like(target_out), delete_trace
        if not pruned_rows:
            return zero_out, zero_in, [], _zero_like(rows[0].segment.in_at_out_max), _zero_like(target_out), delete_trace
        if send_max is None:
            out_take, in_take, next_rows, itr_trace = _consume_clob_exact_out(
                pruned_rows,
                out_cap=target_out,
                limit_quality=limit_quality,
                mode=mode,
                source_account=source_account,
                book_in_transfer_rate=book_in_transfer_rate,
                book_out_transfer_rate=book_out_transfer_rate,
                charge_input_transfer_fee=charge_input_transfer_fee,
                owner_pays_transfer_fee=owner_pays_transfer_fee,
                enable_self_cross_filter=enable_self_cross_filter,
                fix_reduced_offers_v2_enabled=fix_reduced_offers_v2_enabled,
            )
        else:
            out_take, in_take, next_rows, itr_trace = _consume_clob_exact_in(
                pruned_rows,
                in_budget=send_max,
                out_cap=target_out,
                limit_quality=limit_quality,
                mode=mode,
                source_account=source_account,
                book_in_transfer_rate=book_in_transfer_rate,
                book_out_transfer_rate=book_out_transfer_rate,
                charge_input_transfer_fee=charge_input_transfer_fee,
                owner_pays_transfer_fee=owner_pays_transfer_fee,
                enable_self_cross_filter=enable_self_cross_filter,
                fix_reduced_offers_v2_enabled=fix_reduced_offers_v2_enabled,
            )
        return out_take, in_take, next_rows, _zero_like(rows[0].segment.in_at_out_max), _zero_like(target_out), delete_trace + itr_trace

    if send_max is None:
        out_take, in_take, next_rows, trace = _consume_clob_exact_out(
            rows,
            out_cap=target_out,
            limit_quality=limit_quality,
            mode=mode,
            source_account=source_account,
            book_in_transfer_rate=book_in_transfer_rate,
            book_out_transfer_rate=book_out_transfer_rate,
            charge_input_transfer_fee=charge_input_transfer_fee,
            owner_pays_transfer_fee=owner_pays_transfer_fee,
            enable_self_cross_filter=enable_self_cross_filter,
            fix_reduced_offers_v2_enabled=fix_reduced_offers_v2_enabled,
        )
    else:
        out_take, in_take, next_rows, trace = _consume_clob_exact_in(
            rows,
            in_budget=send_max,
            out_cap=target_out,
            limit_quality=limit_quality,
            mode=mode,
            source_account=source_account,
            book_in_transfer_rate=book_in_transfer_rate,
            book_out_transfer_rate=book_out_transfer_rate,
            charge_input_transfer_fee=charge_input_transfer_fee,
            owner_pays_transfer_fee=owner_pays_transfer_fee,
            enable_self_cross_filter=enable_self_cross_filter,
        )

    return out_take, in_take, next_rows, _zero_like(rows[0].segment.in_at_out_max), _zero_like(target_out), trace


@dataclass(slots=True)
class BookStep(Step):
    segments_provider: Callable[[], Iterable[Segment | BookOfferSegment]]
    amm: AMM | None = None
    limit_quality: Quality | None = None
    mode: Literal["payment", "offer_crossing"] = "payment"
    source_account: str | None = None
    book_in_transfer_rate: IOUAmount | None = None
    book_out_transfer_rate: IOUAmount | None = None
    charge_input_transfer_fee: bool = True
    owner_pays_transfer_fee: bool = False
    enable_self_cross_filter: bool = False
    fix_amm_v1_1_enabled: bool = True
    fix_reduced_offers_v2_enabled: bool = True
    apply_quality_limit_out: bool = True
    enforce_limit_quality_result: bool = True

    @classmethod
    def from_static(
        cls,
        segments: list[Segment | BookOfferSegment],
        *,
        amm: AMM | None = None,
        limit_quality: Quality | None = None,
        mode: Literal["payment", "offer_crossing"] = "payment",
        source_account: str | None = None,
        book_in_transfer_rate: IOUAmount | None = None,
        book_out_transfer_rate: IOUAmount | None = None,
        charge_input_transfer_fee: bool = True,
        owner_pays_transfer_fee: bool = False,
        enable_self_cross_filter: bool = False,
        fix_amm_v1_1_enabled: bool = True,
        fix_reduced_offers_v2_enabled: bool = True,
        apply_quality_limit_out: bool = True,
        enforce_limit_quality_result: bool = True,
    ) -> "BookStep":
        return cls(
            segments_provider=lambda: segments,
            amm=amm,
            limit_quality=limit_quality,
            mode=mode,
            source_account=source_account,
            book_in_transfer_rate=book_in_transfer_rate,
            book_out_transfer_rate=book_out_transfer_rate,
            charge_input_transfer_fee=charge_input_transfer_fee,
            owner_pays_transfer_fee=owner_pays_transfer_fee,
            enable_self_cross_filter=enable_self_cross_filter,
            fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
            fix_reduced_offers_v2_enabled=fix_reduced_offers_v2_enabled,
            apply_quality_limit_out=apply_quality_limit_out,
            enforce_limit_quality_result=enforce_limit_quality_result,
        )

    def rev(
        self,
        sandbox: "PaymentSandbox" | None,
        out_req: Amount,
        return_trace: bool = False,
        require_full_fill: bool = False,
    ) -> tuple[Amount, Amount] | tuple[Amount, Amount, list[dict[str, object]]]:
        del sandbox
        total_out = _zero_like(out_req)
        rows = _normalise_rows(self.segments_provider())
        amm_exec = _AMMAggregateExecution.from_amm(self.amm) if self.amm is not None else None
        shadow_amm = amm_exec.shadow_amm() if amm_exec is not None else None
        total_in = _zero_like(rows[0].segment.in_at_out_max) if rows else (
            _zero_like(shadow_amm._x_amount) if shadow_amm is not None else _zero_like(out_req)
        )
        trace: list[dict[str, object]] = []
        remaining_out = out_req
        amm_consumed = False

        for _ in range(MAX_BOOK_STEP_ITERS):
            if remaining_out.is_zero():
                break
            if self.apply_quality_limit_out:
                target_out, adjusted_out = _limit_out_by_quality_function(
                    rows,
                    amm=None if amm_consumed else shadow_amm,
                    remaining_out=remaining_out,
                    limit_quality=self.limit_quality,
                    mode=self.mode,
                    source_account=self.source_account,
                    book_in_transfer_rate=self.book_in_transfer_rate,
                    book_out_transfer_rate=self.book_out_transfer_rate,
                    charge_input_transfer_fee=self.charge_input_transfer_fee,
                    owner_pays_transfer_fee=self.owner_pays_transfer_fee,
                    enable_self_cross_filter=self.enable_self_cross_filter,
                    fix_amm_v1_1_enabled=self.fix_amm_v1_1_enabled,
                )
            else:
                target_out, adjusted_out = remaining_out, False
            next_filled_out, next_spent_in, next_rows, amm_in, amm_out, itr_trace = _execute_one_iteration(
                rows,
                target_out=target_out,
                send_max=None,
                limit_quality=self.limit_quality,
                mode=self.mode,
                amm=None if amm_consumed else shadow_amm,
                source_account=self.source_account,
                book_in_transfer_rate=self.book_in_transfer_rate,
                book_out_transfer_rate=self.book_out_transfer_rate,
                charge_input_transfer_fee=self.charge_input_transfer_fee,
                owner_pays_transfer_fee=self.owner_pays_transfer_fee,
                enable_self_cross_filter=self.enable_self_cross_filter,
                fix_amm_v1_1_enabled=self.fix_amm_v1_1_enabled,
                fix_reduced_offers_v2_enabled=self.fix_reduced_offers_v2_enabled,
            )
            if not _is_positive(next_filled_out) or not _is_positive(next_spent_in):
                break
            itr_quality = None
            if self.enforce_limit_quality_result and self.limit_quality is not None:
                itr_quality = Quality.from_amounts(next_filled_out, next_spent_in)
                if _iteration_violates_limit_quality(
                    mode=self.mode,
                    limit_quality=self.limit_quality,
                    itr_quality=itr_quality,
                    adjusted_out=adjusted_out,
                    amm_in=amm_in,
                    amm_out=amm_out,
                ):
                    break
            filled_out = next_filled_out
            spent_in = next_spent_in
            rows = next_rows
            if amm_exec is not None and _is_positive(amm_in) and _is_positive(amm_out):
                _, _, shadow_amm = amm_exec.apply_exact_out(
                    raw_in=amm_in,
                    raw_out=amm_out,
                )
                filled_out = amm_out
                amm_consumed = True
                itr_trace = []
            total_out = total_out + filled_out
            total_in = total_in + spent_in
            remaining_out = remaining_out - filled_out
            trace.extend(itr_trace)
            if self.mode in {"payment", "offer_crossing"}:
                # Match rippled BookStep::forEachOffer(): one BookStep call
                # processes at most the current best-quality bucket (or one
                # synthetic AMM offer). Outer flow / offer-crossing logic is
                # responsible for iterating across subsequent quality tiers.
                break

        self._cached_out = total_out
        self._cached_in = total_in
        if require_full_fill and total_out < out_req:
            raise RuntimeError("insufficient liquidity for requested output")
        if amm_exec is not None and amm_exec.has_fill():
            trace.append(amm_exec.merged_trace_row())
        if return_trace:
            return total_out, total_in, trace
        return total_out, total_in

    def fwd(
        self,
        sandbox: "PaymentSandbox" | None,
        in_cap: Amount,
        return_trace: bool = False,
    ) -> tuple[Amount, Amount] | tuple[Amount, Amount, list[dict[str, object]]]:
        rows = _normalise_rows(self.segments_provider())
        amm_exec = _AMMAggregateExecution.from_amm(self.amm) if self.amm is not None else None
        shadow_amm = amm_exec.shadow_amm() if amm_exec is not None else None
        if hasattr(self, "_cached_out") and _is_positive(self._cached_out):
            target_out = self._cached_out
        else:
            clob_target_out = None
            if rows:
                filtered_rows = _filtered_rows(_sorted_rows(rows), self.limit_quality)
                if filtered_rows:
                    clob_target_out = _clob_out_capacity(filtered_rows, rows[0].segment.out_max)

            amm_target_out = None
            if shadow_amm is not None:
                # Rippled does not pre-trim the forward exact-in AMM target via
                # a synthetic offer boundary here. Single-path limit-quality
                # enforcement happens later through the strand quality function
                # limitOut() cap. Pre-capping here makes AMM-only offer crossing
                # materially too conservative.
                amm_target_out = shadow_amm.swap_out_given_in_st(in_cap)
                if not _is_positive(amm_target_out):
                    amm_target_out = None

            if clob_target_out is not None and amm_target_out is not None:
                target_out = clob_target_out + amm_target_out
            elif clob_target_out is not None:
                target_out = clob_target_out
            elif amm_target_out is not None:
                target_out = amm_target_out
            else:
                target_out = _zero_like(in_cap)

        total_in = _zero_like(in_cap)
        total_out = _zero_like(target_out)
        trace: list[dict[str, object]] = []
        remaining_in = in_cap
        remaining_out = target_out
        amm_consumed = False

        for _ in range(MAX_BOOK_STEP_ITERS):
            if remaining_in.is_zero() or remaining_out.is_zero():
                break
            if self.apply_quality_limit_out:
                target_out_cap, adjusted_out = _limit_out_by_quality_function(
                    rows,
                    amm=None if amm_consumed else shadow_amm,
                    remaining_out=remaining_out,
                    limit_quality=self.limit_quality,
                    mode=self.mode,
                    source_account=self.source_account,
                    book_in_transfer_rate=self.book_in_transfer_rate,
                    book_out_transfer_rate=self.book_out_transfer_rate,
                    charge_input_transfer_fee=self.charge_input_transfer_fee,
                    owner_pays_transfer_fee=self.owner_pays_transfer_fee,
                    enable_self_cross_filter=self.enable_self_cross_filter,
                    fix_amm_v1_1_enabled=self.fix_amm_v1_1_enabled,
                )
            else:
                target_out_cap, adjusted_out = remaining_out, False
            next_filled_out, next_spent_in, next_rows, amm_in, amm_out, itr_trace = _execute_one_iteration(
                rows,
                target_out=target_out_cap,
                send_max=remaining_in,
                limit_quality=self.limit_quality,
                mode=self.mode,
                amm=None if amm_consumed else shadow_amm,
                source_account=self.source_account,
                book_in_transfer_rate=self.book_in_transfer_rate,
                book_out_transfer_rate=self.book_out_transfer_rate,
                charge_input_transfer_fee=self.charge_input_transfer_fee,
                owner_pays_transfer_fee=self.owner_pays_transfer_fee,
                enable_self_cross_filter=self.enable_self_cross_filter,
                fix_amm_v1_1_enabled=self.fix_amm_v1_1_enabled,
                fix_reduced_offers_v2_enabled=self.fix_reduced_offers_v2_enabled,
            )
            if not _is_positive(next_filled_out) or not _is_positive(next_spent_in):
                break
            itr_quality = None
            if self.enforce_limit_quality_result and self.limit_quality is not None:
                itr_quality = Quality.from_amounts(next_filled_out, next_spent_in)
                if _iteration_violates_limit_quality(
                    mode=self.mode,
                    limit_quality=self.limit_quality,
                    itr_quality=itr_quality,
                    adjusted_out=adjusted_out,
                    amm_in=amm_in,
                    amm_out=amm_out,
                ):
                    break
            filled_out = next_filled_out
            spent_in = next_spent_in
            rows = next_rows
            if amm_exec is not None and _is_positive(amm_in) and _is_positive(amm_out):
                _, _, shadow_amm = amm_exec.apply_exact_in(
                    raw_in=amm_in,
                    raw_out=amm_out,
                )
                filled_out = amm_out
                amm_consumed = True
                itr_trace = []
            total_out = total_out + filled_out
            total_in = total_in + spent_in
            remaining_out = remaining_out - filled_out
            remaining_in = remaining_in - spent_in
            trace.extend(itr_trace)
            if self.mode in {"payment", "offer_crossing"}:
                # Match rippled BookStep::forEachOffer(): one BookStep call
                # processes at most the current best-quality bucket (or one
                # synthetic AMM offer). Outer flow / offer-crossing logic is
                # responsible for iterating across subsequent quality tiers.
                break

        self._cached_out = total_out
        self._cached_in = total_in
        if amm_exec is not None and amm_exec.has_fill():
            trace.append(amm_exec.merged_trace_row())
            if sandbox is not None:
                sandbox.stage_after_iteration(amm_exec.cumulative_in, amm_exec.cumulative_out)
        if return_trace:
            return total_out, total_in, trace
        return total_out, total_in

    def quality_upper_bound(self) -> Quality:
        return _quality_upper_bound_for_state(
            _normalise_rows(self.segments_provider()),
            amm=self.amm,
            mode=self.mode,
            limit_quality=self.limit_quality,
            fix_amm_v1_1_enabled=self.fix_amm_v1_1_enabled,
        ) or Quality(0)


class RouterQuoteView:
    """Read-only exact-out quote view for the rebuilt single-path BookStep."""

    def __init__(
        self,
        segments_provider: Callable[[], Iterable[Segment | BookOfferSegment]],
        *,
        amm: AMM | None = None,
        limit_quality: Quality | None = None,
        mode: Literal["payment", "offer_crossing"] = "payment",
        source_account: str | None = None,
        book_in_transfer_rate: IOUAmount | None = None,
        book_out_transfer_rate: IOUAmount | None = None,
        charge_input_transfer_fee: bool = True,
        owner_pays_transfer_fee: bool = False,
        enable_self_cross_filter: bool = False,
        fix_amm_v1_1_enabled: bool = True,
        fix_reduced_offers_v2_enabled: bool = True,
        apply_quality_limit_out: bool = True,
        enforce_limit_quality_result: bool = True,
    ) -> None:
        self._step = BookStep(
            segments_provider=segments_provider,
            amm=amm,
            limit_quality=limit_quality,
            mode=mode,
            source_account=source_account,
            book_in_transfer_rate=book_in_transfer_rate,
            book_out_transfer_rate=book_out_transfer_rate,
            charge_input_transfer_fee=charge_input_transfer_fee,
            owner_pays_transfer_fee=owner_pays_transfer_fee,
            enable_self_cross_filter=enable_self_cross_filter,
            fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
            fix_reduced_offers_v2_enabled=fix_reduced_offers_v2_enabled,
            apply_quality_limit_out=apply_quality_limit_out,
            enforce_limit_quality_result=enforce_limit_quality_result,
        )

    def preview_out(self, out_req: Amount) -> dict[str, object]:
        total_out, total_in, trace = self._step.rev(None, out_req, return_trace=True)
        avg_quality = None
        if _is_positive(total_out) and _is_positive(total_in):
            avg_quality = Quality.from_amounts(total_out, total_in)

        requested_units = _amount_units_floor(out_req)
        filled_units = _amount_units_floor(total_out)
        fill_ratio = float(filled_units / requested_units) if requested_units > 0 else 1.0

        return {
            "slices": trace,
            "summary": {
                "requested_out": out_req,
                "total_out": total_out,
                "total_in": total_in,
                "avg_quality": avg_quality,
                "is_partial": total_out < out_req,
                "fill_ratio": fill_ratio,
            },
        }

    def preview_in(self, in_cap: Amount) -> dict[str, object]:
        raise NotImplementedError("RouterQuoteView.preview_in is outside rebuild phase 1 scope")


__all__ = ["BookStep", "RouterQuoteView", "MAX_BOOK_STEP_ITERS"]
