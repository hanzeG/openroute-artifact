"""Strict adapter from XRPL `book_offers` rows to kernel CLOB segments."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

from .amm import amount_sub_ripple_number
from .core import (
    IOUAmount,
    IOU_EXP_MIN,
    IOU_EXP_MAX,
    IOU_MANTISSA_MAX,
    IOU_MANTISSA_MIN,
    Quality,
    Segment,
    XRPAmount,
    parse_iou_decimal,
    parse_xrp_drops,
)

XRP = "XRP"
TEN_TO_14 = 10**14
TEN_TO_14_MINUS_1 = TEN_TO_14 - 1
TEN_TO_17 = 10**17
QUALITY_MANTISSA_MASK = 0x00FFFFFFFFFFFFFF
QUALITY_ONE = 1_000_000_000
UINT32_MAX = (2**32) - 1
XRP_MAX_DROPS = 100_000_000_000_000_000


@dataclass(frozen=True, slots=True)
class BookOfferSegment:
    offer_id: str
    quoted_quality: Quality
    segment: Segment
    owner: str | None = None
    owner_is_out_issuer: bool = False
    owner_funds: XRPAmount | IOUAmount | None = None
    in_transfer_rate: IOUAmount | None = None
    out_transfer_rate: IOUAmount | None = None
    non_fill_deleted: bool = False
    book_visible_only: bool = False

    @property
    def quality(self) -> Quality:
        return self.segment.quality

    @property
    def out_max(self) -> XRPAmount | IOUAmount:
        return self.segment.out_max

    @property
    def in_at_out_max(self) -> XRPAmount | IOUAmount:
        return self.segment.in_at_out_max


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise RuntimeError("denominator must be positive")
    return (numerator + denominator - 1) // denominator


def _log10_floor_int(value: int) -> int:
    if value <= 0:
        return -1
    return len(str(value)) - 1


def _log10_ceil_int(value: int) -> int:
    if value <= 1:
        return 0
    floor = len(str(value)) - 1
    return floor if value == 10**floor else floor + 1


def _transfer_rate_to_uint32(rate: IOUAmount) -> int | None:
    scaled = rate.as_fraction() * QUALITY_ONE
    if scaled.denominator != 1:
        return None
    value = scaled.numerator
    if value <= 0 or value > UINT32_MAX:
        return None
    return value


def _mul_iou_by_uint32_ratio(
    amount: IOUAmount,
    numerator: int,
    denominator: int,
    *,
    round_up: bool,
) -> IOUAmount:
    if denominator == 0:
        raise RuntimeError("division by zero")
    if amount.is_zero() or numerator == 0:
        return IOUAmount(0, 0)

    neg = amount.mantissa < 0
    mul = abs(amount.mantissa) * numerator
    low = mul // denominator
    rem = mul - low * denominator
    exponent = amount.exponent

    if rem:
        room_to_grow = _log10_floor_int((2**63) - 1) - _log10_ceil_int(low)
        if room_to_grow > 0:
            scale = 10**room_to_grow
            exponent -= room_to_grow
            low *= scale
            rem *= scale
        add_rem = rem // denominator
        low += add_rem
        rem -= add_rem * denominator

    has_rem = rem != 0
    must_shrink = _log10_ceil_int(low) - _log10_floor_int((2**63) - 1)
    if must_shrink > 0:
        scale = 10**must_shrink
        saved = low
        exponent += must_shrink
        low //= scale
        if not has_rem:
            has_rem = saved != low * scale

    mantissa = -low if neg else low
    result = IOUAmount(mantissa, exponent)
    if has_rem:
        if round_up and not neg:
            if result.is_zero():
                return IOUAmount(IOU_MANTISSA_MIN, IOU_EXP_MIN)
            return IOUAmount(result.mantissa + 1, result.exponent)
        if not round_up and neg:
            if result.is_zero():
                return IOUAmount(-IOU_MANTISSA_MIN, IOU_EXP_MIN)
            return IOUAmount(result.mantissa - 1, result.exponent)
    return result


def _scale_iou_by_transfer_rate(
    amount: IOUAmount,
    rate: IOUAmount,
    *,
    divide: bool,
    round_up: bool,
) -> IOUAmount | None:
    rate_value = _transfer_rate_to_uint32(rate)
    if rate_value is None:
        return None
    numerator = QUALITY_ONE if divide else rate_value
    denominator = rate_value if divide else QUALITY_ONE
    return _mul_iou_by_uint32_ratio(
        amount,
        numerator,
        denominator,
        round_up=round_up,
    )

def _parse_amount_field(
    raw: Any, *, field_name: str
) -> tuple[str, str | None, XRPAmount | IOUAmount]:
    if isinstance(raw, str):
        try:
            amount = parse_xrp_drops(raw, field_name=field_name)
        except Exception as e:
            raise RuntimeError(str(e)) from e
        if amount.drops <= 0:
            raise RuntimeError(f"invalid {field_name}: non-positive amount")
        return XRP, None, amount

    if not isinstance(raw, dict):
        raise RuntimeError(f"invalid {field_name}: expected XRP string or IOU object")

    currency_raw = raw.get("currency")
    issuer_raw = raw.get("issuer")
    value_raw = raw.get("value")
    if currency_raw is None or str(currency_raw) == "":
        raise RuntimeError(f"invalid {field_name}: missing currency")

    currency = str(currency_raw)
    issuer = None if issuer_raw is None else str(issuer_raw)
    if currency == XRP:
        raise RuntimeError(f"invalid {field_name}: XRP must use string drops encoding")
    if issuer is None or issuer == "":
        raise RuntimeError(f"invalid {field_name}: missing issuer for non-XRP amount")

    try:
        amount = parse_iou_decimal(value_raw, field_name=field_name)
    except Exception as e:
        raise RuntimeError(str(e)) from e
    if amount.mantissa <= 0:
        raise RuntimeError(f"invalid {field_name}: non-positive amount")
    return currency, issuer, amount


def _same_asset(
    left_currency: str,
    left_issuer: str | None,
    right_currency: str,
    right_issuer: str | None,
) -> bool:
    return (left_currency, left_issuer) == (right_currency, right_issuer)


def _parse_owner_funds(raw: Any, *, gets_currency: str) -> XRPAmount | IOUAmount:
    if gets_currency == XRP:
        try:
            amount = parse_xrp_drops(raw, field_name="owner_funds")
        except Exception as e:
            raise RuntimeError(str(e)) from e
        if amount.drops < 0:
            raise RuntimeError("invalid owner_funds: negative amount")
        return amount
    try:
        amount = parse_iou_decimal(raw, field_name="owner_funds")
    except Exception as e:
        raise RuntimeError(str(e)) from e
    if amount.mantissa < 0:
        raise RuntimeError("invalid owner_funds: negative amount")
    return amount


def _parse_book_offer_id(offer: dict[str, Any]) -> str:
    if offer.get("index"):
        return str(offer["index"])
    if offer.get("Account") is not None and offer.get("Sequence") is not None:
        return f"{offer['Account']}:{offer['Sequence']}"
    raise RuntimeError("offer row missing stable identifier")


def _parse_quality_from_book_directory(raw_book_directory: Any) -> Quality:
    if raw_book_directory is None:
        raise RuntimeError("missing required offer.BookDirectory in snapshot row")
    book_directory = str(raw_book_directory).strip()
    if len(book_directory) < 16:
        raise RuntimeError("invalid offer.BookDirectory: expected at least 16 hex chars")
    try:
        encoded = int(book_directory[-16:], 16)
    except ValueError as exc:
        raise RuntimeError(f"invalid offer.BookDirectory: bad hex value: {exc}") from exc
    encoded &= (0xFF << 56) | QUALITY_MANTISSA_MASK
    return Quality(encoded)


def _normalize_optional_iou_transfer_rate(
    value: Any,
    *,
    field_name: str,
) -> IOUAmount | None:
    if value is None:
        return None
    if isinstance(value, IOUAmount):
        rate = IOUAmount(value.mantissa, value.exponent)
    else:
        try:
            rate = parse_iou_decimal(value, field_name=field_name)
        except Exception as e:
            raise RuntimeError(str(e)) from e
    if rate.mantissa <= 0:
        raise RuntimeError(f"{field_name} must be strictly positive")
    if rate == IOUAmount(1, 0):
        return None
    return rate


def _same_amount_type(left: XRPAmount | IOUAmount, right: XRPAmount | IOUAmount) -> bool:
    return type(left) is type(right)


def _copy_amount(amount: XRPAmount | IOUAmount) -> XRPAmount | IOUAmount:
    if isinstance(amount, XRPAmount):
        return XRPAmount(amount.drops)
    return IOUAmount(amount.mantissa, amount.exponent)


def _zero_like(amount: XRPAmount | IOUAmount) -> XRPAmount | IOUAmount:
    if isinstance(amount, XRPAmount):
        return XRPAmount(0)
    return IOUAmount(0, 0)


def _is_positive(amount: XRPAmount | IOUAmount) -> bool:
    if isinstance(amount, XRPAmount):
        return amount.drops > 0
    return amount.mantissa > 0


def _min_amount(
    left: XRPAmount | IOUAmount, right: XRPAmount | IOUAmount
) -> XRPAmount | IOUAmount:
    if not _same_amount_type(left, right):
        raise RuntimeError("amount domain mismatch")
    return left if left <= right else right


def _working_components(amount: XRPAmount | IOUAmount) -> tuple[int, int, bool]:
    if isinstance(amount, XRPAmount):
        value = amount.drops
        exponent = 0
        integral = True
    else:
        value = amount.mantissa
        exponent = amount.exponent
        integral = False

    if value <= 0:
        raise RuntimeError("working components require a strictly positive amount")

    if integral:
        while value < IOU_MANTISSA_MIN:
            value *= 10
            exponent -= 1

    return value, exponent, integral


def _amount_to_fraction(amount: XRPAmount | IOUAmount) -> Fraction:
    if isinstance(amount, XRPAmount):
        return Fraction(amount.drops, 1)
    return Fraction(amount.mantissa, 1) * (Fraction(10, 1) ** amount.exponent)


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
    if exponent >= 0:
        denominator = value.denominator * (10**exponent)
        mantissa = _ceil_div(value.numerator, denominator) if round_up else (
            value.numerator // denominator
        )
    else:
        scale = 10 ** (-exponent)
        scaled_num = value.numerator * scale
        mantissa = _ceil_div(scaled_num, value.denominator) if round_up else (
            scaled_num // value.denominator
        )
    if mantissa <= 0:
        return IOUAmount(0, 0)
    return IOUAmount(mantissa, exponent)


def _fraction_to_amount(
    value: Fraction,
    *,
    result_is_xrp: bool,
    round_up: bool,
) -> XRPAmount | IOUAmount:
    if value <= 0:
        return XRPAmount(0) if result_is_xrp else IOUAmount(0, 0)
    if result_is_xrp:
        drops = _ceil_div(value.numerator, value.denominator) if round_up else (
            value.numerator // value.denominator
        )
        return XRPAmount(drops)
    return _fraction_to_iou_amount(value, round_up=round_up)


def _canonicalize_native_positive(value: int, exponent: int, *, round_up: bool) -> XRPAmount:
    if value <= 0 or exponent <= -20:
        return XRPAmount(0)

    if exponent < 0:
        if round_up:
            had_remainder = False
            while exponent < -1:
                new_value = value // 10
                had_remainder |= value != (new_value * 10)
                value = new_value
                exponent += 1
            value += 10 if had_remainder else 9
            value //= 10
            exponent += 1
        else:
            loops = 0
            while exponent < -1:
                value //= 10
                exponent += 1
                loops += 1
            value += 9 if loops >= 2 else 10
            value //= 10
            exponent += 1

    while exponent < 0:
        value //= 10
        exponent += 1

    while exponent > 0:
        if value > 100_000_000_000_000_000:
            raise RuntimeError("native amount out of range")
        value *= 10
        exponent -= 1

    if value > 100_000_000_000_000_000:
        raise RuntimeError("native amount out of range")
    return XRPAmount(value)


def _amount_from_math_value(
    *,
    is_xrp: bool,
    value: int,
    exponent: int,
    round_up: bool,
) -> XRPAmount | IOUAmount:
    if is_xrp:
        result = _canonicalize_native_positive(value, exponent, round_up=round_up)
        if round_up and result.drops == 0:
            return XRPAmount(1)
        return result

    result = IOUAmount(value, exponent)
    if round_up and result.is_zero():
        return IOUAmount(IOU_MANTISSA_MIN, IOU_EXP_MIN)
    return result


def _canonicalize_round_legacy(
    *, integral: bool, value: int, exponent: int, round_up: bool
) -> tuple[int, int]:
    del round_up
    if integral:
        if exponent < 0:
            loops = 0
            while exponent < -1:
                value //= 10
                exponent += 1
                loops += 1
            value += 9 if loops >= 2 else 10
            value //= 10
            exponent += 1
        return value, exponent

    if value > IOU_MANTISSA_MAX:
        while value > 10 * IOU_MANTISSA_MAX:
            value //= 10
            exponent += 1
        value += 9
        value //= 10
        exponent += 1
    return value, exponent


def _canonicalize_round_strict(
    *, integral: bool, value: int, exponent: int, round_up: bool
) -> tuple[int, int]:
    if integral:
        if exponent < 0:
            had_remainder = False
            while exponent < -1:
                new_value = value // 10
                had_remainder |= value != new_value * 10
                value = new_value
                exponent += 1
            value += 10 if (had_remainder and round_up) else 9
            value //= 10
            exponent += 1
        return value, exponent

    if value > IOU_MANTISSA_MAX:
        while value > 10 * IOU_MANTISSA_MAX:
            value //= 10
            exponent += 1
        value += 9
        value //= 10
        exponent += 1
    return value, exponent


def _amount_from_strict_st_components(
    *,
    is_xrp: bool,
    value: int,
    exponent: int,
    round_up: bool,
) -> XRPAmount | IOUAmount:
    if value <= 0:
        return XRPAmount(0) if is_xrp else IOUAmount(0, 0)

    if is_xrp:
        if exponent <= -20:
            return XRPAmount(0)
        while exponent < 0:
            value //= 10
            exponent += 1
        while exponent > 0:
            if value > XRP_MAX_DROPS:
                raise RuntimeError("native amount out of range")
            value *= 10
            exponent -= 1
        if value > XRP_MAX_DROPS:
            raise RuntimeError("native amount out of range")
        result = XRPAmount(value)
        if round_up and result.drops == 0:
            return XRPAmount(1)
        return result

    while value < IOU_MANTISSA_MIN and exponent > IOU_EXP_MIN:
        value *= 10
        exponent -= 1
    while value > IOU_MANTISSA_MAX:
        if exponent >= IOU_EXP_MAX:
            raise RuntimeError("value overflow")
        value //= 10
        exponent += 1
    if exponent < IOU_EXP_MIN or value < IOU_MANTISSA_MIN:
        return IOUAmount(0, 0)
    if exponent > IOU_EXP_MAX:
        raise RuntimeError("value overflow")
    result = IOUAmount(value, exponent)
    if round_up and result.is_zero():
        return IOUAmount(IOU_MANTISSA_MIN, IOU_EXP_MIN)
    return result


def _mul_amount_by_quality_rate_strict(
    amount: XRPAmount | IOUAmount,
    rate: IOUAmount,
    *,
    result_is_xrp: bool,
    round_up: bool,
) -> XRPAmount | IOUAmount:
    if not _is_positive(amount) or not _is_positive(rate):
        return XRPAmount(0) if result_is_xrp else IOUAmount(0, 0)
    value1, exponent1, _ = _working_components(amount)
    value2, exponent2, _ = _working_components(rate)
    increment = TEN_TO_14_MINUS_1 if round_up else 0
    value = (value1 * value2 + increment) // TEN_TO_14
    exponent = exponent1 + exponent2 + 14
    if round_up:
        value, exponent = _canonicalize_round_strict(
            integral=result_is_xrp,
            value=value,
            exponent=exponent,
            round_up=round_up,
        )
    return _amount_from_strict_st_components(
        is_xrp=result_is_xrp,
        value=value,
        exponent=exponent,
        round_up=round_up,
    )


def _divide_amount_by_quality_rate_strict(
    amount: XRPAmount | IOUAmount,
    rate: IOUAmount,
    *,
    result_is_xrp: bool,
    round_up: bool,
) -> XRPAmount | IOUAmount:
    if not _is_positive(amount):
        return XRPAmount(0) if result_is_xrp else IOUAmount(0, 0)
    if not _is_positive(rate):
        raise RuntimeError("rate must be strictly positive")
    num_value, num_exponent, _ = _working_components(amount)
    den_value, den_exponent, _ = _working_components(rate)
    increment = den_value - 1 if round_up else 0
    value = (num_value * TEN_TO_17 + increment) // den_value
    exponent = num_exponent - den_exponent - 17
    if round_up:
        value, exponent = _canonicalize_round_legacy(
            integral=result_is_xrp,
            value=value,
            exponent=exponent,
            round_up=round_up,
        )
    return _amount_from_strict_st_components(
        is_xrp=result_is_xrp,
        value=value,
        exponent=exponent,
        round_up=round_up,
    )


def _mul_amount_by_rate(
    amount: XRPAmount | IOUAmount,
    rate: IOUAmount,
    *,
    result_is_xrp: bool,
    round_up: bool,
) -> XRPAmount | IOUAmount:
    return _fraction_to_amount(
        _amount_to_fraction(amount) * _amount_to_fraction(rate),
        result_is_xrp=result_is_xrp,
        round_up=round_up,
    )


def _divide_amount_by_rate(
    amount: XRPAmount | IOUAmount,
    rate: IOUAmount,
    *,
    result_is_xrp: bool,
    round_up: bool,
) -> XRPAmount | IOUAmount:
    rate_fraction = _amount_to_fraction(rate)
    if rate_fraction <= 0:
        raise RuntimeError("rate must be strictly positive")
    return _fraction_to_amount(
        _amount_to_fraction(amount) / rate_fraction,
        result_is_xrp=result_is_xrp,
        round_up=round_up,
    )


def _owner_funds_limit_out(
    owner_funds: XRPAmount | IOUAmount,
    *,
    out_transfer_rate: IOUAmount | None,
    result_is_xrp: bool,
) -> XRPAmount | IOUAmount:
    if out_transfer_rate is None:
        return _copy_amount(owner_funds)
    if isinstance(owner_funds, IOUAmount) and not result_is_xrp:
        scaled = _scale_iou_by_transfer_rate(
            owner_funds,
            out_transfer_rate,
            divide=True,
            round_up=False,
        )
        if scaled is not None:
            return scaled
    return _divide_amount_by_rate(
        owner_funds,
        out_transfer_rate,
        result_is_xrp=result_is_xrp,
        round_up=False,
    )


def _owner_pays_for_out(
    out_amount: XRPAmount | IOUAmount,
    *,
    out_transfer_rate: IOUAmount | None,
) -> XRPAmount | IOUAmount:
    if out_transfer_rate is None:
        return _copy_amount(out_amount)
    if isinstance(out_amount, IOUAmount):
        scaled = _scale_iou_by_transfer_rate(
            out_amount,
            out_transfer_rate,
            divide=False,
            round_up=False,
        )
        if scaled is not None:
            return scaled
    return _mul_amount_by_rate(
        out_amount,
        out_transfer_rate,
        result_is_xrp=isinstance(out_amount, XRPAmount),
        round_up=False,
    )


def _build_raw_segment(
    *,
    quality: Quality,
    out_amount: XRPAmount | IOUAmount,
    in_amount: XRPAmount | IOUAmount,
) -> Segment:
    return Segment(
        src="CLOB",
        quality=quality,
        out_max=out_amount,
        in_at_out_max=in_amount,
    )


def _collect_owner_funds(
    offers: list[dict[str, Any]],
) -> dict[str, XRPAmount | IOUAmount]:
    out: dict[str, XRPAmount | IOUAmount] = {}
    for offer in offers:
        account = str(offer.get("Account") or "")
        if not account:
            continue
        raw_owner_funds = offer.get("owner_funds")
        if raw_owner_funds is None:
            continue
        gets_currency, _, _ = _parse_amount_field(offer.get("TakerGets"), field_name="TakerGets")
        owner_funds = _parse_owner_funds(raw_owner_funds, gets_currency=gets_currency)
        prior = out.get(account)
        if prior is None:
            out[account] = owner_funds
            continue
        if prior != owner_funds:
            raise RuntimeError(f"ambiguous owner_funds for account {account}")
    return out


def build_clob_segments_from_offers(
    offers: Iterable[dict[str, Any]],
    in_cur: str,
    out_cur: str,
    *,
    iou_in_transfer_rate: Any = None,
    iou_out_transfer_rate: Any = None,
    owner_pays_transfer_fee: bool = True,
    exclude_account: str | None = None,
    preserve_book_visible_offers: bool = False,
) -> list[BookOfferSegment]:
    offer_rows = list(offers)
    in_transfer_rate = None if in_cur == XRP else _normalize_optional_iou_transfer_rate(
        iou_in_transfer_rate,
        field_name="iou_in_transfer_rate",
    )
    out_transfer_rate = None if out_cur == XRP else _normalize_optional_iou_transfer_rate(
        iou_out_transfer_rate,
        field_name="iou_out_transfer_rate",
    )
    owner_funds_out_transfer_rate = out_transfer_rate if owner_pays_transfer_fee else None

    account_remaining = {
        account: _copy_amount(amount) for account, amount in _collect_owner_funds(offer_rows).items()
    }
    exclude_account_norm = str(exclude_account or "")
    parsed_rows: list[dict[str, Any]] = []

    for offer in offer_rows:
        if exclude_account_norm:
            if "Account" not in offer:
                raise RuntimeError("missing offer.Account while exclude_account is set")
            if str(offer.get("Account") or "") == exclude_account_norm:
                continue

        gets_currency, gets_issuer, full_out = _parse_amount_field(
            offer.get("TakerGets"),
            field_name="TakerGets",
        )
        pays_currency, pays_issuer, full_in = _parse_amount_field(
            offer.get("TakerPays"),
            field_name="TakerPays",
        )
        if gets_currency != out_cur or pays_currency != in_cur:
            continue

        offer_id = _parse_book_offer_id(offer)
        quoted_quality = _parse_quality_from_book_directory(
            offer.get("book_directory", offer.get("BookDirectory"))
        )
        quoted_rate = quoted_quality.rate()
        account = str(offer.get("Account") or "") or None
        owner_is_out_issuer = bool(
            account
            and gets_currency != XRP
            and gets_issuer is not None
            and account == gets_issuer
        )

        funded_out_cap: XRPAmount | IOUAmount | None = None
        if "taker_gets_funded" in offer:
            funded_currency, funded_issuer, funded_amount = _parse_amount_field(
                offer.get("taker_gets_funded"),
                field_name="taker_gets_funded",
            )
            if not _same_asset(funded_currency, funded_issuer, gets_currency, gets_issuer):
                raise RuntimeError("taker_gets_funded asset must match TakerGets")
            if funded_amount > full_out:
                raise RuntimeError("taker_gets_funded cannot exceed TakerGets")
            funded_out_cap = funded_amount

        funded_in_cap: XRPAmount | IOUAmount | None = None
        if "taker_pays_funded" in offer:
            funded_currency, funded_issuer, funded_amount = _parse_amount_field(
                offer.get("taker_pays_funded"),
                field_name="taker_pays_funded",
            )
            if not _same_asset(funded_currency, funded_issuer, pays_currency, pays_issuer):
                raise RuntimeError("taker_pays_funded asset must match TakerPays")
            if funded_amount > full_in:
                raise RuntimeError("taker_pays_funded cannot exceed TakerPays")
            funded_in_cap = funded_amount
            if funded_out_cap is None:
                raise RuntimeError("taker_pays_funded without taker_gets_funded is unsupported")

        parsed_rows.append(
            {
                "offer_id": offer_id,
                "quoted_quality": quoted_quality,
                "quoted_rate": quoted_rate,
                "full_out": full_out,
                "full_in": full_in,
                "account": account,
                "owner_is_out_issuer": owner_is_out_issuer,
                "funded_out_cap": funded_out_cap,
                "funded_in_cap": funded_in_cap,
                "non_fill_deleted": bool(offer.get("non_fill_deleted")),
            }
        )

    # Shared owner_funds must be allocated in the same quality-priority order
    # as the book itself, not in snapshot row order.
    parsed_rows.sort(key=lambda row: row["quoted_quality"], reverse=True)

    results: list[BookOfferSegment] = []
    for row in parsed_rows:
        full_out = row["full_out"]
        full_in = row["full_in"]
        quoted_quality = row["quoted_quality"]
        quoted_rate = row["quoted_rate"]
        account = row["account"]
        owner_is_out_issuer = row["owner_is_out_issuer"]
        funded_out_cap = row["funded_out_cap"]
        funded_in_cap = row["funded_in_cap"]
        remaining_out = None if owner_is_out_issuer else account_remaining.get(account or "")

        def append_book_visible_only() -> None:
            if not (
                preserve_book_visible_offers
                and remaining_out is not None
                and _is_positive(full_out)
                and _is_positive(full_in)
            ):
                return
            results.append(
                BookOfferSegment(
                    offer_id=f"{row['offer_id']}#book_visible_residual",
                    quoted_quality=quoted_quality,
                    segment=_build_raw_segment(
                        quality=quoted_quality,
                        out_amount=full_out,
                        in_amount=full_in,
                    ),
                    owner=account,
                    owner_is_out_issuer=owner_is_out_issuer,
                    owner_funds=_zero_like(full_out),
                    in_transfer_rate=in_transfer_rate,
                    out_transfer_rate=out_transfer_rate,
                    non_fill_deleted=bool(row.get("non_fill_deleted")),
                    book_visible_only=True,
                )
            )

        effective_out = full_out
        if funded_out_cap is not None:
            effective_out = _min_amount(effective_out, funded_out_cap)
        if remaining_out is not None:
            owner_funds_limit_out = _owner_funds_limit_out(
                remaining_out,
                out_transfer_rate=owner_funds_out_transfer_rate,
                result_is_xrp=isinstance(full_out, XRPAmount),
            )
            effective_out = _min_amount(effective_out, owner_funds_limit_out)
        if not _is_positive(effective_out):
            append_book_visible_only()
            continue

        effective_in = full_in
        if funded_in_cap is not None:
            effective_in = _min_amount(effective_in, funded_in_cap)
        if effective_out < full_out:
            limited_in = _mul_amount_by_quality_rate_strict(
                effective_out,
                quoted_rate,
                result_is_xrp=isinstance(full_in, XRPAmount),
                round_up=False,
            )
            effective_in = _min_amount(effective_in, limited_in)
        if not _is_positive(effective_in):
            append_book_visible_only()
            continue

        row_owner_funds = _copy_amount(remaining_out) if remaining_out is not None else None
        results.append(
            BookOfferSegment(
                offer_id=row["offer_id"],
                quoted_quality=quoted_quality,
                segment=_build_raw_segment(
                    quality=quoted_quality,
                    out_amount=effective_out,
                    in_amount=effective_in,
                ),
                owner=account,
                owner_is_out_issuer=owner_is_out_issuer,
                owner_funds=row_owner_funds,
                in_transfer_rate=in_transfer_rate,
                out_transfer_rate=out_transfer_rate,
                non_fill_deleted=bool(row.get("non_fill_deleted")),
            )
        )
        if preserve_book_visible_offers and funded_out_cap is None and remaining_out is not None and effective_out < full_out:
            residual_out = full_out - effective_out
            residual_in = full_in - effective_in if effective_in < full_in else _zero_like(full_in)
            if _is_positive(residual_out) and _is_positive(residual_in):
                results.append(
                    BookOfferSegment(
                        offer_id=f"{row['offer_id']}#book_visible_residual",
                        quoted_quality=quoted_quality,
                        segment=_build_raw_segment(
                            quality=quoted_quality,
                            out_amount=residual_out,
                            in_amount=residual_in,
                        ),
                        owner=account,
                        owner_is_out_issuer=owner_is_out_issuer,
                        owner_funds=_zero_like(residual_out),
                        in_transfer_rate=in_transfer_rate,
                        out_transfer_rate=out_transfer_rate,
                        non_fill_deleted=bool(row.get("non_fill_deleted")),
                        book_visible_only=True,
                    )
                )

        if remaining_out is not None:
            owner_pays = _owner_pays_for_out(
                effective_out,
                out_transfer_rate=owner_funds_out_transfer_rate,
            )
            next_remaining = amount_sub_ripple_number(remaining_out, owner_pays)
            account_remaining[account or ""] = (
                next_remaining if _is_positive(next_remaining) else _zero_like(remaining_out)
            )

    return results


__all__ = ["BookOfferSegment", "build_clob_segments_from_offers"]
