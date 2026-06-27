"""Strict single-path AMM model aligned to rippled swap and anchoring semantics."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterator, Literal

from .core import (
    IOU_EXP_MIN,
    IOUAmount,
    IOU_MANTISSA_MAX,
    IOU_MANTISSA_MIN,
    INT64_MAX,
    Quality,
    Segment,
    XRPAmount,
    parse_iou_decimal,
    parse_xrp_decimal,
)

Amount = XRPAmount | IOUAmount

TRADING_FEE_SCALE = 100_000
TRADING_FEE_MAX = 1_000
PRODUCT_RELATIVE_DISTANCE = Fraction(1, 10_000_000)
QUALITY_RELATIVE_DISTANCE = Fraction(1, 10_000_000)
REDUCED_OFFER_NUMERATOR = 9_999
REDUCED_OFFER_DENOMINATOR = 10_000
RippleNumberMantissaScale = Literal["small", "large"]
RIPPLE_NUMBER_SMALL_MIN_MANTISSA = 10**15
RIPPLE_NUMBER_SMALL_MAX_MANTISSA = (10**16) - 1
RIPPLE_NUMBER_LARGE_MIN_MANTISSA = 10**18
RIPPLE_NUMBER_LARGE_MAX_MANTISSA = (10**19) - 1
RIPPLE_NUMBER_MIN_EXPONENT = -32768
RIPPLE_NUMBER_MAX_EXPONENT = 32768
ROUND_TO_NEAREST = "to_nearest"
ROUND_TOWARDS_ZERO = "towards_zero"
ROUND_DOWNWARD = "downward"
ROUND_UPWARD = "upward"
RIPPLE_NUMBER_MAX_REP = (2**63) - 1
RIPPLE_NUMBER_ZERO_EXPONENT = -(2**31)
RIPPLE_NUMBER_DIV_CORRECTION_FACTOR = 1_000
RIPPLE_NUMBER_SMALL_DIV_SCALE = 10**17
RIPPLE_NUMBER_LARGE_DIV_SCALE = 10**19
_ACTIVE_RIPPLE_NUMBER_MANTISSA_SCALE: RippleNumberMantissaScale = "small"


def _validate_ripple_number_mantissa_scale(scale: str) -> RippleNumberMantissaScale:
    if scale not in {"small", "large"}:
        raise RuntimeError(f"unsupported ripple number mantissa scale: {scale}")
    return scale


def _ripple_number_min_mantissa() -> int:
    if _ACTIVE_RIPPLE_NUMBER_MANTISSA_SCALE == "small":
        return RIPPLE_NUMBER_SMALL_MIN_MANTISSA
    return RIPPLE_NUMBER_LARGE_MIN_MANTISSA


def _ripple_number_max_mantissa() -> int:
    if _ACTIVE_RIPPLE_NUMBER_MANTISSA_SCALE == "small":
        return RIPPLE_NUMBER_SMALL_MAX_MANTISSA
    return RIPPLE_NUMBER_LARGE_MAX_MANTISSA


def _ripple_number_mantissa_log() -> int:
    if _ACTIVE_RIPPLE_NUMBER_MANTISSA_SCALE == "small":
        return 15
    return 18


def _ripple_number_div_scale() -> int:
    if _ACTIVE_RIPPLE_NUMBER_MANTISSA_SCALE == "small":
        return RIPPLE_NUMBER_SMALL_DIV_SCALE
    return RIPPLE_NUMBER_LARGE_DIV_SCALE


def _ripple_number_div_exponent_shift() -> int:
    if _ACTIVE_RIPPLE_NUMBER_MANTISSA_SCALE == "small":
        return 17
    return 19


def _ripple_number_use_div_correction() -> bool:
    return _ACTIVE_RIPPLE_NUMBER_MANTISSA_SCALE == "large"


@contextmanager
def _ripple_number_mantissa_scale_guard(scale: RippleNumberMantissaScale) -> Iterator[None]:
    global _ACTIVE_RIPPLE_NUMBER_MANTISSA_SCALE
    saved = _ACTIVE_RIPPLE_NUMBER_MANTISSA_SCALE
    _ACTIVE_RIPPLE_NUMBER_MANTISSA_SCALE = scale
    try:
        yield
    finally:
        _ACTIVE_RIPPLE_NUMBER_MANTISSA_SCALE = saved


@dataclass(frozen=True)
class AMMOverAsk(Exception):
    max_out: Amount
    dx_for_max: Amount

    def __str__(self) -> str:
        return "requested AMM output exceeds non-drain capacity"


class _RippleGuard:
    __slots__ = ("_digits", "_xbit", "_sbit")

    def __init__(self) -> None:
        self._digits = 0
        self._xbit = False
        self._sbit = False

    def set_negative(self) -> None:
        self._sbit = True

    def push(self, digit: int) -> None:
        self._xbit = self._xbit or ((self._digits & 0xF) != 0)
        self._digits >>= 4
        self._digits |= (digit & 0xF) << 60

    def pop(self) -> int:
        digit = (self._digits & 0xF000000000000000) >> 60
        self._digits = (self._digits << 4) & 0xFFFFFFFFFFFFFFFF
        return digit

    def round(self, rounding: str) -> int:
        if rounding == ROUND_TOWARDS_ZERO:
            return -1
        if rounding == ROUND_DOWNWARD:
            if self._sbit and (self._digits > 0 or self._xbit):
                return 1
            return -1
        if rounding == ROUND_UPWARD:
            if self._sbit:
                return -1
            if self._digits > 0 or self._xbit:
                return 1
            return -1
        if rounding != ROUND_TO_NEAREST:
            raise RuntimeError(f"unsupported rounding mode: {rounding}")
        half = 0x5000000000000000
        if self._digits > half:
            return 1
        if self._digits < half:
            return -1
        if self._xbit:
            return 1
        return 0

    def _bring_into_range(self, negative: bool, mantissa: int, exponent: int, *, min_mantissa: int) -> tuple[bool, int, int]:
        if mantissa < min_mantissa:
            mantissa *= 10
            exponent -= 1
        if exponent < RIPPLE_NUMBER_MIN_EXPONENT:
            return False, 0, RIPPLE_NUMBER_ZERO_EXPONENT
        return negative, mantissa, exponent

    def do_round_up(
        self,
        negative: bool,
        mantissa: int,
        exponent: int,
        *,
        min_mantissa: int,
        max_mantissa: int,
        rounding: str,
    ) -> tuple[bool, int, int]:
        rounded = self.round(rounding)
        if rounded == 1 or (rounded == 0 and (mantissa & 1) == 1):
            mantissa += 1
            if mantissa > max_mantissa or mantissa > RIPPLE_NUMBER_MAX_REP:
                mantissa //= 10
                exponent += 1
        negative, mantissa, exponent = self._bring_into_range(
            negative,
            mantissa,
            exponent,
            min_mantissa=min_mantissa,
        )
        if exponent > RIPPLE_NUMBER_MAX_EXPONENT:
            raise RuntimeError("rippled number exponent overflow")
        return negative, mantissa, exponent

    def do_round_down(
        self,
        negative: bool,
        mantissa: int,
        exponent: int,
        *,
        min_mantissa: int,
        rounding: str,
    ) -> tuple[bool, int, int]:
        rounded = self.round(rounding)
        if rounded == 1 or (rounded == 0 and (mantissa & 1) == 1):
            mantissa -= 1
            if mantissa < min_mantissa:
                mantissa *= 10
                exponent -= 1
        return self._bring_into_range(negative, mantissa, exponent, min_mantissa=min_mantissa)

    def do_round_rep(self, drops: int, *, rounding: str) -> int:
        rounded = self.round(rounding)
        if rounded == 1 or (rounded == 0 and (drops & 1) == 1):
            if drops >= RIPPLE_NUMBER_MAX_REP:
                raise RuntimeError("rippled number rep overflow")
            drops += 1
        if self._sbit:
            drops = -drops
        return drops


def _normalize_ripple_number(
    negative: bool,
    mantissa: int,
    exponent: int,
    *,
    min_mantissa: int,
    max_mantissa: int,
    rounding: str,
) -> tuple[bool, int, int]:
    if mantissa == 0:
        return False, 0, RIPPLE_NUMBER_ZERO_EXPONENT

    while mantissa < min_mantissa and exponent > RIPPLE_NUMBER_MIN_EXPONENT:
        mantissa *= 10
        exponent -= 1

    guard = _RippleGuard()
    if negative:
        guard.set_negative()

    while mantissa > max_mantissa:
        if exponent >= RIPPLE_NUMBER_MAX_EXPONENT:
            raise RuntimeError("rippled number exponent overflow")
        mantissa, digit = divmod(mantissa, 10)
        guard.push(digit)
        exponent += 1

    if exponent < RIPPLE_NUMBER_MIN_EXPONENT or mantissa < min_mantissa:
        return False, 0, RIPPLE_NUMBER_ZERO_EXPONENT

    if mantissa > RIPPLE_NUMBER_MAX_REP:
        if exponent >= RIPPLE_NUMBER_MAX_EXPONENT:
            raise RuntimeError("rippled number exponent overflow")
        mantissa, digit = divmod(mantissa, 10)
        guard.push(digit)
        exponent += 1

    return guard.do_round_up(
        negative,
        mantissa,
        exponent,
        min_mantissa=min_mantissa,
        max_mantissa=max_mantissa,
        rounding=rounding,
    )


@dataclass(frozen=True)
class _RippleNumber:
    negative: bool
    mantissa: int
    exponent: int

    @classmethod
    def zero(cls) -> "_RippleNumber":
        return cls(False, 0, RIPPLE_NUMBER_ZERO_EXPONENT)

    @classmethod
    def _from_components(
        cls,
        negative: bool,
        mantissa: int,
        exponent: int,
        *,
        rounding: str = ROUND_TO_NEAREST,
    ) -> "_RippleNumber":
        negative, mantissa, exponent = _normalize_ripple_number(
            negative,
            mantissa,
            exponent,
            min_mantissa=_ripple_number_min_mantissa(),
            max_mantissa=_ripple_number_max_mantissa(),
            rounding=rounding,
        )
        return cls(negative, mantissa, exponent)

    @classmethod
    def from_amount(cls, amount: Amount) -> "_RippleNumber":
        if isinstance(amount, XRPAmount):
            return cls.from_int(amount.drops)
        return cls._from_components(
            amount.mantissa < 0,
            abs(amount.mantissa),
            amount.exponent,
        )

    @classmethod
    def from_int(cls, value: int) -> "_RippleNumber":
        return cls._from_components(value < 0, abs(value), 0)

    def is_zero(self) -> bool:
        return self.mantissa == 0

    def signum(self) -> int:
        if self.is_zero():
            return 0
        return -1 if self.negative else 1

    def external_mantissa(self) -> int:
        mantissa = self.mantissa
        if mantissa > RIPPLE_NUMBER_MAX_REP:
            mantissa //= 10
        return -mantissa if self.negative else mantissa

    def external_exponent(self) -> int:
        exponent = self.exponent
        if self.mantissa > RIPPLE_NUMBER_MAX_REP:
            exponent += 1
        return exponent

    def normalize_to_range(
        self,
        *,
        min_mantissa: int,
        max_mantissa: int,
        rounding: str,
        zero_exponent: int,
    ) -> tuple[int, int]:
        if self.is_zero():
            return 0, zero_exponent
        negative, mantissa, exponent = _normalize_ripple_number(
            self.negative,
            self.mantissa,
            self.exponent,
            min_mantissa=min_mantissa,
            max_mantissa=max_mantissa,
            rounding=rounding,
        )
        if mantissa == 0:
            return 0, zero_exponent
        signed = -mantissa if negative else mantissa
        return signed, exponent

    def to_rep(self, *, rounding: str) -> int:
        drops = self.external_mantissa()
        exponent = self.external_exponent()
        guard = _RippleGuard()
        if drops == 0:
            return 0
        if drops < 0:
            guard.set_negative()
            drops = -drops
        while exponent < 0:
            drops, digit = divmod(drops, 10)
            guard.push(digit)
            exponent += 1
        while exponent > 0:
            if drops > RIPPLE_NUMBER_MAX_REP // 10:
                raise RuntimeError("rippled number rep overflow")
            drops *= 10
            exponent -= 1
        return guard.do_round_rep(drops, rounding=rounding)

    def __neg__(self) -> "_RippleNumber":
        if self.is_zero():
            return self
        return _RippleNumber(not self.negative, self.mantissa, self.exponent)

    def to_amount(self, *, is_xrp: bool, rounding: str) -> Amount:
        if is_xrp:
            return XRPAmount(self.to_rep(rounding=rounding))
        mantissa, exponent = self.normalize_to_range(
            min_mantissa=IOU_MANTISSA_MIN,
            max_mantissa=IOU_MANTISSA_MAX,
            rounding=rounding,
            zero_exponent=-100,
        )
        return IOUAmount(mantissa, exponent)


def _number_compare(left: _RippleNumber, right: _RippleNumber) -> int:
    if left.negative != right.negative:
        return -1 if left.negative else 1
    if left.is_zero() and right.is_zero():
        return 0
    if left.is_zero():
        return -1 if right.signum() > 0 else 1
    if right.is_zero():
        return 1 if left.signum() > 0 else -1
    if left.exponent != right.exponent:
        if left.exponent > right.exponent:
            return 1 if not left.negative else -1
        return -1 if not left.negative else 1
    if left.mantissa == right.mantissa:
        return 0
    if left.mantissa > right.mantissa:
        return 1 if not left.negative else -1
    return -1 if not left.negative else 1


def _number_lt(left: _RippleNumber, right: _RippleNumber) -> bool:
    return _number_compare(left, right) < 0


def _number_le(left: _RippleNumber, right: _RippleNumber) -> bool:
    return _number_compare(left, right) <= 0


def _number_gt(left: _RippleNumber, right: _RippleNumber) -> bool:
    return _number_compare(left, right) > 0


def _number_one() -> _RippleNumber:
    return _RippleNumber.from_int(1)


def _number_shift_exponent(value: _RippleNumber, exponent_delta: int) -> _RippleNumber:
    new_exponent = value.exponent + exponent_delta
    if new_exponent >= RIPPLE_NUMBER_MAX_EXPONENT:
        raise RuntimeError("rippled number shift exponent overflow")
    if new_exponent < RIPPLE_NUMBER_MIN_EXPONENT:
        return _RippleNumber.zero()
    return _RippleNumber(value.negative, value.mantissa, new_exponent)


def _number_root2(value: _RippleNumber) -> _RippleNumber:
    zero = _RippleNumber.zero()
    one = _number_one()

    if value == one:
        return value
    if _number_lt(value, zero):
        raise RuntimeError("rippled number root2 nan")
    if value == zero:
        return value

    exponent = value.exponent + _ripple_number_mantissa_log() + 1
    if exponent % 2 != 0:
        exponent += 1
    scaled = _number_shift_exponent(value, -exponent)

    d = _RippleNumber.from_int(105)
    a0 = _RippleNumber.from_int(18)
    a1 = _RippleNumber.from_int(144)
    a2 = _RippleNumber.from_int(-60)
    guess = _number_div(
        _number_add(
            _number_mul(
                _number_add(_number_mul(a2, scaled, rounding=ROUND_TO_NEAREST), a1, rounding=ROUND_TO_NEAREST),
                scaled,
                rounding=ROUND_TO_NEAREST,
            ),
            a0,
            rounding=ROUND_TO_NEAREST,
        ),
        d,
        rounding=ROUND_TO_NEAREST,
    )

    prev_1 = zero
    prev_2 = zero
    while True:
        prev_2 = prev_1
        prev_1 = guess
        guess = _number_div(
            _number_add(guess, _number_div(scaled, guess, rounding=ROUND_TO_NEAREST), rounding=ROUND_TO_NEAREST),
            _RippleNumber.from_int(2),
            rounding=ROUND_TO_NEAREST,
        )
        if guess == prev_1 or guess == prev_2:
            break

    return _number_shift_exponent(guess, exponent // 2)


def _number_add(left: _RippleNumber, right: _RippleNumber, *, rounding: str) -> _RippleNumber:
    if right.is_zero():
        return left
    if left.is_zero():
        return right
    if left == -right:
        return _RippleNumber.zero()

    left_negative = left.negative
    left_mantissa = left.mantissa
    left_exponent = left.exponent

    right_negative = right.negative
    right_mantissa = right.mantissa
    right_exponent = right.exponent

    guard = _RippleGuard()
    if left_exponent < right_exponent:
        if left_negative:
            guard.set_negative()
        while left_exponent < right_exponent:
            left_mantissa, digit = divmod(left_mantissa, 10)
            guard.push(digit)
            left_exponent += 1
    elif left_exponent > right_exponent:
        if right_negative:
            guard.set_negative()
        while left_exponent > right_exponent:
            right_mantissa, digit = divmod(right_mantissa, 10)
            guard.push(digit)
            right_exponent += 1

    if left_negative == right_negative:
        left_mantissa += right_mantissa
        max_mantissa = _ripple_number_max_mantissa()
        if left_mantissa > max_mantissa or left_mantissa > RIPPLE_NUMBER_MAX_REP:
            left_mantissa, digit = divmod(left_mantissa, 10)
            guard.push(digit)
            left_exponent += 1
        left_negative, left_mantissa, left_exponent = guard.do_round_up(
            left_negative,
            left_mantissa,
            left_exponent,
            min_mantissa=_ripple_number_min_mantissa(),
            max_mantissa=max_mantissa,
            rounding=rounding,
        )
    else:
        if left_mantissa > right_mantissa:
            left_mantissa -= right_mantissa
        else:
            left_mantissa = right_mantissa - left_mantissa
            left_exponent = right_exponent
            left_negative = right_negative
        min_mantissa = _ripple_number_min_mantissa()
        while left_mantissa < min_mantissa and left_mantissa * 10 <= RIPPLE_NUMBER_MAX_REP:
            left_mantissa *= 10
            left_mantissa -= guard.pop()
            left_exponent -= 1
        left_negative, left_mantissa, left_exponent = guard.do_round_down(
            left_negative,
            left_mantissa,
            left_exponent,
            min_mantissa=min_mantissa,
            rounding=rounding,
        )
    return _RippleNumber._from_components(
        left_negative,
        left_mantissa,
        left_exponent,
        rounding=rounding,
    )


def _number_sub(left: _RippleNumber, right: _RippleNumber, *, rounding: str) -> _RippleNumber:
    return _number_add(left, -right, rounding=rounding)


def _number_mul(left: _RippleNumber, right: _RippleNumber, *, rounding: str) -> _RippleNumber:
    if left.is_zero() or right.is_zero():
        return _RippleNumber.zero()

    negative = left.negative != right.negative
    mantissa = left.mantissa * right.mantissa
    exponent = left.exponent + right.exponent
    guard = _RippleGuard()
    if negative:
        guard.set_negative()

    max_mantissa = _ripple_number_max_mantissa()
    while mantissa > max_mantissa or mantissa > RIPPLE_NUMBER_MAX_REP:
        mantissa, digit = divmod(mantissa, 10)
        guard.push(digit)
        exponent += 1

    negative, mantissa, exponent = guard.do_round_up(
        negative,
        mantissa,
        exponent,
        min_mantissa=_ripple_number_min_mantissa(),
        max_mantissa=max_mantissa,
        rounding=rounding,
    )
    return _RippleNumber._from_components(negative, mantissa, exponent, rounding=rounding)


def _number_div(left: _RippleNumber, right: _RippleNumber, *, rounding: str) -> _RippleNumber:
    if right.is_zero():
        raise RuntimeError("division by zero")
    if left.is_zero():
        return _RippleNumber.zero()

    div_scale = _ripple_number_div_scale()
    numerator = left.mantissa * div_scale
    mantissa = numerator // right.mantissa
    exponent = left.exponent - right.exponent - _ripple_number_div_exponent_shift()
    negative = left.negative != right.negative

    remainder = numerator % right.mantissa
    if _ripple_number_use_div_correction() and remainder != 0:
        mantissa *= RIPPLE_NUMBER_DIV_CORRECTION_FACTOR
        mantissa += (remainder * RIPPLE_NUMBER_DIV_CORRECTION_FACTOR) // right.mantissa
        exponent -= 3

    return _RippleNumber._from_components(negative, mantissa, exponent, rounding=rounding)


def _amount_add(left: Amount, right: Amount, *, rounding: str) -> Amount:
    if type(left) is not type(right):
        raise RuntimeError("amount addition requires matching asset domains")
    if isinstance(left, XRPAmount):
        return left + right
    summed = _number_add(
        _RippleNumber.from_amount(left),
        _RippleNumber.from_amount(right),
        rounding=rounding,
    )
    return summed.to_amount(is_xrp=False, rounding=rounding)


def amount_add_ripple_number(left: Amount, right: Amount) -> Amount:
    """Add amounts using rippled's current Number-backed IOUAmount semantics."""

    return _amount_add(left, right, rounding=ROUND_TO_NEAREST)


def _amount_sub(left: Amount, right: Amount, *, rounding: str) -> Amount:
    if type(left) is not type(right):
        raise RuntimeError("amount subtraction requires matching asset domains")
    if isinstance(left, XRPAmount):
        return left - right
    diff = _number_sub(
        _RippleNumber.from_amount(left),
        _RippleNumber.from_amount(right),
        rounding=rounding,
    )
    return diff.to_amount(is_xrp=False, rounding=rounding)


def amount_sub_ripple_number(left: Amount, right: Amount) -> Amount:
    """Subtract amounts using rippled's current Number-backed IOUAmount semantics."""

    return _amount_sub(left, right, rounding=ROUND_TO_NEAREST)


def _parse_positive_reserve(value: Any, *, is_xrp: bool, field_name: str) -> Amount:
    if isinstance(value, XRPAmount):
        if not is_xrp:
            raise RuntimeError(f"invalid {field_name}: expected IOU reserve")
        if value.drops <= 0:
            raise RuntimeError(f"invalid {field_name}: reserve must be strictly positive")
        return value
    if isinstance(value, IOUAmount):
        if is_xrp:
            raise RuntimeError(f"invalid {field_name}: expected XRP reserve")
        if value.mantissa <= 0:
            raise RuntimeError(f"invalid {field_name}: reserve must be strictly positive")
        return value

    try:
        parsed = (
            parse_xrp_decimal(value, field_name=field_name)
            if is_xrp
            else parse_iou_decimal(value, field_name=field_name)
        )
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    if parsed.signum() <= 0:
        raise RuntimeError(f"invalid {field_name}: reserve must be strictly positive")
    return parsed


def _parse_trading_fee_units(value: Any) -> int:
    if type(value) is int:
        fee_units = value
    else:
        text = str(value).strip()
        if not text:
            raise RuntimeError("invalid fee: empty decimal string")
        if text[0] == "+":
            text = text[1:]
        elif text[0] == "-":
            raise RuntimeError("invalid fee: expected decimal fraction in [0, 1)")
        if not text:
            raise RuntimeError("invalid fee: empty decimal string")
        if text.count(".") > 1:
            raise RuntimeError("invalid fee: too many decimal points")
        if "." in text:
            whole_part, frac_part = text.split(".", 1)
        else:
            whole_part, frac_part = text, ""
        if not whole_part:
            whole_part = "0"
        if not whole_part.isdigit() or (frac_part and not frac_part.isdigit()):
            raise RuntimeError("invalid fee: malformed decimal fraction")
        if int(whole_part) != 0:
            raise RuntimeError("invalid fee: expected decimal fraction in [0, 1)")
        if len(frac_part) > 5:
            if any(ch != "0" for ch in frac_part[5:]):
                raise RuntimeError("invalid fee: must be exact in 1/100000 increments")
            frac_part = frac_part[:5]
        fee_units = int(frac_part.ljust(5, "0")) if frac_part else 0
    if fee_units < 0 or fee_units > TRADING_FEE_MAX:
        raise RuntimeError("invalid fee: XRPL trading fee must be in [0, 1000]")
    return fee_units


def _amount_to_fraction(amount: Amount) -> Fraction:
    if isinstance(amount, XRPAmount):
        return Fraction(amount.drops, 1)
    if amount.exponent >= 0:
        return Fraction(amount.mantissa * (10**amount.exponent), 1)
    return Fraction(amount.mantissa, 10 ** (-amount.exponent))


def _from_units(units: int, *, is_xrp: bool) -> Amount:
    if units <= 0:
        return XRPAmount(0) if is_xrp else IOUAmount(0, 0)
    if is_xrp:
        return XRPAmount(units)
    mantissa = units
    exponent = -15
    while mantissa > INT64_MAX:
        mantissa //= 10
        exponent += 1
    return IOUAmount(mantissa, exponent)


def _previous_positive_amount(amount: Amount) -> Amount:
    if isinstance(amount, XRPAmount):
        return XRPAmount(max(0, amount.drops - 1))
    if amount.mantissa <= 0:
        return IOUAmount(0, 0)
    if amount.mantissa > IOU_MANTISSA_MIN:
        return IOUAmount(amount.mantissa - 1, amount.exponent)
    if amount.exponent <= IOU_EXP_MIN:
        return IOUAmount(0, 0)
    return IOUAmount(IOU_MANTISSA_MAX, amount.exponent - 1)


def _amount_min(left: Amount, right: Amount) -> Amount:
    if type(left) is not type(right):
        raise RuntimeError("amount min requires matching asset domains")
    return left if left <= right else right


def _amount_to_units_floor(amount: Amount) -> int:
    if isinstance(amount, XRPAmount):
        return max(0, amount.drops)
    if amount.mantissa <= 0:
        return 0
    shift = amount.exponent + 15
    if shift >= 0:
        return amount.mantissa * (10**shift)
    return amount.mantissa // (10 ** (-shift))


@dataclass(frozen=True)
class _SyntheticProposal:
    start_with_taker_gets: bool
    start_amount: Amount


def _quality_rate_number(quality: Quality) -> _RippleNumber:
    return _RippleNumber.from_amount(quality.rate())


def _fee_multiplier_number(keep_fee_units: int) -> _RippleNumber:
    return _number_div(
        _RippleNumber.from_int(keep_fee_units),
        _RippleNumber.from_int(TRADING_FEE_SCALE),
        rounding=ROUND_TO_NEAREST,
    )


def _solve_quadratic_smallest(
    a: _RippleNumber,
    b: _RippleNumber,
    c: _RippleNumber,
) -> _RippleNumber | None:
    discriminant = _number_sub(
        _number_mul(b, b, rounding=ROUND_TO_NEAREST),
        _number_mul(
            _number_mul(_RippleNumber.from_int(4), a, rounding=ROUND_TO_NEAREST),
            c,
            rounding=ROUND_TO_NEAREST,
        ),
        rounding=ROUND_TO_NEAREST,
    )
    if _number_lt(discriminant, _RippleNumber.zero()):
        return None
    root = _number_root2(discriminant)
    two_c = _number_mul(_RippleNumber.from_int(2), c, rounding=ROUND_TO_NEAREST)
    if _number_gt(b, _RippleNumber.zero()):
        denominator = _number_sub(-b, root, rounding=ROUND_TO_NEAREST)
    else:
        denominator = _number_add(-b, root, rounding=ROUND_TO_NEAREST)
    if denominator.is_zero():
        return None
    return _number_div(two_c, denominator, rounding=ROUND_TO_NEAREST)


def _start_amount_from_number(proposed: _RippleNumber, *, is_xrp: bool) -> Amount | None:
    if _number_le(proposed, _RippleNumber.zero()):
        return None
    amount = proposed.to_amount(is_xrp=is_xrp, rounding=ROUND_DOWNWARD)
    return amount if _amount_to_units_floor(amount) > 0 else None


def _reduce_offer_amount(amount: Amount) -> Amount:
    reduced_pct = _number_shift_exponent(_RippleNumber.from_int(REDUCED_OFFER_NUMERATOR), -4)
    reduced = _number_mul(
        _RippleNumber.from_amount(amount),
        reduced_pct,
        rounding=ROUND_TOWARDS_ZERO,
    )
    return reduced.to_amount(is_xrp=isinstance(amount, XRPAmount), rounding=ROUND_DOWNWARD)


def _solve_quality_crossing_units(
    *,
    x_amount: Amount,
    y_amount: Amount,
    x_is_xrp: bool,
    y_is_xrp: bool,
    keep_fee_units: int,
    q_threshold: Quality,
) -> _SyntheticProposal | None:
    rate = _quality_rate_number(q_threshold)
    if rate.is_zero():
        return None
    fee_multiplier = _fee_multiplier_number(keep_fee_units)
    if _number_le(fee_multiplier, _RippleNumber.zero()):
        return None

    pool_in = _RippleNumber.from_amount(x_amount)
    pool_out = _RippleNumber.from_amount(y_amount)

    start_with_taker_gets = y_is_xrp
    if start_with_taker_gets:
        a = _number_one()
        b = _number_sub(
            _number_div(
                _number_mul(
                    pool_in,
                    _number_sub(
                        _number_one(),
                        _number_div(_number_one(), fee_multiplier, rounding=ROUND_TO_NEAREST),
                        rounding=ROUND_TO_NEAREST,
                    ),
                    rounding=ROUND_TO_NEAREST,
                ),
                rate,
                rounding=ROUND_TO_NEAREST,
            ),
            _number_mul(_RippleNumber.from_int(2), pool_out, rounding=ROUND_TO_NEAREST),
            rounding=ROUND_TO_NEAREST,
        )
        c = _number_sub(
            _number_mul(pool_out, pool_out, rounding=ROUND_TO_NEAREST),
            _number_div(
                _number_mul(pool_in, pool_out, rounding=ROUND_TO_NEAREST),
                rate,
                rounding=ROUND_TO_NEAREST,
            ),
            rounding=ROUND_TO_NEAREST,
        )
        proposed = _solve_quadratic_smallest(a, b, c)
        if proposed is None or _number_le(proposed, _RippleNumber.zero()):
            return None
        constraint = _number_sub(
            pool_out,
            _number_div(
                pool_in,
                _number_mul(rate, fee_multiplier, rounding=ROUND_TO_NEAREST),
                rounding=ROUND_TO_NEAREST,
            ),
            rounding=ROUND_TO_NEAREST,
        )
        if _number_le(constraint, _RippleNumber.zero()):
            return None
        if _number_lt(constraint, proposed):
            proposed = constraint
        start_amount = _start_amount_from_number(proposed, is_xrp=y_is_xrp)
    else:
        a = fee_multiplier
        b = _number_mul(
            pool_in,
            _number_add(_number_one(), fee_multiplier, rounding=ROUND_TO_NEAREST),
            rounding=ROUND_TO_NEAREST,
        )
        c = _number_sub(
            _number_mul(pool_in, pool_in, rounding=ROUND_TO_NEAREST),
            _number_mul(
                _number_mul(pool_in, pool_out, rounding=ROUND_TO_NEAREST),
                rate,
                rounding=ROUND_TO_NEAREST,
            ),
            rounding=ROUND_TO_NEAREST,
        )
        proposed = _solve_quadratic_smallest(a, b, c)
        if proposed is None or _number_le(proposed, _RippleNumber.zero()):
            return None
        constraint = _number_sub(
            _number_mul(pool_out, rate, rounding=ROUND_TO_NEAREST),
            _number_div(pool_in, fee_multiplier, rounding=ROUND_TO_NEAREST),
            rounding=ROUND_TO_NEAREST,
        )
        if _number_le(constraint, _RippleNumber.zero()):
            return None
        if _number_lt(constraint, proposed):
            proposed = constraint
        start_amount = _start_amount_from_number(proposed, is_xrp=x_is_xrp)

    if start_amount is None:
        return None
    return _SyntheticProposal(start_with_taker_gets=start_with_taker_gets, start_amount=start_amount)


def _quality_rate_fraction(quality: Quality) -> Fraction:
    return quality.rate().as_fraction()


def _within_relative_distance(left: Quality, right: Quality, *, dist: Fraction) -> bool:
    left_rate = _quality_rate_fraction(left)
    right_rate = _quality_rate_fraction(right)
    if left_rate == right_rate:
        return True
    low = min(left_rate, right_rate)
    high = max(left_rate, right_rate)
    if high <= 0:
        return False
    return ((high - low) / high) < dist


class AMM:
    """Strict single-path AMM state and quoting surface."""

    def __init__(
        self,
        x_reserve: Any,
        y_reserve: Any,
        fee: Any,
        *,
        x_is_xrp: bool,
        y_is_xrp: bool,
        number_mantissa_scale: RippleNumberMantissaScale = "small",
    ) -> None:
        self.x_is_xrp = bool(x_is_xrp)
        self.y_is_xrp = bool(y_is_xrp)
        self.number_mantissa_scale = _validate_ripple_number_mantissa_scale(number_mantissa_scale)
        self._x_amount = _parse_positive_reserve(x_reserve, is_xrp=self.x_is_xrp, field_name="x_reserve")
        self._y_amount = _parse_positive_reserve(y_reserve, is_xrp=self.y_is_xrp, field_name="y_reserve")
        self.trading_fee = _parse_trading_fee_units(fee)
        self._keep_fee_units = TRADING_FEE_SCALE - self.trading_fee
        self._sync_public_reserves()

    def _sync_public_reserves(self) -> None:
        self._x_units = _amount_to_units_floor(self._x_amount)
        self._y_units = _amount_to_units_floor(self._y_amount)

    def clone(self) -> "AMM":
        return AMM(
            self._x_amount,
            self._y_amount,
            self.trading_fee,
            x_is_xrp=self.x_is_xrp,
            y_is_xrp=self.y_is_xrp,
            number_mantissa_scale=self.number_mantissa_scale,
        )

    def max_out_amount(self) -> Amount:
        return _previous_positive_amount(self._y_amount)

    def max_offer_out_amount(self) -> Amount | None:
        with _ripple_number_mantissa_scale_guard(self.number_mantissa_scale):
            pool_out = _RippleNumber.from_amount(self._y_amount)
            capped_out = _number_mul(
                pool_out,
                _RippleNumber._from_components(False, 99, -2),
                rounding=ROUND_TO_NEAREST,
            ).to_amount(is_xrp=self.y_is_xrp, rounding=ROUND_DOWNWARD)
        if capped_out.signum() <= 0 or capped_out >= self._y_amount:
            return None
        return capped_out

    def max_offer_segment(self, *, max_out_cap: Amount | None = None) -> Segment | None:
        out_amount = self.max_offer_out_amount()
        if out_amount is None:
            return None
        if max_out_cap is not None:
            out_amount = _amount_min(out_amount, max_out_cap)
        if out_amount.signum() <= 0:
            return None
        in_amount = self.swap_in_given_out_st(out_amount)
        if in_amount.signum() <= 0:
            return None
        return Segment(
            src="AMM",
            quality=self.spq_quality_int(),
            out_max=out_amount,
            in_at_out_max=in_amount,
        )

    def spq_quality_int(self) -> Quality:
        if self._x_amount.signum() <= 0 or self._y_amount.signum() <= 0:
            return Quality(0)
        return Quality.from_amounts(self._y_amount, self._x_amount)

    def swap_out_given_in_st(self, dx: Amount) -> Amount:
        if dx.signum() <= 0 or self._x_amount.signum() <= 0 or self._y_amount.signum() <= 0:
            return _from_units(0, is_xrp=self.y_is_xrp)
        with _ripple_number_mantissa_scale_guard(self.number_mantissa_scale):
            pool_in = _RippleNumber.from_amount(self._x_amount)
            pool_out = _RippleNumber.from_amount(self._y_amount)
            asset_in = _RippleNumber.from_amount(dx)
            numerator = _number_mul(pool_in, pool_out, rounding=ROUND_UPWARD)
            fee = _number_div(
                _RippleNumber.from_int(self.trading_fee),
                _RippleNumber.from_int(TRADING_FEE_SCALE),
                rounding=ROUND_UPWARD,
            )
            fee_mult = _number_sub(_RippleNumber.from_int(1), fee, rounding=ROUND_DOWNWARD)
            asset_in_after_fee = _number_mul(asset_in, fee_mult, rounding=ROUND_DOWNWARD)
            denominator = _number_add(pool_in, asset_in_after_fee, rounding=ROUND_DOWNWARD)
            if denominator.signum() <= 0:
                return _from_units(0, is_xrp=self.y_is_xrp)
            ratio = _number_div(numerator, denominator, rounding=ROUND_UPWARD)
            swap_out = _number_sub(pool_out, ratio, rounding=ROUND_DOWNWARD)
            if swap_out.signum() < 0:
                return _from_units(0, is_xrp=self.y_is_xrp)
            return swap_out.to_amount(is_xrp=self.y_is_xrp, rounding=ROUND_DOWNWARD)

    def swap_in_given_out_st(self, dy: Amount) -> Amount:
        if dy.signum() <= 0 or self._x_amount.signum() <= 0 or self._y_amount.signum() <= 0:
            return _from_units(0, is_xrp=self.x_is_xrp)
        if dy >= self._y_amount:
            return _from_units(0, is_xrp=self.x_is_xrp)
        with _ripple_number_mantissa_scale_guard(self.number_mantissa_scale):
            pool_in = _RippleNumber.from_amount(self._x_amount)
            pool_out = _RippleNumber.from_amount(self._y_amount)
            numerator = _number_mul(pool_in, pool_out, rounding=ROUND_UPWARD)
            denominator_amount = _amount_sub(self._y_amount, dy, rounding=ROUND_DOWNWARD)
            if denominator_amount.signum() <= 0:
                return _from_units(0, is_xrp=self.x_is_xrp)
            denominator = _RippleNumber.from_amount(denominator_amount)
            ratio = _number_div(numerator, denominator, rounding=ROUND_UPWARD)
            numerator2 = _number_sub(ratio, pool_in, rounding=ROUND_UPWARD)
            fee = _number_div(
                _RippleNumber.from_int(self.trading_fee),
                _RippleNumber.from_int(TRADING_FEE_SCALE),
                rounding=ROUND_UPWARD,
            )
            fee_mult = _number_sub(_RippleNumber.from_int(1), fee, rounding=ROUND_DOWNWARD)
            if fee_mult.signum() <= 0:
                return _from_units(0, is_xrp=self.x_is_xrp)
            swap_in = _number_div(numerator2, fee_mult, rounding=ROUND_UPWARD)
            if swap_in.signum() < 0:
                return _from_units(0, is_xrp=self.x_is_xrp)
            return swap_in.to_amount(is_xrp=self.x_is_xrp, rounding=ROUND_UPWARD)

    def dx_for_out_st(self, dy: Amount) -> Amount:
        if dy.signum() <= 0:
            return _from_units(0, is_xrp=self.x_is_xrp)
        max_out = self.max_out_amount()
        if dy > max_out:
            raise AMMOverAsk(max_out=max_out, dx_for_max=self.swap_in_given_out_st(max_out))
        return self.swap_in_given_out_st(dy)

    def apply_fill_st(self, dx: Amount, dy: Amount) -> None:
        if dx.signum() < 0 or dy.signum() < 0:
            raise RuntimeError("AMM fill amounts must be non-negative")
        if dx.signum() == 0 and dy.signum() == 0:
            return
        if dy >= self._y_amount:
            raise RuntimeError("AMM fill would drain the output reserve")

        prior_spq = self.spq_quality_int()
        prior_product = _amount_to_fraction(self._x_amount) * _amount_to_fraction(self._y_amount)

        with _ripple_number_mantissa_scale_guard(self.number_mantissa_scale):
            self._x_amount = _amount_add(self._x_amount, dx, rounding=ROUND_TO_NEAREST)
            self._y_amount = _amount_sub(self._y_amount, dy, rounding=ROUND_TO_NEAREST)
        self._sync_public_reserves()

        if self._x_amount.signum() <= 0 or self._y_amount.signum() <= 0:
            raise RuntimeError("AMM reserves must remain strictly positive after fill")
        new_product = _amount_to_fraction(self._x_amount) * _amount_to_fraction(self._y_amount)
        if new_product < prior_product:
            product_gap = (prior_product - new_product) / prior_product
            if product_gap >= PRODUCT_RELATIVE_DISTANCE:
                raise RuntimeError("AMM fill broke the pool product invariant")
        if self.spq_quality_int() > prior_spq:
            raise RuntimeError("AMM SPQ improved after a fill")

    def synthetic_segment_for_quality(
        self,
        q_threshold: Quality,
        *,
        max_out_cap: Amount | None = None,
        require_post_spq_floor: bool = True,
    ) -> Segment | None:
        if q_threshold.is_zero():
            return None

        spq = self.spq_quality_int()
        if spq.is_zero():
            return None
        if require_post_spq_floor:
            if not (spq > q_threshold):
                return None
            if _within_relative_distance(spq, q_threshold, dist=QUALITY_RELATIVE_DISTANCE):
                return None
        elif spq < q_threshold:
            return None
        with _ripple_number_mantissa_scale_guard(self.number_mantissa_scale):
            proposal = _solve_quality_crossing_units(
                x_amount=self._x_amount,
                y_amount=self._y_amount,
                x_is_xrp=self.x_is_xrp,
                y_is_xrp=self.y_is_xrp,
                keep_fee_units=self._keep_fee_units,
                q_threshold=q_threshold,
            )
            if proposal is None:
                return None
            start_with_taker_gets = proposal.start_with_taker_gets

            def build_segment_from_start(start_amount: Amount) -> tuple[Amount, Amount] | None:
                if _amount_to_units_floor(start_amount) <= 0:
                    return None
                if start_with_taker_gets:
                    out_amount = start_amount
                    in_amount = self.swap_in_given_out_st(out_amount)
                    if _amount_to_units_floor(in_amount) <= 0:
                        return None
                    return out_amount, in_amount

                in_amount = start_amount
                out_amount = self.swap_out_given_in_st(in_amount)
                if _amount_to_units_floor(out_amount) <= 0:
                    return None
                return out_amount, in_amount

            built = build_segment_from_start(proposal.start_amount)
            if built is None:
                return None
            out_amount, in_amount = built
            q_slice = Quality.from_amounts(out_amount, in_amount)
            needs_reduced_retry = q_slice < q_threshold
            if needs_reduced_retry:
                reduced_start = _reduce_offer_amount(
                    out_amount if start_with_taker_gets else in_amount
                )
                rebuilt = build_segment_from_start(reduced_start)
                if rebuilt is None:
                    if q_slice < q_threshold:
                        return None
                else:
                    out_amount, in_amount = rebuilt
                    q_slice = Quality.from_amounts(out_amount, in_amount)

            if max_out_cap is not None:
                cap_units = _amount_to_units_floor(max_out_cap)
                if cap_units <= 0:
                    return None
                out_units = _amount_to_units_floor(out_amount)
                if out_units > cap_units:
                    out_amount = _from_units(cap_units, is_xrp=self.y_is_xrp)
                    in_amount = self.swap_in_given_out_st(out_amount)
                    if _amount_to_units_floor(in_amount) <= 0:
                        return None

            q_slice = Quality.from_amounts(out_amount, in_amount)
            if q_slice < q_threshold:
                return None

            if require_post_spq_floor:
                shadow = self.clone()
                shadow.apply_fill_st(in_amount, out_amount)
                if shadow.spq_quality_int() < q_threshold:
                    return None

            return Segment(
                src="AMM",
                quality=q_slice,
                out_max=out_amount,
                in_at_out_max=in_amount,
            )


__all__ = ["AMM", "AMMOverAsk"]
