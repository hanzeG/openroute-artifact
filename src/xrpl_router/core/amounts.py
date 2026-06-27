"""Rippled-aligned amount primitives for the rebuilt single-path kernel."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import total_ordering

from .exc import AmountDomainError, InvariantViolation, NormalisationError, UnsupportedKernelOperation

DROPS_PER_XRP = 1_000_000
INT64_MIN = -(2**63)
INT64_MAX = (2**63) - 1
IOU_MANTISSA_MIN = 10**15
IOU_MANTISSA_MAX = (10**16) - 1
IOU_EXP_MIN = -96
IOU_EXP_MAX = 80


def _require_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise AmountDomainError(f"{name} must be an int")


def _require_int64(name: str, value: int) -> None:
    _require_int(name, value)
    if value < INT64_MIN or value > INT64_MAX:
        raise AmountDomainError(f"{name} must fit signed int64 range")


def _power_ten(power: int) -> int:
    if power < 0:
        raise InvariantViolation("power must be non-negative")
    return 10**power

def _div_toward_zero(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise InvariantViolation("divisor must be positive")
    if value >= 0:
        return value // divisor
    return -((-value) // divisor)


def _div_nearest_even(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise InvariantViolation("divisor must be positive")
    quotient, remainder = divmod(value, divisor)
    doubled = remainder * 2
    if doubled > divisor or (doubled == divisor and quotient % 2 == 1):
        quotient += 1
    return quotient


def floor_div(numerator: int, denominator: int) -> int:
    """Round down on the integer grid."""
    _require_int("numerator", numerator)
    _require_int("denominator", denominator)
    if numerator < 0 or denominator <= 0:
        raise AmountDomainError("floor_div requires numerator >= 0 and denominator > 0")
    return numerator // denominator


def ceil_div(numerator: int, denominator: int) -> int:
    """Round up on the integer grid."""
    _require_int("numerator", numerator)
    _require_int("denominator", denominator)
    if numerator < 0 or denominator <= 0:
        raise AmountDomainError("ceil_div requires numerator >= 0 and denominator > 0")
    if numerator == 0:
        return 0
    return -(-numerator // denominator)


def mul_ratio_round_down(value: int, numerator: int, denominator: int) -> int:
    """Scale `value` by `numerator / denominator`, rounding down."""
    _require_int("value", value)
    _require_int("numerator", numerator)
    _require_int("denominator", denominator)
    if value < 0 or numerator < 0 or denominator <= 0:
        raise AmountDomainError(
            "mul_ratio_round_down requires non-negative inputs and denominator > 0"
        )
    if value == 0 or numerator == 0:
        return 0
    return (value * numerator) // denominator


def mul_ratio_round_up(value: int, numerator: int, denominator: int) -> int:
    """Scale `value` by `numerator / denominator`, rounding up."""
    _require_int("value", value)
    _require_int("numerator", numerator)
    _require_int("denominator", denominator)
    if value < 0 or numerator < 0 or denominator <= 0:
        raise AmountDomainError("mul_ratio_round_up requires non-negative inputs and denominator > 0")
    if value == 0 or numerator == 0:
        return 0
    return ceil_div(value * numerator, denominator)


@dataclass(frozen=True, order=True)
class XRPAmount:
    """XRP represented as signed integer drops, matching rippled value semantics."""

    drops: int

    def __post_init__(self) -> None:
        _require_int64("drops", self.drops)

    def is_zero(self) -> bool:
        return self.drops == 0

    def signum(self) -> int:
        return (self.drops > 0) - (self.drops < 0)

    def as_fraction(self) -> Fraction:
        return Fraction(self.drops, 1)

    def __add__(self, other: object) -> "XRPAmount":
        if not isinstance(other, XRPAmount):
            raise UnsupportedKernelOperation("XRPAmount addition requires XRPAmount")
        return XRPAmount(self.drops + other.drops)

    def __sub__(self, other: object) -> "XRPAmount":
        if not isinstance(other, XRPAmount):
            raise UnsupportedKernelOperation("XRPAmount subtraction requires XRPAmount")
        return XRPAmount(self.drops - other.drops)


def parse_xrp_drops(value: object, *, field_name: str) -> XRPAmount:
    try:
        text = str(value).strip()
        drops = int(text)
    except Exception as exc:
        raise AmountDomainError(f"invalid {field_name}: bad XRP drops: {exc}") from exc
    try:
        return XRPAmount(drops)
    except AmountDomainError as exc:
        raise AmountDomainError(f"invalid {field_name}: bad XRP drops: {exc}") from exc


def parse_xrp_decimal(value: object, *, field_name: str) -> XRPAmount:
    text = str(value).strip()
    if not text:
        raise AmountDomainError(f"invalid {field_name}: empty decimal string")

    sign = 1
    if text[0] == "+":
        text = text[1:]
    elif text[0] == "-":
        sign = -1
        text = text[1:]

    if not text:
        raise AmountDomainError(f"invalid {field_name}: empty decimal string")
    if text.count(".") > 1:
        raise AmountDomainError(f"invalid {field_name}: too many decimal points")

    if "." in text:
        whole_part, frac_part = text.split(".", 1)
    else:
        whole_part, frac_part = text, ""

    if not whole_part:
        whole_part = "0"
    if not whole_part.isdigit() or (frac_part and not frac_part.isdigit()):
        raise AmountDomainError(f"invalid {field_name}: malformed XRP decimal")
    if len(frac_part) > 6:
        if any(ch != "0" for ch in frac_part[6:]):
            raise AmountDomainError(f"invalid {field_name}: XRP amount must be exact on drop grid")
        frac_part = frac_part[:6]

    drops = int(whole_part) * DROPS_PER_XRP
    if frac_part:
        drops += int(frac_part.ljust(6, "0"))
    drops *= sign

    try:
        return XRPAmount(drops)
    except AmountDomainError as exc:
        raise AmountDomainError(f"invalid {field_name}: bad XRP decimal: {exc}") from exc


def _normalize_iou(mantissa: int, exponent: int) -> tuple[int, int]:
    _require_int64("mantissa", mantissa)
    _require_int("exponent", exponent)
    if mantissa == 0:
        return 0, -100

    negative = mantissa < 0
    if negative:
        mantissa = -mantissa

    while mantissa < IOU_MANTISSA_MIN and exponent > IOU_EXP_MIN:
        mantissa *= 10
        exponent -= 1

    if mantissa < IOU_MANTISSA_MIN and exponent == IOU_EXP_MIN:
        return 0, -100

    if mantissa > IOU_MANTISSA_MAX:
        divisor = 1
        while mantissa // divisor > IOU_MANTISSA_MAX:
            divisor *= 10
            exponent += 1
        mantissa = _div_nearest_even(mantissa, divisor)
        if mantissa > IOU_MANTISSA_MAX:
            mantissa = _div_nearest_even(mantissa, 10)
            exponent += 1
        if exponent > IOU_EXP_MAX:
            raise NormalisationError("IOUAmount exponent overflow during normalization")

    if exponent < IOU_EXP_MIN or exponent > IOU_EXP_MAX:
        if exponent > IOU_EXP_MAX:
            raise NormalisationError("IOUAmount exponent is out of supported range")
        return 0, -100

    if negative:
        mantissa = -mantissa
    return mantissa, exponent


@total_ordering
@dataclass(frozen=True)
class IOUAmount:
    """IOU represented as signed canonical mantissa/exponent, matching rippled."""

    mantissa: int
    exponent: int

    def __post_init__(self) -> None:
        mantissa, exponent = _normalize_iou(self.mantissa, self.exponent)
        object.__setattr__(self, "mantissa", mantissa)
        object.__setattr__(self, "exponent", exponent)

    def is_zero(self) -> bool:
        return self.mantissa == 0

    def signum(self) -> int:
        return (self.mantissa > 0) - (self.mantissa < 0)

    def as_fraction(self) -> Fraction:
        if self.is_zero():
            return Fraction(0, 1)
        if self.exponent >= 0:
            return Fraction(self.mantissa * _power_ten(self.exponent), 1)
        return Fraction(self.mantissa, _power_ten(-self.exponent))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, IOUAmount):
            raise UnsupportedKernelOperation("IOUAmount comparison requires IOUAmount")
        return self.as_fraction() < other.as_fraction()

    def __add__(self, other: object) -> "IOUAmount":
        if not isinstance(other, IOUAmount):
            raise UnsupportedKernelOperation("IOUAmount addition requires IOUAmount")
        if self.is_zero():
            return other
        if other.is_zero():
            return self
        left_exponent = self.exponent
        right_exponent = other.exponent
        left_mantissa = self.mantissa
        right_mantissa = other.mantissa

        while left_exponent < right_exponent:
            left_mantissa = _div_toward_zero(left_mantissa, 10)
            left_exponent += 1
        while right_exponent < left_exponent:
            right_mantissa = _div_toward_zero(right_mantissa, 10)
            right_exponent += 1

        mantissa = left_mantissa + right_mantissa
        if -10 <= mantissa <= 10:
            return IOUAmount(0, -100)
        return IOUAmount(mantissa, left_exponent)

    def __sub__(self, other: object) -> "IOUAmount":
        if not isinstance(other, IOUAmount):
            raise UnsupportedKernelOperation("IOUAmount subtraction requires IOUAmount")
        return self + IOUAmount(-other.mantissa, other.exponent)


def parse_iou_decimal(value: object, *, field_name: str) -> IOUAmount:
    text = str(value).strip()
    if not text:
        raise AmountDomainError(f"invalid {field_name}: empty decimal string")

    sign = 1
    if text[0] == "+":
        text = text[1:]
    elif text[0] == "-":
        sign = -1
        text = text[1:]

    if not text:
        raise AmountDomainError(f"invalid {field_name}: empty decimal string")
    if text.count(".") > 1:
        raise AmountDomainError(f"invalid {field_name}: too many decimal points")

    exponent_adjust = 0
    if "e" in text.lower():
        lower = text.lower()
        if lower.count("e") != 1:
            raise AmountDomainError(f"invalid {field_name}: too many exponent markers")
        significand_text, exponent_text = lower.split("e", 1)
        if not significand_text or not exponent_text:
            raise AmountDomainError(f"invalid {field_name}: malformed exponent notation")
        exp_sign = 1
        if exponent_text[0] == "+":
            exponent_text = exponent_text[1:]
        elif exponent_text[0] == "-":
            exp_sign = -1
            exponent_text = exponent_text[1:]
        if not exponent_text or not exponent_text.isdigit():
            raise AmountDomainError(f"invalid {field_name}: invalid exponent digits")
        exponent_adjust = exp_sign * int(exponent_text)
        text = significand_text

    if "." in text:
        whole_part, frac_part = text.split(".", 1)
    else:
        whole_part, frac_part = text, ""

    if not whole_part and not frac_part:
        raise AmountDomainError(f"invalid {field_name}: empty decimal string")
    if whole_part and not whole_part.isdigit():
        raise AmountDomainError(f"invalid {field_name}: invalid whole-number digits")
    if frac_part and not frac_part.isdigit():
        raise AmountDomainError(f"invalid {field_name}: invalid fractional digits")
    if not whole_part:
        whole_part = "0"

    digits_text = f"{whole_part}{frac_part}"
    exponent = exponent_adjust - len(frac_part)
    while digits_text.endswith("0") and digits_text:
        digits_text = digits_text[:-1]
        exponent += 1
    mantissa = 0 if not digits_text else int(digits_text)
    if sign < 0:
        mantissa = -mantissa

    try:
        return IOUAmount(mantissa, exponent)
    except (AmountDomainError, NormalisationError) as exc:
        raise AmountDomainError(
            f"invalid {field_name}: failed to normalize IOU amount: {exc}"
        ) from exc


Amount = XRPAmount | IOUAmount

__all__ = [
    "DROPS_PER_XRP",
    "INT64_MIN",
    "INT64_MAX",
    "IOU_MANTISSA_MIN",
    "IOU_MANTISSA_MAX",
    "IOU_EXP_MIN",
    "IOU_EXP_MAX",
    "floor_div",
    "ceil_div",
    "mul_ratio_round_down",
    "mul_ratio_round_up",
    "XRPAmount",
    "IOUAmount",
    "parse_xrp_drops",
    "parse_xrp_decimal",
    "parse_iou_decimal",
    "Amount",
]
