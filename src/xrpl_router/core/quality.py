"""Rippled-aligned quality encoding for the rebuilt single-path kernel."""

from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering

from .amounts import Amount, IOUAmount, IOU_MANTISSA_MIN, XRPAmount
from .exc import AmountDomainError, NormalisationError, UnsupportedKernelOperation

RATE_SCALE = 10**17
QUALITY_MANTISSA_MASK = 0x00FFFFFFFFFFFFFF
UINT64_MAX = (2**64) - 1


def _positive_components(amount: Amount) -> tuple[int, int]:
    if isinstance(amount, XRPAmount):
        if amount.drops <= 0:
            raise AmountDomainError("quality requires strictly positive amounts")
        mantissa = amount.drops
        exponent = 0
        while mantissa < IOU_MANTISSA_MIN:
            mantissa *= 10
            exponent -= 1
        return mantissa, exponent
    if amount.mantissa <= 0:
        raise AmountDomainError("quality requires strictly positive amounts")
    return amount.mantissa, amount.exponent


def _divide_like_rippled(num: Amount, den: Amount) -> IOUAmount:
    num_mantissa, num_exponent = _positive_components(num)
    den_mantissa, den_exponent = _positive_components(den)
    quotient = ((num_mantissa * RATE_SCALE) // den_mantissa) + 5
    exponent = num_exponent - den_exponent - 17
    return IOUAmount(quotient, exponent)


def _encode_rate(rate: IOUAmount) -> int:
    if rate.is_zero():
        return 0
    stored_exponent = rate.exponent + 100
    if stored_exponent < 0 or stored_exponent > 255:
        return 0
    return (stored_exponent << 56) | abs(rate.mantissa)


@total_ordering
@dataclass(frozen=True)
class Quality:
    """Quality wraps rippled's encoded offer rate; higher quality still compares better."""

    value: int = 0

    def __post_init__(self) -> None:
        if type(self.value) is not int or self.value < 0 or self.value > UINT64_MAX:
            raise AmountDomainError("quality value must fit uint64 range")

    @classmethod
    def from_rate(cls, rate: IOUAmount) -> "Quality":
        if rate.mantissa <= 0:
            raise AmountDomainError("quality rate must be strictly positive")
        return cls(_encode_rate(rate))

    @classmethod
    def from_amounts(cls, out_amount: Amount, in_amount: Amount) -> "Quality":
        try:
            return cls(_encode_rate(_divide_like_rippled(in_amount, out_amount)))
        except NormalisationError:
            return cls(0)

    def is_zero(self) -> bool:
        return self.value == 0

    def rate(self) -> IOUAmount:
        if self.value == 0:
            return IOUAmount(0, -100)
        exponent = (self.value >> 56) - 100
        mantissa = self.value & QUALITY_MANTISSA_MASK
        return IOUAmount(mantissa, exponent)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Quality):
            raise UnsupportedKernelOperation("Quality comparison requires Quality")
        return self.value > other.value


__all__ = ["Quality"]
