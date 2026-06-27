"""Minimal kernel datatypes for the rebuilt single-path calculation layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .amounts import Amount, IOUAmount, XRPAmount
from .exc import InvariantViolation, UnsupportedKernelOperation
from .quality import Quality


def _is_strictly_positive(amount: Amount) -> bool:
    if isinstance(amount, XRPAmount):
        return amount.drops > 0
    if isinstance(amount, IOUAmount):
        return amount.mantissa > 0
    raise UnsupportedKernelOperation("unsupported amount type")


@dataclass(frozen=True)
class Segment:
    """A single executable slice source for later single-path execution assembly."""

    src: Literal["AMM", "CLOB"]
    quality: Quality
    out_max: Amount
    in_at_out_max: Amount

    def __post_init__(self) -> None:
        if self.src not in {"AMM", "CLOB"}:
            raise UnsupportedKernelOperation("segment src must be 'AMM' or 'CLOB'")
        if not _is_strictly_positive(self.out_max) or not _is_strictly_positive(self.in_at_out_max):
            raise InvariantViolation("segment amounts must be strictly positive")


__all__ = ["Segment"]
