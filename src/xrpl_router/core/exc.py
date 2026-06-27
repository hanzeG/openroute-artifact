"""Fail-fast exception types for the rebuilt calculation kernel."""

__all__ = [
    "CoreError",
    "AmountDomainError",
    "NormalisationError",
    "UnsupportedKernelOperation",
    "InvariantViolation",
]


class CoreError(Exception):
    """Base exception for the calculation kernel."""


class AmountDomainError(CoreError):
    """Raised when an amount or arithmetic input violates the supported domain."""


class NormalisationError(CoreError):
    """Raised when an IOU value cannot be canonicalised into the supported range."""


class UnsupportedKernelOperation(CoreError):
    """Raised when code asks core to do work outside its strict supported scope."""


class InvariantViolation(CoreError):
    """Raised when a constructed value would break a kernel invariant."""
