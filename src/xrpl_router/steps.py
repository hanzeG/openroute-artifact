from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from .core import Amount, Quality, XRPAmount

if TYPE_CHECKING:
    from .flow import PaymentSandbox


class Step:
    """Abstract payment step interface (integer domain, whitepaper §1.3.1).

    Reverse (rev): estimate required IN for a requested OUT (out_req).
    Forward (fwd): spend up to in_cap and produce OUT. Both may stage effects into PaymentSandbox.

    Each step supports reverse (rev) and forward (fwd) execution and can
    advertise a conservative quality upper bound for sorting (multi-path).
    """

    _cached_in: Amount
    _cached_out: Amount

    def rev(self, sandbox: "PaymentSandbox", out_req: Amount) -> Tuple[Amount, Amount]:
        raise NotImplementedError

    def fwd(self, sandbox: "PaymentSandbox", in_cap: Amount) -> Tuple[Amount, Amount]:
        raise NotImplementedError

    def quality_upper_bound(self) -> Quality:
        raise NotImplementedError

    def cached_in(self) -> Amount:
        return getattr(self, "_cached_in", XRPAmount(0))

    def cached_out(self) -> Amount:
        return getattr(self, "_cached_out", XRPAmount(0))
