from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Tuple

from .core import Amount, IOUAmount, IOU_MANTISSA_MAX, IOU_MANTISSA_MIN, Quality, XRPAmount

if TYPE_CHECKING:
    from .steps import Step


def _zero_like(x: Amount) -> Amount:
    return XRPAmount(0) if isinstance(x, XRPAmount) else IOUAmount(0, 0)


def _gt_zero(x: Amount) -> bool:
    z = _zero_like(x)
    return (not x.is_zero()) and (x > z)


def _compose_path_quality(step_quals: List[Quality]) -> Optional[Quality]:
    if not step_quals:
        return None
    mantissa = 1
    exponent = 0
    for quality in step_quals:
        rate = quality.rate()
        if rate.is_zero():
            return None
        mantissa *= rate.mantissa
        exponent += rate.exponent
        while mantissa > IOU_MANTISSA_MAX:
            mantissa //= 10
            exponent += 1
        while 0 < mantissa < IOU_MANTISSA_MIN:
            mantissa *= 10
            exponent -= 1
    rate = IOUAmount(mantissa, exponent)
    if rate.is_zero():
        return None
    return Quality.from_rate(rate)


@dataclass
class PaymentSandbox:
    staged: List[Tuple[Amount, Amount]]

    def __init__(self) -> None:
        self.staged = []

    def stage_after_iteration(self, dx: Amount, dy: Amount) -> None:
        self.staged.append((dx, dy))

    def apply(self, sink: Optional[Callable[[Amount, Amount], None]]) -> None:
        if sink is None:
            self.staged.clear()
            return
        for dx, dy in self.staged:
            sink(dx, dy)
        self.staged.clear()


def flow(
    payment_sandbox: PaymentSandbox,
    strands: Iterable[List["Step"]],
    out_req: Amount,
    *,
    send_max: Optional[Amount] = None,
    limit_quality: Optional[Quality] = None,
    apply_sink: Optional[Callable[[Amount, Amount], None]] = None,
) -> Tuple[Amount, Amount]:
    if out_req.is_zero() or not _gt_zero(out_req):
        z = _zero_like(out_req)
        return (z, z)

    strands = list(strands)
    if len(strands) != 1:
        raise RuntimeError("rebuild flow currently expects exactly one strand")

    remaining_out: Amount = out_req
    remaining_in: Optional[Amount] = send_max
    actual_in: Optional[Amount] = None
    actual_out: Amount = _zero_like(out_req)

    iters = 0

    while _gt_zero(remaining_out) and (remaining_in is None or remaining_in >= _zero_like(remaining_in)):
        if remaining_in is not None and remaining_in < _zero_like(remaining_in):
            break

        active: List[Tuple[Quality, List["Step"]]] = []
        for strand in strands:
            try:
                step_qs = [s.quality_upper_bound() for s in strand]
                q = _compose_path_quality(step_qs)
            except Exception:
                q = None
            if q is None:
                continue
            if limit_quality is not None and q.rate() < limit_quality.rate():
                continue
            active.append((q, strand))

        if not active:
            break

        active.sort(key=lambda t: (t[0].rate().exponent, t[0].rate().mantissa), reverse=True)

        attempt_succeeded = False
        for _, best in active:
            need = remaining_out
            sb = payment_sandbox
            limiting_idx: Optional[int] = None
            n_steps = len(best)
            rev_cache: List[Tuple[Amount, Amount] | None] = [None] * n_steps

            for rev_pos, step in enumerate(reversed(best)):
                idx_forward = n_steps - 1 - rev_pos
                out_i, in_i = step.rev(sb, need)
                if not _gt_zero(out_i):
                    need = _zero_like(out_i)
                    limiting_idx = idx_forward
                    break
                if out_i < need:
                    limiting_idx = idx_forward
                    out_i, in_i = step.rev(sb, out_i)
                rev_cache[idx_forward] = (out_i, in_i)
                need = in_i

            if not _gt_zero(need):
                payment_sandbox.apply(None)
                continue

            required_in = need

            in_spent_add = required_in
            out_propagate = remaining_out

            if remaining_in is None and limiting_idx is None:
                pass
            elif remaining_in is not None and remaining_in < required_in:
                cap = remaining_in
                in_spent_add = remaining_in
                ok = True
                for i, step_i in enumerate(best):
                    if not _gt_zero(cap):
                        ok = False
                        break
                    out_i, in_i = step_i.fwd(sb, cap)
                    if not _gt_zero(out_i) or not _gt_zero(in_i):
                        ok = False
                        break
                    if i == 0:
                        in_spent_add = in_i
                    out_propagate = out_i
                    cap = out_i
                if not ok:
                    payment_sandbox.apply(None)
                    continue
            elif limiting_idx is not None:
                cached = rev_cache[limiting_idx]
                if cached is None:
                    payment_sandbox.apply(None)
                    continue
                out_propagate = cached[0]
                ok = True
                for step_i in best[limiting_idx + 1 :]:
                    if not _gt_zero(out_propagate):
                        ok = False
                        break
                    next_out, next_in = step_i.fwd(sb, out_propagate)
                    if not _gt_zero(next_out) or not _gt_zero(next_in):
                        ok = False
                        break
                    out_propagate = next_out
                if not ok:
                    payment_sandbox.apply(None)
                    continue

            actual_out = actual_out + out_propagate
            remaining_out = out_req - actual_out
            if actual_in is None:
                actual_in = in_spent_add
            else:
                actual_in = actual_in + in_spent_add
            if remaining_in is not None:
                remaining_in = remaining_in - in_spent_add

            payment_sandbox.apply(apply_sink)
            attempt_succeeded = True
            if remaining_in is not None and remaining_in < required_in:
                remaining_out = _zero_like(out_req)
            elif limiting_idx is not None:
                remaining_out = _zero_like(out_req)
            break

        if not attempt_succeeded:
            break

        iters += 1
        if iters >= 128:
            break

    if actual_in is None:
        actual_in = _zero_like(out_req)
    return actual_in, actual_out
