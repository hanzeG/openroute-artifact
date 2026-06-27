from __future__ import annotations

import pytest

from xrpl_router.core import Quality, XRPAmount
from xrpl_router.flow import PaymentSandbox, flow
from xrpl_router.steps import Step


class MockStep(Step):
    def __init__(
        self,
        *,
        rev_map: dict[int, tuple[int, int]],
        fwd_map: dict[int, tuple[int, int]] | None = None,
        quality: Quality | None = None,
        stage_on_fwd: bool = False,
    ) -> None:
        self._rev_map = rev_map
        self._fwd_map = fwd_map or {}
        self._quality = quality or Quality.from_amounts(XRPAmount(1), XRPAmount(1))
        self._stage_on_fwd = stage_on_fwd
        self._cached_out = XRPAmount(0)
        self._cached_in = XRPAmount(0)
        self.rev_calls: list[int] = []
        self.fwd_calls: list[int] = []

    def rev(self, sandbox: PaymentSandbox, out_req: XRPAmount) -> tuple[XRPAmount, XRPAmount]:
        del sandbox
        self.rev_calls.append(out_req.drops)
        out_drops, in_drops = self._rev_map[out_req.drops]
        self._cached_out = XRPAmount(out_drops)
        self._cached_in = XRPAmount(in_drops)
        return self._cached_out, self._cached_in

    def fwd(self, sandbox: PaymentSandbox, in_cap: XRPAmount) -> tuple[XRPAmount, XRPAmount]:
        self.fwd_calls.append(in_cap.drops)
        out_drops, in_drops = self._fwd_map[in_cap.drops]
        self._cached_out = XRPAmount(out_drops)
        self._cached_in = XRPAmount(in_drops)
        if self._stage_on_fwd and self._cached_out.drops > 0 and self._cached_in.drops > 0:
            sandbox.stage_after_iteration(self._cached_in, self._cached_out)
        return self._cached_out, self._cached_in

    def quality_upper_bound(self) -> Quality:
        return self._quality


def test_payment_sandbox_apply_commits_then_clears() -> None:
    sandbox = PaymentSandbox()
    sink_calls: list[tuple[XRPAmount, XRPAmount]] = []

    sandbox.stage_after_iteration(XRPAmount(7), XRPAmount(5))
    sandbox.stage_after_iteration(XRPAmount(3), XRPAmount(2))
    sandbox.apply(lambda dx, dy: sink_calls.append((dx, dy)))

    assert sink_calls == [
        (XRPAmount(7), XRPAmount(5)),
        (XRPAmount(3), XRPAmount(2)),
    ]
    assert sandbox.staged == []


def test_flow_exact_out_reverse_only_matches_whitepaper_reverse_first_shape() -> None:
    step = MockStep(rev_map={50: (50, 60)})
    sandbox = PaymentSandbox()

    actual_in, actual_out = flow(sandbox, [[step]], XRPAmount(50))

    assert actual_in == XRPAmount(60)
    assert actual_out == XRPAmount(50)
    assert step.rev_calls == [50]
    assert step.fwd_calls == []
    assert sandbox.staged == []


def test_flow_sendmax_binding_replays_first_step_forward_and_commits_staged_changes() -> None:
    step = MockStep(
        rev_map={50: (50, 60)},
        fwd_map={55: (45, 55)},
        stage_on_fwd=True,
    )
    sandbox = PaymentSandbox()
    sink_calls: list[tuple[XRPAmount, XRPAmount]] = []

    actual_in, actual_out = flow(
        sandbox,
        [[step]],
        XRPAmount(50),
        send_max=XRPAmount(55),
        apply_sink=lambda dx, dy: sink_calls.append((dx, dy)),
    )

    assert actual_in == XRPAmount(55)
    assert actual_out == XRPAmount(45)
    assert step.rev_calls == [50]
    assert step.fwd_calls == [55]
    assert sink_calls == [(XRPAmount(55), XRPAmount(45))]
    assert sandbox.staged == []


def test_flow_output_limited_middle_step_replays_downstream_forward_only() -> None:
    step1 = MockStep(rev_map={20: (20, 10)})
    step2 = MockStep(rev_map={40: (30, 20), 30: (30, 20)})
    step3 = MockStep(rev_map={50: (50, 40)}, fwd_map={30: (35, 30)})
    sandbox = PaymentSandbox()

    actual_in, actual_out = flow(sandbox, [[step1, step2, step3]], XRPAmount(50))

    assert actual_in == XRPAmount(10)
    assert actual_out == XRPAmount(35)
    assert step1.rev_calls == [20]
    assert step1.fwd_calls == []
    assert step2.rev_calls == [40, 30]
    assert step2.fwd_calls == []
    assert step3.rev_calls == [50]
    assert step3.fwd_calls == [30]


def test_flow_quality_floor_rejects_consumed_liquidity_below_threshold() -> None:
    step = MockStep(rev_map={50: (50, 100)})
    sandbox = PaymentSandbox()

    actual_in, actual_out = flow(
        sandbox,
        [[step]],
        XRPAmount(50),
        limit_quality=Quality.from_amounts(XRPAmount(60), XRPAmount(100)),
    )

    assert actual_in == XRPAmount(0)
    assert actual_out == XRPAmount(0)
    assert sandbox.staged == []


def test_flow_failed_forward_replay_rolls_back_staged_changes() -> None:
    step1 = MockStep(
        rev_map={45: (45, 60)},
        fwd_map={55: (40, 55)},
        stage_on_fwd=True,
    )
    step2 = MockStep(
        rev_map={50: (45, 45), 45: (45, 45)},
        fwd_map={40: (0, 0)},
        stage_on_fwd=True,
    )
    sandbox = PaymentSandbox()
    sink_calls: list[tuple[XRPAmount, XRPAmount]] = []

    actual_in, actual_out = flow(
        sandbox,
        [[step1, step2]],
        XRPAmount(50),
        send_max=XRPAmount(55),
        apply_sink=lambda dx, dy: sink_calls.append((dx, dy)),
    )

    assert actual_in == XRPAmount(0)
    assert actual_out == XRPAmount(0)
    assert sink_calls == []
    assert sandbox.staged == []


def test_flow_rejects_multi_strand_scope_in_rebuild() -> None:
    step = MockStep(rev_map={50: (50, 60)})

    with pytest.raises(RuntimeError, match="exactly one strand"):
        flow(PaymentSandbox(), [[step], [step]], XRPAmount(50))
