from decimal import Decimal

from xrpl_router.amm import AMM
from xrpl_router.book_step import _execute_one_iteration
from xrpl_router.core import IOUAmount, Quality, XRPAmount
from xrpl_router.core.datatypes import Segment


def _mk_clob_segment(q: Quality) -> Segment:
    return Segment(
        src="CLOB",
        quality=q,
        out_max=XRPAmount(100),
        in_at_out_max=XRPAmount(100),
        source_id="clob-1",
    )


def _mk_amm() -> AMM:
    return AMM(
        Decimal("1000"),
        Decimal("1000"),
        Decimal("0"),
        x_is_xrp=True,
        y_is_xrp=True,
    )


def test_equal_spq_does_not_call_anchor_and_prefers_clob():
    q_top = Quality.from_amounts(XRPAmount(100), XRPAmount(100))
    clob = _mk_clob_segment(q_top)
    amm = _mk_amm()
    anchor_calls = []

    def anchor(q, cap):
        anchor_calls.append((q, cap))
        return Segment(
            src="AMM",
            quality=q,
            out_max=XRPAmount(100),
            in_at_out_max=XRPAmount(100),
            source_id="amm-1",
        )

    _, _, _, amm_used, _, _, _, trace = _execute_one_iteration(
        [clob],
        target_out=XRPAmount(50),
        send_max=None,
        limit_quality=None,
        amm_anchor=anchor,
        amm_spq=q_top,  # equal to CLOB top quality
        amm_obj=amm,
    )

    assert len(anchor_calls) == 0
    assert amm_used is False
    assert len(trace) == 1
    assert trace[0]["src"] == "CLOB"


def test_strictly_better_spq_calls_anchor_and_can_use_amm():
    q_top = Quality.from_amounts(XRPAmount(100), XRPAmount(100))
    q_better = Quality.from_amounts(XRPAmount(101), XRPAmount(100))
    clob = _mk_clob_segment(q_top)
    amm = _mk_amm()
    anchor_calls = []

    def anchor(q, cap):
        anchor_calls.append((q, cap))
        return Segment(
            src="AMM",
            quality=q,
            out_max=XRPAmount(100),
            in_at_out_max=XRPAmount(100),
            source_id="amm-1",
        )

    _, _, _, amm_used, _, _, _, trace = _execute_one_iteration(
        [clob],
        target_out=XRPAmount(50),
        send_max=None,
        limit_quality=None,
        amm_anchor=anchor,
        amm_spq=q_better,  # strictly better than CLOB top quality
        amm_obj=amm,
    )

    assert len(anchor_calls) == 1
    assert amm_used is True
    assert len(trace) == 1
    assert trace[0]["src"] == "AMM"


def test_strictly_better_spq_does_not_mix_clob_in_same_iteration():
    q_top = Quality.from_amounts(XRPAmount(100), XRPAmount(100))
    q_better = Quality.from_amounts(XRPAmount(101), XRPAmount(100))
    clob = _mk_clob_segment(q_top)
    amm = _mk_amm()

    def anchor(q, cap):
        # Deliberately smaller than target_out so old behaviour would continue into CLOB.
        return Segment(
            src="AMM",
            quality=q,
            out_max=XRPAmount(30),
            in_at_out_max=XRPAmount(30),
            source_id="amm-1",
        )

    filled, spent, next_segs, amm_used, _, _, _, trace = _execute_one_iteration(
        [clob],
        target_out=XRPAmount(100),
        send_max=None,
        limit_quality=None,
        amm_anchor=anchor,
        amm_spq=q_better,
        amm_obj=amm,
    )

    assert amm_used is True
    assert filled == XRPAmount(30)
    assert spent == amm.swap_in_given_out_st(XRPAmount(30))
    assert len(trace) == 1
    assert trace[0]["src"] == "AMM"
    # CLOB must remain untouched for the next iteration.
    assert len(next_segs) == 1
    assert next_segs[0].src == "CLOB"
    assert next_segs[0].out_max == clob.out_max
    assert next_segs[0].in_at_out_max == clob.in_at_out_max


def test_forward_sendmax_caps_amm_anchor_take():
    q_top = Quality.from_amounts(XRPAmount(100), XRPAmount(100))
    q_better = Quality.from_amounts(XRPAmount(101), XRPAmount(100))
    clob = _mk_clob_segment(q_top)
    amm = _mk_amm()

    def anchor(q, cap):
        return Segment(
            src="AMM",
            quality=q,
            out_max=XRPAmount(100),
            in_at_out_max=XRPAmount(100),
            source_id="amm-1",
        )

    filled, spent, _, amm_used, _, _, _, trace = _execute_one_iteration(
        [clob],
        target_out=XRPAmount(100),
        send_max=XRPAmount(40),
        limit_quality=None,
        amm_anchor=anchor,
        amm_spq=q_better,
        amm_obj=amm,
    )

    assert amm_used is True
    assert filled == amm.swap_out_given_in_st(XRPAmount(40))
    assert spent == XRPAmount(40)
    assert len(trace) == 1
    assert trace[0]["src"] == "AMM"


def test_clob_tiny_tail_leg_is_consumed():
    tiny_out = IOUAmount.from_components(200, -15)  # 2e-13
    one_drop = XRPAmount(1)
    seg = Segment(
        src="CLOB",
        quality=Quality.from_amounts(tiny_out, one_drop),
        out_max=tiny_out,
        in_at_out_max=one_drop,
        source_id="clob-dust",
    )

    filled, spent, next_segs, amm_used, _, _, _, trace = _execute_one_iteration(
        [seg],
        target_out=tiny_out,
        send_max=None,
        limit_quality=None,
        amm_anchor=None,
        amm_spq=None,
        amm_obj=None,
    )

    assert amm_used is False
    assert filled == tiny_out
    assert spent == one_drop
    assert len(trace) == 1
    assert trace[0]["src"] == "CLOB"
    assert trace[0]["source_id"] == "clob-dust"
    assert len(next_segs) == 0


def test_quality_filter_empty_clob_still_allows_amm_only_without_index_error():
    q_clob = Quality.from_amounts(XRPAmount(50), XRPAmount(100))
    q_floor = Quality.from_amounts(XRPAmount(90), XRPAmount(100))
    clob = _mk_clob_segment(q_clob)
    amm = _mk_amm()

    def anchor(q, cap):
        return Segment(
            src="AMM",
            quality=q,
            out_max=XRPAmount(50),
            in_at_out_max=XRPAmount(25),
            source_id="amm-1",
        )

    filled, spent, next_segs, amm_used, _, _, _, trace = _execute_one_iteration(
        [clob],
        target_out=XRPAmount(40),
        send_max=None,
        limit_quality=q_floor,
        amm_anchor=anchor,
        amm_spq=amm.spq_quality_int(),
        amm_obj=amm,
    )

    assert amm_used is True
    assert filled == XRPAmount(40)
    assert spent.value > 0
    assert len(next_segs) == 0
    assert len(trace) == 1
    assert trace[0]["src"] == "AMM"


def test_amm_only_overask_keeps_capped_partial_slice():
    amm = _mk_amm()
    filled, spent, next_segs, amm_used, _, _, _, trace = _execute_one_iteration(
        [],
        target_out=XRPAmount(50_000),  # intentionally above AMM non-drain capacity
        send_max=XRPAmount(20),
        limit_quality=None,
        amm_anchor=None,
        amm_spq=amm.spq_quality_int(),
        amm_obj=amm,
    )

    assert amm_used is True
    assert filled.value > 0
    assert spent.value > 0
    assert len(next_segs) == 0
    assert len(trace) == 1
    assert trace[0]["src"] == "AMM"


def test_amm_only_with_limit_quality_does_not_cross_below_floor():
    amm = AMM(
        Decimal("1000000"),
        Decimal("1000000"),
        Decimal("0"),
        x_is_xrp=True,
        y_is_xrp=True,
    )
    q_floor = Quality.from_amounts(XRPAmount(8), XRPAmount(10))  # 0.8

    filled, spent, _, amm_used, _, _, _, trace = _execute_one_iteration(
        [],
        target_out=XRPAmount(400000),
        send_max=None,
        limit_quality=q_floor,
        amm_anchor=lambda q, cap: amm.synthetic_segment_for_quality(q, max_out_cap=cap),
        amm_spq=amm.spq_quality_int(),
        amm_obj=amm,
    )

    assert amm_used is True
    assert len(trace) == 1
    assert trace[0]["src"] == "AMM"
    q_avg = Quality.from_amounts(filled, spent)
    assert q_avg >= q_floor


def test_forward_sendmax_anchored_amm_uses_curve_not_linear_scaling():
    # Mirror a real-world XRP->rUSD shape where AMM SPQ is above CLOB top quality.
    amm = AMM(
        Decimal("691055.850947"),
        Decimal("1438146.053667077"),
        Decimal("0.00327"),
        x_is_xrp=True,
        y_is_xrp=False,
    )
    clob = Segment(
        src="CLOB",
        quality=Quality.from_amounts(IOUAmount.from_decimal(Decimal("4.7625648")), XRPAmount(2_333_914)),
        out_max=IOUAmount.from_decimal(Decimal("4.7625648")),
        in_at_out_max=XRPAmount(2_333_914),
        source_id="clob-top",
    )
    target_out = IOUAmount.from_decimal(Decimal("1847.224218096000"))
    send_max = XRPAmount(148_454_706)

    def anchor(q, cap):
        seg = amm.synthetic_segment_for_quality(q, max_out_cap=cap)
        assert seg is not None
        return seg

    filled, spent, _, amm_used, _, _, _, trace = _execute_one_iteration(
        [clob],
        target_out=target_out,
        send_max=send_max,
        limit_quality=None,
        amm_anchor=anchor,
        amm_spq=amm.spq_quality_int(),
        amm_obj=amm,
    )

    assert amm_used is True
    assert len(trace) == 1
    assert trace[0]["src"] == "AMM"
    # Under send_max binding, AMM forward output should match swap_out_given_in curve quote.
    assert spent == send_max
    assert filled == amm.swap_out_given_in_st(send_max)
