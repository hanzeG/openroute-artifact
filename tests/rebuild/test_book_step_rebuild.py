from __future__ import annotations

from decimal import Decimal

from xrpl_router.amm import AMM
from xrpl_router.book_offers import BookOfferSegment
from xrpl_router.book_step import (
    BookStep,
    RouterQuoteView,
    _AMMAggregateExecution,
    _competing_amm_segment,
    _execute_one_iteration,
    _limit_out_by_quality_function,
    _normalise_rows,
)
from xrpl_router.core import IOUAmount, Quality, Segment, XRPAmount
from xrpl_router.flow import PaymentSandbox


def iou(value: str) -> IOUAmount:
    return IOUAmount(int(Decimal(value) * (Decimal(10) ** 15)), -15)


def test_book_step_equal_spq_prefers_clob_over_amm() -> None:
    amm = AMM(Decimal("1000"), Decimal("1000"), Decimal("0"), x_is_xrp=True, y_is_xrp=True)
    q_top = Quality.from_amounts(XRPAmount(100), XRPAmount(100))
    rows = _normalise_rows(
        [
            Segment(
                src="CLOB",
                quality=q_top,
                out_max=XRPAmount(100),
                in_at_out_max=XRPAmount(100),
            )
        ]
    )

    filled, spent, _, amm_in, amm_out, trace = _execute_one_iteration(
        rows,
        target_out=XRPAmount(50),
        send_max=None,
        limit_quality=None,
        mode="payment",
        amm=amm,
    )

    assert filled == XRPAmount(50)
    assert spent == XRPAmount(50)
    assert amm_in == XRPAmount(0)
    assert amm_out == XRPAmount(0)
    assert len(trace) == 1
    assert trace[0]["src"] == "CLOB"


def test_book_step_strictly_better_spq_prefers_amm() -> None:
    amm = AMM(Decimal("1000"), Decimal("1000"), Decimal("0"), x_is_xrp=True, y_is_xrp=True)
    q_top = Quality.from_amounts(XRPAmount(90), XRPAmount(100))
    rows = _normalise_rows(
        [
            Segment(
                src="CLOB",
                quality=q_top,
                out_max=XRPAmount(100),
                in_at_out_max=XRPAmount(112),
            )
        ]
    )

    filled, spent, next_rows, amm_in, amm_out, trace = _execute_one_iteration(
        rows,
        target_out=XRPAmount(50),
        send_max=None,
        limit_quality=None,
        mode="payment",
        amm=amm,
    )

    assert filled == XRPAmount(50)
    assert spent == XRPAmount(51)
    assert amm_in == XRPAmount(51)
    assert amm_out == XRPAmount(50)
    assert len(trace) == 1
    assert trace[0]["src"] == "AMM"
    assert len(next_rows) == 1
    assert next_rows[0].segment.src == "CLOB"


def test_competing_amm_segment_accepts_quality_tie_after_anchor() -> None:
    top_quality = Quality.from_amounts(XRPAmount(9), XRPAmount(10))
    tied_segment = Segment(
        src="AMM",
        quality=top_quality,
        out_max=XRPAmount(90),
        in_at_out_max=XRPAmount(100),
    )

    class DummyAMM:
        def spq_quality_int(self) -> Quality:
            return Quality.from_amounts(XRPAmount(10), XRPAmount(10))

        def synthetic_segment_for_quality(
            self,
            quality: Quality,
            *,
            max_out_cap: XRPAmount | None,
            require_post_spq_floor: bool,
        ) -> Segment:
            assert quality == top_quality
            assert max_out_cap is None
            assert require_post_spq_floor is True
            return tied_segment

    segment = _competing_amm_segment(
        DummyAMM(),
        top_quality=top_quality,
        out_cap=XRPAmount(90),
        require_post_spq_floor=True,
        mode="offer_crossing",
        limit_quality=None,
    )

    assert segment == tied_segment


def test_competing_amm_segment_offer_crossing_loose_limit_uses_rippled_max_offer() -> None:
    top_quality = Quality.from_amounts(XRPAmount(9), XRPAmount(10))
    looser_limit_quality = Quality.from_amounts(XRPAmount(10), XRPAmount(10))
    max_offer_segment = Segment(
        src="AMM",
        quality=Quality.from_amounts(XRPAmount(91), XRPAmount(100)),
        out_max=XRPAmount(91),
        in_at_out_max=XRPAmount(100),
    )

    class DummyAMM:
        def spq_quality_int(self) -> Quality:
            return Quality.from_amounts(XRPAmount(11), XRPAmount(10))

        def max_offer_segment(self, *, max_out_cap: XRPAmount | None) -> Segment:
            assert max_out_cap == XRPAmount(90)
            return max_offer_segment

        def max_out_amount(self) -> XRPAmount:
            raise AssertionError("offer-crossing loose-limit path must not use raw near-drain max_out_amount")

        def swap_in_given_out_st(self, out: XRPAmount) -> XRPAmount:
            raise AssertionError("offer-crossing loose-limit path must not rebuild from raw max_out")

    segment = _competing_amm_segment(
        DummyAMM(),
        top_quality=top_quality,
        out_cap=XRPAmount(90),
        require_post_spq_floor=True,
        mode="offer_crossing",
        limit_quality=looser_limit_quality,
    )

    assert segment is max_offer_segment


def test_book_step_forward_budgeted_amm_uses_curve_not_linear_scaling() -> None:
    amm = AMM(Decimal("1000"), Decimal("1000"), Decimal("0"), x_is_xrp=True, y_is_xrp=True)
    q_top = Quality.from_amounts(XRPAmount(90), XRPAmount(100))
    rows = _normalise_rows(
        [
            Segment(
                src="CLOB",
                quality=q_top,
                out_max=XRPAmount(100),
                in_at_out_max=XRPAmount(112),
            )
        ]
    )
    budget = XRPAmount(40_000_000)

    filled, spent, _, amm_in, amm_out, trace = _execute_one_iteration(
        rows,
        target_out=XRPAmount(500_000_000),
        send_max=budget,
        limit_quality=None,
        mode="payment",
        amm=amm,
    )

    assert spent == budget
    assert filled == amm.swap_out_given_in_st(budget)
    assert amm_in == budget
    assert amm_out == filled
    assert len(trace) == 1
    assert trace[0]["src"] == "AMM"


def test_book_step_forward_budgeted_amm_spends_full_input_on_discrete_output_plateau() -> None:
    amm = AMM(
        Decimal("1306896.196063654"),
        Decimal("678673.182809"),
        Decimal("0.00519"),
        x_is_xrp=False,
        y_is_xrp=True,
    )
    budget = iou("16.057776")

    filled, spent, _, amm_in, amm_out, trace = _execute_one_iteration(
        [],
        target_out=XRPAmount(1_000_000_000_000),
        send_max=budget,
        limit_quality=None,
        mode="offer_crossing",
        amm=amm,
    )

    assert filled == XRPAmount(8_295_447)
    assert spent == budget
    assert amm_in == budget
    assert amm_out == filled
    assert len(trace) == 1
    assert trace[0]["src"] == "AMM"


def test_book_step_offer_crossing_amm_only_honors_limit_quality_without_clob_rows() -> None:
    amm = AMM(
        Decimal("680074.500291"),
        Decimal("1316371.18551031"),
        Decimal("0.00519"),
        x_is_xrp=True,
        y_is_xrp=False,
    )
    limit_quality = Quality.from_amounts(IOUAmount(4_927_436_017_353_250, -11), XRPAmount(25_599_596_934))
    remaining_out = IOUAmount(4_925_970_846_698_436, -11)
    send_max = XRPAmount(25_550_430_295)
    capped_out, adjusted = _limit_out_by_quality_function(
        [],
        amm=amm,
        remaining_out=remaining_out,
        limit_quality=limit_quality,
        mode="offer_crossing",
    )

    filled, spent, _, amm_in, amm_out, trace = _execute_one_iteration(
        [],
        target_out=capped_out,
        send_max=send_max,
        limit_quality=limit_quality,
        mode="offer_crossing",
        amm=amm,
    )

    assert adjusted is True
    assert filled == IOUAmount(5277592494635926, -13)
    assert spent == XRPAmount(274_187_712)
    assert amm_in == spent
    assert amm_out == filled
    assert len(trace) == 1
    assert trace[0]["src"] == "AMM"
    assert amm.spq_quality_int() >= limit_quality


def test_book_step_offer_crossing_partial_clob_fill_rechecks_actual_quality() -> None:
    limit_quality = Quality.from_rate(IOUAmount(5_259_282_634_883_098, -10))
    quoted_quality = Quality.from_rate(IOUAmount(5_259_282_633_848_744, -10))
    row = Segment(
        src="CLOB",
        quality=quoted_quality,
        out_max=IOUAmount(3_707_730_000_000_000, -12),
        in_at_out_max=XRPAmount(1_950_000_000),
    )

    step = BookStep.from_static([row], mode="offer_crossing", limit_quality=limit_quality)
    filled, spent = step.rev(None, IOUAmount(1_194_356_510_000_000, -12))

    assert filled == IOUAmount(0, 0)
    assert spent == XRPAmount(0)


def test_book_step_payment_amm_only_uses_rippled_max_offer_cap_in_reverse() -> None:
    amm = AMM(
        Decimal("702869.551393"),
        Decimal("1426782.263627074"),
        Decimal("0.00529"),
        x_is_xrp=True,
        y_is_xrp=False,
    )
    target_out = IOUAmount(9_999_999_999_999_999, 64)
    max_offer = amm.max_offer_segment(max_out_cap=target_out)

    assert max_offer is not None

    filled, spent, _, amm_in, amm_out, trace = _execute_one_iteration(
        [],
        target_out=target_out,
        send_max=None,
        limit_quality=None,
        mode="payment",
        amm=amm,
    )

    assert filled == max_offer.out_max
    assert spent == max_offer.in_at_out_max
    assert amm_in == spent
    assert amm_out == filled
    assert len(trace) == 1
    assert trace[0]["src"] == "AMM"
    assert trace[0]["quality"] == max_offer.quality


def test_book_step_payment_amm_only_huge_reverse_stops_after_single_synthetic_offer() -> None:
    amm = AMM(
        Decimal("702869.551393"),
        Decimal("1426782.263627074"),
        Decimal("0.00529"),
        x_is_xrp=True,
        y_is_xrp=False,
    )
    step = BookStep.from_static([], amm=amm, mode="payment")
    huge_target_out = IOUAmount(9_999_999_999_999_999, 79)
    max_offer = amm.max_offer_segment(max_out_cap=huge_target_out)

    assert max_offer is not None

    filled, spent, trace = step.rev(None, huge_target_out, return_trace=True)

    assert filled == max_offer.out_max
    assert spent == max_offer.in_at_out_max
    assert len(trace) == 1
    assert trace[0]["src"] == "AMM"
    assert trace[0]["out_take"] == max_offer.out_max
    assert trace[0]["in_take"] == max_offer.in_at_out_max


def test_book_step_consumes_tiny_clob_tail_without_losing_the_leg() -> None:
    tiny_out = IOUAmount(200, -15)
    rows = _normalise_rows(
        [
            Segment(
                src="CLOB",
                quality=Quality.from_amounts(tiny_out, XRPAmount(1)),
                out_max=tiny_out,
                in_at_out_max=XRPAmount(1),
            )
        ]
    )

    filled, spent, next_rows, _, _, trace = _execute_one_iteration(
        rows,
        target_out=tiny_out,
        send_max=None,
        limit_quality=None,
        mode="payment",
        amm=None,
    )

    assert filled == tiny_out
    assert spent == XRPAmount(1)
    assert next_rows == []
    assert len(trace) == 1
    assert trace[0]["src"] == "CLOB"


def test_book_step_exact_in_tiny_budget_that_cannot_buy_positive_output_returns_zero_without_invariant() -> None:
    rows = _normalise_rows(
        [
            Segment(
                src="CLOB",
                quality=Quality.from_amounts(XRPAmount(1), iou("1")),
                out_max=XRPAmount(1),
                in_at_out_max=iou("1"),
            )
        ]
    )

    filled, spent, next_rows, _, _, trace = _execute_one_iteration(
        rows,
        target_out=XRPAmount(1),
        send_max=iou("0.000000000000001"),
        limit_quality=None,
        mode="offer_crossing",
        amm=None,
    )

    assert filled == XRPAmount(0)
    assert spent == iou("0")
    assert next_rows == []
    assert trace == []


def test_book_step_exact_in_tiny_budget_with_input_transfer_rate_that_rounds_raw_limit_to_zero_returns_zero() -> None:
    row = BookOfferSegment(
        "fee-row",
        Quality.from_amounts(XRPAmount(1), iou("0.000001")),
        Segment(
            src="CLOB",
            quality=Quality.from_amounts(XRPAmount(1), iou("0.000001")),
            out_max=XRPAmount(1),
            in_at_out_max=iou("0.000001"),
        ),
        owner="rFee",
        in_transfer_rate=IOUAmount(1250000000000000, -15),
    )
    rows = _normalise_rows([row])

    filled, spent, next_rows, _, _, trace = _execute_one_iteration(
        rows,
        target_out=XRPAmount(1),
        send_max=iou("0.000000000000001"),
        limit_quality=None,
        mode="offer_crossing",
        amm=None,
    )

    assert filled == XRPAmount(0)
    assert spent == iou("0")
    assert next_rows == []
    assert trace == []


def test_book_step_exact_in_applies_output_cap_then_input_cap_on_same_row() -> None:
    rows = _normalise_rows(
        [
            Segment(
                src="CLOB",
                quality=Quality.from_amounts(XRPAmount(10), iou("5")),
                out_max=XRPAmount(10),
                in_at_out_max=iou("5"),
            )
        ]
    )

    filled, spent, next_rows, _, _, trace = _execute_one_iteration(
        rows,
        target_out=XRPAmount(6),
        send_max=iou("2"),
        limit_quality=None,
        mode="offer_crossing",
        amm=None,
    )

    assert filled == XRPAmount(4)
    assert spent == iou("2")
    assert len(next_rows) == 1
    assert next_rows[0].segment.out_max == XRPAmount(6)
    assert next_rows[0].segment.in_at_out_max == iou("3")
    assert len(trace) == 1
    assert trace[0]["src"] == "CLOB"
    assert trace[0]["out_take"] == XRPAmount(4)
    assert trace[0]["in_take"] == iou("2")


def test_book_step_residual_row_preserves_original_offer_quality() -> None:
    quoted_quality = Quality.from_rate(IOUAmount(5000000000000000, -10))
    rows = _normalise_rows(
        [
            BookOfferSegment(
                "offer-quality",
                quoted_quality,
                Segment(
                    src="CLOB",
                    quality=quoted_quality,
                    out_max=iou("10"),
                    in_at_out_max=XRPAmount(5_000_000),
                ),
            )
        ]
    )

    filled, spent, next_rows, _, _, trace = _execute_one_iteration(
        rows,
        target_out=iou("4"),
        send_max=None,
        limit_quality=None,
        mode="payment",
        amm=None,
    )

    assert filled == iou("4")
    assert spent == XRPAmount(2_000_000)
    assert len(trace) == 1
    assert len(next_rows) == 1
    assert next_rows[0].segment.quality == quoted_quality
    assert next_rows[0].quoted_quality == quoted_quality


def test_router_quote_view_preserves_offer_ids_from_book_offer_rows() -> None:
    segment = Segment(
        src="CLOB",
        quality=Quality.from_amounts(XRPAmount(10), XRPAmount(10)),
        out_max=XRPAmount(10),
        in_at_out_max=XRPAmount(10),
    )
    row = BookOfferSegment("offer-1", segment.quality, segment)
    view = RouterQuoteView(lambda: [row])

    quote = view.preview_out(XRPAmount(5))

    assert quote["summary"]["total_out"] == XRPAmount(5)
    assert quote["summary"]["total_in"] == XRPAmount(5)
    assert quote["slices"][0]["source_id"] == "offer-1"


def test_book_step_forward_stages_amm_deltas_in_sandbox() -> None:
    amm = AMM(Decimal("1000"), Decimal("1000"), Decimal("0"), x_is_xrp=True, y_is_xrp=True)
    q_top = Quality.from_amounts(XRPAmount(90), XRPAmount(100))
    step = BookStep.from_static(
        [
            Segment(
                src="CLOB",
                quality=q_top,
                out_max=XRPAmount(100),
                in_at_out_max=XRPAmount(112),
            )
        ],
        amm=amm,
    )
    step._cached_out = XRPAmount(500_000_000)
    sandbox = PaymentSandbox()

    total_out, total_in, trace = step.fwd(sandbox, XRPAmount(40_000_000), return_trace=True)

    assert total_in == XRPAmount(40_000_000)
    assert total_out == amm.swap_out_given_in_st(XRPAmount(40_000_000))
    assert len(trace) == 1
    assert trace[0]["src"] == "AMM"
    assert sandbox.staged == [(XRPAmount(40_000_000), total_out)]


def test_amm_aggregate_execution_exact_in_preserves_sequential_consumption_and_merges_trace() -> None:
    amm = AMM(
        Decimal("1306912.253839654"),
        Decimal("678664.887362"),
        Decimal("0.00519"),
        x_is_xrp=False,
        y_is_xrp=True,
    )
    agg = _AMMAggregateExecution.from_amm(amm)
    in1 = iou("164.7067178238257")
    in2 = iou("315.3139714996734")
    raw_out1 = amm.swap_out_given_in_st(in1)
    raw_shadow = amm.clone()
    raw_shadow.apply_fill_st(in1, raw_out1)
    raw_out2 = raw_shadow.swap_out_given_in_st(in2)

    out1, spent1, shadow1 = agg.apply_exact_in(raw_in=in1, raw_out=raw_out1)
    out2, spent2, shadow2 = agg.apply_exact_in(raw_in=in2, raw_out=raw_out2)

    assert spent1 == in1
    assert spent2 == in2
    assert out1 == raw_out1
    assert out2 == raw_out2
    assert shadow1.spq_quality_int() > shadow2.spq_quality_int()
    sequential_shadow = amm.clone()
    sequential_shadow.apply_fill_st(in1, raw_out1)
    sequential_shadow.apply_fill_st(in2, raw_out2)
    assert shadow2.spq_quality_int() == sequential_shadow.spq_quality_int()
    assert agg.merged_trace_row()["out_take"] == raw_out1 + raw_out2
    assert agg.merged_trace_row()["in_take"] == in1 + in2


def test_amm_aggregate_execution_exact_out_preserves_sequential_consumption_and_merges_trace() -> None:
    amm = AMM(
        Decimal("1306912.253839654"),
        Decimal("678664.887362"),
        Decimal("0.00519"),
        x_is_xrp=False,
        y_is_xrp=True,
    )
    agg = _AMMAggregateExecution.from_amm(amm)
    out1 = XRPAmount(85_075_784)
    out2 = XRPAmount(162_809_192)
    raw_in1 = amm.swap_in_given_out_st(out1)
    raw_shadow = amm.clone()
    raw_shadow.apply_fill_st(raw_in1, out1)
    raw_in2 = raw_shadow.swap_in_given_out_st(out2)

    filled1, in1, shadow1 = agg.apply_exact_out(raw_in=raw_in1, raw_out=out1)
    filled2, in2, shadow2 = agg.apply_exact_out(raw_in=raw_in2, raw_out=out2)

    assert filled1 == out1
    assert filled2 == out2
    assert in1 == raw_in1
    assert in2 == raw_in2
    sequential_shadow = amm.clone()
    sequential_shadow.apply_fill_st(raw_in1, out1)
    sequential_shadow.apply_fill_st(raw_in2, out2)
    assert shadow1.spq_quality_int() > shadow2.spq_quality_int()
    assert shadow2.spq_quality_int() == sequential_shadow.spq_quality_int()
    assert agg.merged_trace_row()["out_take"] == XRPAmount(out1.drops + out2.drops)
    assert agg.merged_trace_row()["in_take"] == raw_in1 + raw_in2


def test_book_step_offer_crossing_self_waives_input_transfer_rate_on_direct_book_step() -> None:
    row = BookOfferSegment(
        "self-offer",
        Quality.from_amounts(XRPAmount(1_000_000), iou("2")),
        Segment(
            src="CLOB",
            quality=Quality.from_amounts(XRPAmount(1_000_000), iou("2")),
            out_max=XRPAmount(1_000_000),
            in_at_out_max=iou("2"),
        ),
        owner="rSelf",
        in_transfer_rate=IOUAmount(1250000000000000, -15),
    )

    self_step = BookStep.from_static([row], mode="offer_crossing", source_account="rSelf")
    other_step = BookStep.from_static([row], mode="offer_crossing", source_account="rOther")

    self_out, self_in = self_step.rev(None, XRPAmount(1_000_000))
    other_out, other_in = other_step.rev(None, XRPAmount(1_000_000))

    assert self_out == XRPAmount(1_000_000)
    assert self_in == iou("2")
    assert other_out == XRPAmount(1_000_000)
    assert other_in == iou("2.5")


def test_book_step_offer_crossing_deletes_self_cross_before_moving_to_next_quality_bucket_exact_out() -> None:
    better_quality = Quality.from_amounts(iou("2"), XRPAmount(1_000_000))
    limit_quality = Quality.from_amounts(iou("2"), XRPAmount(1_100_000))
    external_row = BookOfferSegment(
        "external-row",
        limit_quality,
        Segment(
            src="CLOB",
            quality=limit_quality,
            out_max=iou("2"),
            in_at_out_max=XRPAmount(1_100_000),
        ),
        owner="rExternal",
    )
    self_row = BookOfferSegment(
        "self-row",
        better_quality,
        Segment(
            src="CLOB",
            quality=better_quality,
            out_max=iou("2"),
            in_at_out_max=XRPAmount(1_000_000),
        ),
        owner="rSelf",
    )
    rows = _normalise_rows([self_row, external_row])

    filled, spent, next_rows, _, _, trace = _execute_one_iteration(
        rows,
        target_out=iou("2"),
        send_max=None,
        limit_quality=limit_quality,
        mode="offer_crossing",
        amm=None,
        source_account="rSelf",
    )

    assert filled == iou("2")
    assert spent == XRPAmount(1_100_000)
    assert next_rows == []
    fill_trace = [slice_ for slice_ in trace if slice_["src"] != "CLOB_DELETE"]
    delete_trace = [slice_ for slice_ in trace if slice_["src"] == "CLOB_DELETE"]
    assert [slice_["source_id"] for slice_ in fill_trace] == ["external-row"]
    assert [slice_["source_id"] for slice_ in delete_trace] == ["self-row"]


def test_book_step_offer_crossing_deletes_self_cross_before_moving_to_next_quality_bucket_exact_in() -> None:
    better_quality = Quality.from_amounts(iou("2"), XRPAmount(1_000_000))
    limit_quality = Quality.from_amounts(iou("2"), XRPAmount(1_100_000))
    external_row = BookOfferSegment(
        "external-row",
        limit_quality,
        Segment(
            src="CLOB",
            quality=limit_quality,
            out_max=iou("2"),
            in_at_out_max=XRPAmount(1_100_000),
        ),
        owner="rExternal",
    )
    self_row = BookOfferSegment(
        "self-row",
        better_quality,
        Segment(
            src="CLOB",
            quality=better_quality,
            out_max=iou("2"),
            in_at_out_max=XRPAmount(1_000_000),
        ),
        owner="rSelf",
    )
    rows = _normalise_rows([self_row, external_row])

    filled, spent, next_rows, _, _, trace = _execute_one_iteration(
        rows,
        target_out=iou("2"),
        send_max=XRPAmount(1_100_000),
        limit_quality=limit_quality,
        mode="offer_crossing",
        amm=None,
        source_account="rSelf",
    )

    assert filled == iou("2")
    assert spent == XRPAmount(1_100_000)
    assert next_rows == []
    fill_trace = [slice_ for slice_ in trace if slice_["src"] != "CLOB_DELETE"]
    delete_trace = [slice_ for slice_ in trace if slice_["src"] == "CLOB_DELETE"]
    assert [slice_["source_id"] for slice_ in fill_trace] == ["external-row"]
    assert [slice_["source_id"] for slice_ in delete_trace] == ["self-row"]


def test_book_step_owner_funds_cap_applies_after_output_transfer_rate() -> None:
    row = BookOfferSegment(
        "funded-offer",
        Quality.from_amounts(iou("10"), XRPAmount(10_000_000)),
        Segment(
            src="CLOB",
            quality=Quality.from_amounts(iou("10"), XRPAmount(10_000_000)),
            out_max=iou("10"),
            in_at_out_max=XRPAmount(10_000_000),
        ),
        owner="rMaker",
        owner_funds=iou("12"),
        out_transfer_rate=IOUAmount(1250000000000000, -15),
    )

    step = BookStep.from_static([row], mode="payment")

    total_out, total_in = step.rev(None, iou("10"))

    assert total_out == iou("9.6")
    assert total_in == XRPAmount(9_600_000)


def test_book_step_consumes_same_quoted_quality_bucket_even_when_step_qualities_differ() -> None:
    base_quality = Quality.from_amounts(XRPAmount(1_000_000), iou("2"))
    rows = _normalise_rows(
        [
            BookOfferSegment(
                "fee-row",
                base_quality,
                Segment(
                    src="CLOB",
                    quality=base_quality,
                    out_max=XRPAmount(1_000_000),
                    in_at_out_max=iou("2"),
                ),
                owner="rFee",
                in_transfer_rate=IOUAmount(1250000000000000, -15),
            ),
            BookOfferSegment(
                "plain-row",
                base_quality,
                Segment(
                    src="CLOB",
                    quality=base_quality,
                    out_max=XRPAmount(1_000_000),
                    in_at_out_max=iou("2"),
                ),
                owner="rPlain",
            ),
        ]
    )

    filled, spent, next_rows, _, _, trace = _execute_one_iteration(
        rows,
        target_out=XRPAmount(2_000_000),
        send_max=None,
        limit_quality=None,
        mode="payment",
        amm=None,
    )

    assert filled == XRPAmount(2_000_000)
    assert spent == iou("4.5")
    assert next_rows == []
    assert [slice_["source_id"] for slice_ in trace] == ["fee-row", "plain-row"]


def test_limit_out_by_quality_function_caps_single_path_amm_tip_like_rippled_number_semantics() -> None:
    amm = AMM(
        Decimal("684735.71968"),
        Decimal("1307456.560406163"),
        Decimal("0.00519"),
        x_is_xrp=True,
        y_is_xrp=False,
    )
    remaining_out = IOUAmount(1907176970000000, -12)
    limit_quality = Quality(6490451417912776046)
    tip_row = BookOfferSegment(
        "lob-tip",
        Quality(6490449198745816255),
        Segment(
            src="CLOB",
            quality=Quality(6490449198745816255),
            out_max=IOUAmount(1905268470000000, -12),
            in_at_out_max=XRPAmount(1_003_263_950),
        ),
    )

    capped_out, adjusted = _limit_out_by_quality_function(
        _normalise_rows([tip_row]),
        amm=amm,
        remaining_out=remaining_out,
        limit_quality=limit_quality,
        mode="offer_crossing",
    )

    assert adjusted is True
    assert capped_out == IOUAmount(8620586963544793, -13)


def test_limit_out_by_quality_function_floors_xrp_cap_on_real_boundary_cases() -> None:
    cases = [
        (
            Decimal("1313060.488435156"),
            Decimal("682291.248212"),
            Quality.from_amounts(XRPAmount(717_523_140), IOUAmount(1388536430065200, -12)),
            XRPAmount(230_238_063),
        ),
        (
            Decimal("1312906.427976155"),
            Decimal("682370.894833"),
            Quality.from_amounts(XRPAmount(796_699_730), IOUAmount(1541056287739000, -12)),
            XRPAmount(79_646_621),
        ),
        (
            Decimal("1308391.097719879"),
            Decimal("685664.851604"),
            Quality.from_amounts(XRPAmount(992_067_940), IOUAmount(1903371630000000, -12)),
            XRPAmount(152_535_972),
        ),
    ]

    for x_reserve, y_reserve, limit_quality, expected_cap in cases:
        amm = AMM(x_reserve, y_reserve, Decimal("0.00519"), x_is_xrp=False, y_is_xrp=True)
        capped_out, adjusted = _limit_out_by_quality_function(
            [],
            amm=amm,
            remaining_out=XRPAmount(10_000_000_000_000),
            limit_quality=limit_quality,
            mode="offer_crossing",
        )
        assert adjusted is True
        assert capped_out == expected_cap


def test_limit_out_by_quality_function_leaves_constant_clob_tip_unchanged() -> None:
    amm = AMM(
        Decimal("1000"),
        Decimal("1000"),
        Decimal("0.00519"),
        x_is_xrp=True,
        y_is_xrp=False,
    )
    remaining_out = iou("100")
    better_clob = BookOfferSegment(
        "better-clob",
        Quality.from_amounts(iou("100"), XRPAmount(90_000_000)),
        Segment(
            src="CLOB",
            quality=Quality.from_amounts(iou("100"), XRPAmount(90_000_000)),
            out_max=iou("100"),
            in_at_out_max=XRPAmount(90_000_000),
        ),
    )

    capped_out, adjusted = _limit_out_by_quality_function(
        _normalise_rows([better_clob]),
        amm=amm,
        remaining_out=remaining_out,
        limit_quality=Quality.from_amounts(iou("100"), XRPAmount(95_000_000)),
        mode="offer_crossing",
    )

    assert adjusted is False
    assert capped_out == remaining_out
