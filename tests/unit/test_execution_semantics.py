from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pytest

from xrpl_router.amm import AMM
from xrpl_router.book_offers import build_clob_segments_from_offers
from xrpl_router.core import IOUAmount, Quality, XRPAmount
from xrpl_router.core.datatypes import Segment
from xrpl_router.execution_semantics import execute_single_path_intent
from xrpl_router.tx_intent import TxIntent


def _one_to_one_clob_segment(*, out_value: str = "10", in_drops: int = 10_000_000) -> Segment:
    out_amt = IOUAmount.from_decimal(Decimal(out_value))
    in_amt = XRPAmount(in_drops)
    return Segment(
        src="CLOB",
        quality=Quality.from_amounts(out_amt, in_amt),
        out_max=out_amt,
        in_at_out_max=in_amt,
        raw_quality=Quality.from_amounts(out_amt, in_amt),
        source_id="offer-1",
    )


def _one_to_one_clob_segment_iou_to_xrp(*, out_drops: int = 10_000_000, in_value: str = "10") -> Segment:
    out_amt = XRPAmount(out_drops)
    in_amt = IOUAmount.from_decimal(Decimal(in_value))
    return Segment(
        src="CLOB",
        quality=Quality.from_amounts(out_amt, in_amt),
        out_max=out_amt,
        in_at_out_max=in_amt,
        raw_quality=Quality.from_amounts(out_amt, in_amt),
        source_id="offer-1-iou-to-xrp",
    )


def _executed_slices(quote: dict) -> list[dict]:
    return [
        s
        for s in (quote.get("slices") or [])
        if s.get("out_take") is not None and s.get("in_take") is not None
    ]


def test_payment_partial_without_flag_fails() -> None:
    seg = _one_to_one_clob_segment(out_value="10", in_drops=10_000_000)
    intent = TxIntent(
        transaction_type="PAYMENT",
        flags=0,
        in_cur="XRP",
        out_cur="USD",
        payment_sendmax_amt=XRPAmount(5_000_000),
        payment_delivermax_amt=IOUAmount.from_decimal(Decimal("10")),
        payment_partial_allowed=False,
    )

    with pytest.raises(RuntimeError, match="tfPartialPayment"):
        execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="XRP", out_cur="USD")


def test_payment_no_ripple_direct_without_paths_fails() -> None:
    seg = _one_to_one_clob_segment(out_value="10", in_drops=10_000_000)
    intent = TxIntent(
        transaction_type="PAYMENT",
        flags=0,
        in_cur="XRP",
        out_cur="USD",
        payment_sendmax_amt=XRPAmount(10_000_000),
        payment_delivermax_amt=IOUAmount.from_decimal(Decimal("10")),
        payment_no_ripple_direct=True,
        payment_paths_present=False,
    )

    with pytest.raises(RuntimeError, match="tfNoRippleDirect"):
        execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="XRP", out_cur="USD")


def test_payment_no_ripple_direct_with_paths_can_execute() -> None:
    seg = _one_to_one_clob_segment(out_value="10", in_drops=10_000_000)
    intent = TxIntent(
        transaction_type="PAYMENT",
        flags=0,
        in_cur="XRP",
        out_cur="USD",
        payment_sendmax_amt=XRPAmount(10_000_000),
        payment_delivermax_amt=IOUAmount.from_decimal(Decimal("10")),
        payment_no_ripple_direct=True,
        payment_paths_present=True,
        payment_paths_has_external_asset=False,
    )

    quote, _ = execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="XRP", out_cur="USD")
    summary = quote.get("summary") or {}
    total_out = summary.get("total_out")
    assert total_out is not None
    assert Decimal(total_out.to_decimal()) == Decimal("10")


def test_offercreate_fok_partial_fails() -> None:
    seg = _one_to_one_clob_segment_iou_to_xrp(out_drops=5_000_000, in_value="5")
    limit_q = Quality.from_amounts(XRPAmount(1_000_000), IOUAmount.from_decimal(Decimal("1")))
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=IOUAmount.from_decimal(Decimal("10")),
        offer_taker_pays_amt=XRPAmount(10_000_000),
        offer_limit_quality=limit_q,
        offer_is_sell=False,
        offer_is_fok=True,
    )

    with pytest.raises(RuntimeError, match="tfFillOrKill"):
        execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="USD", out_cur="XRP")


def test_offercreate_fok_buy_fails_when_budget_exhausted_before_target() -> None:
    # Non-sell FOK must acquire the full TakerPays amount; exhausting TakerGets
    # budget first is still a partial fill.
    seg = Segment(
        src="CLOB",
        quality=Quality.from_amounts(XRPAmount(5_000_000), IOUAmount.from_decimal(Decimal("10"))),
        out_max=XRPAmount(5_000_000),
        in_at_out_max=IOUAmount.from_decimal(Decimal("10")),
        raw_quality=Quality.from_amounts(XRPAmount(5_000_000), IOUAmount.from_decimal(Decimal("10"))),
        source_id="offer-budget-exhausted",
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=IOUAmount.from_decimal(Decimal("10")),
        offer_taker_pays_amt=XRPAmount(10_000_000),
        offer_limit_quality=Quality.from_amounts(XRPAmount(100_000), IOUAmount.from_decimal(Decimal("10"))),
        offer_is_sell=False,
        offer_is_fok=True,
    )

    with pytest.raises(RuntimeError, match="tfFillOrKill"):
        execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="USD", out_cur="XRP")


def test_offercreate_buy_budget_exhausted_before_target_posts_remainder() -> None:
    seg = Segment(
        src="CLOB",
        quality=Quality.from_amounts(XRPAmount(5_000_000), IOUAmount.from_decimal(Decimal("10"))),
        out_max=XRPAmount(5_000_000),
        in_at_out_max=IOUAmount.from_decimal(Decimal("10")),
        raw_quality=Quality.from_amounts(XRPAmount(5_000_000), IOUAmount.from_decimal(Decimal("10"))),
        source_id="offer-budget-exhausted-post",
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=IOUAmount.from_decimal(Decimal("10")),
        offer_taker_pays_amt=XRPAmount(10_000_000),
        offer_limit_quality=Quality.from_amounts(XRPAmount(100_000), IOUAmount.from_decimal(Decimal("10"))),
        offer_is_sell=False,
        offer_is_fok=False,
        offer_is_ioc=False,
    )

    quote, meta = execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="USD", out_cur="XRP")
    summary = quote.get("summary") or {}
    assert bool(summary.get("is_partial")) is True
    assert meta.get("post_to_book_remainder") is True


def test_offercreate_partial_without_ioc_or_fok_posts_remainder() -> None:
    seg = _one_to_one_clob_segment_iou_to_xrp(out_drops=5_000_000, in_value="5")
    limit_q = Quality.from_amounts(XRPAmount(1_000_000), IOUAmount.from_decimal(Decimal("1")))
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=IOUAmount.from_decimal(Decimal("10")),
        offer_taker_pays_amt=XRPAmount(10_000_000),
        offer_limit_quality=limit_q,
        offer_is_sell=False,
        offer_is_ioc=False,
        offer_is_fok=False,
    )

    quote, meta = execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="USD", out_cur="XRP")
    summary = quote.get("summary") or {}

    assert bool(summary.get("is_partial")) is True
    assert meta.get("post_to_book_remainder") is True


def test_offercreate_sell_path_no_amount_domain_type_error() -> None:
    seg = _one_to_one_clob_segment(out_value="10", in_drops=10_000_000)
    # Quality floor equals segment quality (1e-6), so crossing is allowed.
    limit_q = Quality.from_amounts(IOUAmount.from_decimal(Decimal("1")), XRPAmount(1_000_000))
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="XRP",
        out_cur="USD",
        offer_taker_gets_amt=XRPAmount(10_000_000),
        offer_taker_pays_amt=IOUAmount.from_decimal(Decimal("10")),
        offer_limit_quality=limit_q,
        offer_is_sell=True,
    )

    quote, meta = execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="XRP", out_cur="USD")
    summary = quote.get("summary") or {}

    assert meta.get("mode") is not None
    assert "+sell" in str(meta.get("mode"))
    assert summary.get("total_in") is not None
    assert summary.get("total_out") is not None


def test_offercreate_sell_fok_treats_sub_micro_input_residual_as_complete() -> None:
    seg = Segment(
        src="CLOB",
        quality=Quality.from_amounts(XRPAmount(2_406_252), IOUAmount.from_decimal(Decimal("4.999999156421"))),
        out_max=XRPAmount(2_406_252),
        in_at_out_max=IOUAmount.from_decimal(Decimal("4.999999156421")),
        raw_quality=Quality.from_amounts(XRPAmount(2_406_252), IOUAmount.from_decimal(Decimal("4.999999156421"))),
        source_id="offer-sell-near-cap",
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=IOUAmount.from_decimal(Decimal("5")),
        offer_taker_pays_amt=XRPAmount(2_274_658),
        offer_limit_quality=Quality.from_amounts(XRPAmount(2_274_658), IOUAmount.from_decimal(Decimal("5"))),
        offer_is_sell=True,
        offer_is_fok=True,
    )

    quote, _ = execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="USD", out_cur="XRP")
    summary = quote.get("summary") or {}
    assert bool(summary.get("is_partial")) is False


def test_offercreate_sell_fok_treats_sub_quantum_residual_as_complete() -> None:
    # Residual input is > 2e-6 IOU, but still below the minimum input needed
    # to produce 1 drop XRP at available quality; should be treated as exhausted.
    seg = Segment(
        src="CLOB",
        quality=Quality.from_amounts(XRPAmount(19_231_403), IOUAmount.from_decimal(Decimal("40.018857279536"))),
        out_max=XRPAmount(19_231_403),
        in_at_out_max=IOUAmount.from_decimal(Decimal("40.018857279536")),
        raw_quality=Quality.from_amounts(XRPAmount(19_231_403), IOUAmount.from_decimal(Decimal("40.018857279536"))),
        source_id="offer-sell-sub-quantum-residual",
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=IOUAmount.from_decimal(Decimal("40.0188593")),
        offer_taker_pays_amt=XRPAmount(19_067_569),
        offer_limit_quality=Quality.from_amounts(XRPAmount(19_067_569), IOUAmount.from_decimal(Decimal("40.0188593"))),
        offer_is_sell=True,
        offer_is_fok=True,
    )

    quote, _ = execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="USD", out_cur="XRP")
    summary = quote.get("summary") or {}
    assert bool(summary.get("is_partial")) is False


def test_offercreate_buy_side_still_crosses_when_limit_floor_is_tight() -> None:
    seg = _one_to_one_clob_segment_iou_to_xrp(out_drops=5_000_000, in_value="5")
    # Set a tighter floor than available segment quality; buy-side execution
    # should still run under SendMax/DeliverMax constraints.
    tight_limit = Quality.from_amounts(XRPAmount(1_000_000), IOUAmount.from_decimal(Decimal("3")))
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=IOUAmount.from_decimal(Decimal("10")),
        offer_taker_pays_amt=XRPAmount(10_000_000),
        offer_limit_quality=tight_limit,
        offer_is_sell=False,
    )

    quote, _ = execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="USD", out_cur="XRP")
    summary = quote.get("summary") or {}
    assert summary.get("total_out") is not None
    assert bool(summary.get("is_partial")) is True


def test_payment_stops_after_sendmax_triggered_forward_rerun() -> None:
    seg = _one_to_one_clob_segment(out_value="10", in_drops=10_000_000)
    intent = TxIntent(
        transaction_type="PAYMENT",
        flags=0x00020000,  # tfPartialPayment
        in_cur="XRP",
        out_cur="USD",
        payment_sendmax_amt=XRPAmount(9_000_001),
        payment_delivermax_amt=IOUAmount.from_decimal(Decimal("10")),
        payment_delivermin_amt=IOUAmount.from_decimal(Decimal("9")),
        payment_partial_allowed=True,
    )

    reverse_quote = {
        "slices": [],
        "summary": {
            "total_out": IOUAmount.from_decimal(Decimal("10")),
            "total_in": XRPAmount(10_000_000),
            "requested_out": IOUAmount.from_decimal(Decimal("10")),
        },
    }

    fwd_trace = [
        {
            "src": "CLOB",
            "source_id": "offer-1",
            "take_out": IOUAmount.from_decimal(Decimal("9")),
            "take_in": XRPAmount(9_000_000),
            "quality": Quality.from_amounts(
                IOUAmount.from_decimal(Decimal("9")),
                XRPAmount(9_000_000),
            ),
        }
    ]

    class _MockView:
        def preview_out(self, requested_out):
            return reverse_quote

    class _MockStep:
        def __init__(self):
            self._cached_out = None

        def fwd(self, sb, remaining_in, return_trace=False):
            assert remaining_in == XRPAmount(9_000_001)
            return IOUAmount.from_decimal(Decimal("9")), XRPAmount(9_000_000), fwd_trace

    with patch("xrpl_router.execution_semantics.RouterQuoteView", return_value=_MockView()) as mock_view, patch(
        "xrpl_router.execution_semantics.BookStep.from_static",
        return_value=_MockStep(),
    ) as mock_step:
        quote, _ = execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="XRP", out_cur="USD")

    summary = quote.get("summary") or {}
    total_out = summary.get("total_out")
    total_in = summary.get("total_in")
    slices = _executed_slices(quote)

    assert mock_view.call_count == 1
    assert mock_step.call_count == 1
    assert total_out is not None and Decimal(total_out.to_decimal()) == Decimal("9")
    assert total_in == XRPAmount(9_000_000)
    assert len(slices) == 1
    assert slices[0]["src"] == "CLOB"


def test_offercreate_buy_stops_after_sendmax_triggered_forward_rerun() -> None:
    seg = _one_to_one_clob_segment_iou_to_xrp(out_drops=10_000_000, in_value="10")
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=IOUAmount.from_decimal(Decimal("9.000001")),
        offer_taker_pays_amt=XRPAmount(10_000_000),
        offer_limit_quality=Quality.from_amounts(XRPAmount(10_000_000), IOUAmount.from_decimal(Decimal("10"))),
        offer_is_sell=False,
        offer_is_ioc=True,
    )

    reverse_quote = {
        "slices": [],
        "summary": {
            "total_out": XRPAmount(10_000_000),
            "total_in": IOUAmount.from_decimal(Decimal("10")),
            "requested_out": XRPAmount(10_000_000),
        },
    }

    fwd_trace = [
        {
            "src": "CLOB",
            "source_id": "offer-1",
            "take_out": XRPAmount(9_000_000),
            "take_in": IOUAmount.from_decimal(Decimal("9")),
            "quality": Quality.from_amounts(
                XRPAmount(9_000_000),
                IOUAmount.from_decimal(Decimal("9")),
            ),
        }
    ]

    class _MockView:
        def preview_out(self, requested_out):
            return reverse_quote

    class _MockStep:
        def __init__(self):
            self._cached_out = None

        def fwd(self, sb, remaining_in, return_trace=False):
            assert remaining_in == IOUAmount.from_decimal(Decimal("9.000001"))
            return XRPAmount(9_000_000), IOUAmount.from_decimal(Decimal("9")), fwd_trace

    with patch("xrpl_router.execution_semantics.RouterQuoteView", return_value=_MockView()) as mock_view, patch(
        "xrpl_router.execution_semantics.BookStep.from_static",
        return_value=_MockStep(),
    ) as mock_step:
        quote, meta = execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="USD", out_cur="XRP")

    summary = quote.get("summary") or {}
    total_out = summary.get("total_out")
    total_in = summary.get("total_in")
    slices = _executed_slices(quote)

    assert mock_view.call_count == 1
    assert mock_step.call_count == 1
    assert total_out == XRPAmount(9_000_000)
    assert total_in is not None and Decimal(total_in.to_decimal()) == Decimal("9")
    assert len(slices) == 1
    assert slices[0]["src"] == "CLOB"
    assert meta.get("post_to_book_remainder") is False


def test_offercreate_passive_does_not_cross_at_equal_quality() -> None:
    seg = _one_to_one_clob_segment_iou_to_xrp(out_drops=5_000_000, in_value="5")
    equal_limit = Quality.from_amounts(XRPAmount(5_000_000), IOUAmount.from_decimal(Decimal("5")))
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=IOUAmount.from_decimal(Decimal("5")),
        offer_taker_pays_amt=XRPAmount(5_000_000),
        offer_limit_quality=equal_limit,
        offer_is_sell=False,
        offer_is_passive=True,
    )

    quote, meta = execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="USD", out_cur="XRP")
    summary = quote.get("summary") or {}
    total_out = summary.get("total_out")
    total_in = summary.get("total_in")
    assert total_out is not None
    assert total_in is not None
    assert total_out.is_zero()
    assert total_in.is_zero()
    assert bool(summary.get("is_partial")) is True
    assert meta.get("post_to_book_remainder") is True


def test_offercreate_empty_ioc_is_success_with_no_post() -> None:
    seg = _one_to_one_clob_segment_iou_to_xrp(out_drops=5_000_000, in_value="5")
    equal_limit = Quality.from_amounts(XRPAmount(5_000_000), IOUAmount.from_decimal(Decimal("5")))
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=IOUAmount.from_decimal(Decimal("5")),
        offer_taker_pays_amt=XRPAmount(5_000_000),
        offer_limit_quality=equal_limit,
        offer_is_sell=False,
        offer_is_passive=True,
        offer_is_ioc=True,
    )

    quote, meta = execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="USD", out_cur="XRP")
    summary = quote.get("summary") or {}
    total_out = summary.get("total_out")
    total_in = summary.get("total_in")
    assert total_out is not None
    assert total_in is not None
    assert total_out.is_zero()
    assert total_in.is_zero()
    assert bool(summary.get("is_partial")) is True
    assert meta.get("post_to_book_remainder") is False


def test_offercreate_non_passive_crosses_at_equal_quality() -> None:
    seg = _one_to_one_clob_segment_iou_to_xrp(out_drops=5_000_000, in_value="5")
    equal_limit = Quality.from_amounts(XRPAmount(5_000_000), IOUAmount.from_decimal(Decimal("5")))
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=IOUAmount.from_decimal(Decimal("5")),
        offer_taker_pays_amt=XRPAmount(5_000_000),
        offer_limit_quality=equal_limit,
        offer_is_sell=False,
        offer_is_passive=False,
    )

    quote, _ = execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="USD", out_cur="XRP")
    summary = quote.get("summary") or {}
    total_out = summary.get("total_out")
    assert total_out is not None
    assert Decimal(total_out.to_decimal()) == Decimal("5")


def test_offercreate_buy_side_sendmax_grid_mismatch_still_crosses() -> None:
    # Segment IN grid is exponent -14, while sendmax stays on -15.
    # Budget conversion must floor-divide, not collapse to zero.
    out_amt = XRPAmount(9_286_859)
    in_amt = IOUAmount.from_components(1_997_560_601_682_221, -14)
    seg = Segment(
        src="CLOB",
        quality=Quality.from_amounts(out_amt, in_amt),
        out_max=out_amt,
        in_at_out_max=in_amt,
        raw_quality=Quality.from_amounts(out_amt, in_amt),
        source_id="offer-grid-mismatch",
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=IOUAmount.from_components(19_014_882_408_157_803, -15),
        offer_taker_pays_amt=XRPAmount(9_286_859),
        offer_limit_quality=Quality.from_amounts(out_amt, in_amt),
        offer_is_sell=False,
    )

    quote, _ = execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="USD", out_cur="XRP")
    summary = quote.get("summary") or {}
    total_out = summary.get("total_out")
    assert total_out is not None
    assert total_out.is_zero() is False


def test_offercreate_buy_side_respects_source_funds_cap() -> None:
    seg = Segment(
        src="CLOB",
        quality=Quality.from_amounts(XRPAmount(10_000_000), IOUAmount.from_decimal(Decimal("10"))),
        out_max=XRPAmount(10_000_000),
        in_at_out_max=IOUAmount.from_decimal(Decimal("10")),
        raw_quality=Quality.from_amounts(XRPAmount(10_000_000), IOUAmount.from_decimal(Decimal("10"))),
        source_id="offer-funded-cap",
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=IOUAmount.from_decimal(Decimal("10")),
        offer_taker_pays_amt=XRPAmount(10_000_000),
        offer_limit_quality=Quality.from_amounts(XRPAmount(10_000_000), IOUAmount.from_decimal(Decimal("10"))),
        offer_is_sell=False,
        source_funds_cap_amt=IOUAmount.from_decimal(Decimal("2")),
    )

    quote, _ = execute_single_path_intent(intent=intent, amm=None, segs=[seg], in_cur="USD", out_cur="XRP")
    summary = quote.get("summary") or {}
    total_in = summary.get("total_in")
    total_out = summary.get("total_out")
    assert total_in is not None and total_out is not None
    assert Decimal(total_in.to_decimal()) == Decimal("2")
    assert Decimal(total_out.to_decimal()) == Decimal("2")


def test_offercreate_ioc_hybrid_relaxes_amm_anchor_to_slice_quality_floor() -> None:
    seg = Segment(
        src="CLOB",
        quality=Quality.from_amounts(XRPAmount(4_675_089), IOUAmount.from_decimal(Decimal("9.599902554024"))),
        out_max=XRPAmount(4_675_089),
        in_at_out_max=IOUAmount.from_decimal(Decimal("9.599902554024")),
        raw_quality=Quality.from_amounts(XRPAmount(4_675_089), IOUAmount.from_decimal(Decimal("9.599902554024"))),
        source_id="offer-hybrid-anchor",
    )
    amm = AMM(
        Decimal("1425836.824179917"),
        Decimal("696794.819156"),
        Decimal("0.9646301091686775") / Decimal("294.9939171769656"),
        x_is_xrp=False,
        y_is_xrp=True,
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=IOUAmount.from_decimal(Decimal("45787.5016942301")),
        offer_taker_pays_amt=XRPAmount(22_297_406_704),
        offer_limit_quality=Quality.from_amounts(
            XRPAmount(22_297_406_704),
            IOUAmount.from_decimal(Decimal("45787.5016942301")),
        ),
        offer_is_sell=False,
        offer_is_ioc=True,
        source_funds_cap_amt=IOUAmount.from_decimal(Decimal("45785.85168513401")),
    )

    quote, meta = execute_single_path_intent(intent=intent, amm=amm, segs=[seg], in_cur="USD", out_cur="XRP")
    summary = quote.get("summary") or {}
    total_in = summary.get("total_in")
    total_out = summary.get("total_out")
    assert total_in == IOUAmount.from_components(3_044_816_407_890_024, -13)
    assert total_out == XRPAmount(148_280_558)
    eff_slices = _executed_slices(quote)
    assert [s.get("src") for s in eff_slices] == ["AMM", "CLOB"]
    assert summary.get("is_partial") is True
    assert meta.get("post_to_book_remainder") is False


def test_offercreate_ioc_hybrid_keeps_tier_anchor_when_clob_competes() -> None:
    seg_top = Segment(
        src="CLOB",
        quality=Quality.from_amounts(
            IOUAmount.from_decimal(Decimal("0.3312607677695808")),
            XRPAmount(161_973),
        ),
        out_max=IOUAmount.from_decimal(Decimal("0.3312607677695808")),
        in_at_out_max=XRPAmount(161_973),
        raw_quality=Quality.from_amounts(
            IOUAmount.from_decimal(Decimal("0.3312607677695808")),
            XRPAmount(161_973),
        ),
        source_id="offer-7AEF",
    )
    seg_rest = Segment(
        src="CLOB",
        quality=Quality.from_amounts(
            IOUAmount.from_decimal(Decimal("391.750556919236")),
            XRPAmount(191_653_187),
        ),
        out_max=IOUAmount.from_decimal(Decimal("391.750556919236")),
        in_at_out_max=XRPAmount(191_653_187),
        raw_quality=Quality.from_amounts(
            IOUAmount.from_decimal(Decimal("391.750556919236")),
            XRPAmount(191_653_187),
        ),
        source_id="offer-63A7",
    )
    amm = AMM(
        Decimal("695635.480496"),
        Decimal("1428187.908423381"),
        Decimal("0.00327"),
        x_is_xrp=True,
        y_is_xrp=False,
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="XRP",
        out_cur="USD",
        offer_taker_gets_amt=XRPAmount(600_000_000),
        offer_taker_pays_amt=IOUAmount.from_decimal(Decimal("1226.313180608006")),
        offer_limit_quality=Quality.from_amounts(
            IOUAmount.from_decimal(Decimal("1226.313180608006")),
            XRPAmount(600_000_000),
        ),
        offer_is_sell=False,
        offer_is_ioc=True,
    )

    quote, meta = execute_single_path_intent(
        intent=intent,
        amm=amm,
        segs=[seg_top, seg_rest],
        in_cur="XRP",
        out_cur="USD",
    )
    summary = quote.get("summary") or {}
    total_in = summary.get("total_in")
    total_out = summary.get("total_out")
    assert total_in is not None and total_out is not None
    assert Decimal(total_out.to_decimal()) == Decimal("1226.313180608006")
    assert total_in == XRPAmount(599_720_236)

    eff_slices = _executed_slices(quote)
    assert [s.get("src") for s in eff_slices] == ["CLOB", "CLOB", "AMM"]

    amm_out_total = sum(
        Decimal(str(s["out_take"].to_decimal()))
        for s in eff_slices
        if s.get("src") == "AMM"
    )
    assert amm_out_total > Decimal("834.22")
    assert amm_out_total < Decimal("834.25")
    amm_slices = [s for s in eff_slices if s.get("src") == "AMM"]
    assert len(amm_slices) == 1
    assert amm_slices[0]["in_take"] == XRPAmount(407_905_076)
    assert meta.get("post_to_book_remainder") is False


def test_offercreate_sell_fok_forward_not_capped_by_clob_only_capacity() -> None:
    seg = Segment(
        src="CLOB",
        quality=Quality.from_amounts(IOUAmount.from_decimal(Decimal("10")), XRPAmount(5_000_000)),
        out_max=IOUAmount.from_decimal(Decimal("10")),
        in_at_out_max=XRPAmount(5_000_000),
        raw_quality=Quality.from_amounts(IOUAmount.from_decimal(Decimal("10")), XRPAmount(5_000_000)),
        source_id="clob-cap",
    )
    amm = AMM(
        Decimal("1000"),
        Decimal("1000"),
        Decimal("0"),
        x_is_xrp=True,
        y_is_xrp=False,
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="XRP",
        out_cur="USD",
        offer_taker_gets_amt=XRPAmount(50_000_000),
        offer_taker_pays_amt=IOUAmount.from_decimal(Decimal("40")),
        offer_limit_quality=Quality.from_amounts(IOUAmount.from_decimal(Decimal("40")), XRPAmount(50_000_000)),
        offer_is_sell=True,
        offer_is_fok=True,
    )

    quote, _ = execute_single_path_intent(intent=intent, amm=amm, segs=[seg], in_cur="XRP", out_cur="USD")
    summary = quote.get("summary") or {}
    total_in = summary.get("total_in")
    assert total_in is not None
    assert Decimal(total_in.to_decimal()) > Decimal("49")


def test_offercreate_amm_only_with_limit_quality_tiny_amount_can_cross() -> None:
    # Regression: AMM-only + limit quality on tiny amounts must not collapse to
    # zero execution because of synthetic segment integer-grid misses.
    amm = AMM(
        Decimal("1428744.50589053"),
        Decimal("695147.830816"),
        Decimal("0.00327"),
        x_is_xrp=False,
        y_is_xrp=True,
    )
    taker_gets = IOUAmount.from_decimal(Decimal("0.000589"))
    taker_pays = XRPAmount(285)
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=taker_gets,
        offer_taker_pays_amt=taker_pays,
        offer_limit_quality=Quality.from_amounts(taker_pays, taker_gets),
        offer_is_sell=False,
    )

    quote, _ = execute_single_path_intent(intent=intent, amm=amm, segs=[], in_cur="USD", out_cur="XRP")
    summary = quote.get("summary") or {}
    total_out = summary.get("total_out")
    total_in = summary.get("total_in")
    assert total_out is not None and total_in is not None
    assert total_out.is_zero() is False
    assert int(total_out.value) == 285
    assert Decimal(total_in.to_decimal()) <= Decimal("0.000589")


def test_offercreate_passive_amm_only_limit_quality_allows_post_spq_below_floor() -> None:
    amm = AMM(
        Decimal("1424675.121794072"),
        Decimal("697361.138093"),
        Decimal("0.00327"),
        x_is_xrp=False,
        y_is_xrp=True,
    )
    taker_gets = IOUAmount.from_decimal(Decimal("1910.99693"))
    taker_pays = XRPAmount(932_043_590)
    limit_q = Quality.from_amounts(taker_pays, taker_gets)
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=65536,
        in_cur="USD",
        out_cur="XRP",
        offer_taker_gets_amt=taker_gets,
        offer_taker_pays_amt=taker_pays,
        offer_limit_quality=limit_q,
        offer_is_sell=False,
        offer_is_passive=True,
    )

    quote, meta = execute_single_path_intent(intent=intent, amm=amm, segs=[], in_cur="USD", out_cur="XRP")
    summary = quote.get("summary") or {}
    total_out = summary.get("total_out")
    total_in = summary.get("total_in")

    assert total_out is not None and total_out.value > 200_000_000
    assert total_in is not None
    assert Decimal(total_in.to_decimal()) > Decimal("470")
    assert Decimal(total_in.to_decimal()) < Decimal("472")
    assert summary.get("is_partial") is True
    assert meta.get("post_to_book_remainder") is True

    shadow = amm.clone()
    shadow.apply_fill_st(total_in, total_out)
    assert shadow.spq_quality_int() < limit_q


def test_payment_partial_funded_offer_can_meet_delivermin_without_false_plus_one_drop() -> None:
    segs = build_clob_segments_from_offers(
        [
            {
                "index": "223B76C68E23D96482EEB713B61F26A1788F687DB64FA3A65547E5A5F3A3EAED",
                "Account": "rfmdBKhtJw2J22rw1JxQcchQTM68qzE4N2",
                "Sequence": 117327390,
                "TakerGets": {"currency": "524C555344000000000000000000000000000000", "value": "9998.467046823989"},
                "TakerPays": "4813759960",
                "owner_funds": "20.770597474545",
                "quality": "2.0770597474544594865922645632",
            }
        ],
        in_cur="XRP",
        out_cur="524C555344000000000000000000000000000000",
    )
    intent = TxIntent(
        transaction_type="PAYMENT",
        flags=131072,
        in_cur="XRP",
        out_cur="524C555344000000000000000000000000000000",
        payment_sendmax_amt=XRPAmount(10_000_000),
        payment_delivermax_amt=IOUAmount.from_decimal(Decimal("20.978303")),
        payment_delivermin_amt=IOUAmount.from_decimal(Decimal("20.770597")),
        payment_partial_allowed=True,
    )

    quote, _ = execute_single_path_intent(
        intent=intent,
        amm=None,
        segs=segs,
        in_cur="XRP",
        out_cur="524C555344000000000000000000000000000000",
    )
    summary = quote.get("summary") or {}
    total_in = summary.get("total_in")
    total_out = summary.get("total_out")

    assert total_in == XRPAmount(10_000_000)
    assert total_out == IOUAmount.from_components(2_077_059_747_454_500, -14)
