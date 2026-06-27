from __future__ import annotations

from decimal import Decimal
import pytest

from xrpl_router.amm import AMM
from xrpl_router.book_offers import BookOfferSegment
from xrpl_router.book_step import BookStep
from xrpl_router.core import IOUAmount, Quality, Segment, XRPAmount
from xrpl_router.execution_semantics import _amt_from_units, execute_single_path_intent
from xrpl_router.tx_intent import TxIntent

RUSD = "524C555344000000000000000000000000000000"
XRP = "XRP"


def iou(value: str) -> IOUAmount:
    return IOUAmount(int(Decimal(value) * (Decimal(10) ** 15)), -15)


def test_payment_partial_without_flag_fails_fast() -> None:
    segment = Segment(
        src="CLOB",
        quality=Quality.from_amounts(iou("10"), XRPAmount(10_000_000)),
        out_max=iou("10"),
        in_at_out_max=XRPAmount(10_000_000),
    )
    intent = TxIntent(
        transaction_type="PAYMENT",
        flags=0,
        in_cur=XRP,
        out_cur=RUSD,
        payment_sendmax_amt=XRPAmount(5_000_000),
        payment_delivermax_amt=iou("10"),
        payment_partial_allowed=False,
    )

    with pytest.raises(RuntimeError, match="tfPartialPayment"):
        execute_single_path_intent(intent=intent, amm=None, segs=[segment], in_cur=XRP, out_cur=RUSD)


def test_payment_one_quantum_shortfall_is_not_treated_as_partial() -> None:
    requested_out = IOUAmount(10000000000000001, -15)
    segment = Segment(
        src="CLOB",
        quality=Quality.from_amounts(IOUAmount(10000000000000000, -15), XRPAmount(10_000_000)),
        out_max=IOUAmount(10000000000000000, -15),
        in_at_out_max=XRPAmount(10_000_000),
    )
    intent = TxIntent(
        transaction_type="PAYMENT",
        flags=0,
        in_cur=XRP,
        out_cur=RUSD,
        payment_sendmax_amt=XRPAmount(10_000_000),
        payment_delivermax_amt=requested_out,
        payment_partial_allowed=False,
    )

    quote, meta = execute_single_path_intent(intent=intent, amm=None, segs=[segment], in_cur=XRP, out_cur=RUSD)

    assert meta["mode"] == "payment"
    assert quote["summary"]["is_partial"] is False
    assert quote["summary"]["total_out"] == IOUAmount(10000000000000000, -15)


def test_offercreate_buy_fill_or_kill_requires_full_target() -> None:
    segment = Segment(
        src="CLOB",
        quality=Quality.from_amounts(XRPAmount(5_000_000), iou("10")),
        out_max=XRPAmount(5_000_000),
        in_at_out_max=iou("10"),
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur=RUSD,
        out_cur=XRP,
        offer_taker_gets_amt=iou("10"),
        offer_taker_pays_amt=XRPAmount(10_000_000),
        offer_limit_quality=Quality.from_amounts(XRPAmount(100_000), iou("10")),
        offer_is_fok=True,
    )

    with pytest.raises(RuntimeError, match="tfFillOrKill"):
        execute_single_path_intent(intent=intent, amm=None, segs=[segment], in_cur=RUSD, out_cur=XRP)


def test_offercreate_buy_fill_or_kill_accepts_one_quantum_iou_shortfall() -> None:
    requested_out = IOUAmount(10000000000000001, -15)
    segment = Segment(
        src="CLOB",
        quality=Quality.from_amounts(IOUAmount(10000000000000000, -15), XRPAmount(10_000_000)),
        out_max=IOUAmount(10000000000000000, -15),
        in_at_out_max=XRPAmount(10_000_000),
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur=XRP,
        out_cur=RUSD,
        offer_taker_gets_amt=XRPAmount(10_000_000),
        offer_taker_pays_amt=requested_out,
        offer_limit_quality=Quality.from_amounts(requested_out, XRPAmount(10_000_000)),
        offer_is_sell=False,
        offer_is_fok=True,
    )

    quote, meta = execute_single_path_intent(intent=intent, amm=None, segs=[segment], in_cur=XRP, out_cur=RUSD)

    assert meta["mode"] == "offer-crossing-buy+fok"
    assert meta["post_to_book_remainder"] is False
    assert quote["summary"]["is_partial"] is False
    assert quote["summary"]["total_out"] == IOUAmount(10000000000000000, -15)


def test_offercreate_passive_raises_floor_above_equal_quality_liquidity() -> None:
    segment = Segment(
        src="CLOB",
        quality=Quality.from_amounts(XRPAmount(10_000_000), iou("10")),
        out_max=XRPAmount(10_000_000),
        in_at_out_max=iou("10"),
    )
    base_quality = Quality.from_amounts(XRPAmount(10_000_000), iou("10"))
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur=RUSD,
        out_cur=XRP,
        offer_taker_gets_amt=iou("10"),
        offer_taker_pays_amt=XRPAmount(10_000_000),
        offer_limit_quality=base_quality,
        offer_is_passive=True,
    )

    quote, meta = execute_single_path_intent(intent=intent, amm=None, segs=[segment], in_cur=RUSD, out_cur=XRP)

    assert quote["summary"]["is_partial"] is True
    assert meta["post_to_book_remainder"] is True
    assert quote["summary"]["total_out"] == XRPAmount(0)


def test_offercreate_amm_only_with_limit_quality_uses_unbounded_single_path_quote() -> None:
    amm = AMM(
        Decimal("1428744.50589053"),
        Decimal("695147.830816"),
        Decimal("0.00327"),
        x_is_xrp=False,
        y_is_xrp=True,
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur=RUSD,
        out_cur=XRP,
        offer_taker_gets_amt=iou("0.000589"),
        offer_taker_pays_amt=XRPAmount(285),
        offer_limit_quality=Quality.from_amounts(XRPAmount(285), iou("0.000589")),
        offer_is_sell=False,
    )

    quote, meta = execute_single_path_intent(intent=intent, amm=amm, segs=[], in_cur=RUSD, out_cur=XRP)

    assert meta["mode"] == "offer-crossing-buy"
    assert quote["summary"]["total_out"] == XRPAmount(285)
    assert quote["summary"]["total_in"] <= iou("0.000589")


def test_offercreate_sell_executes_amm_when_clob_is_fully_filtered_by_limit_quality() -> None:
    amm = AMM(
        Decimal("1315152.213956053"),
        Decimal("681211.711684"),
        Decimal("0.00519"),
        x_is_xrp=False,
        y_is_xrp=True,
        number_mantissa_scale="large",
    )
    worse_clob = Segment(
        src="CLOB",
        quality=Quality.from_amounts(XRPAmount(30_000_000), iou("79.53053385")),
        out_max=XRPAmount(30_000_000),
        in_at_out_max=iou("79.53053385"),
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur=RUSD,
        out_cur=XRP,
        source_account="rNDni1mYwggZzriRzs1YB5E4PUBpNdckQ6",
        offer_taker_gets_amt=iou("79.530535"),
        offer_taker_pays_amt=XRPAmount(40_978_601),
        offer_limit_quality=Quality.from_amounts(XRPAmount(40_978_601), iou("79.530535")),
        offer_is_sell=True,
        offer_is_passive=True,
    )

    quote, meta = execute_single_path_intent(
        intent=intent,
        amm=amm,
        segs=[worse_clob],
        in_cur=RUSD,
        out_cur=XRP,
    )

    assert meta["mode"] == "offer-crossing-sell+sell+passive"
    assert meta["post_to_book_remainder"] is True
    assert quote["summary"]["total_out"] == XRPAmount(36_082_084)
    assert quote["summary"]["total_in"] == IOUAmount(7_002_746_249_036_701, -14)


def test_offercreate_sell_amm_only_passive_uses_quality_function_limit_out_cap() -> None:
    amm = AMM(
        Decimal("1315152.213956053"),
        Decimal("681211.711684"),
        Decimal("0.00519"),
        x_is_xrp=False,
        y_is_xrp=True,
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=589824,
        source_account="rNDni1mYwggZzriRzs1YB5E4PUBpNdckQ6",
        in_cur=RUSD,
        out_cur=XRP,
        source_funds_cap_amt=IOUAmount(8836726088280955, -14),
        offer_taker_gets_amt=IOUAmount(7953053500000000, -14),
        offer_taker_pays_amt=XRPAmount(40_978_601),
        offer_limit_quality=Quality.from_amounts(XRPAmount(40_978_601), IOUAmount(7953053500000000, -14)),
        offer_is_sell=True,
        offer_is_passive=True,
    )

    quote, meta = execute_single_path_intent(intent=intent, amm=amm, segs=[], in_cur=RUSD, out_cur=XRP)

    assert meta["mode"] == "offer-crossing-sell+sell+passive"
    assert meta["post_to_book_remainder"] is False
    assert quote["summary"]["total_out"] == XRPAmount(36_082_084)
    assert quote["summary"]["total_in"] == IOUAmount(7_002_746_249_133_001, -14)


def test_offercreate_sell_treats_sub_quantum_residual_budget_as_complete() -> None:
    segment = Segment(
        src="CLOB",
        quality=Quality.from_amounts(XRPAmount(19_231_403), iou("40.018857279536")),
        out_max=XRPAmount(19_231_403),
        in_at_out_max=iou("40.018857279536"),
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur=RUSD,
        out_cur=XRP,
        offer_taker_gets_amt=iou("40.018857279537"),
        offer_taker_pays_amt=XRPAmount(18_000_000),
        offer_limit_quality=Quality.from_amounts(XRPAmount(18_000_000), iou("40.018857279537")),
        offer_is_sell=True,
        offer_is_fok=True,
    )

    quote, meta = execute_single_path_intent(intent=intent, amm=None, segs=[segment], in_cur=RUSD, out_cur=XRP)

    assert quote["summary"]["is_partial"] is False
    assert meta["post_to_book_remainder"] is False


def test_payment_forward_prefers_maxoffer_budget_slice_when_synthetic_offer_cannot_cross_tip() -> None:
    amm = AMM(
        Decimal("682399.189688"),
        Decimal("1312735.253177556"),
        Decimal("0.00519"),
        x_is_xrp=True,
        y_is_xrp=False,
    )
    amm.apply_fill_st(XRPAmount(7_787_937), IOUAmount(1_490_377_505_300_000, -14))

    clob = Segment(
        src="CLOB",
        quality=Quality.from_amounts(IOUAmount(1_666_613_443_299_999, -13), XRPAmount(87_120_410)),
        out_max=IOUAmount(1_666_613_443_299_999, -13),
        in_at_out_max=XRPAmount(87_120_410),
    )

    out_take, in_take, trace = BookStep.from_static([clob], amm=amm, mode="payment").fwd(
        None,
        XRPAmount(87_120_410),
        return_trace=True,
    )

    assert out_take == IOUAmount(1_666_992_546_620_000, -13)
    assert in_take == XRPAmount(87_120_410)
    assert [row["src"] for row in trace] == ["AMM"]


def test_amt_from_units_canonicalizes_large_iou_without_int64_overflow() -> None:
    amount = _amt_from_units(RUSD, 10_007_538_952_502_700_000)

    assert amount == IOUAmount(1_000_753_895_250_270, -11)


def test_offercreate_preserves_non_fill_deleted_marker_across_mutable_rounds() -> None:
    external_quality = Quality.from_amounts(iou("2"), XRPAmount(1_000_000))
    self_quality = Quality.from_amounts(iou("1"), XRPAmount(500_000))
    external = BookOfferSegment(
        "external",
        external_quality,
        Segment(
            src="CLOB",
            quality=external_quality,
            out_max=iou("2"),
            in_at_out_max=XRPAmount(1_000_000),
        ),
        owner="rMaker",
    )
    self_deleted = BookOfferSegment(
        "self-deleted",
        self_quality,
        Segment(
            src="CLOB",
            quality=self_quality,
            out_max=iou("1"),
            in_at_out_max=XRPAmount(500_000),
        ),
        owner="rSelf",
        non_fill_deleted=True,
    )
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        source_account="rSelf",
        in_cur=XRP,
        out_cur=RUSD,
        offer_taker_gets_amt=XRPAmount(10_000_000),
        offer_taker_pays_amt=iou("10"),
        offer_limit_quality=self_quality,
        offer_is_sell=True,
    )

    quote, _ = execute_single_path_intent(
        intent=intent,
        amm=None,
        segs=[external, self_deleted],
        in_cur=XRP,
        out_cur=RUSD,
    )

    fill_slices = [slice_ for slice_ in quote["slices"] if slice_["src"] == "CLOB"]
    delete_slices = [slice_ for slice_ in quote["slices"] if slice_["src"] == "CLOB_DELETE"]
    assert [slice_["source_id"] for slice_ in fill_slices] == ["external"]
    assert [slice_["source_id"] for slice_ in delete_slices] == ["self-deleted"]
    assert quote["summary"]["total_out"] == iou("2")
    assert quote["summary"]["total_in"] == XRPAmount(1_000_000)
