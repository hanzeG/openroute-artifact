from decimal import Decimal, ROUND_FLOOR

from xrpl_router.amm import AMM
from xrpl_router.core import IOUAmount, Quality, XRPAmount
from xrpl_router.core.constants import IOU_QUANTUM, XRP_QUANTUM
from xrpl_router.core.fmt import amount_to_decimal


def _xrp_amount(v: Decimal) -> XRPAmount:
    drops = int((v / XRP_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
    return XRPAmount(drops)


def _iou_amount(v: Decimal) -> IOUAmount:
    units = int((v / IOU_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
    return IOUAmount.from_components(units, -15)


def test_anchor_threshold_uses_same_quality_domain_as_quality_objects():
    # Real rUSD->XRP-like state where AMM SPQ is above a CLOB threshold.
    amm = AMM(
        Decimal("1411396.414475868"),
        Decimal("717363.789664"),
        Decimal("0.00529"),
        x_is_xrp=False,
        y_is_xrp=True,
    )

    q_threshold = Quality.from_amounts(
        _xrp_amount(Decimal("966.3005")),
        _iou_amount(Decimal("1911.43902")),
    )
    out_cap = _xrp_amount(Decimal("20.610728"))

    seg = amm.synthetic_segment_for_quality(q_threshold, max_out_cap=out_cap)
    assert seg is not None

    # Regression guard: with mixed-unit quality-domain mismatch this used to return a tiny AMM slice.
    assert amount_to_decimal(seg.out_max) == Decimal("20.610728")

    # Full-cap AMM still keeps marginal quality above threshold, so over-cap fallback is expected.
    shadow = amm.clone()
    shadow.apply_fill_st(seg.in_at_out_max, seg.out_max)
    assert shadow.spq_quality_int() > q_threshold
