from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR

import pytest

from xrpl_router.amm import AMM, AMMOverAsk
from xrpl_router.core import IOUAmount, IOU_MANTISSA_MIN, Quality, XRPAmount


def xrp(value: str) -> XRPAmount:
    drops = int((Decimal(value) / Decimal("0.000001")).to_integral_value(rounding=ROUND_FLOOR))
    return XRPAmount(drops)


def iou(value: str) -> IOUAmount:
    units = int(
        (Decimal(value) / Decimal("0.000000000000001")).to_integral_value(rounding=ROUND_FLOOR)
    )
    return IOUAmount(units, -15)


def iou_exact(value: str) -> IOUAmount:
    dec = Decimal(value)
    sign, digits, exponent = dec.as_tuple()
    mantissa = 0 if not digits else int("".join(str(d) for d in digits))
    if sign:
        mantissa = -mantissa
    return IOUAmount(mantissa, exponent)


def rippled_pool_amount(value: str, *, is_xrp: bool) -> XRPAmount | IOUAmount:
    return XRPAmount(int(value)) if is_xrp else iou_exact(value)


def test_amm_constructor_rejects_off_grid_reserves_and_invalid_fee() -> None:
    with pytest.raises(RuntimeError, match="drop grid|exact on ledger grid"):
        AMM(Decimal("1.0000005"), Decimal("2"), Decimal("0"), x_is_xrp=True, y_is_xrp=False)

    with pytest.raises(RuntimeError, match="1/100000 increments"):
        AMM(Decimal("1"), Decimal("2"), Decimal("0.000001"), x_is_xrp=False, y_is_xrp=False)


def test_amm_spq_is_pure_balance_ratio_and_not_fee_adjusted() -> None:
    zero_fee = AMM(Decimal("1000"), Decimal("2000"), Decimal("0"), x_is_xrp=True, y_is_xrp=True)
    nonzero_fee = AMM(
        Decimal("1000"),
        Decimal("2000"),
        Decimal("0.005"),
        x_is_xrp=True,
        y_is_xrp=True,
    )

    expected = Quality.from_amounts(XRPAmount(2_000_000_000), XRPAmount(1_000_000_000))

    assert zero_fee.spq_quality_int() == nonzero_fee.spq_quality_int()
    assert zero_fee.spq_quality_int() == expected


def test_amm_swap_in_and_swap_out_are_inverse_safe_on_grid() -> None:
    amm = AMM(Decimal("1000"), Decimal("1000"), Decimal("0"), x_is_xrp=True, y_is_xrp=True)

    out_amount = amm.swap_out_given_in_st(XRPAmount(10_000_000))

    assert out_amount == XRPAmount(9_900_990)
    assert amm.swap_in_given_out_st(out_amount) == XRPAmount(10_000_000)


def test_amm_dx_for_out_raises_overask_with_exact_non_drain_cap() -> None:
    amm = AMM(Decimal("1000"), Decimal("1000"), Decimal("0"), x_is_xrp=True, y_is_xrp=True)

    with pytest.raises(AMMOverAsk) as exc:
        amm.dx_for_out_st(XRPAmount(1_000_000_000))

    assert exc.value.max_out == XRPAmount(999_999_999)
    assert exc.value.dx_for_max == amm.swap_in_given_out_st(exc.value.max_out)


def test_amm_apply_fill_preserves_monotone_spq_and_non_decreasing_product() -> None:
    amm = AMM(Decimal("1000"), Decimal("1000"), Decimal("0.003"), x_is_xrp=True, y_is_xrp=True)
    prior_spq = amm.spq_quality_int()
    prior_product = amm._x_units * amm._y_units
    dx = XRPAmount(10_000_000)
    dy = amm.swap_out_given_in_st(dx)

    amm.apply_fill_st(dx, dy)

    assert amm._x_units * amm._y_units >= prior_product
    assert amm.spq_quality_int() <= prior_spq


def test_amm_synthetic_segment_matches_real_like_anchored_cap_case() -> None:
    amm = AMM(
        Decimal("1411396.414475868"),
        Decimal("717363.789664"),
        Decimal("0.00529"),
        x_is_xrp=False,
        y_is_xrp=True,
    )
    threshold = Quality.from_amounts(xrp("966.3005"), iou("1911.43902"))
    out_cap = xrp("20.610728")

    segment = amm.synthetic_segment_for_quality(threshold, max_out_cap=out_cap)

    assert segment is not None
    assert segment.src == "AMM"
    assert segment.out_max == out_cap
    assert segment.quality >= threshold

    shadow = amm.clone()
    shadow.apply_fill_st(segment.in_at_out_max, segment.out_max)
    assert shadow.spq_quality_int() > threshold


def test_amm_synthetic_segment_rejects_equal_spq_competing_threshold() -> None:
    amm = AMM(Decimal("1000"), Decimal("2000"), Decimal("0"), x_is_xrp=True, y_is_xrp=True)

    assert amm.synthetic_segment_for_quality(amm.spq_quality_int()) is None


def test_amm_synthetic_segment_supports_start_with_taker_pays_solver_branch() -> None:
    amm = AMM(Decimal("1000"), Decimal("1000"), Decimal("0.003"), x_is_xrp=True, y_is_xrp=False)
    sample_in = XRPAmount(1_000_000)
    sample_out = amm.swap_out_given_in_st(sample_in)
    threshold = Quality.from_amounts(sample_out, sample_in)

    segment = amm.synthetic_segment_for_quality(threshold, max_out_cap=sample_out)

    assert segment is not None
    assert segment.src == "AMM"
    assert segment.out_max == sample_out
    assert segment.quality >= threshold


def test_amm_forward_quote_matches_rippled_large_mantissa_rounding_case() -> None:
    amm = AMM(
        Decimal("696932.590304"),
        Decimal("1425548.297477277"),
        Decimal("0.00327"),
        x_is_xrp=True,
        y_is_xrp=False,
        number_mantissa_scale="large",
    )

    assert amm.swap_out_given_in_st(XRPAmount(3_385_350)) == IOUAmount(6_901_923_876_705_000, -15)


def test_amm_reverse_quote_matches_rippled_iou_denom_amount_subtraction_case() -> None:
    amm = AMM(
        Decimal("696932.590304"),
        Decimal("1425548.297477277"),
        Decimal("0.00327"),
        x_is_xrp=True,
        y_is_xrp=False,
    )

    assert amm.swap_in_given_out_st(IOUAmount(6_901_923_876_022_920, -15)) == XRPAmount(3_385_351)


def test_amm_max_out_handles_iou_min_boundary_without_name_errors() -> None:
    amm = AMM(
        XRPAmount(1_000_000),
        IOUAmount(IOU_MANTISSA_MIN, -96),
        Decimal("0"),
        x_is_xrp=True,
        y_is_xrp=False,
    )

    assert amm.max_out_amount().is_zero()


@pytest.mark.parametrize(
    ("pool_in", "pool_out", "quality", "fee_units"),
    [
        ("0.001519763260828713", "1558701", Quality(5414253689393440221), 1000),
        ("105439.2955578965", "49398693", Quality(5910869983721805038), 400),
        ("1892611", "0.01099814367603737", Quality(6703103457950430139), 1000),
    ],
)
def test_amm_synthetic_segment_matches_rippled_change_spot_price_quality_cases(
    pool_in: str,
    pool_out: str,
    quality: Quality,
    fee_units: int,
) -> None:
    x_is_xrp = pool_in.isdigit()
    y_is_xrp = pool_out.isdigit()
    amm = AMM(
        rippled_pool_amount(pool_in, is_xrp=x_is_xrp),
        rippled_pool_amount(pool_out, is_xrp=y_is_xrp),
        Decimal(fee_units) / Decimal("100000"),
        x_is_xrp=x_is_xrp,
        y_is_xrp=y_is_xrp,
    )

    segment = amm.synthetic_segment_for_quality(quality, require_post_spq_floor=False)

    assert segment is not None
    assert segment.quality >= quality


def test_amm_synthetic_segment_rejects_rippled_known_fail_case() -> None:
    amm = AMM(
        rippled_pool_amount("350131413924", is_xrp=True),
        rippled_pool_amount("1576879.110907892", is_xrp=False),
        Decimal("0.0065"),
        x_is_xrp=True,
        y_is_xrp=False,
    )

    assert amm.synthetic_segment_for_quality(Quality(6487411636539049449), require_post_spq_floor=False) is None


def test_amm_synthetic_segment_keeps_boundary_tie() -> None:
    amm = AMM(
        Decimal("706462.025694"),
        Decimal("1430605.814158099"),
        Decimal("0.00529"),
        x_is_xrp=True,
        y_is_xrp=False,
    )
    threshold = Quality(6490153102872721632)

    segment = amm.synthetic_segment_for_quality(threshold, require_post_spq_floor=False)

    assert segment is not None
    assert segment.quality == threshold
