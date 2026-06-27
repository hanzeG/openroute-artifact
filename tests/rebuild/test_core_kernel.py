from __future__ import annotations

import pytest

from xrpl_router import (
    AmountDomainError,
    DROPS_PER_XRP,
    IOUAmount,
    IOU_EXP_MAX,
    IOU_EXP_MIN,
    IOU_MANTISSA_MAX,
    IOU_MANTISSA_MIN,
    INT64_MAX,
    INT64_MIN,
    InvariantViolation,
    NormalisationError,
    Quality,
    Segment,
    UnsupportedKernelOperation,
    XRPAmount,
    ceil_div,
    floor_div,
    mul_ratio_round_down,
    mul_ratio_round_up,
    parse_iou_decimal,
    parse_xrp_decimal,
    parse_xrp_drops,
)


def test_top_level_kernel_exports_are_available() -> None:
    assert DROPS_PER_XRP == 1_000_000
    assert IOU_MANTISSA_MIN == 10**15
    assert IOU_MANTISSA_MAX == (10**16) - 1
    assert IOU_EXP_MIN == -96
    assert IOU_EXP_MAX == 80
    assert floor_div(7, 3) == 2
    assert ceil_div(7, 3) == 3
    assert mul_ratio_round_down(10, 2, 3) == 6
    assert mul_ratio_round_up(10, 2, 3) == 7


def test_xrp_amount_rejects_negative_drops() -> None:
    assert XRPAmount(-1).drops == -1


def test_xrp_amount_rejects_values_outside_signed_int64_range() -> None:
    with pytest.raises(AmountDomainError, match="int64"):
        XRPAmount(INT64_MAX + 1)
    with pytest.raises(AmountDomainError, match="int64"):
        XRPAmount(INT64_MIN - 1)


def test_xrp_amount_add_and_subtract_in_integer_drops() -> None:
    left = XRPAmount(9)
    right = XRPAmount(4)

    assert left + right == XRPAmount(13)
    assert left - right == XRPAmount(5)


def test_xrp_amount_subtraction_underflow_fails_fast() -> None:
    assert XRPAmount(3) - XRPAmount(4) == XRPAmount(-1)


def test_xrp_amount_mixed_type_arithmetic_is_unsupported() -> None:
    with pytest.raises(UnsupportedKernelOperation, match="requires XRPAmount"):
        XRPAmount(3) + IOUAmount(IOU_MANTISSA_MIN, -15)  # type: ignore[arg-type]


def test_iou_amount_zero_is_canonical() -> None:
    assert IOUAmount(0, -15) == IOUAmount(0, 0)
    assert IOUAmount(0, -15).exponent == -100


def test_iou_amount_rejects_mantissa_outside_signed_int64_range() -> None:
    with pytest.raises(AmountDomainError, match="int64"):
        IOUAmount(INT64_MAX + 1, 0)
    with pytest.raises(AmountDomainError, match="int64"):
        IOUAmount(INT64_MIN - 1, 0)


def test_iou_amount_normalizes_to_canonical_representation() -> None:
    amount = IOUAmount(123, -2)
    assert amount == IOUAmount(1230000000000000, -15)
    assert amount.mantissa == 1230000000000000
    assert amount.exponent == -15


def test_iou_amount_underflow_to_zero_at_minimum_exponent() -> None:
    amount = IOUAmount(1, IOU_EXP_MIN)
    assert amount.is_zero() is True
    assert amount.exponent == -100


def test_iou_amount_exponent_overflow_fails_fast() -> None:
    with pytest.raises(NormalisationError, match="overflow"):
        IOUAmount(IOU_MANTISSA_MAX * 10, IOU_EXP_MAX)


def test_iou_amount_lossy_normalization_rounds_nearest_even_like_rippled_stnumber() -> None:
    amount = IOUAmount(53619302949061667, -11)
    assert amount.mantissa == 5361930294906167
    assert amount.exponent == -10

    tie_to_even = IOUAmount(12345678901234565, -15)
    assert tie_to_even.mantissa == 1234567890123456
    assert tie_to_even.exponent == -14

    carry = IOUAmount(99999999999999999, -15)
    assert carry.mantissa == 1000000000000000
    assert carry.exponent == -13


def test_iou_amount_comparison_is_stable_across_representations() -> None:
    smaller = IOUAmount(1, -15)
    larger = IOUAmount(2, -15)
    assert smaller < larger
    assert larger > smaller


def test_iou_amount_addition_preserves_exact_value_then_recannonicalizes() -> None:
    left = IOUAmount(IOU_MANTISSA_MIN, -15)
    right = IOUAmount(IOU_MANTISSA_MIN, -15)
    assert left + right == IOUAmount(2000000000000000, -15)


def test_iou_amount_large_exponent_gap_addition_truncates_small_tail_like_rippled() -> None:
    left = IOUAmount(IOU_MANTISSA_MIN, 0)
    right = IOUAmount(IOU_MANTISSA_MIN, -20)
    assert left + right == left


def test_iou_amount_subtraction_can_go_negative_like_rippled() -> None:
    assert IOUAmount(IOU_MANTISSA_MIN, -15) - IOUAmount(2000000000000000, -15) == IOUAmount(
        -1000000000000000, -15
    )


def test_integer_rounding_helpers_fail_on_invalid_domain() -> None:
    with pytest.raises(AmountDomainError):
        floor_div(-1, 3)
    with pytest.raises(AmountDomainError):
        ceil_div(1, 0)
    with pytest.raises(AmountDomainError):
        mul_ratio_round_down(1, 1, 0)
    with pytest.raises(AmountDomainError):
        mul_ratio_round_up(-1, 1, 1)


def test_rounding_helpers_distinguish_input_up_vs_output_down() -> None:
    assert mul_ratio_round_down(10, 2, 3) == 6
    assert mul_ratio_round_up(10, 2, 3) == 7


def test_xrp_parsers_distinguish_drops_vs_decimal_surface() -> None:
    assert parse_xrp_drops("1000000", field_name="drops") == XRPAmount(1_000_000)
    assert parse_xrp_decimal("1", field_name="xrp") == XRPAmount(1_000_000)
    assert parse_xrp_decimal("1.25", field_name="xrp") == XRPAmount(1_250_000)


def test_xrp_decimal_parser_rejects_sub_drop_precision() -> None:
    with pytest.raises(AmountDomainError, match="drop grid"):
        parse_xrp_decimal("1.0000001", field_name="xrp")


def test_iou_decimal_parser_matches_canonical_amount() -> None:
    assert parse_iou_decimal("18.61431502790658", field_name="iou") == IOUAmount(
        1861431502790658,
        -14,
    )
    assert parse_iou_decimal("-0.5", field_name="iou") == IOUAmount(-5000000000000000, -16)
    assert parse_iou_decimal("9999999999999990e-1", field_name="iou") == IOUAmount(
        999999999999999,
        0,
    )
    assert parse_iou_decimal("1000000000000000e4", field_name="iou") == IOUAmount(
        1000000000000000,
        4,
    )
    assert parse_iou_decimal("0.0005889999437679493000000000000", field_name="iou") == IOUAmount(
        5889999437679493,
        -19,
    )


def test_quality_from_amounts_matches_rippled_getrate_encoding_for_simple_ratio() -> None:
    quality = Quality.from_amounts(XRPAmount(1), XRPAmount(10))
    assert quality.value == (((100 - 14) << 56) | 1000000000000000)


def test_quality_equal_ratios_compare_equal_across_valid_representations() -> None:
    left = Quality.from_amounts(IOUAmount(1500000000000000, -15), XRPAmount(3))
    right = Quality.from_amounts(IOUAmount(3000000000000000, -15), XRPAmount(6))
    assert left == right


def test_quality_scales_native_input_like_rippled_divide_before_encoding() -> None:
    quality = Quality.from_amounts(IOUAmount(3000000000000000, -15), XRPAmount(1))
    assert quality.rate() == IOUAmount(3333333333333334, -16)


def test_quality_scales_native_output_like_rippled_divide_before_encoding() -> None:
    quality = Quality.from_amounts(XRPAmount(1), IOUAmount(3000000000000000, -15))
    assert quality.rate() == IOUAmount(3000000000000000, -15)


def test_quality_from_amounts_uses_stnumber_rounding_at_offercreate_boundary() -> None:
    quality = Quality.from_amounts(
        parse_iou_decimal("53171.15", field_name="out"),
        parse_xrp_drops("28510000000", field_name="in"),
    )
    assert quality.value == int("5A130CA5AE89CD37", 16)


def test_quality_is_monotonic_and_higher_is_better() -> None:
    better = Quality.from_amounts(XRPAmount(11), XRPAmount(10))
    worse = Quality.from_amounts(XRPAmount(10), XRPAmount(10))
    assert better > worse


def test_quality_needs_reverse_sort_for_best_first_rippled_order() -> None:
    better = Quality.from_amounts(XRPAmount(11), XRPAmount(10))
    worse = Quality.from_amounts(XRPAmount(10), XRPAmount(10))
    assert sorted([better, worse]) == [worse, better]
    assert sorted([better, worse], reverse=True) == [better, worse]


def test_quality_rate_roundtrips_to_iou_amount() -> None:
    quality = Quality.from_amounts(XRPAmount(1), XRPAmount(10))
    assert quality.rate() == IOUAmount(1000000000000000, -14)


def test_quality_from_rate_roundtrips_to_same_rate() -> None:
    rate = IOUAmount(5173278979415522, -10)
    quality = Quality.from_rate(rate)
    assert quality.rate() == rate


def test_quality_rejects_zero_or_invalid_inputs() -> None:
    with pytest.raises(AmountDomainError, match="strictly positive"):
        Quality.from_amounts(XRPAmount(0), XRPAmount(1))
    with pytest.raises(AmountDomainError, match="strictly positive"):
        Quality.from_amounts(XRPAmount(1), XRPAmount(0))
    with pytest.raises(AmountDomainError, match="strictly positive"):
        Quality.from_rate(IOUAmount(0, 0))


def test_quality_returns_zero_for_worthless_ratio_like_rippled_getrate() -> None:
    quality = Quality.from_amounts(IOUAmount(IOU_MANTISSA_MAX, IOU_EXP_MAX), XRPAmount(1))
    assert quality.is_zero() is True


def test_segment_accepts_minimal_valid_kernel_shape() -> None:
    out_amount = XRPAmount(10)
    in_amount = IOUAmount(IOU_MANTISSA_MIN, -15)
    quality = Quality.from_amounts(out_amount, in_amount)
    segment = Segment(
        src="CLOB",
        quality=quality,
        out_max=out_amount,
        in_at_out_max=in_amount,
    )

    assert segment.src == "CLOB"
    assert segment.quality == quality


def test_segment_rejects_invalid_src() -> None:
    with pytest.raises(UnsupportedKernelOperation, match="segment src"):
        Segment(
            src="BAD",  # type: ignore[arg-type]
            quality=Quality.from_amounts(XRPAmount(1), XRPAmount(1)),
            out_max=XRPAmount(1),
            in_at_out_max=XRPAmount(1),
        )


def test_segment_rejects_zero_amounts() -> None:
    with pytest.raises(InvariantViolation, match="strictly positive"):
        Segment(
            src="AMM",
            quality=Quality.from_amounts(XRPAmount(1), XRPAmount(1)),
            out_max=XRPAmount(0),
            in_at_out_max=XRPAmount(1),
        )


def test_segment_rejects_negative_amounts() -> None:
    with pytest.raises(InvariantViolation, match="strictly positive"):
        Segment(
            src="AMM",
            quality=Quality.from_amounts(XRPAmount(1), XRPAmount(1)),
            out_max=XRPAmount(-1),
            in_at_out_max=XRPAmount(1),
        )


def test_segment_allows_quoted_quality_to_differ_from_current_amount_pair_like_rippled_offer() -> None:
    segment = Segment(
        src="CLOB",
        quality=Quality.from_amounts(XRPAmount(1), XRPAmount(1)),
        out_max=XRPAmount(2),
        in_at_out_max=XRPAmount(7),
    )

    assert segment.quality == Quality.from_amounts(XRPAmount(1), XRPAmount(1))
