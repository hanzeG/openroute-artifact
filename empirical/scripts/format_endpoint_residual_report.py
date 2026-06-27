#!/usr/bin/env python3
"""Format endpoint-residual aggregate results for reporting."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from statistics import NormalDist
from typing import Any


SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _percent(num: int, den: int) -> str:
    return f"{(num / den * 100) if den else 0:.2f}%"


def _sci(value: float) -> str:
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10**exponent)
    return f"{mantissa:.4f}×10{str(exponent).translate(SUPERSCRIPT)}"


def _p_value_two_tailed(t_stat: float, df: int) -> float:
    if not math.isfinite(t_stat):
        return 0.0
    try:
        from scipy import stats  # type: ignore

        return float(2 * stats.t.sf(abs(t_stat), df))
    except Exception:
        # The two-week datasets have large df; normal fallback is sufficient for
        # report generation if scipy is unavailable.
        return float(2 * (1 - NormalDist().cdf(abs(t_stat))))


def _bucket_lines(title: str, buckets: dict[str, int], denominator: int) -> list[str]:
    zero = int(buckets.get("=0", 0))
    le12 = zero
    le9 = zero
    gt9 = int(buckets.get("> 1e-9", 0))
    for label, count in buckets.items():
        if label in {"=0", "> 1e-9"}:
            continue
        if not label.startswith("<="):
            continue
        threshold = float(label.split("<=", 1)[1].strip())
        if threshold <= 1e-12:
            le12 += int(count)
        if threshold <= 1e-9:
            le9 += int(count)
    return [
        f"- {title}:",
        f"  - =0: {zero:,} / {denominator:,}, {_percent(zero, denominator)}",
        f"  - <= 1×10⁻¹²: {le12:,} / {denominator:,}, {_percent(le12, denominator)}",
        f"  - <= 1×10⁻⁹: {le9:,} / {denominator:,}, {_percent(le9, denominator)}",
        f"  - > 1×10⁻⁹: {gt9:,} / {denominator:,}, {_percent(gt9, denominator)}",
    ]


def _stats_payload(aggregate: dict[str, Any], side: str, signed: bool) -> dict[str, Any]:
    endpoint_keys = {
        "input": "structure_matched_endpoint_ttest_inputs",
        "output": "structure_matched_endpoint_ttest_outputs",
        "predicted": "structure_matched_endpoint_ttest_predicted_side",
        "constraint": "structure_matched_endpoint_ttest_constraint_side",
    }
    endpoint_key = endpoint_keys[side]
    error_key = "signed_relative_error" if signed else "unsigned_relative_error"
    return aggregate[endpoint_key][error_key]


def _stats_count(aggregate: dict[str, Any], side: str) -> int:
    if side == "predicted" and "structure_matched_endpoint_ttest_predicted_side" not in aggregate:
        return 0
    if side == "constraint" and "structure_matched_endpoint_ttest_constraint_side" not in aggregate:
        return 0
    return int(_stats_payload(aggregate, side, signed=False).get("count") or 0)


def _stats_lines(aggregate: dict[str, Any], side: str, signed: bool, *, title: str | None = None) -> list[str]:
    payload = _stats_payload(aggregate, side, signed)
    count = int(payload.get("count") or 0)
    side_titles = {
        "input": "Input side",
        "output": "Output side",
        "predicted": "Predicted side",
        "constraint": "Constraint side",
    }
    side_title = title or side_titles[side]
    if count == 0 or payload.get("mean") is None:
        return [f"- {side_title}:", "  - no classified measurements"]
    mean = float(payload["mean"])
    stddev = float(payload.get("sample_standard_deviation") or 0)
    sqrt_n = math.sqrt(count)
    standard_error = stddev / sqrt_n if count else 0.0
    if standard_error == 0:
        t_stat = 0.0 if mean == 0 else math.copysign(math.inf, mean)
    else:
        t_stat = mean / standard_error
    df = count - 1
    if count < 2:
        p_value = 1.0 if mean == 0 else 0.0
    else:
        p_value = 1.0 if standard_error == 0 and mean == 0 else _p_value_two_tailed(t_stat, df)
    t_stat_text = "inf" if math.isinf(t_stat) else f"{t_stat:.4f}"
    return [
        f"- {side_title}:",
        f"  - average error = {_sci(mean)}",
        f"  - standard deviation = {_sci(stddev)}",
        f"  - sqrt(n) = sqrt({count:,}) = {sqrt_n:.3f}",
        f"  - standard error = standard deviation / sqrt(n)",
        f"  - = {_sci(stddev)} / {sqrt_n:.3f} = {_sci(standard_error)}",
        f"  - t-statistic = {_sci(mean)} / {_sci(standard_error)} = {t_stat_text}",
        f"  - degrees of freedom = {df:,}",
        f"  - two-tailed p-value = {p_value:.4f}",
        f"  - result: {'reject H0' if p_value < 0.05 else 'fail to reject H0'}",
    ]


def _measurement_stats_lines(
    payload: dict[str, Any],
    *,
    prefix: str,
    tx_count: int,
    side: str,
    include_tx_count: bool = True,
) -> list[str]:
    count = int(payload.get("count") or 0)
    tx_count_lines = [f"{prefix}tx_count = {tx_count:,}"] if include_tx_count else []
    if count == 0 or payload.get("mean") is None:
        return tx_count_lines + [
            f"{prefix}measurement_count = 0",
            f"{prefix}side = {side}",
            f"{prefix}no measurements",
        ]
    mean = float(payload["mean"])
    stddev = float(payload.get("sample_standard_deviation") or 0)
    sqrt_n = math.sqrt(count)
    standard_error = stddev / sqrt_n if count else 0.0
    if standard_error == 0:
        t_stat = 0.0 if mean == 0 else math.copysign(math.inf, mean)
    else:
        t_stat = mean / standard_error
    df = count - 1
    if count < 2:
        p_value = 1.0 if mean == 0 else 0.0
    else:
        p_value = 1.0 if standard_error == 0 and mean == 0 else _p_value_two_tailed(t_stat, df)
    t_stat_text = "inf" if math.isinf(t_stat) else f"{t_stat:.4f}"
    return tx_count_lines + [
        f"{prefix}measurement_count = {count:,}",
        f"{prefix}side = {side}",
        f"{prefix}average error = {_sci(mean)}",
        f"{prefix}standard deviation = {_sci(stddev)}",
        f"{prefix}sqrt(n) = sqrt({count:,}) = {sqrt_n:.3f}",
        f"{prefix}standard error = standard deviation / sqrt(n)",
        f"{prefix}= {_sci(stddev)} / {sqrt_n:.3f} = {_sci(standard_error)}",
        f"{prefix}t-statistic = {_sci(mean)} / {_sci(standard_error)} = {t_stat_text}",
        f"{prefix}degrees of freedom = {df:,}",
        f"{prefix}two-tailed p-value = {p_value:.4f}",
        f"{prefix}result: {'reject H0' if p_value < 0.05 else 'fail to reject H0'}",
    ]


def _group_measurement(
    aggregate: dict[str, Any],
    group_name: str,
    side: str,
    error_key: str,
) -> tuple[int, dict[str, Any]]:
    groups = aggregate.get("endpoint_ttest_groups") or {}
    group = groups.get(group_name) or {}
    measurements = group.get("measurements") or {}
    side_payload = measurements.get(side) or {}
    return int(group.get("tx_count") or 0), side_payload.get(error_key) or {"count": 0}


def _strict_group_ttest_lines(aggregate: dict[str, Any], *, signed: bool) -> list[str]:
    error_key = "signed_relative_error" if signed else "unsigned_relative_error"
    title = "Signed Relative Error t-Test:" if signed else "Unsigned Relative Error t-Test:"
    description = (
        "Signed relative error tests endpoint residual directional bias."
        if signed
        else "Unsigned relative error measures endpoint residual magnitude."
    )
    formula = (
        "signed_error = ((pre_state ± model_amount) - post_state) / |real_amount|"
        if signed
        else "unsigned_error = |(pre_state ± model_amount) - post_state| / |real_amount|"
    )
    h0 = "H0: mean_signed_error = 0" if signed else "H0: mean_unsigned_error = 0"
    lines = ["", title, description, h0, formula, "alpha = 0.05", ""]

    tx_count, payload = _group_measurement(aggregate, "input_predicted", "in", error_key)
    lines.append("Input-predicted group:")
    lines.extend(_measurement_stats_lines(payload, prefix="", tx_count=tx_count, side="input"))
    lines.append("")

    tx_count, payload = _group_measurement(aggregate, "output_predicted", "out", error_key)
    lines.append("Output-predicted group:")
    lines.extend(_measurement_stats_lines(payload, prefix="", tx_count=tx_count, side="output"))
    lines.append("")

    tx_count, input_payload = _group_measurement(aggregate, "both_computed", "in", error_key)
    if tx_count == 0:
        tx_count, input_payload = _group_measurement(aggregate, "both_computed_partial", "in", error_key)
    _, output_payload = _group_measurement(aggregate, "both_computed", "out", error_key)
    if not output_payload.get("count"):
        _, output_payload = _group_measurement(aggregate, "both_computed_partial", "out", error_key)
    lines.append("Input/output both-computed group:")
    lines.append(f"tx_count = {tx_count:,}")
    lines.extend(
        _measurement_stats_lines(
            input_payload,
            prefix="input ",
            tx_count=tx_count,
            side="input",
            include_tx_count=False,
        )
    )
    lines.append("")
    lines.extend(
        _measurement_stats_lines(
            output_payload,
            prefix="output ",
            tx_count=tx_count,
            side="output",
            include_tx_count=False,
        )
    )
    return lines


def format_report(aggregate: dict[str, Any], *, title: str | None, updated: str) -> str:
    sample_count = int(aggregate["sample_count"])
    strict_count = int(aggregate["strict_exact_match_count"])
    structure_match_count = int(aggregate["structure_match_count"])
    structure_false_count = int(aggregate["structure_false_count"])

    lines: list[str] = []
    if title:
        lines.extend([f"# {title}", ""])

    lines.extend(
        [
            "Error Analysis:",
            f"- Strict match: {strict_count:,} / {sample_count:,}, {_percent(strict_count, sample_count)}",
            f"- Structure match: {structure_match_count:,} / {sample_count:,}, {_percent(structure_match_count, sample_count)}",
            f"- Structure mismatch: {structure_false_count:,} / {sample_count:,}, {_percent(structure_false_count, sample_count)}",
        ]
    )
    lines.extend(
        _bucket_lines(
            "Relative error among structure-matched txs",
            aggregate["structure_matched_max_side_endpoint_rel_err_buckets"],
            structure_match_count,
        )
    )
    lines.extend(
        _bucket_lines(
            "Relative error by input side among structure-matched txs",
            aggregate["structure_matched_in_endpoint_rel_err_buckets"],
            structure_match_count,
        )
    )
    lines.extend(
        _bucket_lines(
            "Relative error by output side among structure-matched txs",
            aggregate["structure_matched_out_endpoint_rel_err_buckets"],
            structure_match_count,
        )
    )
    groups = aggregate.get("endpoint_ttest_groups") or {}
    if groups:
        input_count = int((groups.get("input_predicted") or {}).get("tx_count") or 0)
        output_count = int((groups.get("output_predicted") or {}).get("tx_count") or 0)
        both_count = int(
            (groups.get("both_computed") or groups.get("both_computed_partial") or {}).get("tx_count") or 0
        )
        lines.extend(
            [
                "",
                "Computation Groups:",
                f"Input-predicted txs: {input_count:,}",
                f"Output-predicted txs: {output_count:,}",
                f"Input/output both-computed txs: {both_count:,}",
            ]
        )
        lines.extend(_strict_group_ttest_lines(aggregate, signed=False))
        lines.extend(_strict_group_ttest_lines(aggregate, signed=True))
    lines.extend(["", f"Updated: {updated}"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title")
    parser.add_argument(
        "--updated",
        default=date.today().strftime("%-d %B, %Y"),
        help="Updated date text to print at the end of the report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    report = format_report(aggregate, title=args.title, updated=args.updated)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
