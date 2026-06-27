#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import NormalDist
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise per-pair baseline replay aggregates for paper-data checks."
    )
    parser.add_argument("--pair-config", required=True, help="Resolved paper pair config JSON")
    parser.add_argument("--compare-root", default="artifacts/compare")
    parser.add_argument("--fit-name", default="final_fit")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _dec(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _pct(num: int, den: int) -> str:
    return f"{(Decimal(num) / Decimal(den) * Decimal(100)).quantize(Decimal('0.0001'))}" if den else "0"


def _p_value_two_tailed(t_stat: float, df: int) -> float:
    if not math.isfinite(t_stat):
        return 0.0
    try:
        from scipy import stats  # type: ignore

        return float(2 * stats.t.sf(abs(t_stat), df))
    except Exception:
        return float(2 * (1 - NormalDist().cdf(abs(t_stat))))


def _ttest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    count = int(payload.get("count") or 0)
    mean = payload.get("mean")
    if count < 1 or mean is None:
        return {"count": count, "p_value": None, "mean": None, "sample_standard_deviation": None}
    mean_f = float(mean)
    stddev_f = float(payload.get("sample_standard_deviation") or 0.0)
    if count < 2:
        p_value = 1.0 if mean_f == 0 else 0.0
    else:
        standard_error = stddev_f / math.sqrt(count) if count else 0.0
        if standard_error == 0:
            p_value = 1.0 if mean_f == 0 else 0.0
        else:
            p_value = _p_value_two_tailed(mean_f / standard_error, count - 1)
    return {
        "count": count,
        "p_value": p_value,
        "mean": mean,
        "sample_standard_deviation": payload.get("sample_standard_deviation"),
    }


def _group_ttests(aggregate: dict[str, Any]) -> dict[str, Any]:
    groups = aggregate.get("endpoint_ttest_groups") or {}
    out: dict[str, Any] = {}
    for group_name, group in sorted(groups.items()):
        measurements = group.get("measurements") or {}
        out[group_name] = {
            "tx_count": int(group.get("tx_count") or 0),
            "measurements": {
                side: {
                    error_kind: _ttest_payload(payload or {})
                    for error_kind, payload in sorted((side_payload or {}).items())
                    if error_kind in {"signed_relative_error", "unsigned_relative_error"}
                }
                for side, side_payload in sorted(measurements.items())
            },
        }
    return out


def _amount_only_errors(aggregate: dict[str, Any], expected_count: int) -> dict[str, Any]:
    values: list[Decimal] = []
    for entry in aggregate.get("top_relative_errors") or []:
        if not bool(entry.get("structure_match", True)):
            continue
        value = _dec(entry.get("max_side_endpoint_unsigned_rel_err"))
        if value is not None:
            values.append(abs(value))
    values.sort()
    complete = len(values) >= expected_count
    selected = values[:expected_count] if complete else values
    return {
        "count": expected_count,
        "values_available_from_top_relative_errors": len(values),
        "complete": complete,
        "min": selected[0].to_eng_string() if selected else None,
        "median": _median(selected).to_eng_string() if selected else None,
        "max": selected[-1].to_eng_string() if selected else None,
    }


def _median(values: list[Decimal]) -> Decimal:
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / Decimal(2)


def _load_pair_config(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("pairs") or [])


def _pair_summary(compare_root: Path, fit_name: str, row: dict[str, Any]) -> dict[str, Any]:
    pair = str(row["pair"])
    window = str(row["window"])
    aggregate_path = compare_root / pair / window / fit_name / "aggregate.json"
    if not aggregate_path.exists():
        raise FileNotFoundError(aggregate_path)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    sample_count = int(aggregate.get("sample_count") or 0)
    strict_count = int(aggregate.get("strict_exact_match_count") or 0)
    structure_count = int(aggregate.get("structure_match_count") or 0)
    structure_false_count = int(aggregate.get("structure_false_count") or 0)
    amount_only_count = int(aggregate.get("strict_false_structure_true_count") or 0)
    return {
        "pair": pair,
        "pair_label": str(row.get("pair_label") or pair),
        "window": window,
        "aggregate_path": str(aggregate_path),
        "transactions": sample_count,
        "fully_fitted": strict_count,
        "fully_fitted_rate_pct": _pct(strict_count, sample_count),
        "amount_only_error": _amount_only_errors(aggregate, amount_only_count),
        "structure_error": structure_false_count,
        "structure_matched": structure_count,
        "endpoint_ttests": _group_ttests(aggregate),
    }


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Baseline Fit Summary",
        "",
        "| Pair | Transactions | Fully fitted | Amount-only | Structure error |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pair_label'].replace('_', '/')} | {row['transactions']:,} | "
            f"{row['fully_fitted']:,} ({row['fully_fitted_rate_pct']}%) | "
            f"{row['amount_only_error']['count']:,} | {row['structure_error']:,} |"
        )
    lines.append("")
    lines.append(
        "Amount-only min/median/max are available when `top_relative_errors` covers all amount-only rows."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    pairs = _load_pair_config(Path(args.pair_config))
    rows = [_pair_summary(Path(args.compare_root), args.fit_name, row) for row in pairs]
    total_tx = sum(int(row["transactions"]) for row in rows)
    total_strict = sum(int(row["fully_fitted"]) for row in rows)
    total_amount = sum(int(row["amount_only_error"]["count"]) for row in rows)
    total_structure = sum(int(row["structure_error"]) for row in rows)
    payload = {
        "pair_config": str(args.pair_config),
        "compare_root": str(args.compare_root),
        "fit_name": str(args.fit_name),
        "summary": {
            "transactions": total_tx,
            "fully_fitted": total_strict,
            "fully_fitted_rate_pct": _pct(total_strict, total_tx),
            "amount_only_error": total_amount,
            "structure_error": total_structure,
        },
        "pairs": rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_fit_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(output_dir / "baseline_fit_summary.md", rows)
    print(json.dumps({"output_json": str(output_dir / "baseline_fit_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
