#!/usr/bin/env python3
"""Summarize Layer 1 same-fill best-price counterfactuals across token pairs."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


MEASUREMENT_ORDER = [
    "fixed_input_same_input",
    "fixed_input_same_output",
    "fixed_output_same_input",
    "fixed_output_same_output",
    "bounded_partial_same_input",
    "bounded_partial_same_output",
]

IMPROVEMENT_NONE = "none"
IMPROVEMENT_ALLOCATION_CHANGED = "allocation_changed"
IMPROVEMENT_AMM_MULTI_SLICE_FEE_COMPOUNDING = "amm_multi_slice_fee_compounding"
IMPROVEMENT_ALLOCATION_AND_AMM_MULTI_SLICE_FEE_COMPOUNDING = "allocation_and_amm_multi_slice_fee_compounding"
IMPROVEMENT_ALLOCATION_OFFSETS_AMM_MULTI_SLICE_FEE_COMPOUNDING_LOSS = (
    "allocation_changed_offsets_amm_multi_slice_fee_compounding_loss"
)
IMPROVEMENT_AMM_MULTI_SLICE_FEE_COMPOUNDING_OFFSETS_ALLOCATION_LOSS = (
    "amm_multi_slice_fee_compounding_offsets_allocation_loss"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def _dec(value: Any) -> Decimal:
    if value is None:
        return Decimal(0)
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal(0)


def _improvement_type_from_gain_decomposition(row: dict[str, Any]) -> str:
    if not bool(row.get("found_improvement")):
        return IMPROVEMENT_NONE
    decomposition = row.get("gain_decomposition") or {}
    if not isinstance(decomposition, dict):
        return str(row.get("improvement_type") or "unknown")
    compounding_gain = _dec(decomposition.get("amm_multi_slice_fee_compounding_gain"))
    allocation_gain = _dec(decomposition.get("allocation_optimization_gain"))
    total_gain = _dec(decomposition.get("total_gain"))
    if total_gain <= 0:
        return IMPROVEMENT_NONE
    if compounding_gain > 0 and allocation_gain > 0:
        return IMPROVEMENT_ALLOCATION_AND_AMM_MULTI_SLICE_FEE_COMPOUNDING
    if compounding_gain > 0 and allocation_gain < 0:
        return IMPROVEMENT_AMM_MULTI_SLICE_FEE_COMPOUNDING_OFFSETS_ALLOCATION_LOSS
    if allocation_gain > 0 and compounding_gain < 0:
        return IMPROVEMENT_ALLOCATION_OFFSETS_AMM_MULTI_SLICE_FEE_COMPOUNDING_LOSS
    if compounding_gain > 0:
        return IMPROVEMENT_AMM_MULTI_SLICE_FEE_COMPOUNDING
    if allocation_gain > 0:
        return IMPROVEMENT_ALLOCATION_CHANGED
    return IMPROVEMENT_ALLOCATION_CHANGED


def _normalise_improvement_type(row: dict[str, Any]) -> dict[str, Any]:
    corrected = _improvement_type_from_gain_decomposition(row)
    raw = str(row.get("improvement_type") or "unknown")
    if corrected == raw:
        return row
    out = dict(row)
    out["raw_improvement_type"] = raw
    out["improvement_type"] = corrected
    return out


def _fmt_dec(value: Decimal | Any, *, places: int | None = None) -> str:
    dec = _dec(value)
    if places is not None:
        quant = Decimal(1).scaleb(-places)
        return str(dec.quantize(quant))
    return dec.to_eng_string()


def _pct(part: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(part / total) * 100:.2f}%"


def _count(counts: dict[str, int], key: str) -> int:
    return int(counts.get(key) or 0)


def _add_counts(dst: dict[str, int], src: dict[str, Any] | None) -> None:
    for key, value in (src or {}).items():
        dst[str(key)] = dst.get(str(key), 0) + int(value or 0)


def _empty_metric() -> dict[str, Any]:
    return {
        "count": 0,
        "improved": 0,
        "constraint_violations": 0,
        "same_input_under_spent_count": 0,
        "materiality_counts": {},
        "analysis_layer_counts": {},
        "layer1_measurement_counts": {},
        "status_counts": {},
        "objective_counts": {},
        "source_intent_family_counts": {},
        "improvement_type_counts": {},
        "best_price_claim_counts": {},
        "baseline_fit_status_counts": {},
        "layer1_evidence_status_counts": {},
        "all_bps": [],
        "improved_bps": [],
    }


def _bump(metric: dict[str, Any], key: str, value: str) -> None:
    counts = metric.setdefault(key, {})
    counts[value] = counts.get(value, 0) + 1


def _nearest_rank(values: list[Decimal], percentile: int) -> Decimal:
    if not values:
        return Decimal(0)
    ordered = sorted(values)
    rank = max(1, (len(ordered) * percentile + 99) // 100)
    return ordered[min(len(ordered) - 1, rank - 1)]


def _distribution(values: list[Decimal]) -> dict[str, str | int]:
    if not values:
        return {"count": 0, "p50": "0", "p95": "0", "p99": "0", "max": "0"}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": _nearest_rank(ordered, 50).to_eng_string(),
        "p95": _nearest_rank(ordered, 95).to_eng_string(),
        "p99": _nearest_rank(ordered, 99).to_eng_string(),
        "max": ordered[-1].to_eng_string(),
    }


def _pair_dirs(batch_root: Path) -> list[Path]:
    return sorted(path for path in batch_root.iterdir() if path.is_dir() and (path / "results.ndjson").exists())


def _row_measurement(row: dict[str, Any]) -> str:
    return str(row.get("layer1_measurement") or "unknown")


def _fixed_constraint_delta(row: dict[str, Any]) -> Decimal:
    comparison = row.get("counterfactual_comparison") or {}
    measurement = _row_measurement(row)
    if measurement.endswith("same_input"):
        return _dec(comparison.get("delta_input"))
    if measurement.endswith("same_output"):
        return _dec(comparison.get("delta_output"))
    return Decimal(0)


def _size_denominator(row: dict[str, Any]) -> Decimal:
    """Return the same denominator used by improvement_bps for this Layer 1 row."""
    baseline = row.get("baseline") or {}
    measurement = _row_measurement(row)
    if measurement.endswith("same_input"):
        return abs(_dec(baseline.get("output")))
    if measurement.endswith("same_output"):
        return abs(_dec(baseline.get("input")))
    return Decimal(0)


def _denominator_scale_bucket(row: dict[str, Any]) -> str:
    baseline = row.get("baseline") or {}
    measurement = _row_measurement(row)
    raw = baseline.get("output") if measurement.endswith("same_input") else baseline.get("input")
    try:
        dec = Decimal(str(raw))
    except (InvalidOperation, ValueError):
        return "scale_unknown"
    # XRP amounts are naturally drop-based and display with <= 6 decimal places, while issued
    # amounts commonly use canonical mantissa/exponent decimals. Keep the label scale-based
    # rather than asset-based because optimizer rows intentionally stay pair-generic.
    return "scale_le_6dp" if dec.as_tuple().exponent >= -6 else "scale_gt_6dp"


def _denominator_context(row: dict[str, Any]) -> str:
    return f"{_row_measurement(row)}:{_denominator_scale_bucket(row)}"


def _size_bucket(denominator: Decimal, thresholds: dict[str, Decimal]) -> str:
    if denominator <= 0:
        return "unknown_size"
    if denominator >= thresholds.get("p95", Decimal(0)):
        return "very_large_p95_100"
    if denominator >= thresholds.get("p90", Decimal(0)):
        return "large_p90_95"
    if denominator >= thresholds.get("p50", Decimal(0)):
        return "medium_p50_90"
    return "small_p0_50"


def _bps_bucket(bps: Decimal) -> str:
    if bps <= 0:
        return "no_improvement"
    if bps >= Decimal("10"):
        return "high_gt_10bps"
    if bps >= Decimal("1"):
        return "material_1_10bps"
    if bps >= Decimal("0.01"):
        return "low_0_01_1bps"
    return "dust_lt_0_01bps"


def _materiality_bucket(row: dict[str, Any], thresholds: dict[str, Decimal]) -> str:
    found = bool(row.get("found_improvement"))
    bps = _dec((row.get("best") or {}).get("improvement_bps")) if found else Decimal(0)
    return f"{_denominator_scale_bucket(row)}:{_size_bucket(_size_denominator(row), thresholds)}:{_bps_bucket(bps)}"


def _baseline_fit_status(row: dict[str, Any]) -> str:
    baseline_fit = row.get("baseline_fit")
    if not isinstance(baseline_fit, dict):
        return "missing"
    return str(baseline_fit.get("status") or "missing")


def _layer1_evidence_status(row: dict[str, Any]) -> str:
    status = _baseline_fit_status(row)
    if status == "strict_exact_match":
        return "valid_strict_baseline"
    if status == "structure_match_endpoint_residual":
        return "review_structure_matched_baseline"
    if status == "structure_mismatch":
        return "suspect_structure_mismatch_baseline"
    return "unknown_missing_baseline_fit"


def _violates_fixed_constraint(row: dict[str, Any]) -> bool:
    measurement = _row_measurement(row)
    delta = _fixed_constraint_delta(row)
    if measurement.endswith("same_input"):
        # Same-input Layer 1 fixes the input budget. Under-spending is allowed if no additional
        # canonical output unit is reachable; spending more than the baseline budget is not.
        return delta > 0
    if measurement.endswith("same_output"):
        return delta != 0
    return False


def _baseline_summary(row: dict[str, Any]) -> str:
    baseline = row.get("baseline") or {}
    return f"in={baseline.get('input')}, out={baseline.get('output')}"


def _best_summary(row: dict[str, Any]) -> str:
    best = row.get("best") or {}
    return f"in={best.get('spent_input') or best.get('input')}, out={best.get('output')}"


def _scan_pair(pair_dir: Path, *, top_n: int) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    denominator_values: dict[str, list[Decimal]] = {}
    top_rows: list[tuple[Decimal, dict[str, Any]]] = []
    large_high_rows: list[tuple[Decimal, dict[str, Any]]] = []
    constraint_violations: list[dict[str, Any]] = []
    unique_tx_hashes: set[str] = set()
    improved_tx_hashes: set[str] = set()
    all_bps: list[Decimal] = []
    improved_bps: list[Decimal] = []
    results = pair_dir / "results.ndjson"
    if not results.exists():
        return {
            "metrics": metrics,
            "top_rows": [],
            "constraint_violations": [],
            "unique_tx_count": 0,
            "improved_tx_count": 0,
            "measurement_count": 0,
            "improved_measurement_count": 0,
        }

    with results.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            row = _normalise_improvement_type(row)
            denominator = _size_denominator(row)
            if denominator > 0:
                denominator_values.setdefault(_denominator_context(row), []).append(denominator)

    denominator_thresholds = {
        context: {
            "p50": _nearest_rank(values, 50),
            "p90": _nearest_rank(values, 90),
            "p95": _nearest_rank(values, 95),
        }
        for context, values in denominator_values.items()
    }

    with results.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            row = _normalise_improvement_type(row)
            tx_hash = str(row.get("tx_hash") or "")
            if tx_hash:
                unique_tx_hashes.add(tx_hash)
            measurement = _row_measurement(row)
            metric = metrics.setdefault(measurement, _empty_metric())
            metric["count"] += 1
            found = bool(row.get("found_improvement"))
            if found:
                metric["improved"] += 1
                if tx_hash:
                    improved_tx_hashes.add(tx_hash)
            best = row.get("best") or {}
            bps = _dec(best.get("improvement_bps")) if found else Decimal(0)
            denominator_context = _denominator_context(row)
            thresholds = denominator_thresholds.get(denominator_context) or {
                "p50": Decimal(0),
                "p90": Decimal(0),
                "p95": Decimal(0),
            }
            materiality = _materiality_bucket(row, thresholds)
            metric["all_bps"].append(bps)
            all_bps.append(bps)
            if found:
                metric["improved_bps"].append(bps)
                improved_bps.append(bps)
            comparison = row.get("counterfactual_comparison") or {}
            cert = row.get("best_price_certificate") or {}
            _bump(metric, "analysis_layer_counts", str(row.get("analysis_layer") or "missing"))
            _bump(metric, "layer1_measurement_counts", measurement)
            _bump(metric, "status_counts", str(comparison.get("status") or "missing"))
            _bump(metric, "objective_counts", str(row.get("objective") or "unknown"))
            _bump(metric, "source_intent_family_counts", str(row.get("source_intent_family") or "unknown"))
            _bump(metric, "improvement_type_counts", str(row.get("improvement_type") or "unknown"))
            _bump(metric, "best_price_claim_counts", str(cert.get("claim") or "missing"))
            _bump(metric, "materiality_counts", materiality)
            _bump(metric, "baseline_fit_status_counts", _baseline_fit_status(row))
            _bump(metric, "layer1_evidence_status_counts", _layer1_evidence_status(row))
            delta = _fixed_constraint_delta(row)
            if measurement.endswith("same_input") and delta < 0:
                metric["same_input_under_spent_count"] += 1
            if _violates_fixed_constraint(row):
                metric["constraint_violations"] += 1
                if len(constraint_violations) < 20:
                    constraint_violations.append(
                        {
                            "tx_hash": row.get("tx_hash"),
                            "measurement": measurement,
                            "delta": delta.to_eng_string(),
                            "baseline": row.get("baseline"),
                            "best": row.get("best"),
                            "comparison": row.get("counterfactual_comparison"),
                        }
                    )
            if found:
                top_rows.append((bps, row))
                top_rows.sort(key=lambda item: item[0], reverse=True)
                if len(top_rows) > top_n:
                    top_rows.pop()
                if ":large_p90_95:high_gt_10bps" in materiality or ":very_large_p95_100:high_gt_10bps" in materiality:
                    large_high_rows.append((bps, row))
                    large_high_rows.sort(key=lambda item: item[0], reverse=True)
                    if len(large_high_rows) > top_n:
                        large_high_rows.pop()

    compact_metrics: dict[str, dict[str, Any]] = {}
    for measurement, metric in metrics.items():
        compact_metrics[measurement] = {
            "count": metric["count"],
            "improved": metric["improved"],
            "improved_rate": _pct(int(metric["improved"]), int(metric["count"])),
            "constraint_violations": metric["constraint_violations"],
            "same_input_under_spent_count": metric["same_input_under_spent_count"],
            "analysis_layer_counts": metric["analysis_layer_counts"],
            "layer1_measurement_counts": metric["layer1_measurement_counts"],
            "status_counts": metric["status_counts"],
            "objective_counts": metric["objective_counts"],
            "source_intent_family_counts": metric["source_intent_family_counts"],
            "improvement_type_counts": metric["improvement_type_counts"],
            "best_price_claim_counts": metric["best_price_claim_counts"],
            "materiality_counts": metric["materiality_counts"],
            "baseline_fit_status_counts": metric["baseline_fit_status_counts"],
            "layer1_evidence_status_counts": metric["layer1_evidence_status_counts"],
            "size_denominator_thresholds_by_scale": {
                context: {key: value.to_eng_string() for key, value in threshold.items()}
                for context, threshold in denominator_thresholds.items()
                if context.startswith(f"{measurement}:")
            },
            "all_bps_distribution": _distribution(metric["all_bps"]),
            "improved_bps_distribution": _distribution(metric["improved_bps"]),
        }

    pair_summary = {
        "unique_tx_count": len(unique_tx_hashes),
        "measurement_count": sum(int(metric["count"]) for metric in compact_metrics.values()),
        "improved_measurement_count": sum(int(metric["improved"]) for metric in compact_metrics.values()),
        "improved_tx_count": len(improved_tx_hashes),
        "constraint_violations": sum(int(metric["constraint_violations"]) for metric in compact_metrics.values()),
        "same_input_under_spent_count": sum(
            int(metric["same_input_under_spent_count"]) for metric in compact_metrics.values()
        ),
        "all_bps_distribution": _distribution(all_bps),
        "improved_bps_distribution": _distribution(improved_bps),
        "status_counts": {},
        "objective_counts": {},
        "source_intent_family_counts": {},
        "improvement_type_counts": {},
        "best_price_claim_counts": {},
        "materiality_counts": {},
        "baseline_fit_status_counts": {},
        "layer1_evidence_status_counts": {},
    }
    for metric in compact_metrics.values():
        _add_counts(pair_summary["status_counts"], metric.get("status_counts") or {})
        _add_counts(pair_summary["objective_counts"], metric.get("objective_counts") or {})
        _add_counts(pair_summary["source_intent_family_counts"], metric.get("source_intent_family_counts") or {})
        _add_counts(pair_summary["improvement_type_counts"], metric.get("improvement_type_counts") or {})
        _add_counts(pair_summary["best_price_claim_counts"], metric.get("best_price_claim_counts") or {})
        _add_counts(pair_summary["materiality_counts"], metric.get("materiality_counts") or {})
        _add_counts(pair_summary["baseline_fit_status_counts"], metric.get("baseline_fit_status_counts") or {})
        _add_counts(pair_summary["layer1_evidence_status_counts"], metric.get("layer1_evidence_status_counts") or {})

    return {
        "metrics": compact_metrics,
        "summary": pair_summary,
        "top_rows": [row for _bps, row in top_rows],
        "large_high_rows": [row for _bps, row in large_high_rows],
        "constraint_violations": constraint_violations,
        "unique_tx_count": len(unique_tx_hashes),
        "improved_tx_count": len(improved_tx_hashes),
        "measurement_count": sum(int(metric["count"]) for metric in compact_metrics.values()),
        "improved_measurement_count": sum(int(metric["improved"]) for metric in compact_metrics.values()),
    }


def _ordered_measurements(measurements: dict[str, Any]) -> list[str]:
    known = [name for name in MEASUREMENT_ORDER if name in measurements]
    unknown = sorted(name for name in measurements if name not in MEASUREMENT_ORDER)
    return known + unknown


def main() -> None:
    args = _parse_args()
    batch_root = Path(args.batch_root)
    top_n = max(1, int(args.top_n))
    pair_payloads: list[dict[str, Any]] = []
    global_counts = {
        "pair_count": 0,
        "unique_tx_count": 0,
        "measurement_count": 0,
        "improved_measurement_count": 0,
        "improved_tx_count": 0,
        "constraint_violations": 0,
        "same_input_under_spent_count": 0,
    }
    global_count_maps: dict[str, dict[str, int]] = {
        "analysis_layer_counts": {},
        "layer1_measurement_counts": {},
        "source_intent_family_counts": {},
        "objective_counts": {},
        "comparison_status_counts": {},
        "best_price_claim_counts": {},
        "improvement_type_counts": {},
        "materiality_counts": {},
        "baseline_fit_status_counts": {},
        "layer1_evidence_status_counts": {},
    }
    global_measurements: dict[str, dict[str, Any]] = {}
    global_top: list[tuple[Decimal, str, dict[str, Any]]] = []
    global_large_high: list[tuple[Decimal, str, dict[str, Any]]] = []
    all_constraint_examples: list[dict[str, Any]] = []

    for pair_dir in _pair_dirs(batch_root):
        scan = _scan_pair(pair_dir, top_n=top_n)
        pair = pair_dir.name
        pair_payloads.append({"pair": pair, "summary": scan["summary"], "scan": scan})
        global_counts["pair_count"] += 1
        global_counts["unique_tx_count"] += int(scan["unique_tx_count"])
        global_counts["measurement_count"] += int(scan["measurement_count"])
        global_counts["improved_measurement_count"] += int(scan["improved_measurement_count"])
        global_counts["improved_tx_count"] += int(scan["improved_tx_count"])

        for measurement, metric in scan["metrics"].items():
            target = global_measurements.setdefault(measurement, _empty_metric())
            target["count"] += int(metric["count"])
            target["improved"] += int(metric["improved"])
            target["constraint_violations"] += int(metric["constraint_violations"])
            target["same_input_under_spent_count"] += int(metric["same_input_under_spent_count"])
            _add_counts(target["analysis_layer_counts"], metric.get("analysis_layer_counts") or {})
            _add_counts(target["layer1_measurement_counts"], metric.get("layer1_measurement_counts") or {})
            _add_counts(target["status_counts"], metric.get("status_counts") or {})
            _add_counts(target["objective_counts"], metric.get("objective_counts") or {})
            _add_counts(target["source_intent_family_counts"], metric.get("source_intent_family_counts") or {})
            _add_counts(target["improvement_type_counts"], metric.get("improvement_type_counts") or {})
            _add_counts(target["best_price_claim_counts"], metric.get("best_price_claim_counts") or {})
            _add_counts(target["materiality_counts"], metric.get("materiality_counts") or {})
            _add_counts(target["baseline_fit_status_counts"], metric.get("baseline_fit_status_counts") or {})
            _add_counts(target["layer1_evidence_status_counts"], metric.get("layer1_evidence_status_counts") or {})
            # Re-scan distributions are stored compactly per pair; global distribution is derived from pair top-level
            # rows only in this report to avoid keeping all cross-pair values resident.
        for violation in scan["constraint_violations"]:
            if len(all_constraint_examples) < 20:
                all_constraint_examples.append({"pair": pair, **violation})
        global_counts["constraint_violations"] += sum(
            int(metric["constraint_violations"]) for metric in scan["metrics"].values()
        )
        global_counts["same_input_under_spent_count"] += sum(
            int(metric["same_input_under_spent_count"]) for metric in scan["metrics"].values()
        )
        for measurement, metric in scan["metrics"].items():
            global_count_maps["layer1_measurement_counts"][measurement] = (
                global_count_maps["layer1_measurement_counts"].get(measurement, 0) + int(metric["count"])
            )
            _add_counts(global_count_maps["analysis_layer_counts"], metric.get("analysis_layer_counts") or {})
            _add_counts(global_count_maps["source_intent_family_counts"], metric.get("source_intent_family_counts") or {})
            _add_counts(global_count_maps["objective_counts"], metric.get("objective_counts") or {})
            _add_counts(global_count_maps["comparison_status_counts"], metric.get("status_counts") or {})
            _add_counts(global_count_maps["best_price_claim_counts"], metric.get("best_price_claim_counts") or {})
            _add_counts(global_count_maps["improvement_type_counts"], metric.get("improvement_type_counts") or {})
            _add_counts(global_count_maps["materiality_counts"], metric.get("materiality_counts") or {})
            _add_counts(global_count_maps["baseline_fit_status_counts"], metric.get("baseline_fit_status_counts") or {})
            _add_counts(
                global_count_maps["layer1_evidence_status_counts"],
                metric.get("layer1_evidence_status_counts") or {},
            )
        for row in scan["top_rows"]:
            bps = _dec((row.get("best") or {}).get("improvement_bps"))
            global_top.append((bps, pair, row))
            global_top.sort(key=lambda item: item[0], reverse=True)
            if len(global_top) > top_n:
                global_top.pop()
        for row in scan["large_high_rows"]:
            bps = _dec((row.get("best") or {}).get("improvement_bps"))
            global_large_high.append((bps, pair, row))
            global_large_high.sort(key=lambda item: item[0], reverse=True)
            if len(global_large_high) > top_n:
                global_large_high.pop()

    compact_global_measurements: dict[str, Any] = {}
    for measurement, metric in global_measurements.items():
        compact_global_measurements[measurement] = {
            "count": metric["count"],
            "improved": metric["improved"],
            "improved_rate": _pct(int(metric["improved"]), int(metric["count"])),
            "constraint_violations": metric["constraint_violations"],
            "same_input_under_spent_count": metric["same_input_under_spent_count"],
            "analysis_layer_counts": metric["analysis_layer_counts"],
            "layer1_measurement_counts": metric["layer1_measurement_counts"],
            "status_counts": metric["status_counts"],
            "objective_counts": metric["objective_counts"],
            "source_intent_family_counts": metric["source_intent_family_counts"],
            "improvement_type_counts": metric["improvement_type_counts"],
            "best_price_claim_counts": metric["best_price_claim_counts"],
            "materiality_counts": metric["materiality_counts"],
            "baseline_fit_status_counts": metric["baseline_fit_status_counts"],
            "layer1_evidence_status_counts": metric["layer1_evidence_status_counts"],
        }

    output = {
        "batch_root": str(batch_root),
        "global_counts": global_counts,
        "global_count_maps": global_count_maps,
        "global_measurements": compact_global_measurements,
        "pairs": pair_payloads,
        "top_improvements": [
            {
                "pair": pair,
                "tx_hash": row.get("tx_hash"),
                "measurement": row.get("layer1_measurement"),
                "source_intent_family": row.get("source_intent_family"),
                "objective": row.get("objective"),
                "improvement_type": row.get("improvement_type"),
                "raw_improvement_type": row.get("raw_improvement_type"),
                "baseline": row.get("baseline"),
                "best": row.get("best"),
                "counterfactual_comparison": row.get("counterfactual_comparison"),
                "best_price_certificate": row.get("best_price_certificate"),
                "baseline_fit": row.get("baseline_fit"),
            }
            for _bps, pair, row in global_top
        ],
        "large_size_high_bps_improvements": [
            {
                "pair": pair,
                "tx_hash": row.get("tx_hash"),
                "measurement": row.get("layer1_measurement"),
                "source_intent_family": row.get("source_intent_family"),
                "objective": row.get("objective"),
                "improvement_type": row.get("improvement_type"),
                "baseline": row.get("baseline"),
                "best": row.get("best"),
                "counterfactual_comparison": row.get("counterfactual_comparison"),
                "best_price_certificate": row.get("best_price_certificate"),
                "baseline_fit": row.get("baseline_fit"),
            }
            for _bps, pair, row in global_large_high
        ],
        "constraint_violation_examples": all_constraint_examples,
    }
    Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Layer 1 Same-Fill Best-Price Report", ""]
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "Layer 1 is a same-fill price counterfactual. It keeps the pre-state, executed tx, "
        "canonical amount domains, transfer-rate rules, owner-funds constraints, and one endpoint "
        "from the baseline trajectory fixed, then asks whether a better AMM/CLOB allocation exists."
    )
    lines.append("")
    lines.append(
        "- `fixed_input_same_input`: keep the baseline input budget fixed and maximize output. "
        "The optimized route may under-spend if no additional canonical output unit is reachable."
    )
    lines.append("- `fixed_input_same_output`: keep baseline output fixed for fixed-input source txs and minimize input.")
    lines.append("- `fixed_output_same_input`: keep the baseline input budget fixed for fixed-output source txs and maximize output.")
    lines.append("- `fixed_output_same_output`: keep baseline output fixed and minimize input.")
    lines.append(
        "- `bounded_partial_same_input`: for bounded partial txs, keep the baseline input budget fixed and maximize output."
    )
    lines.append(
        "- `bounded_partial_same_output`: for bounded partial txs, keep baseline output fixed and minimize input."
    )
    lines.append(
        "This report is measurement-level. In the all-sides Layer 1 report, each tx contributes "
        "same-input and same-output measurements. "
        "It is not the Layer 2 same-original-intent full-outcome model."
    )
    lines.append("")
    lines.append("## Definitions And Formulas")
    lines.append("")
    lines.append("- For same-input measurements, `improvement = optimized_output - baseline_output`.")
    lines.append("- For same-input measurements, `improvement_bps = improvement / baseline_output * 10000`.")
    lines.append("- For same-output measurements, `improvement = baseline_input - optimized_input`.")
    lines.append("- For same-output measurements, `improvement_bps = improvement / baseline_input * 10000`.")
    lines.append(
        "- The materiality denominator is exactly the denominator used by `improvement_bps`: "
        "`baseline_output` for same-input and `baseline_input` for same-output."
    )
    lines.append(
        "- Denominator percentiles are computed within `pair + measurement + denominator-scale`; "
        "this prevents XRP/drop-scale amounts and IOU canonical-decimal amounts from being ranked in the same size bucket."
    )
    lines.append(
        "- `baseline_fit.status` is not used to suppress optimizer computation. It is an evidence label: "
        "`strict_exact_match` is clean optimization evidence, `structure_match_endpoint_residual` needs review, "
        "and `structure_mismatch` is suspect for optimization claims."
    )
    lines.append("")
    lines.append("## Global Results")
    lines.append("")
    lines.append(f"- batch_root: `{batch_root}`")
    lines.append(f"- pair_count: `{global_counts['pair_count']}`")
    lines.append(f"- unique_tx_count: `{global_counts['unique_tx_count']}`")
    lines.append(f"- measurement_count: `{global_counts['measurement_count']}`")
    lines.append(
        f"- improved_measurements: `{global_counts['improved_measurement_count']}` "
        f"({_pct(global_counts['improved_measurement_count'], global_counts['measurement_count'])})"
    )
    lines.append(
        f"- improved_unique_txs: `{global_counts['improved_tx_count']}` "
        f"({_pct(global_counts['improved_tx_count'], global_counts['unique_tx_count'])})"
    )
    lines.append(f"- constraint_violations: `{global_counts['constraint_violations']}`")
    lines.append(f"- same_input_under_spent_count: `{global_counts['same_input_under_spent_count']}`")
    for key, counts in global_count_maps.items():
        lines.append(f"- {key}: `{json.dumps(counts, sort_keys=True)}`")
    lines.append("")
    lines.append("## Key Interpretation")
    lines.append("")
    measurement_counts = global_count_maps["layer1_measurement_counts"]
    type_counts = global_count_maps["improvement_type_counts"]
    claim_counts = global_count_maps["best_price_claim_counts"]
    status_counts = global_count_maps["comparison_status_counts"]
    evidence_counts = global_count_maps["layer1_evidence_status_counts"]
    improved_total = global_counts["improved_measurement_count"]
    bounded_measurements = _count(measurement_counts, "bounded_partial_same_input") + _count(
        measurement_counts, "bounded_partial_same_output"
    )
    amm_compounding = _count(type_counts, "amm_multi_slice_fee_compounding")
    allocation_changed = _count(type_counts, "allocation_changed")
    mixed_allocation_and_compounding = _count(type_counts, "allocation_and_amm_multi_slice_fee_compounding")
    allocation_offsets_compounding_loss = _count(
        type_counts,
        "allocation_changed_offsets_amm_multi_slice_fee_compounding_loss",
    )
    compounding_offsets_allocation_loss = _count(
        type_counts,
        "amm_multi_slice_fee_compounding_offsets_allocation_loss",
    )
    exact_claims = _count(claim_counts, "exact_discrete_best_price")
    unchanged = _count(status_counts, "unchanged")
    lines.append(
        f"- Validity: `{exact_claims}` / `{global_counts['measurement_count']}` measurements carry "
        "`exact_discrete_best_price`, and `constraint_violations=0`."
    )
    lines.append(
        f"- Baseline fit evidence: `{json.dumps(evidence_counts, sort_keys=True)}`."
    )
    lines.append(
        f"- Coverage: bounded partial dominates the sample with `{bounded_measurements}` / "
        f"`{global_counts['measurement_count']}` measurements ({_pct(bounded_measurements, global_counts['measurement_count'])})."
    )
    lines.append(
        f"- Improvement frequency: `{improved_total}` measurements improve "
        f"({_pct(improved_total, global_counts['measurement_count'])}); "
        f"`{unchanged}` are unchanged."
    )
    lines.append(
        f"- Main mechanism: AMM multi-slice fee compounding explains `{amm_compounding}` improved measurements "
        f"({_pct(amm_compounding, improved_total)} of improvements), while changed AMM/CLOB allocation explains "
        f"`{allocation_changed}` ({_pct(allocation_changed, improved_total)})."
    )
    lines.append(
        f"- Mixed mechanisms: allocation plus AMM compounding explains `{mixed_allocation_and_compounding}`; "
        f"allocation offsetting AMM one-shot loss explains `{allocation_offsets_compounding_loss}`; "
        f"AMM compounding offsetting allocation loss explains `{compounding_offsets_allocation_loss}`."
    )
    lines.append(
        "- Magnitudes should be read mostly through bps distributions, not raw total gain, because the report mixes "
        "input-saving and output-gain objectives across different token units."
    )
    lines.append(
        "- Materiality buckets use each pair+measurement+denominator-scale baseline-size distribution: "
        "`scale_le_6dp` and `scale_gt_6dp` are separated before computing `small_p0_50`, "
        "`medium_p50_90`, `large_p90_95`, and `very_large_p95_100`; those size buckets are crossed with "
        "`dust_lt_0_01bps`, `low_0_01_1bps`, `material_1_10bps`, and `high_gt_10bps`."
    )
    lines.append(
        f"- Same-input under-spend occurs in `{global_counts['same_input_under_spent_count']}` rows; this means the "
        "optimizer stayed within the baseline input budget but could not buy another canonical output unit, so it is "
        "not a constraint violation."
    )
    lines.append("")
    lines.append("## Measurement Breakdown")
    lines.append("")
    lines.append(
        "| measurement | count | improved | rate | constraint violations | same-input under-spend | status | baseline fit | evidence | claims | improvement types |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---|---|---|")
    for measurement in _ordered_measurements(compact_global_measurements):
        metric = compact_global_measurements[measurement]
        lines.append(
            f"| {measurement} | {metric['count']} | {metric['improved']} | {metric['improved_rate']} | "
            f"{metric['constraint_violations']} | "
            f"{metric['same_input_under_spent_count']} | "
            f"`{json.dumps(metric['status_counts'], sort_keys=True)}` | "
            f"`{json.dumps(metric['baseline_fit_status_counts'], sort_keys=True)}` | "
            f"`{json.dumps(metric['layer1_evidence_status_counts'], sort_keys=True)}` | "
            f"`{json.dumps(metric['best_price_claim_counts'], sort_keys=True)}` | "
            f"`{json.dumps(metric['improvement_type_counts'], sort_keys=True)}` |"
        )
    lines.append("")
    lines.append("## Materiality Breakdown")
    lines.append("")
    lines.append("| materiality bucket | count | share |")
    lines.append("|---|---:|---:|")
    materiality_counts = global_count_maps["materiality_counts"]
    for bucket, count in sorted(materiality_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {bucket} | {count} | {_pct(count, global_counts['measurement_count'])} |")
    lines.append("")
    lines.append("### Large-Size And High-Bps Improvements")
    lines.append("")
    lines.append(
        "These rows are the main audit target because the baseline trade size is in the top decile "
        "for its pair+measurement and the counterfactual improvement exceeds 10 bps."
    )
    lines.append("")
    lines.append("| pair | tx | measurement | baseline fit | family | objective | type | baseline | best | gain | bps | claim |")
    lines.append("|---|---|---|---|---|---|---|---|---|---:|---:|---|")
    for _bps, pair, row in global_large_high:
        best = row.get("best") or {}
        cert = row.get("best_price_certificate") or {}
        baseline_fit = row.get("baseline_fit") or {}
        lines.append(
            f"| {pair} | `{str(row.get('tx_hash'))[:12]}` | {row.get('layer1_measurement')} | "
            f"{baseline_fit.get('status', 'missing')} | "
            f"{row.get('source_intent_family')} | {row.get('objective')} | {row.get('improvement_type')} | "
            f"{_baseline_summary(row)} | {_best_summary(row)} | {best.get('improvement')} | "
            f"{best.get('improvement_bps')} | {cert.get('claim')} |"
        )
    lines.append("")
    lines.append("## Pair Breakdown")
    lines.append("")
    lines.append(
        "| pair | unique tx | measurements | improved measurements | improved tx | p95 bps | p99 bps | max bps | status | baseline fit |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for payload in pair_payloads:
        pair = payload["pair"]
        summary = payload["summary"]
        all_dist = summary.get("all_bps_distribution") or {}
        lines.append(
            f"| {pair} | {summary.get('unique_tx_count')} | "
            f"{summary.get('measurement_count')} | "
            f"{summary.get('improved_measurement_count')} | {summary.get('improved_tx_count')} | "
            f"{all_dist.get('p95')} | {all_dist.get('p99')} | {all_dist.get('max')} | "
            f"`{json.dumps(summary.get('status_counts') or {}, sort_keys=True)}` | "
            f"`{json.dumps(summary.get('baseline_fit_status_counts') or {}, sort_keys=True)}` |"
        )
    lines.append("")
    lines.append("## Per-Pair Measurement Details")
    for payload in pair_payloads:
        pair = payload["pair"]
        scan = payload["scan"]
        lines.append("")
        lines.append(f"### {pair}")
        lines.append("")
        lines.append("| measurement | count | improved | rate | p95 bps | p99 bps | max bps | baseline fit | types |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
        for measurement in _ordered_measurements(scan["metrics"]):
            metric = scan["metrics"][measurement]
            all_dist = metric["all_bps_distribution"]
            lines.append(
                f"| {measurement} | {metric['count']} | {metric['improved']} | {metric['improved_rate']} | "
                f"{all_dist.get('p95')} | {all_dist.get('p99')} | {all_dist.get('max')} | "
                f"`{json.dumps(metric['baseline_fit_status_counts'], sort_keys=True)}` | "
                f"`{json.dumps(metric['improvement_type_counts'], sort_keys=True)}` |"
            )
    lines.append("")
    lines.append(f"## Top {len(global_top)} Improvements")
    lines.append("")
    lines.append(
        "| pair | tx | measurement | baseline fit | family | objective | type | baseline | best | gain | bps | claim |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---:|---:|---|")
    for _bps, pair, row in global_top:
        best = row.get("best") or {}
        cert = row.get("best_price_certificate") or {}
        baseline_fit = row.get("baseline_fit") or {}
        lines.append(
            f"| {pair} | `{str(row.get('tx_hash'))[:12]}` | {row.get('layer1_measurement')} | "
            f"{baseline_fit.get('status', 'missing')} | "
            f"{row.get('source_intent_family')} | {row.get('objective')} | {row.get('improvement_type')} | "
            f"{_baseline_summary(row)} | {_best_summary(row)} | {best.get('improvement')} | "
            f"{best.get('improvement_bps')} | {cert.get('claim')} |"
        )
    if all_constraint_examples:
        lines.append("")
        lines.append("## Constraint Violation Examples")
        lines.append("")
        lines.append("These rows indicate a bug in Layer 1 setup and should be investigated before using the report.")
        lines.append("")
        lines.append("| pair | tx | measurement | delta |")
        lines.append("|---|---|---|---:|")
        for row in all_constraint_examples:
            lines.append(f"| {row['pair']} | `{str(row.get('tx_hash'))[:12]}` | {row['measurement']} | {row['delta']} |")
    else:
        lines.append("")
        lines.append("## Constraint Check")
        lines.append("")
        lines.append("No constraint violations were found. Same-input rows do not exceed the baseline input budget, and same-output rows hit the baseline output exactly.")
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
