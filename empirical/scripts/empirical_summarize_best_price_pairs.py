#!/usr/bin/env python3
"""Summarize full best-price optimizer outputs across token pairs."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def _dec(value: Any) -> Decimal:
    return Decimal(str(value or "0"))


def _add_counts(dst: dict[str, int], src: dict[str, Any]) -> None:
    for key, value in (src or {}).items():
        dst[str(key)] = dst.get(str(key), 0) + int(value or 0)


def _add_decimals(dst: dict[str, Decimal], src: dict[str, Any]) -> None:
    for key, value in (src or {}).items():
        dst[str(key)] = dst.get(str(key), Decimal(0)) + _dec(value)


def _max_decimals(dst: dict[str, Decimal], src: dict[str, Any]) -> None:
    for key, value in (src or {}).items():
        dst[str(key)] = max(dst.get(str(key), Decimal(0)), _dec(value))


def _nearest_rank(values: list[Decimal], percentile: int) -> Decimal:
    if not values:
        return Decimal(0)
    rank = max(1, (len(values) * int(percentile) + 99) // 100)
    return values[min(len(values) - 1, rank - 1)]


def _bps_distribution(values: list[Decimal]) -> dict[str, str | int]:
    values = sorted(values)
    if not values:
        return {"count": 0, "p50": "0", "p95": "0", "p99": "0", "max": "0"}
    return {
        "count": len(values),
        "p50": _nearest_rank(values, 50).to_eng_string(),
        "p95": _nearest_rank(values, 95).to_eng_string(),
        "p99": _nearest_rank(values, 99).to_eng_string(),
        "max": values[-1].to_eng_string(),
    }


def _iter_pair_dirs(batch_root: Path) -> list[Path]:
    if (batch_root / "aggregate.json").exists():
        return [batch_root]
    return sorted(path for path in batch_root.iterdir() if path.is_dir() and (path / "aggregate.json").exists())


def _scan_results_ndjson(
    path: Path, *, top_n: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, int], dict[str, dict[str, str | int]]]:
    objective_claim_counts: dict[str, dict[str, int]] = {}
    objective_improved_counts: dict[str, int] = {}
    if not path.exists():
        empty = {"all_txs": _bps_distribution([]), "improved_txs": _bps_distribution([])}
        return [], objective_claim_counts, objective_improved_counts, empty
    rows: list[tuple[Decimal, dict[str, Any]]] = []
    all_bps: list[Decimal] = []
    improved_bps: list[Decimal] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            objective = str(row.get("objective") or "unknown")
            claim = str((row.get("best_price_certificate") or {}).get("claim") or "missing")
            objective_claim_counts.setdefault(objective, {})
            objective_claim_counts[objective][claim] = objective_claim_counts[objective].get(claim, 0) + 1
            found = bool(row.get("found_improvement"))
            best = row.get("best") or {}
            bps = _dec(best.get("improvement_bps")) if found else Decimal(0)
            all_bps.append(bps)
            if not row.get("found_improvement"):
                continue
            objective_improved_counts[objective] = objective_improved_counts.get(objective, 0) + 1
            improved_bps.append(bps)
            rows.append((bps, row))
            rows.sort(key=lambda item: item[0], reverse=True)
            if len(rows) > top_n:
                rows.pop()
    return (
        [row for _bps, row in rows],
        objective_claim_counts,
        objective_improved_counts,
        {"all_txs": _bps_distribution(all_bps), "improved_txs": _bps_distribution(improved_bps)},
    )


def _format_baseline(row: dict[str, Any]) -> str:
    baseline = row.get("baseline") or {}
    return f"in={baseline.get('input')}, out={baseline.get('output')}"


def _format_best(row: dict[str, Any]) -> str:
    best = row.get("best") or {}
    if row.get("objective") != "bounded_pareto":
        return f"in={best.get('input') or best.get('spent_input')}, out={best.get('output')}"
    same_input = best.get("same_input") or {}
    same_output = best.get("same_output") or {}
    parts: list[str] = []
    if same_input:
        parts.append(f"same_input: in={same_input.get('spent_input')}, out={same_input.get('output')}")
    if same_output:
        parts.append(f"same_output: in={same_output.get('spent_input')}, out={same_output.get('output')}")
    return "; ".join(parts) if parts else "none"


def main() -> None:
    args = _parse_args()
    batch_root = Path(args.batch_root)
    top_n = max(1, int(args.top_n))
    pair_payloads: list[dict[str, Any]] = []
    global_counts = {
        "pair_count": 0,
        "tx_count": 0,
        "improved_count": 0,
        "unchanged_or_no_better_count": 0,
    }
    global_objective_counts: dict[str, int] = {}
    global_improvement_type_counts: dict[str, int] = {}
    global_claim_counts: dict[str, int] = {}
    global_not_certified_reason_counts: dict[str, int] = {}
    global_gain_total: dict[str, Decimal] = {}
    global_gain_max_bps: dict[str, Decimal] = {}
    global_bounded_total: dict[str, Decimal] = {}
    global_top: list[tuple[Decimal, str, dict[str, Any]]] = []
    global_objective_claim_counts: dict[str, dict[str, int]] = {}
    global_objective_improved_counts: dict[str, int] = {}

    for pair_dir in _iter_pair_dirs(batch_root):
        aggregate = json.loads((pair_dir / "aggregate.json").read_text(encoding="utf-8"))
        pair = pair_dir.name
        payload = {"pair": pair, "aggregate": aggregate}
        pair_payloads.append(payload)
        global_counts["pair_count"] += 1
        for key in ["tx_count", "improved_count", "unchanged_or_no_better_count"]:
            global_counts[key] += int(aggregate.get(key) or 0)
        _add_counts(global_objective_counts, aggregate.get("objective_counts") or {})
        _add_counts(global_improvement_type_counts, aggregate.get("improvement_type_counts") or {})
        _add_counts(global_claim_counts, aggregate.get("best_price_claim_counts") or {})
        _add_counts(global_not_certified_reason_counts, aggregate.get("not_certified_reason_counts") or {})
        _add_decimals(global_gain_total, aggregate.get("gain_decomposition_total") or {})
        _max_decimals(global_gain_max_bps, aggregate.get("gain_decomposition_max_bps") or {})
        _add_decimals(global_bounded_total, aggregate.get("bounded_anchor_decomposition_total") or {})

        results_path = Path(aggregate.get("results_ndjson") or pair_dir / "results.ndjson")
        top_rows, objective_claim_counts, objective_improved_counts, result_distribution = _scan_results_ndjson(
            results_path, top_n=top_n
        )
        if not aggregate.get("improvement_bps_distribution"):
            aggregate["improvement_bps_distribution"] = result_distribution
        for objective, claim_counts in objective_claim_counts.items():
            global_objective_claim_counts.setdefault(objective, {})
            _add_counts(global_objective_claim_counts[objective], claim_counts)
        _add_counts(global_objective_improved_counts, objective_improved_counts)

        for row in top_rows:
            bps = _dec((row.get("best") or {}).get("improvement_bps"))
            global_top.append((bps, pair, row))
            global_top.sort(key=lambda item: item[0], reverse=True)
            if len(global_top) > top_n:
                global_top.pop()

    output = {
        "batch_root": str(batch_root),
        "global_counts": global_counts,
        "global_objective_counts": global_objective_counts,
        "global_improvement_type_counts": global_improvement_type_counts,
        "global_best_price_claim_counts": global_claim_counts,
        "global_not_certified_reason_counts": global_not_certified_reason_counts,
        "global_objective_claim_counts": global_objective_claim_counts,
        "global_objective_improved_counts": global_objective_improved_counts,
        "global_gain_decomposition_total": {
            key: value.to_eng_string() for key, value in sorted(global_gain_total.items())
        },
        "global_gain_decomposition_max_bps": {
            key: value.to_eng_string() for key, value in sorted(global_gain_max_bps.items())
        },
        "global_bounded_anchor_decomposition_total": {
            key: value.to_eng_string() for key, value in sorted(global_bounded_total.items())
        },
        "pairs": pair_payloads,
        "top_improvements": [
            {
                "pair": pair,
                "tx_hash": row.get("tx_hash"),
                "objective": row.get("objective"),
                "improvement_type": row.get("improvement_type"),
                "improvement": (row.get("best") or {}).get("improvement"),
                "improvement_bps": (row.get("best") or {}).get("improvement_bps"),
                "certificate": (row.get("best_price_certificate") or {}).get("claim"),
            }
            for _bps, pair, row in global_top
        ],
    }
    Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Ten-Pair Best-Price Optimizer Report", ""]
    lines.append("## Scope And Interpretation")
    lines.append("")
    lines.append(
        "This report treats the rippled-aligned execution model as the baseline and changes only "
        "the AMM/CLOB allocation under the same transaction intent, pre-state, amount domain, "
        "transfer-rate rules, and funded CLOB/AMM liquidity."
    )
    lines.append("")
    lines.append("- `fixed_input_max_output`: input is fixed; the optimizer maximizes output.")
    lines.append("- `fixed_output_min_input`: output is fixed; the optimizer minimizes input.")
    lines.append(
        "- `bounded_partial`: neither endpoint is uniquely fixed; the optimizer reports Pareto-anchor "
        "improvements by checking same-input max-output and same-output min-input anchors."
    )
    lines.append("")
    lines.append("- `allocation_changed`: a different AMM/CLOB split improves the objective.")
    lines.append(
        "- `amm_multi_slice_fee_compounding`: the AMM/CLOB allocation is unchanged, but replacing "
        "multi-slice AMM execution by one-shot AMM execution removes intermediate fee compounding."
    )
    lines.append(
        "- `bounded_same_input_output_gain`, `bounded_same_output_input_saved`, and "
        "`bounded_pareto_both` are bounded-partial Pareto-anchor gains."
    )
    lines.append("")
    lines.append(
        "`exact_discrete_best_price` and `exact_discrete_pareto_anchor` mean the discrete "
        "candidate search covered the relevant CLOB-level intervals under the current solver. "
        "`not_certified` means the evaluated candidate is reported, but the run does not claim a "
        "complete best-price proof for that tx."
    )
    lines.append("")
    lines.append("## Global Summary")
    lines.append("")
    lines.append(f"- batch_root: `{batch_root}`")
    lines.append(f"- pair_count: `{global_counts['pair_count']}`")
    lines.append(f"- tx_count: `{global_counts['tx_count']}`")
    lines.append(f"- improved_count: `{global_counts['improved_count']}`")
    lines.append(f"- unchanged_or_no_better_count: `{global_counts['unchanged_or_no_better_count']}`")
    lines.append(f"- objective_counts: `{json.dumps(global_objective_counts, sort_keys=True)}`")
    lines.append(f"- improvement_type_counts: `{json.dumps(global_improvement_type_counts, sort_keys=True)}`")
    lines.append(f"- best_price_claim_counts: `{json.dumps(global_claim_counts, sort_keys=True)}`")
    lines.append(f"- not_certified_reason_counts: `{json.dumps(global_not_certified_reason_counts, sort_keys=True)}`")
    lines.append(f"- objective_claim_counts: `{json.dumps(global_objective_claim_counts, sort_keys=True)}`")
    lines.append(f"- objective_improved_counts: `{json.dumps(global_objective_improved_counts, sort_keys=True)}`")
    lines.append(f"- gain_decomposition_total: `{json.dumps(output['global_gain_decomposition_total'], sort_keys=True)}`")
    lines.append(
        f"- bounded_anchor_decomposition_total: "
        f"`{json.dumps(output['global_bounded_anchor_decomposition_total'], sort_keys=True)}`"
    )
    lines.append("")
    lines.append("## Per-Pair Summary")
    lines.append("")
    lines.append("| pair | tx | improved | objectives | claims | p95 bps | p95 improved bps | max bps |")
    lines.append("|---|---:|---:|---|---|---:|---:|---:|")
    for payload in pair_payloads:
        pair = payload["pair"]
        aggregate = payload["aggregate"]
        distribution = aggregate.get("improvement_bps_distribution") or {}
        all_dist = distribution.get("all_txs") or {}
        improved_dist = distribution.get("improved_txs") or {}
        lines.append(
            f"| {pair} | {aggregate.get('tx_count')} | {aggregate.get('improved_count')} | "
            f"`{json.dumps(aggregate.get('objective_counts') or {}, sort_keys=True)}` | "
            f"`{json.dumps(aggregate.get('best_price_claim_counts') or {}, sort_keys=True)}` | "
            f"{all_dist.get('p95')} | "
            f"{improved_dist.get('p95')} | "
            f"{aggregate.get('max_improvement_bps')} |"
        )
    lines.append("")
    lines.append(f"## Top {len(global_top)} Improvements")
    lines.append("")
    lines.append("| pair | tx | objective | type | baseline | best | gain | bps | certificate |")
    lines.append("|---|---|---|---|---|---|---:|---:|---|")
    for _bps, pair, row in global_top:
        best = row.get("best") or {}
        cert = row.get("best_price_certificate") or {}
        lines.append(
            f"| {pair} | `{str(row.get('tx_hash'))[:8]}` | {row.get('objective')} | "
            f"{row.get('improvement_type')} | {_format_baseline(row)} | {_format_best(row)} | {best.get('improvement')} | "
            f"{best.get('improvement_bps')} | {cert.get('claim')} |"
        )
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
