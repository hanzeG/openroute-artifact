#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected transaction rows from same-fill optimisation outputs."
    )
    parser.add_argument("--results-root", required=True, help="Root containing PAIR_LABEL/results.ndjson")
    parser.add_argument("--tx-hash", action="append", required=True, help="Transaction hash to extract")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _pair_result_paths(results_root: Path) -> list[Path]:
    return sorted(path / "results.ndjson" for path in results_root.iterdir() if (path / "results.ndjson").exists())


def _extract(results_root: Path, wanted: set[str]) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {tx_hash: [] for tx_hash in wanted}
    for path in _pair_result_paths(results_root):
        pair = path.parent.name
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                tx_hash = str(row.get("tx_hash") or "").upper()
                if tx_hash not in found:
                    continue
                extracted = {
                    "pair": pair,
                    "line_no": line_no,
                    "tx_hash": tx_hash,
                    "layer1_measurement": row.get("layer1_measurement"),
                    "found_improvement": row.get("found_improvement"),
                    "improvement_type": row.get("improvement_type"),
                    "baseline_fit": row.get("baseline_fit"),
                    "baseline": row.get("baseline"),
                    "best": row.get("best"),
                    "counterfactual_comparison": row.get("counterfactual_comparison"),
                    "gain_decomposition": row.get("gain_decomposition"),
                    "best_point_diagnostic": row.get("best_point_diagnostic"),
                }
                found[tx_hash].append(extracted)
    missing = sorted(tx_hash for tx_hash, rows in found.items() if not rows)
    if missing:
        raise RuntimeError(f"missing requested tx hashes: {missing}")
    return found


def main() -> None:
    args = _parse_args()
    wanted = {str(tx_hash).upper() for tx_hash in args.tx_hash}
    payload = {
        "results_root": str(args.results_root),
        "cases": _extract(Path(args.results_root), wanted),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
