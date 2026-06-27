#!/usr/bin/env python3
"""Run the best-price optimizer over multiple pair datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
OPTIMIZER = SCRIPT_DIR / "empirical_best_price_intent_optimizer.py"


def _parse_pair_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--pair must be LABEL=DATASET_MANIFEST")
    label, manifest = raw.split("=", 1)
    label = label.strip()
    manifest = manifest.strip()
    if not label or not manifest:
        raise argparse.ArgumentTypeError("--pair must be LABEL=DATASET_MANIFEST")
    return label, Path(manifest)


def _parse_pair_jobs(raw: str) -> tuple[str, int]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--pair-jobs must be LABEL=N")
    label, value = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("--pair-jobs must be LABEL=N")
    try:
        jobs = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--pair-jobs must use an integer N") from exc
    if jobs <= 0:
        raise argparse.ArgumentTypeError("--pair-jobs N must be positive")
    return label, jobs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", type=_parse_pair_arg, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pair-workers", type=int, default=1)
    parser.add_argument("--jobs-per-pair", type=int, default=1)
    parser.add_argument("--pair-jobs", action="append", type=_parse_pair_jobs, default=[])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--candidate-pool-size", type=int, default=0)
    parser.add_argument("--max-txs", type=int, default=0)
    parser.add_argument(
        "--optimizer-mode",
        choices=["all", "fixed_input_max_output", "fixed_output_min_input", "bounded_partial"],
        default="all",
    )
    parser.add_argument("--max-clob-segments", type=int, default=0)
    parser.add_argument("--rounding-window-drops", type=int, default=8)
    parser.add_argument(
        "--layer1-measurement-policy",
        choices=["natural", "both"],
        default="both",
    )
    parser.add_argument("--context-batch-size", type=int, default=2_000)
    parser.add_argument("--stream-required", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in label)


def _optimizer_command(
    *,
    args: argparse.Namespace,
    label: str,
    manifest: Path,
    output_dir: Path,
    pair_jobs: int,
) -> list[str]:
    cmd = [
        str(args.python),
        str(OPTIMIZER),
        "--dataset-manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--sample-from-required",
        "--optimizer-mode",
        str(args.optimizer_mode),
        "--candidate-pool-size",
        str(max(0, int(args.candidate_pool_size))),
        "--max-txs",
        str(max(0, int(args.max_txs))),
        "--max-clob-segments",
        str(int(args.max_clob_segments)),
        "--rounding-window-drops",
        str(max(0, int(args.rounding_window_drops))),
        "--layer1-measurement-policy",
        str(args.layer1_measurement_policy),
        "--jobs",
        str(pair_jobs),
        "--context-batch-size",
        str(max(1, int(args.context_batch_size))),
    ]
    if args.stream_required:
        cmd.append("--stream-required")
    return cmd


def _run_one(
    *,
    args: argparse.Namespace,
    label: str,
    manifest: Path,
    output_root: Path,
    pair_jobs: int,
) -> dict[str, Any]:
    started = time.monotonic()
    pair_dir = output_root / _safe_label(label)
    pair_dir.mkdir(parents=True, exist_ok=True)
    log_path = pair_dir / "optimizer.log"
    cmd = _optimizer_command(
        args=args,
        label=label,
        manifest=manifest,
        output_dir=pair_dir,
        pair_jobs=pair_jobs,
    )
    if args.dry_run:
        elapsed = time.monotonic() - started
        return {
            "pair": label,
            "manifest": str(manifest),
            "output_dir": str(pair_dir),
            "log": str(log_path),
            "jobs": pair_jobs,
            "returncode": None,
            "elapsed_seconds": elapsed,
            "command": cmd,
            "aggregate": None,
        }
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    elapsed = time.monotonic() - started
    aggregate_path = pair_dir / "aggregate.json"
    aggregate = None
    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    return {
        "pair": label,
        "manifest": str(manifest),
        "output_dir": str(pair_dir),
        "log": str(log_path),
        "jobs": pair_jobs,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "command": cmd,
        "aggregate": aggregate,
    }


def _write_batch_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: str(row["pair"]))
    payload = {
        "pair_count": len(rows),
        "failed_count": sum(1 for row in rows if row.get("returncode") not in {0, None}),
        "total_elapsed_seconds": sum(float(row.get("elapsed_seconds") or 0) for row in rows),
        "pairs": rows,
    }
    (output_root / "batch_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = ["# Best Price Batch", ""]
    lines.append(f"- pair_count: `{payload['pair_count']}`")
    lines.append(f"- failed_count: `{payload['failed_count']}`")
    lines.append(f"- total_elapsed_seconds: `{payload['total_elapsed_seconds']:.3f}`")
    lines.append("")
    lines.append("| pair | rc | seconds | tx_count | improved | certificate claims | output |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for row in rows:
        aggregate = row.get("aggregate") or {}
        claims = aggregate.get("best_price_claim_counts")
        lines.append(
            f"| {row['pair']} | {row.get('returncode')} | "
            f"{float(row.get('elapsed_seconds') or 0):.3f} | "
            f"{aggregate.get('tx_count')} | "
            f"{aggregate.get('improved_count')} | "
            f"`{json.dumps(claims, sort_keys=True)}` | "
            f"`{row.get('output_dir')}` |"
        )
    (output_root / "batch_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pair_jobs = {label: jobs for label, jobs in args.pair_jobs}
    workers = max(1, int(args.pair_workers))
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_one,
                args=args,
                label=label,
                manifest=manifest,
                output_root=output_root,
                pair_jobs=pair_jobs.get(label, max(1, int(args.jobs_per_pair))),
            )
            for label, manifest in args.pair
        ]
        for future in as_completed(futures):
            rows.append(future.result())
            _write_batch_summary(output_root, rows)
    _write_batch_summary(output_root, rows)
    failed = [row for row in rows if row.get("returncode") not in {0, None}]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
