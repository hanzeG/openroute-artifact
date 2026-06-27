#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run isolated direct-pair fit/compare for one strict-direct day directory "
            "using full-ledger metadata and tx-level prebook snapshots."
        )
    )
    p.add_argument(
        "--day-dir",
        required=True,
        help="Path like artifacts/metadata/.../strict_direct_targets/date_YYYY-MM-DD",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional explicit fit output dir. Defaults to "
            "artifacts/compare/<pair>/<window>/strict_direct_targets/date_YYYY-MM-DD/isolated_fit"
        ),
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample cap for smoke/debug runs.",
    )
    p.add_argument(
        "--issuer-transfer-rates-json",
        required=True,
        help=(
            "JSON containing raw issuer transfer rates/ranges. Required so day fits "
            "use the same fail-fast transfer-rate path as two-week fits."
        ),
    )
    return p.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _derive_pair_window(day_dir: Path, repo_root: Path) -> tuple[str, str]:
    metadata_root = (repo_root / "artifacts/metadata").resolve()
    try:
        rel = day_dir.resolve().relative_to(metadata_root)
    except ValueError as exc:
        raise RuntimeError(f"day dir must live under {metadata_root}: {day_dir}") from exc
    if len(rel.parts) < 4:
        raise RuntimeError(f"unexpected strict-direct day dir layout: {day_dir}")
    return rel.parts[0], rel.parts[1]


def _validate_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"missing {label}: {path}")


def main() -> int:
    args = parse_args()
    repo_root = _repo_root()
    day_dir = Path(args.day_dir)
    pair, window = _derive_pair_window(day_dir, repo_root)
    day_name = day_dir.name

    full_ledger_dir = day_dir / "full_ledger_metadata"
    metadata_ndjson = full_ledger_dir / "tx_metadata_full_merged.ndjson"
    required_tx_csv = full_ledger_dir / "required_tx.csv"
    tx_prebook_dir = (
        repo_root
        / "artifacts/prebook"
        / pair
        / window
        / "strict_direct_targets"
        / day_name
        / "tx_prebook_replay_full_ledger_metadata"
    )
    tx_prebook_snapshots = tx_prebook_dir / "tx_prebook_snapshots.ndjson"
    window_tables = repo_root / "artifacts/metadata" / pair / window / "window_tables"
    amm_swaps = window_tables / "amm_swaps_two_week.parquet"
    amm_fees = window_tables / "amm_fees_two_week.parquet"

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root
        / "artifacts/compare"
        / pair
        / window
        / "strict_direct_targets"
        / day_name
        / "isolated_fit"
    )

    _validate_exists(day_dir, "day dir")
    _validate_exists(full_ledger_dir, "full-ledger metadata dir")
    _validate_exists(metadata_ndjson, "metadata ndjson")
    _validate_exists(required_tx_csv, "required tx csv")
    _validate_exists(tx_prebook_snapshots, "tx-prebook snapshots")
    _validate_exists(amm_swaps, "AMM swaps parquet")
    _validate_exists(amm_fees, "AMM fees parquet")

    output_dir.mkdir(parents=True, exist_ok=True)

    runner = repo_root / "empirical/scripts/empirical_fit_single_direct.py"
    cmd = [
        sys.executable,
        str(runner),
        "--root",
        str(repo_root),
        "--metadata-ndjson",
        str(metadata_ndjson),
        "--tx-prebook-snapshots",
        str(tx_prebook_snapshots),
        "--amm-swaps",
        str(amm_swaps),
        "--amm-fees",
        str(amm_fees),
        "--target-tx-file",
        str(required_tx_csv),
        "--output-dir",
        str(output_dir),
        "--issuer-transfer-rates-json",
        str(args.issuer_transfer_rates_json),
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])

    print("[day-fit] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(repo_root))

    manifest_path = output_dir / "manifest.json"
    aggregate_path = output_dir / "aggregate.json"
    _validate_exists(manifest_path, "fit manifest")
    _validate_exists(aggregate_path, "fit aggregate")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    day_manifest = {
        "scope": "strict_direct_day_isolated_fit",
        "day_dir": str(day_dir),
        "pair": pair,
        "window": window,
        "day_name": day_name,
        "metadata_ndjson": str(metadata_ndjson),
        "required_tx_csv": str(required_tx_csv),
        "tx_prebook_snapshots": str(tx_prebook_snapshots),
        "amm_swaps": str(amm_swaps),
        "amm_fees": str(amm_fees),
        "output_dir": str(output_dir),
        "fit_manifest": manifest,
        "aggregate": aggregate,
    }
    day_manifest_path = output_dir / "day_fit_manifest.json"
    day_manifest_path.write_text(json.dumps(day_manifest, indent=2), encoding="utf-8")
    print(json.dumps(day_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
