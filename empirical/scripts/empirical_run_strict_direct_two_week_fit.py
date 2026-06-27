#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run isolated direct-pair fit/compare for the two-week strict-direct window "
            "from one canonical dataset root."
        )
    )
    p.add_argument(
        "--dataset-root",
        required=True,
        help=(
            "Canonical two-week isolated dataset root containing "
            "tx_metadata_full_merged.ndjson, tx_prebook_snapshots.ndjson, "
            "amm_swaps_two_week.parquet, amm_fees_two_week.parquet, and required_tx.csv."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional explicit fit output dir. Defaults to "
            "artifacts/compare/<pair>/<window>/strict_direct_targets/two_week_isolated/"
            "isolated_fit_canonical"
        ),
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample cap for smoke/debug runs.",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Forwarded to empirical_fit_single_direct.py to skip per-tx reports and large summaries.",
    )
    p.add_argument(
        "--write-structure-false-reports",
        action="store_true",
        help="Forwarded to empirical_fit_single_direct.py with --aggregate-only for compact debugging.",
    )
    p.add_argument(
        "--top-relative-errors",
        type=int,
        default=0,
        help="Forwarded to empirical_fit_single_direct.py to keep compact top relative-error samples.",
    )
    p.add_argument(
        "--report-only-output",
        action="store_true",
        help=(
            "After generating error_analysis.md, remove machine/debug outputs from the "
            "output directory. Use when only the formatted report should remain."
        ),
    )
    return p.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"missing {label}: {path}")


def _required_tx_contains_offercreate(path: Path) -> bool:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tx_type = str(row.get("transaction_type") or "").upper()
            if tx_type == "OFFERCREATE":
                return True
    return False


def _offer_tick_size_snapshots(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.glob("offer_tick_sizes_at_ledger_*.json"))


def _derive_pair_window(dataset_root: Path, repo_root: Path) -> tuple[str, str]:
    fit_inputs_root = (repo_root / "artifacts/fit_inputs").resolve()
    try:
        rel = dataset_root.resolve().relative_to(fit_inputs_root)
    except ValueError as exc:
        manifest_path = dataset_root / "dataset_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            pair = str(manifest.get("pair") or "")
            window = str(manifest.get("window") or "")
            if pair and window:
                return pair, window
        raise RuntimeError(
            f"dataset root must live under {fit_inputs_root} or provide pair/window in dataset_manifest.json: {dataset_root}"
        ) from exc
    if len(rel.parts) < 4:
        raise RuntimeError(f"unexpected canonical dataset layout: {dataset_root}")
    return rel.parts[0], rel.parts[1]


def main() -> int:
    args = parse_args()
    repo_root = _repo_root()
    dataset_root = Path(args.dataset_root)
    pair, window = _derive_pair_window(dataset_root, repo_root)

    metadata_ndjson = dataset_root / "tx_metadata_full_merged.ndjson"
    tx_prebook_snapshots = dataset_root / "tx_prebook_snapshots.ndjson"
    amm_swaps = dataset_root / "amm_swaps_two_week.parquet"
    amm_fees = dataset_root / "amm_fees_two_week.parquet"
    required_tx_csv = dataset_root / "required_tx.csv"
    ledger_list = dataset_root / "ledger_list.txt"
    dataset_manifest = dataset_root / "dataset_manifest.json"

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root
        / "artifacts/compare"
        / pair
        / window
        / "strict_direct_targets"
        / "two_week_isolated"
        / "isolated_fit_canonical"
    )

    _validate_exists(dataset_root, "dataset root")
    _validate_exists(metadata_ndjson, "metadata ndjson")
    _validate_exists(tx_prebook_snapshots, "tx-prebook snapshots")
    _validate_exists(amm_swaps, "AMM swaps parquet")
    _validate_exists(amm_fees, "AMM fees parquet")
    _validate_exists(required_tx_csv, "required tx csv")
    _validate_exists(ledger_list, "ledger list")
    _validate_exists(dataset_manifest, "dataset manifest")
    tick_size_snapshots = _offer_tick_size_snapshots(dataset_root)
    if _required_tx_contains_offercreate(required_tx_csv) and not tick_size_snapshots:
        raise RuntimeError(
            "dataset root is missing offer tick-size snapshots for OfferCreate replay: "
            f"{dataset_root}"
        )
    dataset_manifest_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    base_currency = str(
        dataset_manifest_payload.get("base_currency")
        or "524C555344000000000000000000000000000000"
    )
    base_label = str(dataset_manifest_payload.get("base_label") or "rUSD")

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
        str(dataset_manifest),
        "--base-currency",
        base_currency,
        "--base-label",
        base_label,
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.aggregate_only:
        cmd.append("--aggregate-only")
    if args.write_structure_false_reports:
        cmd.append("--write-structure-false-reports")
    if args.top_relative_errors > 0:
        cmd.extend(["--top-relative-errors", str(args.top_relative_errors)])

    print("[two-week-fit] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(repo_root))

    manifest_path = output_dir / "manifest.json"
    aggregate_path = output_dir / "aggregate.json"
    _validate_exists(manifest_path, "fit manifest")
    _validate_exists(aggregate_path, "fit aggregate")
    report_path = output_dir / "error_analysis.md"
    formatter = repo_root / "empirical/scripts/format_endpoint_residual_report.py"
    subprocess.run(
        [
            sys.executable,
            str(formatter),
            "--aggregate",
            str(aggregate_path),
            "--output",
            str(report_path),
            "--title",
            pair.replace("_", "/").upper(),
        ],
        check=True,
        cwd=str(repo_root),
    )

    fit_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    run_manifest = {
        "scope": "strict_direct_two_week_isolated_fit",
        "dataset_root": str(dataset_root),
        "pair": pair,
        "window": window,
        "metadata_ndjson": str(metadata_ndjson),
        "tx_prebook_snapshots": str(tx_prebook_snapshots),
        "amm_swaps": str(amm_swaps),
        "amm_fees": str(amm_fees),
        "required_tx_csv": str(required_tx_csv),
        "ledger_list": str(ledger_list),
        "offer_tick_size_snapshots": [str(path) for path in tick_size_snapshots],
        "output_dir": str(output_dir),
        "error_analysis": str(report_path),
        "fit_manifest": fit_manifest,
        "aggregate": aggregate,
    }
    run_manifest_path = output_dir / "two_week_fit_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    if args.report_only_output:
        for path in (
            output_dir / "aggregate.json",
            output_dir / "intent_error_breakdown.json",
            output_dir / "manifest.json",
            run_manifest_path,
        ):
            if path.exists():
                path.unlink()
    print(json.dumps(run_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
