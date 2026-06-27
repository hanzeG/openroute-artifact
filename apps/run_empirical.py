#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


SCRIPT_MAP = {
    "delta-sharing-check-freshness": "empirical/scripts/delta_sharing_check_freshness.py",
    "delta-sharing-test-profile": "empirical/scripts/delta_sharing_test_profile.py",
    "empirical-export-window": "empirical/scripts/empirical_export_window.py",
    "empirical-build-model-input": "empirical/scripts/empirical_build_model_input.py",
    "empirical-compare-rolling": "empirical/scripts/empirical_compare_rolling.py",
    "empirical-replay-tx-prebook": "empirical/scripts/empirical_replay_tx_prebook_rust.py",
    "empirical-fit-single-direct": "empirical/scripts/empirical_fit_single_direct.py",
}

LEGACY_ALIAS_MAP = {
    "check-freshness": "delta-sharing-check-freshness",
    "test-share-profile": "delta-sharing-test-profile",
    "pipeline-export-window": "empirical-export-window",
    "pipeline-build-model-input": "empirical-build-model-input",
    "research-compare-rolling": "empirical-compare-rolling",
    "research-replay-tx-prebook-rust": "empirical-replay-tx-prebook",
    "empirical-replay-tx-prebook-rust": "empirical-replay-tx-prebook",
}


def _strip_double_dash(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an empirical artifact script by alias.")
    parser.add_argument("alias")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _run_script(root: Path, rel_path: str, script_args: list[str]) -> int:
    target = root / rel_path
    sys.argv = [str(target), *_strip_double_dash(script_args)]
    runpy.run_path(str(target), run_name="__main__")
    return 0


def _parse_pipeline_args(raw: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run the legacy empirical pipeline shape.")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--ledger-start", required=True)
    parser.add_argument("--ledger-end", required=True)
    parser.add_argument("--tx-prebook-snapshots", required=True)
    parser.add_argument("--metadata-ndjson", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(_strip_double_dash(raw))


def _pipeline_run(raw: list[str]) -> int:
    args = _parse_pipeline_args(raw)
    if not args.dry_run:
        raise SystemExit("pipeline-run is kept only for dry-run compatibility; use README commands for reproduction.")
    print("[dry-run] planned pipeline steps:")
    for path in (
        "empirical/scripts/empirical_export_window.py",
        "empirical/scripts/empirical_build_model_input.py",
        "empirical/scripts/empirical_compare_rolling.py",
    ):
        print(path)
    return 0


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.alias == "pipeline-run":
        return _pipeline_run(args.script_args)
    alias = LEGACY_ALIAS_MAP.get(args.alias, args.alias)
    rel_path = SCRIPT_MAP.get(alias)
    if rel_path is None:
        raise SystemExit(f"unknown empirical alias: {args.alias}")
    return _run_script(root, rel_path, args.script_args)


if __name__ == "__main__":
    raise SystemExit(main())
