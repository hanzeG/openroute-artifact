#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble one canonical fit dataset root from pipeline outputs.")
    parser.add_argument("--pair-config", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--metadata-ndjson", required=True)
    parser.add_argument("--tx-prebook-snapshots", required=True)
    parser.add_argument("--required-tx", required=True)
    parser.add_argument("--ledger-list", required=True)
    parser.add_argument("--amm-swaps", required=True)
    parser.add_argument("--amm-fees", required=True)
    parser.add_argument("--account-offers-snapshots", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_pair(path: Path, pair: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("pairs") or []:
        if str(row.get("pair")) == pair:
            return row
    raise RuntimeError(f"pair not found in resolved config: {pair}")


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise RuntimeError(f"missing input: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise RuntimeError(f"missing input: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def main() -> None:
    args = _parse_args()
    pair_row = _load_pair(Path(args.pair_config), args.pair)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _copy_file(Path(args.metadata_ndjson), out / "tx_metadata_full_merged.ndjson")
    _copy_file(Path(args.tx_prebook_snapshots), out / "tx_prebook_snapshots.ndjson")
    _copy_file(Path(args.required_tx), out / "required_tx.csv")
    _copy_file(Path(args.ledger_list), out / "ledger_list.txt")
    _copy_tree(Path(args.amm_swaps), out / "amm_swaps_two_week.parquet")
    _copy_tree(Path(args.amm_fees), out / "amm_fees_two_week.parquet")
    if args.account_offers_snapshots:
        _copy_file(Path(args.account_offers_snapshots), out / "account_offers_snapshots.ndjson")

    manifest = {
        "pair": pair_row["pair"],
        "pair_label": pair_row["pair_label"],
        "window": pair_row["window"],
        "base_currency": pair_row["base_currency"],
        "base_label": pair_row["base_label"],
        "counter_currency": pair_row["counter_currency"],
        "counter_label": "XRP",
        "issuer_transfer_rates": {
            f"{pair_row['base_currency']}:{pair_row['base_issuer']}": 1000000000
        },
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"dataset_root": str(out), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
