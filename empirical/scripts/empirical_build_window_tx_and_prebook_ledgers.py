#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full transaction sequence and prebook ledger list from one exported window."
    )
    parser.add_argument("--export-dir", required=True, help="Directory containing amm_swaps and clob_legs parquet dirs")
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def _load_amm(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    required = ["transaction_hash", "ledger_index", "transaction_index"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"AMM parquet missing columns {missing}: {path}")
    out = df[required].copy().rename(columns={"transaction_hash": "tx_hash"})
    out["has_amm"] = 1
    out["has_clob"] = 0
    return out


def _load_clob(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    tx_col = "tx_hash" if "tx_hash" in df.columns else "transaction_hash"
    required = [tx_col, "ledger_index", "transaction_index"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"CLOB parquet missing columns {missing}: {path}")
    out = df[required].copy().rename(columns={tx_col: "tx_hash"})
    out["has_amm"] = 0
    out["has_clob"] = 1
    return out


def main() -> None:
    args = _parse_args()
    export_dir = Path(args.export_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = pd.concat(
        [_load_amm(export_dir / "amm_swaps"), _load_clob(export_dir / "clob_legs")],
        ignore_index=True,
    )
    rows["tx_hash"] = rows["tx_hash"].astype(str).str.upper()
    rows["ledger_index"] = pd.to_numeric(rows["ledger_index"], errors="coerce")
    rows["transaction_index"] = pd.to_numeric(rows["transaction_index"], errors="coerce")
    rows = rows.dropna(subset=["tx_hash", "ledger_index", "transaction_index"]).copy()
    rows["ledger_index"] = rows["ledger_index"].astype("int64")
    rows["transaction_index"] = rows["transaction_index"].astype("int64")

    full_tx = (
        rows.groupby("tx_hash", as_index=False)
        .agg(
            ledger_index=("ledger_index", "min"),
            ledger_max=("ledger_index", "max"),
            transaction_index=("transaction_index", "min"),
            txi_max=("transaction_index", "max"),
            has_amm=("has_amm", "max"),
            has_clob=("has_clob", "max"),
        )
        .copy()
    )
    full_tx["ledger_mismatch"] = (full_tx["ledger_index"] != full_tx["ledger_max"]).astype("int64")
    full_tx["txi_mismatch"] = (full_tx["transaction_index"] != full_tx["txi_max"]).astype("int64")
    full_tx = full_tx[
        [
            "tx_hash",
            "ledger_index",
            "transaction_index",
            "has_amm",
            "has_clob",
            "ledger_mismatch",
            "txi_mismatch",
        ]
    ].sort_values(["ledger_index", "transaction_index", "tx_hash"])

    prebook = (full_tx["ledger_index"] - 1).drop_duplicates().sort_values().astype("int64")
    full_tx.to_parquet(outdir / "full_tx_sequence.parquet", index=False)
    full_tx.to_csv(outdir / "full_tx_sequence.csv", index=False)
    prebook.to_frame(name="ledger_index").to_parquet(outdir / "prebook_ledgers_full.parquet", index=False)
    (outdir / "prebook_ledgers_full.txt").write_text("".join(f"{value}\n" for value in prebook), encoding="utf-8")

    stats = {
        "export_dir": str(export_dir),
        "tx_unique": int(len(full_tx)),
        "tx_has_amm": int((full_tx["has_amm"] == 1).sum()),
        "tx_has_clob": int((full_tx["has_clob"] == 1).sum()),
        "tx_has_both": int(((full_tx["has_amm"] == 1) & (full_tx["has_clob"] == 1)).sum()),
        "ledger_mismatch_txs": int(full_tx["ledger_mismatch"].sum()),
        "txi_mismatch_txs": int(full_tx["txi_mismatch"].sum()),
        "ledger_min": int(full_tx["ledger_index"].min()),
        "ledger_max": int(full_tx["ledger_index"].max()),
        "prebook_unique_ledgers": int(len(prebook)),
        "prebook_min": int(prebook.min()),
        "prebook_max": int(prebook.max()),
    }
    (outdir / "full_tx_prebook_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
