#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prepare inputs for full-ledger metadata fetch from a tx sequence table. "
            "Emits ledger_list.txt and required_tx.csv."
        )
    )
    p.add_argument(
        "--tx-sequence",
        required=True,
        help="Path to full_tx_sequence.parquet or .csv",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Output directory for ledger_list.txt and required_tx.csv",
    )
    p.add_argument(
        "--ledger-start",
        type=int,
        default=None,
        help="Optional inclusive lower ledger bound",
    )
    p.add_argument(
        "--ledger-end",
        type=int,
        default=None,
        help="Optional inclusive upper ledger bound",
    )
    return p.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise RuntimeError(f"unsupported tx-sequence format: {path}")
    need = {"tx_hash", "ledger_index", "transaction_index"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise RuntimeError(f"tx-sequence missing required columns {missing}: {path}")
    return df


def main() -> None:
    args = parse_args()
    tx_path = Path(args.tx_sequence)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_table(tx_path).copy()
    df["tx_hash"] = df["tx_hash"].astype(str).str.upper()
    df["ledger_index"] = pd.to_numeric(df["ledger_index"], errors="coerce")
    df["transaction_index"] = pd.to_numeric(df["transaction_index"], errors="coerce")
    df = df.dropna(subset=["tx_hash", "ledger_index", "transaction_index"]).copy()
    df["ledger_index"] = df["ledger_index"].astype("int64")
    df["transaction_index"] = df["transaction_index"].astype("int64")

    if args.ledger_start is not None:
        df = df[df["ledger_index"] >= int(args.ledger_start)].copy()
    if args.ledger_end is not None:
        df = df[df["ledger_index"] <= int(args.ledger_end)].copy()

    df = df.sort_values(
        ["ledger_index", "transaction_index", "tx_hash"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("no rows remain after filtering")

    ledger_list = sorted(df["ledger_index"].drop_duplicates().astype("int64").tolist())
    required_tx = df[["ledger_index", "transaction_index", "tx_hash"]].rename(
        columns={"tx_hash": "transaction_hash"}
    )

    ledger_list_path = outdir / "ledger_list.txt"
    required_tx_path = outdir / "required_tx.csv"

    ledger_list_path.write_text("".join(f"{x}\n" for x in ledger_list), encoding="utf-8")
    required_tx.to_csv(required_tx_path, index=False)

    print(f"rows={len(df)}")
    print(f"unique_ledgers={len(ledger_list)}")
    print(f"ledger_min={int(min(ledger_list))}")
    print(f"ledger_max={int(max(ledger_list))}")
    print(f"ledger_list={ledger_list_path}")
    print(f"required_tx_csv={required_tx_path}")


if __name__ == "__main__":
    main()
