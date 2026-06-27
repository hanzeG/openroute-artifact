#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build full transaction sequence (AMM + CLOB) and prebook ledger list "
            "for a date window."
        )
    )
    p.add_argument(
        "--base-dir",
        default="artifacts/exports/rlusd_xrp",
        help="Base exports directory that contains date_YYYY-MM-DD folders",
    )
    p.add_argument("--date-start", default="2025-12-08", help="Inclusive start date")
    p.add_argument("--date-end", default="2025-12-22", help="Exclusive end date")
    p.add_argument("--amm-subdir", default="amm_swaps", help="AMM swaps subdir")
    p.add_argument(
        "--clob-subdir",
        default="clob_legs_with_idx",
        help="CLOB legs (enriched) subdir",
    )
    p.add_argument(
        "--outdir",
        default="artifacts/metadata/rlusd_xrp/ledger_100725835_101035981",
        help="Output directory",
    )
    return p.parse_args()


def iter_days(start_s: str, end_s: str) -> list[str]:
    d0 = date.fromisoformat(start_s)
    d1 = date.fromisoformat(end_s)
    out: list[str] = []
    d = d0
    while d < d1:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def load_daily_tx(base: Path, day: str, amm_subdir: str, clob_subdir: str) -> pd.DataFrame:
    day_dir = base / f"date_{day}"
    amm_path = day_dir / amm_subdir
    clob_path = day_dir / clob_subdir

    if not amm_path.exists():
        raise FileNotFoundError(f"missing AMM path: {amm_path}")
    if not clob_path.exists():
        raise FileNotFoundError(f"missing CLOB path: {clob_path}")

    amm = pd.read_parquet(amm_path)[["transaction_hash", "ledger_index", "transaction_index"]].copy()
    amm = amm.rename(columns={"transaction_hash": "tx_hash"})
    amm["has_amm"] = 1
    amm["has_clob"] = 0

    clob = pd.read_parquet(clob_path)[["tx_hash", "ledger_index", "transaction_index"]].copy()
    clob["has_amm"] = 0
    clob["has_clob"] = 1

    both = pd.concat([amm, clob], ignore_index=True)
    both["tx_hash"] = both["tx_hash"].astype(str).str.upper()
    both["ledger_index"] = pd.to_numeric(both["ledger_index"], errors="coerce")
    both["transaction_index"] = pd.to_numeric(both["transaction_index"], errors="coerce")
    both = both.dropna(subset=["tx_hash", "ledger_index", "transaction_index"])
    both["ledger_index"] = both["ledger_index"].astype("int64")
    both["transaction_index"] = both["transaction_index"].astype("int64")
    return both


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for day in iter_days(args.date_start, args.date_end):
        df = load_daily_tx(base, day, args.amm_subdir, args.clob_subdir)
        frames.append(df)
        print(f"[ok] {day} rows={len(df)}")

    all_rows = pd.concat(frames, ignore_index=True)

    agg = (
        all_rows.groupby("tx_hash", as_index=False)
        .agg(
            ledger_min=("ledger_index", "min"),
            ledger_max=("ledger_index", "max"),
            txi_min=("transaction_index", "min"),
            txi_max=("transaction_index", "max"),
            has_amm=("has_amm", "max"),
            has_clob=("has_clob", "max"),
        )
        .copy()
    )

    agg["ledger_mismatch"] = (agg["ledger_min"] != agg["ledger_max"]).astype("int64")
    agg["txi_mismatch"] = (agg["txi_min"] != agg["txi_max"]).astype("int64")

    full_tx = agg.rename(
        columns={
            "ledger_min": "ledger_index",
            "txi_min": "transaction_index",
        }
    )[
        [
            "tx_hash",
            "ledger_index",
            "transaction_index",
            "has_amm",
            "has_clob",
            "ledger_mismatch",
            "txi_mismatch",
        ]
    ]

    full_tx = full_tx.sort_values(
        ["ledger_index", "transaction_index", "tx_hash"], ascending=[True, True, True]
    ).reset_index(drop=True)

    prebook = (full_tx["ledger_index"] - 1).drop_duplicates().sort_values().astype("int64")

    out_full_parquet = outdir / "full_tx_sequence.parquet"
    out_full_csv = outdir / "full_tx_sequence.csv"
    out_prebook_txt = outdir / "prebook_ledgers_full.txt"
    out_prebook_parquet = outdir / "prebook_ledgers_full.parquet"
    out_stats = outdir / "full_tx_prebook_stats.json"

    full_tx.to_parquet(out_full_parquet, index=False)
    full_tx.to_csv(out_full_csv, index=False)
    prebook.to_frame(name="ledger_index").to_parquet(out_prebook_parquet, index=False)
    out_prebook_txt.write_text("".join(f"{x}\n" for x in prebook.tolist()), encoding="utf-8")

    stats = {
        "date_start": args.date_start,
        "date_end": args.date_end,
        "rows_input_union": int(len(all_rows)),
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
        "outputs": {
            "full_tx_parquet": str(out_full_parquet),
            "full_tx_csv": str(out_full_csv),
            "prebook_txt": str(out_prebook_txt),
            "prebook_parquet": str(out_prebook_parquet),
        },
    }
    out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"tx_unique={stats['tx_unique']}")
    print(f"prebook_unique_ledgers={stats['prebook_unique_ledgers']}")
    print(f"ledger_range={stats['ledger_min']}-{stats['ledger_max']}")
    print(f"prebook_range={stats['prebook_min']}-{stats['prebook_max']}")
    print(f"ledger_mismatch_txs={stats['ledger_mismatch_txs']} txi_mismatch_txs={stats['txi_mismatch_txs']}")
    print(f"out_full_parquet={out_full_parquet}")
    print(f"out_prebook_txt={out_prebook_txt}")
    print(f"out_stats={out_stats}")


if __name__ == "__main__":
    main()
