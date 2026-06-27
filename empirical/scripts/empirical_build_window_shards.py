#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ledger-indexed shards for two-week window tables.")
    p.add_argument(
        "--meta-base",
        default="artifacts/metadata/rlusd_xrp/ledger_100725835_101035981",
        help="Base metadata directory containing window_tables/",
    )
    p.add_argument("--shards", type=int, default=256, help="Number of shards (recommend 128/256/512).")
    p.add_argument(
        "--window-dir-name",
        default="window_tables",
        help="Directory name under meta-base for merged two-week tables.",
    )
    return p.parse_args()


def _load_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_parquet(path)


def _shard_one(df: pd.DataFrame, table_name: str, out_dir: Path, shards: int) -> dict:
    if "ledger_index" not in df.columns:
        raise RuntimeError(f"{table_name}: missing required column `ledger_index`")

    out_table_dir = out_dir / table_name
    out_table_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["ledger_index"] = df["ledger_index"].astype("int64")
    df["_sid"] = df["ledger_index"] % shards

    sid_counts = {}
    for sid, g in df.groupby("_sid", sort=True):
        sid_i = int(sid)
        part = g.drop(columns=["_sid"])
        f = out_table_dir / f"shard_{sid_i:03d}.parquet"
        part.to_parquet(f, index=False)
        sid_counts[sid_i] = int(len(part))

    return {
        "rows": int(len(df)),
        "shards_non_empty": int(len(sid_counts)),
        "sid_row_counts": sid_counts,
    }


def main() -> None:
    args = _parse_args()
    if args.shards <= 0:
        raise ValueError("--shards must be positive")

    meta_base = Path(args.meta_base).resolve()
    window_dir = meta_base / args.window_dir_name
    out_dir = meta_base / f"window_cache_shards_{args.shards}"
    out_dir.mkdir(parents=True, exist_ok=True)

    amm_path = window_dir / "amm_swaps_two_week.parquet"
    fee_path = window_dir / "amm_fees_two_week.parquet"
    clob_path = window_dir / "clob_legs_two_week_with_idx.parquet"

    amm = _load_required(amm_path)
    fees = _load_required(fee_path)
    clob = _load_required(clob_path)

    summary = {
        "meta_base": str(meta_base),
        "window_dir": str(window_dir),
        "shards": int(args.shards),
        "tables": {},
    }
    summary["tables"]["amm_swaps"] = _shard_one(amm, "amm_swaps", out_dir, args.shards)
    summary["tables"]["amm_fees"] = _shard_one(fees, "amm_fees", out_dir, args.shards)
    summary["tables"]["clob_legs_with_idx"] = _shard_one(clob, "clob_legs_with_idx", out_dir, args.shards)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"built shard cache: {out_dir}")
    print(f"summary: {summary_path}")
    for k, v in summary["tables"].items():
        print(f"{k}: rows={v['rows']} non_empty_shards={v['shards_non_empty']}")


if __name__ == "__main__":
    main()

