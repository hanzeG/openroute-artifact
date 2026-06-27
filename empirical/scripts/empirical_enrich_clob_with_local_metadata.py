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
            "Enrich exported daily clob_legs parquet with ledger_index/transaction_index "
            "using local tx metadata NDJSON."
        )
    )
    p.add_argument(
        "--base-dir",
        default="artifacts/exports/rlusd_xrp",
        help="Base exports directory that contains date_YYYY-MM-DD folders",
    )
    p.add_argument(
        "--metadata-ndjson",
        required=True,
        help="Path to tx_metadata_ok.ndjson",
    )
    p.add_argument("--date-start", default="2025-12-08", help="Inclusive start date")
    p.add_argument("--date-end", default="2025-12-22", help="Exclusive end date")
    p.add_argument(
        "--input-subdir",
        default="clob_legs",
        help="Input subdir under each day directory",
    )
    p.add_argument(
        "--output-subdir",
        default="clob_legs_with_idx",
        help="Output subdir under each day directory",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output subdir if it already exists",
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


def load_metadata_map(path: Path) -> dict[str, tuple[int | None, int | None]]:
    m: dict[str, tuple[int | None, int | None]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            txh = str(obj.get("tx_hash") or "").upper()
            if not txh:
                continue
            result = obj.get("result") or {}
            li_raw = result.get("ledger_index")
            meta = result.get("meta") or {}
            ti_raw = meta.get("TransactionIndex") if isinstance(meta, dict) else None
            li = int(li_raw) if li_raw is not None else None
            ti = int(ti_raw) if ti_raw is not None else None
            m[txh] = (li, ti)
    return m


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    md_path = Path(args.metadata_ndjson)
    if not base.exists():
        raise SystemExit(f"base dir not found: {base}")
    if not md_path.exists():
        raise SystemExit(f"metadata file not found: {md_path}")

    meta_map = load_metadata_map(md_path)
    print(f"metadata_map_size={len(meta_map)}")

    total_rows = 0
    total_enriched = 0
    total_missing = 0

    for day in iter_days(args.date_start, args.date_end):
        day_dir = base / f"date_{day}"
        in_path = day_dir / args.input_subdir
        out_path = day_dir / args.output_subdir

        if not in_path.exists():
            print(f"[skip] {day} missing input: {in_path}")
            continue
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {day} output exists: {out_path}")
            continue

        df = pd.read_parquet(in_path)
        if "tx_hash" not in df.columns:
            raise RuntimeError(f"{in_path} missing tx_hash column")

        txu = df["tx_hash"].astype(str).str.upper()
        mapped = txu.map(meta_map)
        df["ledger_index"] = mapped.map(lambda x: x[0] if isinstance(x, tuple) else None)
        df["transaction_index"] = mapped.map(lambda x: x[1] if isinstance(x, tuple) else None)

        rows = len(df)
        missing = int(df["ledger_index"].isna().sum() + df["transaction_index"].isna().sum())
        enriched_rows = int(((~df["ledger_index"].isna()) & (~df["transaction_index"].isna())).sum())

        if out_path.exists():
            # pandas -> parquet with pyarrow writes directory/file depending on engine settings.
            # Remove existing output path explicitly when overwrite is requested.
            if out_path.is_dir():
                for p in sorted(out_path.rglob("*"), reverse=True):
                    if p.is_file():
                        p.unlink()
                    else:
                        p.rmdir()
                out_path.rmdir()
            else:
                out_path.unlink()

        df.to_parquet(out_path, index=False)

        total_rows += rows
        total_enriched += enriched_rows
        total_missing += missing

        print(
            f"[ok] {day} rows={rows} enriched={enriched_rows} "
            f"missing_any_field={missing} -> {out_path}"
        )

    print(
        f"done total_rows={total_rows} total_enriched={total_enriched} "
        f"missing_any_field={total_missing}"
    )


if __name__ == "__main__":
    main()

