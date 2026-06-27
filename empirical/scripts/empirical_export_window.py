#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession, functions as F


RUSD_HEX = "524C555344000000000000000000000000000000"
RUSD_ISSUER = "rMxCKbEDwqr76QuheSUMdEGf4B9xJ8m5De"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export AMM/CLOB/fees window from Delta Sharing with reusable filters "
            "(ledger/date/time) and pair selection."
        )
    )
    p.add_argument(
        "--share-profile",
        default=os.environ.get("XRPL_SHARE_PROFILE", "data/config.share"),
        help="Delta Sharing profile path",
    )
    p.add_argument("--share", default="ripple-ubri-share", help="Delta Sharing share name")
    p.add_argument("--schema", default="ripplex", help="Delta Sharing schema name")
    p.add_argument("--table-amm", default="fact_amm_swaps", help="AMM swaps table")
    p.add_argument("--table-clob", default="offers_fact_tx", help="CLOB legs table")
    p.add_argument("--table-fees", default="fact_amm_fees", help="AMM fees table")

    p.add_argument("--ledger-start", type=int, default=None, help="ledger_index inclusive lower bound")
    p.add_argument("--ledger-end", type=int, default=None, help="ledger_index inclusive upper bound")
    p.add_argument("--date-start", default=None, help="close_time_date inclusive lower bound (YYYY-MM-DD)")
    p.add_argument("--date-end", default=None, help="close_time_date exclusive upper bound (YYYY-MM-DD)")
    p.add_argument("--time-start", default=None, help="timestamp inclusive lower bound (ISO8601)")
    p.add_argument("--time-end", default=None, help="timestamp exclusive upper bound (ISO8601)")

    p.add_argument("--base-currency", default=RUSD_HEX, help="Base currency in pair filter")
    p.add_argument("--base-issuer", default=RUSD_ISSUER, help="Base issuer in pair filter")
    p.add_argument("--counter-currency", default="XRP", help="Counter currency in pair filter")
    p.add_argument("--counter-issuer", default="", help="Counter issuer in pair filter")

    p.add_argument("--pair", default="rlusd_xrp", help="Pair key used in output path naming")
    p.add_argument("--output-dir", "--out-root", dest="output_dir", default=None, help="Output directory")
    p.add_argument("--with-amm-ledger-csv", action="store_true", help="Also export AMM ledger stats CSV")
    p.add_argument(
        "--sort-output",
        action="store_true",
        help="Sort rows before parquet write (slower). Default is unsorted for speed.",
    )
    p.add_argument(
        "--with-row-counts",
        action="store_true",
        help="Print output row counts after write (slower due to extra scans).",
    )

    p.add_argument("--skip-amm", action="store_true", help="Skip AMM swaps export")
    p.add_argument("--skip-clob", action="store_true", help="Skip CLOB legs export")
    p.add_argument("--skip-fees", action="store_true", help="Skip AMM fees export")
    return p.parse_args()


def ds_url(profile: str, share: str, schema: str, table: str) -> str:
    return f"{profile}#{share}.{schema}.{table}"


def apply_ledger_filter(df, ledger_start: int | None, ledger_end: int | None):
    if ledger_start is not None:
        df = df.filter(F.col("ledger_index") >= F.lit(int(ledger_start)))
    if ledger_end is not None:
        df = df.filter(F.col("ledger_index") <= F.lit(int(ledger_end)))
    return df


def apply_date_filter(df, date_col: str, date_start: str | None, date_end: str | None):
    if date_start:
        df = df.filter(F.col(date_col) >= F.to_date(F.lit(date_start)))
    if date_end:
        df = df.filter(F.col(date_col) < F.to_date(F.lit(date_end)))
    return df


def apply_time_filter(df, time_col: str, time_start: str | None, time_end: str | None):
    if time_start:
        df = df.filter(F.col(time_col) >= F.to_timestamp(F.lit(time_start)))
    if time_end:
        df = df.filter(F.col(time_col) < F.to_timestamp(F.lit(time_end)))
    return df


def default_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        return args.output_dir
    if args.ledger_start is not None and args.ledger_end is not None:
        window = f"ledger_{args.ledger_start}_{args.ledger_end}"
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        window = f"adhoc_{stamp}"
    out_dir = os.path.join("artifacts", "exports", args.pair, window)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def write_manifest(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def validate_share_profile(profile: str) -> None:
    profile_path = Path(profile)
    if not profile_path.exists():
        raise SystemExit(
            f"share profile not found: {profile_path}. "
            "Provide --share-profile or set XRPL_SHARE_PROFILE."
        )

    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"share profile is not valid JSON: {profile_path} ({exc})")

    exp_raw = payload.get("expirationTime")
    if exp_raw:
        try:
            exp = datetime.fromisoformat(exp_raw.replace("Z", "+00:00"))
        except Exception:
            raise SystemExit(f"share profile has unparseable expirationTime: {exp_raw}")
        if exp <= datetime.now(timezone.utc):
            raise SystemExit(
                f"share profile is expired at {exp.isoformat()} (UTC): {profile_path}"
            )

    mode = stat.S_IMODE(profile_path.stat().st_mode)
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        print(
            f"[warn] {profile_path} is group/other-readable "
            f"(mode {oct(mode)}). Recommended: chmod 600 {profile_path}"
        )


def main() -> None:
    args = parse_args()
    if args.ledger_start is not None and args.ledger_end is not None and args.ledger_end < args.ledger_start:
        raise SystemExit("--ledger-end must be >= --ledger-start")
    validate_share_profile(args.share_profile)

    out_root = default_output_dir(args)

    spark = SparkSession.builder.appName("empirical_export_window").getOrCreate()

    # AMM swaps
    if not args.skip_amm:
        amm = spark.read.format("deltaSharing").load(
            ds_url(args.share_profile, args.share, args.schema, args.table_amm)
        )
        amm = apply_ledger_filter(amm, args.ledger_start, args.ledger_end)
        amm = apply_date_filter(amm, "close_time_date", args.date_start, args.date_end)
        amm = apply_time_filter(amm, "close_time_datetime", args.time_start, args.time_end)

        bcur = args.base_currency
        biss = args.base_issuer
        ccur = args.counter_currency
        ciss = args.counter_issuer

        amm = amm.filter(
            (
                (F.col("amm_asset_currency") == F.lit(bcur))
                & (F.col("amm_asset_issuer") == F.lit(biss))
                & (F.col("amm_asset2_currency") == F.lit(ccur))
                & (F.coalesce(F.col("amm_asset2_issuer"), F.lit("")) == F.lit(ciss))
            )
            |
            (
                (F.col("amm_asset_currency") == F.lit(ccur))
                & (F.coalesce(F.col("amm_asset_issuer"), F.lit("")) == F.lit(ciss))
                & (F.col("amm_asset2_currency") == F.lit(bcur))
                & (F.col("amm_asset2_issuer") == F.lit(biss))
            )
        )

        out_amm = os.path.join(out_root, "amm_swaps")
        amm_to_write = amm.orderBy(F.col("close_time_datetime").asc_nulls_last()) if args.sort_output else amm
        amm_to_write.write.mode("overwrite").parquet(out_amm)
        if args.with_row_counts:
            amm_rows = amm.count()
            print(f"[ok] AMM rows={amm_rows} -> {out_amm}")
        else:
            print(f"[ok] AMM -> {out_amm}")

        if args.with_amm_ledger_csv:
            out_amm_csv = os.path.join(out_root, "amm_ledgers")
            (
                amm.groupBy("ledger_index")
                .agg(
                    F.count("*").alias("amm_swap_count"),
                    F.min("close_time_datetime").alias("amm_first_time"),
                    F.max("close_time_datetime").alias("amm_last_time"),
                    F.min("transaction_index").alias("amm_min_tx_index"),
                    F.max("transaction_index").alias("amm_max_tx_index"),
                )
                .orderBy(F.col("amm_swap_count").desc(), F.col("ledger_index").asc())
                .coalesce(1)
                .write.mode("overwrite")
                .option("header", True)
                .csv(out_amm_csv)
            )
            print(f"[ok] AMM ledger CSV -> {out_amm_csv}")

    # CLOB legs
    if not args.skip_clob:
        clob = spark.read.format("deltaSharing").load(
            ds_url(args.share_profile, args.share, args.schema, args.table_clob)
        )
        clob = apply_ledger_filter(clob, args.ledger_start, args.ledger_end)
        clob = apply_date_filter(clob, "close_time_date", args.date_start, args.date_end)
        clob = apply_time_filter(clob, "close_time", args.time_start, args.time_end)

        bcur = args.base_currency
        biss = args.base_issuer
        ccur = args.counter_currency
        ciss = args.counter_issuer

        clob = clob.filter(
            (
                (F.col("offer_base_currency") == F.lit(bcur))
                & (F.col("offer_base_issuer") == F.lit(biss))
                & (F.col("offer_counter_currency") == F.lit(ccur))
                & (F.coalesce(F.col("offer_counter_issuer"), F.lit("")) == F.lit(ciss))
            )
            |
            (
                (F.col("offer_base_currency") == F.lit(ccur))
                & (F.coalesce(F.col("offer_base_issuer"), F.lit("")) == F.lit(ciss))
                & (F.col("offer_counter_currency") == F.lit(bcur))
                & (F.col("offer_counter_issuer") == F.lit(biss))
            )
            |
            (
                (F.col("base_currency") == F.lit(bcur))
                & (F.col("base_issuer") == F.lit(biss))
                & (F.col("counter_currency") == F.lit(ccur))
                & (F.coalesce(F.col("counter_issuer"), F.lit("")) == F.lit(ciss))
            )
            |
            (
                (F.col("base_currency") == F.lit(ccur))
                & (F.coalesce(F.col("base_issuer"), F.lit("")) == F.lit(ciss))
                & (F.col("counter_currency") == F.lit(bcur))
                & (F.col("counter_issuer") == F.lit(biss))
            )
        )

        out_clob = os.path.join(out_root, "clob_legs")
        clob_to_write = clob.orderBy(F.col("close_time").asc_nulls_last()) if args.sort_output else clob
        clob_to_write.write.mode("overwrite").parquet(out_clob)
        if args.with_row_counts:
            clob_rows = clob.count()
            print(f"[ok] CLOB rows={clob_rows} -> {out_clob}")
        else:
            print(f"[ok] CLOB -> {out_clob}")

    # Fees
    if not args.skip_fees:
        fees = spark.read.format("deltaSharing").load(
            ds_url(args.share_profile, args.share, args.schema, args.table_fees)
        )
        fees = apply_ledger_filter(fees, args.ledger_start, args.ledger_end)
        fees = apply_date_filter(fees, "close_time_date", args.date_start, args.date_end)
        fees = apply_time_filter(fees, "close_time_datetime", args.time_start, args.time_end)
        out_fees = os.path.join(out_root, "amm_fees")
        fees_to_write = fees.orderBy(F.col("close_time_datetime").asc_nulls_last()) if args.sort_output else fees
        fees_to_write.write.mode("overwrite").parquet(out_fees)
        if args.with_row_counts:
            fee_rows = fees.count()
            print(f"[ok] FEES rows={fee_rows} -> {out_fees}")
        else:
            print(f"[ok] FEES -> {out_fees}")

    spark.stop()
    manifest = {
        "script": "empirical_export_window.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair": args.pair,
        "filters": {
            "ledger_start": args.ledger_start,
            "ledger_end": args.ledger_end,
            "date_start": args.date_start,
            "date_end": args.date_end,
            "time_start": args.time_start,
            "time_end": args.time_end,
        },
        "outputs": {
            "amm_swaps": None if args.skip_amm else os.path.join(out_root, "amm_swaps"),
            "clob_legs": None if args.skip_clob else os.path.join(out_root, "clob_legs"),
            "amm_fees": None if args.skip_fees else os.path.join(out_root, "amm_fees"),
        },
    }
    write_manifest(os.path.join(out_root, "manifest.json"), manifest)
    print(f"[done] out_root={out_root}")


if __name__ == "__main__":
    main()
