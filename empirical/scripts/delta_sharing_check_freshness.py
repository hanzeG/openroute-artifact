#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pyspark.sql import SparkSession, functions as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="deltaSharing table string, e.g. data/config.share#...fact_amm_swaps")
    ap.add_argument("--time-col", default="close_time_datetime", help="timestamp column (default: close_time_datetime)")
    ap.add_argument("--ledger-col", default="ledger_index", help="ledger index column (default: ledger_index)")
    ap.add_argument("--limit", type=int, default=50, help="top-N latest rows to display (default: 50)")
    ap.add_argument("--with-count", action="store_true", help="also compute count(*) (can be slow)")
    ap.add_argument("--where", default="", help="optional SQL filter, e.g. close_time_date >= '2025-12-15'")
    args = ap.parse_args()

    spark = SparkSession.builder.appName("delta_sharing_check_freshness").getOrCreate()

    df = spark.read.format("deltaSharing").load(args.table)

    if args.where.strip():
        df = df.where(args.where)

    # 1) Fast-ish bounds
    agg_exprs = [
        F.max(F.col(args.ledger_col)).alias("max_ledger"),
        F.min(F.col(args.ledger_col)).alias("min_ledger"),
        F.max(F.col(args.time_col)).alias("max_time"),
        F.min(F.col(args.time_col)).alias("min_time"),
    ]
    if args.with_count:
        agg_exprs.append(F.count(F.lit(1)).alias("n"))

    print("\n=== Bounds ===")
    df.agg(*agg_exprs).show(truncate=False)

    # 2) Show latest rows (helps confirm table is actually readable)
    cols = [c for c in [args.ledger_col, args.time_col, "transaction_hash", "tx_hash", "transaction_index"] if c in df.columns]
    if not cols:
        cols = [args.ledger_col, args.time_col]

    print(f"\n=== Latest {args.limit} rows (ordered by {args.time_col} desc) ===")
    (df.select(*cols)
       .orderBy(F.col(args.time_col).desc_nulls_last(), F.col(args.ledger_col).desc_nulls_last())
       .limit(args.limit)
       .show(truncate=False))

    spark.stop()


if __name__ == "__main__":
    main()