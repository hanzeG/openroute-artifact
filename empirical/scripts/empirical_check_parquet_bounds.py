#!/usr/bin/env python3

import argparse
from pyspark.sql import SparkSession, functions as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check ledger/transaction index bounds for a parquet dataset.")
    p.add_argument("--path", required=True, help="Parquet path to inspect")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("empirical_check_parquet_bounds").getOrCreate()
    df = spark.read.parquet(args.path)

    df.agg(
        F.min("ledger_index").alias("min_ledger_index"),
        F.max("ledger_index").alias("max_ledger_index"),
    ).show()

    df.agg(
        F.min("transaction_index").alias("min_transaction_index"),
        F.max("transaction_index").alias("max_transaction_index"),
    ).show()

    spark.stop()


if __name__ == "__main__":
    main()
