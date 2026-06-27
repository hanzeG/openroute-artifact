#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from decimal import Decimal, getcontext

from pyspark.sql import SparkSession, Window, functions as F

getcontext().prec = 28

RUSD_HEX = "524C555344000000000000000000000000000000"
XRP = "XRP"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build AMM-only model input JSON from exported AMM/CLOB/fees parquet windows."
    )
    p.add_argument("--input-amm", "--amm", dest="input_amm", required=True, help="AMM swaps parquet path")
    p.add_argument("--input-clob", "--clob", dest="input_clob", required=True, help="CLOB legs parquet path")
    p.add_argument("--input-fees", "--fees", dest="input_fees", required=True, help="AMM fees parquet path")
    p.add_argument("--seed", type=int, default=7, help="Random seed for anchor selection")
    p.add_argument(
        "--use-discounted-fee",
        action="store_true",
        help="Use discounted_fee (ex-post) instead of trading_fee (ex-ante)",
    )
    p.add_argument("--pair", default="rlusd_xrp", help="Pair key used in output naming")
    p.add_argument("--ledger-start", type=int, default=None, help="Ledger lower bound in output naming")
    p.add_argument("--ledger-end", type=int, default=None, help="Ledger upper bound in output naming")
    p.add_argument("--output-dir", default=None, help="Output directory (default: artifacts/model_input/<pair>/<window>)")
    p.add_argument("--output-json", "--out-json", dest="output_json", default=None, help="Output JSON path")
    return p.parse_args()


def to_decimal(x) -> Decimal:
    if x is None:
        return Decimal("0")
    return Decimal(str(x))


def pretty_time(ts) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def default_output_paths(args: argparse.Namespace) -> tuple[str, str]:
    if args.output_json:
        output_json = args.output_json
        output_dir = args.output_dir or os.path.dirname(os.path.abspath(output_json))
    else:
        if args.ledger_start is not None and args.ledger_end is not None:
            window = f"ledger_{args.ledger_start}_{args.ledger_end}"
        else:
            window = f"adhoc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')}"
        output_dir = args.output_dir or os.path.join("artifacts", "model_input", args.pair, window)
        output_json = os.path.join(
            output_dir,
            f"model_input_{args.pair}_{window}_v1.json",
        )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, output_json


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("empirical_build_model_input").getOrCreate()

    output_dir, output_json = default_output_paths(args)

    amm = spark.read.parquet(args.input_amm)
    clob = spark.read.parquet(args.input_clob)
    fees = spark.read.parquet(args.input_fees)

    amm_tx = amm.select(F.col("transaction_hash").alias("tx_hash")).distinct()
    clob_tx = clob.select("tx_hash").distinct()
    dup_tx = amm_tx.join(clob_tx, on="tx_hash", how="inner").cache()
    dup_cnt = dup_tx.count()

    amm_pure = amm.join(dup_tx, amm.transaction_hash == dup_tx.tx_hash, how="left_anti").cache()
    amm_pure_cnt = amm_pure.count()

    w = Window.orderBy(F.col("close_time_datetime").asc())
    amm_ranked = amm_pure.withColumn("rn", F.row_number().over(w))
    n = amm_pure_cnt
    low = int(n * 0.2) + 1
    high = int(n * 0.8)
    candidates = amm_ranked.filter((F.col("rn") >= low) & (F.col("rn") <= high)).cache()
    cand_cnt = candidates.count()
    random.seed(args.seed)
    pick_rn = random.randint(low, high)

    anchor_rows = candidates.filter(F.col("rn") == pick_rn).collect()
    if not anchor_rows:
        raise RuntimeError("No anchor selected; check window/candidate filtering.")
    anchor = anchor_rows[0]

    prev_row = (
        amm_pure.filter(F.col("close_time_datetime") < anchor["close_time_datetime"])
        .orderBy(F.col("close_time_datetime").desc())
        .select("transaction_fee")
        .limit(1)
        .collect()
    )
    if prev_row:
        net_fee_approx_drops = Decimal(str(prev_row[0]["transaction_fee"] or 0))
        net_fee_approx_source = "prev_swap_fee"
    else:
        net_fee_approx_drops = Decimal("0")
        net_fee_approx_source = "no_prev_swap"
    net_fee_approx_xrp = net_fee_approx_drops / Decimal("1000000")

    in_cur = anchor["asset_in_currency"]
    out_cur = anchor["asset_out_currency"]
    side1_cur = anchor["amm_asset_currency"]
    side2_cur = anchor["amm_asset2_currency"]
    side1_bal_before = to_decimal(anchor["amm_asset_balance_before"])
    side2_bal_before = to_decimal(anchor["amm_asset2_balance_before"])
    side1_bal_after = to_decimal(anchor["amm_asset_balance_after"])
    side2_bal_after = to_decimal(anchor["amm_asset2_balance_after"])

    if in_cur == side1_cur and out_cur == side2_cur:
        x_currency, y_currency = side1_cur, side2_cur
        x_reserve, y_reserve = side1_bal_before, side2_bal_before
        x_after, y_after = side1_bal_after, side2_bal_after
    elif in_cur == side2_cur and out_cur == side1_cur:
        x_currency, y_currency = side2_cur, side1_cur
        x_reserve, y_reserve = side2_bal_before, side1_bal_before
        x_after, y_after = side2_bal_after, side1_bal_after
    else:
        raise ValueError(
            f"Anchor currencies do not match AMM pool sides: "
            f"in={in_cur}, out={out_cur}, side1={side1_cur}, side2={side2_cur}"
        )
    real_post_spot_price_inout = (x_after / y_after) if y_after != 0 else None

    fee_row = (
        fees.filter(
            (F.col("transaction_hash") == anchor["transaction_hash"])
            & (F.col("amm_account") == anchor["amm_account"])
        )
        .select("trading_fee", "discounted_fee")
        .limit(1)
        .collect()
    )
    fee_fallback = Decimal("0.003")
    trading_fee = fee_row[0]["trading_fee"] if fee_row else None
    discounted_fee = fee_row[0]["discounted_fee"] if fee_row else None
    if args.use_discounted_fee and discounted_fee is not None:
        fee_dec = to_decimal(discounted_fee)
        fee_src = "discounted_fee (ex-post)"
    elif trading_fee is not None:
        fee_dec = to_decimal(trading_fee)
        fee_src = "trading_fee (ex-ante)"
    elif discounted_fee is not None:
        fee_dec = to_decimal(discounted_fee)
        fee_src = "discounted_fee (fallback)"
    else:
        fee_dec = fee_fallback
        fee_src = "fallback_const"

    target_out = to_decimal(anchor["asset_out_value"])
    real_in_val = to_decimal(anchor["asset_in_value"])
    real_out_val = to_decimal(anchor["asset_out_value"])
    avg_price_real = (real_in_val / real_out_val) if real_out_val != 0 else None

    model_input = {
        "amm": {
            "x_reserve": str(x_reserve),
            "y_reserve": str(y_reserve),
            "x_currency": x_currency,
            "y_currency": y_currency,
            "fee": str(fee_dec),
            "fee_source": fee_src,
        },
        "trade": {
            "target_out": str(target_out),
            "out_currency": out_cur,
            "in_currency": in_cur,
        },
        "anchor_meta": {
            "T": pretty_time(anchor["close_time_datetime"]),
            "ledger_index": int(anchor["ledger_index"]),
            "transaction_index": int(anchor["transaction_index"]),
            "tx_hash": anchor["transaction_hash"],
            "amm_account": anchor["amm_account"],
        },
        "real_world": {
            "asset_in_value": str(real_in_val),
            "asset_out_value": str(real_out_val),
            "avg_price_real": str(avg_price_real) if avg_price_real is not None else None,
            "post_x_reserve": str(x_after),
            "post_y_reserve": str(y_after),
            "post_spot_price_inout": str(real_post_spot_price_inout) if real_post_spot_price_inout is not None else None,
        },
        "window_meta": {
            "dup_tx_count": int(dup_cnt),
            "pure_amm_count": int(amm_pure_cnt),
            "candidate_count": int(cand_cnt),
            "seed": int(args.seed),
            "use_discounted_fee": bool(args.use_discounted_fee),
            "net_fee_approx_drops": str(net_fee_approx_drops),
            "net_fee_approx_xrp": str(net_fee_approx_xrp),
            "net_fee_approx_source": net_fee_approx_source,
        },
    }

    ensure_parent(output_json)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(model_input, f, ensure_ascii=False, indent=2)
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "script": "empirical_build_model_input.py",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "pair": args.pair,
                "window": {"ledger_start": args.ledger_start, "ledger_end": args.ledger_end},
                "inputs": {
                    "amm_swaps": args.input_amm,
                    "clob_legs": args.input_clob,
                    "amm_fees": args.input_fees,
                },
                "output_json": output_json,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[ok] Wrote model input: {output_json}")
    spark.stop()


if __name__ == "__main__":
    main()
