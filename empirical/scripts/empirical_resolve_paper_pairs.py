#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve paper pair currency/issuer identifiers from Delta Sharing AMM rows."
    )
    parser.add_argument("--pair-config", required=True)
    parser.add_argument(
        "--share-profile",
        default=os.environ.get("XRPL_SHARE_PROFILE", "data/config.share"),
    )
    parser.add_argument("--share", default="ripple-ubri-share")
    parser.add_argument("--schema", default="ripplex")
    parser.add_argument("--table-amm", default="fact_amm_swaps")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _ds_url(profile: str, share: str, schema: str, table: str) -> str:
    return f"{profile}#{share}.{schema}.{table}"


def _validate_share_profile(profile: str) -> None:
    path = Path(profile)
    if not path.exists():
        raise SystemExit(f"share profile not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"share profile is not valid JSON: {path} ({exc})")
    exp_raw = payload.get("expirationTime")
    if exp_raw:
        exp = datetime.fromisoformat(str(exp_raw).replace("Z", "+00:00"))
        if exp <= datetime.now(timezone.utc):
            raise SystemExit(f"share profile is expired: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        print(f"[warn] {path} is group/other-readable; recommended: chmod 600 {path}")


def _ascii_currency_hex(label: str) -> str | None:
    try:
        raw = label.encode("ascii")
    except UnicodeEncodeError:
        return None
    if not raw or len(raw) > 20:
        return None
    return (raw.hex().upper() + ("00" * (20 - len(raw))))


def _currency_candidates(entry: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for value in entry.get("base_currency_candidates") or []:
        if value is not None:
            candidates.append(str(value))
    label = str(entry["base_label"])
    candidates.append(label)
    hex_value = _ascii_currency_hex(label)
    if hex_value:
        candidates.append(hex_value)
    out: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _candidate_df(df, *, counter_currency: str, counter_issuer: str, candidates: list[str]):
    from pyspark.sql import functions as F

    ciss = F.lit(counter_issuer)
    candidate_array = [F.lit(item) for item in candidates]
    asset1_is_counter = (
        (F.col("amm_asset_currency") == F.lit(counter_currency))
        & (F.coalesce(F.col("amm_asset_issuer"), F.lit("")) == ciss)
    )
    asset2_is_counter = (
        (F.col("amm_asset2_currency") == F.lit(counter_currency))
        & (F.coalesce(F.col("amm_asset2_issuer"), F.lit("")) == ciss)
    )
    return (
        df.withColumn(
            "base_currency",
            F.when(asset1_is_counter, F.col("amm_asset2_currency")).otherwise(F.col("amm_asset_currency")),
        )
        .withColumn(
            "base_issuer",
            F.when(asset1_is_counter, F.col("amm_asset2_issuer")).otherwise(F.col("amm_asset_issuer")),
        )
        .filter(asset1_is_counter | asset2_is_counter)
        .filter(F.col("base_currency").isin(candidates))
        .select("base_currency", "base_issuer")
    )


def main() -> None:
    args = _parse_args()
    config_path = Path(args.pair_config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ledger_start = int(config["ledger_start"])
    ledger_end = int(config["ledger_end"])
    counter_currency = str(config.get("counter_currency") or "XRP")
    counter_issuer = str(config.get("counter_issuer") or "")

    _validate_share_profile(args.share_profile)
    from pyspark.sql import SparkSession, functions as F

    spark = SparkSession.builder.appName("empirical_resolve_paper_pairs").getOrCreate()
    amm = spark.read.format("deltaSharing").load(
        _ds_url(args.share_profile, args.share, args.schema, args.table_amm)
    )
    amm = amm.filter((F.col("ledger_index") >= ledger_start) & (F.col("ledger_index") <= ledger_end))

    resolved: list[dict[str, Any]] = []
    for entry in config["pairs"]:
        pair = str(entry["pair"])
        base_label = str(entry["base_label"])
        candidates = _currency_candidates(entry)
        grouped = (
            _candidate_df(
                amm,
                counter_currency=counter_currency,
                counter_issuer=counter_issuer,
                candidates=candidates,
            )
            .groupBy("base_currency", "base_issuer")
            .agg(F.count(F.lit(1)).alias("amm_rows"))
            .orderBy(F.col("amm_rows").desc(), F.col("base_currency").asc(), F.col("base_issuer").asc())
            .limit(2)
            .collect()
        )
        if not grouped:
            raise RuntimeError(f"no AMM rows found for pair={pair} candidates={candidates}")
        best = grouped[0].asDict()
        if len(grouped) > 1 and int(grouped[1].asDict()["amm_rows"]) == int(best["amm_rows"]):
            raise RuntimeError(f"ambiguous pair resolution for pair={pair}; add base_currency_candidates")
        resolved.append(
            {
                "pair": pair,
                "pair_label": f"{base_label}_XRP",
                "base_label": base_label,
                "base_currency": str(best["base_currency"]),
                "base_issuer": str(best["base_issuer"] or ""),
                "counter_currency": counter_currency,
                "counter_issuer": counter_issuer,
                "ledger_start": ledger_start,
                "ledger_end": ledger_end,
                "window": f"ledger_{ledger_start}_{ledger_end}",
                "amm_rows_for_resolution": int(best["amm_rows"]),
            }
        )

    payload = {
        "source_pair_config": str(config_path),
        "share": args.share,
        "schema": args.schema,
        "table_amm": args.table_amm,
        "ledger_start": ledger_start,
        "ledger_end": ledger_end,
        "pairs": resolved,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
