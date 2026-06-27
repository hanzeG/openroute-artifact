#!/usr/bin/env python3
"""Reconstruct real intra-tx AMM/CLOB execution paths from XRPL tx metadata."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from pyspark.sql import SparkSession, functions as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reconstruct real tx path sequence (AMM/CLOB) from XRPL tx metadata."
    )
    p.add_argument("--amm-swaps", required=True, help="AMM swaps parquet path")
    p.add_argument("--ledger-start", type=int, default=None, help="Ledger lower bound (inclusive)")
    p.add_argument("--ledger-end", type=int, default=None, help="Ledger upper bound (inclusive)")
    p.add_argument("--rpc", default="https://s1.ripple.com:51234/", help="XRPL JSON-RPC endpoint")
    p.add_argument("--pair", default="rlusd_xrp", help="Pair key used in output naming")
    p.add_argument("--output-dir", default=None, help="Output directory")
    p.add_argument("--max-workers", type=int, default=12, help="Parallel metadata fetch workers")
    p.add_argument("--retry-n", type=int, default=3, help="Retry attempts per tx")
    p.add_argument("--retry-sleep-sec", type=float, default=0.6, help="Initial retry sleep seconds")
    p.add_argument("--limit", type=int, default=None, help="Optional cap on tx count for debugging")
    return p.parse_args()


def _assert_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input path: {path}")


def _fetch_tx(rpc: str, tx_hash: str, retry_n: int, retry_sleep_sec: float) -> dict[str, Any]:
    payload = {
        "method": "tx",
        "params": [{"transaction": tx_hash, "binary": False}],
    }
    data = json.dumps(payload).encode("utf-8")

    last_error: str | None = None
    for attempt in range(1, retry_n + 1):
        req = urllib.request.Request(
            rpc,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                out = json.loads(resp.read().decode("utf-8"))
                return out
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
            time.sleep(retry_sleep_sec * (2 ** (attempt - 1)))
    raise RuntimeError(f"tx fetch failed after retries: {last_error}")


def _changed_amount(prev: Any, final: Any) -> bool:
    return prev is not None and final is not None and prev != final


def _classify_node(node: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    node_kind = next((k for k in ("CreatedNode", "ModifiedNode", "DeletedNode") if k in node), None)
    if node_kind is None:
        return None, {}
    body = node[node_kind]
    ledger_type = body.get("LedgerEntryType")
    prev = body.get("PreviousFields", {}) or {}
    final = body.get("FinalFields", {}) or body.get("NewFields", {}) or {}

    details = {
        "node_type": node_kind,
        "ledger_entry_type": ledger_type,
    }

    if ledger_type == "AMM":
        return "AMM", details

    if ledger_type == "Offer":
        changed_gets = _changed_amount(prev.get("TakerGets"), final.get("TakerGets"))
        changed_pays = _changed_amount(prev.get("TakerPays"), final.get("TakerPays"))
        removed_offer = node_kind == "DeletedNode"
        if changed_gets or changed_pays or removed_offer:
            return "CLOB", details

    return None, details


def _collapse_steps(kinds: list[str]) -> list[str]:
    out: list[str] = []
    for k in kinds:
        if not out or out[-1] != k:
            out.append(k)
    return out


def _extract_real_path(meta: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    affected = meta.get("AffectedNodes", []) or []
    steps: list[dict[str, Any]] = []
    kinds: list[str] = []

    for idx, node in enumerate(affected):
        kind, details = _classify_node(node)
        if kind is None:
            continue
        steps.append(
            {
                "idx": idx,
                "kind": kind,
                "node_type": details.get("node_type"),
                "ledger_entry_type": details.get("ledger_entry_type"),
            }
        )
        kinds.append(kind)
    return steps, _collapse_steps(kinds)


def main() -> None:
    args = parse_args()
    _assert_exists(args.amm_swaps)

    spark = SparkSession.builder.appName("empirical_reconstruct_real_paths").getOrCreate()
    amm = spark.read.parquet(args.amm_swaps)
    if args.ledger_start is not None:
        amm = amm.filter(F.col("ledger_index") >= F.lit(int(args.ledger_start)))
    if args.ledger_end is not None:
        amm = amm.filter(F.col("ledger_index") <= F.lit(int(args.ledger_end)))

    tx_rows = (
        amm.select("transaction_hash", "ledger_index", "transaction_index")
        .where(F.col("transaction_hash").isNotNull())
        .distinct()
        .orderBy(F.col("ledger_index").asc(), F.col("transaction_index").asc_nulls_last())
        .collect()
    )
    if args.limit is not None:
        tx_rows = tx_rows[: int(args.limit)]
    if not tx_rows:
        raise RuntimeError("No tx rows found in selected window.")

    if args.output_dir:
        out_dir = args.output_dir
    else:
        lmin = args.ledger_start if args.ledger_start is not None else "auto"
        lmax = args.ledger_end if args.ledger_end is not None else "auto"
        out_dir = os.path.join("artifacts", "real_paths", args.pair, f"ledger_{lmin}_{lmax}")
    os.makedirs(out_dir, exist_ok=True)
    out_ndjson = os.path.join(out_dir, "real_path_sequence.ndjson")
    out_manifest = os.path.join(out_dir, "manifest.json")

    results: list[dict[str, Any]] = []

    def _one(row: Any) -> dict[str, Any]:
        tx_hash = str(row["transaction_hash"])
        ledger_index = int(row["ledger_index"]) if row["ledger_index"] is not None else None
        tx_index = int(row["transaction_index"]) if row["transaction_index"] is not None else None
        try:
            resp = _fetch_tx(args.rpc, tx_hash, args.retry_n, args.retry_sleep_sec)
            result = resp.get("result", {}) or {}
            validated = bool(result.get("validated"))
            meta = result.get("meta", {}) or {}
            steps, collapsed = _extract_real_path(meta)
            return {
                "tx_hash": tx_hash,
                "ledger_index": ledger_index,
                "transaction_index": tx_index,
                "validated": validated,
                "path_steps": steps,
                "path_kinds_collapsed": collapsed,
                "path_sig_real": "+".join(collapsed),
                "clob_node_count": sum(1 for s in steps if s["kind"] == "CLOB"),
                "amm_node_count": sum(1 for s in steps if s["kind"] == "AMM"),
                "error": None,
            }
        except Exception as exc:
            return {
                "tx_hash": tx_hash,
                "ledger_index": ledger_index,
                "transaction_index": tx_index,
                "validated": False,
                "path_steps": [],
                "path_kinds_collapsed": [],
                "path_sig_real": "",
                "clob_node_count": 0,
                "amm_node_count": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        futures = [ex.submit(_one, r) for r in tx_rows]
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 100 == 0 or done == total:
                print(f"[progress] {done}/{total}")

    results.sort(key=lambda x: ((x["ledger_index"] or 0), (x["transaction_index"] or -1)))

    with open(out_ndjson, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in results if not r["error"]]
    fail = [r for r in results if r["error"]]

    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(
            {
                "script": "empirical_reconstruct_real_paths.py",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": {
                    "amm_swaps": args.amm_swaps,
                    "ledger_start": args.ledger_start,
                    "ledger_end": args.ledger_end,
                    "rpc": args.rpc,
                },
                "outputs": {"real_path_sequence_ndjson": out_ndjson},
                "stats": {
                    "tx_total": len(results),
                    "tx_ok": len(ok),
                    "tx_failed": len(fail),
                    "with_amm_nodes": sum(1 for r in ok if r["amm_node_count"] > 0),
                    "with_clob_nodes": sum(1 for r in ok if r["clob_node_count"] > 0),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    spark.stop()
    print(f"[done] wrote {len(results)} tx paths -> {out_ndjson}")


if __name__ == "__main__":
    main()
