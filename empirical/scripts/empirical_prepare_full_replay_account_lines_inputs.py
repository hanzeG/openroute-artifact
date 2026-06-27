#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


XRP = "XRP"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare account_lines fetch targets needed by full-ledger tx_prebook replay. "
            "This covers same-ledger OfferCreate transactions whose TakerGets is an IOU, "
            "so replay can seed maker fundedness from ledger-1 trust lines before applying "
            "in-ledger metadata updates."
        )
    )
    parser.add_argument("--metadata-ndjson", required=True, help="Full-ledger tx metadata NDJSON")
    parser.add_argument(
        "--required-tx-csv",
        required=True,
        help="Target required_tx.csv. Its ledgers define the replay ledger set.",
    )
    parser.add_argument("--output-csv", required=True, help="Output CSV for account_lines fetch")
    parser.add_argument(
        "--include-all-metadata-ledgers",
        action="store_true",
        help="Do not filter to ledgers present in required_tx.csv.",
    )
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args()


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_required_ledger_cutoffs(path: Path) -> dict[int, int | None]:
    out: dict[int, int | None] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_ledger = _as_str(row.get("ledger_index"))
            if not raw_ledger:
                continue
            ledger_index = int(raw_ledger)
            raw_tx_index = _as_str(row.get("transaction_index"))
            if not raw_tx_index:
                out[ledger_index] = None
                continue
            tx_index = int(raw_tx_index)
            if ledger_index not in out:
                out[ledger_index] = tx_index
                continue
            prev = out[ledger_index]
            if prev is None:
                continue
            out[ledger_index] = max(prev, tx_index)
    if not out:
        raise RuntimeError(f"no ledger_index values found in {path}")
    return out


def _result_obj(row: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result")
    if isinstance(result, dict):
        return result
    return row


def _ledger_index(tx: dict[str, Any]) -> int | None:
    for key in ("ledger_index", "inLedger"):
        raw = tx.get(key)
        if raw is not None:
            return int(raw)
    meta = tx.get("meta")
    if isinstance(meta, dict) and meta.get("ledger_index") is not None:
        return int(meta["ledger_index"])
    return None


def _transaction_index(tx: dict[str, Any]) -> int | None:
    meta = tx.get("meta")
    if isinstance(meta, dict) and meta.get("TransactionIndex") is not None:
        return int(meta["TransactionIndex"])
    meta = tx.get("metaData")
    if isinstance(meta, dict) and meta.get("TransactionIndex") is not None:
        return int(meta["TransactionIndex"])
    return None


def _iou_taker_gets(tx: dict[str, Any]) -> tuple[str, str] | None:
    taker_gets = tx.get("TakerGets")
    if not isinstance(taker_gets, dict):
        return None
    currency = _as_str(taker_gets.get("currency"))
    issuer = _as_str(taker_gets.get("issuer"))
    if not currency or not issuer or currency == XRP:
        return None
    return currency, issuer


def main() -> None:
    args = _parse_args()
    metadata_path = Path(args.metadata_ndjson)
    required_tx_path = Path(args.required_tx_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    ledger_cutoffs = _load_required_ledger_cutoffs(required_tx_path)
    rows_by_key: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    scanned = 0
    matched_offercreate = 0
    emitted_candidates = 0

    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            scanned += 1
            if args.progress_every > 0 and scanned % args.progress_every == 0:
                print(
                    f"[prepare-full-replay-account-lines] scanned={scanned} "
                    f"matched_offercreate={matched_offercreate} unique_keys={len(rows_by_key)}",
                    flush=True,
                )
            obj = json.loads(payload)
            tx = _result_obj(obj)
            ledger_index = _ledger_index(tx)
            if ledger_index is None:
                continue
            tx_index = _transaction_index(tx)
            if not args.include_all_metadata_ledgers:
                max_target_tx_index = ledger_cutoffs.get(ledger_index)
                if max_target_tx_index is None and ledger_index not in ledger_cutoffs:
                    continue
                if (
                    max_target_tx_index is not None
                    and tx_index is not None
                    and tx_index > max_target_tx_index
                ):
                    continue
            if _as_str(tx.get("TransactionType")) != "OfferCreate":
                continue
            account = _as_str(tx.get("Account"))
            if not account:
                continue
            iou = _iou_taker_gets(tx)
            if iou is None:
                continue

            matched_offercreate += 1
            currency, issuer = iou
            boundary_ledger_index = ledger_index - 1
            key = (boundary_ledger_index, account, currency, issuer)
            candidate = {
                "boundary_ledger_index": boundary_ledger_index,
                "owner_account": account,
                "currency": currency,
                "issuer": issuer,
                "first_target_ledger_index": ledger_index,
                "first_tx_hash": _as_str(tx.get("hash")).upper(),
                "first_transaction_index": tx_index if tx_index is not None else "",
                "reason": "full_replay_offercreate_iou_taker_gets",
            }
            existing = rows_by_key.get(key)
            if existing is None:
                rows_by_key[key] = candidate
                emitted_candidates += 1
                continue
            existing_index = existing.get("first_transaction_index")
            existing_sort = (
                int(existing_index)
                if str(existing_index).strip()
                else 1_000_000_000
            )
            candidate_sort = tx_index if tx_index is not None else 1_000_000_000
            if candidate_sort < existing_sort:
                rows_by_key[key] = {
                    **candidate,
                    "reason": "full_replay_offercreate_iou_taker_gets",
                }

    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    fieldnames = [
        "boundary_ledger_index",
        "owner_account",
        "currency",
        "issuer",
        "first_target_ledger_index",
        "first_transaction_index",
        "first_tx_hash",
        "reason",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "scope": "full_replay_account_lines_inputs",
        "metadata_ndjson": str(metadata_path),
        "required_tx_csv": str(required_tx_path),
        "output_csv": str(output_csv),
        "required_ledger_count": len(ledger_cutoffs),
        "required_ledgers_with_tx_index_cutoff": sum(
            1 for cutoff in ledger_cutoffs.values() if cutoff is not None
        ),
        "include_all_metadata_ledgers": bool(args.include_all_metadata_ledgers),
        "scanned_metadata_rows": scanned,
        "matched_offercreate_iou_taker_gets": matched_offercreate,
        "unique_account_line_keys": len(rows),
        "emitted_candidates": emitted_candidates,
        "sample": rows[:20],
    }
    manifest_path = output_csv.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
