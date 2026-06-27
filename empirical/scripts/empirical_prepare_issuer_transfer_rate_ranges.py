#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Populate dataset_manifest issuer_transfer_rate_ranges from raw XRPL AccountRoot.TransferRate."
    )
    p.add_argument("--dataset-root", required=True)
    p.add_argument(
        "--issuer",
        action="append",
        default=[],
        help="Issuer key '<currency>:<account>'. Defaults to existing manifest issuer_transfer_rates keys.",
    )
    p.add_argument("--rpc-url", default="https://s2.ripple.com:51234/")
    p.add_argument("--write", action="store_true", help="Write ranges back to dataset_manifest.json.")
    return p.parse_args()


def _rpc(url: str, method: str, params: list[dict[str, Any]]) -> dict[str, Any]:
    payload = json.dumps({"method": method, "params": params}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    result = data.get("result") or {}
    if result.get("status") != "success":
        raise RuntimeError(f"{method} failed: {result.get('error') or result}")
    return result


def _account_transfer_rate(url: str, account: str, ledger_index: int) -> int:
    result = _rpc(url, "account_info", [{"account": account, "ledger_index": ledger_index}])
    raw = result.get("account_data", {}).get("TransferRate", 1_000_000_000)
    if not isinstance(raw, int):
        raise RuntimeError(f"unexpected TransferRate at ledger {ledger_index}: {raw!r}")
    return raw


def _ledger_bounds(ledger_list: Path) -> tuple[int, int]:
    first: int | None = None
    last: int | None = None
    with ledger_list.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            value = int(text)
            if first is None or value < first:
                first = value
            if last is None or value > last:
                last = value
    if first is None or last is None:
        raise RuntimeError(f"empty ledger list: {ledger_list}")
    return first, last


def _target_ledger_bounds(required_tx_csv: Path) -> tuple[int, int]:
    first: int | None = None
    last: int | None = None
    with required_tx_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("ledger_index") or row.get("ledger")
            if raw is None:
                raise RuntimeError(f"required tx row missing ledger_index: {row}")
            value = int(raw)
            if first is None or value < first:
                first = value
            if last is None or value > last:
                last = value
    if first is None or last is None:
        raise RuntimeError(f"empty required tx csv: {required_tx_csv}")
    return first, last


def _transfer_rate_account_sets(metadata_ndjson: Path, issuer: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with metadata_ndjson.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if "AccountSet" not in line or issuer not in line:
                continue
            payload = json.loads(line)
            tx = payload.get("result") or {}
            if tx.get("TransactionType") != "AccountSet" or tx.get("Account") != issuer:
                continue
            if "TransferRate" not in tx:
                continue
            ledger_index = tx.get("ledger_index") or payload.get("ledger_index")
            if ledger_index is None:
                raise RuntimeError(f"AccountSet missing ledger_index at line {line_no}")
            out.append(
                {
                    "ledger_index": int(ledger_index),
                    "transaction_index": (tx.get("meta") or {}).get("TransactionIndex"),
                    "tx_hash": tx.get("hash") or payload.get("tx_hash"),
                    "TransferRate": tx.get("TransferRate"),
                }
            )
    return out


def _issuer_keys(manifest: dict[str, Any], explicit: list[str]) -> list[str]:
    if explicit:
        return explicit
    raw = manifest.get("issuer_transfer_rates") or manifest.get("issuer_transfer_rate_ranges") or {}
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError("no issuer keys found; pass --issuer '<currency>:<account>'")
    return sorted(str(key) for key in raw)


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root)
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_ndjson = Path(manifest.get("metadata_ndjson") or dataset_root / "tx_metadata_full_merged.ndjson")
    ledger_list = Path(manifest.get("ledger_list") or dataset_root / "ledger_list.txt")
    required_tx_csv = Path(manifest.get("required_tx_csv") or dataset_root / "required_tx.csv")
    if required_tx_csv.exists():
        first_ledger, last_ledger = _target_ledger_bounds(required_tx_csv)
        bounds_source = str(required_tx_csv)
    else:
        first_ledger, last_ledger = _ledger_bounds(ledger_list)
        bounds_source = str(ledger_list)

    ranges: dict[str, list[dict[str, Any]]] = {}
    provenance: dict[str, Any] = {
        "source": "XRPL account_info plus full metadata AccountSet scan",
        "rpc_url": args.rpc_url,
        "metadata_ndjson": str(metadata_ndjson),
        "ledger_list": str(ledger_list),
        "required_tx_csv": str(required_tx_csv),
        "bounds_source": bounds_source,
    }
    for key in _issuer_keys(manifest, args.issuer):
        if ":" not in key:
            raise RuntimeError(f"invalid issuer key {key!r}; expected '<currency>:<issuer>'")
        _, issuer = key.rsplit(":", 1)
        account_sets = _transfer_rate_account_sets(metadata_ndjson, issuer)
        if account_sets:
            raise RuntimeError(
                f"{key} has AccountSet TransferRate changes inside the dataset window; "
                f"tx-index-aware transfer-rate input is required: {account_sets[:5]}"
            )
        start_rate = _account_transfer_rate(args.rpc_url, issuer, first_ledger)
        end_rate = _account_transfer_rate(args.rpc_url, issuer, last_ledger)
        if start_rate != end_rate:
            raise RuntimeError(
                f"{key} TransferRate differs between ledger bounds: "
                f"{first_ledger}={start_rate}, {last_ledger}={end_rate}"
            )
        ranges[key] = [
            {
                "first_ledger": first_ledger,
                "last_ledger": last_ledger,
                "TransferRate": start_rate,
            }
        ]

    manifest["issuer_transfer_rate_ranges"] = ranges
    manifest["issuer_transfer_rate_ranges_provenance"] = provenance
    if args.write:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(json.dumps({"dataset_manifest": str(manifest_path), "issuer_transfer_rate_ranges": ranges}, indent=2))


if __name__ == "__main__":
    main()
