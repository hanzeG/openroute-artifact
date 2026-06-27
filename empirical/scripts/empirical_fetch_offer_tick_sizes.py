#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


XRP = "XRP"


def _default_rpc() -> str:
    for key in ("XRPL_RIPPLE_S2_RPC", "XRPL_RIPPLE_S1_RPC", "XRPL_CLUSTER_RPC", "XRPL_EXTRA_RPC_1"):
        value = os.environ.get(key)
        if value:
            return str(value)
    return "https://s2.ripple.com:51234/"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch issuer TickSize snapshots required for OfferCreate replay."
    )
    parser.add_argument("--required-tx", required=True)
    parser.add_argument("--metadata-ndjson", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--rpc", default=_default_rpc())
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def _required_hashes(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tx_hash = str(row.get("tx_hash") or row.get("transaction_hash") or "").strip().upper()
            if tx_hash:
                out.add(tx_hash)
    if not out:
        raise RuntimeError(f"no tx hashes in required tx file: {path}")
    return out


def _tx_hash(obj: dict[str, Any]) -> str:
    result = obj.get("result") if isinstance(obj.get("result"), dict) else obj
    return str(result.get("hash") or obj.get("tx_hash") or "").strip().upper()


def _result(obj: dict[str, Any]) -> dict[str, Any]:
    return obj.get("result") if isinstance(obj.get("result"), dict) else obj


def _ledger_index(tx: dict[str, Any]) -> int:
    for key in ("ledger_index", "inLedger"):
        if tx.get(key) is not None:
            return int(tx[key])
    meta = tx.get("meta") if isinstance(tx.get("meta"), dict) else {}
    if meta.get("ledger_index") is not None:
        return int(meta["ledger_index"])
    raise RuntimeError("metadata row missing ledger_index")


def _iou_issuers_from_offercreate(tx: dict[str, Any]) -> set[str]:
    if str(tx.get("TransactionType") or "").upper() != "OFFERCREATE":
        return set()
    out: set[str] = set()
    for key in ("TakerGets", "TakerPays"):
        amount = tx.get(key)
        if isinstance(amount, dict):
            currency = str(amount.get("currency") or "")
            issuer = str(amount.get("issuer") or "")
            if issuer and currency != XRP:
                out.add(issuer)
    return out


def _load_targets(*, required_tx: Path, metadata_ndjson: Path) -> dict[int, set[str]]:
    required = _required_hashes(required_tx)
    targets: dict[int, set[str]] = {}
    seen = 0
    with metadata_ndjson.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            tx_hash = _tx_hash(obj)
            if tx_hash not in required:
                continue
            seen += 1
            tx = _result(obj)
            issuers = _iou_issuers_from_offercreate(tx)
            if issuers:
                targets.setdefault(_ledger_index(tx) - 1, set()).update(issuers)
    if seen == 0:
        raise RuntimeError("no required tx metadata rows found")
    return targets


class _Limiter:
    def __init__(self, rps: float) -> None:
        self._interval = 1.0 / max(rps, 0.1)
        self._next = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if wait:
            time.sleep(wait)


def _rpc(url: str, payload: dict[str, Any], *, timeout: float, retries: int, limiter: _Limiter) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    last: str | None = None
    for attempt in range(retries + 1):
        try:
            limiter.acquire()
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8")).get("result") or {}
            if result.get("status") != "success":
                raise RuntimeError(str(result.get("error") or result))
            return result
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            if attempt >= retries:
                break
            time.sleep(min(0.25 * (2**attempt), 4.0))
    raise RuntimeError(last or "unknown rpc failure")


def _fetch_tick_size(
    *,
    rpc: str,
    ledger_index: int,
    issuer: str,
    timeout: float,
    retries: int,
    limiter: _Limiter,
) -> tuple[int, str, int | None]:
    result = _rpc(
        rpc,
        {
            "method": "account_info",
            "params": [{"account": issuer, "ledger_index": ledger_index}],
        },
        timeout=timeout,
        retries=retries,
        limiter=limiter,
    )
    raw = (result.get("account_data") or {}).get("TickSize")
    return ledger_index, issuer, int(raw) if raw is not None else None


def main() -> None:
    args = _parse_args()
    targets = _load_targets(required_tx=Path(args.required_tx), metadata_ndjson=Path(args.metadata_ndjson))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not targets:
        manifest = {"scope": "offer_tick_sizes", "target_count": 0, "snapshot_files": []}
        (outdir / "offer_tick_sizes_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return

    limiter = _Limiter(rps=max(1.0, float(args.workers)))
    mapping: dict[int, dict[str, int | None]] = {ledger: {} for ledger in targets}
    jobs = [
        (ledger_index, issuer)
        for ledger_index, issuers in sorted(targets.items())
        for issuer in sorted(issuers)
    ]
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [
            executor.submit(
                _fetch_tick_size,
                rpc=args.rpc,
                ledger_index=ledger,
                issuer=issuer,
                timeout=args.timeout,
                retries=args.retries,
                limiter=limiter,
            )
            for ledger, issuer in jobs
        ]
        for future in as_completed(futures):
            ledger, issuer, tick_size = future.result()
            mapping[ledger][issuer] = tick_size

    snapshot_files: list[str] = []
    for ledger, tick_sizes in sorted(mapping.items()):
        path = outdir / f"offer_tick_sizes_at_ledger_{ledger}.json"
        path.write_text(json.dumps({"ledger_index": ledger, "tick_sizes": tick_sizes}, indent=2), encoding="utf-8")
        snapshot_files.append(str(path))
    manifest = {
        "scope": "offer_tick_sizes",
        "target_count": len(jobs),
        "snapshot_count": len(snapshot_files),
        "snapshot_files": snapshot_files,
    }
    (outdir / "offer_tick_sizes_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
