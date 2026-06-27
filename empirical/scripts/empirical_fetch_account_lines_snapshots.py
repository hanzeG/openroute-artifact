#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import requests


def _default_rpc() -> str:
    for key in (
        "XRPL_RIPPLE_S2_RPC",
        "XRPL_RIPPLE_S1_RPC",
        "XRPL_CLUSTER_RPC",
        "XRPL_EXTRA_RPC_1",
        "XRPL_EXTRA_RPC_2",
        "XRPL_EXTRA_RPC_3",
        "XRPL_EXTRA_RPC_4",
    ):
        value = os.environ.get(key)
        if value:
            return str(value)
    return ""


class _GlobalRateLimiter:
    def __init__(self, *, rps: float) -> None:
        if rps <= 0:
            raise RuntimeError("rps must be > 0")
        self._interval = 1.0 / rps
        self._next = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                delay = self._next - now
                self._next += self._interval
            else:
                delay = 0.0
                self._next = now + self._interval
        if delay > 0:
            time.sleep(delay)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch historical XRPL account_lines snapshots keyed by "
            "(boundary_ledger_index, owner_account, currency, issuer)."
        )
    )
    parser.add_argument("--input-csv", required=True, help="CSV with boundary_ledger_index, owner_account, currency, issuer")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--rpc", default=_default_rpc(), help="XRPL JSON-RPC endpoint")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rps", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--backoff-base", type=float, default=0.35)
    parser.add_argument("--backoff-max", type=float, default=6.0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--limit", type=int, default=400)
    return parser.parse_args()


def _load_targets(path: Path) -> list[tuple[int, str, str, str]]:
    rows: list[tuple[int, str, str, str]] = []
    seen: set[tuple[int, str, str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ledger_index_raw = row.get("boundary_ledger_index") or row.get("ledger_index")
            owner_account = str(row.get("owner_account") or row.get("account") or "").strip()
            currency = str(row.get("currency") or "").strip()
            issuer = str(row.get("issuer") or "").strip()
            if not ledger_index_raw or not owner_account or not currency or not issuer:
                continue
            key = (int(ledger_index_raw), owner_account, currency, issuer)
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
    if not rows:
        raise RuntimeError(f"no targets found in {path}")
    return rows


def _group_pending_targets(
    all_targets: list[tuple[int, str, str, str]],
    done: set[tuple[int, str, str, str]],
) -> list[tuple[int, str, str, tuple[str, ...]]]:
    grouped: dict[tuple[int, str, str], set[str]] = {}
    for ledger_index, owner_account, currency, issuer in all_targets:
        key = (ledger_index, owner_account, currency, issuer)
        if key in done:
            continue
        grouped.setdefault((ledger_index, owner_account, issuer), set()).add(currency)
    return [
        (ledger_index, owner_account, issuer, tuple(sorted(currencies)))
        for (ledger_index, owner_account, issuer), currencies in sorted(grouped.items())
    ]


def _load_done_keys(path: Path) -> set[tuple[int, str, str, str]]:
    done: set[tuple[int, str, str, str]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                row = json.loads(payload)
            except Exception:
                continue
            ledger_index = row.get("ledger_index")
            owner_account = str(row.get("owner_account") or "").strip()
            currency = str(row.get("currency") or "").strip()
            issuer = str(row.get("issuer") or "").strip()
            if ledger_index is None or not owner_account or not currency or not issuer:
                continue
            done.add((int(ledger_index), owner_account, currency, issuer))
    return done


def main() -> None:
    args = _parse_args()
    if not args.rpc:
        raise RuntimeError("missing --rpc and no XRPL RPC env var is set")

    input_csv = Path(args.input_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ok_file = outdir / "account_lines_snapshots.ndjson"
    fail_file = outdir / "account_lines_fail.ndjson"

    all_targets = _load_targets(input_csv)
    done = _load_done_keys(ok_file)
    pending = _group_pending_targets(all_targets, done)
    pending_key_count = sum(len(currencies) for *_prefix, currencies in pending)
    resume_skipped_keys = len([target for target in all_targets if target in done])
    print(
        f"input_total={len(all_targets)} resume_skipped_keys={resume_skipped_keys} "
        f"pending_keys={pending_key_count} pending_rpc_groups={len(pending)} "
        f"workers={args.workers} rps={args.rps}"
    )
    if not pending:
        print("nothing to do")
        return

    tls = threading.local()
    write_lock = threading.Lock()
    limiter = _GlobalRateLimiter(rps=args.rps)

    def _session() -> requests.Session:
        session = getattr(tls, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"Content-Type": "application/json"})
            tls.session = session
        return session

    def _append_ok(row: dict[str, Any]) -> None:
        with write_lock:
            with ok_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

    def _append_fail(row: dict[str, Any]) -> None:
        with write_lock:
            with fail_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

    def _fetch_one(
        ledger_index: int,
        owner_account: str,
        issuer: str,
        currencies: tuple[str, ...],
    ) -> tuple[int, int, str | None]:
        marker: Any = None
        lines: list[dict[str, Any]] = []
        attempts = 0
        while True:
            payload = {
                "method": "account_lines",
                "params": [
                    {
                        "account": owner_account,
                        "peer": issuer,
                        "ledger_index": ledger_index,
                        "limit": args.limit,
                    }
                ],
            }
            if marker is not None:
                payload["params"][0]["marker"] = marker

            for attempt in range(args.retries + 1):
                attempts += 1
                try:
                    limiter.acquire()
                    response = _session().post(args.rpc, json=payload, timeout=args.timeout)
                    if response.status_code >= 500:
                        raise RuntimeError(f"http_{response.status_code}")
                    body = response.json()
                    result = body.get("result") or {}
                    error = result.get("error")
                    if error:
                        raise RuntimeError(str(error))
                    batch = result.get("lines")
                    if not isinstance(batch, list):
                        raise RuntimeError("account_lines result missing lines")
                    lines.extend(batch)
                    marker = result.get("marker")
                    if marker is None:
                        by_currency: dict[str, list[dict[str, Any]]] = {}
                        for line in lines:
                            line_currency = str(line.get("currency") or "").strip()
                            if line_currency in currencies:
                                by_currency.setdefault(line_currency, []).append(line)
                        for currency in currencies:
                            matching = by_currency.get(currency, [])
                            if len(matching) > 1:
                                raise RuntimeError(
                                    f"multiple matching trust lines for account={owner_account} currency={currency} issuer={issuer} ledger={ledger_index}"
                                )
                            if matching:
                                balance = str(matching[0].get("balance"))
                                source = "account_lines"
                            else:
                                balance = "0"
                                source = "account_lines_missing_line_zero"
                            _append_ok(
                                {
                                    "ledger_index": ledger_index,
                                    "owner_account": owner_account,
                                    "currency": currency,
                                    "issuer": issuer,
                                    "balance": balance,
                                    "source": source,
                                    "validated": bool(result.get("validated")),
                                }
                            )
                        return len(currencies), 0, None
                    break
                except Exception as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                    if attempt >= args.retries:
                        for currency in currencies:
                            _append_fail(
                                {
                                    "ledger_index": ledger_index,
                                    "owner_account": owner_account,
                                    "currency": currency,
                                    "issuer": issuer,
                                    "status": "error",
                                    "error": last_err,
                                    "attempts": attempts,
                                }
                            )
                        return 0, len(currencies), last_err
                    backoff = min(args.backoff_base * (2**attempt), args.backoff_max)
                    time.sleep(backoff)
        return 0, len(currencies), "unreachable"

    completed = 0
    successes = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inflight = {
            executor.submit(_fetch_one, ledger_index, owner_account, issuer, currencies): (
                ledger_index,
                owner_account,
                issuer,
                currencies,
            )
            for ledger_index, owner_account, issuer, currencies in pending[: args.workers * 2]
        }
        cursor = len(inflight)
        while inflight:
            done_futures, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
            for future in done_futures:
                ledger_index, owner_account, issuer, currencies = inflight.pop(future)
                ok_count, fail_count, _err = future.result()
                completed += 1
                successes += ok_count
                failures += fail_count
                if completed % args.progress_every == 0 or completed == len(pending):
                    print(
                        f"completed_rpc_groups={completed}/{len(pending)} ok_keys={successes} fail_keys={failures} "
                        f"last=({ledger_index}, {owner_account}, {issuer}, currencies={len(currencies)})"
                    )
                if cursor < len(pending):
                    next_target = pending[cursor]
                    inflight[executor.submit(_fetch_one, *next_target)] = next_target
                    cursor += 1

    manifest = {
        "scope": "account_lines_snapshots",
        "input_csv": str(input_csv),
        "rpc": args.rpc,
        "ok_file": str(ok_file),
        "fail_file": str(fail_file),
        "workers": args.workers,
        "rps": args.rps,
        "successes": successes,
        "failures": failures,
        "input_keys": len(all_targets),
        "rpc_groups": len(pending),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
