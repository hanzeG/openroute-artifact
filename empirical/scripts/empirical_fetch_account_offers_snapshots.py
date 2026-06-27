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
            "Fetch XRPL account_offers snapshots keyed by (ledger_index, account) "
            "with append-only resume support."
        )
    )
    parser.add_argument("--input-csv", required=True, help="CSV with columns ledger_index,account")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--rpc", default=_default_rpc(), help="XRPL JSON-RPC endpoint")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--rps", type=float, default=12.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--backoff-base", type=float, default=0.35)
    parser.add_argument("--backoff-max", type=float, default=6.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=400)
    return parser.parse_args()


def _load_targets(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ledger_index = row.get("ledger_index")
            account = str(row.get("account") or "")
            if ledger_index is None or not account:
                continue
            rows.append((int(ledger_index), account))
    if not rows:
        raise RuntimeError(f"no targets found in {path}")
    return rows


def _load_done_keys(path: Path) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
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
            account = str(row.get("account") or "")
            if ledger_index is None or not account:
                continue
            done.add((int(ledger_index), account))
    return done


def main() -> None:
    args = _parse_args()
    if not args.rpc:
        raise RuntimeError("missing --rpc and no XRPL RPC endpoint found in environment")

    input_csv = Path(args.input_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ok_file = outdir / "account_offers_ok.ndjson"
    fail_file = outdir / "account_offers_fail.ndjson"

    all_targets = _load_targets(input_csv)
    done = _load_done_keys(ok_file)
    pending = [target for target in all_targets if target not in done]
    print(
        f"input_total={len(all_targets)} resume_skipped={len(all_targets) - len(pending)} "
        f"pending={len(pending)} workers={args.workers} rps={args.rps}"
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

    def _fetch_one(ledger_index: int, account: str) -> tuple[bool, str | None]:
        marker: Any = None
        offers: list[dict[str, Any]] = []
        attempts = 0
        while True:
            payload = {
                "method": "account_offers",
                "params": [
                    {
                        "account": account,
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
                    batch = result.get("offers")
                    if not isinstance(batch, list):
                        raise RuntimeError("account_offers result missing offers")
                    offers.extend(batch)
                    marker = result.get("marker")
                    if marker is None:
                        out = {
                            "ledger_index": ledger_index,
                            "account": account,
                            "offers": offers,
                            "validated": bool(result.get("validated")),
                        }
                        with write_lock:
                            with ok_file.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                                handle.flush()
                        return True, None
                    break
                except Exception as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                    if attempt >= args.retries:
                        out = {
                            "ledger_index": ledger_index,
                            "account": account,
                            "status": "error",
                            "error": last_err,
                            "attempts": attempts,
                        }
                        with write_lock:
                            with fail_file.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                                handle.flush()
                        return False, last_err
                    backoff = min(args.backoff_base * (2**attempt), args.backoff_max)
                    time.sleep(backoff)

        return False, "unreachable"

    completed = 0
    successes = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inflight = {
            executor.submit(_fetch_one, ledger_index, account): (ledger_index, account)
            for ledger_index, account in pending[: args.workers * 2]
        }
        cursor = len(inflight)
        while inflight:
            done_futures, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
            for future in done_futures:
                ledger_index, account = inflight.pop(future)
                ok, _err = future.result()
                completed += 1
                successes += int(ok)
                failures += int(not ok)
                if completed % args.progress_every == 0 or completed == len(pending):
                    print(
                        f"completed={completed}/{len(pending)} ok={successes} fail={failures} "
                        f"last=({ledger_index}, {account})"
                    )
                if cursor < len(pending):
                    next_ledger, next_account = pending[cursor]
                    inflight[executor.submit(_fetch_one, next_ledger, next_account)] = (
                        next_ledger,
                        next_account,
                    )
                    cursor += 1

    manifest = {
        "scope": "account_offers_snapshots",
        "input_csv": str(input_csv),
        "rpc": args.rpc,
        "ok_file": str(ok_file),
        "fail_file": str(fail_file),
        "workers": args.workers,
        "rps": args.rps,
        "successes": successes,
        "failures": failures,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
