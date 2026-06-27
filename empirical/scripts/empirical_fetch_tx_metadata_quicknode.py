#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import requests


def default_rpc() -> str:
    for key in ("XRPL_QN_RPC", "XRPL_QN_RPC_1", "XRPL_QN_RPC_2", "XRPL_QN_RPC_3", "XRPL_QN_RPC_4"):
        value = os.environ.get(key)
        if value:
            return value
    return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fetch XRPL tx metadata from QuickNode with stable RPS limiting, "
            "immediate append writes, and resume support."
        )
    )
    p.add_argument(
        "--input",
        required=True,
        help=(
            "Input tx hash source. Can be: "
            "1) one txt file (one hash per line), or "
            "2) a directory containing tx_hashes_part_*_of_*.txt"
        ),
    )
    p.add_argument("--rpc", default=default_rpc(), help="QuickNode RPC URL")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--workers", type=int, default=12, help="Thread workers (recommend 12-15)")
    p.add_argument("--rps", type=float, default=12.0, help="Global request rate limit")
    p.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds")
    p.add_argument("--retries", type=int, default=4, help="Retries per tx")
    p.add_argument("--backoff-base", type=float, default=0.35, help="Base backoff seconds")
    p.add_argument("--backoff-max", type=float, default=6.0, help="Max backoff seconds")
    p.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Update single-line progress every N completed tx",
    )
    p.add_argument(
        "--max-ok",
        type=int,
        default=0,
        help="Stop after this many successful rows in current run (0 = no limit).",
    )
    p.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=0,
        help="Stop when consecutive failures reach this threshold (0 = disable).",
    )
    p.add_argument(
        "--inflight-factor",
        type=int,
        default=2,
        help="Max in-flight futures as workers * inflight-factor (default: 2).",
    )
    return p.parse_args()


def load_hashes(inp: Path) -> list[str]:
    out: list[str] = []
    if inp.is_dir():
        files = sorted(inp.glob("tx_hashes_part_*_of_*.txt"))
        if not files:
            raise SystemExit(f"no shard files matched in directory: {inp}")
        for fp in files:
            for ln in fp.read_text(encoding="utf-8").splitlines():
                h = ln.strip()
                if h:
                    out.append(h.upper())
    else:
        for ln in inp.read_text(encoding="utf-8").splitlines():
            h = ln.strip()
            if h:
                out.append(h.upper())

    seen: set[str] = set()
    uniq: list[str] = []
    for h in out:
        if h in seen:
            continue
        seen.add(h)
        uniq.append(h)
    return uniq


def load_done_hashes(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            h = obj.get("tx_hash")
            if not h:
                continue
            # Resume should skip only fully usable rows.
            result = obj.get("result") or {}
            li = result.get("ledger_index")
            meta = result.get("meta") or {}
            ti = meta.get("TransactionIndex") if isinstance(meta, dict) else None
            if li is not None and ti is not None:
                done.add(str(h).upper())
    return done


class GlobalRateLimiter:
    def __init__(self, rps: float):
        if rps <= 0:
            raise ValueError("rps must be > 0")
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


def main() -> None:
    args = parse_args()
    if not args.rpc:
        raise SystemExit("missing --rpc and no XRPL_QN_RPC found in environment")
    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"input not found: {inp}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ok_file = outdir / "tx_metadata_ok.ndjson"
    fail_file = outdir / "tx_metadata_fail.ndjson"

    hashes = load_hashes(inp)
    done_hashes = load_done_hashes(ok_file)
    pending = [h for h in hashes if h not in done_hashes]

    total = len(hashes)
    skipped = len(done_hashes & set(hashes))
    todo = len(pending)

    print(
        f"input_total={total} resume_skipped={skipped} pending={todo} "
        f"workers={args.workers} rps={args.rps} max_ok={args.max_ok} "
        f"max_consecutive_failures={args.max_consecutive_failures}"
    )
    if todo == 0:
        print("nothing to do")
        return

    limiter = GlobalRateLimiter(args.rps)
    tls = threading.local()
    write_lock = threading.Lock()

    def get_session() -> requests.Session:
        s = getattr(tls, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"Content-Type": "application/json"})
            tls.session = s
        return s

    def fetch_one(tx_hash: str) -> tuple[bool, int, str | None]:
        payload = {"method": "tx", "params": [{"transaction": tx_hash, "binary": False}]}
        last_err: str | None = None
        attempts = 0

        for attempt in range(args.retries + 1):
            attempts += 1
            try:
                limiter.acquire()
                r = get_session().post(args.rpc, json=payload, timeout=args.timeout)
                if r.status_code >= 500:
                    raise RuntimeError(f"http_{r.status_code}")
                body = r.json()

                # tx method returns errors inside result envelope.
                result = body.get("result", {})
                err = result.get("error")
                if err:
                    msg = str(err)
                    if msg == "txnNotFound":
                        out = {"tx_hash": tx_hash, "status": "not_found", "result": result}
                        with write_lock:
                            with fail_file.open("a", encoding="utf-8") as fp:
                                fp.write(json.dumps(out, ensure_ascii=False) + "\n")
                                fp.flush()
                        return False, attempts, "txnNotFound"
                    raise RuntimeError(msg)

                li = result.get("ledger_index")
                meta = result.get("meta") or {}
                ti = meta.get("TransactionIndex") if isinstance(meta, dict) else None
                if li is not None and ti is not None:
                    out = {"tx_hash": tx_hash, "status": "ok", "result": result}
                    with write_lock:
                        with ok_file.open("a", encoding="utf-8") as fp:
                            fp.write(json.dumps(out, ensure_ascii=False) + "\n")
                            fp.flush()
                    return True, attempts, None

                # Non-empty but incomplete result: retry, then record as failure.
                raise RuntimeError("missing_required_fields")

            except Exception as e:
                last_err = type(e).__name__
                if attempt >= args.retries:
                    out = {"tx_hash": tx_hash, "status": "error", "error": str(e)}
                    with write_lock:
                        with fail_file.open("a", encoding="utf-8") as fp:
                            fp.write(json.dumps(out, ensure_ascii=False) + "\n")
                            fp.flush()
                    return False, attempts, last_err
                sleep_s = min(args.backoff_max, args.backoff_base * (2**attempt))
                sleep_s = sleep_s * (0.6 + random.random() * 0.8)
                time.sleep(sleep_s)

        return False, attempts, last_err

    started = time.time()
    ok = 0
    fail = 0
    attempts_sum = 0

    consecutive_failures = 0
    stop_reason: str | None = None
    done = 0
    pending_q: collections.deque[str] = collections.deque(pending)
    max_inflight = max(1, int(args.workers) * max(1, int(args.inflight_factor)))

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        inflight: set[Future[tuple[bool, int, str | None]]] = set()

        def fill_inflight() -> None:
            while pending_q and len(inflight) < max_inflight and stop_reason is None:
                h = pending_q.popleft()
                inflight.add(ex.submit(fetch_one, h))

        fill_inflight()

        while inflight:
            done_set, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done_set:
                inflight.remove(fut)
                success, attempts, _ = fut.result()
                done += 1
                attempts_sum += attempts
                if success:
                    ok += 1
                    consecutive_failures = 0
                else:
                    fail += 1
                    consecutive_failures += 1

                if args.max_ok > 0 and ok >= args.max_ok and stop_reason is None:
                    stop_reason = f"reached --max-ok={args.max_ok}"

                if (
                    args.max_consecutive_failures > 0
                    and consecutive_failures >= args.max_consecutive_failures
                    and stop_reason is None
                ):
                    stop_reason = (
                        "reached --max-consecutive-failures="
                        f"{args.max_consecutive_failures}"
                    )

                if done % max(1, args.progress_every) == 0 or done == todo:
                    elapsed = time.time() - started
                    txps = ok / elapsed if elapsed > 0 else 0.0
                    pct = (done / todo) * 100.0
                    tail = f" | stop={stop_reason}" if stop_reason else ""
                    print(
                        f"\r[{done}/{todo} {pct:6.2f}%] ok={ok} fail={fail} "
                        f"attempts/tx={attempts_sum/done:4.2f} effective_txps={txps:5.2f}{tail}",
                        end="",
                        flush=True,
                    )

            if stop_reason is None:
                fill_inflight()
            else:
                # Graceful stop: do not submit new work; finish current in-flight only.
                pass

    elapsed = time.time() - started
    print()
    print(
        f"done pending={todo} ok={ok} fail={fail} elapsed_s={elapsed:.1f} "
        f"effective_txps={ok/elapsed if elapsed>0 else 0.0:.2f}"
    )
    if stop_reason:
        print(f"stopped_early reason={stop_reason} remaining_unsubmitted={len(pending_q)}")
    print(f"ok_file={ok_file}")
    print(f"fail_file={fail_file}")


if __name__ == "__main__":
    main()
