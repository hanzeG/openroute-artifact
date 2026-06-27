#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import threading
import time
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Tuple

import requests


def rpc(
    session: requests.Session,
    url: str,
    method: str,
    params: Dict[str, Any],
    *,
    retries: int = 8,
    backoff_base: float = 0.25,
    backoff_max: float = 10.0,
) -> Dict[str, Any]:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": [params]}

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = session.post(url, json=payload, timeout=60)

            # Hard rate limit: HTTP 429
            if r.status_code == 429:
                raise RuntimeError("HTTP 429")

            r.raise_for_status()
            out = r.json()

            # Some providers return JSON-RPC errors (sometimes with HTTP 200)
            if "error" in out and out["error"] is not None:
                msg = str(out["error"].get("message", ""))
                # Heuristic: treat rate-limit style errors as retryable
                if "rate" in msg.lower() or "limit" in msg.lower() or "too many" in msg.lower():
                    raise RuntimeError(f"JSON-RPC rate limit: {msg}")
                raise RuntimeError(json.dumps(out, indent=2))

            res = out.get("result", {})
            if res.get("status") != "success":
                # Also treat certain result-status errors as retryable
                msg = str(res.get("error_message") or res.get("error") or "")
                if "rate" in msg.lower() or "limit" in msg.lower() or "too many" in msg.lower():
                    raise RuntimeError(f"result rate limit: {msg}")
                raise RuntimeError(json.dumps(out, indent=2))
            return res

        except Exception as e:
            last_exc = e
            if attempt >= retries:
                break
            # Exponential backoff with jitter
            sleep_s = min(backoff_max, backoff_base * (2 ** attempt))
            sleep_s = sleep_s * (0.5 + random.random())
            time.sleep(sleep_s)

    raise last_exc if last_exc is not None else RuntimeError("rpc failed")


def parse_complete_ledgers(s: str) -> Tuple[int, int]:
    a, b = s.split("-")
    return int(a), int(b)


def book_offers_one_page(
    session: requests.Session,
    url: str,
    ledger_index: int,
    taker_gets: Dict[str, Any],
    taker_pays: Dict[str, Any],
    limit: int,
    retries: int,
    backoff_base: float,
    backoff_max: float,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "ledger_index": ledger_index,
        "taker_gets": taker_gets,
        "taker_pays": taker_pays,
        "limit": limit,
    }

    res = rpc(session, url, "book_offers", params, retries=retries, backoff_base=backoff_base, backoff_max=backoff_max)
    offers = res.get("offers", [])

    return {
        "ledger_index": ledger_index,
        "ledger_hash": res.get("ledger_hash"),
        "taker_gets": taker_gets,
        "taker_pays": taker_pays,
        "offers_count": len(offers),
        "offers": offers,
        # Note: marker intentionally ignored/omitted; this script stores only 1 page.
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc", default="https://xrplcluster.com/")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--issuer", required=True)
    ap.add_argument("--currency_hex", required=True)  # rUSD hex
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)  # 1 = every ledger
    ap.add_argument("--limit", type=int, default=200, help="Offers per side per ledger (one page only)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between ledgers")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite ndjson outputs if exist")
    ap.add_argument("--flush_every", type=int, default=20, help="Flush every N ledgers")
    ap.add_argument("--retries", type=int, default=8, help="Retries for 429 / rate-limit errors")
    ap.add_argument("--backoff_base", type=float, default=0.25, help="Base seconds for exponential backoff")
    ap.add_argument("--backoff_max", type=float, default=10.0, help="Max seconds for backoff between retries")
    ap.add_argument(
        "--max_inflight",
        type=int,
        default=None,
        help="Max queued ledgers when using --workers > 1 (default: 2*workers)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent ledgers to fetch (each ledger still fetches one page per side).",
    )
    ap.add_argument(
        "--parallel_sides",
        action="store_true",
        help="Fetch the two book sides concurrently per ledger (2 in-flight requests).",
    )
    args = ap.parse_args()

    # Thread-local sessions for safe connection reuse when using --workers > 1.
    _tls = threading.local()

    def get_sessions() -> Tuple[requests.Session, requests.Session]:
        sa = getattr(_tls, "sess_a", None)
        sb = getattr(_tls, "sess_b", None)
        if sa is None or sb is None:
            _tls.sess_a = requests.Session()
            _tls.sess_b = requests.Session()
        return _tls.sess_a, _tls.sess_b

    # Use separate sessions so we can safely run 2 concurrent requests (one per side).
    sess_a = requests.Session()
    sess_b = requests.Session()

    os.makedirs(args.outdir, exist_ok=True)

    info = rpc(sess_a, args.rpc, "server_info", {})
    complete = info["info"]["complete_ledgers"]
    lo, hi = parse_complete_ledgers(complete)

    start = args.start if args.start is not None else lo
    end = args.end if args.end is not None else hi
    if start < lo:
        start = lo
    if end > hi:
        end = hi

    print(f"complete_ledgers: {lo}-{hi}")
    print(
        f"downloading ledgers: {start}-{end} step={args.step} (one page only, limit={args.limit}) "
        f"workers={args.workers} parallel_sides={args.parallel_sides} max_inflight={args.max_inflight} "
        f"retries={args.retries} backoff_base={args.backoff_base} backoff_max={args.backoff_max}"
    )

    total = ((end - start) // args.step) + 1
    t0 = time.time()
    ok = 0
    err = 0
    last_line_len = 0

    def render_progress(current_ledger: int):
        nonlocal last_line_len
        done = ok + err
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        pct = (done / total * 100.0) if total > 0 else 0.0
        line = (
            f"[{done}/{total} {pct:6.2f}%] "
            f"lgr={current_ledger} ok={ok} err={err} "
            f"rate={rate:6.2f} lgr/s elapsed={elapsed:7.1f}s"
        )
        # Pad with spaces to fully overwrite the previous line
        pad = max(0, last_line_len - len(line))
        print("\r" + line + (" " * pad), end="", flush=True)
        last_line_len = len(line)

    RUSD = args.currency_hex
    ISS = args.issuer

    gets_xrp = {"currency": "XRP"}
    pays_rusd = {"currency": RUSD, "issuer": ISS}
    gets_rusd = {"currency": RUSD, "issuer": ISS}
    pays_xrp = {"currency": "XRP"}

    nd1 = os.path.join(args.outdir, "book_rusd_xrp_getsXRP.ndjson")
    nd2 = os.path.join(args.outdir, "book_rusd_xrp_getsrUSD.ndjson")

    if args.overwrite:
        if os.path.exists(nd1):
            os.remove(nd1)
        if os.path.exists(nd2):
            os.remove(nd2)

    def fetch_ledger(lgr: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sa, sb = get_sessions() if args.workers > 1 else (sess_a, sess_b)
        # If using multi-ledger concurrency, do NOT additionally parallelise sides;
        # keep per-ledger work simple to reduce thread overhead.
        if args.parallel_sides and args.workers == 1:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fa = ex.submit(
                    book_offers_one_page,
                    sa, args.rpc, lgr, gets_xrp, pays_rusd, args.limit,
                    args.retries, args.backoff_base, args.backoff_max,
                )
                fb = ex.submit(
                    book_offers_one_page,
                    sb, args.rpc, lgr, gets_rusd, pays_xrp, args.limit,
                    args.retries, args.backoff_base, args.backoff_max,
                )
                return fa.result(), fb.result()
        a = book_offers_one_page(
            sa, args.rpc, lgr, gets_xrp, pays_rusd, args.limit,
            args.retries, args.backoff_base, args.backoff_max,
        )
        b = book_offers_one_page(
            sa, args.rpc, lgr, gets_rusd, pays_xrp, args.limit,
            args.retries, args.backoff_base, args.backoff_max,
        )
        return a, b

    with open(nd1, "a", encoding="utf-8") as fp1, open(nd2, "a", encoding="utf-8") as fp2:
        i = 0

        if args.workers <= 1:
            # Single-thread (optionally parallelise sides)
            for lgr in range(start, end + 1, args.step):
                i += 1
                try:
                    a, b = fetch_ledger(lgr)

                    fp1.write(json.dumps(a, ensure_ascii=False) + "\n")
                    fp2.write(json.dumps(b, ensure_ascii=False) + "\n")

                    if args.flush_every > 0 and (i % args.flush_every == 0):
                        fp1.flush()
                        fp2.flush()

                    ok += 1
                    render_progress(lgr)

                    if args.sleep > 0:
                        time.sleep(args.sleep)

                except Exception as e:
                    err += 1
                    render_progress(lgr)
                    print(f"\nledger {lgr}: ERROR {e}")
                    continue

        else:
            # Multi-ledger concurrency; writes remain single-threaded.
            # Note: --parallel_sides is ignored here to avoid nested thread pools.
            inflight = deque()
            max_inflight = max(1, args.max_inflight if args.max_inflight is not None else (args.workers * 2))

            def submit(ex: ThreadPoolExecutor, lgr: int):
                inflight.append((lgr, ex.submit(fetch_ledger, lgr)))

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                lgr_iter = iter(range(start, end + 1, args.step))

                # Prime the queue
                for _ in range(min(max_inflight, total)):
                    try:
                        lgr = next(lgr_iter)
                    except StopIteration:
                        break
                    submit(ex, lgr)

                while inflight:
                    lgr, fut = inflight.popleft()
                    i += 1
                    try:
                        a, b = fut.result()
                        fp1.write(json.dumps(a, ensure_ascii=False) + "\n")
                        fp2.write(json.dumps(b, ensure_ascii=False) + "\n")

                        if args.flush_every > 0 and (i % args.flush_every == 0):
                            fp1.flush()
                            fp2.flush()

                        ok += 1
                        render_progress(lgr)

                    except Exception as e:
                        err += 1
                        render_progress(lgr)
                        print(f"\nledger {lgr}: ERROR {e}")

                    # Top up the queue
                    try:
                        nxt = next(lgr_iter)
                        submit(ex, nxt)
                    except StopIteration:
                        pass

                    if args.sleep > 0:
                        time.sleep(args.sleep)

    print("\n\nWrote NDJSON:")
    print(f"  {nd1}")
    print(f"  {nd2}")


if __name__ == "__main__":
    main()