#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download XRPL book_offers (issued asset/XRP both sides) for a given ledger list, "
            "with resume support and endpoint-switch friendly stop conditions."
        )
    )
    p.add_argument("--rpc", required=True, help="RPC endpoint URL")
    p.add_argument("--ledger-list", required=True, help="Text file: one ledger_index per line")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--issuer", required=True, help="Issued asset issuer")
    p.add_argument("--currency-hex", required=True, help="Issued asset currency")
    p.add_argument(
        "--output-prefix",
        default="book_rusd_xrp",
        help="Output file prefix; emits <prefix>_getsXRP.ndjson and <prefix>_getsTOKEN.ndjson",
    )
    p.add_argument("--limit", type=int, default=100, help="book_offers limit per side")
    p.add_argument("--workers", type=int, default=8, help="Concurrent ledgers")
    p.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds")
    p.add_argument("--retries", type=int, default=4, help="Retries per side request")
    p.add_argument("--backoff-base", type=float, default=0.35, help="Retry backoff base seconds")
    p.add_argument("--backoff-max", type=float, default=8.0, help="Retry backoff max seconds")
    p.add_argument("--progress-every", type=int, default=100, help="Single-line progress every N ledgers")
    p.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=100,
        help="Stop submitting new work after N consecutive failed ledgers (0 disables)",
    )
    p.add_argument(
        "--inflight-factor",
        type=int,
        default=2,
        help="Max in-flight futures = workers * inflight-factor",
    )
    return p.parse_args()


def load_ledgers(path: Path) -> list[int]:
    vals: list[int] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        vals.append(int(s))
    vals = sorted(set(vals))
    return vals


def load_done_ledgers(path: Path) -> set[int]:
    done: set[int] = set()
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
            li = obj.get("ledger_index")
            offers = obj.get("offers")
            if li is None or not isinstance(offers, list):
                continue
            done.add(int(li))
    return done


def is_quota_like(err_text: str, http_status: int | None) -> bool:
    if http_status == 429:
        return True
    t = (err_text or "").lower()
    keys = [
        "429",
        "rate limit",
        "too many",
        "quota",
        "credit",
        "credits",
        "exceeded",
        "limit reached",
    ]
    return any(k in t for k in keys)


def rpc_book_offers(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    *,
    timeout: float,
    retries: int,
    backoff_base: float,
    backoff_max: float,
) -> dict[str, Any]:
    payload = {"method": "book_offers", "params": [params]}
    last_err = "unknown"
    last_status: int | None = None

    for attempt in range(retries + 1):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            last_status = r.status_code
            text = r.text
            if r.status_code >= 500:
                raise RuntimeError(f"http_{r.status_code}")
            if r.status_code == 429:
                raise RuntimeError("http_429")
            body = json.loads(text)
            res = body.get("result", {})
            err = res.get("error") or body.get("error")
            if err:
                raise RuntimeError(str(err))
            if res.get("status") != "success":
                raise RuntimeError(str(res))
            if not isinstance(res.get("offers"), list):
                raise RuntimeError("missing_offers")
            return res
        except Exception as e:
            last_err = str(e)
            if attempt >= retries:
                break
            sleep_s = min(backoff_max, backoff_base * (2**attempt))
            sleep_s = sleep_s * (0.6 + random.random() * 0.8)
            time.sleep(sleep_s)

    raise RuntimeError(f"{last_err}|http={last_status}")


def main() -> None:
    args = parse_args()

    ledgers = load_ledgers(Path(args.ledger_list))
    if not ledgers:
        raise SystemExit("ledger list is empty")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_getsxrp = outdir / f"{args.output_prefix}_getsXRP.ndjson"
    out_getsrusd = outdir / f"{args.output_prefix}_getsTOKEN.ndjson"
    out_fail = outdir / f"{args.output_prefix}_fail.ndjson"

    done_x = load_done_ledgers(out_getsxrp)
    done_r = load_done_ledgers(out_getsrusd)
    done = done_x & done_r
    pending = [x for x in ledgers if x not in done]

    total = len(ledgers)
    skipped = len(done)
    todo = len(pending)
    print(
        f"input_total={total} resume_skipped={skipped} pending={todo} "
        f"workers={args.workers} limit={args.limit} rpc={args.rpc}"
    )
    if todo == 0:
        print("nothing to do")
        return

    tls = threading.local()
    write_lock = threading.Lock()
    started = time.time()

    RUSD = args.currency_hex
    ISS = args.issuer
    gets_xrp = {"currency": "XRP"}
    pays_rusd = {"currency": RUSD, "issuer": ISS}
    gets_rusd = {"currency": RUSD, "issuer": ISS}
    pays_xrp = {"currency": "XRP"}

    def get_session() -> requests.Session:
        s = getattr(tls, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"Content-Type": "application/json"})
            tls.session = s
        return s

    def fetch_ledger(ledger_index: int) -> tuple[bool, bool, str | None, int]:
        sess = get_session()
        attempts = 0
        try:
            pa = {
                "ledger_index": ledger_index,
                "taker_gets": gets_xrp,
                "taker_pays": pays_rusd,
                "limit": args.limit,
            }
            ra = rpc_book_offers(
                sess,
                args.rpc,
                pa,
                timeout=args.timeout,
                retries=args.retries,
                backoff_base=args.backoff_base,
                backoff_max=args.backoff_max,
            )
            attempts += 1

            pb = {
                "ledger_index": ledger_index,
                "taker_gets": gets_rusd,
                "taker_pays": pays_xrp,
                "limit": args.limit,
            }
            rb = rpc_book_offers(
                sess,
                args.rpc,
                pb,
                timeout=args.timeout,
                retries=args.retries,
                backoff_base=args.backoff_base,
                backoff_max=args.backoff_max,
            )
            attempts += 1

            out_a = {
                "ledger_index": ledger_index,
                "ledger_hash": ra.get("ledger_hash"),
                "taker_gets": gets_xrp,
                "taker_pays": pays_rusd,
                "offers_count": len(ra.get("offers", [])),
                "offers": ra.get("offers", []),
            }
            out_b = {
                "ledger_index": ledger_index,
                "ledger_hash": rb.get("ledger_hash"),
                "taker_gets": gets_rusd,
                "taker_pays": pays_xrp,
                "offers_count": len(rb.get("offers", [])),
                "offers": rb.get("offers", []),
            }
            with write_lock:
                with out_getsxrp.open("a", encoding="utf-8") as fx:
                    fx.write(json.dumps(out_a, ensure_ascii=False) + "\n")
                    fx.flush()
                with out_getsrusd.open("a", encoding="utf-8") as fr:
                    fr.write(json.dumps(out_b, ensure_ascii=False) + "\n")
                    fr.flush()
            return True, False, None, attempts

        except Exception as e:
            msg = str(e)
            quota_like = is_quota_like(msg, None)
            out = {
                "ledger_index": ledger_index,
                "status": "error",
                "error": msg,
                "quota_like": quota_like,
            }
            with write_lock:
                with out_fail.open("a", encoding="utf-8") as ff:
                    ff.write(json.dumps(out, ensure_ascii=False) + "\n")
                    ff.flush()
            return False, quota_like, msg, attempts

    ok = 0
    fail = 0
    done_n = 0
    attempts_sum = 0
    consecutive_failures = 0
    stop_reason: str | None = None
    saw_quota_like = False

    q: collections.deque[int] = collections.deque(pending)
    max_inflight = max(1, int(args.workers) * max(1, int(args.inflight_factor)))

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        inflight: set[Future[tuple[bool, bool, str | None, int]]] = set()

        def fill_inflight() -> None:
            while q and len(inflight) < max_inflight and stop_reason is None:
                li = q.popleft()
                inflight.add(ex.submit(fetch_ledger, li))

        fill_inflight()

        while inflight:
            finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in finished:
                inflight.remove(fut)
                success, quota_like, _, attempts = fut.result()
                done_n += 1
                attempts_sum += attempts
                if success:
                    ok += 1
                    consecutive_failures = 0
                else:
                    fail += 1
                    consecutive_failures += 1
                    if quota_like:
                        saw_quota_like = True
                        if stop_reason is None:
                            stop_reason = "quota_like_error_detected"

                if (
                    args.max_consecutive_failures > 0
                    and consecutive_failures >= args.max_consecutive_failures
                    and stop_reason is None
                ):
                    stop_reason = (
                        "reached --max-consecutive-failures="
                        f"{args.max_consecutive_failures}"
                    )

                if done_n % max(1, args.progress_every) == 0 or done_n == todo:
                    elapsed = time.time() - started
                    ledgers_ps = ok / elapsed if elapsed > 0 else 0.0
                    req_ps = (ok * 2) / elapsed if elapsed > 0 else 0.0
                    pct = (done_n / todo) * 100.0
                    tail = f" | stop={stop_reason}" if stop_reason else ""
                    print(
                        f"\r[{done_n}/{todo} {pct:6.2f}%] ok={ok} fail={fail} "
                        f"attempts/ledger={attempts_sum/max(1,done_n):4.2f} "
                        f"eff_reqps={req_ps:5.2f} eff_lgrps={ledgers_ps:5.2f}{tail}",
                        end="",
                        flush=True,
                    )

            if stop_reason is None:
                fill_inflight()

    elapsed = time.time() - started
    print()
    print(
        f"done pending={todo} ok={ok} fail={fail} elapsed_s={elapsed:.1f} "
        f"eff_reqps={(ok*2/elapsed) if elapsed>0 else 0.0:.2f}"
    )
    if stop_reason:
        print(f"stopped_early reason={stop_reason} remaining_unsubmitted={len(q)}")
    if saw_quota_like:
        print("hint=quota-like errors observed, switch to next endpoint and rerun same command")
    print(f"ok_file_getsXRP={out_getsxrp}")
    print(f"ok_file_getsrUSD={out_getsrusd}")
    print(f"fail_file={out_fail}")


if __name__ == "__main__":
    main()
