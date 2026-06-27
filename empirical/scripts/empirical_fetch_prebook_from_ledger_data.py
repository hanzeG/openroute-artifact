#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Callable

import requests

getcontext().prec = 50

XRP = "XRP"
XRP_QUANTUM = Decimal("0.000001")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build historical prebook snapshots via ledger_data pagination, then filter "
            "Offer entries for an issued-asset/XRP pair in both directions."
        )
    )
    p.add_argument("--rpc", required=True, help="XRPL RPC endpoint")
    p.add_argument("--ledger-list", required=True, help="Text file: one ledger index per line")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--issuer", required=True, help="Issued asset issuer")
    p.add_argument("--currency-hex", required=True, help="Issued asset currency hex")
    p.add_argument(
        "--output-prefix",
        default="book_rusd_xrp",
        help=(
            "Output file prefix; emits <prefix>_getsXRP.ndjson and "
            "<prefix>_getsrUSD.ndjson. Default preserves legacy RLUSD/XRP filenames."
        ),
    )
    p.add_argument("--limit", type=int, default=400, help="Max offers to keep per side per ledger")
    p.add_argument("--page-limit", type=int, default=2000, help="ledger_data page size")
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    p.add_argument("--retries", type=int, default=4, help="Retries per rpc call")
    p.add_argument("--backoff-base", type=float, default=0.4, help="Backoff base seconds")
    p.add_argument("--backoff-max", type=float, default=8.0, help="Backoff max seconds")
    p.add_argument("--workers", type=int, default=1, help="Ledger workers (single-ledger scan is still marker-serial)")
    p.add_argument(
        "--stop-getsxrp-at",
        type=int,
        default=0,
        help="Stop scanning a ledger once candidate getsXRP offers reach this count (0 disables)",
    )
    p.add_argument(
        "--checkpoint-pages",
        type=int,
        default=0,
        help="If >0, write checkpoint snapshots every N pages while scanning each ledger",
    )
    p.add_argument(
        "--progress-pages",
        type=int,
        default=20,
        help="Print progress every N ledger_data pages (0 disables)",
    )
    return p.parse_args()


def load_ledgers(path: Path) -> list[int]:
    out: list[int] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        out.append(int(s))
    return sorted(set(out))


def rpc_call(
    session: requests.Session,
    url: str,
    method: str,
    params: dict[str, Any],
    *,
    timeout: float,
    retries: int,
    backoff_base: float,
    backoff_max: float,
) -> dict[str, Any]:
    payload = {"method": method, "params": [params]}
    last_err = "unknown"
    last_status: int | None = None
    for i in range(retries + 1):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            last_status = r.status_code
            if r.status_code >= 500:
                raise RuntimeError(f"http_{r.status_code}")
            body = r.json()
            res = body.get("result", {})
            err = res.get("error") or body.get("error")
            if err:
                raise RuntimeError(str(err))
            if res.get("status") != "success":
                raise RuntimeError(str(res))
            return res
        except Exception as e:
            last_err = str(e)
            if i >= retries:
                break
            sleep_s = min(backoff_max, backoff_base * (2**i))
            sleep_s *= 0.6 + random.random() * 0.8
            time.sleep(sleep_s)
    raise RuntimeError(f"{last_err}|http={last_status}")


def amt_currency(a: Any) -> str | None:
    if isinstance(a, str):
        return XRP
    if isinstance(a, dict):
        return str(a.get("currency")) if a.get("currency") is not None else None
    return None


def amt_value(a: Any) -> Decimal | None:
    if isinstance(a, str):
        return Decimal(a) * XRP_QUANTUM
    if isinstance(a, dict) and a.get("value") is not None:
        return Decimal(str(a.get("value")))
    return None


def is_rusd_amt(a: Any, *, currency_hex: str, issuer: str) -> bool:
    return (
        isinstance(a, dict)
        and str(a.get("currency")) == currency_hex
        and str(a.get("issuer")) == issuer
        and a.get("value") is not None
    )


def quality_out_over_in(offer: dict[str, Any], *, out_cur: str, in_cur: str) -> Decimal:
    gets = offer.get("TakerGets")
    pays = offer.get("TakerPays")
    gets_cur = amt_currency(gets)
    pays_cur = amt_currency(pays)
    gets_v = amt_value(gets)
    pays_v = amt_value(pays)
    if gets_v is None or pays_v is None or gets_v <= 0 or pays_v <= 0:
        return Decimal("-1")
    if gets_cur == out_cur and pays_cur == in_cur:
        return gets_v / pays_v
    if pays_cur == out_cur and gets_cur == in_cur:
        return pays_v / gets_v
    return Decimal("-1")


def gather_offers_for_ledger(
    session: requests.Session,
    rpc: str,
    ledger_index: int,
    *,
    timeout: float,
    retries: int,
    backoff_base: float,
    backoff_max: float,
    page_limit: int,
    currency_hex: str,
    issuer: str,
    keep_limit: int,
    progress_pages: int,
    stop_getsxrp_at: int,
    progress_cb: Callable[[int, str | None, list[dict[str, Any]], list[dict[str, Any]], bool], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker: Any = None
    offers_getsxrp: list[dict[str, Any]] = []
    offers_getsrusd: list[dict[str, Any]] = []
    ledger_hash: str | None = None
    pages = 0
    started = time.time()
    stop_requested = False

    while True:
        params: dict[str, Any] = {
            "ledger_index": ledger_index,
            "limit": page_limit,
            "type": "offer",
        }
        if marker is not None:
            params["marker"] = marker
        res = rpc_call(
            session,
            rpc,
            "ledger_data",
            params,
            timeout=timeout,
            retries=retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
        )
        if ledger_hash is None:
            ledger_hash = str(res.get("ledger_hash") or "")
        pages += 1

        for e in res.get("state", []) or []:
            if not isinstance(e, dict):
                continue
            if str(e.get("LedgerEntryType")) != "Offer":
                continue
            gets = e.get("TakerGets")
            pays = e.get("TakerPays")
            if gets is None or pays is None:
                continue

            is_getsxrp = (amt_currency(gets) == XRP and is_rusd_amt(pays, currency_hex=currency_hex, issuer=issuer))
            is_getsrusd = (is_rusd_amt(gets, currency_hex=currency_hex, issuer=issuer) and amt_currency(pays) == XRP)
            if not is_getsxrp and not is_getsrusd:
                continue

            row = dict(e)
            row.setdefault("index", row.get("index"))
            row.setdefault("owner_funds", None)
            row.setdefault("taker_gets_funded", None)
            row.setdefault("taker_pays_funded", None)
            if is_getsxrp:
                offers_getsxrp.append(row)
                if stop_getsxrp_at > 0 and len(offers_getsxrp) >= stop_getsxrp_at:
                    stop_requested = True
            if is_getsrusd:
                offers_getsrusd.append(row)
            if stop_requested:
                break

        marker = res.get("marker")
        if progress_pages > 0 and (pages % progress_pages) == 0:
            elapsed = time.time() - started
            reqps = (pages / elapsed) if elapsed > 0 else 0.0
            print(
                f"  ledger={ledger_index} pages={pages} "
                f"cand_getsXRP={len(offers_getsxrp)} cand_getsrUSD={len(offers_getsrusd)} "
                f"reqps={reqps:.2f} elapsed_s={elapsed:.1f}",
                flush=True,
            )
        if progress_cb is not None and progress_pages > 0 and (pages % progress_pages) == 0:
            progress_cb(pages, ledger_hash, offers_getsxrp, offers_getsrusd, stop_requested)
        if stop_requested:
            print(
                f"  ledger={ledger_index} early-stop: cand_getsXRP={len(offers_getsxrp)} >= stop_getsxrp_at={stop_getsxrp_at}",
                flush=True,
            )
            break
        if marker is None:
            break

    offers_getsxrp.sort(
        key=lambda o: quality_out_over_in(o, out_cur=XRP, in_cur=currency_hex),
        reverse=True,
    )
    offers_getsrusd.sort(
        key=lambda o: quality_out_over_in(o, out_cur=currency_hex, in_cur=XRP),
        reverse=True,
    )
    if keep_limit > 0:
        offers_getsxrp = offers_getsxrp[:keep_limit]
        offers_getsrusd = offers_getsrusd[:keep_limit]

    out_a = {
        "ledger_index": int(ledger_index),
        "ledger_hash": ledger_hash,
        "taker_gets": {"currency": XRP},
        "taker_pays": {"currency": currency_hex, "issuer": issuer},
        "offers_count": len(offers_getsxrp),
        "offers": offers_getsxrp,
    }
    out_b = {
        "ledger_index": int(ledger_index),
        "ledger_hash": ledger_hash,
        "taker_gets": {"currency": currency_hex, "issuer": issuer},
        "taker_pays": {"currency": XRP},
        "offers_count": len(offers_getsrusd),
        "offers": offers_getsrusd,
    }
    if progress_cb is not None:
        progress_cb(pages, ledger_hash, offers_getsxrp, offers_getsrusd, stop_requested)
    return out_a, out_b


def main() -> None:
    args = parse_args()
    ledgers = load_ledgers(Path(args.ledger_list))
    if not ledgers:
        raise SystemExit("ledger list is empty")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_getsxrp = outdir / f"{args.output_prefix}_getsXRP.ndjson"
    out_getsrusd = outdir / f"{args.output_prefix}_getsrUSD.ndjson"
    out_fail = outdir / f"{args.output_prefix}_fail.ndjson"
    ckpt_getsxrp = outdir / f"{args.output_prefix}_getsXRP.checkpoint.ndjson"
    ckpt_getsrusd = outdir / f"{args.output_prefix}_getsrUSD.checkpoint.ndjson"

    # Overwrite for deterministic rebuild.
    for p in [out_getsxrp, out_getsrusd, out_fail, ckpt_getsxrp, ckpt_getsrusd]:
        if p.exists():
            p.unlink()

    print(
        f"input_total={len(ledgers)} rpc={args.rpc} limit={args.limit} page_limit={args.page_limit} "
        f"workers={args.workers} stop_getsxrp_at={args.stop_getsxrp_at} checkpoint_pages={args.checkpoint_pages} ledger_data_mode=1"
    )
    if len(ledgers) == 1 and args.workers > 1:
        print("note: single-ledger ledger_data scan is marker-serial; workers has no practical speedup")
    started = time.time()
    ok = 0
    fail = 0

    sess = requests.Session()
    sess.headers.update({"Content-Type": "application/json"})

    def _quality_sort_xrp(o: dict[str, Any]) -> Decimal:
        return quality_out_over_in(o, out_cur=XRP, in_cur=args.currency_hex)

    def _quality_sort_rusd(o: dict[str, Any]) -> Decimal:
        return quality_out_over_in(o, out_cur=args.currency_hex, in_cur=XRP)

    def _write_checkpoint(
        ledger_index: int,
        pages: int,
        ledger_hash: str | None,
        x_rows: list[dict[str, Any]],
        r_rows: list[dict[str, Any]],
    ) -> None:
        x = sorted(x_rows, key=_quality_sort_xrp, reverse=True)
        r = sorted(r_rows, key=_quality_sort_rusd, reverse=True)
        if args.limit > 0:
            x = x[: args.limit]
            r = r[: args.limit]
        rec_x = {
            "scope": "checkpoint",
            "ledger_index": int(ledger_index),
            "pages_scanned": int(pages),
            "ledger_hash": ledger_hash,
            "offers_count": len(x),
            "offers": x,
        }
        rec_r = {
            "scope": "checkpoint",
            "ledger_index": int(ledger_index),
            "pages_scanned": int(pages),
            "ledger_hash": ledger_hash,
            "offers_count": len(r),
            "offers": r,
        }
        ckpt_getsxrp.write_text(json.dumps(rec_x, ensure_ascii=False) + "\n", encoding="utf-8")
        ckpt_getsrusd.write_text(json.dumps(rec_r, ensure_ascii=False) + "\n", encoding="utf-8")

    for i, li in enumerate(ledgers, start=1):
        try:
            progress_cb = None
            if args.checkpoint_pages > 0:
                def progress_cb(
                    pages: int,
                    ledger_hash: str | None,
                    x_rows: list[dict[str, Any]],
                    r_rows: list[dict[str, Any]],
                    _stopped: bool,
                ) -> None:
                    if pages % args.checkpoint_pages == 0:
                        _write_checkpoint(li, pages, ledger_hash, x_rows, r_rows)

            out_a, out_b = gather_offers_for_ledger(
                sess,
                args.rpc,
                li,
                timeout=args.timeout,
                retries=args.retries,
                backoff_base=args.backoff_base,
                backoff_max=args.backoff_max,
                page_limit=args.page_limit,
                currency_hex=args.currency_hex,
                issuer=args.issuer,
                keep_limit=args.limit,
                progress_pages=args.progress_pages,
                stop_getsxrp_at=args.stop_getsxrp_at,
                progress_cb=progress_cb,
            )
            with out_getsxrp.open("a", encoding="utf-8") as fa:
                fa.write(json.dumps(out_a, ensure_ascii=False) + "\n")
            with out_getsrusd.open("a", encoding="utf-8") as fb:
                fb.write(json.dumps(out_b, ensure_ascii=False) + "\n")
            if args.checkpoint_pages > 0:
                _write_checkpoint(
                    li,
                    pages=-1,
                    ledger_hash=str(out_a.get("ledger_hash") or ""),
                    x_rows=out_a.get("offers", []) or [],
                    r_rows=out_b.get("offers", []) or [],
                )
            ok += 1
        except Exception as e:
            fail += 1
            with out_fail.open("a", encoding="utf-8") as ff:
                ff.write(
                    json.dumps(
                        {
                            "ledger_index": int(li),
                            "error": str(e),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        elapsed = time.time() - started
        eff = (ok / elapsed) if elapsed > 0 else 0.0
        print(
            f"[{i}/{len(ledgers)} {100.0*i/len(ledgers):6.2f}%] ok={ok} fail={fail} "
            f"ledger={li} eff_lgrps={eff:5.2f}"
        )

    elapsed = time.time() - started
    print(f"done ok={ok} fail={fail} elapsed_s={elapsed:.1f} eff_lgrps={(ok/elapsed if elapsed>0 else 0.0):.2f}")
    print(f"ok_file_getsXRP={out_getsxrp}")
    print(f"ok_file_getsrUSD={out_getsrusd}")
    print(f"fail_file={out_fail}")


if __name__ == "__main__":
    main()
