#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    keys = [
        "XRPL_RIPPLE_S2_RPC",
        "XRPL_RIPPLE_S1_RPC",
        "XRPL_CLUSTER_RPC",
        "XRPL_EXTRA_RPC_1",
        "XRPL_EXTRA_RPC_2",
        "XRPL_EXTRA_RPC_3",
        "XRPL_EXTRA_RPC_4",
    ]
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fetch full expanded XRPL ledger transactions by ledger index and emit "
            "rows compatible with tx_metadata_full_merged.ndjson."
        )
    )
    p.add_argument("--ledger-list", required=True, help="Text file: one ledger_index per line")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--rpc", default=default_rpc(), help="XRPL RPC endpoint")
    p.add_argument("--workers", type=int, default=6, help="Concurrent ledgers")
    p.add_argument(
        "--required-tx-csv",
        default=None,
        help=(
            "Optional CSV/TSV with at least ledger_index and transaction_hash columns. "
            "If provided, each fetched ledger must contain all expected tx hashes."
        ),
    )
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    p.add_argument("--retries", type=int, default=4, help="Retries per ledger fetch")
    p.add_argument("--backoff-base", type=float, default=0.4, help="Retry backoff base seconds")
    p.add_argument("--backoff-max", type=float, default=8.0, help="Retry backoff max seconds")
    p.add_argument("--progress-every", type=int, default=20, help="Single-line progress every N ledgers")
    p.add_argument(
        "--merged-name",
        default="tx_metadata_full_merged.ndjson",
        help="Merged NDJSON output filename",
    )
    return p.parse_args()


def load_ledgers(path: Path) -> list[int]:
    vals: list[int] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        vals.append(int(s))
    return sorted(set(vals))


def load_required_hashes_by_ledger(path: Path | None) -> dict[int, set[str]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"required tx csv not found: {path}")

    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        fieldnames = {str(x) for x in (reader.fieldnames or [])}
        if "ledger_index" not in fieldnames or "transaction_hash" not in fieldnames:
            raise RuntimeError(
                f"required tx csv must contain ledger_index and transaction_hash columns: {path}"
            )
        out: dict[int, set[str]] = {}
        for row in reader:
            try:
                li = int(str(row.get("ledger_index") or "").strip())
            except Exception:
                continue
            txh = str(row.get("transaction_hash") or "").strip().upper()
            if not txh:
                continue
            out.setdefault(li, set()).add(txh)
        return out


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
            result = body.get("result", {})
            err = result.get("error") or body.get("error")
            if err:
                raise RuntimeError(str(err))
            status = result.get("status")
            if status not in (None, "success"):
                raise RuntimeError(str(result))
            return result
        except Exception as e:
            last_err = str(e)
            if i >= retries:
                break
            sleep_s = min(backoff_max, backoff_base * (2**i))
            sleep_s *= 0.6 + random.random() * 0.8
            time.sleep(sleep_s)
    raise RuntimeError(f"{last_err}|http={last_status}")


def _normalize_tx_row(
    tx_obj: dict[str, Any],
    *,
    ledger_index: int | None,
    validated: Any,
) -> dict[str, Any] | None:
    if not isinstance(tx_obj, dict):
        return None
    tx_hash = tx_obj.get("hash")
    if tx_hash is None:
        return None
    out = dict(tx_obj)
    meta = out.get("meta")
    if meta is None and out.get("metaData") is not None:
        meta = out.pop("metaData")
        out["meta"] = meta
    if ledger_index is not None:
        out.setdefault("ledger_index", int(ledger_index))
        out.setdefault("inLedger", int(ledger_index))
    if validated is not None:
        out.setdefault("validated", bool(validated))
    return {
        "tx_hash": str(tx_hash).upper(),
        "status": "ok",
        "result": out,
    }


def normalize_ledger_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    ledger_obj = result.get("ledger")
    if isinstance(ledger_obj, dict):
        txs = ledger_obj.get("transactions") or []
        ledger_index = ledger_obj.get("ledger_index") or result.get("ledger_index")
        validated = result.get("validated")
    else:
        txs = result.get("transactions") or []
        ledger_index = result.get("ledger_index")
        validated = result.get("validated")

    out: list[dict[str, Any]] = []
    for tx_obj in txs:
        row = _normalize_tx_row(tx_obj, ledger_index=ledger_index, validated=validated)
        if row is not None:
            out.append(row)

    def _sort_key(row: dict[str, Any]) -> tuple[int, str]:
        result_obj = row.get("result") or {}
        meta = result_obj.get("meta") or {}
        txi = meta.get("TransactionIndex")
        try:
            txi_i = int(txi)
        except Exception:
            txi_i = 10**9
        return txi_i, str(row.get("tx_hash") or "")

    out.sort(key=_sort_key)
    return out


def _ledger_shard_path(raw_dir: Path, ledger_index: int) -> Path:
    return raw_dir / f"ledger_{int(ledger_index)}.ndjson"


def _load_done_ledgers(raw_dir: Path) -> set[int]:
    done: set[int] = set()
    for fp in raw_dir.glob("ledger_*.ndjson"):
        stem = fp.stem
        try:
            li = int(stem.split("_", 1)[1])
        except Exception:
            continue
        try:
            first = fp.read_text(encoding="utf-8").strip().splitlines()
        except Exception:
            continue
        if first:
            done.add(li)
    return done


def main() -> None:
    args = parse_args()
    if not args.rpc:
        raise SystemExit("missing --rpc and no XRPL RPC endpoint found in environment")
    ledgers = load_ledgers(Path(args.ledger_list))
    if not ledgers:
        raise SystemExit("ledger list is empty")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = outdir / "raw_ledgers"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fail_file = outdir / "ledger_fetch_fail.ndjson"
    merged_path = outdir / args.merged_name
    required_hashes_by_ledger = load_required_hashes_by_ledger(
        Path(args.required_tx_csv) if args.required_tx_csv else None
    )

    done_ledgers = _load_done_ledgers(raw_dir)
    pending = [li for li in ledgers if li not in done_ledgers]
    print(
        f"input_total={len(ledgers)} resume_skipped={len(done_ledgers)} pending={len(pending)} "
        f"workers={args.workers}"
    )

    tls = threading.local()
    write_lock = threading.Lock()
    started = time.time()

    def get_session() -> requests.Session:
        s = getattr(tls, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"Content-Type": "application/json"})
            tls.session = s
        return s

    def fetch_one(ledger_index: int) -> tuple[bool, int, str | None]:
        try:
            result = rpc_call(
                get_session(),
                args.rpc,
                "ledger",
                {
                    "ledger_index": int(ledger_index),
                    "transactions": True,
                    "expand": True,
                    "owner_funds": False,
                    "binary": False,
                },
                timeout=args.timeout,
                retries=args.retries,
                backoff_base=args.backoff_base,
                backoff_max=args.backoff_max,
            )
            rows = normalize_ledger_result(result)
            required_hashes = required_hashes_by_ledger.get(int(ledger_index), set())
            if required_hashes:
                present_hashes = {str(row.get("tx_hash") or "").upper() for row in rows}
                missing = sorted(required_hashes - present_hashes)
                if missing:
                    preview = ",".join(missing[:3])
                    raise RuntimeError(
                        f"required_tx_missing count={len(missing)} preview={preview}"
                    )
            shard_path = _ledger_shard_path(raw_dir, ledger_index)
            payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
            if payload:
                payload += "\n"
            with write_lock:
                shard_path.write_text(payload, encoding="utf-8")
            return True, len(rows), None
        except Exception as e:
            with write_lock:
                with fail_file.open("a", encoding="utf-8") as fp:
                    fp.write(
                        json.dumps(
                            {
                                "ledger_index": int(ledger_index),
                                "status": "error",
                                "error": str(e),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            return False, 0, str(e)

    if pending:
        done_n = 0
        ok_n = 0
        fail_n = 0
        tx_rows_n = 0
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            inflight: set[Future[tuple[bool, int, str | None]]] = set()
            pending_iter = iter(pending)

            def fill() -> None:
                while len(inflight) < max(1, int(args.workers)):
                    try:
                        li = next(pending_iter)
                    except StopIteration:
                        break
                    inflight.add(ex.submit(fetch_one, li))

            fill()
            while inflight:
                done_set, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    inflight.remove(fut)
                    ok, rows_n, _err = fut.result()
                    done_n += 1
                    tx_rows_n += int(rows_n)
                    if ok:
                        ok_n += 1
                    else:
                        fail_n += 1
                    if done_n % max(1, int(args.progress_every)) == 0 or done_n == len(pending):
                        elapsed = time.time() - started
                        print(
                            f"\r[{done_n}/{len(pending)}] ok={ok_n} fail={fail_n} "
                            f"tx_rows={tx_rows_n} elapsed_s={elapsed:.1f}",
                            end="",
                            flush=True,
                        )
                fill()
        print()

    # Always rebuild merged output from shard files in requested ledger order.
    with merged_path.open("w", encoding="utf-8") as out_f:
        merged_rows = 0
        merged_ledgers = 0
        for li in ledgers:
            shard_path = _ledger_shard_path(raw_dir, li)
            if not shard_path.exists():
                continue
            with shard_path.open("r", encoding="utf-8") as in_f:
                wrote_any = False
                for line in in_f:
                    s = line.strip()
                    if not s:
                        continue
                    out_f.write(s + "\n")
                    merged_rows += 1
                    wrote_any = True
                if wrote_any:
                    merged_ledgers += 1

    print(
        f"done ledgers={len(ledgers)} merged_ledgers={merged_ledgers} merged_rows={merged_rows} "
        f"elapsed_s={time.time() - started:.1f}"
    )
    print(f"merged_path={merged_path}")
    print(f"raw_dir={raw_dir}")
    print(f"fail_file={fail_file}")


if __name__ == "__main__":
    main()
