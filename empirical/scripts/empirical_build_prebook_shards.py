#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build contiguous shards for prebook ndjson files.")
    p.add_argument(
        "--prebook-dir",
        default="artifacts/prebook/rlusd_xrp/final_prebook_20260304",
        help="Directory containing book_rusd_xrp_getsXRP.ndjson and book_rusd_xrp_getsrUSD.ndjson",
    )
    p.add_argument("--shards", type=int, default=256, help="Number of shards.")
    p.add_argument(
        "--outdir-name",
        default="shards_256",
        help="Output subdirectory name under --prebook-dir",
    )
    return p.parse_args()


def _ledger_index_of(obj: dict[str, Any]) -> int:
    li = obj.get("ledger_index")
    if li is None:
        li = obj.get("ledger") or obj.get("ledger_current_index")
    if li is None:
        raise RuntimeError("line missing ledger_index/ledger/ledger_current_index")
    return int(li)


def _load_rows(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                li = _ledger_index_of(obj)
            except Exception as e:
                raise RuntimeError(f"{path}:{i} parse failed: {e}") from e
            rows.append((li, s))
    rows.sort(key=lambda x: x[0])
    return rows


def _write_shards(rows: list[tuple[int, str]], out_dir: Path, shards: int, label: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    if n == 0:
        return {"rows": 0, "shards": 0, "ranges": []}

    ranges: list[dict[str, Any]] = []
    for sid in range(shards):
        lo = (n * sid) // shards
        hi = (n * (sid + 1)) // shards
        if lo >= hi:
            continue
        chunk = rows[lo:hi]
        shard_path = out_dir / f"{label}_part_{sid:03d}_of_{shards:03d}.ndjson"
        with shard_path.open("w", encoding="utf-8") as w:
            for _, line in chunk:
                w.write(line + "\n")
        ranges.append(
            {
                "sid": sid,
                "path": str(shard_path),
                "rows": len(chunk),
                "ledger_min": int(chunk[0][0]),
                "ledger_max": int(chunk[-1][0]),
            }
        )
    return {"rows": n, "shards": len(ranges), "ranges": ranges}


def main() -> None:
    args = _parse_args()
    if args.shards <= 0:
        raise ValueError("--shards must be positive")

    base = Path(args.prebook_dir).resolve()
    gets_xrp = base / "book_rusd_xrp_getsXRP.ndjson"
    gets_rusd = base / "book_rusd_xrp_getsrUSD.ndjson"
    if not gets_xrp.exists() or not gets_rusd.exists():
        raise FileNotFoundError(f"missing prebook files under {base}")

    outdir = base / args.outdir_name
    outdir.mkdir(parents=True, exist_ok=True)

    x_rows = _load_rows(gets_xrp)
    r_rows = _load_rows(gets_rusd)

    x_summary = _write_shards(x_rows, outdir / "getsXRP", args.shards, "getsXRP")
    r_summary = _write_shards(r_rows, outdir / "getsrUSD", args.shards, "getsrUSD")

    summary = {
        "prebook_dir": str(base),
        "shards": int(args.shards),
        "outdir": str(outdir),
        "getsXRP": x_summary,
        "getsrUSD": r_summary,
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"built prebook shards: {outdir}")
    print(f"summary: {summary_path}")
    print(f"getsXRP: rows={x_summary['rows']} shards={x_summary['shards']}")
    print(f"getsrUSD: rows={r_summary['rows']} shards={r_summary['shards']}")


if __name__ == "__main__":
    main()

