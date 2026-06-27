#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build pending tx hash list by subtracting fully successful metadata rows "
            "(ledger_index + meta.TransactionIndex present) from input tx list."
        )
    )
    p.add_argument("--input", required=True, help="Input tx hash file (one hash per line)")
    p.add_argument("--ok-ndjson", required=True, help="tx_metadata_ok.ndjson path")
    p.add_argument("--output", required=True, help="Output pending tx hash file")
    return p.parse_args()


def load_input_hashes(path: Path) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for ln in path.read_text(encoding="utf-8").splitlines():
        h = ln.strip().upper()
        if not h or h in seen:
            continue
        seen.add(h)
        out.append(h)
    return out


def load_fully_successful_hashes(path: Path) -> set[str]:
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
            h = str(obj.get("tx_hash") or "").upper()
            if not h:
                continue
            result = obj.get("result") or {}
            li = result.get("ledger_index")
            meta = result.get("meta") or {}
            ti = meta.get("TransactionIndex") if isinstance(meta, dict) else None
            if li is not None and ti is not None:
                done.add(h)
    return done


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    ok_path = Path(args.ok_ndjson)
    out_path = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")

    all_hashes = load_input_hashes(in_path)
    done = load_fully_successful_hashes(ok_path)
    pending = [h for h in all_hashes if h not in done]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(pending) + ("\n" if pending else ""), encoding="utf-8")

    print(f"input_total={len(all_hashes)}")
    print(f"done_full={len(done)}")
    print(f"pending={len(pending)}")
    print(f"output={out_path}")


if __name__ == "__main__":
    main()

