#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split a ledger list into balanced shard files.")
    p.add_argument("--input", required=True, help="Input text file: one ledger index per line")
    p.add_argument("--parts", type=int, required=True, help="Number of shards")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <input_dir>/ledger_shards_<parts>)",
    )
    p.add_argument(
        "--prefix",
        default="prebook_ledgers_part",
        help="Output filename prefix",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")
    if args.parts <= 0:
        raise SystemExit("--parts must be > 0")

    lines = [ln.strip() for ln in in_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    lines = sorted(set(lines), key=int)

    out_dir = Path(args.output_dir) if args.output_dir else (in_path.parent / f"ledger_shards_{args.parts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    shards: list[list[str]] = [[] for _ in range(args.parts)]
    for i, x in enumerate(lines):
        shards[i % args.parts].append(x)

    written = 0
    for i, shard in enumerate(shards, start=1):
        fp = out_dir / f"{args.prefix}_{i:02d}_of_{args.parts:02d}.txt"
        fp.write_text("\n".join(shard) + ("\n" if shard else ""), encoding="utf-8")
        written += len(shard)
        print(f"{fp}\t{len(shard)}")
    print(f"done input_unique={len(lines)} written={written} parts={args.parts} out_dir={out_dir}")


if __name__ == "__main__":
    main()
