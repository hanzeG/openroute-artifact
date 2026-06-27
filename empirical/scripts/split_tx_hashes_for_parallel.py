#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split tx hash list into balanced shard files.")
    p.add_argument("--input", required=True, help="Input tx hash file (one hash per line)")
    p.add_argument("--parts", type=int, default=8, help="Number of shards to create")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for shard files (default: <input_dir>/shards_<parts>)",
    )
    p.add_argument(
        "--prefix",
        default="tx_hashes_part",
        help="Output filename prefix (default: tx_hashes_part)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.parts <= 0:
        raise SystemExit("--parts must be > 0")

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"input file not found: {in_path}")

    lines = [ln.strip() for ln in in_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    total = len(lines)
    if total == 0:
        raise SystemExit("input file is empty")

    out_dir = Path(args.output_dir) if args.output_dir else (in_path.parent / f"shards_{args.parts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    shards: list[list[str]] = [[] for _ in range(args.parts)]
    for i, h in enumerate(lines):
        shards[i % args.parts].append(h)

    width = max(2, len(str(args.parts)))
    written = 0
    for i, shard in enumerate(shards, start=1):
        out_file = out_dir / f"{args.prefix}_{i:0{width}d}_of_{args.parts}.txt"
        out_file.write_text("\n".join(shard) + ("\n" if shard else ""), encoding="utf-8")
        print(f"{out_file}\t{len(shard)}")
        written += len(shard)

    print(f"TOTAL\t{written}")


if __name__ == "__main__":
    main()

