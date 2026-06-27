#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Finalize a multi-endpoint full-ledger metadata fetch run into one canonical, "
            "reusable directory."
        )
    )
    p.add_argument("--run-dir", required=True, help="Source run directory containing full_ledger_* outputs")
    p.add_argument("--assignment-dir", required=True, help="Assignment directory with ledgers_*.txt and required_tx_*.csv")
    p.add_argument("--master-ledger-list", required=True, help="Canonical ledger_list.txt in desired merged order")
    p.add_argument("--master-required-tx", required=True, help="Canonical required_tx.csv for the full scope")
    p.add_argument("--outdir", required=True, help="Canonical output directory")
    p.add_argument(
        "--low-space-stream-merge",
        action="store_true",
        help=(
            "Build the canonical merged NDJSON directly from source shards without first "
            "moving all raw_ledgers into the output directory."
        ),
    )
    p.add_argument(
        "--delete-source-shards-after-merge",
        action="store_true",
        help=(
            "Only valid with --low-space-stream-merge. Delete each source shard after it "
            "has been merged successfully."
        ),
    )
    p.add_argument(
        "--cleanup-source",
        action="store_true",
        help="Remove the source run directory after successful consolidation",
    )
    return p.parse_args()


def load_ledger_list(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"ledger list not found: {path}")
    out = [int(s.strip()) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    return out


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    assignment_dir = Path(args.assignment_dir)
    master_ledger_list = Path(args.master_ledger_list)
    master_required_tx = Path(args.master_required_tx)
    outdir = Path(args.outdir)

    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")
    if not assignment_dir.exists():
        raise SystemExit(f"assignment dir not found: {assignment_dir}")
    if outdir.exists():
        raise SystemExit(f"output directory already exists; refuse to overwrite: {outdir}")

    expected_ledgers = load_ledger_list(master_ledger_list)
    expected_set = set(expected_ledgers)
    if not expected_ledgers:
        raise SystemExit("master ledger list is empty")

    endpoint_map = {
        "qn": "full_ledger_qn",
        "s1": "full_ledger_s1",
        "s2": "full_ledger_s2",
        "rusty": "full_ledger_rusty",
        "cluster": "full_ledger_cluster",
    }

    outdir.mkdir(parents=True, exist_ok=False)
    out_raw = outdir / "raw_ledgers"
    out_raw.mkdir(parents=True, exist_ok=False)

    assignment_manifest_src = assignment_dir / "assignment_manifest.json"
    _copy_file(assignment_manifest_src, outdir / "assignment_manifest.json")
    _copy_file(master_ledger_list, outdir / "ledger_list.txt")
    _copy_file(master_required_tx, outdir / "required_tx.csv")

    endpoint_stats: dict[str, dict[str, object]] = {}
    seen_ledgers: dict[int, str] = {}
    source_shards: dict[int, Path] = {}

    if args.delete_source_shards_after_merge and not args.low_space_stream_merge:
        raise SystemExit("--delete-source-shards-after-merge requires --low-space-stream-merge")

    out_raw = None
    if not args.low_space_stream_merge:
        out_raw = outdir / "raw_ledgers"
        out_raw.mkdir(parents=True, exist_ok=False)

    for short_name, dirname in endpoint_map.items():
        endpoint_dir = run_dir / dirname
        raw_dir = endpoint_dir / "raw_ledgers"
        if not raw_dir.exists():
            raise RuntimeError(f"missing raw_ledgers dir for endpoint {short_name}: {raw_dir}")

        raw_files = sorted(raw_dir.glob("ledger_*.ndjson"))
        moved = 0
        for shard in raw_files:
            try:
                ledger_index = int(shard.stem.split("_", 1)[1])
            except Exception as exc:
                raise RuntimeError(f"invalid raw ledger filename: {shard}") from exc
            if ledger_index not in expected_set:
                raise RuntimeError(f"unexpected ledger shard outside master scope: {shard}")
            if ledger_index in seen_ledgers:
                raise RuntimeError(
                    f"duplicate ledger shard across endpoints: ledger={ledger_index} "
                    f"first={seen_ledgers[ledger_index]} second={short_name}"
                )
            seen_ledgers[ledger_index] = short_name
            source_shards[ledger_index] = shard
            if out_raw is not None:
                dst = out_raw / shard.name
                shutil.move(str(shard), str(dst))
                source_shards[ledger_index] = dst
            moved += 1

        fail_history = endpoint_dir / "ledger_fetch_fail_history.ndjson"
        if not fail_history.exists():
            fail_history = endpoint_dir / "ledger_fetch_fail.ndjson"
        fail_rows = 0
        if fail_history.exists():
            with fail_history.open("r", encoding="utf-8") as handle:
                fail_rows = sum(1 for _ in handle)
            _copy_file(fail_history, outdir / f"ledger_fetch_fail_history_{short_name}.ndjson")

        endpoint_stats[short_name] = {
            "source_dir": str(endpoint_dir),
            "raw_ledgers_found": moved,
            "finalize_mode": "stream" if args.low_space_stream_merge else "move",
            "fail_history_rows": fail_rows,
        }

    missing = [li for li in expected_ledgers if li not in seen_ledgers]
    if missing:
        sample = ", ".join(str(x) for x in missing[:10])
        raise RuntimeError(f"missing consolidated ledgers: count={len(missing)} sample=[{sample}]")

    merged_path = outdir / "tx_metadata_full_merged.ndjson"
    merged_rows = 0
    merged_ledgers = 0
    with merged_path.open("w", encoding="utf-8") as out_handle:
        for ledger_index in expected_ledgers:
            shard = source_shards.get(ledger_index)
            if shard is None or not shard.exists():
                raise RuntimeError(f"missing raw shard during merge: {shard}")
            wrote_any = False
            with shard.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    payload = line.strip()
                    if not payload:
                        continue
                    out_handle.write(payload + "\n")
                    merged_rows += 1
                    wrote_any = True
            if wrote_any:
                merged_ledgers += 1
            if args.low_space_stream_merge and args.delete_source_shards_after_merge:
                shard.unlink()

    fetch_manifest = {
        "scope": "full_ledger_metadata_finalize",
        "source_run_dir": str(run_dir),
        "assignment_dir": str(assignment_dir),
        "master_ledger_list": str(master_ledger_list),
        "master_required_tx": str(master_required_tx),
        "outdir": str(outdir),
        "merged_ledgers": merged_ledgers,
        "merged_rows": merged_rows,
        "finalize_mode": "stream" if args.low_space_stream_merge else "move",
        "source_shards_deleted_after_merge": bool(args.delete_source_shards_after_merge),
        "endpoint_stats": endpoint_stats,
    }
    (outdir / "fetch_manifest.json").write_text(
        json.dumps(fetch_manifest, indent=2),
        encoding="utf-8",
    )

    if args.cleanup_source:
        shutil.rmtree(run_dir)

    print(f"outdir={outdir}")
    print(f"merged_ledgers={merged_ledgers}")
    print(f"merged_rows={merged_rows}")
    print(f"merged_path={merged_path}")
    print(f"cleanup_source={'1' if args.cleanup_source else '0'}")


if __name__ == "__main__":
    main()
