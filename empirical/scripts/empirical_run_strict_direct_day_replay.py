#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run tx-prebook replay for one strict-direct day directory backed by "
            "full-ledger metadata, and emit a reusable day-scoped output directory."
        )
    )
    p.add_argument(
        "--day-dir",
        required=True,
        help="Path like artifacts/metadata/.../strict_direct_targets/date_YYYY-MM-DD",
    )
    p.add_argument(
        "--prebook-root",
        default=None,
        help="Optional explicit prebook root. Defaults to artifacts/prebook/<pair>/<window>.",
    )
    p.add_argument(
        "--book-gets-xrp",
        default=None,
        help="Optional explicit ledger-level prebook NDJSON for offers whose TakerGets is XRP.",
    )
    p.add_argument(
        "--book-gets-issued",
        default=None,
        help="Optional explicit ledger-level prebook NDJSON for offers whose TakerGets is the issued side.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional explicit replay output dir. Defaults to "
            "artifacts/prebook/<pair>/<window>/strict_direct_targets/date_YYYY-MM-DD/"
            "tx_prebook_replay_full_ledger_metadata"
        ),
    )
    p.add_argument(
        "--with-events",
        action="store_true",
        help="Also emit tx_prebook_events.ndjson. Default is snapshots only.",
    )
    p.add_argument(
        "--max-offers-per-side",
        type=int,
        default=0,
        help="Optional trim of emitted offers per side. Default keeps all visible offers.",
    )
    p.add_argument(
        "--account-lines-snapshots",
        default=None,
        help=(
            "Boundary account_lines snapshot NDJSON for full-replay fundedness. If omitted, "
            "auto-detect full_replay_account_lines_snapshots.ndjson, then account_lines_snapshots.ndjson."
        ),
    )
    p.add_argument(
        "--allow-missing-account-lines-snapshots",
        action="store_true",
        help=(
            "Allow replay without boundary account_lines snapshots. This can produce stale "
            "pre-state for in-ledger OfferCreate replay and is intended only for legacy debugging."
        ),
    )
    p.add_argument(
        "--prepare-account-lines-targets",
        action="store_true",
        help="Generate full-replay account_lines target CSV before replay.",
    )
    p.add_argument(
        "--account-lines-targets-output",
        default=None,
        help=(
            "Output CSV for --prepare-account-lines-targets. Defaults to "
            "full_ledger_metadata/full_replay_account_lines_targets.csv."
        ),
    )
    p.add_argument(
        "--fetch-account-lines",
        action="store_true",
        help=(
            "Run prepare + account_lines fetch before replay and use the fetched snapshots. "
            "The fetch is pair-agnostic but can be large; tune --fetch-workers/--fetch-rps."
        ),
    )
    p.add_argument("--rpc", default=None, help="XRPL JSON-RPC endpoint for --fetch-account-lines")
    p.add_argument("--fetch-workers", type=int, default=8)
    p.add_argument("--fetch-rps", type=float, default=8.0)
    p.add_argument("--fetch-timeout", type=float, default=20.0)
    p.add_argument(
        "--account-lines-fetch-outdir",
        default=None,
        help=(
            "Output dir for --fetch-account-lines. Defaults to "
            "full_ledger_metadata/full_replay_account_lines_fetch."
        ),
    )
    return p.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _derive_pair_window(day_dir: Path, repo_root: Path) -> tuple[str, str]:
    metadata_root = (repo_root / "artifacts/metadata").resolve()
    try:
        rel = day_dir.resolve().relative_to(metadata_root)
    except ValueError as exc:
        raise RuntimeError(
            f"day dir must live under {metadata_root}: {day_dir}"
        ) from exc
    if len(rel.parts) < 4:
        raise RuntimeError(f"unexpected strict-direct day dir layout: {day_dir}")
    return rel.parts[0], rel.parts[1]


def _read_required_tx(required_tx_csv: Path) -> tuple[int, int, set[str]]:
    ledger_min: int | None = None
    ledger_max: int | None = None
    hashes: set[str] = set()
    with required_tx_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_ledger = str(row.get("ledger_index") or "").strip()
            raw_hash = str(row.get("transaction_hash") or row.get("tx_hash") or "").strip().upper()
            if not raw_ledger or not raw_hash:
                continue
            ledger = int(raw_ledger)
            ledger_min = ledger if ledger_min is None else min(ledger_min, ledger)
            ledger_max = ledger if ledger_max is None else max(ledger_max, ledger)
            hashes.add(raw_hash)
    if ledger_min is None or ledger_max is None or not hashes:
        raise RuntimeError(f"required tx csv is empty or malformed: {required_tx_csv}")
    return ledger_min, ledger_max, hashes


def _validate_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"missing {label}: {path}")


def _resolve_single_prebook(prebook_root: Path, patterns: list[str], label: str) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(prebook_root.glob(pattern)))
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(f"missing {label}: no {patterns} under {prebook_root}")
    raise RuntimeError(
        f"ambiguous {label}: {len(matches)} matches for {patterns} under {prebook_root}: "
        + ", ".join(str(path) for path in matches[:10])
    )


def _default_account_lines_snapshots(full_ledger_dir: Path, day_dir: Path) -> Path:
    candidates = [
        full_ledger_dir / "full_replay_account_lines_snapshots.ndjson",
        full_ledger_dir / "account_lines_snapshots.ndjson",
        day_dir / "full_replay_account_lines_snapshots.ndjson",
        day_dir / "account_lines_snapshots.ndjson",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _scan_snapshot_coverage(snapshots_path: Path, target_hashes: set[str]) -> dict[str, object]:
    seen_by_tx: dict[str, set[str]] = defaultdict(set)
    total_rows = 0
    with snapshots_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            obj = json.loads(line)
            tx_hash = str(obj.get("transaction_hash") or "").strip().upper()
            side = str(obj.get("side") or "").strip()
            if tx_hash:
                seen_by_tx[tx_hash].add(side)

    missing = sorted(target_hashes - set(seen_by_tx))
    side_counter = Counter()
    incomplete: list[str] = []
    for tx_hash, sides in seen_by_tx.items():
        for side in sides:
            side_counter[side] += 1
        if sides != {"getsXRP", "getsrUSD"}:
            incomplete.append(tx_hash)

    return {
        "snapshot_rows": total_rows,
        "snapshot_unique_tx": len(seen_by_tx),
        "target_unique_tx": len(target_hashes),
        "missing_target_tx_count": len(missing),
        "missing_target_tx_sample": missing[:20],
        "incomplete_side_tx_count": len(incomplete),
        "incomplete_side_tx_sample": sorted(incomplete)[:20],
        "side_rows_by_name": dict(sorted(side_counter.items())),
    }


def main() -> int:
    args = parse_args()
    repo_root = _repo_root()
    day_dir = Path(args.day_dir)
    pair, window = _derive_pair_window(day_dir, repo_root)
    day_name = day_dir.name

    full_ledger_dir = day_dir / "full_ledger_metadata"
    metadata_ndjson = full_ledger_dir / "tx_metadata_full_merged.ndjson"
    required_tx_csv = full_ledger_dir / "required_tx.csv"
    if not required_tx_csv.exists():
        required_tx_csv = day_dir / "required_tx.csv"
    account_lines_snapshots = (
        Path(args.account_lines_snapshots)
        if args.account_lines_snapshots
        else _default_account_lines_snapshots(full_ledger_dir, day_dir)
    )

    prebook_root = (
        Path(args.prebook_root)
        if args.prebook_root
        else repo_root / "artifacts/prebook" / pair / window
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root
        / "artifacts/prebook"
        / pair
        / window
        / "strict_direct_targets"
        / day_name
        / "tx_prebook_replay_full_ledger_metadata"
    )

    book_gets_xrp = (
        Path(args.book_gets_xrp)
        if args.book_gets_xrp
        else _resolve_single_prebook(prebook_root, ["book_*_getsXRP.ndjson"], "book getsXRP prebook")
    )
    book_gets_rusd = (
        Path(args.book_gets_issued)
        if args.book_gets_issued
        else _resolve_single_prebook(
            prebook_root,
            ["book_*_getsrUSD.ndjson", "book_*_getsTOKEN.ndjson"],
            "book gets issued prebook",
        )
    )

    _validate_exists(day_dir, "day dir")
    _validate_exists(full_ledger_dir, "full_ledger_metadata dir")
    _validate_exists(metadata_ndjson, "merged metadata ndjson")
    _validate_exists(required_tx_csv, "required tx csv")
    _validate_exists(book_gets_xrp, "book getsXRP prebook")
    _validate_exists(book_gets_rusd, "book gets issued prebook")

    ledger_start, ledger_end, target_hashes = _read_required_tx(required_tx_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    account_lines_targets_output = (
        Path(args.account_lines_targets_output)
        if args.account_lines_targets_output
        else full_ledger_dir / "full_replay_account_lines_targets.csv"
    )
    if args.prepare_account_lines_targets or args.fetch_account_lines:
        prepare_script = repo_root / "empirical/scripts/empirical_prepare_full_replay_account_lines_inputs.py"
        prepare_cmd = [
            sys.executable,
            str(prepare_script),
            "--metadata-ndjson",
            str(metadata_ndjson),
            "--required-tx-csv",
            str(required_tx_csv),
            "--output-csv",
            str(account_lines_targets_output),
        ]
        print("[day-replay:prepare-account-lines] " + " ".join(prepare_cmd), flush=True)
        subprocess.run(prepare_cmd, check=True, cwd=str(repo_root))

    if args.fetch_account_lines:
        fetch_outdir = (
            Path(args.account_lines_fetch_outdir)
            if args.account_lines_fetch_outdir
            else full_ledger_dir / "full_replay_account_lines_fetch"
        )
        fetch_script = repo_root / "empirical/scripts/empirical_fetch_account_lines_snapshots.py"
        fetch_cmd = [
            sys.executable,
            str(fetch_script),
            "--input-csv",
            str(account_lines_targets_output),
            "--outdir",
            str(fetch_outdir),
            "--workers",
            str(args.fetch_workers),
            "--rps",
            str(args.fetch_rps),
            "--timeout",
            str(args.fetch_timeout),
        ]
        if args.rpc:
            fetch_cmd.extend(["--rpc", str(args.rpc)])
        print("[day-replay:fetch-account-lines] " + " ".join(fetch_cmd), flush=True)
        subprocess.run(fetch_cmd, check=True, cwd=str(repo_root))
        account_lines_snapshots = fetch_outdir / "account_lines_snapshots.ndjson"

    if not account_lines_snapshots.exists() and not args.allow_missing_account_lines_snapshots:
        raise RuntimeError(
            "missing full-replay account_lines snapshots. Generate them with "
            "--prepare-account-lines-targets and empirical_fetch_account_lines_snapshots.py, "
            "or pass --fetch-account-lines to let this runner fetch them. "
            "Use --allow-missing-account-lines-snapshots only for legacy stale-prestate debugging. "
            f"expected={account_lines_snapshots}"
        )

    runner = repo_root / "empirical/scripts/empirical_replay_tx_prebook_rust.py"
    snapshots_name = "tx_prebook_snapshots.ndjson"
    events_name = "tx_prebook_events.ndjson"
    cmd = [
        sys.executable,
        str(runner),
        "--ledger-start",
        str(ledger_start),
        "--ledger-end",
        str(ledger_end),
        "--book-gets-xrp",
        str(book_gets_xrp),
        "--book-gets-rusd",
        str(book_gets_rusd),
        "--metadata-ndjson",
        str(metadata_ndjson),
        "--target-tx-file",
        str(required_tx_csv),
        "--output-dir",
        str(output_dir),
        "--snapshots-name",
        snapshots_name,
        "--events-name",
        events_name,
        "--max-offers-per-side",
        str(args.max_offers_per_side),
    ]
    if account_lines_snapshots.exists():
        cmd.extend(["--account-lines-snapshots", str(account_lines_snapshots)])
    if not args.with_events:
        cmd.append("--no-events")

    print("[day-replay] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(repo_root))

    snapshots_path = output_dir / snapshots_name
    _validate_exists(snapshots_path, "replay snapshots output")
    coverage = _scan_snapshot_coverage(snapshots_path, target_hashes)
    manifest = {
        "scope": "strict_direct_day_tx_prebook_replay",
        "day_dir": str(day_dir),
        "pair": pair,
        "window": window,
        "day_name": day_name,
        "ledger_start": ledger_start,
        "ledger_end": ledger_end,
        "input_metadata_ndjson": str(metadata_ndjson),
        "input_required_tx_csv": str(required_tx_csv),
        "input_prebook_gets_xrp": str(book_gets_xrp),
        "input_prebook_gets_rusd": str(book_gets_rusd),
        "input_account_lines_snapshots": (
            str(account_lines_snapshots) if account_lines_snapshots.exists() else None
        ),
        "input_account_lines_targets": (
            str(account_lines_targets_output) if account_lines_targets_output.exists() else None
        ),
        "account_lines_required": not bool(args.allow_missing_account_lines_snapshots),
        "output_dir": str(output_dir),
        "snapshots_path": str(snapshots_path),
        "events_enabled": bool(args.with_events),
        "coverage": coverage,
    }
    manifest_path = output_dir / "replay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
