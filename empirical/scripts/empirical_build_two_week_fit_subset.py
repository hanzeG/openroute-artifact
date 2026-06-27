#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a smaller canonical dataset root for the two-week isolated direct fit "
            "by filtering the source dataset to an explicit target-tx subset."
        )
    )
    parser.add_argument(
        "--source-dataset-root",
        required=True,
        help="Canonical two-week isolated dataset root under artifacts/fit_inputs/...",
    )
    parser.add_argument(
        "--target-tx-file",
        required=True,
        help=(
            "Target tx list. Supports .txt, .csv, .parquet, or fit manifest .json "
            "(reads fit_manifest.samples or top-level samples)."
        ),
    )
    parser.add_argument(
        "--output-dataset-root",
        required=True,
        help="Output subset dataset root under artifacts/fit_inputs/...",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files in the output dataset root.",
    )
    return parser.parse_args()


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"missing {label}: {path}")


def _normalise_tx_hash(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text


def _rel_or_abs(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_target_hashes(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".list"}:
        raw = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        hashes = [_normalise_tx_hash(x) for x in raw if str(x).strip()]
    elif suffix == ".csv":
        frame = pd.read_csv(path)
        hashes = _load_hashes_from_frame(frame=frame, source=str(path))
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
        hashes = _load_hashes_from_frame(frame=frame, source=str(path))
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        hashes = _load_hashes_from_json(payload=payload, source=str(path))
    else:
        raise RuntimeError(f"unsupported target tx file type: {path}")
    out: list[str] = []
    seen: set[str] = set()
    for tx_hash in hashes:
        if not tx_hash:
            continue
        if tx_hash in seen:
            continue
        seen.add(tx_hash)
        out.append(tx_hash)
    if not out:
        raise RuntimeError(f"no target tx hashes loaded from {path}")
    return out


def _load_hashes_from_frame(*, frame: pd.DataFrame, source: str) -> list[str]:
    for column in ("transaction_hash", "tx_hash"):
        if column not in frame.columns:
            continue
        values = [_normalise_tx_hash(x) for x in frame[column].dropna().astype(str).tolist()]
        values = [x for x in values if x]
        if values:
            return values
    raise RuntimeError(
        f"target tx file missing supported hash column ['transaction_hash', 'tx_hash']: {source}"
    )


def _load_hashes_from_json(*, payload: Any, source: str) -> list[str]:
    if isinstance(payload, list):
        return [_normalise_tx_hash(x) for x in payload]
    if isinstance(payload, dict):
        candidates = [
            payload.get("samples"),
            payload.get("transaction_hashes"),
            payload.get("tx_hashes"),
            (payload.get("fit_manifest") or {}).get("samples") if isinstance(payload.get("fit_manifest"), dict) else None,
        ]
        for candidate in candidates:
            if not isinstance(candidate, list):
                continue
            values = [_normalise_tx_hash(x) for x in candidate]
            values = [x for x in values if x]
            if values:
                return values
    raise RuntimeError(f"unsupported json target tx payload: {source}")


def _read_required_tx_rows(required_tx_csv: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    with required_tx_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"required tx csv is empty: {required_tx_csv}")
    by_hash: dict[str, dict[str, str]] = {}
    for row in rows:
        tx_hash = _normalise_tx_hash(row.get("transaction_hash") or row.get("tx_hash"))
        if not tx_hash:
            raise RuntimeError(f"required tx csv row missing transaction hash: {row}")
        if tx_hash in by_hash:
            raise RuntimeError(f"duplicate transaction hash in required tx csv: {tx_hash}")
        by_hash[tx_hash] = row
    return rows, by_hash


def _write_required_tx_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["ledger_index", "transaction_index", "transaction_hash"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def _filter_ndjson_by_hash(
    *,
    input_path: Path,
    output_path: Path,
    target_hashes: set[str],
    hash_key_candidates: tuple[str, ...],
) -> int:
    matched: set[str] = set()
    line_count = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                obj = json.loads(payload)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to parse ndjson row at {input_path}:{line_no}: {type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                continue
            tx_hash = ""
            for key in hash_key_candidates:
                maybe = obj.get(key)
                if maybe:
                    tx_hash = _normalise_tx_hash(maybe)
                    break
            if tx_hash not in target_hashes:
                continue
            dst.write(line)
            line_count += 1
            matched.add(tx_hash)
    missing = sorted(target_hashes - matched)
    if missing:
        raise RuntimeError(
            f"missing {len(missing)} target txs while filtering {input_path.name}: {', '.join(missing[:5])}"
        )
    return line_count


def _filter_parquet_by_hash(*, input_path: Path, output_path: Path, target_hashes: list[str]) -> int:
    if not target_hashes:
        raise RuntimeError("target_hashes must be non-empty")
    try:
        frame = pd.read_parquet(input_path, filters=[("transaction_hash", "in", target_hashes)])
    except Exception:
        frame = pd.read_parquet(input_path)
    if "transaction_hash" not in frame.columns:
        raise RuntimeError(f"parquet file missing transaction_hash column: {input_path}")
    filtered = frame[frame["transaction_hash"].astype(str).str.upper().isin(set(target_hashes))].copy()
    filtered.to_parquet(output_path, index=False)
    return int(len(filtered))


def _copy_tick_size_snapshots(*, source_root: Path, output_root: Path, max_ledger_index: int) -> list[str]:
    copied: list[str] = []
    for path in sorted(source_root.glob("offer_tick_sizes_at_ledger_*.json")):
        suffix = path.stem.removeprefix("offer_tick_sizes_at_ledger_")
        try:
            ledger_index = int(suffix)
        except ValueError as exc:
            raise RuntimeError(f"invalid tick-size snapshot filename: {path}") from exc
        if ledger_index > max_ledger_index:
            continue
        target = output_root / path.name
        target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        copied.append(path.name)
    return copied


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    source_root = Path(args.source_dataset_root).resolve()
    output_root = Path(args.output_dataset_root).resolve()
    target_file = Path(args.target_tx_file).resolve()

    _ensure_exists(source_root, "source dataset root")
    _ensure_exists(target_file, "target tx file")

    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise RuntimeError(f"output dataset root already exists and is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_path = source_root / "tx_metadata_full_merged.ndjson"
    snapshots_path = source_root / "tx_prebook_snapshots.ndjson"
    amm_swaps_path = source_root / "amm_swaps_two_week.parquet"
    amm_fees_path = source_root / "amm_fees_two_week.parquet"
    required_tx_path = source_root / "required_tx.csv"
    ledger_list_path = source_root / "ledger_list.txt"
    dataset_manifest_path = source_root / "dataset_manifest.json"

    for path, label in (
        (metadata_path, "metadata ndjson"),
        (snapshots_path, "tx-prebook snapshots"),
        (amm_swaps_path, "amm swaps parquet"),
        (amm_fees_path, "amm fees parquet"),
        (required_tx_path, "required tx csv"),
        (ledger_list_path, "ledger list"),
        (dataset_manifest_path, "dataset manifest"),
    ):
        _ensure_exists(path, label)

    target_hashes = _load_target_hashes(target_file)
    target_set = set(target_hashes)

    _source_rows, source_required_by_hash = _read_required_tx_rows(required_tx_path)
    missing_required = [tx_hash for tx_hash in target_hashes if tx_hash not in source_required_by_hash]
    if missing_required:
        raise RuntimeError(
            f"{len(missing_required)} target txs missing from source required_tx.csv: {', '.join(missing_required[:5])}"
        )
    subset_required_rows = [source_required_by_hash[tx_hash] for tx_hash in target_hashes]
    _write_required_tx_csv(output_root / "required_tx.csv", subset_required_rows)

    selected_ledgers = {int(row["ledger_index"]) for row in subset_required_rows}
    with ledger_list_path.open("r", encoding="utf-8") as handle:
        source_ledger_lines = [line.strip() for line in handle if line.strip()]
    filtered_ledger_lines = [line for line in source_ledger_lines if int(line) in selected_ledgers]
    if len(filtered_ledger_lines) != len(selected_ledgers):
        missing_ledgers = sorted(selected_ledgers - {int(line) for line in filtered_ledger_lines})
        raise RuntimeError(f"missing ledgers in source ledger list: {missing_ledgers[:5]}")
    (output_root / "ledger_list.txt").write_text("\n".join(filtered_ledger_lines) + "\n", encoding="utf-8")

    metadata_rows = _filter_ndjson_by_hash(
        input_path=metadata_path,
        output_path=output_root / "tx_metadata_full_merged.ndjson",
        target_hashes=target_set,
        hash_key_candidates=("tx_hash", "transaction_hash"),
    )
    snapshot_rows = _filter_ndjson_by_hash(
        input_path=snapshots_path,
        output_path=output_root / "tx_prebook_snapshots.ndjson",
        target_hashes=target_set,
        hash_key_candidates=("transaction_hash", "tx_hash"),
    )
    amm_swap_rows = _filter_parquet_by_hash(
        input_path=amm_swaps_path,
        output_path=output_root / "amm_swaps_two_week.parquet",
        target_hashes=target_hashes,
    )
    amm_fee_rows = _filter_parquet_by_hash(
        input_path=amm_fees_path,
        output_path=output_root / "amm_fees_two_week.parquet",
        target_hashes=target_hashes,
    )
    copied_tick_size_files = _copy_tick_size_snapshots(
        source_root=source_root,
        output_root=output_root,
        max_ledger_index=max(selected_ledgers),
    )

    subset_manifest = {
        "scope": "strict_direct_two_week_isolated_fit_input_subset",
        "dataset_root": _rel_or_abs(output_root, repo_root),
        "source_dataset_root": _rel_or_abs(source_root, repo_root),
        "target_tx_file": _rel_or_abs(target_file, repo_root),
        "selected_tx_count": len(target_hashes),
        "selected_ledger_count": len(selected_ledgers),
        "metadata_ndjson": _rel_or_abs(output_root / "tx_metadata_full_merged.ndjson", repo_root),
        "tx_prebook_snapshots": _rel_or_abs(output_root / "tx_prebook_snapshots.ndjson", repo_root),
        "amm_swaps": _rel_or_abs(output_root / "amm_swaps_two_week.parquet", repo_root),
        "amm_fees": _rel_or_abs(output_root / "amm_fees_two_week.parquet", repo_root),
        "required_tx_csv": _rel_or_abs(output_root / "required_tx.csv", repo_root),
        "ledger_list": _rel_or_abs(output_root / "ledger_list.txt", repo_root),
        "tick_size_snapshot_files": copied_tick_size_files,
        "row_counts": {
            "required_tx": len(subset_required_rows),
            "metadata_ndjson": metadata_rows,
            "tx_prebook_snapshots": snapshot_rows,
            "amm_swaps": amm_swap_rows,
            "amm_fees": amm_fee_rows,
        },
        "notes": [
            "Subset root generated from the canonical two-week isolated fit input root.",
            "Metadata and tx_prebook NDJSON were filtered by explicit target tx hash membership.",
            "AMM parquet tables were filtered by transaction_hash.",
            "Tick-size snapshots, when present, were copied up to the maximum selected ledger index.",
        ],
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(subset_manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(subset_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
