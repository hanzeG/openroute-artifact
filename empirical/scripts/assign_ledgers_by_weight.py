#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Assign ledgers to named endpoint buckets by integer weights, and optionally "
            "split required_tx.csv to the same buckets."
        )
    )
    p.add_argument("--ledger-list", required=True, help="Input text file: one ledger per line")
    p.add_argument(
        "--weights",
        required=True,
        nargs="+",
        help="Bucket weights as name=weight, for example qn=12 s1=18 s2=16 rusty=12 cluster=2",
    )
    p.add_argument(
        "--required-tx-csv",
        default=None,
        help="Optional required_tx.csv containing ledger_index, transaction_hash",
    )
    p.add_argument("--outdir", required=True, help="Output directory")
    return p.parse_args()


def load_ledgers(path: Path) -> list[int]:
    values = [int(s.strip()) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    return sorted(set(values))


def parse_weights(specs: list[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    for spec in specs:
        if "=" not in spec:
            raise RuntimeError(f"invalid weight spec (expected name=weight): {spec}")
        name, raw_weight = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise RuntimeError(f"invalid empty bucket name in weight spec: {spec}")
        if name in seen:
            raise RuntimeError(f"duplicate bucket name in weights: {name}")
        try:
            weight = int(raw_weight.strip())
        except Exception as exc:
            raise RuntimeError(f"invalid bucket weight in spec: {spec}") from exc
        if weight <= 0:
            raise RuntimeError(f"bucket weight must be > 0: {spec}")
        seen.add(name)
        out.append((name, weight))
    return out


def largest_remainder_counts(total: int, weighted_names: list[tuple[str, int]]) -> dict[str, int]:
    weight_sum = sum(weight for _, weight in weighted_names)
    floors: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0
    for name, weight in weighted_names:
        exact = total * weight / weight_sum
        floor_count = int(exact)
        floors[name] = floor_count
        allocated += floor_count
        remainders.append((exact - floor_count, name))
    remaining = total - allocated
    for _fraction, name in sorted(remainders, key=lambda x: (-x[0], x[1]))[:remaining]:
        floors[name] += 1
    return floors


def split_fair_interleaved(
    ledgers: list[int],
    counts: dict[str, int],
    ordered_names: list[str],
) -> dict[str, list[int]]:
    total = len(ledgers)
    assigned_counts = {name: 0 for name in ordered_names}
    out = {name: [] for name in ordered_names}
    name_rank = {name: idx for idx, name in enumerate(ordered_names)}

    for position, ledger in enumerate(ledgers, start=1):
        best_name = None
        best_deficit = None
        for name in ordered_names:
            target = counts[name]
            if assigned_counts[name] >= target:
                continue
            ideal = position * target / total
            deficit = ideal - assigned_counts[name]
            rank_key = (deficit, target, -name_rank[name])
            if best_deficit is None or rank_key > best_deficit:
                best_deficit = rank_key
                best_name = name
        if best_name is None:
            raise RuntimeError(f"failed to assign ledger at position={position} ledger={ledger}")
        out[best_name].append(ledger)
        assigned_counts[best_name] += 1

    for name in ordered_names:
        if assigned_counts[name] != counts[name]:
            raise RuntimeError(
                f"assignment count mismatch for {name}: assigned={assigned_counts[name]} target={counts[name]}"
            )
    return out


def load_required_tx_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        fieldnames = set(reader.fieldnames or [])
        need = {"ledger_index", "transaction_hash"}
        if not need.issubset(fieldnames):
            raise RuntimeError(f"required_tx csv missing columns {sorted(need - fieldnames)}: {path}")
        rows: list[dict[str, str]] = []
        for row in reader:
            ledger_index = str(row.get("ledger_index") or "").strip()
            tx_hash = str(row.get("transaction_hash") or "").strip().upper()
            if not ledger_index or not tx_hash:
                continue
            rows.append({"ledger_index": ledger_index, "transaction_hash": tx_hash})
        return rows


def write_required_tx_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ledger_index", "transaction_hash"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ledger_path = Path(args.ledger_list)
    if not ledger_path.exists():
        raise SystemExit(f"ledger list not found: {ledger_path}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ledgers = load_ledgers(ledger_path)
    if not ledgers:
        raise SystemExit("ledger list is empty")

    weighted_names = parse_weights(args.weights)
    bucket_names = [name for name, _ in weighted_names]
    counts = largest_remainder_counts(len(ledgers), weighted_names)
    assignments = split_fair_interleaved(ledgers, counts, bucket_names)

    required_rows = None
    if args.required_tx_csv:
        required_tx_path = Path(args.required_tx_csv)
        if not required_tx_path.exists():
            raise SystemExit(f"required tx csv not found: {required_tx_path}")
        required_rows = load_required_tx_rows(required_tx_path)

    manifest: dict[str, object] = {
        "input": {
            "ledger_list": str(ledger_path),
            "required_tx_csv": str(args.required_tx_csv) if args.required_tx_csv else None,
            "input_ledgers": len(ledgers),
            "ledger_min": min(ledgers),
            "ledger_max": max(ledgers),
        },
        "weights": {name: weight for name, weight in weighted_names},
        "assignments": {},
    }

    for name in bucket_names:
        bucket_ledgers = assignments[name]
        ledger_file = outdir / f"ledgers_{name}.txt"
        ledger_file.write_text(
            "".join(f"{ledger}\n" for ledger in bucket_ledgers),
            encoding="utf-8",
        )

        bucket_entry: dict[str, object] = {
            "ledgers": len(bucket_ledgers),
            "ledger_list": str(ledger_file),
            "ledger_min": min(bucket_ledgers) if bucket_ledgers else None,
            "ledger_max": max(bucket_ledgers) if bucket_ledgers else None,
        }

        if required_rows is not None:
            ledger_set = {str(x) for x in bucket_ledgers}
            bucket_rows = [row for row in required_rows if row["ledger_index"] in ledger_set]
            required_tx_file = outdir / f"required_tx_{name}.csv"
            write_required_tx_csv(required_tx_file, bucket_rows)
            bucket_entry["required_tx_rows"] = len(bucket_rows)
            bucket_entry["required_tx_csv"] = str(required_tx_file)

        manifest["assignments"][name] = bucket_entry

    manifest_path = outdir / "assignment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for name in bucket_names:
        entry = manifest["assignments"][name]
        ledgers_n = entry["ledgers"]
        tx_rows_n = entry.get("required_tx_rows")
        if tx_rows_n is None:
            print(f"{name}\tledgers={ledgers_n}")
        else:
            print(f"{name}\tledgers={ledgers_n}\trequired_tx_rows={tx_rows_n}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
