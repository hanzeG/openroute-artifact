#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


XRP = "XRP"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build (ledger, account) fetch targets for account_offers snapshots from "
            "isolated-fit reports and tx-prebook snapshots."
        )
    )
    parser.add_argument("--report-dir", required=True, help="Directory containing isolated-fit report json files")
    parser.add_argument("--tx-prebook-snapshots", required=True, help="tx_prebook_snapshots.ndjson path")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument(
        "--mag-threshold",
        type=float,
        default=None,
        help="Select reports whose max(abs(total_in_diff), abs(total_out_diff)) exceeds this threshold",
    )
    parser.add_argument(
        "--tx-hash",
        action="append",
        default=[],
        help="Explicit tx hash to include. May be repeated.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write empty target files instead of failing when no reports are selected.",
    )
    return parser.parse_args()


def _normalise_tx_hash(value: str) -> str:
    return str(value or "").strip().upper()


def _selected_side(report: dict[str, Any]) -> str:
    out_cur = str(((report.get("intent") or {}).get("out_cur")) or "")
    return "getsXRP" if out_cur == XRP else "getsrUSD"


def _magnitude(report: dict[str, Any]) -> float:
    cmp = report.get("comparison") or {}
    return max(abs(float(cmp.get("total_in_diff") or 0)), abs(float(cmp.get("total_out_diff") or 0)))


def _load_selected_reports(report_dir: Path, *, mag_threshold: float | None, explicit_txs: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        tx_hash = _normalise_tx_hash(report.get("tx_hash") or "")
        if not tx_hash:
            continue
        if tx_hash in explicit_txs:
            out.append(report)
            continue
        if mag_threshold is not None and _magnitude(report) > mag_threshold:
            out.append(report)
    dedup: dict[str, dict[str, Any]] = {}
    for report in out:
        dedup[_normalise_tx_hash(report.get("tx_hash") or "")] = report
    return list(dedup.values())


def _load_snapshot_rows(path: Path, *, selected_targets: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                row = json.loads(payload)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to parse tx_prebook snapshot at {path}:{line_no}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            tx_hash = _normalise_tx_hash(row.get("transaction_hash") or "")
            side = str(row.get("side") or "")
            key = (tx_hash, side)
            if key not in selected_targets:
                continue
            out[key] = row
            if len(out) == len(selected_targets):
                break
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = _parse_args()
    report_dir = Path(args.report_dir)
    snapshots_path = Path(args.tx_prebook_snapshots)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    explicit_txs = {_normalise_tx_hash(tx_hash) for tx_hash in args.tx_hash}
    reports = _load_selected_reports(
        report_dir,
        mag_threshold=args.mag_threshold,
        explicit_txs=explicit_txs,
    )
    if not reports:
        if not args.allow_empty:
            raise RuntimeError("no reports selected")
        _write_csv(
            outdir / "selected_txs.csv",
            [],
            fieldnames=[
                "tx_hash",
                "ledger_index",
                "transaction_index",
                "selected_side",
                "used_prebook_ledger",
                "magnitude",
            ],
        )
        _write_csv(
            outdir / "account_offers_targets.csv",
            [],
            fieldnames=["ledger_index", "account"],
        )
        manifest = {
            "scope": "account_offers_snapshot_inputs",
            "report_dir": str(report_dir),
            "tx_prebook_snapshots": str(snapshots_path),
            "mag_threshold": args.mag_threshold,
            "explicit_tx_hashes": sorted(explicit_txs),
            "selected_tx_count": 0,
            "target_pair_count": 0,
            "selected_txs_csv": str(outdir / "selected_txs.csv"),
            "account_offers_targets_csv": str(outdir / "account_offers_targets.csv"),
        }
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return

    selected_targets = {
        (_normalise_tx_hash(report["tx_hash"]), _selected_side(report))
        for report in reports
    }
    snapshot_rows = _load_snapshot_rows(snapshots_path, selected_targets=selected_targets)

    selected_rows: list[dict[str, Any]] = []
    target_rows_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for report in sorted(reports, key=lambda row: (int(row["ledger_index"]), int(row["transaction_index"]))):
        tx_hash = _normalise_tx_hash(report["tx_hash"])
        side = _selected_side(report)
        snapshot = snapshot_rows.get((tx_hash, side))
        if snapshot is None:
            raise RuntimeError(f"missing tx-prebook snapshot for tx={tx_hash} side={side}")
        used_prebook_ledger = snapshot.get("used_prebook_ledger")
        if used_prebook_ledger is None:
            raise RuntimeError(f"snapshot missing used_prebook_ledger for tx={tx_hash} side={side}")
        fetch_ledger_index = int(used_prebook_ledger)
        mag = _magnitude(report)
        selected_rows.append(
            {
                "tx_hash": tx_hash,
                "ledger_index": int(report["ledger_index"]),
                "transaction_index": int(report["transaction_index"]),
                "selected_side": side,
                "used_prebook_ledger": fetch_ledger_index,
                "magnitude": format(mag, "f"),
            }
        )
        seen_accounts: set[str] = set()
        for offer in snapshot.get("offers") or []:
            if not isinstance(offer, dict):
                continue
            account = str(offer.get("account") or "")
            if not account or account in seen_accounts:
                continue
            seen_accounts.add(account)
            key = (fetch_ledger_index, account)
            target_rows_by_key.setdefault(
                key,
                {
                    "ledger_index": fetch_ledger_index,
                    "account": account,
                },
            )

    target_rows = sorted(target_rows_by_key.values(), key=lambda row: (int(row["ledger_index"]), str(row["account"])))
    _write_csv(
        outdir / "selected_txs.csv",
        selected_rows,
        fieldnames=["tx_hash", "ledger_index", "transaction_index", "selected_side", "used_prebook_ledger", "magnitude"],
    )
    _write_csv(
        outdir / "account_offers_targets.csv",
        target_rows,
        fieldnames=["ledger_index", "account"],
    )

    manifest = {
        "scope": "account_offers_snapshot_inputs",
        "report_dir": str(report_dir),
        "tx_prebook_snapshots": str(snapshots_path),
        "mag_threshold": args.mag_threshold,
        "explicit_tx_hashes": sorted(explicit_txs),
        "selected_tx_count": len(selected_rows),
        "target_pair_count": len(target_rows),
        "selected_txs_csv": str(outdir / "selected_txs.csv"),
        "account_offers_targets_csv": str(outdir / "account_offers_targets.csv"),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
