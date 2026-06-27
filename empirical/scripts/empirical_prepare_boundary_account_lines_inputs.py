#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract boundary account-lines fetch targets from a fit output by finding "
            "structure-mismatch tx where the model-only CLOB offer is an overlay-created "
            "offer with null owner_funds."
        )
    )
    parser.add_argument("--fit-output-dir", required=True, help="Fit output dir containing summary.json and reports/")
    parser.add_argument("--tx-prebook-snapshots", required=True, help="tx_prebook_snapshots.ndjson used by the fit run")
    parser.add_argument("--output-csv", required=True, help="Output CSV of boundary account-lines fetch keys")
    return parser.parse_args()


def _load_structure_mismatch_reports(fit_output_dir: Path) -> dict[str, dict[str, Any]]:
    summary_path = fit_output_dir / "summary.json"
    report_dir = fit_output_dir / "reports"
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        comparison = row.get("comparison") or {}
        if comparison.get("structure_match"):
            continue
        tx_hash = str(row.get("tx_hash") or "").strip().upper()
        if not tx_hash:
            continue
        report = json.loads((report_dir / f"{tx_hash}.json").read_text(encoding="utf-8"))
        model_ids = {
            leg["source_id"]
            for leg in report.get("model", {}).get("legs", [])
            if leg.get("src") == "CLOB" and leg.get("source_id")
        }
        real_ids = {
            leg["offer_id"]
            for leg in report.get("real", {}).get("legs", [])
            if leg.get("src") == "CLOB" and leg.get("offer_id")
        }
        out[tx_hash] = {
            "ledger_index": int(report["ledger_index"]),
            "transaction_index": int(report["transaction_index"]),
            "selected_side": report["snapshot"]["selected_side"],
            "model_only_offer_ids": sorted(model_ids - real_ids),
        }
    return out


def _csv_rows_for_candidates(
    structure_mismatch: dict[str, dict[str, Any]],
    snapshots_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    with snapshots_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            snapshot = json.loads(payload)
            tx_hash = str(snapshot.get("transaction_hash") or "").strip().upper()
            meta = structure_mismatch.get(tx_hash)
            if meta is None:
                continue
            if snapshot.get("side") != meta["selected_side"]:
                continue

            pre_overlay_offer_ids = set(snapshot.get("pre_overlay_offer_ids") or [])
            offers = {offer["offer_id"]: offer for offer in snapshot.get("offers", []) if offer.get("offer_id")}
            for offer_id in meta["model_only_offer_ids"]:
                offer = offers.get(offer_id)
                if offer is None:
                    continue
                if offer.get("owner_funds") is not None:
                    continue
                if offer_id in pre_overlay_offer_ids:
                    continue

                taker_gets = offer.get("taker_gets")
                if not isinstance(taker_gets, dict):
                    unsupported.append(
                        {
                            "tx_hash": tx_hash,
                            "ledger_index": meta["ledger_index"],
                            "transaction_index": meta["transaction_index"],
                            "offer_id": offer_id,
                            "owner_account": offer.get("account"),
                            "reason": "model-only null owner_funds offer has XRP TakerGets; account_lines cannot seed this domain",
                        }
                    )
                    continue

                currency = str(taker_gets.get("currency") or "").strip()
                issuer = str(taker_gets.get("issuer") or "").strip()
                if not currency or not issuer:
                    unsupported.append(
                        {
                            "tx_hash": tx_hash,
                            "ledger_index": meta["ledger_index"],
                            "transaction_index": meta["transaction_index"],
                            "offer_id": offer_id,
                            "owner_account": offer.get("account"),
                            "reason": "model-only null owner_funds IOU offer missing currency/issuer",
                        }
                    )
                    continue

                rows.append(
                    {
                        "tx_hash": tx_hash,
                        "target_ledger_index": meta["ledger_index"],
                        "boundary_ledger_index": meta["ledger_index"] - 1,
                        "transaction_index": meta["transaction_index"],
                        "selected_side": meta["selected_side"],
                        "offer_id": offer_id,
                        "owner_account": str(offer.get("account") or ""),
                        "currency": currency,
                        "issuer": issuer,
                        "quality": offer.get("quality"),
                    }
                )
    rows.sort(key=lambda row: (row["boundary_ledger_index"], row["owner_account"], row["currency"], row["issuer"], row["tx_hash"], row["offer_id"]))
    unsupported.sort(key=lambda row: (row["ledger_index"], row["transaction_index"], row["tx_hash"], row["offer_id"]))
    return rows, unsupported


def main() -> None:
    args = _parse_args()
    fit_output_dir = Path(args.fit_output_dir)
    snapshots_path = Path(args.tx_prebook_snapshots)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    structure_mismatch = _load_structure_mismatch_reports(fit_output_dir)
    rows, unsupported = _csv_rows_for_candidates(structure_mismatch, snapshots_path)

    fieldnames = [
        "tx_hash",
        "target_ledger_index",
        "boundary_ledger_index",
        "transaction_index",
        "selected_side",
        "offer_id",
        "owner_account",
        "currency",
        "issuer",
        "quality",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    unique_keys = {
        (
            row["boundary_ledger_index"],
            row["owner_account"],
            row["currency"],
            row["issuer"],
        )
        for row in rows
    }
    manifest = {
        "scope": "boundary_account_lines_inputs",
        "fit_output_dir": str(fit_output_dir),
        "tx_prebook_snapshots": str(snapshots_path),
        "output_csv": str(output_csv),
        "candidate_rows": len(rows),
        "unique_boundary_account_line_keys": len(unique_keys),
        "unsupported_candidates": len(unsupported),
        "unsupported_sample": unsupported[:20],
    }
    manifest_path = output_csv.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
