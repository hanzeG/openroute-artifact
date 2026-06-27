#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from xrpl_router.tx_intent import (
    TxIntent,
    build_intent_from_metadata_result,
    extract_tx_hash_from_metadata_obj,
)

RUSD_HEX = "524C555344000000000000000000000000000000"
XRP = "XRP"
RULE_NAME = "strict_direct_v2"
RIPPLE_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build the strict direct target-tx table used for isolated exact-fit work. "
            "This scope keeps pair-direct OfferCreate plus pair-direct Payment with no Paths "
            "or one pair-local explicit path."
        )
    )
    p.add_argument(
        "--tx-sequence",
        required=True,
        help="Path to full_tx_sequence.parquet or .csv",
    )
    p.add_argument(
        "--metadata-ndjson",
        required=True,
        help="Path to tx_metadata_full_merged.ndjson",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Output directory for strict target artifacts",
    )
    p.add_argument(
        "--utc-date",
        default=None,
        help="Optional UTC calendar day filter in YYYY-MM-DD",
    )
    p.add_argument(
        "--base-currency",
        default=RUSD_HEX,
        help="Issued asset currency used for pair filtering",
    )
    p.add_argument(
        "--base-label",
        default="rUSD",
        help="Human-readable label for the issued asset in direction output",
    )
    p.add_argument(
        "--counter-currency",
        default=XRP,
        help="Counter asset currency used for pair filtering",
    )
    p.add_argument(
        "--counter-label",
        default="XRP",
        help="Human-readable label for the counter asset in direction output",
    )
    return p.parse_args()


def load_tx_sequence(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise RuntimeError(f"unsupported tx-sequence format: {path}")

    required = {"tx_hash", "ledger_index", "transaction_index"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"tx-sequence missing required columns {missing}: {path}")

    df = df.copy()
    df["tx_hash"] = df["tx_hash"].astype(str).str.upper()
    df["ledger_index"] = pd.to_numeric(df["ledger_index"], errors="coerce")
    df["transaction_index"] = pd.to_numeric(df["transaction_index"], errors="coerce")
    df = df.dropna(subset=["tx_hash", "ledger_index", "transaction_index"]).copy()
    df["ledger_index"] = df["ledger_index"].astype("int64")
    df["transaction_index"] = df["transaction_index"].astype("int64")
    if df["tx_hash"].duplicated().any():
        dup = df.loc[df["tx_hash"].duplicated(), "tx_hash"].iloc[0]
        raise RuntimeError(f"tx-sequence contains duplicate tx_hash: {dup}")
    return df


def _is_target_pair_direction(intent: TxIntent, *, base_currency: str, counter_currency: str) -> bool:
    return {str(intent.in_cur), str(intent.out_cur)} == {str(counter_currency), str(base_currency)}


def _direction_label(
    intent: TxIntent,
    *,
    base_currency: str,
    base_label: str,
    counter_currency: str,
    counter_label: str,
) -> str:
    left = counter_label if str(intent.in_cur) == str(counter_currency) else base_label
    right = counter_label if str(intent.out_cur) == str(counter_currency) else base_label
    return f"{left}->{right}"


def _classify_intent(intent: TxIntent, *, base_currency: str, counter_currency: str) -> tuple[bool, str]:
    if not _is_target_pair_direction(
        intent,
        base_currency=base_currency,
        counter_currency=counter_currency,
    ):
        return False, "non_pair_direct_intent"

    tx_type = str(intent.transaction_type or "").upper()
    if tx_type == "OFFERCREATE":
        return True, "offercreate_pair_direct"
    if tx_type == "PAYMENT":
        if not intent.payment_paths_present:
            return True, "payment_pair_direct_no_paths"
        if intent.payment_paths_has_external_asset:
            return False, "payment_paths_external_asset"
        if int(intent.payment_paths_count) != 1:
            return False, "payment_paths_not_single"
        return True, "payment_pair_direct_single_pair_local_path"
    raise RuntimeError(f"unsupported transaction type after strict intent parse: {tx_type or 'unknown'}")


def _result_utc_date(result: dict[str, Any]) -> str | None:
    ripple_seconds = result.get("date")
    if ripple_seconds is None:
        return None
    try:
        dt = RIPPLE_EPOCH + timedelta(seconds=int(ripple_seconds))
    except Exception as exc:
        raise RuntimeError(f"invalid metadata date field: {ripple_seconds}") from exc
    return dt.date().isoformat()


def load_metadata_map(*, metadata_ndjson: Path, target_tx_hashes: set[str]) -> dict[str, tuple[dict[str, Any], TxIntent]]:
    out: dict[str, tuple[dict[str, Any], TxIntent]] = {}
    with metadata_ndjson.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                obj = json.loads(payload)
            except Exception as exc:
                raise RuntimeError(
                    f"metadata parse failed at {metadata_ndjson}:{line_no}: {type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                continue
            tx_hash = extract_tx_hash_from_metadata_obj(obj)
            if tx_hash not in target_tx_hashes:
                continue
            if tx_hash in out:
                raise RuntimeError(f"duplicate metadata rows found for tx={tx_hash}")
            result = obj.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(
                    f"metadata row missing result object for tx={tx_hash} at {metadata_ndjson}:{line_no}"
                )
            intent = build_intent_from_metadata_result(result)
            out[tx_hash] = (result, intent)
    missing = sorted(target_tx_hashes - set(out))
    if missing:
        sample = ", ".join(missing[:5])
        raise RuntimeError(f"missing metadata rows for {len(missing)} txs; sample=[{sample}]")
    return out


def _build_metadata_entry(
    *,
    obj: dict[str, Any],
    tx_hash: str,
    source: str,
) -> tuple[dict[str, Any], TxIntent]:
    result = obj.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"metadata row missing result object for tx={tx_hash} at {source}")
    intent = build_intent_from_metadata_result(result)
    return result, intent


def _parse_target_lines_by_index(
    *,
    shard_path: Path,
    ledger_index: int,
    txi_to_hashes: dict[int, set[str]],
) -> tuple[dict[str, tuple[dict[str, Any], TxIntent]], set[str]]:
    found: dict[str, tuple[dict[str, Any], TxIntent]] = {}
    wanted_indices = set(txi_to_hashes)
    max_index = max(wanted_indices) if wanted_indices else -1

    with shard_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if line_idx > max_index:
                break
            if line_idx not in wanted_indices:
                continue
            payload = line.strip()
            if not payload:
                continue
            obj = json.loads(payload)
            if not isinstance(obj, dict):
                continue
            tx_hash = extract_tx_hash_from_metadata_obj(obj)
            if tx_hash in txi_to_hashes[line_idx]:
                found[tx_hash] = _build_metadata_entry(
                    obj=obj,
                    tx_hash=tx_hash,
                    source=f"{shard_path}:line_index={line_idx}",
                )

    expected = {h for hashes in txi_to_hashes.values() for h in hashes}
    return found, expected - set(found)


def _parse_target_lines_by_hash(
    *,
    shard_path: Path,
    wanted_hashes: set[str],
) -> dict[str, tuple[dict[str, Any], TxIntent]]:
    found: dict[str, tuple[dict[str, Any], TxIntent]] = {}
    with shard_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            obj = json.loads(payload)
            if not isinstance(obj, dict):
                continue
            tx_hash = extract_tx_hash_from_metadata_obj(obj)
            if tx_hash not in wanted_hashes:
                continue
            found[tx_hash] = _build_metadata_entry(
                obj=obj,
                tx_hash=tx_hash,
                source=f"{shard_path}:{line_no}",
            )
            if len(found) == len(wanted_hashes):
                break
    return found


def load_metadata_map_from_raw_ledgers(
    *,
    raw_dir: Path,
    tx_df: pd.DataFrame,
) -> dict[str, tuple[dict[str, Any], TxIntent]]:
    by_ledger: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in tx_df[["tx_hash", "ledger_index", "transaction_index"]].to_dict(orient="records"):
        by_ledger[int(row["ledger_index"])][int(row["transaction_index"])].add(
            str(row["tx_hash"]).upper()
        )

    out: dict[str, tuple[dict[str, Any], TxIntent]] = {}
    fallback_ledgers = 0
    for i, (ledger_index, txi_to_hashes) in enumerate(sorted(by_ledger.items()), start=1):
        shard_path = raw_dir / f"ledger_{ledger_index}.ndjson"
        if not shard_path.exists():
            raise FileNotFoundError(f"missing raw ledger metadata shard: {shard_path}")

        found, missing = _parse_target_lines_by_index(
            shard_path=shard_path,
            ledger_index=ledger_index,
            txi_to_hashes=txi_to_hashes,
        )
        if missing:
            fallback_ledgers += 1
            wanted = {h for hashes in txi_to_hashes.values() for h in hashes}
            found = _parse_target_lines_by_hash(shard_path=shard_path, wanted_hashes=wanted)
            missing = wanted - set(found)
            if missing:
                sample = ", ".join(sorted(missing)[:5])
                raise RuntimeError(
                    f"missing metadata rows in shard for ledger={ledger_index} "
                    f"count={len(missing)} sample=[{sample}]"
                )

        duplicate = set(out).intersection(found)
        if duplicate:
            sample = ", ".join(sorted(duplicate)[:5])
            raise RuntimeError(f"duplicate metadata rows found in raw ledgers: {sample}")
        out.update(found)

        if i % 5000 == 0 or i == len(by_ledger):
            print(
                f"metadata_shards {i}/{len(by_ledger)} "
                f"target_rows={len(out)} fallback_ledgers={fallback_ledgers}",
                flush=True,
            )

    return out


def main() -> None:
    args = parse_args()
    tx_path = Path(args.tx_sequence)
    metadata_path = Path(args.metadata_ndjson)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    utc_date_filter = None if args.utc_date is None else str(args.utc_date)
    if utc_date_filter is not None:
        try:
            datetime.strptime(utc_date_filter, "%Y-%m-%d")
        except ValueError as exc:
            raise RuntimeError(f"invalid --utc-date: {utc_date_filter}") from exc

    tx_df = load_tx_sequence(tx_path)
    target_tx_hashes = set(tx_df["tx_hash"].tolist())
    raw_dir = metadata_path.parent / "raw_ledgers"
    if raw_dir.exists():
        metadata_map = load_metadata_map_from_raw_ledgers(raw_dir=raw_dir, tx_df=tx_df)
    else:
        metadata_map = load_metadata_map(metadata_ndjson=metadata_path, target_tx_hashes=target_tx_hashes)

    selected_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}

    tx_df = tx_df.sort_values(
        ["ledger_index", "transaction_index", "tx_hash"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    for row in tx_df.to_dict(orient="records"):
        tx_hash = str(row["tx_hash"]).upper()
        result, intent = metadata_map[tx_hash]
        utc_date = _result_utc_date(result)
        if utc_date_filter is not None and utc_date != utc_date_filter:
            continue
        include, reason = _classify_intent(
            intent,
            base_currency=str(args.base_currency),
            counter_currency=str(args.counter_currency),
        )
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        base = {
            "tx_hash": tx_hash,
            "ledger_index": int(row["ledger_index"]),
            "transaction_index": int(row["transaction_index"]),
            "transaction_type": str(intent.transaction_type or "").upper(),
            "direction": _direction_label(
                intent,
                base_currency=str(args.base_currency),
                base_label=str(args.base_label),
                counter_currency=str(args.counter_currency),
                counter_label=str(args.counter_label),
            ),
            "selection_reason": reason,
            "payment_paths_present": bool(intent.payment_paths_present),
            "payment_paths_count": int(intent.payment_paths_count),
            "payment_paths_has_external_asset": bool(intent.payment_paths_has_external_asset),
            "utc_date": "" if utc_date is None else utc_date,
            "metadata_ledger_index": int(result.get("ledger_index")),
            "metadata_transaction_index": int((result.get("meta") or {}).get("TransactionIndex")),
        }
        if include:
            selected_rows.append(base)
        else:
            excluded_rows.append(base)

    selected_df = pd.DataFrame(selected_rows)
    excluded_df = pd.DataFrame(excluded_rows)
    if selected_df.empty:
        raise RuntimeError("strict direct target selection produced no rows")

    selected_df = selected_df.sort_values(
        ["ledger_index", "transaction_index", "tx_hash"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    excluded_df = excluded_df.sort_values(
        ["ledger_index", "transaction_index", "tx_hash"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    ledger_list = selected_df["ledger_index"].drop_duplicates().astype("int64").tolist()

    target_parquet = outdir / "strict_target_tx.parquet"
    target_csv = outdir / "strict_target_tx.csv"
    excluded_csv = outdir / "strict_target_tx_excluded.csv"
    ledger_list_txt = outdir / "ledger_list.txt"
    required_tx_csv = outdir / "required_tx.csv"
    stats_json = outdir / "strict_target_tx_stats.json"

    selected_df.to_parquet(target_parquet, index=False)
    selected_df.to_csv(target_csv, index=False)
    excluded_df.to_csv(excluded_csv, index=False)
    ledger_list_txt.write_text("".join(f"{x}\n" for x in ledger_list), encoding="utf-8")
    selected_df[["ledger_index", "transaction_index", "tx_hash"]].rename(
        columns={"tx_hash": "transaction_hash"}
    ).to_csv(required_tx_csv, index=False)

    stats = {
        "rule_name": RULE_NAME,
        "rule_summary": {
            "pair_direction_only": True,
            "offercreate_included": True,
            "payment_no_paths_included": True,
            "payment_single_pair_local_path_included": True,
            "payment_external_asset_paths_excluded": True,
            "payment_multi_path_excluded": True,
        },
        "counts": {
            "rows_input": int(len(tx_df)),
            "rows_selected": int(len(selected_df)),
            "rows_excluded": int(len(excluded_df)),
            "unique_ledgers_selected": int(len(ledger_list)),
            "ledger_min_selected": int(min(ledger_list)),
            "ledger_max_selected": int(max(ledger_list)),
        },
        "filters": {
            "utc_date": utc_date_filter,
            "base_currency": str(args.base_currency),
            "counter_currency": str(args.counter_currency),
        },
        "selection_reason_counts": reason_counts,
        "outputs": {
            "strict_target_tx_parquet": str(target_parquet),
            "strict_target_tx_csv": str(target_csv),
            "strict_target_tx_excluded_csv": str(excluded_csv),
            "ledger_list_txt": str(ledger_list_txt),
            "required_tx_csv": str(required_tx_csv),
        },
    }
    stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"rule_name={RULE_NAME}")
    print(f"rows_input={len(tx_df)}")
    print(f"rows_selected={len(selected_df)}")
    print(f"rows_excluded={len(excluded_df)}")
    print(f"unique_ledgers_selected={len(ledger_list)}")
    print(f"ledger_min_selected={min(ledger_list)}")
    print(f"ledger_max_selected={max(ledger_list)}")
    print(f"target_parquet={target_parquet}")
    print(f"ledger_list={ledger_list_txt}")
    print(f"stats_json={stats_json}")


if __name__ == "__main__":
    main()
