#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise raw pair activity and strict-direct target composition."
    )
    parser.add_argument("--pair-config", required=True, help="Resolved paper pair config JSON")
    parser.add_argument("--input-root", default="artifacts/inputs")
    parser.add_argument("--target-root", default="artifacts/targets")
    parser.add_argument("--export-root", default="artifacts/exports")
    parser.add_argument("--target-scope", default="strict_direct")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _dec(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _range(values: list[Decimal]) -> dict[str, str | None]:
    if not values:
        return {"min": None, "max": None}
    return {"min": min(values).to_eng_string(), "max": max(values).to_eng_string()}


def _load_pair_config(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("pairs") or [])


def _composition(df: pd.DataFrame) -> dict[str, int]:
    has_amm = pd.to_numeric(df["has_amm"], errors="coerce").fillna(0).astype(int) > 0
    has_clob = pd.to_numeric(df["has_clob"], errors="coerce").fillna(0).astype(int) > 0
    return {
        "transactions": int(len(df)),
        "amm": int(has_amm.sum()),
        "clob": int(has_clob.sum()),
        "amm_clob": int((has_amm & has_clob).sum()),
    }


def _target_composition(input_dir: Path, target_dir: Path) -> dict[str, int]:
    full_tx = pd.read_csv(input_dir / "full_tx_sequence.csv")
    target_tx = pd.read_csv(target_dir / "strict_target_tx.csv")
    tx_col = "tx_hash"
    full_tx[tx_col] = full_tx[tx_col].astype(str).str.upper()
    target_tx[tx_col] = target_tx[tx_col].astype(str).str.upper()
    merged = target_tx[[tx_col]].merge(full_tx[[tx_col, "has_amm", "has_clob"]], on=tx_col, how="left")
    if merged[["has_amm", "has_clob"]].isna().any().any():
        raise RuntimeError(f"target rows missing from full transaction sequence: {target_dir}")
    return _composition(merged)


def _raw_composition(input_dir: Path) -> dict[str, int]:
    stats_path = input_dir / "full_tx_prebook_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        return {
            "transactions": int(stats["tx_unique"]),
            "amm": int(stats["tx_has_amm"]),
            "clob": int(stats["tx_has_clob"]),
            "amm_clob": int(stats["tx_has_both"]),
        }
    full_tx = pd.read_csv(input_dir / "full_tx_sequence.csv")
    return _composition(full_tx)


def _amm_summary(export_dir: Path) -> dict[str, Any]:
    amm_path = export_dir / "amm_swaps"
    if not amm_path.exists():
        return {"account": None, "xrp_reserve": {"min": None, "max": None}, "token_reserve": {"min": None, "max": None}}
    df = pd.read_parquet(amm_path)
    required = {
        "amm_account",
        "amm_asset_currency",
        "amm_asset_balance_before",
        "amm_asset2_currency",
        "amm_asset2_balance_before",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"AMM parquet missing setup-summary columns {missing}: {amm_path}")
    accounts = sorted(str(value) for value in df["amm_account"].dropna().unique())
    xrp_values: list[Decimal] = []
    token_values: list[Decimal] = []
    for row in df[
        [
            "amm_asset_currency",
            "amm_asset_balance_before",
            "amm_asset2_currency",
            "amm_asset2_balance_before",
        ]
    ].to_dict(orient="records"):
        for currency_key, balance_key in (
            ("amm_asset_currency", "amm_asset_balance_before"),
            ("amm_asset2_currency", "amm_asset2_balance_before"),
        ):
            value = _dec(row.get(balance_key))
            if value is None:
                continue
            if str(row.get(currency_key) or "") == "XRP":
                xrp_values.append(value)
            else:
                token_values.append(value)
    return {
        "account": accounts[0] if len(accounts) == 1 else accounts,
        "xrp_reserve": _range(xrp_values),
        "token_reserve": _range(token_values),
    }


def _summarise_pair(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    pair = str(row["pair"])
    window = str(row["window"])
    input_dir = Path(args.input_root) / pair / window
    target_dir = Path(args.target_root) / pair / window / args.target_scope
    export_dir = Path(args.export_root) / pair / window
    return {
        "pair": pair,
        "pair_label": str(row.get("pair_label") or pair),
        "window": window,
        "raw_pair_activity": _raw_composition(input_dir),
        "direct_target_set": _target_composition(input_dir, target_dir),
        "amm": _amm_summary(export_dir),
    }


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Pair Setup Summary",
        "",
        "| Pair | Raw tx | Raw AMM | Raw CLOB | Raw AMM+CLOB | Target tx | Target AMM | Target CLOB | Target AMM+CLOB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        raw = row["raw_pair_activity"]
        target = row["direct_target_set"]
        lines.append(
            f"| {row['pair_label'].replace('_', '/')} | {raw['transactions']:,} | {raw['amm']:,} | "
            f"{raw['clob']:,} | {raw['amm_clob']:,} | {target['transactions']:,} | "
            f"{target['amm']:,} | {target['clob']:,} | {target['amm_clob']:,} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    rows = [_summarise_pair(args, row) for row in _load_pair_config(Path(args.pair_config))]
    payload = {
        "pair_config": str(args.pair_config),
        "input_root": str(args.input_root),
        "target_root": str(args.target_root),
        "export_root": str(args.export_root),
        "pairs": rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pair_setup_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(output_dir / "pair_setup_summary.md", rows)
    print(json.dumps({"output_json": str(output_dir / "pair_setup_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
