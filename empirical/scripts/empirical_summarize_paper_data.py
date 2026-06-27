#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Any


getcontext().prec = 80

PAYMENT_FLAG_PARTIAL = 0x00020000
PAYMENT_FLAG_NO_RIPPLE_DIRECT = 0x00010000
PAYMENT_FLAG_LIMIT_QUALITY = 0x00040000
OFFER_FLAG_PASSIVE = 0x00010000
OFFER_FLAG_IOC = 0x00020000
OFFER_FLAG_FOK = 0x00040000
OFFER_FLAG_SELL = 0x00080000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compact paper-data summaries from same-fill optimisation results."
    )
    parser.add_argument("--results-root", required=True, help="Root containing PAIR_LABEL/results.ndjson")
    parser.add_argument("--pair-config", required=True, help="Resolved paper pair config JSON")
    parser.add_argument(
        "--fit-input-root",
        default="artifacts/fit_inputs",
        help="Root containing pair/window/strict_direct_targets/two_week_isolated datasets",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dust-quantile", type=float, default=0.05)
    return parser.parse_args()


def _dec(value: Any) -> Decimal:
    if value is None:
        return Decimal(0)
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal(0)


def _positive(value: Any) -> Decimal:
    value_dec = _dec(value)
    return value_dec if value_dec > 0 else Decimal(0)


def _side(measurement: str) -> str | None:
    if measurement.endswith("same_input"):
        return "same_input"
    if measurement.endswith("same_output"):
        return "same_output"
    return None


def _is_strict(row: dict[str, Any]) -> bool:
    fit = row.get("baseline_fit") or {}
    return bool(fit.get("strict_exact_match")) or fit.get("status") == "strict_exact_match"


def _fixed_size(row: dict[str, Any], side: str) -> Decimal:
    baseline = row.get("baseline") or {}
    return _positive(baseline.get("input") if side == "same_input" else baseline.get("output"))


def _weighted_components(row: dict[str, Any]) -> tuple[Decimal, Decimal]:
    measurement = str(row.get("layer1_measurement") or "")
    baseline = row.get("baseline") or {}
    best = row.get("best") or {}
    comparison = row.get("counterfactual_comparison") or {}
    if measurement.endswith("same_input"):
        numerator = _positive(comparison.get("delta_output")) or _positive(best.get("improvement"))
        return numerator, _positive(baseline.get("output"))
    if measurement.endswith("same_output"):
        numerator = _positive(comparison.get("input_saved"))
        if numerator == 0:
            numerator = abs(_dec(comparison.get("delta_input")))
        if numerator == 0:
            numerator = _positive(best.get("improvement"))
        return numerator, _positive(baseline.get("input"))
    return Decimal(0), Decimal(0)


def _improvement_bps(row: dict[str, Any]) -> Decimal:
    best = row.get("best") or {}
    if best.get("improvement_bps") is not None:
        return _dec(best.get("improvement_bps"))
    comparison = row.get("counterfactual_comparison") or {}
    return _dec(comparison.get("delta_bps"))


def _percentile(values: list[Decimal], q: float) -> Decimal:
    if not values:
        return Decimal(0)
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = Decimal(str(pos - lo))
    return ordered[lo] * (Decimal(1) - weight) + ordered[hi] * weight


def _nearest_rank(values: list[Decimal], percentile: int) -> Decimal:
    if not values:
        return Decimal(0)
    ordered = sorted(values)
    rank = max(1, (len(ordered) * int(percentile) + 99) // 100)
    return ordered[min(len(ordered) - 1, rank - 1)]


def _dist(values: list[Decimal]) -> dict[str, str | int]:
    positives = [value for value in values if value > 0]
    return {
        "count": len(values),
        "positive_count": len(positives),
        "positive_median_bps": _nearest_rank(positives, 50).to_eng_string() if positives else "0",
        "positive_p95_bps": _nearest_rank(positives, 95).to_eng_string() if positives else "0",
        "positive_max_bps": max(positives).to_eng_string() if positives else "0",
    }


@dataclass
class Stat:
    checks: int = 0
    improved: int = 0
    weighted_num: Decimal = Decimal(0)
    weighted_den: Decimal = Decimal(0)
    bps: list[Decimal] = field(default_factory=list)

    def add(self, row: dict[str, Any]) -> None:
        self.checks += 1
        bps = _improvement_bps(row)
        num, den = _weighted_components(row)
        self.weighted_num += max(num, Decimal(0))
        self.weighted_den += den
        self.bps.append(bps)
        if bool(row.get("found_improvement")) and bps > 0:
            self.improved += 1

    def as_dict(self) -> dict[str, Any]:
        weighted = self.weighted_num / self.weighted_den * Decimal(10000) if self.weighted_den else Decimal(0)
        return {
            "checks": self.checks,
            "improved": self.improved,
            "improved_rate": (Decimal(self.improved) / Decimal(self.checks)).to_eng_string() if self.checks else "0",
            "weighted_bps": weighted.to_eng_string(),
            "weighted_numerator": self.weighted_num.to_eng_string(),
            "weighted_denominator": self.weighted_den.to_eng_string(),
            "bps_distribution": _dist(self.bps),
        }


@dataclass
class TxStat:
    tx: int = 0
    improved: int = 0
    weighted_num: Decimal = Decimal(0)
    weighted_den: Decimal = Decimal(0)
    bps: list[Decimal] = field(default_factory=list)

    def add_tx(self, *, improved: bool, weighted_num: Decimal, weighted_den: Decimal, max_bps: Decimal) -> None:
        self.tx += 1
        self.weighted_num += weighted_num
        self.weighted_den += weighted_den
        if improved:
            self.improved += 1
            self.bps.append(max_bps)

    def as_dict(self) -> dict[str, Any]:
        weighted = self.weighted_num / self.weighted_den * Decimal(10000) if self.weighted_den else Decimal(0)
        return {
            "transactions": self.tx,
            "improved_transactions": self.improved,
            "improved_rate": (Decimal(self.improved) / Decimal(self.tx)).to_eng_string() if self.tx else "0",
            "weighted_bps": weighted.to_eng_string(),
            "weighted_numerator": self.weighted_num.to_eng_string(),
            "weighted_denominator": self.weighted_den.to_eng_string(),
            "positive_bps_distribution": _dist(self.bps),
        }


def _venue_flags(payload: dict[str, Any]) -> tuple[bool, bool]:
    amm = _positive(payload.get("amm_input")) > 0 or _positive(payload.get("amm_output")) > 0
    clob = _positive(payload.get("clob_input")) > 0 or _positive(payload.get("clob_output")) > 0
    return amm, clob


def _venue_name(payload: dict[str, Any]) -> str:
    amm, clob = _venue_flags(payload)
    if amm and clob:
        return "AMM+CLOB"
    if amm:
        return "AMM-only"
    if clob:
        return "CLOB-only"
    return "No filled venue"


def _mechanism(row: dict[str, Any]) -> tuple[Decimal, Decimal]:
    decomp = row.get("gain_decomposition") or {}
    return (
        _positive(decomp.get("allocation_optimization_gain")),
        _positive(decomp.get("amm_multi_slice_fee_compounding_gain")),
    )


def _price_gap_bps(row: dict[str, Any]) -> Decimal | None:
    diagnostic = row.get("best_point_diagnostic") or {}
    matched = diagnostic.get("matched_interval") or {}
    side = str(diagnostic.get("side") or "")
    if side == "input":
        current_raw = matched.get("clob_out_per_input")
        current = Decimal(1) / _dec(current_raw) if _dec(current_raw) > 0 else Decimal(0)
    else:
        current = _dec(matched.get("clob_unit_cost"))
    if current <= 0:
        return None
    point_units = diagnostic.get("point_units")
    try:
        point_units_i = int(point_units)
    except (TypeError, ValueError):
        point_units_i = None
    matched_lo = _int_or_none(matched.get("interval_lo"))
    matched_hi = _int_or_none(matched.get("interval_hi"))
    gaps: list[Decimal] = []
    for interval in diagnostic.get("intervals_sample") or []:
        lo = _int_or_none(interval.get("interval_lo"))
        hi = _int_or_none(interval.get("interval_hi"))
        adjacent = (matched_lo is not None and hi == matched_lo) or (matched_hi is not None and lo == matched_hi)
        if point_units_i is not None and (lo == point_units_i or hi == point_units_i):
            adjacent = True
        if not adjacent:
            continue
        if side == "input":
            raw = _dec(interval.get("clob_out_per_input"))
            price = Decimal(1) / raw if raw > 0 else Decimal(0)
        else:
            price = _dec(interval.get("clob_unit_cost"))
        if price > 0 and price != current:
            gaps.append(abs(price / current - Decimal(1)) * Decimal(10000))
    return min(gaps) if gaps else None


def _amm_movement_bps(row: dict[str, Any]) -> Decimal | None:
    diagnostic = row.get("best_point_diagnostic") or {}
    matched = diagnostic.get("matched_interval") or {}
    side = str(diagnostic.get("side") or "")
    lo = _dec(matched.get("amm_marginal_lo"))
    hi = _dec(matched.get("amm_marginal_hi"))
    if side == "input":
        lo = Decimal(1) / lo if lo > 0 else Decimal(0)
        hi = Decimal(1) / hi if hi > 0 else Decimal(0)
    if lo <= 0 or hi <= 0:
        return None
    return abs(hi / lo - Decimal(1)) * Decimal(10000)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bucket_clob(value: Decimal | None) -> str:
    if value is None:
        return "none"
    if value <= Decimal("0.01"):
        return "le_0_01"
    if value <= Decimal("0.1"):
        return "0_01_0_1"
    if value <= Decimal("1"):
        return "0_1_1"
    if value <= Decimal("10"):
        return "1_10"
    return "gt_10"


def _bucket_amm(value: Decimal | None) -> str:
    if value is None:
        return "none"
    if value <= Decimal("0.001"):
        return "le_0_001"
    if value <= Decimal("0.01"):
        return "0_001_0_01"
    if value <= Decimal("0.1"):
        return "0_01_0_1"
    return "gt_0_1"


def _amount_kind(value: Any) -> str:
    if isinstance(value, str):
        return "XRP"
    if isinstance(value, dict):
        return "XRP" if value.get("currency") == "XRP" else "IOU"
    return "unknown"


def _field_groups(tx: dict[str, Any]) -> list[tuple[str, str]]:
    tx_type = str(tx.get("TransactionType") or "").upper()
    flags = int(tx.get("Flags") or 0)
    groups: list[tuple[str, str]] = []
    if tx_type == "PAYMENT":
        deliver = tx.get("DeliverMax", tx.get("Amount"))
        sendmax = tx.get("SendMax")
        paths = tx.get("Paths") or []
        groups.extend(
            [
                ("Payment", "Payment total"),
                ("Payment", "tfPartialPayment" if flags & PAYMENT_FLAG_PARTIAL else "Exact delivery"),
                ("Payment", f"DeliverMax = {_amount_kind(deliver)}"),
                ("Payment", f"SendMax = {_amount_kind(sendmax)}"),
                ("Payment", "DeliverMin present" if tx.get("DeliverMin") is not None else "No DeliverMin"),
                ("Payment", "Paths present" if paths else "No Paths"),
                ("Payment", "tfLimitQuality" if flags & PAYMENT_FLAG_LIMIT_QUALITY else "No tfLimitQuality"),
                ("Payment", "tfNoRippleDirect" if flags & PAYMENT_FLAG_NO_RIPPLE_DIRECT else "No tfNoRippleDirect"),
            ]
        )
    elif tx_type == "OFFERCREATE":
        groups.extend(
            [
                ("OfferCreate", "OfferCreate total"),
                ("OfferCreate", "tfSell" if flags & OFFER_FLAG_SELL else "No tfSell"),
                ("OfferCreate", "tfFillOrKill" if flags & OFFER_FLAG_FOK else "No tfFillOrKill"),
                ("OfferCreate", "tfImmediateOrCancel" if flags & OFFER_FLAG_IOC else "No tfImmediateOrCancel"),
                ("OfferCreate", "tfPassive" if flags & OFFER_FLAG_PASSIVE else "No tfPassive"),
                ("OfferCreate", f"TakerGets = {_amount_kind(tx.get('TakerGets'))}"),
                ("OfferCreate", "OfferSequence" if tx.get("OfferSequence") is not None else "No OfferSequence"),
                ("OfferCreate", "Expiration" if tx.get("Expiration") is not None else "No Expiration"),
            ]
        )
    return groups


def _fast_tx_hash(line: str) -> str | None:
    pos = line.find('"tx_hash"')
    if pos < 0:
        return None
    colon = line.find(":", pos)
    quote = line.find('"', colon + 1)
    end = line.find('"', quote + 1)
    if colon < 0 or quote < 0 or end < 0:
        return None
    return line[quote + 1 : end]


def _metadata_for_hashes(path: Path, tx_hashes: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not tx_hashes:
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tx_hash = _fast_tx_hash(line)
            if tx_hash not in tx_hashes:
                continue
            obj = json.loads(line)
            result = obj.get("result") if isinstance(obj.get("result"), dict) else obj
            actual_hash = str(result.get("hash") or tx_hash)
            out[actual_hash] = result
            if len(out) == len(tx_hashes):
                break
    missing = tx_hashes - set(out)
    if missing:
        raise RuntimeError(f"metadata missing for {len(missing)} txs from {path}")
    return out


def _pair_rows(results_root: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(results_root.iterdir()):
        result_path = path / "results.ndjson"
        if not result_path.exists():
            continue
        rows: list[dict[str, Any]] = []
        with result_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        out[path.name] = rows
    return out


def _load_pair_config(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["pair_label"]): row for row in payload.get("pairs") or []}


def _thresholds(pair_rows: dict[str, list[dict[str, Any]]], quantile: float) -> dict[tuple[str, str], Decimal]:
    sizes: dict[tuple[str, str], list[Decimal]] = defaultdict(list)
    for pair, rows in pair_rows.items():
        for row in rows:
            if not _is_strict(row):
                continue
            side = _side(str(row.get("layer1_measurement") or ""))
            if side:
                size = _fixed_size(row, side)
                if size > 0:
                    sizes[(pair, side)].append(size)
    return {key: _percentile(values, quantile) for key, values in sizes.items()}


def _analyse(pair_rows: dict[str, list[dict[str, Any]]], thresholds: dict[tuple[str, str], Decimal]) -> dict[str, Any]:
    materiality: dict[str, Stat] = defaultdict(Stat)
    materiality_trimmed: dict[str, Stat] = defaultdict(Stat)
    venue_transition: dict[str, Stat] = defaultdict(Stat)
    clob_gap: dict[str, Stat] = defaultdict(Stat)
    amm_movement: dict[str, Stat] = defaultdict(Stat)
    pair_stats: dict[str, Stat] = defaultdict(Stat)
    mechanism = {
        "allocation_positive_checks": 0,
        "amm_slice_positive_checks": 0,
        "both_positive_checks": 0,
    }
    metadata_needed: dict[str, set[str]] = defaultdict(set)

    for pair, rows in pair_rows.items():
        for row in rows:
            if not _is_strict(row):
                continue
            side = _side(str(row.get("layer1_measurement") or ""))
            if side is None:
                continue
            size = _fixed_size(row, side)
            if size <= 0:
                continue
            materiality[side].add(row)
            pair_stats[pair].add(row)
            if size > thresholds.get((pair, side), Decimal(0)):
                materiality_trimmed[side].add(row)
                venue_transition[f"{_venue_name(row.get('baseline') or {})}->{_venue_name(row.get('best') or {})}"].add(row)
                clob_gap[_bucket_clob(_price_gap_bps(row))].add(row)
                amm_movement[_bucket_amm(_amm_movement_bps(row))].add(row)
                allocation, amm_slice = _mechanism(row)
                mechanism["allocation_positive_checks"] += int(allocation > 0)
                mechanism["amm_slice_positive_checks"] += int(amm_slice > 0)
                mechanism["both_positive_checks"] += int(allocation > 0 and amm_slice > 0)
                metadata_needed[pair].add(str(row.get("tx_hash") or ""))

    return {
        "materiality_full": {key: stat.as_dict() for key, stat in sorted(materiality.items())},
        "materiality_dust_filtered": {key: stat.as_dict() for key, stat in sorted(materiality_trimmed.items())},
        "pair_materiality": {key: stat.as_dict() for key, stat in sorted(pair_stats.items())},
        "route_transition": {key: stat.as_dict() for key, stat in sorted(venue_transition.items())},
        "local_clob_gap": {key: stat.as_dict() for key, stat in sorted(clob_gap.items())},
        "local_amm_movement": {key: stat.as_dict() for key, stat in sorted(amm_movement.items())},
        "mechanism_counts": mechanism,
        "metadata_needed": {key: sorted(value) for key, value in metadata_needed.items()},
    }


def _analyse_submitted_fields(
    *,
    pair_rows: dict[str, list[dict[str, Any]]],
    pair_config: dict[str, dict[str, Any]],
    fit_input_root: Path,
    thresholds: dict[tuple[str, str], Decimal],
) -> dict[str, Any]:
    field_stats: dict[str, TxStat] = defaultdict(TxStat)
    for pair, rows in pair_rows.items():
        pair_meta = pair_config[pair]
        dataset_root = fit_input_root / pair_meta["pair"] / pair_meta["window"] / "strict_direct_targets" / "two_week_isolated"
        tx_hashes: set[str] = set()
        row_by_tx: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if not _is_strict(row):
                continue
            side = _side(str(row.get("layer1_measurement") or ""))
            if side is None:
                continue
            if _fixed_size(row, side) <= thresholds.get((pair, side), Decimal(0)):
                continue
            tx_hash = str(row.get("tx_hash") or "")
            tx_hashes.add(tx_hash)
            row_by_tx[tx_hash].append(row)
        metadata = _metadata_for_hashes(dataset_root / "tx_metadata_full_merged.ndjson", tx_hashes)
        for tx_hash, tx_rows in row_by_tx.items():
            groups = _field_groups(metadata[tx_hash])
            if not groups:
                continue
            improved, weighted_num, weighted_den, max_bps = _tx_metrics(tx_rows)
            for tx_type, field_name in groups:
                field_stats[f"{tx_type}|{field_name}"].add_tx(
                    improved=improved,
                    weighted_num=weighted_num,
                    weighted_den=weighted_den,
                    max_bps=max_bps,
                )
    return {key: stat.as_dict() for key, stat in sorted(field_stats.items())}


def _tx_metrics(rows: list[dict[str, Any]]) -> tuple[bool, Decimal, Decimal, Decimal]:
    found = any(bool(row.get("found_improvement")) and _improvement_bps(row) > 0 for row in rows)
    total_num = Decimal(0)
    total_den = Decimal(0)
    max_bps = Decimal(0)
    for row in rows:
        num, den = _weighted_components(row)
        total_num += max(num, Decimal(0))
        total_den += den
        max_bps = max(max_bps, _improvement_bps(row))
    return found, total_num, total_den, max_bps


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Paper Data Summary", ""]
    for section in ("materiality_full", "materiality_dust_filtered", "route_transition", "local_clob_gap", "local_amm_movement"):
        lines.extend([f"## {section}", "", "| key | checks | improved | weighted bps |", "|---|---:|---:|---:|"])
        for key, row in payload["analysis"][section].items():
            lines.append(f"| {key} | {row['checks']} | {row['improved']} | {row['weighted_bps']} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    results_root = Path(args.results_root)
    pair_config = _load_pair_config(Path(args.pair_config))
    pair_rows = _pair_rows(results_root)
    missing = sorted(set(pair_config) - set(pair_rows))
    if missing:
        raise RuntimeError(f"missing optimiser result directories for pairs: {missing}")
    thresholds = _thresholds(pair_rows, float(args.dust_quantile))
    analysis = _analyse(pair_rows, thresholds)
    analysis["submitted_fields"] = _analyse_submitted_fields(
        pair_rows=pair_rows,
        pair_config=pair_config,
        fit_input_root=Path(args.fit_input_root),
        thresholds=thresholds,
    )
    payload = {
        "results_root": str(results_root),
        "pair_config": str(args.pair_config),
        "dust_quantile": args.dust_quantile,
        "dust_thresholds_by_pair_side": {
            f"{pair}|{side}": value.to_eng_string() for (pair, side), value in sorted(thresholds.items())
        },
        "analysis": analysis,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_data_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(output_dir / "paper_data_summary.md", payload)
    print(json.dumps({"output_json": str(output_dir / "paper_data_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
