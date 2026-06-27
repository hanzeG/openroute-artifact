#!/usr/bin/env python3
"""Counterfactual best-price intent optimizer over the current model baseline.

This optimizer supports three one-dimensional intent families:

    fixed-input maximize-output:
        exact input intents, such as OfferCreate + tfSell + tfFillOrKill

    fixed-output minimize-input:
        OfferCreate buy-side fills and non-partial Payment exact-output fills

    bounded partial:
        intents with input/output bounds but no exact endpoint. The optimizer
        first resolves the same primary fill condition used by the baseline
        execution path, then optimizes price for that resolved fill condition.

The optimizer reuses the current rippled-aligned amount math, AMM/CLOB execution
kernels, transfer-rate handling, and pre-state loaders, but it intentionally
changes the routing objective from "match rippled's current execution order" to
"find the best executable price" under the same tx intent and pre-state.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Iterable

getcontext().prec = 60

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

import empirical_fit_single_direct as fit  # noqa: E402
from xrpl_router.amm import amount_add_ripple_number  # noqa: E402
from xrpl_router.book_step import (  # noqa: E402
    _apply_rate,
    _filtered_rows,
    _iteration_violates_limit_quality,
    _normalise_rows,
    _offer_crossing_candidate_floor,
    _offer_stream_rows,
    _prepare_row,
    _remove_rate,
    _should_delete_self_cross_offer,
    _sum_amounts_sorted,
)
from xrpl_router.book_offers import build_clob_segments_from_offers  # noqa: E402
from xrpl_router.core import Amount, INT64_MAX, IOUAmount, Quality, Segment, XRPAmount  # noqa: E402
from xrpl_router.execution_semantics import (  # noqa: E402
    _amount_eq_or_one_quantum_short,
    _amount_lt,
    _amt_from_units,
    _amt_units_floor,
    _cap_input_budget,
    _offer_gateway_sendmax,
    _offer_input_rate_for_bookstep,
    _offer_output_owner_fee_applies,
    _payment_book_output_target,
    _payment_output_endpoint_rate,
    _require_quote_totals,
    _sell_deliver_ceiling,
    _strictly_better_quality_floor,
    execute_single_path_intent,
    run_payment_constrained_quote,
)
from xrpl_router.tx_intent import TxIntent, apply_offer_tick_size_to_intent  # noqa: E402


IMPROVEMENT_NONE = "none"
IMPROVEMENT_ALLOCATION_CHANGED = "allocation_changed"
IMPROVEMENT_AMM_MULTI_SLICE_FEE_COMPOUNDING = "amm_multi_slice_fee_compounding"
IMPROVEMENT_ALLOCATION_AND_AMM_MULTI_SLICE_FEE_COMPOUNDING = "allocation_and_amm_multi_slice_fee_compounding"
IMPROVEMENT_ALLOCATION_OFFSETS_AMM_MULTI_SLICE_FEE_COMPOUNDING_LOSS = (
    "allocation_changed_offsets_amm_multi_slice_fee_compounding_loss"
)
IMPROVEMENT_AMM_MULTI_SLICE_FEE_COMPOUNDING_OFFSETS_ALLOCATION_LOSS = (
    "amm_multi_slice_fee_compounding_offsets_allocation_loss"
)
MODE_FIXED_INPUT_MAX_OUTPUT = "fixed_input_max_output"
MODE_FIXED_OUTPUT_MIN_INPUT = "fixed_output_min_input"
MODE_BOUNDED_PARTIAL = "bounded_partial"
CLAIM_EXACT_BEST_PRICE = "exact_discrete_best_price"
CLAIM_EXACT_INFEASIBLE = "exact_discrete_infeasible"
CLAIM_NOT_CERTIFIED = "not_certified"
STRUCTURAL_CERT_METHOD = "theorem_driven_single_structural_anchor"
ANALYSIS_LAYER_LAYER1_SAME_FILL = "layer1_same_fill"
LAYER1_MEASUREMENT_POLICY_NATURAL = "natural"
LAYER1_MEASUREMENT_POLICY_BOTH = "both"


def _nearest_rank(values: list[Decimal], percentile: int) -> Decimal:
    if not values:
        return Decimal(0)
    rank = max(1, (len(values) * int(percentile) + 99) // 100)
    return values[min(len(values) - 1, rank - 1)]


def _bps_distribution(values: list[Decimal]) -> dict[str, str | int]:
    values = sorted(values)
    if not values:
        return {"count": 0, "p50": "0", "p95": "0", "p99": "0", "max": "0"}
    return {
        "count": len(values),
        "p50": _nearest_rank(values, 50).to_eng_string(),
        "p95": _nearest_rank(values, 95).to_eng_string(),
        "p99": _nearest_rank(values, 99).to_eng_string(),
        "max": values[-1].to_eng_string(),
    }


@dataclass(frozen=True)
class PairDataset:
    manifest_path: Path
    dataset_root: Path
    metadata_ndjson: Path
    tx_prebook_snapshots: Path
    amm_swaps: Path
    amm_fees: Path
    required_tx_csv: Path
    base_currency: str
    base_label: str


@dataclass(frozen=True)
class TxContext:
    tx_hash: str
    ledger_index: int
    intent: TxIntent
    amm: Any
    clob_segments: list[Any]
    baseline_quote: dict[str, Any]
    baseline_fit: dict[str, Any]
    baseline_in: Amount
    baseline_out: Amount
    optimizer_mode: str
    effective_input: Amount
    target_output: Amount | None
    offer_input_rate: IOUAmount | None
    book_out_transfer_rate: IOUAmount
    owner_pays_transfer_fee: bool
    charge_input_transfer_fee: bool
    effective_limit_quality: Quality | None
    execution_mode: str
    fix_amm_v1_1_enabled: bool
    fix_reduced_offers_v2_enabled: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tx-hash", action="append", default=[])
    parser.add_argument(
        "--tx-hash-file",
        action="append",
        default=[],
        help="File containing tx hashes, one per line or in a CSV column named tx_hash/transaction_hash.",
    )
    parser.add_argument("--sample-from-required", action="store_true")
    parser.add_argument(
        "--stream-required",
        action="store_true",
        help=(
            "Stream required tx metadata and tx_prebook snapshots in ledger order. "
            "Use this for full-pair runs to avoid materializing large metadata files in memory."
        ),
    )
    parser.add_argument(
        "--optimizer-mode",
        choices=["all", MODE_FIXED_INPUT_MAX_OUTPUT, MODE_FIXED_OUTPUT_MIN_INPUT, MODE_BOUNDED_PARTIAL],
        default="all",
        help="Restrict target selection to one optimizer family.",
    )
    parser.add_argument(
        "--layer1-measurement-policy",
        choices=[LAYER1_MEASUREMENT_POLICY_NATURAL, LAYER1_MEASUREMENT_POLICY_BOTH],
        default=LAYER1_MEASUREMENT_POLICY_BOTH,
        help=(
            "Controls same-fill measurement emission. "
            "natural emits the same-fill side implied by the submitted intent "
            "for fixed-input/fixed-output txs, while bounded partial emits both sides. "
            "both emits same-input and same-output measurements for fixed-input and "
            "fixed-output txs as well."
        ),
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=0,
        help="Number of required txs to scan when sampling. Use 0 for all required txs.",
    )
    parser.add_argument(
        "--max-txs",
        type=int,
        default=0,
        help="Maximum optimizer targets to run. Use 0 for all selected optimizer targets.",
    )
    parser.add_argument(
        "--max-clob-segments",
        type=int,
        default=0,
        help="Maximum positive CLOB segments to use for structural certification. Use 0 for all segments.",
    )
    parser.add_argument("--rounding-window-drops", type=int, default=8)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes for per-transaction optimization after inputs are loaded.",
    )
    parser.add_argument(
        "--task-chunksize",
        type=int,
        default=8,
        help="ProcessPool chunksize for per-transaction optimization when --jobs > 1.",
    )
    parser.add_argument(
        "--context-batch-size",
        type=int,
        default=2_000,
        help="Number of selected transactions to materialize into optimisation contexts at a time.",
    )
    return parser.parse_args()


def _load_dataset(path: Path) -> PairDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pair = str(raw.get("pair") or "")
    inferred_base_label = pair.split("_", 1)[0].upper() if pair else ""
    inferred_base_currency = None
    ranges = raw.get("issuer_transfer_rate_ranges")
    if isinstance(ranges, dict) and ranges:
        first_key = next(iter(ranges))
        inferred_base_currency = str(first_key).split(":", 1)[0]
    return PairDataset(
        manifest_path=path,
        dataset_root=Path(raw["dataset_root"]),
        metadata_ndjson=Path(raw["metadata_ndjson"]),
        tx_prebook_snapshots=Path(raw["tx_prebook_snapshots"]),
        amm_swaps=Path(raw["amm_swaps"]),
        amm_fees=Path(raw["amm_fees"]),
        required_tx_csv=Path(raw["required_tx_csv"]),
        base_currency=str(raw.get("base_currency") or inferred_base_currency or inferred_base_label),
        base_label=str(raw.get("base_label") or inferred_base_label),
    )


def _read_required_hashes(path: Path, *, limit: int) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tx_hash = fit._normalise_tx_hash(row.get("transaction_hash") or row.get("tx_hash") or "")
            if not tx_hash:
                continue
            out.append(tx_hash)
            if limit > 0 and len(out) >= limit:
                break
    return out


def _read_hash_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if "," in lines[0] and ("tx_hash" in lines[0] or "transaction_hash" in lines[0]):
        out: list[str] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                tx_hash = fit._normalise_tx_hash(row.get("transaction_hash") or row.get("tx_hash") or "")
                if tx_hash:
                    out.append(tx_hash)
        return out
    return [tx_hash for line in lines if (tx_hash := fit._normalise_tx_hash(line.split(",", 1)[0]))]


def _amount_text(amount: Amount | None) -> str | None:
    if amount is None:
        return None
    return format(fit._amount_to_decimal(amount), "f")


def _zero_like(proto: Amount) -> Amount:
    if isinstance(proto, XRPAmount):
        return XRPAmount(0)
    return IOUAmount(0, 0)


def _amount_from_units_like(proto: Amount, units: int) -> Amount:
    units = max(0, int(units))
    if isinstance(proto, XRPAmount):
        return XRPAmount(units)
    mantissa = units
    exponent = -15
    while mantissa > INT64_MAX:
        mantissa //= 10
        exponent += 1
    return IOUAmount(mantissa, exponent)


def _amount_sub(left: Amount, right: Amount) -> Amount:
    if isinstance(left, XRPAmount) and isinstance(right, XRPAmount):
        return left - right
    if isinstance(left, IOUAmount) and isinstance(right, IOUAmount):
        return left - right
    raise RuntimeError("amount domain mismatch")


def _amount_units_exactly_equal(left: Amount, right: Amount) -> bool:
    return _amt_units_floor(left) == _amt_units_floor(right)


def _amount_canonical_quantum_units(amount: Amount) -> int:
    if isinstance(amount, XRPAmount):
        return 1
    if isinstance(amount, IOUAmount):
        return 10 ** max(0, amount.exponent + 15)
    return 1


def _build_executable_clob_prepared_rows(ctx_kwargs: dict[str, Any], clob_segments: list[Any]) -> list[Any]:
    intent = ctx_kwargs["intent"]
    rows = _normalise_rows(clob_segments)
    executable_rows = _offer_stream_rows(
        rows,
        mode=ctx_kwargs["execution_mode"],
        source_account=intent.source_account,
        book_in_transfer_rate=ctx_kwargs["offer_input_rate"],
        book_out_transfer_rate=ctx_kwargs["book_out_transfer_rate"],
        charge_input_transfer_fee=ctx_kwargs["charge_input_transfer_fee"],
        owner_pays_transfer_fee=ctx_kwargs["owner_pays_transfer_fee"],
        enable_self_cross_filter=ctx_kwargs["execution_mode"] == "offer_crossing",
    )
    if ctx_kwargs["execution_mode"] == "offer_crossing":
        executable_rows = _filtered_rows(
            executable_rows,
            _offer_crossing_candidate_floor(ctx_kwargs["effective_limit_quality"]),
        )
    prepared_rows: list[Any] = []
    for row in executable_rows:
        if _should_delete_self_cross_offer(
            row,
            mode=ctx_kwargs["execution_mode"],
            source_account=intent.source_account,
            limit_quality=ctx_kwargs["effective_limit_quality"],
            enable_self_cross_filter=ctx_kwargs["execution_mode"] == "offer_crossing",
        ):
            continue
        prepared = _prepare_row(
            row,
            mode=ctx_kwargs["execution_mode"],
            source_account=intent.source_account,
            book_in_transfer_rate=ctx_kwargs["offer_input_rate"],
            book_out_transfer_rate=ctx_kwargs["book_out_transfer_rate"],
            charge_input_transfer_fee=ctx_kwargs["charge_input_transfer_fee"],
            owner_pays_transfer_fee=ctx_kwargs["owner_pays_transfer_fee"],
            enable_self_cross_filter=ctx_kwargs["execution_mode"] == "offer_crossing",
        )
        if prepared is not None:
            prepared_rows.append(prepared)
    return prepared_rows


def _ctx_kwargs_for_clob_prepare(ctx: TxContext) -> dict[str, Any]:
    return {
        "intent": ctx.intent,
        "target_output": ctx.target_output,
        "offer_input_rate": ctx.offer_input_rate,
        "book_out_transfer_rate": ctx.book_out_transfer_rate,
        "charge_input_transfer_fee": ctx.charge_input_transfer_fee,
        "owner_pays_transfer_fee": ctx.owner_pays_transfer_fee,
        "effective_limit_quality": ctx.effective_limit_quality,
        "execution_mode": ctx.execution_mode,
        "effective_input": ctx.effective_input,
    }


def _executable_clob_segments(ctx: TxContext) -> list[Segment]:
    out: list[Segment] = []
    for prepared in _build_executable_clob_prepared_rows(_ctx_kwargs_for_clob_prepare(ctx), ctx.clob_segments):
        if _amt_units_floor(prepared.step_in) <= 0 or _amt_units_floor(prepared.step_out) <= 0:
            continue
        out.append(
            Segment(
                src="CLOB",
                quality=prepared.base_row.quoted_quality,
                out_max=prepared.step_out,
                in_at_out_max=prepared.step_in,
            )
        )
    return out


def _normalised_intent(sample: Any, tick_size_snapshots: list[tuple[int, dict[str, int | None]]]) -> TxIntent:
    tick_size = fit._effective_offer_tick_size_for_intent(
        intent=sample.intent,
        ledger_index=sample.ledger_index,
        tick_size_snapshots=tick_size_snapshots,
    )
    return apply_offer_tick_size_to_intent(sample.intent, tick_size)


def _optimizer_mode_for_intent(intent: TxIntent) -> str | None:
    tx_type = str(intent.transaction_type or "").upper()
    if tx_type == "OFFERCREATE":
        if intent.offer_taker_gets_amt is None or intent.offer_taker_pays_amt is None:
            return None
        if bool(intent.offer_is_sell):
            if bool(intent.offer_is_fok):
                return MODE_FIXED_INPUT_MAX_OUTPUT
            return MODE_BOUNDED_PARTIAL
        if bool(intent.offer_is_fok):
            return MODE_FIXED_OUTPUT_MIN_INPUT
        return MODE_BOUNDED_PARTIAL
    if tx_type == "PAYMENT":
        if intent.payment_sendmax_amt is None or intent.payment_delivermax_amt is None:
            return None
        if bool(intent.payment_partial_allowed):
            if intent.payment_delivermin_amt is not None and _amount_units_exactly_equal(
                intent.payment_delivermin_amt,
                intent.payment_delivermax_amt,
            ):
                return MODE_FIXED_OUTPUT_MIN_INPUT
            return MODE_BOUNDED_PARTIAL
        return MODE_FIXED_OUTPUT_MIN_INPUT
    return None


def _is_supported_optimizer_intent(intent: TxIntent, *, optimizer_mode_filter: str = "all") -> bool:
    optimizer_mode = _optimizer_mode_for_intent(intent)
    if optimizer_mode is None:
        return False
    return optimizer_mode_filter == "all" or optimizer_mode == optimizer_mode_filter


def _select_optimizer_targets(
    *,
    dataset: PairDataset,
    explicit_hashes: Iterable[str],
    sample_from_required: bool,
    candidate_pool_size: int,
    max_txs: int,
    optimizer_mode_filter: str,
) -> tuple[list[str], dict[str, Any], list[tuple[int, dict[str, int | None]]], dict[str, int]]:
    fit.TARGET_IOU_CURRENCY = dataset.base_currency
    fit.TARGET_IOU_LABEL = dataset.base_label

    explicit_ordered: list[str] = []
    for tx_hash in explicit_hashes:
        normalised = fit._normalise_tx_hash(tx_hash)
        if normalised and normalised not in explicit_ordered:
            explicit_ordered.append(normalised)
    if not explicit_ordered and not sample_from_required:
        raise RuntimeError("provide --tx-hash or --sample-from-required")

    tick_size_snapshots = fit._load_offer_tick_size_snapshots(dataset.metadata_ndjson)

    selected: list[str] = []
    samples: dict[str, Any] = {}
    selection_stats = {
        "explicit_supported_count": 0,
        "context_filtered_count": 0,
    }

    if explicit_ordered:
        explicit_samples = fit._load_metadata_samples(
            metadata_path=dataset.metadata_ndjson,
            target_tx_hashes=set(explicit_ordered),
        )
        samples.update(explicit_samples)
        for tx_hash in explicit_ordered:
            sample = samples[tx_hash]
            intent = _normalised_intent(sample, tick_size_snapshots)
            if _is_supported_optimizer_intent(intent, optimizer_mode_filter=optimizer_mode_filter):
                selection_stats["explicit_supported_count"] += 1
                selected.append(tx_hash)
                if max_txs > 0 and len(selected) >= max_txs:
                    break

    if sample_from_required and (max_txs <= 0 or len(selected) < max_txs):
        required_order = _read_required_hashes(dataset.required_tx_csv, limit=candidate_pool_size)
        required_set = set(required_order)
        already_seen = set(samples)
        with dataset.metadata_ndjson.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    obj = json.loads(payload)
                except Exception as exc:
                    raise RuntimeError(
                        f"failed to parse metadata at {dataset.metadata_ndjson}:{line_no}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                if not isinstance(obj, dict):
                    continue
                tx_hash = fit._normalise_tx_hash(fit.extract_tx_hash_from_metadata_obj(obj))
                if tx_hash not in required_set or tx_hash in already_seen:
                    continue
                result = obj.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError(f"metadata row missing result object for tx={tx_hash}")
                intent = fit.build_intent_from_metadata_result(result)
                if not fit._is_direct_pair(intent.in_cur, intent.out_cur):
                    continue
                ledger_index = result.get("ledger_index")
                tx_index = (result.get("meta") or {}).get("TransactionIndex")
                if ledger_index is None or tx_index is None:
                    raise RuntimeError(f"metadata row missing ledger_index/TransactionIndex for tx={tx_hash}")
                sample = fit.SampleInput(
                    tx_hash=tx_hash,
                    ledger_index=int(ledger_index),
                    transaction_index=int(tx_index),
                    result=result,
                    intent=intent,
                )
                already_seen.add(tx_hash)
                normalised = _normalised_intent(sample, tick_size_snapshots)
                if not _is_supported_optimizer_intent(normalised, optimizer_mode_filter=optimizer_mode_filter):
                    continue
                samples[tx_hash] = sample
                selected.append(tx_hash)
                if max_txs > 0 and len(selected) >= max_txs:
                    break

    for tx_hash in list(selected):
        sample = samples[tx_hash]
        intent = _normalised_intent(sample, tick_size_snapshots)
        if not _is_supported_optimizer_intent(intent, optimizer_mode_filter=optimizer_mode_filter):
            selected.remove(tx_hash)
    if not selected:
        raise RuntimeError("no supported fixed-input/fixed-output optimizer targets found in candidate pool")
    selection_stats["selected_count"] = len(selected)
    return selected, samples, tick_size_snapshots, selection_stats


def _sample_from_metadata_obj(obj: dict[str, Any], *, tx_hash: str) -> Any:
    result = obj.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"metadata row missing result object for tx={tx_hash}")
    intent = fit.build_intent_from_metadata_result(result)
    if not fit._is_direct_pair(intent.in_cur, intent.out_cur):
        raise RuntimeError(f"sample tx is not direct pair scoped: tx={tx_hash}")
    ledger_index = result.get("ledger_index")
    tx_index = (result.get("meta") or {}).get("TransactionIndex")
    if ledger_index is None or tx_index is None:
        raise RuntimeError(f"metadata row missing ledger_index/TransactionIndex for tx={tx_hash}")
    return fit.SampleInput(
        tx_hash=tx_hash,
        ledger_index=int(ledger_index),
        transaction_index=int(tx_index),
        result=result,
        intent=intent,
    )


def _iter_required_metadata_samples_in_order(
    *,
    metadata_path: Path,
    required_order: list[str],
) -> Iterable[tuple[str, Any, dict[str, int]]]:
    required_set = set(required_order)
    buffered: dict[str, Any] = {}
    next_index = 0
    max_buffered = 0
    progress_every = 20_000 if len(required_order) > 10_000 else 0

    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            if progress_every and line_no % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "metadata_scan_lines": line_no,
                            "yielded": next_index,
                            "required": len(required_order),
                            "buffered": len(buffered),
                            "max_buffered": max_buffered,
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            try:
                obj = json.loads(payload)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to parse metadata at {metadata_path}:{line_no}: {type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                continue
            tx_hash = fit._normalise_tx_hash(fit.extract_tx_hash_from_metadata_obj(obj))
            if tx_hash not in required_set:
                continue
            if tx_hash in buffered:
                raise RuntimeError(f"duplicate metadata row for required tx={tx_hash}")
            buffered[tx_hash] = _sample_from_metadata_obj(obj, tx_hash=tx_hash)
            max_buffered = max(max_buffered, len(buffered))

            while next_index < len(required_order):
                expected_hash = required_order[next_index]
                matched = buffered.pop(expected_hash, None)
                if matched is None:
                    break
                next_index += 1
                yield expected_hash, matched, {
                    "metadata_scan_lines": line_no,
                    "metadata_buffered": len(buffered),
                    "metadata_max_buffered": max_buffered,
                }
            if next_index == len(required_order):
                break

    if next_index != len(required_order):
        missing = [tx_hash for tx_hash in required_order[next_index:] if tx_hash not in buffered]
        raise RuntimeError(
            f"missing metadata rows for {len(missing)} required txs after streaming metadata: "
            f"{', '.join(missing[:5])}"
        )


class SnapshotSideStream:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("r", encoding="utf-8")
        self.buffered: dict[tuple[str, str], dict[str, Any]] = {}
        self.line_no = 0
        self.max_buffered = 0

    def close(self) -> None:
        self.handle.close()

    def next_for(self, *, tx_hash: str, side: str) -> dict[str, Any]:
        for line in self.handle:
            self.line_no += 1
            payload = line.strip()
            if not payload:
                continue
            try:
                row = json.loads(payload)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to parse tx_prebook snapshot at {self.path}:{self.line_no}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                continue
            row_hash = fit._normalise_tx_hash(row.get("transaction_hash") or "")
            row_side = str(row.get("side") or "")
            if row_hash == tx_hash and row_side == side:
                return row
        raise RuntimeError(f"missing tx_prebook side row for tx={tx_hash} side={side}")


def _build_contexts(
    *,
    dataset: PairDataset,
    selected_hashes: list[str],
    samples: dict[str, Any],
    tick_size_snapshots: list[tuple[int, dict[str, int | None]]],
    snapshot_payloads_by_hash: dict[str, dict[str, Any]] | None = None,
    require_strict_baseline_fit: bool = False,
) -> list[TxContext]:
    issuer_transfer_rates, issuer_transfer_rate_ranges = fit._load_issuer_transfer_rates(
        str(dataset.manifest_path)
    )
    amendment_feature_flags = fit._load_amendment_feature_flags(str(dataset.manifest_path))
    fix_amm_v1_1_enabled = amendment_feature_flags.get(
        "fixAMMv1_1",
        amendment_feature_flags.get("AMMv1_1", True),
    )
    fix_reduced_offers_v2_enabled = amendment_feature_flags.get("fixReducedOffersV2", True)

    intents: dict[str, TxIntent] = {}
    ordered_targets: list[tuple[str, str]] = []
    for tx_hash in selected_hashes:
        sample = samples[tx_hash]
        intent = _normalised_intent(sample, tick_size_snapshots)
        intents[tx_hash] = intent
        ordered_targets.append((tx_hash, fit._prebook_primary_side_for_output_currency(intent.out_cur)))

    if snapshot_payloads_by_hash is None:
        snapshots = {
            tx_hash: payload
            for tx_hash, payload in fit._iter_snapshot_rows_in_sample_order(
                snapshots_path=dataset.tx_prebook_snapshots,
                ordered_targets=ordered_targets,
            )
        }
    else:
        snapshots = snapshot_payloads_by_hash
    amm_rows = fit._load_amm_rows(dataset.amm_swaps, tx_column="transaction_hash", target_tx_hashes=set(selected_hashes))
    fee_rows = fit._load_amm_rows(dataset.amm_fees, tx_column="transaction_hash", target_tx_hashes=set(selected_hashes))

    contexts: list[TxContext] = []
    for tx_hash in selected_hashes:
        sample = samples[tx_hash]
        intent = intents[tx_hash]
        tx_type = str(intent.transaction_type or "").upper()
        optimizer_mode = _optimizer_mode_for_intent(intent)
        if optimizer_mode is None:
            continue

        tx_in_transfer_rate = fit._transfer_rate_for_asset(
            currency=intent.in_cur,
            issuer=(
                intent.payment_sendmax_issuer
                if tx_type == "PAYMENT"
                else intent.offer_taker_gets_issuer
            ),
            issuer_transfer_rates=issuer_transfer_rates,
            issuer_transfer_rate_ranges=issuer_transfer_rate_ranges,
            ledger_index=sample.ledger_index,
        )
        tx_out_transfer_rate = fit._transfer_rate_for_asset(
            currency=intent.out_cur,
            issuer=(
                intent.payment_delivermax_issuer
                if tx_type == "PAYMENT"
                else intent.offer_taker_pays_issuer
            ),
            issuer_transfer_rates=issuer_transfer_rates,
            issuer_transfer_rate_ranges=issuer_transfer_rate_ranges,
            ledger_index=sample.ledger_index,
        )
        if tx_type == "OFFERCREATE":
            book_in_transfer_rate = _offer_input_rate_for_bookstep(intent, tx_in_transfer_rate)
            owner_pays_transfer_fee = _offer_output_owner_fee_applies(intent, tx_out_transfer_rate)
            charge_input_transfer_fee = book_in_transfer_rate is not None
            execution_mode = "offer_crossing"
            gateway_sendmax = _offer_gateway_sendmax(intent.offer_taker_gets_amt, book_in_transfer_rate)
            effective_input = _cap_input_budget(gateway_sendmax, intent.source_funds_cap_amt)
            base_limit_quality = Quality.from_amounts(intent.offer_taker_pays_amt, gateway_sendmax)
            effective_limit_quality = (
                _strictly_better_quality_floor(base_limit_quality)
                if bool(intent.offer_is_passive)
                else base_limit_quality
            )
            target_output = None if bool(intent.offer_is_sell) else intent.offer_taker_pays_amt
        else:
            book_in_transfer_rate = tx_in_transfer_rate
            owner_pays_transfer_fee = False
            charge_input_transfer_fee = True
            execution_mode = "payment"
            effective_input = _cap_input_budget(intent.payment_sendmax_amt, intent.source_funds_cap_amt)
            endpoint_out_rate = _payment_output_endpoint_rate(intent, tx_out_transfer_rate)
            target_output = _payment_book_output_target(intent.payment_delivermax_amt, endpoint_out_rate)
            effective_limit_quality = intent.payment_limit_quality

        selected_side = fit._prebook_primary_side_for_output_currency(intent.out_cur)
        snapshot_payload = snapshots[tx_hash]
        owner_funds_overlay = fit._extract_overlay_owner_funds_from_metadata_result(sample.result)
        current_tx_pre_overlay_rows = fit._metadata_current_tx_pre_offer_rows(
            result=sample.result,
            selected_side=selected_side,
        )
        snapshot_payload, _overlay_count = fit._overlay_current_tx_pre_offer_rows(
            side_payload=snapshot_payload,
            current_tx_pre_offer_rows=current_tx_pre_overlay_rows,
        )
        raw_funded_caps_by_offer_id, raw_overridden_accounts = fit._build_external_funded_caps_for_snapshot(
            side_payload=snapshot_payload,
            ledger_index=int(snapshot_payload["used_prebook_ledger"]),
            account_offers_snapshots={},
        )
        overridden_accounts = set(raw_overridden_accounts)
        funded_caps_by_offer_id: dict[str, Any] = {}
        if overridden_accounts:
            for row in snapshot_payload.get("offers", []):
                if not isinstance(row, dict):
                    continue
                account = str(row.get("account") or "")
                offer_id = str(row.get("offer_id") or "")
                if account in overridden_accounts and offer_id in raw_funded_caps_by_offer_id:
                    funded_caps_by_offer_id[offer_id] = raw_funded_caps_by_offer_id[offer_id]

        book_offers = fit._materialize_snapshot_book_offers(
            side_payload=snapshot_payload,
            derived_owner_funds_by_offer_id=owner_funds_overlay,
            funded_caps_by_offer_id=funded_caps_by_offer_id,
            strip_owner_funds_accounts=overridden_accounts,
        )
        book_offers = fit._filter_non_fill_deleted_offers(
            book_offers=book_offers,
            offer_ids=fit._metadata_deleted_offer_without_fill_ids(sample.result),
            keep_owner=str(intent.source_account or "") if tx_type == "OFFERCREATE" else None,
        )
        book_offers = fit._apply_offercreate_cancel_sequence(book_offers=book_offers, intent=intent)
        clob_segments = build_clob_segments_from_offers(
            book_offers,
            in_cur=intent.in_cur,
            out_cur=intent.out_cur,
            iou_in_transfer_rate=tx_in_transfer_rate,
            iou_out_transfer_rate=tx_out_transfer_rate,
            owner_pays_transfer_fee=tx_type == "OFFERCREATE",
            exclude_account=None,
            preserve_book_visible_offers=tx_type == "OFFERCREATE",
        )

        amm, amm_state_report = fit._build_model_amm(
            tx_hash=tx_hash,
            metadata_result=sample.result,
            amm_rows=amm_rows.get(tx_hash, []),
            fee_rows=fee_rows.get(tx_hash, []),
            in_cur=intent.in_cur,
            out_cur=intent.out_cur,
        )
        baseline_quote, _baseline_meta = execute_single_path_intent(
            intent=intent,
            amm=amm,
            segs=clob_segments,
            in_cur=intent.in_cur,
            out_cur=intent.out_cur,
            book_in_transfer_rate=tx_in_transfer_rate,
            book_out_transfer_rate=tx_out_transfer_rate,
            fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
            fix_reduced_offers_v2_enabled=fix_reduced_offers_v2_enabled,
        )
        baseline_fit = _baseline_fit_summary(
            intent=intent,
            sample_result=sample.result,
            baseline_quote=baseline_quote,
            amm_rows=amm_rows.get(tx_hash, []),
            amm_state_report=amm_state_report,
        )
        if require_strict_baseline_fit and baseline_fit.get("status") != "strict_exact_match":
            continue
        _summary, baseline_out, baseline_in = _require_quote_totals(baseline_quote)
        if (
            optimizer_mode == MODE_FIXED_OUTPUT_MIN_INPUT
            and target_output is not None
            and _amount_lt(baseline_out, target_output)
        ):
            # A buy-side OfferCreate can have a nominal TakerPays target while
            # still partially crossing. That is a bounded-partial optimization
            # problem, not an exact-output minimize-input problem.
            continue
        contexts.append(
            TxContext(
                tx_hash=tx_hash,
                ledger_index=int(sample.ledger_index),
                intent=intent,
                amm=amm,
                clob_segments=clob_segments,
                baseline_quote=baseline_quote,
                baseline_fit=baseline_fit,
                baseline_in=baseline_in,
                baseline_out=baseline_out,
                optimizer_mode=optimizer_mode,
                effective_input=effective_input,
                target_output=target_output,
                offer_input_rate=book_in_transfer_rate,
                book_out_transfer_rate=tx_out_transfer_rate,
                owner_pays_transfer_fee=owner_pays_transfer_fee,
                charge_input_transfer_fee=charge_input_transfer_fee,
                effective_limit_quality=effective_limit_quality,
                execution_mode=execution_mode,
                fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
                fix_reduced_offers_v2_enabled=fix_reduced_offers_v2_enabled,
            )
        )
    return contexts


def _sum_source_amount_from_rows(rows: list[Any], *, source: str, side: str, proto: Amount) -> Amount:
    key = "in_take" if side == "in" else "out_take"
    total = _zero_like(proto)
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("src") or "") != source:
            continue
        amount = row.get(key)
        if amount is not None:
            total = amount_add_ripple_number(total, amount)
    return total


def _sum_source_amount(quote: dict[str, Any], *, source: str, side: str, proto: Amount, rows_key: str = "slices") -> Amount:
    return _sum_source_amount_from_rows(
        quote.get(rows_key, []) or [],
        source=source,
        side=side,
        proto=proto,
    )


def _fit_amount_value(report: dict[str, Any], side: str) -> str | None:
    summary = report.get("summary") or {}
    key = "total_in" if side == "in" else "total_out"
    amount = summary.get(key)
    if not isinstance(amount, dict):
        return None
    return str(amount.get("value"))


def _baseline_fit_status(comparison: dict[str, Any]) -> str:
    if bool(comparison.get("strict_exact_match")):
        return "strict_exact_match"
    if bool(comparison.get("structure_match")):
        return "structure_match_endpoint_residual"
    return "structure_mismatch"


def _baseline_fit_summary(
    *,
    intent: TxIntent,
    sample_result: dict[str, Any],
    baseline_quote: dict[str, Any],
    amm_rows: list[dict[str, Any]],
    amm_state_report: dict[str, Any] | None,
) -> dict[str, Any]:
    model_report = fit._model_report_from_quote(
        baseline_quote,
        in_cur=intent.in_cur,
        out_cur=intent.out_cur,
        amm_account=amm_state_report.get("amm_account") if isinstance(amm_state_report, dict) else None,
    )
    real_legs = fit._build_real_offer_legs(sample_result, in_cur=intent.in_cur, out_cur=intent.out_cur)
    real_amm_legs = fit._build_real_amm_legs_from_metadata(
        sample_result,
        in_cur=intent.in_cur,
        out_cur=intent.out_cur,
    )
    if real_amm_legs is None:
        if amm_rows:
            raise RuntimeError(f"missing metadata-exact real AMM legs for tx intent {intent.transaction_type}")
        real_amm_legs = []
    real_legs.extend(real_amm_legs)
    real_total_out, real_total_in = fit._sum_legs(real_legs, in_cur=intent.in_cur, out_cur=intent.out_cur)
    real_report = {
        "legs": real_legs,
        "summary": {
            "total_out": fit._amount_to_report(real_total_out, currency=intent.out_cur),
            "total_in": fit._amount_to_report(real_total_in, currency=intent.in_cur),
        },
    }
    comparison = fit._compare_reports(model=model_report, real=real_report)
    comparison["fit_bucket"] = fit._classify_fit_bucket(
        model=model_report,
        real=real_report,
        comparison=comparison,
    )
    comparison.update(
        fit._endpoint_prediction_role_fields(
            {
                "intent": fit._intent_report(intent),
                "model": model_report,
                "real": real_report,
                "comparison": comparison,
            }
        )
    )
    status = _baseline_fit_status(comparison)
    return {
        "status": status,
        "valid_for_layer1_evidence": status == "strict_exact_match",
        "strict_exact_match": bool(comparison.get("strict_exact_match")),
        "structure_match": bool(comparison.get("structure_match")),
        "fit_bucket": comparison.get("fit_bucket"),
        "prediction_role": comparison.get("prediction_role"),
        "predicted_side": comparison.get("predicted_side"),
        "constraint_side": comparison.get("constraint_side"),
        "model_total_in": _fit_amount_value(model_report, "in"),
        "model_total_out": _fit_amount_value(model_report, "out"),
        "real_total_in": _fit_amount_value(real_report, "in"),
        "real_total_out": _fit_amount_value(real_report, "out"),
        "total_in_match": bool(comparison.get("total_in_match")),
        "total_out_match": bool(comparison.get("total_out_match")),
        "total_in_diff": comparison.get("total_in_diff"),
        "total_out_diff": comparison.get("total_out_diff"),
        "total_in_endpoint_unsigned_rel_err_bps": comparison.get("total_in_endpoint_unsigned_rel_err_bps"),
        "total_out_endpoint_unsigned_rel_err_bps": comparison.get("total_out_endpoint_unsigned_rel_err_bps"),
        "max_side_endpoint_unsigned_rel_err_bps": comparison.get("max_side_endpoint_unsigned_rel_err_bps"),
        "model_leg_count": comparison.get("model_leg_count"),
        "real_leg_count": comparison.get("real_leg_count"),
        "clob_offer_multiset_match": bool(comparison.get("clob_offer_multiset_match")),
        "amm_participation_match": bool(comparison.get("amm_participation_match")),
    }


def _price_out_per_in(out_amount: Amount, in_amount: Amount) -> Decimal | None:
    in_dec = fit._amount_to_decimal(in_amount)
    if in_dec == 0:
        return None
    return fit._amount_to_decimal(out_amount) / in_dec


def _price_in_per_out(in_amount: Amount, out_amount: Amount) -> Decimal | None:
    out_dec = fit._amount_to_decimal(out_amount)
    if out_dec == 0:
        return None
    return fit._amount_to_decimal(in_amount) / out_dec


def _unit_decimal_like(proto: Amount) -> Decimal:
    return fit._amount_to_decimal(_amount_from_units_like(proto, 1))


def _rate_decimal(rate: IOUAmount | None) -> Decimal:
    if rate is None:
        return Decimal(1)
    return fit._amount_to_decimal(rate)


def _amm_keep_fee_decimal(ctx: TxContext) -> Decimal:
    if ctx.amm is None:
        return Decimal(0)
    return Decimal(100_000 - int(ctx.amm.trading_fee)) / Decimal(100_000)


def _amm_marginal_after_input(ctx: TxContext, consumed_units: int) -> Decimal | None:
    """Continuous AMM marginal output per gross input at a split point.

    Structural localization must use the AMM curve's asset-level marginal, not
    a one-ledger-unit finite difference. Tiny IOU output units and XRP drop
    rounding otherwise invert the derivative near boundaries.
    """
    if ctx.amm is None:
        return None
    keep = _amm_keep_fee_decimal(ctx)
    rate = _rate_decimal(_active_amm_input_transfer_rate(ctx))
    if keep <= 0 or rate <= 0:
        return Decimal(0)
    pool_in = fit._amount_to_decimal(ctx.amm._x_amount)
    pool_out = fit._amount_to_decimal(ctx.amm._y_amount)
    if pool_in <= 0 or pool_out <= 0:
        return Decimal(0)
    gross_consumed = fit._amount_to_decimal(_amount_from_units_like(ctx.effective_input, consumed_units))
    raw_consumed = gross_consumed / rate
    denominator = pool_in + keep * raw_consumed
    if denominator <= 0:
        return None
    return (keep * pool_in * pool_out) / (denominator * denominator * rate)


def _add_candidate(
    candidates: set[int],
    total_units: int,
    value: int,
    *,
    window: int,
    lower_bound: int = 0,
    upper_bound: int | None = None,
) -> None:
    upper = total_units if upper_bound is None else min(total_units, int(upper_bound))
    lower = max(0, int(lower_bound))
    for delta in range(-window, window + 1):
        candidate = int(value) + delta
        if lower <= candidate <= upper:
            candidates.add(candidate)


def _clob_segment_intervals(ctx: TxContext, *, max_clob_segments: int) -> list[tuple[int, int, Any]]:
    total_units = _amt_units_floor(ctx.effective_input)
    intervals: list[tuple[int, int, Any]] = []
    cum_before = 0
    for seg in _executable_clob_segments(ctx)[:max_clob_segments]:
        seg_in = getattr(seg, "in_at_out_max")
        seg_units = _amt_units_floor(seg_in)
        if seg_units <= 0:
            continue
        cum_after = cum_before + seg_units
        interval_lo = max(0, total_units - cum_after)
        interval_hi = min(total_units, total_units - cum_before)
        if interval_lo <= interval_hi:
            intervals.append((interval_lo, interval_hi, seg))
        cum_before = cum_after
        if cum_before >= total_units:
            break
    return intervals


def _fixed_input_interval_role(
    ctx: TxContext,
    *,
    interval_lo: int,
    interval_hi: int,
    seg: Any,
) -> dict[str, Any]:
    seg_price = _price_out_per_in(getattr(seg, "out_max"), getattr(seg, "in_at_out_max"))
    amm_lo = _amm_marginal_after_input(ctx, interval_lo)
    amm_hi = _amm_marginal_after_input(ctx, interval_hi)
    role = "unknown"
    derivative_lo: Decimal | None = None
    derivative_hi: Decimal | None = None
    if seg_price is not None and amm_lo is not None and amm_hi is not None:
        derivative_lo = amm_lo - seg_price
        derivative_hi = amm_hi - seg_price
        if derivative_lo >= 0 >= derivative_hi:
            role = "crossing_interval"
        elif derivative_hi > 0:
            role = "amm_better_through_interval"
        elif derivative_lo < 0:
            role = "clob_better_through_interval"
        else:
            role = "non_monotone_or_rounding_irregularity"
    return {
        "role": role,
        "clob_out_per_input": _decimal_text(seg_price),
        "amm_marginal_lo": _decimal_text(amm_lo),
        "amm_marginal_hi": _decimal_text(amm_hi),
        "derivative_lo": _decimal_text(derivative_lo),
        "derivative_hi": _decimal_text(derivative_hi),
    }


def _fixed_input_feasible_domain_units(ctx: TxContext) -> dict[str, Any]:
    total_units = _amt_units_floor(ctx.effective_input)
    _amm_out, amm_spent = _quote_source_exact_in(ctx, source="AMM", input_amount=ctx.effective_input)
    _clob_out, clob_spent = _quote_source_exact_in(ctx, source="CLOB", input_amount=ctx.effective_input)
    max_amm_units = min(total_units, _amt_units_floor(amm_spent))
    max_clob_units = min(total_units, _amt_units_floor(clob_spent))

    left = 0
    right = total_units
    while left < right:
        mid = (left + right) // 2
        amm_amount = _amount_from_units_like(ctx.effective_input, mid)
        clob_amount = _max_complement_for_canonical_sum(total=ctx.effective_input, fixed=amm_amount)
        if _amt_units_floor(clob_amount) <= max_clob_units:
            right = mid
        else:
            left = mid + 1

    domain_lo = left
    domain_hi = max_amm_units
    return {
        "total_units": int(total_units),
        "domain_lo": int(domain_lo),
        "domain_hi": int(domain_hi),
        "max_amm_input_units": int(max_amm_units),
        "max_clob_input_units": int(max_clob_units),
        "max_amm_spent": _amount_text(amm_spent),
        "max_clob_spent": _amount_text(clob_spent),
        "feasible": domain_lo <= domain_hi,
    }


def _fixed_input_structural_anchor(
    ctx: TxContext,
    *,
    max_clob_segments: int,
) -> dict[str, Any]:
    """Locate the unique continuous optimum for fixed-input max-output.

    The objective is concave in AMM input allocation. Within a CLOB interval the
    derivative is AMM marginal output per input minus that interval's constant
    CLOB output per input. The optimum is therefore either the unique derivative
    crossing or the relevant domain boundary.
    """
    total_units = _amt_units_floor(ctx.effective_input)
    feasible_domain = _fixed_input_feasible_domain_units(ctx)
    domain_lo = int(feasible_domain["domain_lo"])
    domain_hi = int(feasible_domain["domain_hi"])
    intervals = [
        (max(interval_lo, domain_lo), min(interval_hi, domain_hi), seg)
        for interval_lo, interval_hi, seg in sorted(
            _clob_segment_intervals(ctx, max_clob_segments=max_clob_segments),
            key=lambda item: item[0],
        )
        if max(interval_lo, domain_lo) <= min(interval_hi, domain_hi)
    ]
    coverage = _interval_coverage_report(
        ctx,
        side="input",
        required_units=total_units,
        max_clob_segments=max_clob_segments,
    )

    def exact_boundary(point: int, *, reason: str, boundary: str) -> dict[str, Any]:
        return {
            "method": STRUCTURAL_CERT_METHOD,
            "side": "input",
            "status": "exact",
            "reason": reason,
            "structural_kind": "allocation_boundary" if boundary in {"domain_lower", "domain_upper"} else "clob_boundary",
            "anchor_units": [int(point)],
            "boundary": boundary,
            "feasible_domain": feasible_domain,
            "domain_coverage": coverage,
        }

    if not bool(feasible_domain["feasible"]):
        return {
            "method": STRUCTURAL_CERT_METHOD,
            "side": "input",
            "status": "exact",
            "reason": "empty_source_feasible_domain",
            "structural_kind": "infeasible_domain",
            "anchor_units": [],
            "feasible_domain": feasible_domain,
            "domain_coverage": coverage,
        }
    if ctx.amm is None:
        return exact_boundary(domain_lo, reason="no_amm_available", boundary="domain_lower")
    if not intervals:
        return exact_boundary(domain_hi, reason="no_clob_interval_available", boundary="domain_upper")
    if coverage.get("complete") is False:
        return {
            "method": STRUCTURAL_CERT_METHOD,
            "side": "input",
            "status": "not_certified",
            "reason": "clob_domain_truncated_by_max_clob_segments",
            "anchor_units": [],
            "feasible_domain": feasible_domain,
            "domain_coverage": coverage,
        }

    roles: list[dict[str, Any]] = []
    for index, (interval_lo, interval_hi, seg) in enumerate(intervals):
        role = _fixed_input_interval_role(ctx, interval_lo=interval_lo, interval_hi=interval_hi, seg=seg)
        role_info = {
            "interval_index": index,
            "interval_lo": int(interval_lo),
            "interval_hi": int(interval_hi),
            **role,
        }
        roles.append(role_info)
        seg_price = _price_out_per_in(getattr(seg, "out_max"), getattr(seg, "in_at_out_max"))
        price_lo = _amm_marginal_after_input(ctx, interval_lo)
        price_hi = _amm_marginal_after_input(ctx, interval_hi)
        if seg_price is None or price_lo is None or price_hi is None:
            return {
                "method": STRUCTURAL_CERT_METHOD,
                "side": "input",
                "status": "not_certified",
                "reason": "missing_marginal_price",
                "anchor_units": [],
                "feasible_domain": feasible_domain,
                "domain_coverage": coverage,
                "failed_interval": role_info,
                "roles_sample": roles[:10],
            }
        if price_lo >= seg_price >= price_hi:
            left = interval_lo
            right = interval_hi
            for _ in range(80):
                if right - left <= 1:
                    break
                mid = (left + right) // 2
                price_mid = _amm_marginal_after_input(ctx, mid)
                if price_mid is None:
                    return {
                        "method": STRUCTURAL_CERT_METHOD,
                        "side": "input",
                        "status": "not_certified",
                        "reason": "missing_crossing_midpoint_price",
                        "anchor_units": [],
                        "feasible_domain": feasible_domain,
                        "domain_coverage": coverage,
                        "failed_interval": role_info,
                        "roles_sample": roles[:10],
                    }
                if price_mid >= seg_price:
                    left = mid
                else:
                    right = mid
            return {
                "method": STRUCTURAL_CERT_METHOD,
                "side": "input",
                "status": "exact",
                "reason": "marginal_crossing",
                "structural_kind": "amm_clob_marginal_crossing",
                "anchor_units": sorted({int(left), int(right)}),
                "crossing_interval": role_info,
                "feasible_domain": feasible_domain,
                "domain_coverage": coverage,
                "roles_sample": roles[:10],
            }
        if price_lo < seg_price:
            return {
                "method": STRUCTURAL_CERT_METHOD,
                "side": "input",
                "status": "exact",
                "reason": "derivative_negative_at_domain_point",
                "structural_kind": "clob_offer_boundary",
                "anchor_units": [int(interval_lo)],
                "boundary": "interval_left",
                "boundary_interval": role_info,
                "feasible_domain": feasible_domain,
                "domain_coverage": coverage,
                "roles_sample": roles[:10],
            }

    return {
        "method": STRUCTURAL_CERT_METHOD,
        "side": "input",
        "status": "exact",
        "reason": "derivative_positive_through_domain",
        "structural_kind": "allocation_boundary",
        "anchor_units": [int(domain_hi)],
        "boundary": "domain_upper",
        "feasible_domain": feasible_domain,
        "domain_coverage": coverage,
        "roles_sample": roles[:10],
    }


def _candidate_set_from_fixed_input_structural_anchor(
    ctx: TxContext,
    *,
    max_clob_segments: int,
    rounding_window_drops: int,
) -> tuple[set[int], dict[str, Any]]:
    total_units = _amt_units_floor(ctx.effective_input)
    certificate = _fixed_input_structural_anchor(ctx, max_clob_segments=max_clob_segments)
    candidates: set[int] = set()
    feasible_domain = certificate.get("feasible_domain") or {}
    lower_bound = int(feasible_domain.get("domain_lo", 0))
    upper_bound = int(feasible_domain.get("domain_hi", total_units))
    for anchor in certificate.get("anchor_units") or []:
        _add_candidate(
            candidates,
            total_units,
            int(anchor),
            window=rounding_window_drops,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    certificate["rounding_window_units"] = int(rounding_window_drops)
    certificate["candidate_units"] = sorted(candidates)
    return candidates, certificate


def _fixed_input_best_point_diagnostic(
    ctx: TxContext,
    *,
    best: dict[str, Any] | None,
    max_clob_segments: int,
    rounding_window_units: int,
) -> dict[str, Any] | None:
    if best is None:
        return None
    point = best.get("amm_input_units", best.get("amm_input_drops"))
    if point is None:
        return None
    point = int(point)
    total_units = _amt_units_floor(ctx.effective_input)
    intervals = _clob_segment_intervals(ctx, max_clob_segments=max_clob_segments)
    interval_infos: list[dict[str, Any]] = []
    crossing_indices: list[int] = []
    matched: dict[str, Any] | None = None
    for index, (interval_lo, interval_hi, seg) in enumerate(intervals):
        role = _fixed_input_interval_role(ctx, interval_lo=interval_lo, interval_hi=interval_hi, seg=seg)
        info = {
            "interval_index": index,
            "interval_lo": int(interval_lo),
            "interval_hi": int(interval_hi),
            **role,
        }
        if role.get("role") == "crossing_interval":
            crossing_indices.append(index)
        interval_infos.append(info)
        if interval_lo <= point <= interval_hi and matched is None:
            distance_to_left = point - interval_lo
            distance_to_right = interval_hi - point
            matched = {
                **info,
                "point_location": _point_location_in_interval(
                    point=point,
                    interval_lo=interval_lo,
                    interval_hi=interval_hi,
                ),
                "distance_to_left_boundary_units": int(distance_to_left),
                "distance_to_right_boundary_units": int(distance_to_right),
                "nearest_boundary_distance_units": int(min(distance_to_left, distance_to_right)),
            }

    if matched is None:
        origin = "global_boundary" if point in (0, total_units) else "outside_scanned_intervals"
        return {
            "side": "input",
            "point_units": point,
            "total_units": total_units,
            "origin": origin,
            "matched_interval": None,
            "crossing_interval_indices": crossing_indices,
            "crossing_interval_count": len(crossing_indices),
            "intervals_sample": interval_infos[:10],
        }

    matched_index = int(matched["interval_index"])
    location = str(matched["point_location"])
    role = str(matched["role"])
    boundary_neighborhood_units = max(1, int(rounding_window_units) * 4)
    near_boundary = int(matched["nearest_boundary_distance_units"]) <= boundary_neighborhood_units
    if role == "crossing_interval":
        origin = "main_crossing_boundary" if "boundary" in location else "main_crossing_interior_or_neighbor"
    elif crossing_indices and any(abs(matched_index - idx) <= 1 for idx in crossing_indices) and "boundary" in location:
        origin = "adjacent_to_crossing_boundary"
    elif point in (0, total_units):
        origin = "global_boundary"
    elif near_boundary:
        origin = "monotone_boundary_neighbor"
    elif "boundary" in location:
        origin = "other_interval_boundary"
    else:
        origin = "other_interval_interior"
    return {
        "side": "input",
        "point_units": point,
        "total_units": total_units,
        "origin": origin,
        "matched_interval": matched,
        "crossing_interval_indices": crossing_indices,
        "crossing_interval_count": len(crossing_indices),
        "is_in_crossing_interval": role == "crossing_interval",
        "is_adjacent_to_crossing_interval": bool(
            crossing_indices and any(abs(matched_index - idx) <= 1 for idx in crossing_indices)
        ),
        "boundary_neighborhood_units": boundary_neighborhood_units,
        "is_near_interval_boundary": near_boundary,
        "intervals_sample": interval_infos[:10],
    }


def _active_amm_input_transfer_rate(ctx: TxContext) -> IOUAmount | None:
    return ctx.offer_input_rate if ctx.charge_input_transfer_fee else None


def _amm_exact_output_input(ctx: TxContext, output_amount: Amount) -> tuple[Amount, Amount] | None:
    if ctx.amm is None or _amt_units_floor(output_amount) <= 0:
        return None
    raw_in = ctx.amm.swap_in_given_out_st(output_amount)
    if _amt_units_floor(raw_in) <= 0:
        return None
    gross_in = _apply_rate(
        raw_in,
        _active_amm_input_transfer_rate(ctx),
        round_up=True,
    )
    if _amt_units_floor(gross_in) <= 0:
        return None
    return gross_in, raw_in


def _amm_take_violates_limit_quality(
    ctx: TxContext,
    *,
    output_amount: Amount,
    gross_input: Amount,
    raw_input: Amount,
) -> bool:
    if ctx.effective_limit_quality is None:
        return False
    itr_quality = Quality.from_amounts(output_amount, gross_input)
    return _iteration_violates_limit_quality(
        mode=ctx.execution_mode,
        limit_quality=ctx.effective_limit_quality,
        itr_quality=itr_quality,
        adjusted_out=False,
        amm_in=raw_input,
        amm_out=output_amount,
    )


def _quote_amm_exact_out(ctx: TxContext, *, output_amount: Amount) -> tuple[Amount, Amount] | None:
    if _amt_units_floor(output_amount) <= 0:
        return _zero_like(output_amount), _zero_like(ctx.effective_input)
    quoted = _amm_exact_output_input(ctx, output_amount)
    if quoted is None:
        return None
    gross_in, raw_in = quoted
    if _amm_take_violates_limit_quality(
        ctx,
        output_amount=output_amount,
        gross_input=gross_in,
        raw_input=raw_in,
    ):
        return None
    return output_amount, gross_in


def _quote_amm_max_output_under_input(ctx: TxContext, *, input_amount: Amount) -> tuple[Amount, Amount]:
    if _amt_units_floor(input_amount) <= 0:
        return _zero_like(ctx.baseline_out), _zero_like(ctx.effective_input)
    if ctx.amm is None:
        return _zero_like(ctx.baseline_out), _zero_like(ctx.effective_input)
    try:
        quote, _meta = run_payment_constrained_quote(
            amm=ctx.amm.clone(),
            segs=[],
            in_cur=ctx.intent.in_cur,
            out_cur=ctx.intent.out_cur,
            deliver_max_out=_sell_deliver_ceiling(ctx.intent.out_cur),
            send_max_in=input_amount,
            deliver_min_out=None,
            limit_quality=ctx.effective_limit_quality,
            source_account=ctx.intent.source_account,
            execution_mode=ctx.execution_mode,
            book_in_transfer_rate=ctx.offer_input_rate,
            book_out_transfer_rate=ctx.book_out_transfer_rate,
            charge_input_transfer_fee=ctx.charge_input_transfer_fee,
            owner_pays_transfer_fee=ctx.owner_pays_transfer_fee,
            enable_self_cross_filter=ctx.execution_mode == "offer_crossing",
            fix_amm_v1_1_enabled=ctx.fix_amm_v1_1_enabled,
            fix_reduced_offers_v2_enabled=ctx.fix_reduced_offers_v2_enabled,
        )
    except RuntimeError:
        return _zero_like(ctx.baseline_out), _zero_like(ctx.effective_input)
    _summary, total_out, total_in = _require_quote_totals(quote)
    if _amt_units_floor(total_out) <= 0 or _amt_units_floor(total_in) <= 0:
        return _zero_like(ctx.baseline_out), _zero_like(ctx.effective_input)
    return total_out, total_in


def _quote_source_exact_in(ctx: TxContext, *, source: str, input_amount: Amount) -> tuple[Amount, Amount]:
    if source == "AMM":
        return _quote_amm_max_output_under_input(ctx, input_amount=input_amount)
    if _amt_units_floor(input_amount) <= 0:
        return _zero_like(ctx.baseline_out), _zero_like(ctx.effective_input)
    quote, _meta = run_payment_constrained_quote(
        amm=ctx.amm.clone() if source == "AMM" and ctx.amm is not None else None,
        segs=[] if source == "AMM" else ctx.clob_segments,
        in_cur=ctx.intent.in_cur,
        out_cur=ctx.intent.out_cur,
        deliver_max_out=_sell_deliver_ceiling(ctx.intent.out_cur),
        send_max_in=input_amount,
        deliver_min_out=None,
        limit_quality=ctx.effective_limit_quality,
        source_account=ctx.intent.source_account,
        execution_mode=ctx.execution_mode,
        book_in_transfer_rate=ctx.offer_input_rate,
        book_out_transfer_rate=ctx.book_out_transfer_rate,
        charge_input_transfer_fee=ctx.charge_input_transfer_fee,
        owner_pays_transfer_fee=ctx.owner_pays_transfer_fee,
        enable_self_cross_filter=ctx.execution_mode == "offer_crossing",
        fix_amm_v1_1_enabled=ctx.fix_amm_v1_1_enabled,
        fix_reduced_offers_v2_enabled=ctx.fix_reduced_offers_v2_enabled,
    )
    _summary, total_out, total_in = _require_quote_totals(quote)
    return total_out, total_in


def _quote_source_exact_out(ctx: TxContext, *, source: str, output_amount: Amount) -> tuple[Amount, Amount] | None:
    if source == "AMM":
        return _quote_amm_exact_out(ctx, output_amount=output_amount)
    if _amt_units_floor(output_amount) <= 0:
        return _zero_like(output_amount), _zero_like(ctx.effective_input)
    try:
        quote, _meta = run_payment_constrained_quote(
            amm=ctx.amm.clone() if source == "AMM" and ctx.amm is not None else None,
            segs=[] if source == "AMM" else ctx.clob_segments,
            in_cur=ctx.intent.in_cur,
            out_cur=ctx.intent.out_cur,
            deliver_max_out=output_amount,
            send_max_in=ctx.effective_input,
            deliver_min_out=output_amount,
            limit_quality=ctx.effective_limit_quality,
            source_account=ctx.intent.source_account,
            execution_mode=ctx.execution_mode,
            book_in_transfer_rate=ctx.offer_input_rate,
            book_out_transfer_rate=ctx.book_out_transfer_rate,
            charge_input_transfer_fee=ctx.charge_input_transfer_fee,
            owner_pays_transfer_fee=ctx.owner_pays_transfer_fee,
            enable_self_cross_filter=ctx.execution_mode == "offer_crossing",
            fix_amm_v1_1_enabled=ctx.fix_amm_v1_1_enabled,
            fix_reduced_offers_v2_enabled=ctx.fix_reduced_offers_v2_enabled,
        )
    except RuntimeError:
        return None
    _summary, total_out, total_in = _require_quote_totals(quote)
    if not _amount_units_exactly_equal(total_out, output_amount):
        return None
    if _amt_units_floor(total_in) <= 0:
        return None
    return total_out, total_in


def _max_complement_for_canonical_sum(*, total: Amount, fixed: Amount) -> Amount:
    """Largest complement whose rippled Number sum still canonicalizes to total."""
    base = _amount_sub(total, fixed)
    if _amt_units_floor(base) <= 0:
        return _zero_like(total)
    if not _amount_units_exactly_equal(amount_add_ripple_number(fixed, base), total):
        return base

    base_units = _amt_units_floor(base)
    step = max(
        1,
        _amount_canonical_quantum_units(total),
        _amount_canonical_quantum_units(fixed),
        _amount_canonical_quantum_units(base),
    )
    lo = base_units
    hi = base_units

    while True:
        probe_units = hi + step
        probe = _amount_from_units_like(base, probe_units)
        if _amount_units_exactly_equal(amount_add_ripple_number(fixed, probe), total):
            hi = probe_units
            step *= 2
            continue
        break

    left = hi + 1
    right = hi + step - 1
    best = hi
    while left <= right:
        mid = (left + right) // 2
        probe = _amount_from_units_like(base, mid)
        if _amount_units_exactly_equal(amount_add_ripple_number(fixed, probe), total):
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return _amount_from_units_like(base, best)


def _quote_source_exact_out_for_candidate(
    ctx: TxContext,
    *,
    source: str,
    output_amount: Amount,
    other_output_amount: Amount | None = None,
) -> tuple[Amount, Amount] | None:
    return _quote_source_exact_out(ctx, source=source, output_amount=output_amount)


def _evaluate_candidate(ctx: TxContext, amm_drops: int) -> dict[str, Any]:
    total_units = _amt_units_floor(ctx.effective_input)
    amm_drops = max(0, min(total_units, int(amm_drops)))
    amm_input_amount = _amount_from_units_like(ctx.effective_input, amm_drops)
    clob_input_amount = _max_complement_for_canonical_sum(
        total=ctx.effective_input,
        fixed=amm_input_amount,
    )
    amm_out, amm_spent = _quote_source_exact_in(ctx, source="AMM", input_amount=amm_input_amount)
    clob_out, clob_spent = _quote_source_exact_in(ctx, source="CLOB", input_amount=clob_input_amount)
    total_out = amount_add_ripple_number(amm_out, clob_out)
    total_spent = amount_add_ripple_number(amm_spent, clob_spent)
    spent_units = _amt_units_floor(total_spent)
    full_fill = spent_units == total_units
    improvement = fit._amount_to_decimal(total_out) - fit._amount_to_decimal(ctx.baseline_out)
    baseline_out = fit._amount_to_decimal(ctx.baseline_out)
    improvement_bps = None if baseline_out == 0 else improvement / baseline_out * Decimal(10_000)
    return {
        "amm_input_drops": amm_drops,
        "amm_input_units": amm_drops,
        "candidate_kind": "structural_anchor_neighbor",
        "amm_input": _amount_text(amm_input_amount),
        "clob_input": _amount_text(clob_input_amount),
        "spent_input": _amount_text(total_spent),
        "full_fill": full_fill,
        "output": _amount_text(total_out),
        "improvement": format(improvement, "f"),
        "improvement_bps": None if improvement_bps is None else format(improvement_bps, "f"),
    }


def _fixed_input_incumbent_candidate(ctx: TxContext) -> dict[str, Any]:
    baseline_amm = _sum_source_amount(ctx.baseline_quote, source="AMM", side="in", proto=ctx.effective_input)
    baseline_clob = _sum_source_amount(ctx.baseline_quote, source="CLOB", side="in", proto=ctx.effective_input)
    baseline_out = fit._amount_to_decimal(ctx.baseline_out)
    baseline_meets_input = _amount_eq_or_one_quantum_short(ctx.baseline_in, ctx.effective_input)
    return {
        "amm_input_drops": _amt_units_floor(baseline_amm),
        "amm_input_units": _amt_units_floor(baseline_amm),
        "candidate_kind": "baseline_incumbent",
        "amm_input": _amount_text(baseline_amm),
        "clob_input": _amount_text(baseline_clob),
        "spent_input": _amount_text(ctx.baseline_in),
        "full_fill": baseline_meets_input,
        "output": _amount_text(ctx.baseline_out),
        "improvement": "0" if baseline_meets_input else None,
        "improvement_bps": None if baseline_out == 0 or not baseline_meets_input else "0",
    }


def _evaluate_fixed_output_candidate(ctx: TxContext, amm_output_units: int) -> dict[str, Any]:
    if ctx.target_output is None:
        raise RuntimeError("fixed-output optimizer missing target output")
    target_units = _amt_units_floor(ctx.target_output)
    amm_output_units = max(0, min(target_units, int(amm_output_units)))
    amm_requested_out = _amt_from_units(ctx.intent.out_cur, amm_output_units)
    clob_requested_out = _amount_sub(ctx.target_output, amm_requested_out)
    amm_quote = _quote_source_exact_out_for_candidate(ctx, source="AMM", output_amount=amm_requested_out)
    clob_quote = _quote_source_exact_out_for_candidate(
        ctx,
        source="CLOB",
        output_amount=clob_requested_out,
        other_output_amount=amm_quote[0] if amm_quote is not None else None,
    )
    source_quotes_fill = amm_quote is not None and clob_quote is not None
    meets_target = False
    within_input_cap = False
    full_fill = source_quotes_fill
    if source_quotes_fill:
        amm_out, amm_spent = amm_quote
        clob_out, clob_spent = clob_quote
        total_out = amount_add_ripple_number(amm_out, clob_out)
        total_spent = amount_add_ripple_number(amm_spent, clob_spent)
        meets_target = _amount_units_exactly_equal(total_out, ctx.target_output)
        within_input_cap = not _amount_lt(ctx.effective_input, total_spent)
        spends_positive_input = _amt_units_floor(total_spent) > 0
        full_fill = meets_target and within_input_cap and spends_positive_input
    else:
        amm_out = amm_requested_out
        clob_out = clob_requested_out
        total_out = _zero_like(ctx.target_output)
        total_spent = _zero_like(ctx.effective_input)
    baseline_in = fit._amount_to_decimal(ctx.baseline_in)
    spent_in = fit._amount_to_decimal(total_spent)
    improvement = baseline_in - spent_in if full_fill else Decimal("-Infinity")
    improvement_bps = None if baseline_in == 0 or not full_fill else improvement / baseline_in * Decimal(10_000)
    return {
        "amm_output_units": amm_output_units,
        "candidate_kind": "structural_anchor_neighbor",
        "amm_requested_output": _amount_text(amm_requested_out),
        "clob_requested_output": _amount_text(clob_requested_out),
        "amm_output": _amount_text(amm_out),
        "clob_output": _amount_text(clob_out),
        "amm_input": _amount_text(amm_quote[1]) if amm_quote is not None else None,
        "clob_input": _amount_text(clob_quote[1]) if clob_quote is not None else None,
        "spent_input": _amount_text(total_spent),
        "output": _amount_text(total_out),
        "source_quotes_fill": source_quotes_fill,
        "meets_target": meets_target,
        "within_input_cap": within_input_cap,
        "full_fill": full_fill,
        "improvement": None if not full_fill else format(improvement, "f"),
        "improvement_bps": None if improvement_bps is None else format(improvement_bps, "f"),
    }


def _fixed_output_incumbent_candidate(ctx: TxContext) -> dict[str, Any]:
    if ctx.target_output is None:
        raise RuntimeError("fixed-output optimizer missing target output")
    baseline_amm_out = _sum_source_amount(
        ctx.baseline_quote,
        source="AMM",
        side="out",
        proto=ctx.target_output,
        rows_key="endpoint_slices",
    )
    baseline_clob_out = _sum_source_amount(
        ctx.baseline_quote,
        source="CLOB",
        side="out",
        proto=ctx.target_output,
        rows_key="endpoint_slices",
    )
    baseline_amm_in = _sum_source_amount(
        ctx.baseline_quote,
        source="AMM",
        side="in",
        proto=ctx.effective_input,
        rows_key="endpoint_slices",
    )
    baseline_clob_in = _sum_source_amount(
        ctx.baseline_quote,
        source="CLOB",
        side="in",
        proto=ctx.effective_input,
        rows_key="endpoint_slices",
    )
    baseline_meets_target = _amount_units_exactly_equal(ctx.baseline_out, ctx.target_output)
    return {
        "amm_output_units": _amt_units_floor(baseline_amm_out),
        "candidate_kind": "baseline_incumbent",
        "amm_requested_output": _amount_text(baseline_amm_out),
        "clob_requested_output": _amount_text(baseline_clob_out),
        "amm_output": _amount_text(baseline_amm_out),
        "clob_output": _amount_text(baseline_clob_out),
        "amm_input": _amount_text(baseline_amm_in),
        "clob_input": _amount_text(baseline_clob_in),
        "spent_input": _amount_text(ctx.baseline_in),
        "output": _amount_text(ctx.baseline_out),
        "source_quotes_fill": True,
        "meets_target": baseline_meets_target,
        "within_input_cap": True,
        "full_fill": baseline_meets_target,
        "improvement": "0" if baseline_meets_target else None,
        "improvement_bps": "0" if baseline_meets_target else None,
    }


def _fixed_output_segment_intervals(ctx: TxContext, *, max_clob_segments: int) -> list[tuple[int, int, Any]]:
    if ctx.target_output is None:
        return []
    target_units = _amt_units_floor(ctx.target_output)
    intervals: list[tuple[int, int, Any]] = []
    cum_before = 0
    for seg in ctx.clob_segments[:max_clob_segments]:
        seg_out = getattr(seg, "out_max")
        seg_units = _amt_units_floor(seg_out)
        if seg_units <= 0:
            continue
        cum_after = cum_before + seg_units
        interval_lo = max(0, target_units - cum_after)
        interval_hi = min(target_units, target_units - cum_before)
        if interval_lo <= interval_hi:
            intervals.append((interval_lo, interval_hi, seg))
        cum_before = cum_after
        if cum_before >= target_units:
            break
    return intervals


def _positive_segment_units(ctx: TxContext, *, side: str) -> list[int]:
    units: list[int] = []
    attr = "in_at_out_max" if side == "input" else "out_max"
    for seg in ctx.clob_segments:
        seg_units = _amt_units_floor(getattr(seg, attr))
        if seg_units > 0:
            units.append(seg_units)
    return units


def _interval_coverage_report(
    ctx: TxContext,
    *,
    side: str,
    required_units: int,
    max_clob_segments: int,
) -> dict[str, Any]:
    segment_units = _positive_segment_units(ctx, side=side)
    scanned_units = sum(segment_units[:max_clob_segments])
    positive_segment_count = len(segment_units)
    scanned_segment_count = min(positive_segment_count, max_clob_segments)
    complete = positive_segment_count <= max_clob_segments or scanned_units >= required_units
    return {
        "complete": complete,
        "side": side,
        "required_units": int(required_units),
        "scanned_units": int(scanned_units),
        "scanned_segment_count": int(scanned_segment_count),
        "positive_segment_count": int(positive_segment_count),
        "max_clob_segments": int(max_clob_segments),
    }


def _clob_input_per_output(seg: Any) -> Decimal | None:
    return _price_in_per_out(getattr(seg, "in_at_out_max"), getattr(seg, "out_max"))


def _fixed_output_amm_cost_decimal_direct(ctx: TxContext, output_units: int) -> Decimal | None:
    if ctx.amm is None or ctx.target_output is None:
        return None
    output_units = int(output_units)
    if output_units <= 0:
        return Decimal(0)
    target_units = _amt_units_floor(ctx.target_output)
    if output_units > target_units:
        return None
    requested_out = _amt_from_units(ctx.intent.out_cur, output_units)
    quoted = _amm_exact_output_input(ctx, requested_out)
    if quoted is None:
        return None
    spent_in, _raw_in = quoted
    if _amt_units_floor(spent_in) <= 0:
        return None
    return fit._amount_to_decimal(spent_in)


def _fixed_output_amm_marginal_input_per_output_unit(
    ctx: TxContext,
    *,
    output_units: int,
    side: str,
) -> Decimal | None:
    """Continuous AMM marginal gross input per output at a split point."""
    if side not in {"left", "right"}:
        raise RuntimeError(f"unknown AMM marginal side: {side}")
    if ctx.amm is None or ctx.target_output is None:
        return None
    keep = _amm_keep_fee_decimal(ctx)
    rate = _rate_decimal(_active_amm_input_transfer_rate(ctx))
    if keep <= 0 or rate <= 0:
        return None
    pool_in = fit._amount_to_decimal(ctx.amm._x_amount)
    pool_out = fit._amount_to_decimal(ctx.amm._y_amount)
    output_dec = fit._amount_to_decimal(_amt_from_units(ctx.intent.out_cur, output_units))
    if pool_in <= 0 or pool_out <= 0 or output_dec < 0 or output_dec >= pool_out:
        return None
    remaining_out = pool_out - output_dec
    if remaining_out <= 0:
        return None
    return (pool_in * pool_out * rate) / (keep * remaining_out * remaining_out)


def _point_location_in_interval(*, point: int, interval_lo: int, interval_hi: int) -> str:
    if point == interval_lo and point == interval_hi:
        return "single_point_interval"
    if point == interval_lo:
        return "interval_left_boundary"
    if point == interval_hi:
        return "interval_right_boundary"
    return "interval_interior"


def _fixed_output_interval_role(
    ctx: TxContext,
    *,
    interval_lo: int,
    interval_hi: int,
    seg: Any,
) -> dict[str, Any]:
    clob_unit_cost = _clob_input_per_output(seg)
    amm_lo = _fixed_output_amm_marginal_input_per_output_unit(
        ctx,
        output_units=interval_lo,
        side="right",
    )
    amm_hi = _fixed_output_amm_marginal_input_per_output_unit(
        ctx,
        output_units=interval_hi,
        side="left",
    )
    role = "unknown"
    derivative_lo: Decimal | None = None
    derivative_hi: Decimal | None = None
    if clob_unit_cost is not None and amm_lo is not None and amm_hi is not None:
        derivative_lo = amm_lo - clob_unit_cost
        derivative_hi = amm_hi - clob_unit_cost
        if derivative_lo <= 0 <= derivative_hi:
            role = "crossing_interval"
        elif derivative_hi < 0:
            role = "amm_cheaper_through_interval"
        elif derivative_lo > 0:
            role = "clob_cheaper_through_interval"
        else:
            role = "non_monotone_or_rounding_irregularity"
    return {
        "role": role,
        "clob_unit_cost": _decimal_text(clob_unit_cost),
        "amm_marginal_lo": _decimal_text(amm_lo),
        "amm_marginal_hi": _decimal_text(amm_hi),
        "derivative_lo": _decimal_text(derivative_lo),
        "derivative_hi": _decimal_text(derivative_hi),
    }


def _fixed_output_best_point_diagnostic(
    ctx: TxContext,
    *,
    best: dict[str, Any] | None,
    max_clob_segments: int,
    rounding_window_units: int,
) -> dict[str, Any] | None:
    if best is None or ctx.target_output is None:
        return None
    point = best.get("amm_output_units")
    if point is None:
        return None
    point = int(point)
    target_units = _amt_units_floor(ctx.target_output)
    intervals = _fixed_output_segment_intervals(ctx, max_clob_segments=max_clob_segments)
    interval_infos: list[dict[str, Any]] = []
    crossing_indices: list[int] = []
    matched: dict[str, Any] | None = None
    for index, (interval_lo, interval_hi, seg) in enumerate(intervals):
        role = _fixed_output_interval_role(ctx, interval_lo=interval_lo, interval_hi=interval_hi, seg=seg)
        info = {
            "interval_index": index,
            "interval_lo": int(interval_lo),
            "interval_hi": int(interval_hi),
            **role,
        }
        if role.get("role") == "crossing_interval":
            crossing_indices.append(index)
        interval_infos.append(info)
        if interval_lo <= point <= interval_hi and matched is None:
            distance_to_left = point - interval_lo
            distance_to_right = interval_hi - point
            matched = {
                **info,
                "point_location": _point_location_in_interval(
                    point=point,
                    interval_lo=interval_lo,
                    interval_hi=interval_hi,
                ),
                "distance_to_left_boundary_units": int(distance_to_left),
                "distance_to_right_boundary_units": int(distance_to_right),
                "nearest_boundary_distance_units": int(min(distance_to_left, distance_to_right)),
            }

    if matched is None:
        origin = "global_boundary" if point in (0, target_units) else "outside_scanned_intervals"
        return {
            "side": "output",
            "point_units": point,
            "target_units": target_units,
            "origin": origin,
            "matched_interval": None,
            "crossing_interval_indices": crossing_indices,
            "crossing_interval_count": len(crossing_indices),
            "intervals_sample": interval_infos[:10],
        }

    matched_index = int(matched["interval_index"])
    location = str(matched["point_location"])
    role = str(matched["role"])
    boundary_neighborhood_units = max(1, int(rounding_window_units) * 4)
    near_boundary = int(matched["nearest_boundary_distance_units"]) <= boundary_neighborhood_units
    if role == "crossing_interval":
        origin = "main_crossing_boundary" if "boundary" in location else "main_crossing_interior_or_neighbor"
    elif crossing_indices and any(abs(matched_index - idx) <= 1 for idx in crossing_indices) and "boundary" in location:
        origin = "adjacent_to_crossing_boundary"
    elif point in (0, target_units):
        origin = "global_boundary"
    elif near_boundary:
        origin = "monotone_boundary_neighbor"
    elif "boundary" in location:
        origin = "other_interval_boundary"
    else:
        origin = "other_interval_interior"
    return {
        "side": "output",
        "point_units": point,
        "target_units": target_units,
        "origin": origin,
        "matched_interval": matched,
        "crossing_interval_indices": crossing_indices,
        "crossing_interval_count": len(crossing_indices),
        "is_in_crossing_interval": role == "crossing_interval",
        "is_adjacent_to_crossing_interval": bool(
            crossing_indices and any(abs(matched_index - idx) <= 1 for idx in crossing_indices)
        ),
        "boundary_neighborhood_units": boundary_neighborhood_units,
        "is_near_interval_boundary": near_boundary,
        "intervals_sample": interval_infos[:10],
    }


def _fixed_output_max_source_output_units(ctx: TxContext, *, source: str) -> int:
    if ctx.target_output is None:
        return 0
    target_units = _amt_units_floor(ctx.target_output)
    if target_units <= 0:
        return 0
    out, spent = _quote_source_exact_in(ctx, source=source, input_amount=ctx.effective_input)
    if _amt_units_floor(out) <= 0 or _amt_units_floor(spent) <= 0:
        return 0
    return min(target_units, _amt_units_floor(out))


def _fixed_output_feasible_domain_units(ctx: TxContext) -> dict[str, Any]:
    if ctx.target_output is None:
        raise RuntimeError("fixed-output optimizer missing target output")
    target_units = _amt_units_floor(ctx.target_output)
    max_amm_units = _fixed_output_max_source_output_units(ctx, source="AMM")
    max_clob_units = _fixed_output_max_source_output_units(ctx, source="CLOB")
    domain_lo = max(0, target_units - max_clob_units)
    domain_hi = min(target_units, max_amm_units)
    return {
        "target_units": int(target_units),
        "domain_lo": int(domain_lo),
        "domain_hi": int(domain_hi),
        "max_amm_output_units": int(max_amm_units),
        "max_clob_output_units": int(max_clob_units),
        "feasible": domain_lo <= domain_hi,
    }


def _fixed_output_structural_anchor(
    ctx: TxContext,
    *,
    max_clob_segments: int,
) -> dict[str, Any]:
    """Locate the unique continuous optimum for fixed-output min-input."""
    if ctx.target_output is None:
        raise RuntimeError("fixed-output optimizer missing target output")
    target_units = _amt_units_floor(ctx.target_output)
    feasible_domain = _fixed_output_feasible_domain_units(ctx)
    domain_lo = int(feasible_domain["domain_lo"])
    domain_hi = int(feasible_domain["domain_hi"])
    intervals = [
        (max(interval_lo, domain_lo), min(interval_hi, domain_hi), seg)
        for interval_lo, interval_hi, seg in sorted(
            _fixed_output_segment_intervals(ctx, max_clob_segments=max_clob_segments),
            key=lambda item: item[0],
        )
        if max(interval_lo, domain_lo) <= min(interval_hi, domain_hi)
    ]
    coverage = _interval_coverage_report(
        ctx,
        side="output",
        required_units=target_units,
        max_clob_segments=max_clob_segments,
    )

    def exact_boundary(point: int, *, reason: str, boundary: str) -> dict[str, Any]:
        return {
            "method": STRUCTURAL_CERT_METHOD,
            "side": "output",
            "status": "exact",
            "reason": reason,
            "structural_kind": "allocation_boundary" if boundary in {"domain_lower", "domain_upper"} else "clob_boundary",
            "anchor_units": [int(point)],
            "boundary": boundary,
            "feasible_domain": feasible_domain,
            "domain_coverage": coverage,
        }

    if not bool(feasible_domain["feasible"]):
        return {
            "method": STRUCTURAL_CERT_METHOD,
            "side": "output",
            "status": "exact",
            "reason": "empty_source_feasible_domain",
            "structural_kind": "infeasible_domain",
            "anchor_units": [],
            "feasible_domain": feasible_domain,
            "domain_coverage": coverage,
        }
    if ctx.amm is None:
        return exact_boundary(domain_lo, reason="no_amm_available", boundary="domain_lower")
    if not intervals:
        return exact_boundary(domain_hi, reason="no_clob_interval_available", boundary="domain_upper")
    if coverage.get("complete") is False:
        return {
            "method": STRUCTURAL_CERT_METHOD,
            "side": "output",
            "status": "not_certified",
            "reason": "clob_domain_truncated_by_max_clob_segments",
            "anchor_units": [],
            "feasible_domain": feasible_domain,
            "domain_coverage": coverage,
        }

    roles: list[dict[str, Any]] = []
    for index, (interval_lo, interval_hi, seg) in enumerate(intervals):
        role = _fixed_output_interval_role(ctx, interval_lo=interval_lo, interval_hi=interval_hi, seg=seg)
        role_info = {
            "interval_index": index,
            "interval_lo": int(interval_lo),
            "interval_hi": int(interval_hi),
            **role,
        }
        roles.append(role_info)
        clob_unit_cost = _clob_input_per_output(seg)
        amm_lo = _fixed_output_amm_marginal_input_per_output_unit(
            ctx,
            output_units=interval_lo,
            side="right",
        )
        amm_hi = _fixed_output_amm_marginal_input_per_output_unit(
            ctx,
            output_units=interval_hi,
            side="left",
        )
        if clob_unit_cost is None or amm_lo is None or amm_hi is None:
            return {
                "method": STRUCTURAL_CERT_METHOD,
                "side": "output",
                "status": "not_certified",
                "reason": "missing_marginal_cost",
                "anchor_units": [],
                "feasible_domain": feasible_domain,
                "domain_coverage": coverage,
                "failed_interval": role_info,
                "roles_sample": roles[:10],
            }
        if amm_lo <= clob_unit_cost <= amm_hi:
            left = interval_lo
            right = interval_hi
            for _ in range(80):
                if right - left <= 1:
                    break
                mid = (left + right) // 2
                mid_cost = _fixed_output_amm_marginal_input_per_output_unit(
                    ctx,
                    output_units=mid,
                    side="right",
                )
                if mid_cost is None:
                    return {
                        "method": STRUCTURAL_CERT_METHOD,
                        "side": "output",
                        "status": "not_certified",
                        "reason": "missing_crossing_midpoint_cost",
                        "anchor_units": [],
                        "feasible_domain": feasible_domain,
                        "domain_coverage": coverage,
                        "failed_interval": role_info,
                        "roles_sample": roles[:10],
                    }
                if mid_cost <= clob_unit_cost:
                    left = mid
                else:
                    right = mid
            return {
                "method": STRUCTURAL_CERT_METHOD,
                "side": "output",
                "status": "exact",
                "reason": "marginal_crossing",
                "structural_kind": "amm_clob_marginal_crossing",
                "anchor_units": sorted({int(left), int(right)}),
                "crossing_interval": role_info,
                "feasible_domain": feasible_domain,
                "domain_coverage": coverage,
                "roles_sample": roles[:10],
            }
        if amm_lo > clob_unit_cost:
            return {
                "method": STRUCTURAL_CERT_METHOD,
                "side": "output",
                "status": "exact",
                "reason": "derivative_positive_at_domain_point",
                "structural_kind": "clob_offer_boundary",
                "anchor_units": [int(interval_lo)],
                "boundary": "interval_left",
                "boundary_interval": role_info,
                "feasible_domain": feasible_domain,
                "domain_coverage": coverage,
                "roles_sample": roles[:10],
            }

    return {
        "method": STRUCTURAL_CERT_METHOD,
        "side": "output",
        "status": "exact",
        "reason": "derivative_negative_through_domain",
        "structural_kind": "allocation_boundary",
        "anchor_units": [int(domain_hi)],
        "boundary": "domain_upper",
        "feasible_domain": feasible_domain,
        "domain_coverage": coverage,
        "roles_sample": roles[:10],
    }


def _candidate_set_from_fixed_output_structural_anchor(
    ctx: TxContext,
    *,
    max_clob_segments: int,
    rounding_window_units: int,
) -> tuple[set[int], dict[str, Any]]:
    if ctx.target_output is None:
        raise RuntimeError("fixed-output optimizer missing target output")
    target_units = _amt_units_floor(ctx.target_output)
    certificate = _fixed_output_structural_anchor(ctx, max_clob_segments=max_clob_segments)
    candidates: set[int] = set()
    feasible_domain = certificate.get("feasible_domain") or {}
    lower_bound = int(feasible_domain.get("domain_lo", 0))
    upper_bound = int(feasible_domain.get("domain_hi", target_units))
    for anchor in certificate.get("anchor_units") or []:
        _add_candidate(
            candidates,
            target_units,
            int(anchor),
            window=rounding_window_units,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    certificate["rounding_window_units"] = int(rounding_window_units)
    certificate["candidate_units"] = sorted(candidates)
    return candidates, certificate


def _best_valid_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if row["full_fill"]]
    if not valid:
        return None
    return max(valid, key=lambda row: Decimal(str(row["output"])))


def _least_cost_valid_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if row["full_fill"] and row.get("spent_input") is not None]
    if not valid:
        return None
    return min(valid, key=lambda row: Decimal(str(row["spent_input"])))


def _decimal_field(mapping: dict[str, Any], key: str) -> Decimal:
    value = mapping.get(key)
    if value is None:
        return Decimal(0)
    return Decimal(str(value))


def _improvement_type_from_gain_decomposition(
    *,
    found_improvement: bool,
    gain_decomposition: dict[str, Any],
) -> str:
    if not found_improvement:
        return IMPROVEMENT_NONE
    compounding_gain = _decimal_field(gain_decomposition, "amm_multi_slice_fee_compounding_gain")
    allocation_gain = _decimal_field(gain_decomposition, "allocation_optimization_gain")
    total_gain = _decimal_field(gain_decomposition, "total_gain")
    if total_gain <= 0:
        return IMPROVEMENT_NONE
    if compounding_gain > 0 and allocation_gain > 0:
        return IMPROVEMENT_ALLOCATION_AND_AMM_MULTI_SLICE_FEE_COMPOUNDING
    if compounding_gain > 0 and allocation_gain < 0:
        return IMPROVEMENT_AMM_MULTI_SLICE_FEE_COMPOUNDING_OFFSETS_ALLOCATION_LOSS
    if allocation_gain > 0 and compounding_gain < 0:
        return IMPROVEMENT_ALLOCATION_OFFSETS_AMM_MULTI_SLICE_FEE_COMPOUNDING_LOSS
    if compounding_gain > 0:
        return IMPROVEMENT_AMM_MULTI_SLICE_FEE_COMPOUNDING
    if allocation_gain > 0:
        return IMPROVEMENT_ALLOCATION_CHANGED
    return IMPROVEMENT_ALLOCATION_CHANGED


def _dec(amount: Amount) -> Decimal:
    return fit._amount_to_decimal(amount)


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, "f")


def _status_from_delta(value: Decimal | None) -> str:
    if value is None:
        return "infeasible"
    if value > 0:
        return "improved"
    if value < 0:
        return "worse"
    return "unchanged"


def _fixed_input_counterfactual_comparison(ctx: TxContext, *, best: dict[str, Any] | None) -> dict[str, Any]:
    baseline_out = _dec(ctx.baseline_out)
    if best is None or best.get("output") is None:
        return {
            "status": "infeasible",
            "baseline_input": _amount_text(ctx.baseline_in),
            "baseline_output": _amount_text(ctx.baseline_out),
            "optimized_input": None,
            "optimized_output": None,
            "delta_input": None,
            "delta_output": None,
            "delta_bps": None,
        }
    optimized_out = Decimal(str(best.get("output")))
    optimized_in = Decimal(str(best.get("spent_input") or "0"))
    baseline_in = _dec(ctx.baseline_in)
    delta_out = optimized_out - baseline_out
    delta_in = optimized_in - baseline_in
    delta_bps = None if baseline_out == 0 else delta_out / baseline_out * Decimal(10_000)
    return {
        "status": _status_from_delta(delta_out),
        "baseline_input": _amount_text(ctx.baseline_in),
        "baseline_output": _amount_text(ctx.baseline_out),
        "optimized_input": str(best.get("spent_input")),
        "optimized_output": str(best.get("output")),
        "delta_input": _decimal_text(delta_in),
        "delta_output": _decimal_text(delta_out),
        "delta_bps": _decimal_text(delta_bps),
    }


def _fixed_output_counterfactual_comparison(ctx: TxContext, *, best: dict[str, Any] | None) -> dict[str, Any]:
    baseline_in = _dec(ctx.baseline_in)
    if best is None or best.get("spent_input") is None:
        return {
            "status": "infeasible",
            "baseline_input": _amount_text(ctx.baseline_in),
            "baseline_output": _amount_text(ctx.baseline_out),
            "optimized_input": None,
            "optimized_output": None,
            "input_saved": None,
            "delta_output": None,
            "delta_bps": None,
        }
    optimized_in = Decimal(str(best.get("spent_input")))
    optimized_out = Decimal(str(best.get("output") or "0"))
    baseline_out = _dec(ctx.baseline_out)
    input_saved = baseline_in - optimized_in
    delta_out = optimized_out - baseline_out
    delta_bps = None if baseline_in == 0 else input_saved / baseline_in * Decimal(10_000)
    return {
        "status": _status_from_delta(input_saved),
        "baseline_input": _amount_text(ctx.baseline_in),
        "baseline_output": _amount_text(ctx.baseline_out),
        "optimized_input": str(best.get("spent_input")),
        "optimized_output": str(best.get("output")),
        "input_saved": _decimal_text(input_saved),
        "delta_output": _decimal_text(delta_out),
        "delta_bps": _decimal_text(delta_bps),
    }


def _best_row_full_fill(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict):
        return False
    best = row.get("best")
    return bool(isinstance(best, dict) and best.get("full_fill"))


def _bounded_improvement_type(
    *,
    resolution: str,
    selected_row: dict[str, Any] | None,
) -> str:
    if not isinstance(selected_row, dict) or not bool(selected_row.get("found_improvement")):
        return IMPROVEMENT_NONE
    selected_type = str(selected_row.get("improvement_type") or IMPROVEMENT_NONE)
    if selected_type == IMPROVEMENT_NONE:
        return IMPROVEMENT_NONE
    if resolution == "output_cap_reached":
        return f"bounded_output_cap_{selected_type}"
    return f"bounded_partial_input_{selected_type}"


def _bounded_result_from_selected_subproblem(
    ctx: TxContext,
    *,
    resolution: str,
    selected_row: dict[str, Any],
    resolved_input_cap: Amount,
    fixed_output_row: dict[str, Any] | None,
    fixed_input_row: dict[str, Any] | None,
) -> dict[str, Any]:
    cert = dict(selected_row.get("best_price_certificate") or {})
    cert["bounded_resolution"] = resolution
    cert["bounded_method"] = "baseline_semantics_primary_fill_then_best_price"
    selected_best = dict(selected_row.get("best") or {})
    comparison = selected_row.get("counterfactual_comparison")
    found_improvement = bool(selected_row.get("found_improvement"))
    gain_decomposition = dict(selected_row.get("gain_decomposition") or {})
    gain_side = str(selected_row.get("gain_side") or "bounded_fill_price")
    improvement_type = _bounded_improvement_type(
        resolution=resolution,
        selected_row=selected_row,
    )
    baseline_out_dec = _dec(ctx.baseline_out)
    optimized_out_dec = Decimal(str(selected_best.get("output") or baseline_out_dec))
    output_delta = optimized_out_dec - baseline_out_dec
    output_delta_bps = None if baseline_out_dec == 0 else output_delta / baseline_out_dec * Decimal(10_000)

    baseline_in_dec = _dec(ctx.baseline_in)
    optimized_in_dec = Decimal(str(selected_best.get("spent_input") or baseline_in_dec))
    input_delta = optimized_in_dec - baseline_in_dec

    baseline_already_reached_output_cap = (
        ctx.target_output is not None
        and _amount_eq_or_one_quantum_short(ctx.baseline_out, ctx.target_output)
    )
    if resolution == "output_cap_reached" and not baseline_already_reached_output_cap:
        status = "improved_primary_fill" if output_delta > 0 else _status_from_delta(output_delta)
        comparison = {
            "status": status,
            "baseline_input": _amount_text(ctx.baseline_in),
            "baseline_output": _amount_text(ctx.baseline_out),
            "optimized_input": str(selected_best.get("spent_input")),
            "optimized_output": str(selected_best.get("output")),
            "delta_input": _decimal_text(input_delta),
            "delta_output": _decimal_text(output_delta),
            "delta_bps": _decimal_text(output_delta_bps),
            "primary_fill_rule": "output_cap",
        }
        found_improvement = output_delta > 0
        improvement_type = "bounded_output_cap_primary_fill" if found_improvement else IMPROVEMENT_NONE
        gain_side = "output_gain"
        selected_best["improvement"] = _decimal_text(output_delta)
        selected_best["improvement_bps"] = _decimal_text(output_delta_bps)
        gain_decomposition = {
            "baseline_out": _decimal_text(baseline_out_dec),
            "baseline_allocation_one_shot_out": _decimal_text(baseline_out_dec),
            "best_out": _decimal_text(optimized_out_dec),
            "amm_multi_slice_fee_compounding_gain": "0",
            "allocation_optimization_gain": _decimal_text(output_delta),
            "total_gain": _decimal_text(output_delta),
            "amm_multi_slice_fee_compounding_gain_bps": "0",
            "allocation_optimization_gain_bps": _decimal_text(output_delta_bps),
            "total_gain_bps": _decimal_text(output_delta_bps),
        }
    return {
        "tx_hash": ctx.tx_hash,
        "ledger_index": ctx.ledger_index,
        "intent": "bounded partial",
        "optimizer_mode": ctx.optimizer_mode,
        "objective": (
            "bounded_output_cap_minimize_input"
            if resolution == "output_cap_reached"
            else "bounded_baseline_input_maximize_output"
        ),
        "gain_side": gain_side,
        "input_cap": _amount_text(ctx.effective_input),
        "resolved_input_cap": _amount_text(resolved_input_cap),
        "output_cap": _amount_text(ctx.target_output),
        "baseline": {
            "input": _amount_text(ctx.baseline_in),
            "output": _amount_text(ctx.baseline_out),
            "slice_count": len(ctx.baseline_quote.get("slices", []) or []),
        },
        "candidate_count": int(selected_row.get("candidate_count") or 0),
        "candidate_set": str(selected_row.get("candidate_set") or "theorem_driven_structural_anchor_neighborhood"),
        "baseline_incumbent_candidate_included": True,
        "best_price_certificate": cert,
        "best_point_diagnostic": selected_row.get("best_point_diagnostic"),
        "valid_full_fill_candidate_count": int(selected_row.get("valid_full_fill_candidate_count") or 0),
        "best": selected_best,
        "counterfactual_comparison": comparison,
        "bounded_resolution": {
            "method": "baseline_semantics_primary_fill_then_best_price",
            "resolution": resolution,
            "selected_objective": selected_row.get("objective"),
            "output_cap_reachable": bool(resolution == "output_cap_reached"),
            "same_definition_basis": (
                "Payment/OfferCreate buy first tries the baseline output cap; "
                "if that cap is not reachable, bounded partials preserve the "
                "baseline-resolved executed input and optimize price at that "
                "same fill definition."
            ),
            "fixed_output_subproblem": _bounded_subproblem_summary(fixed_output_row),
            "fixed_input_subproblem": _bounded_subproblem_summary(fixed_input_row),
        },
        "found_improvement": found_improvement,
        "improvement_type": improvement_type,
        "amm_one_shot_diagnostic": selected_row.get("amm_one_shot_diagnostic"),
        "gain_decomposition": gain_decomposition,
    }


def _bounded_infeasible_result(
    ctx: TxContext,
    *,
    reason: str,
    fixed_output_row: dict[str, Any] | None,
    fixed_input_row: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "tx_hash": ctx.tx_hash,
        "ledger_index": ctx.ledger_index,
        "intent": "bounded partial",
        "optimizer_mode": ctx.optimizer_mode,
        "objective": "bounded_infeasible",
        "gain_side": "none",
        "input_cap": _amount_text(ctx.effective_input),
        "output_cap": _amount_text(ctx.target_output),
        "baseline": {
            "input": _amount_text(ctx.baseline_in),
            "output": _amount_text(ctx.baseline_out),
            "slice_count": len(ctx.baseline_quote.get("slices", []) or []),
        },
        "candidate_count": 0,
        "candidate_set": "baseline_semantics_primary_fill_then_best_price",
        "best_price_certificate": {
            "method": "baseline_semantics_primary_fill_then_best_price",
            "claim": CLAIM_EXACT_INFEASIBLE,
            "reason": reason,
            "searched_interval_count": 0,
            "certified_interval_count": 0,
            "uncertified_interval_count": 0,
            "uncertified_intervals": [],
        },
        "valid_full_fill_candidate_count": 0,
        "best": None,
        "counterfactual_comparison": {
            "status": "infeasible",
            "reason": reason,
            "baseline_input": _amount_text(ctx.baseline_in),
            "baseline_output": _amount_text(ctx.baseline_out),
        },
        "bounded_resolution": {
            "method": "baseline_semantics_primary_fill_then_best_price",
            "resolution": "infeasible",
            "reason": reason,
            "fixed_output_subproblem": _bounded_subproblem_summary(fixed_output_row),
            "fixed_input_subproblem": _bounded_subproblem_summary(fixed_input_row),
        },
        "found_improvement": False,
        "improvement_type": IMPROVEMENT_NONE,
        "amm_one_shot_diagnostic": None,
        "gain_decomposition": {
            "amm_multi_slice_fee_compounding_gain": "0",
            "allocation_optimization_gain": "0",
            "total_gain": "0",
            "amm_multi_slice_fee_compounding_gain_bps": "0",
            "allocation_optimization_gain_bps": "0",
            "total_gain_bps": "0",
        },
    }


def _amm_one_shot_diagnostic(
    ctx: TxContext,
    *,
    baseline_amm: Amount,
    baseline_clob: Amount,
    best: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if ctx.amm is None:
        return None
    endpoint_rows = ctx.baseline_quote.get("endpoint_slices", []) or []
    endpoint_amm_input = _sum_source_amount_from_rows(
        endpoint_rows,
        source="AMM",
        side="in",
        proto=ctx.effective_input,
    )
    endpoint_amm_output = _sum_source_amount_from_rows(
        endpoint_rows,
        source="AMM",
        side="out",
        proto=ctx.baseline_out,
    )
    endpoint_clob_output = _sum_source_amount_from_rows(
        endpoint_rows,
        source="CLOB",
        side="out",
        proto=ctx.baseline_out,
    )
    one_shot_amm_output, one_shot_amm_spent = _quote_source_exact_in(
        ctx,
        source="AMM",
        input_amount=baseline_amm,
    )
    isolated_clob_output, isolated_clob_spent = _quote_source_exact_in(
        ctx,
        source="CLOB",
        input_amount=baseline_clob,
    )
    one_shot_total_output = amount_add_ripple_number(one_shot_amm_output, isolated_clob_output)

    amm_gain = _dec(one_shot_amm_output) - _dec(endpoint_amm_output)
    clob_delta = _dec(isolated_clob_output) - _dec(endpoint_clob_output)
    total_gain = _dec(one_shot_total_output) - _dec(ctx.baseline_out)
    best_gain = None
    residual = None
    if best is not None:
        best_gain = Decimal(str(best["improvement"]))
        residual = best_gain - total_gain

    endpoint_amm_slice_count = sum(
        1 for row in endpoint_rows if isinstance(row, dict) and str(row.get("src") or "") == "AMM"
    )
    return {
        "mechanism": "amm_fee_intermediate_compounding",
        "endpoint_amm_slice_count": endpoint_amm_slice_count,
        "endpoint_amm_input": _amount_text(endpoint_amm_input),
        "baseline_amm_input": _amount_text(baseline_amm),
        "endpoint_amm_output_multi_slice": _amount_text(endpoint_amm_output),
        "one_shot_amm_spent": _amount_text(one_shot_amm_spent),
        "one_shot_amm_output": _amount_text(one_shot_amm_output),
        "amm_one_shot_gain": _decimal_text(amm_gain),
        "endpoint_clob_output": _amount_text(endpoint_clob_output),
        "isolated_clob_spent": _amount_text(isolated_clob_spent),
        "isolated_clob_output": _amount_text(isolated_clob_output),
        "isolated_clob_delta": _decimal_text(clob_delta),
        "one_shot_total_output": _amount_text(one_shot_total_output),
        "one_shot_total_gain": _decimal_text(total_gain),
        "best_gain": _decimal_text(best_gain),
        "best_minus_one_shot_total_gain": _decimal_text(residual),
        "explained_by_amm_one_shot": bool(residual is not None and residual == 0 and clob_delta == 0),
    }


def _gain_bps(gain: Decimal, baseline_out: Amount) -> Decimal | None:
    baseline = _dec(baseline_out)
    if baseline == 0:
        return None
    return gain / baseline * Decimal(10_000)


def _gain_decomposition(
    ctx: TxContext,
    *,
    best: dict[str, Any] | None,
    amm_one_shot: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_out = _dec(ctx.baseline_out)
    if amm_one_shot is not None and amm_one_shot.get("one_shot_total_output") is not None:
        baseline_allocation_one_shot_out = Decimal(str(amm_one_shot["one_shot_total_output"]))
    else:
        baseline_allocation_one_shot_out = baseline_out

    if best is not None and best.get("output") is not None:
        best_out = Decimal(str(best["output"]))
    else:
        best_out = baseline_out

    compounding_gain = baseline_allocation_one_shot_out - baseline_out
    allocation_gain = best_out - baseline_allocation_one_shot_out
    total_gain = best_out - baseline_out

    compounding_bps = _gain_bps(compounding_gain, ctx.baseline_out)
    allocation_bps = _gain_bps(allocation_gain, ctx.baseline_out)
    total_bps = _gain_bps(total_gain, ctx.baseline_out)

    return {
        "baseline_out": _decimal_text(baseline_out),
        "baseline_allocation_one_shot_out": _decimal_text(baseline_allocation_one_shot_out),
        "best_out": _decimal_text(best_out),
        "amm_multi_slice_fee_compounding_gain": _decimal_text(compounding_gain),
        "allocation_optimization_gain": _decimal_text(allocation_gain),
        "total_gain": _decimal_text(total_gain),
        "amm_multi_slice_fee_compounding_gain_bps": _decimal_text(compounding_bps),
        "allocation_optimization_gain_bps": _decimal_text(allocation_bps),
        "total_gain_bps": _decimal_text(total_bps),
    }


def _input_savings_bps(savings: Decimal, baseline_in: Amount) -> Decimal | None:
    baseline = _dec(baseline_in)
    if baseline == 0:
        return None
    return savings / baseline * Decimal(10_000)


def _fixed_output_gain_decomposition(ctx: TxContext, *, best: dict[str, Any] | None) -> dict[str, Any]:
    baseline_in = _dec(ctx.baseline_in)
    best_in = Decimal(str(best["spent_input"])) if best is not None and best.get("spent_input") is not None else baseline_in
    total_gain = baseline_in - best_in
    total_bps = _input_savings_bps(total_gain, ctx.baseline_in)
    return {
        "gain_side": "input_saved",
        "baseline_in": _decimal_text(baseline_in),
        "baseline_allocation_one_shot_in": _decimal_text(baseline_in),
        "best_in": _decimal_text(best_in),
        "amm_multi_slice_fee_compounding_gain": _decimal_text(Decimal(0)),
        "allocation_optimization_gain": _decimal_text(total_gain),
        "total_gain": _decimal_text(total_gain),
        "amm_multi_slice_fee_compounding_gain_bps": _decimal_text(Decimal(0)),
        "allocation_optimization_gain_bps": _decimal_text(total_bps),
        "total_gain_bps": _decimal_text(total_bps),
    }


def _best_price_claim(*, exact_structural_anchor: bool, best: dict[str, Any] | None) -> str:
    if not exact_structural_anchor:
        return CLAIM_NOT_CERTIFIED
    if best is None:
        return CLAIM_EXACT_INFEASIBLE
    return CLAIM_EXACT_BEST_PRICE


def _optimize_fixed_input_tx(
    ctx: TxContext,
    *,
    max_clob_segments: int,
    rounding_window_drops: int,
) -> dict[str, Any]:
    evaluation_cache: dict[int, dict[str, Any]] = {}
    candidates, structural_certificate = _candidate_set_from_fixed_input_structural_anchor(
        ctx,
        max_clob_segments=max_clob_segments,
        rounding_window_drops=rounding_window_drops,
    )

    evaluated = []
    for drops in sorted(candidates):
        if drops not in evaluation_cache:
            evaluation_cache[drops] = _evaluate_candidate(ctx, drops)
        evaluated.append(evaluation_cache[drops])
    evaluated.append(_fixed_input_incumbent_candidate(ctx))
    valid = [row for row in evaluated if row["full_fill"]]
    best = _best_valid_candidate(evaluated)
    best_point_diagnostic = _fixed_input_best_point_diagnostic(
        ctx,
        best=best,
        max_clob_segments=max_clob_segments,
        rounding_window_units=rounding_window_drops,
    )
    counterfactual_comparison = _fixed_input_counterfactual_comparison(ctx, best=best)
    baseline_amm = _sum_source_amount(ctx.baseline_quote, source="AMM", side="in", proto=ctx.effective_input)
    baseline_clob = _sum_source_amount(ctx.baseline_quote, source="CLOB", side="in", proto=ctx.effective_input)
    baseline_improvement = Decimal(0)
    best_improvement = Decimal(str(best["improvement"])) if best else None
    found_improvement = bool(best is not None and best_improvement is not None and best_improvement > baseline_improvement)
    amm_one_shot = _amm_one_shot_diagnostic(
        ctx,
        baseline_amm=baseline_amm,
        baseline_clob=baseline_clob,
        best=best,
    )
    gain_decomposition = _gain_decomposition(
        ctx,
        best=best if found_improvement else None,
        amm_one_shot=amm_one_shot,
    )
    improvement_type = _improvement_type_from_gain_decomposition(
        found_improvement=found_improvement,
        gain_decomposition=gain_decomposition,
    )
    exact_structural_anchor = structural_certificate.get("status") == "exact"
    best_price_certificate = {
        **structural_certificate,
        "claim": _best_price_claim(exact_structural_anchor=exact_structural_anchor, best=best),
        "searched_interval_count": 0,
        "certified_interval_count": 0,
        "uncertified_interval_count": 0 if exact_structural_anchor else 1,
        "uncertified_intervals": [] if exact_structural_anchor else [structural_certificate],
    }
    return {
        "tx_hash": ctx.tx_hash,
        "ledger_index": ctx.ledger_index,
        "intent": "fixed-input",
        "optimizer_mode": ctx.optimizer_mode,
        "objective": "maximize_output",
        "gain_side": "output_gain",
        "input": _amount_text(ctx.effective_input),
        "baseline": {
            "input": _amount_text(ctx.baseline_in),
            "output": _amount_text(ctx.baseline_out),
            "amm_input": _amount_text(baseline_amm),
            "clob_input": _amount_text(baseline_clob),
            "slice_count": len(ctx.baseline_quote.get("slices", []) or []),
        },
        "candidate_count": len(evaluated),
        "candidate_set": "theorem_driven_structural_anchor_neighborhood",
        "baseline_incumbent_candidate_included": True,
        "best_price_certificate": best_price_certificate,
        "best_point_diagnostic": best_point_diagnostic,
        "valid_full_fill_candidate_count": len(valid),
        "best": best,
        "counterfactual_comparison": counterfactual_comparison,
        "found_improvement": found_improvement,
        "improvement_type": improvement_type,
        "amm_one_shot_diagnostic": amm_one_shot,
        "gain_decomposition": gain_decomposition,
    }


def _bounded_subproblem_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "objective": row.get("objective"),
        "found_improvement": bool(row.get("found_improvement")),
        "improvement_type": row.get("improvement_type"),
        "best": row.get("best"),
        "counterfactual_comparison": row.get("counterfactual_comparison"),
        "gain_decomposition": row.get("gain_decomposition"),
        "candidate_count": row.get("candidate_count"),
        "valid_full_fill_candidate_count": row.get("valid_full_fill_candidate_count"),
        "best_price_certificate": row.get("best_price_certificate"),
        "best_point_diagnostic": row.get("best_point_diagnostic"),
    }


def _anchor_certificate_status(
    *,
    anchor_name: str,
    row: dict[str, Any] | None,
) -> dict[str, Any]:
    cert = row.get("best_price_certificate") if isinstance(row, dict) else None
    if not isinstance(cert, dict):
        return {
            "anchor": anchor_name,
            "claim": "missing",
            "exact": False,
            "searched_interval_count": 0,
            "certified_interval_count": 0,
            "uncertified_interval_count": 0,
            "uncertified_intervals": [],
        }
    claim = str(cert.get("claim") or "missing")
    uncertified_count = int(cert.get("uncertified_interval_count") or 0)
    return {
        "anchor": anchor_name,
        "claim": claim,
        "exact": claim in {CLAIM_EXACT_BEST_PRICE, CLAIM_EXACT_INFEASIBLE} and uncertified_count == 0,
        "searched_interval_count": int(cert.get("searched_interval_count") or 0),
        "certified_interval_count": int(cert.get("certified_interval_count") or 0),
        "uncertified_interval_count": uncertified_count,
        "uncertified_intervals": list(cert.get("uncertified_intervals") or [])[:20],
    }


def _optimize_bounded_partial_tx(
    ctx: TxContext,
    *,
    max_clob_segments: int,
    rounding_window_drops: int,
) -> dict[str, Any]:
    fixed_input_row: dict[str, Any] | None = None
    fixed_output_row: dict[str, Any] | None = None

    if ctx.target_output is not None and _amt_units_floor(ctx.target_output) > 0:
        fixed_output_ctx = replace(
            ctx,
            optimizer_mode=MODE_FIXED_OUTPUT_MIN_INPUT,
        )
        fixed_output_row = _optimize_fixed_output_tx(
            fixed_output_ctx,
            max_clob_segments=max_clob_segments,
            rounding_window_units=rounding_window_drops,
        )
        if _best_row_full_fill(fixed_output_row):
            return _bounded_result_from_selected_subproblem(
                ctx,
                resolution="output_cap_reached",
                selected_row=fixed_output_row,
                resolved_input_cap=ctx.effective_input,
                fixed_output_row=fixed_output_row,
                fixed_input_row=None,
            )

    resolved_input = ctx.baseline_in
    if _amt_units_floor(resolved_input) <= 0:
        return _bounded_infeasible_result(
            ctx,
            reason="baseline_partial_has_no_executed_input",
            fixed_output_row=fixed_output_row,
            fixed_input_row=None,
        )

    fixed_input_ctx = replace(
        ctx,
        optimizer_mode=MODE_FIXED_INPUT_MAX_OUTPUT,
        effective_input=resolved_input,
    )
    fixed_input_row = _optimize_fixed_input_tx(
        fixed_input_ctx,
        max_clob_segments=max_clob_segments,
        rounding_window_drops=rounding_window_drops,
    )
    if not _best_row_full_fill(fixed_input_row):
        return _bounded_infeasible_result(
            ctx,
            reason="baseline_input_subproblem_has_no_full_fill_candidate",
            fixed_output_row=fixed_output_row,
            fixed_input_row=fixed_input_row,
        )

    return _bounded_result_from_selected_subproblem(
        ctx,
        resolution="baseline_partial_input",
        selected_row=fixed_input_row,
        resolved_input_cap=resolved_input,
        fixed_output_row=fixed_output_row,
        fixed_input_row=fixed_input_row,
    )


def _source_intent_family(ctx: TxContext) -> str:
    if ctx.optimizer_mode == MODE_FIXED_INPUT_MAX_OUTPUT:
        return "fixed_input"
    if ctx.optimizer_mode == MODE_FIXED_OUTPUT_MIN_INPUT:
        return "fixed_output"
    if ctx.optimizer_mode == MODE_BOUNDED_PARTIAL:
        return "bounded_partial"
    return str(ctx.optimizer_mode or "unknown")


def _with_layer1_fields(ctx: TxContext, row: dict[str, Any], *, measurement: str) -> dict[str, Any]:
    out = dict(row)
    out["analysis_layer"] = ANALYSIS_LAYER_LAYER1_SAME_FILL
    out["layer1_measurement"] = measurement
    out["source_optimizer_mode"] = ctx.optimizer_mode
    out["source_intent_family"] = _source_intent_family(ctx)
    out["layer1_definition"] = (
        "same-fill price counterfactual: keep the baseline fill side fixed, "
        "then optimize the opposite side under the same pre-state and tx constraints"
    )
    out["baseline_fit"] = dict(ctx.baseline_fit)
    return out


def _optimize_tx_layer1_same_fill(
    ctx: TxContext,
    *,
    max_clob_segments: int,
    rounding_window_drops: int,
    measurement_policy: str = LAYER1_MEASUREMENT_POLICY_NATURAL,
) -> list[dict[str, Any]]:
    if ctx.optimizer_mode == MODE_FIXED_INPUT_MAX_OUTPUT:
        rows: list[dict[str, Any]] = []
        fixed_input_ctx = replace(
            ctx,
            effective_input=ctx.baseline_in,
        )
        row = _optimize_fixed_input_tx(
            fixed_input_ctx,
            max_clob_segments=max_clob_segments,
            rounding_window_drops=rounding_window_drops,
        )
        rows.append(_with_layer1_fields(ctx, row, measurement="fixed_input_same_input"))
        if measurement_policy == LAYER1_MEASUREMENT_POLICY_BOTH and _amt_units_floor(ctx.baseline_out) > 0:
            fixed_output_ctx = replace(
                ctx,
                optimizer_mode=MODE_FIXED_OUTPUT_MIN_INPUT,
                target_output=ctx.baseline_out,
            )
            row = _optimize_fixed_output_tx(
                fixed_output_ctx,
                max_clob_segments=max_clob_segments,
                rounding_window_units=rounding_window_drops,
            )
            rows.append(_with_layer1_fields(ctx, row, measurement="fixed_input_same_output"))
        return rows

    if ctx.optimizer_mode == MODE_FIXED_OUTPUT_MIN_INPUT:
        rows = []
        if measurement_policy == LAYER1_MEASUREMENT_POLICY_BOTH and _amt_units_floor(ctx.baseline_in) > 0:
            fixed_input_ctx = replace(
                ctx,
                optimizer_mode=MODE_FIXED_INPUT_MAX_OUTPUT,
                effective_input=ctx.baseline_in,
            )
            row = _optimize_fixed_input_tx(
                fixed_input_ctx,
                max_clob_segments=max_clob_segments,
                rounding_window_drops=rounding_window_drops,
            )
            rows.append(_with_layer1_fields(ctx, row, measurement="fixed_output_same_input"))
        fixed_output_ctx = replace(
            ctx,
            target_output=ctx.baseline_out,
        )
        row = _optimize_fixed_output_tx(
            fixed_output_ctx,
            max_clob_segments=max_clob_segments,
            rounding_window_units=rounding_window_drops,
        )
        rows.append(_with_layer1_fields(ctx, row, measurement="fixed_output_same_output"))
        return rows

    if ctx.optimizer_mode != MODE_BOUNDED_PARTIAL:
        return []

    rows: list[dict[str, Any]] = []
    if _amt_units_floor(ctx.baseline_in) > 0:
        fixed_input_ctx = replace(
            ctx,
            optimizer_mode=MODE_FIXED_INPUT_MAX_OUTPUT,
            effective_input=ctx.baseline_in,
        )
        row = _optimize_fixed_input_tx(
            fixed_input_ctx,
            max_clob_segments=max_clob_segments,
            rounding_window_drops=rounding_window_drops,
        )
        rows.append(_with_layer1_fields(ctx, row, measurement="bounded_partial_same_input"))

    if _amt_units_floor(ctx.baseline_out) > 0:
        fixed_output_ctx = replace(
            ctx,
            optimizer_mode=MODE_FIXED_OUTPUT_MIN_INPUT,
            target_output=ctx.baseline_out,
        )
        row = _optimize_fixed_output_tx(
            fixed_output_ctx,
            max_clob_segments=max_clob_segments,
            rounding_window_units=rounding_window_drops,
        )
        rows.append(_with_layer1_fields(ctx, row, measurement="bounded_partial_same_output"))
    return rows


def _flatten_result_rows(values: Iterable[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, list):
            out.extend(row for row in value if isinstance(row, dict))
        elif isinstance(value, dict):
            out.append(value)
    return out


def _optimize_tx_worker(item: tuple[TxContext, dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
    ctx, opts = item
    return _optimize_tx_layer1_same_fill(
        ctx,
        max_clob_segments=int(opts["max_clob_segments"]),
        rounding_window_drops=int(opts["rounding_window_drops"]),
        measurement_policy=str(opts.get("layer1_measurement_policy") or LAYER1_MEASUREMENT_POLICY_NATURAL),
    )


def _optimize_context_batch(
    contexts: list[TxContext],
    *,
    optimize_opts: dict[str, Any],
    jobs: int,
    chunksize: int,
    executor: ProcessPoolExecutor | None = None,
) -> list[dict[str, Any]]:
    if jobs == 1 or len(contexts) <= 1:
        return _flatten_result_rows(_optimize_tx_worker((ctx, optimize_opts)) for ctx in contexts)
    owns_executor = executor is None
    if executor is None:
        executor = ProcessPoolExecutor(max_workers=jobs)
    try:
        return _flatten_result_rows(
            executor.map(
                _optimize_tx_worker,
                ((ctx, optimize_opts) for ctx in contexts),
                chunksize=chunksize,
            )
        )
    finally:
        if owns_executor:
            executor.shutdown()


def _optimize_fixed_output_tx(
    ctx: TxContext,
    *,
    max_clob_segments: int,
    rounding_window_units: int,
) -> dict[str, Any]:
    if ctx.target_output is None:
        raise RuntimeError("fixed-output optimizer missing target output")
    evaluation_cache: dict[int, dict[str, Any]] = {}
    candidates, structural_certificate = _candidate_set_from_fixed_output_structural_anchor(
        ctx,
        max_clob_segments=max_clob_segments,
        rounding_window_units=rounding_window_units,
    )

    evaluated = []
    for units in sorted(candidates):
        if units not in evaluation_cache:
            evaluation_cache[units] = _evaluate_fixed_output_candidate(ctx, units)
        evaluated.append(evaluation_cache[units])
    evaluated.append(_fixed_output_incumbent_candidate(ctx))
    valid = [row for row in evaluated if row["full_fill"]]
    best = _least_cost_valid_candidate(evaluated)
    best_point_diagnostic = _fixed_output_best_point_diagnostic(
        ctx,
        best=best,
        max_clob_segments=max_clob_segments,
        rounding_window_units=rounding_window_units,
    )
    counterfactual_comparison = _fixed_output_counterfactual_comparison(ctx, best=best)
    baseline_amm_out = _sum_source_amount(
        ctx.baseline_quote,
        source="AMM",
        side="out",
        proto=ctx.target_output,
        rows_key="endpoint_slices",
    )
    baseline_clob_out = _sum_source_amount(
        ctx.baseline_quote,
        source="CLOB",
        side="out",
        proto=ctx.target_output,
        rows_key="endpoint_slices",
    )
    best_improvement = Decimal(str(best["improvement"])) if best and best.get("improvement") is not None else None
    found_improvement = bool(best is not None and best_improvement is not None and best_improvement > 0)
    gain_decomposition = _fixed_output_gain_decomposition(ctx, best=best if found_improvement else None)
    improvement_type = _improvement_type_from_gain_decomposition(
        found_improvement=found_improvement,
        gain_decomposition=gain_decomposition,
    )
    exact_structural_anchor = structural_certificate.get("status") == "exact"
    best_price_certificate = {
        **structural_certificate,
        "claim": _best_price_claim(exact_structural_anchor=exact_structural_anchor, best=best),
        "searched_interval_count": 0,
        "certified_interval_count": 0,
        "uncertified_interval_count": 0 if exact_structural_anchor else 1,
        "uncertified_intervals": [] if exact_structural_anchor else [structural_certificate],
    }
    tx_type = str(ctx.intent.transaction_type or "").upper()
    intent_label = "OfferCreate buy fixed-output" if tx_type == "OFFERCREATE" else "Payment full fixed-output"
    return {
        "tx_hash": ctx.tx_hash,
        "ledger_index": ctx.ledger_index,
        "intent": intent_label,
        "optimizer_mode": ctx.optimizer_mode,
        "objective": "minimize_input",
        "gain_side": "input_saved",
        "input_cap": _amount_text(ctx.effective_input),
        "target_output": _amount_text(ctx.target_output),
        "baseline": {
            "input": _amount_text(ctx.baseline_in),
            "output": _amount_text(ctx.baseline_out),
            "amm_output": _amount_text(baseline_amm_out),
            "clob_output": _amount_text(baseline_clob_out),
            "slice_count": len(ctx.baseline_quote.get("slices", []) or []),
        },
        "candidate_count": len(evaluated),
        "candidate_set": "theorem_driven_structural_anchor_neighborhood",
        "baseline_incumbent_candidate_included": True,
        "best_price_certificate": best_price_certificate,
        "best_point_diagnostic": best_point_diagnostic,
        "valid_full_fill_candidate_count": len(valid),
        "best": best,
        "counterfactual_comparison": counterfactual_comparison,
        "found_improvement": found_improvement,
        "improvement_type": improvement_type,
        "amm_one_shot_diagnostic": None,
        "gain_decomposition": gain_decomposition,
    }


def _best_price_not_certified_reasons(cert: dict[str, Any]) -> list[str]:
    if not isinstance(cert, dict) or not cert:
        return ["missing_certificate"]
    reasons: list[str] = []
    if cert.get("domain_coverage_complete") is False:
        reasons.append("clob_domain_truncated")
    for interval in cert.get("uncertified_intervals") or []:
        if not isinstance(interval, dict):
            continue
        status = str(interval.get("status") or "uncertified_interval")
        reasons.append(status)
    if not reasons:
        reasons.append(str(cert.get("claim") or "not_certified"))
    return sorted(set(reasons))


def _write_summary(
    output_dir: Path,
    results: list[dict[str, Any]],
    *,
    selection_stats: dict[str, int] | None = None,
    run_config: dict[str, Any] | None = None,
) -> None:
    improved = [row for row in results if row.get("found_improvement")]
    improvement_type_counts: dict[str, int] = {}
    improvement_type_total_gain: dict[str, Decimal] = {}
    improvement_type_max_bps: dict[str, Decimal] = {}
    objective_counts: dict[str, int] = {}
    gain_side_counts: dict[str, int] = {}
    decomposition_total_gain = {
        "amm_multi_slice_fee_compounding_gain": Decimal(0),
        "allocation_optimization_gain": Decimal(0),
        "total_gain": Decimal(0),
    }
    decomposition_positive_counts = {
        "amm_multi_slice_fee_compounding_gain": 0,
        "allocation_optimization_gain": 0,
        "total_gain": 0,
    }
    decomposition_max_bps = {
        "amm_multi_slice_fee_compounding_gain_bps": Decimal(0),
        "allocation_optimization_gain_bps": Decimal(0),
        "total_gain_bps": Decimal(0),
    }
    comparison_status_counts: dict[str, int] = {}
    certified_comparison_status_counts: dict[str, int] = {}
    analysis_layer_counts: dict[str, int] = {}
    layer1_measurement_counts: dict[str, int] = {}
    source_intent_family_counts: dict[str, int] = {}
    unique_tx_hashes: set[str] = set()
    improved_tx_hashes: set[str] = set()
    for row in results:
        objective = str(row.get("objective") or "unknown")
        gain_side = str(row.get("gain_side") or "unknown")
        objective_counts[objective] = objective_counts.get(objective, 0) + 1
        gain_side_counts[gain_side] = gain_side_counts.get(gain_side, 0) + 1
        analysis_layer = str(row.get("analysis_layer") or "missing")
        measurement = str(row.get("layer1_measurement") or "single_outcome")
        family = str(row.get("source_intent_family") or row.get("optimizer_mode") or "unknown")
        analysis_layer_counts[analysis_layer] = analysis_layer_counts.get(analysis_layer, 0) + 1
        layer1_measurement_counts[measurement] = layer1_measurement_counts.get(measurement, 0) + 1
        source_intent_family_counts[family] = source_intent_family_counts.get(family, 0) + 1
        key = str(row.get("improvement_type") or "unknown")
        improvement_type_counts[key] = improvement_type_counts.get(key, 0) + 1
        status = str((row.get("counterfactual_comparison") or {}).get("status") or "missing")
        comparison_status_counts[status] = comparison_status_counts.get(status, 0) + 1
        cert_claim = str((row.get("best_price_certificate") or {}).get("claim") or "missing")
        certified_status = "not_certified" if cert_claim in {CLAIM_NOT_CERTIFIED, "missing"} else status
        certified_comparison_status_counts[certified_status] = (
            certified_comparison_status_counts.get(certified_status, 0) + 1
        )
        best = row.get("best") or {}
        gain = Decimal(str(best.get("improvement") or "0")) if row.get("found_improvement") else Decimal(0)
        bps = Decimal(str(best.get("improvement_bps") or "0")) if row.get("found_improvement") else Decimal(0)
        improvement_type_total_gain[key] = improvement_type_total_gain.get(key, Decimal(0)) + gain
        improvement_type_max_bps[key] = max(improvement_type_max_bps.get(key, Decimal(0)), bps)
        decomposition = row.get("gain_decomposition") or {}
        for gain_key in decomposition_total_gain:
            value = Decimal(str(decomposition.get(gain_key) or "0"))
            decomposition_total_gain[gain_key] += value
            if value > 0:
                decomposition_positive_counts[gain_key] += 1
        for bps_key in decomposition_max_bps:
            value = Decimal(str(decomposition.get(bps_key) or "0"))
            decomposition_max_bps[bps_key] = max(decomposition_max_bps[bps_key], value)
    amm_compounding = [
        row for row in results if row.get("improvement_type") == IMPROVEMENT_AMM_MULTI_SLICE_FEE_COMPOUNDING
    ]
    amm_compounding_explained = [
        row
        for row in amm_compounding
        if (row.get("amm_one_shot_diagnostic") or {}).get("explained_by_amm_one_shot")
    ]
    best_price_claim_counts: dict[str, int] = {}
    not_certified_reason_counts: dict[str, int] = {}
    best_price_certified_interval_count = 0
    best_price_uncertified_interval_count = 0
    for row in results:
        cert = row.get("best_price_certificate") or {}
        claim = str(cert.get("claim") or "missing")
        best_price_claim_counts[claim] = best_price_claim_counts.get(claim, 0) + 1
        best_price_certified_interval_count += int(cert.get("certified_interval_count") or 0)
        best_price_uncertified_interval_count += int(cert.get("uncertified_interval_count") or 0)
        if claim in {CLAIM_NOT_CERTIFIED, "missing"}:
            for reason in _best_price_not_certified_reasons(cert):
                not_certified_reason_counts[reason] = not_certified_reason_counts.get(reason, 0) + 1
    all_bps = [
        Decimal(str((row.get("best") or {}).get("improvement_bps") or "0"))
        if row.get("found_improvement")
        else Decimal(0)
        for row in results
    ]
    improved_bps = [
        Decimal(str((row.get("best") or {}).get("improvement_bps") or "0"))
        for row in results
        if row.get("found_improvement")
    ]
    aggregate = {
        "tx_count": len(results),
        "measurement_count": len(results),
        "unique_tx_count": len({str(row.get("tx_hash") or "") for row in results if row.get("tx_hash")}),
        "improved_tx_count": len({str(row.get("tx_hash") or "") for row in improved if row.get("tx_hash")}),
        "selection_stats": selection_stats or {},
        "run_config": run_config or {},
        "improved_count": len(improved),
        "unchanged_or_no_better_count": len(results) - len(improved),
        "comparison_status_counts": comparison_status_counts,
        "certified_comparison_status_counts": certified_comparison_status_counts,
        "improvement_type_counts": improvement_type_counts,
        "objective_counts": objective_counts,
        "gain_side_counts": gain_side_counts,
        "analysis_layer_counts": analysis_layer_counts,
        "layer1_measurement_counts": layer1_measurement_counts,
        "source_intent_family_counts": source_intent_family_counts,
        "improvement_type_total_gain": {
            key: value.to_eng_string()
            for key, value in sorted(improvement_type_total_gain.items())
        },
        "improvement_type_max_bps": {
            key: value.to_eng_string()
            for key, value in sorted(improvement_type_max_bps.items())
        },
        "gain_decomposition_total": {
            key: value.to_eng_string()
            for key, value in sorted(decomposition_total_gain.items())
        },
        "gain_decomposition_positive_counts": decomposition_positive_counts,
        "gain_decomposition_max_bps": {
            key: value.to_eng_string()
            for key, value in sorted(decomposition_max_bps.items())
        },
        "amm_multi_slice_fee_compounding_explained_count": len(amm_compounding_explained),
        "amm_multi_slice_fee_compounding_unexplained_count": len(amm_compounding) - len(amm_compounding_explained),
        "best_price_claim_counts": best_price_claim_counts,
        "not_certified_reason_counts": not_certified_reason_counts,
        "best_price_certified_interval_count": best_price_certified_interval_count,
        "best_price_uncertified_interval_count": best_price_uncertified_interval_count,
        "max_improvement_bps": max(
            all_bps,
            default=Decimal(0),
        ).to_eng_string(),
        "improvement_bps_distribution": {
            "all_txs": _bps_distribution(all_bps),
            "improved_txs": _bps_distribution(improved_bps),
        },
    }
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Best Price Intent Optimizer", ""]
    lines.append(f"- tx_count: `{aggregate['tx_count']}`")
    lines.append(f"- selection_stats: `{json.dumps(aggregate['selection_stats'], sort_keys=True)}`")
    lines.append(f"- run_config: `{json.dumps(aggregate['run_config'], sort_keys=True)}`")
    lines.append(f"- improved_count: `{aggregate['improved_count']}`")
    lines.append(f"- unique_tx_count: `{aggregate['unique_tx_count']}`")
    lines.append(f"- improved_tx_count: `{aggregate['improved_tx_count']}`")
    lines.append(f"- comparison_status_counts: `{json.dumps(comparison_status_counts, sort_keys=True)}`")
    lines.append(
        f"- certified_comparison_status_counts: "
        f"`{json.dumps(certified_comparison_status_counts, sort_keys=True)}`"
    )
    lines.append(f"- improvement_type_counts: `{json.dumps(improvement_type_counts, sort_keys=True)}`")
    lines.append(f"- objective_counts: `{json.dumps(objective_counts, sort_keys=True)}`")
    lines.append(f"- gain_side_counts: `{json.dumps(gain_side_counts, sort_keys=True)}`")
    lines.append(f"- analysis_layer_counts: `{json.dumps(analysis_layer_counts, sort_keys=True)}`")
    lines.append(f"- layer1_measurement_counts: `{json.dumps(layer1_measurement_counts, sort_keys=True)}`")
    lines.append(f"- source_intent_family_counts: `{json.dumps(source_intent_family_counts, sort_keys=True)}`")
    lines.append(f"- improvement_type_total_gain: `{json.dumps(aggregate['improvement_type_total_gain'], sort_keys=True)}`")
    lines.append(f"- improvement_type_max_bps: `{json.dumps(aggregate['improvement_type_max_bps'], sort_keys=True)}`")
    lines.append(f"- gain_decomposition_total: `{json.dumps(aggregate['gain_decomposition_total'], sort_keys=True)}`")
    lines.append(
        f"- gain_decomposition_positive_counts: "
        f"`{json.dumps(aggregate['gain_decomposition_positive_counts'], sort_keys=True)}`"
    )
    lines.append(f"- gain_decomposition_max_bps: `{json.dumps(aggregate['gain_decomposition_max_bps'], sort_keys=True)}`")
    lines.append(
        f"- amm_multi_slice_fee_compounding_explained: "
        f"`{aggregate['amm_multi_slice_fee_compounding_explained_count']} / "
        f"{improvement_type_counts.get(IMPROVEMENT_AMM_MULTI_SLICE_FEE_COMPOUNDING, 0)}`"
    )
    lines.append(f"- best_price_claim_counts: `{json.dumps(best_price_claim_counts, sort_keys=True)}`")
    lines.append(f"- not_certified_reason_counts: `{json.dumps(not_certified_reason_counts, sort_keys=True)}`")
    lines.append(
        f"- best_price_intervals: `certified={aggregate['best_price_certified_interval_count']}, "
        f"uncertified={aggregate['best_price_uncertified_interval_count']}`"
    )
    lines.append(f"- max_improvement_bps: `{aggregate['max_improvement_bps']}`")
    lines.append(
        f"- improvement_bps_distribution: "
        f"`{json.dumps(aggregate['improvement_bps_distribution'], sort_keys=True)}`"
    )
    lines.append("")
    lines.append("| tx | objective | type | baseline | best | AMM/CLOB baseline -> best | compounding gain | allocation gain | total gain bps | candidates | best-price certificate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in results:
        baseline = row["baseline"]
        best = row.get("best") or {}
        decomposition = row.get("gain_decomposition") or {}
        certificate = row.get("best_price_certificate") or {}
        certificate_text = (
            f"{certificate.get('claim')} "
            f"({certificate.get('certified_interval_count')}/"
            f"{certificate.get('searched_interval_count')} intervals)"
            if certificate
            else "n/a"
        )
        if row.get("objective") == "minimize_input":
            baseline_value = decomposition.get("baseline_in")
            best_value = decomposition.get("best_in")
            baseline_alloc = f"{baseline.get('amm_output')} / {baseline.get('clob_output')}"
            best_alloc = f"{best.get('amm_output')} / {best.get('clob_output')}"
        else:
            baseline_value = decomposition.get("baseline_out")
            best_value = decomposition.get("best_out")
            baseline_alloc = f"{baseline.get('amm_input')} / {baseline.get('clob_input')}"
            best_alloc = f"{best.get('amm_input')} / {best.get('clob_input')}"
        lines.append(
            f"| `{row['tx_hash'][:8]}` | {row.get('objective')} | {row.get('improvement_type')} | "
            f"{baseline_value} | "
            f"{best_value} | "
            f"{baseline_alloc} -> {best_alloc} | "
            f"{decomposition.get('amm_multi_slice_fee_compounding_gain')} | "
            f"{decomposition.get('allocation_optimization_gain')} | "
            f"{decomposition.get('total_gain_bps')} | "
            f"{row['candidate_count']} | "
            f"{certificate_text} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_stream_summary(
    output_dir: Path,
    *,
    results_ndjson: Path,
    selection_stats: dict[str, Any] | None = None,
    run_config: dict[str, Any] | None = None,
    top_n: int = 50,
) -> None:
    tx_count = 0
    improved_count = 0
    improvement_type_counts: dict[str, int] = {}
    improvement_type_total_gain: dict[str, Decimal] = {}
    improvement_type_max_bps: dict[str, Decimal] = {}
    objective_counts: dict[str, int] = {}
    gain_side_counts: dict[str, int] = {}
    decomposition_total_gain = {
        "amm_multi_slice_fee_compounding_gain": Decimal(0),
        "allocation_optimization_gain": Decimal(0),
        "total_gain": Decimal(0),
    }
    decomposition_positive_counts = {
        "amm_multi_slice_fee_compounding_gain": 0,
        "allocation_optimization_gain": 0,
        "total_gain": 0,
    }
    decomposition_max_bps = {
        "amm_multi_slice_fee_compounding_gain_bps": Decimal(0),
        "allocation_optimization_gain_bps": Decimal(0),
        "total_gain_bps": Decimal(0),
    }
    best_price_claim_counts: dict[str, int] = {}
    not_certified_reason_counts: dict[str, int] = {}
    comparison_status_counts: dict[str, int] = {}
    certified_comparison_status_counts: dict[str, int] = {}
    analysis_layer_counts: dict[str, int] = {}
    layer1_measurement_counts: dict[str, int] = {}
    source_intent_family_counts: dict[str, int] = {}
    unique_tx_hashes: set[str] = set()
    improved_tx_hashes: set[str] = set()
    best_price_certified_interval_count = 0
    best_price_uncertified_interval_count = 0
    amm_compounding_count = 0
    amm_compounding_explained_count = 0
    max_improvement_bps = Decimal(0)
    top_rows: list[tuple[Decimal, dict[str, Any]]] = []
    all_bps: list[Decimal] = []
    improved_bps: list[Decimal] = []

    with results_ndjson.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            tx_count += 1
            tx_hash = str(row.get("tx_hash") or "")
            if tx_hash:
                unique_tx_hashes.add(tx_hash)
            objective = str(row.get("objective") or "unknown")
            gain_side = str(row.get("gain_side") or "unknown")
            objective_counts[objective] = objective_counts.get(objective, 0) + 1
            gain_side_counts[gain_side] = gain_side_counts.get(gain_side, 0) + 1
            analysis_layer = str(row.get("analysis_layer") or "missing")
            measurement = str(row.get("layer1_measurement") or "single_outcome")
            family = str(row.get("source_intent_family") or row.get("optimizer_mode") or "unknown")
            analysis_layer_counts[analysis_layer] = analysis_layer_counts.get(analysis_layer, 0) + 1
            layer1_measurement_counts[measurement] = layer1_measurement_counts.get(measurement, 0) + 1
            source_intent_family_counts[family] = source_intent_family_counts.get(family, 0) + 1
            key = str(row.get("improvement_type") or "unknown")
            improvement_type_counts[key] = improvement_type_counts.get(key, 0) + 1
            status = str((row.get("counterfactual_comparison") or {}).get("status") or "missing")
            comparison_status_counts[status] = comparison_status_counts.get(status, 0) + 1
            best = row.get("best") or {}
            found = bool(row.get("found_improvement"))
            if found:
                improved_count += 1
                if tx_hash:
                    improved_tx_hashes.add(tx_hash)
            gain = Decimal(str(best.get("improvement") or "0")) if found else Decimal(0)
            bps = Decimal(str(best.get("improvement_bps") or "0")) if found else Decimal(0)
            all_bps.append(bps)
            improvement_type_total_gain[key] = improvement_type_total_gain.get(key, Decimal(0)) + gain
            improvement_type_max_bps[key] = max(improvement_type_max_bps.get(key, Decimal(0)), bps)
            max_improvement_bps = max(max_improvement_bps, bps)
            if found:
                improved_bps.append(bps)
                top_rows.append((bps, row))
                top_rows.sort(key=lambda item: item[0], reverse=True)
                if len(top_rows) > top_n:
                    top_rows.pop()
            decomposition = row.get("gain_decomposition") or {}
            for gain_key in decomposition_total_gain:
                value = Decimal(str(decomposition.get(gain_key) or "0"))
                decomposition_total_gain[gain_key] += value
                if value > 0:
                    decomposition_positive_counts[gain_key] += 1
            for bps_key in decomposition_max_bps:
                value = Decimal(str(decomposition.get(bps_key) or "0"))
                decomposition_max_bps[bps_key] = max(decomposition_max_bps[bps_key], value)
            if key == IMPROVEMENT_AMM_MULTI_SLICE_FEE_COMPOUNDING:
                amm_compounding_count += 1
                if (row.get("amm_one_shot_diagnostic") or {}).get("explained_by_amm_one_shot"):
                    amm_compounding_explained_count += 1
            cert = row.get("best_price_certificate") or {}
            claim = str(cert.get("claim") or "missing")
            best_price_claim_counts[claim] = best_price_claim_counts.get(claim, 0) + 1
            certified_status = "not_certified" if claim in {CLAIM_NOT_CERTIFIED, "missing"} else status
            certified_comparison_status_counts[certified_status] = (
                certified_comparison_status_counts.get(certified_status, 0) + 1
            )
            best_price_certified_interval_count += int(cert.get("certified_interval_count") or 0)
            best_price_uncertified_interval_count += int(cert.get("uncertified_interval_count") or 0)
            if claim in {CLAIM_NOT_CERTIFIED, "missing"}:
                for reason in _best_price_not_certified_reasons(cert):
                    not_certified_reason_counts[reason] = not_certified_reason_counts.get(reason, 0) + 1

    aggregate = {
        "tx_count": tx_count,
        "measurement_count": tx_count,
        "unique_tx_count": len(unique_tx_hashes),
        "improved_tx_count": len(improved_tx_hashes),
        "selection_stats": selection_stats or {},
        "run_config": run_config or {},
        "results_format": "ndjson",
        "results_ndjson": str(results_ndjson),
        "improved_count": improved_count,
        "unchanged_or_no_better_count": tx_count - improved_count,
        "comparison_status_counts": comparison_status_counts,
        "certified_comparison_status_counts": certified_comparison_status_counts,
        "improvement_type_counts": improvement_type_counts,
        "objective_counts": objective_counts,
        "gain_side_counts": gain_side_counts,
        "analysis_layer_counts": analysis_layer_counts,
        "layer1_measurement_counts": layer1_measurement_counts,
        "source_intent_family_counts": source_intent_family_counts,
        "improvement_type_total_gain": {
            key: value.to_eng_string()
            for key, value in sorted(improvement_type_total_gain.items())
        },
        "improvement_type_max_bps": {
            key: value.to_eng_string()
            for key, value in sorted(improvement_type_max_bps.items())
        },
        "gain_decomposition_total": {
            key: value.to_eng_string()
            for key, value in sorted(decomposition_total_gain.items())
        },
        "gain_decomposition_positive_counts": decomposition_positive_counts,
        "gain_decomposition_max_bps": {
            key: value.to_eng_string()
            for key, value in sorted(decomposition_max_bps.items())
        },
        "amm_multi_slice_fee_compounding_explained_count": amm_compounding_explained_count,
        "amm_multi_slice_fee_compounding_unexplained_count": amm_compounding_count - amm_compounding_explained_count,
        "best_price_claim_counts": best_price_claim_counts,
        "not_certified_reason_counts": not_certified_reason_counts,
        "best_price_certified_interval_count": best_price_certified_interval_count,
        "best_price_uncertified_interval_count": best_price_uncertified_interval_count,
        "max_improvement_bps": max_improvement_bps.to_eng_string(),
        "improvement_bps_distribution": {
            "all_txs": _bps_distribution(all_bps),
            "improved_txs": _bps_distribution(improved_bps),
        },
    }
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Best Price Intent Optimizer", ""]
    for key in [
        "tx_count",
        "improved_count",
        "unchanged_or_no_better_count",
        "max_improvement_bps",
    ]:
        lines.append(f"- {key}: `{aggregate[key]}`")
    lines.append(f"- unique_tx_count: `{aggregate['unique_tx_count']}`")
    lines.append(f"- improved_tx_count: `{aggregate['improved_tx_count']}`")
    lines.append(f"- selection_stats: `{json.dumps(aggregate['selection_stats'], sort_keys=True)}`")
    lines.append(f"- run_config: `{json.dumps(aggregate['run_config'], sort_keys=True)}`")
    lines.append(f"- objective_counts: `{json.dumps(objective_counts, sort_keys=True)}`")
    lines.append(f"- analysis_layer_counts: `{json.dumps(analysis_layer_counts, sort_keys=True)}`")
    lines.append(f"- layer1_measurement_counts: `{json.dumps(layer1_measurement_counts, sort_keys=True)}`")
    lines.append(f"- source_intent_family_counts: `{json.dumps(source_intent_family_counts, sort_keys=True)}`")
    lines.append(f"- improvement_type_counts: `{json.dumps(improvement_type_counts, sort_keys=True)}`")
    lines.append(f"- comparison_status_counts: `{json.dumps(comparison_status_counts, sort_keys=True)}`")
    lines.append(
        f"- certified_comparison_status_counts: "
        f"`{json.dumps(certified_comparison_status_counts, sort_keys=True)}`"
    )
    lines.append(f"- best_price_claim_counts: `{json.dumps(best_price_claim_counts, sort_keys=True)}`")
    lines.append(f"- not_certified_reason_counts: `{json.dumps(not_certified_reason_counts, sort_keys=True)}`")
    lines.append(
        f"- improvement_bps_distribution: "
        f"`{json.dumps(aggregate['improvement_bps_distribution'], sort_keys=True)}`"
    )
    lines.append(f"- gain_decomposition_total: `{json.dumps(aggregate['gain_decomposition_total'], sort_keys=True)}`")
    lines.append("")
    lines.append(f"## Top {len(top_rows)} Improvements By Bps")
    lines.append("")
    lines.append("| tx | objective | type | gain | bps | certificate |")
    lines.append("|---|---|---|---:|---:|---|")
    for bps, row in top_rows:
        cert = row.get("best_price_certificate") or {}
        best = row.get("best") or {}
        lines.append(
            f"| `{str(row.get('tx_hash'))[:8]}` | {row.get('objective')} | {row.get('improvement_type')} | "
            f"{best.get('improvement')} | {bps.to_eng_string()} | {cert.get('claim')} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    dataset = _load_dataset(Path(args.dataset_manifest))
    if bool(args.stream_required):
        if not bool(args.sample_from_required):
            raise RuntimeError("--stream-required requires --sample-from-required")
        if args.tx_hash or args.tx_hash_file:
            raise RuntimeError("--stream-required does not support explicit --tx-hash targets")
        _run_stream_required(args=args, dataset=dataset)
        return

    explicit_hashes = list(args.tx_hash)
    for hash_file in args.tx_hash_file:
        explicit_hashes.extend(_read_hash_file(Path(hash_file)))
    selected, samples, tick_size_snapshots, selection_stats = _select_optimizer_targets(
        dataset=dataset,
        explicit_hashes=explicit_hashes,
        sample_from_required=bool(args.sample_from_required),
        candidate_pool_size=max(0, int(args.candidate_pool_size)),
        max_txs=max(0, int(args.max_txs)),
        optimizer_mode_filter=str(args.optimizer_mode),
    )
    max_clob_segments = int(args.max_clob_segments)
    if max_clob_segments <= 0:
        max_clob_segments = 1_000_000_000
    optimize_opts = {
        "max_clob_segments": max_clob_segments,
        "rounding_window_drops": max(0, int(args.rounding_window_drops)),
        "layer1_measurement_policy": str(args.layer1_measurement_policy),
    }
    jobs = max(1, int(args.jobs))
    chunksize = max(1, int(args.task_chunksize))
    context_batch_size = max(1, int(args.context_batch_size))
    results: list[dict[str, Any]] = []
    for offset in range(0, len(selected), context_batch_size):
        batch_hashes = selected[offset : offset + context_batch_size]
        contexts = _build_contexts(
            dataset=dataset,
            selected_hashes=batch_hashes,
            samples=samples,
            tick_size_snapshots=tick_size_snapshots,
            require_strict_baseline_fit=True,
        )
        if len(contexts) < len(batch_hashes):
            selection_stats["context_filtered_count"] += len(batch_hashes) - len(contexts)
        if jobs == 1 or len(contexts) <= 1:
            results.extend(_flatten_result_rows(_optimize_tx_worker((ctx, optimize_opts)) for ctx in contexts))
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                results.extend(
                    _flatten_result_rows(
                        executor.map(
                            _optimize_tx_worker,
                            ((ctx, optimize_opts) for ctx in contexts),
                            chunksize=chunksize,
                        )
                    )
                )
        print(
            json.dumps(
                {
                    "batch_done": offset + len(batch_hashes),
                    "selected_count": len(selected),
                    "contexts": len(contexts),
                    "results": len(results),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "dataset_manifest": str(Path(args.dataset_manifest)),
        "optimizer_mode": str(args.optimizer_mode),
        "analysis_layer": ANALYSIS_LAYER_LAYER1_SAME_FILL,
        "layer1_measurement_policy": str(args.layer1_measurement_policy),
        "max_clob_segments": optimize_opts["max_clob_segments"],
        "rounding_window_drops": optimize_opts["rounding_window_drops"],
        "jobs": jobs,
        "task_chunksize": chunksize,
        "context_batch_size": context_batch_size,
    }
    _write_summary(output_dir, results, selection_stats=selection_stats, run_config=run_config)
    print(json.dumps(json.loads((output_dir / "aggregate.json").read_text()), ensure_ascii=False, indent=2))


def _run_stream_required(*, args: argparse.Namespace, dataset: PairDataset) -> None:
    fit.TARGET_IOU_CURRENCY = dataset.base_currency
    fit.TARGET_IOU_LABEL = dataset.base_label

    required_order = _read_required_hashes(
        dataset.required_tx_csv,
        limit=max(0, int(args.candidate_pool_size)),
    )
    max_txs = max(0, int(args.max_txs))
    optimizer_mode_filter = str(args.optimizer_mode)
    tick_size_snapshots = fit._load_offer_tick_size_snapshots(dataset.metadata_ndjson)
    max_clob_segments = int(args.max_clob_segments)
    if max_clob_segments <= 0:
        max_clob_segments = 1_000_000_000
    optimize_opts = {
        "max_clob_segments": max_clob_segments,
        "rounding_window_drops": max(0, int(args.rounding_window_drops)),
        "layer1_measurement_policy": str(args.layer1_measurement_policy),
    }
    jobs = max(1, int(args.jobs))
    chunksize = max(1, int(args.task_chunksize))
    context_batch_size = max(1, int(args.context_batch_size))

    selection_stats: dict[str, int] = {
        "required_tx_count": len(required_order),
        "stream_required": 1,
        "explicit_supported_count": 0,
        "unsupported_count": 0,
        "context_filtered_count": 0,
        "selected_count": 0,
    }
    intent_mode_counts: dict[str, int] = {}
    tx_type_counts: dict[str, int] = {}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_stream = SnapshotSideStream(dataset.tx_prebook_snapshots)
    executor: ProcessPoolExecutor | None = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    results_ndjson = output_dir / "results.ndjson"
    results_handle = results_ndjson.open("w", encoding="utf-8")
    result_count = 0
    batch_hashes: list[str] = []
    batch_samples: dict[str, Any] = {}
    batch_snapshots: dict[str, dict[str, Any]] = {}

    def flush_batch(*, final: bool = False) -> None:
        nonlocal batch_hashes, batch_samples, batch_snapshots, result_count
        if not batch_hashes:
            return
        contexts = _build_contexts(
            dataset=dataset,
            selected_hashes=batch_hashes,
            samples=batch_samples,
            tick_size_snapshots=tick_size_snapshots,
            snapshot_payloads_by_hash=batch_snapshots,
            require_strict_baseline_fit=True,
        )
        if len(contexts) < len(batch_hashes):
            selection_stats["context_filtered_count"] += len(batch_hashes) - len(contexts)
        batch_results = _optimize_context_batch(
            contexts,
            optimize_opts=optimize_opts,
            jobs=jobs,
            chunksize=chunksize,
            executor=executor,
        )
        for row in batch_results:
            results_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        results_handle.flush()
        result_count += len(batch_results)
        print(
            json.dumps(
                {
                    "stream_batch_done": selection_stats["selected_count"],
                    "required_tx_count": len(required_order),
                    "batch_contexts": len(contexts),
                    "results": result_count,
                    "snapshot_line": snapshot_stream.line_no,
                    "snapshot_buffered": len(snapshot_stream.buffered),
                    "snapshot_max_buffered": snapshot_stream.max_buffered,
                    "final": final,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        batch_hashes = []
        batch_samples = {}
        batch_snapshots = {}

    try:
        for tx_hash, sample, meta_stats in _iter_required_metadata_samples_in_order(
            metadata_path=dataset.metadata_ndjson,
            required_order=required_order,
        ):
            intent = _normalised_intent(sample, tick_size_snapshots)
            tx_type = str(intent.transaction_type or "").upper()
            tx_type_counts[tx_type or "unknown"] = tx_type_counts.get(tx_type or "unknown", 0) + 1
            optimizer_mode = _optimizer_mode_for_intent(intent)
            if optimizer_mode is None:
                selection_stats["unsupported_count"] += 1
                intent_mode_counts["unsupported"] = intent_mode_counts.get("unsupported", 0) + 1
                continue
            intent_mode_counts[optimizer_mode] = intent_mode_counts.get(optimizer_mode, 0) + 1
            if optimizer_mode_filter != "all" and optimizer_mode != optimizer_mode_filter:
                continue

            side = fit._prebook_primary_side_for_output_currency(intent.out_cur)
            snapshot_payload = snapshot_stream.next_for(tx_hash=tx_hash, side=side)
            batch_hashes.append(tx_hash)
            batch_samples[tx_hash] = sample
            batch_snapshots[tx_hash] = snapshot_payload
            selection_stats["selected_count"] += 1
            if len(batch_hashes) >= context_batch_size:
                flush_batch()
            if max_txs > 0 and selection_stats["selected_count"] >= max_txs:
                break
        flush_batch(final=True)
    finally:
        snapshot_stream.close()
        results_handle.close()
        if executor is not None:
            executor.shutdown()

    selection_stats["intent_mode_counts"] = intent_mode_counts  # type: ignore[assignment]
    selection_stats["tx_type_counts"] = tx_type_counts  # type: ignore[assignment]
    run_config = {
        "dataset_manifest": str(Path(args.dataset_manifest)),
        "optimizer_mode": str(args.optimizer_mode),
        "analysis_layer": ANALYSIS_LAYER_LAYER1_SAME_FILL,
        "layer1_measurement_policy": str(args.layer1_measurement_policy),
        "max_clob_segments": optimize_opts["max_clob_segments"],
        "rounding_window_drops": optimize_opts["rounding_window_drops"],
        "jobs": jobs,
        "task_chunksize": chunksize,
        "context_batch_size": context_batch_size,
        "stream_required": True,
    }
    _write_stream_summary(
        output_dir,
        results_ndjson=results_ndjson,
        selection_stats=selection_stats,
        run_config=run_config,
    )
    print(json.dumps(json.loads((output_dir / "aggregate.json").read_text()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
