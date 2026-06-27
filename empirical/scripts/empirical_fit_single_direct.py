#!/usr/bin/env python3
"""Replay isolated direct-pair single transactions against the rebuilt model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from xrpl_router.amm import (
    AMM,
    amount_add_ripple_number,
    amount_sub_ripple_number,
)
from xrpl_router.book_offers import build_clob_segments_from_offers
from xrpl_router.core import Amount, IOUAmount, Quality, XRPAmount, parse_iou_decimal, parse_xrp_decimal, parse_xrp_drops
from xrpl_router.execution_semantics import execute_single_path_intent
from xrpl_router.tx_intent import (
    TxIntent,
    apply_offer_tick_size_to_intent,
    build_intent_from_metadata_result,
    extract_tx_hash_from_metadata_obj,
)

RUSD_HEX = "524C555344000000000000000000000000000000"
XRP = "XRP"
TARGET_IOU_CURRENCY = RUSD_HEX
TARGET_IOU_LABEL = "rUSD"
CURATED_SAMPLES_PATH = Path("empirical/single_direct/curated_samples.txt")
DEFAULT_ROOT = Path(".tmp/rolling_two_week_top100_direct_latest")

FIT_BUCKET_STRICT_EXACT = "strict_exact_match"
FIT_BUCKET_STRICT_FAILED_BUT_STRUCTURE_MATCHED = "strict_failed_but_structure_matched"
FIT_BUCKET_STRUCTURE_MISMATCH = "structure_mismatch"

STATE_CHANGE_DECREASE = "decrease"
STATE_CHANGE_INCREASE = "increase"


@dataclass(frozen=True, slots=True)
class SampleInput:
    tx_hash: str
    ledger_index: int
    transaction_index: int
    result: dict[str, Any]
    intent: TxIntent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit isolated direct-pair samples against the rebuilt single-path model."
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Dataset root containing metadata / tx_prebook / amm / clob cut files.",
    )
    parser.add_argument("--metadata-ndjson", default=None)
    parser.add_argument("--tx-prebook-snapshots", default=None)
    parser.add_argument(
        "--account-offers-snapshots",
        default=None,
        help=(
            "Optional NDJSON of account_offers snapshots keyed by (ledger_index, account). "
            "Used to reconstruct cross-book owner-funds occupancy."
        ),
    )
    parser.add_argument("--amm-swaps", default=None)
    parser.add_argument("--amm-fees", default=None)
    parser.add_argument(
        "--target-tx-file",
        default=None,
        help=(
            "Optional explicit target tx file (.csv/.parquet/.txt). "
            "If set, tx hashes are loaded from transaction_hash/tx_hash."
        ),
    )
    parser.add_argument(
        "--samples-file",
        default=str(CURATED_SAMPLES_PATH),
        help="Plain-text file of tx hashes to replay.",
    )
    parser.add_argument(
        "--tx-hash",
        action="append",
        default=[],
        help="Replay an explicit tx hash. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for reports. Defaults under .tmp/single_direct_fit_latest.",
    )
    parser.add_argument(
        "--issuer-transfer-rates-json",
        required=True,
        help=(
            "JSON containing issuer_transfer_rates keyed as '<currency>:<issuer>' with raw "
            "XRPL AccountRoot.TransferRate integers. Missing non-XRP issuer rates fail fast."
        ),
    )
    parser.add_argument(
        "--base-currency",
        default=RUSD_HEX,
        help="Issued asset currency for the direct pair. Defaults to RUSD hex.",
    )
    parser.add_argument(
        "--base-label",
        default="rUSD",
        help="Human-readable label for the issued asset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples after loading target tx hashes.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only write aggregate/manifest outputs; skip per-tx reports and large summaries.",
    )
    parser.add_argument(
        "--write-structure-false-reports",
        action="store_true",
        help="With --aggregate-only, still write reports for structure-mismatch transactions.",
    )
    parser.add_argument(
        "--write-strict-false-reports",
        action="store_true",
        help="With --aggregate-only, still write reports for non-strict transactions.",
    )
    parser.add_argument(
        "--top-relative-errors",
        type=int,
        default=0,
        help="Keep the top N structure-matched non-exact transactions by max-side relative error.",
    )
    return parser.parse_args()


def _normalise_tx_hash(value: str) -> str:
    return str(value or "").strip().upper()


def _is_xrp(currency: str) -> bool:
    return str(currency) == XRP


def _is_direct_pair(currency_a: str, currency_b: str) -> bool:
    return {str(currency_a), str(currency_b)} == {XRP, TARGET_IOU_CURRENCY}


def _pretty_currency(currency: str) -> str:
    if str(currency) == TARGET_IOU_CURRENCY:
        return TARGET_IOU_LABEL
    return str(currency)


def _amount_to_decimal(amount: Amount) -> Decimal:
    if isinstance(amount, XRPAmount):
        return Decimal(amount.drops) / Decimal(1_000_000)
    return Decimal(amount.mantissa) * (Decimal(10) ** amount.exponent)


def _amount_to_string(amount: Amount) -> str:
    return format(_amount_to_decimal(amount), "f")


def _compact_decimal_text(value: str) -> str:
    try:
        dec = Decimal(str(value))
    except Exception:
        return str(value)
    if dec == 0:
        return "0"
    text = format(dec, "f")
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".")


def _relative_error_decimal(*, model_value: Decimal, real_value: Decimal) -> Decimal | None:
    abs_real = abs(real_value)
    if abs_real == 0:
        return Decimal(0) if model_value == 0 else None
    return abs(model_value - real_value) / abs_real


def _signed_relative_error_decimal(*, model_value: Decimal, real_value: Decimal) -> Decimal | None:
    abs_real = abs(real_value)
    if abs_real == 0:
        return Decimal(0) if model_value == 0 else None
    return (model_value - real_value) / abs_real


def _relative_error_text(*, model_value: Decimal, real_value: Decimal) -> str | None:
    rel = _relative_error_decimal(model_value=model_value, real_value=real_value)
    if rel is None:
        return None
    return _compact_decimal_text(format(rel, "f"))


def _relative_error_bps_text(*, model_value: Decimal, real_value: Decimal) -> str | None:
    rel = _relative_error_decimal(model_value=model_value, real_value=real_value)
    if rel is None:
        return None
    return _compact_decimal_text(format(rel * Decimal(10_000), "f"))


REL_ERR_BUCKETS: tuple[tuple[str, Decimal], ...] = (
    ("=0", Decimal(0)),
    ("<= 1e-15", Decimal("1e-15")),
    ("<= 1e-14", Decimal("1e-14")),
    ("<= 1e-13", Decimal("1e-13")),
    ("<= 1e-12", Decimal("1e-12")),
    ("<= 1e-9", Decimal("1e-9")),
)
REL_ERR_BUCKET_FALLBACK = "> 1e-9"
REL_ERR_BUCKET_UNDEFINED = "undefined(real=0, model!=0)"


def _relative_error_bucket(value: Decimal | None) -> str:
    if value is None:
        return REL_ERR_BUCKET_UNDEFINED
    if value == 0:
        return "=0"
    for label, upper_bound in REL_ERR_BUCKETS[1:]:
        if value <= upper_bound:
            return label
    return REL_ERR_BUCKET_FALLBACK


def _ordered_bucket_counts(counter: Counter[str]) -> dict[str, int]:
    ordered: dict[str, int] = {}
    for label, _ in REL_ERR_BUCKETS:
        if counter.get(label):
            ordered[label] = int(counter[label])
    if counter.get(REL_ERR_BUCKET_FALLBACK):
        ordered[REL_ERR_BUCKET_FALLBACK] = int(counter[REL_ERR_BUCKET_FALLBACK])
    if counter.get(REL_ERR_BUCKET_UNDEFINED):
        ordered[REL_ERR_BUCKET_UNDEFINED] = int(counter[REL_ERR_BUCKET_UNDEFINED])
    return ordered


def _max_defined_decimal(values: Iterable[Decimal | None]) -> Decimal | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return max(defined)


def _relative_error_fields_from_decimals(
    *,
    model_total_out_decimal: Decimal,
    model_total_in_decimal: Decimal,
    real_total_out_decimal: Decimal,
    real_total_in_decimal: Decimal,
) -> dict[str, Any]:
    total_out_rel_err_decimal = _relative_error_decimal(
        model_value=model_total_out_decimal,
        real_value=real_total_out_decimal,
    )
    total_in_rel_err_decimal = _relative_error_decimal(
        model_value=model_total_in_decimal,
        real_value=real_total_in_decimal,
    )
    max_side_rel_err_decimal = _max_defined_decimal(
        [total_out_rel_err_decimal, total_in_rel_err_decimal]
    )
    total_out_signed_rel_err_decimal = _signed_relative_error_decimal(
        model_value=model_total_out_decimal,
        real_value=real_total_out_decimal,
    )
    total_in_signed_rel_err_decimal = _signed_relative_error_decimal(
        model_value=model_total_in_decimal,
        real_value=real_total_in_decimal,
    )

    return {
        "total_out_rel_err_decimal": total_out_rel_err_decimal,
        "total_in_rel_err_decimal": total_in_rel_err_decimal,
        "max_side_rel_err_decimal": max_side_rel_err_decimal,
        "total_out_signed_rel_err_decimal": total_out_signed_rel_err_decimal,
        "total_in_signed_rel_err_decimal": total_in_signed_rel_err_decimal,
        "total_out_rel_err": _relative_error_text(
            model_value=model_total_out_decimal,
            real_value=real_total_out_decimal,
        ),
        "total_in_rel_err": _relative_error_text(
            model_value=model_total_in_decimal,
            real_value=real_total_in_decimal,
        ),
        "max_side_rel_err": (
            _compact_decimal_text(format(max_side_rel_err_decimal, "f"))
            if max_side_rel_err_decimal is not None
            else None
        ),
        "total_out_signed_rel_err": (
            _compact_decimal_text(format(total_out_signed_rel_err_decimal, "f"))
            if total_out_signed_rel_err_decimal is not None
            else None
        ),
        "total_in_signed_rel_err": (
            _compact_decimal_text(format(total_in_signed_rel_err_decimal, "f"))
            if total_in_signed_rel_err_decimal is not None
            else None
        ),
        "total_out_rel_err_bps": _relative_error_bps_text(
            model_value=model_total_out_decimal,
            real_value=real_total_out_decimal,
        ),
        "total_in_rel_err_bps": _relative_error_bps_text(
            model_value=model_total_in_decimal,
            real_value=real_total_in_decimal,
        ),
        "max_side_rel_err_bps": (
            _compact_decimal_text(format(max_side_rel_err_decimal * Decimal(10_000), "f"))
            if max_side_rel_err_decimal is not None
            else None
        ),
    }


def _report_total_relative_error_fields(report: dict[str, Any]) -> dict[str, Any]:
    model_total_out_decimal, model_total_in_decimal = _exact_report_totals_decimal(report["model"]["legs"])
    real_total_out_decimal, real_total_in_decimal = _exact_report_totals_decimal(report["real"]["legs"])
    return _relative_error_fields_from_decimals(
        model_total_out_decimal=model_total_out_decimal,
        model_total_in_decimal=model_total_in_decimal,
        real_total_out_decimal=real_total_out_decimal,
        real_total_in_decimal=real_total_in_decimal,
    )


def _comparison_with_derived_relative_fields(report: dict[str, Any]) -> dict[str, Any]:
    cmp = dict(report["comparison"])
    rel_fields = _report_total_relative_error_fields(report)
    cmp.setdefault("total_out_rel_err", rel_fields["total_out_rel_err"])
    cmp.setdefault("total_in_rel_err", rel_fields["total_in_rel_err"])
    cmp.setdefault("max_side_rel_err", rel_fields["max_side_rel_err"])
    cmp.setdefault("total_out_rel_err_bps", rel_fields["total_out_rel_err_bps"])
    cmp.setdefault("total_in_rel_err_bps", rel_fields["total_in_rel_err_bps"])
    cmp.setdefault("max_side_rel_err_bps", rel_fields["max_side_rel_err_bps"])
    return cmp


def _amount_kind(amount: Amount) -> str:
    return "XRP" if isinstance(amount, XRPAmount) else "IOU"


def _amount_to_report(amount: Amount, *, currency: str, issuer: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "currency": currency,
        "issuer": issuer,
        "kind": _amount_kind(amount),
        "value": _amount_to_string(amount),
    }
    if isinstance(amount, XRPAmount):
        payload["drops"] = amount.drops
    else:
        payload["mantissa"] = amount.mantissa
        payload["exponent"] = amount.exponent
    return payload


def _state_delta_amount(*, pre_state: Amount, post_state: Amount, state_change: str) -> Amount:
    if state_change == STATE_CHANGE_DECREASE:
        return pre_state - post_state
    if state_change == STATE_CHANGE_INCREASE:
        return post_state - pre_state
    raise RuntimeError(f"unsupported state_change={state_change}")


def _state_transition_report(
    *,
    pre_state: Amount,
    post_state: Amount,
    state_change: str,
    currency: str,
    issuer: str | None = None,
) -> dict[str, Any]:
    delta = _state_delta_amount(pre_state=pre_state, post_state=post_state, state_change=state_change)
    payload = _amount_to_report(delta, currency=currency, issuer=issuer)
    payload["state_change"] = state_change
    payload["pre_state"] = _amount_to_report(pre_state, currency=currency, issuer=issuer)
    payload["post_state"] = _amount_to_report(post_state, currency=currency, issuer=issuer)
    return payload


def _with_state_source(report: dict[str, Any], source: str) -> dict[str, Any]:
    out = dict(report)
    out["state_source"] = source
    return out


def _parse_xrp_amount(raw: Any, *, field_name: str) -> XRPAmount:
    try:
        return parse_xrp_drops(raw, field_name=field_name)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def _parse_iou_amount(raw: Any, *, field_name: str) -> IOUAmount:
    if not isinstance(raw, dict):
        raise RuntimeError(f"invalid {field_name}: expected IOU object")
    value = raw.get("value")
    if value is None:
        raise RuntimeError(f"invalid {field_name}: missing value")
    try:
        return parse_iou_decimal(value, field_name=field_name)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def _parse_amount(raw: Any, *, field_name: str) -> tuple[str, str | None, Amount]:
    if isinstance(raw, str):
        return XRP, None, _parse_xrp_amount(raw, field_name=field_name)
    if not isinstance(raw, dict):
        raise RuntimeError(f"invalid {field_name}: unsupported amount encoding")
    currency = str(raw.get("currency") or "")
    issuer = str(raw.get("issuer") or "") or None
    if not currency:
        raise RuntimeError(f"invalid {field_name}: missing currency")
    if currency == XRP:
        raise RuntimeError(f"invalid {field_name}: XRP must use drop string encoding")
    if issuer is None:
        raise RuntimeError(f"invalid {field_name}: missing issuer")
    return currency, issuer, _parse_iou_amount(raw, field_name=field_name)


def _zero_like(amount: Amount) -> Amount:
    if isinstance(amount, XRPAmount):
        return XRPAmount(0)
    return IOUAmount(0, 0)


def _decimal_to_iou_amount(value: Decimal) -> IOUAmount:
    try:
        return parse_iou_decimal(format(value, "f"), field_name="decimal_to_iou_amount")
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def _amount_delta(before: Amount, after: Amount) -> Amount:
    return before - after


def _positive_amount(amount: Amount) -> bool:
    if isinstance(amount, XRPAmount):
        return amount.drops > 0
    return amount.mantissa > 0


def _quality_to_string(quality: Quality | None) -> str | None:
    if quality is None:
        return None
    return format(_amount_to_decimal(quality.rate()), "f")


def _amount_lt(left: Amount, right: Amount) -> bool:
    if type(left) is not type(right):
        raise RuntimeError("amount domain mismatch")
    return left < right


def _fraction_from_decimal_text(value: Any, *, field_name: str) -> Fraction:
    text = str(value).strip()
    if not text:
        raise RuntimeError(f"invalid {field_name}: empty decimal string")
    sign = 1
    if text[0] == "+":
        text = text[1:]
    elif text[0] == "-":
        sign = -1
        text = text[1:]
    if not text:
        raise RuntimeError(f"invalid {field_name}: empty decimal string")
    if text.count(".") > 1:
        raise RuntimeError(f"invalid {field_name}: too many decimal points")

    exponent_adjust = 0
    if "e" in text.lower():
        lower = text.lower()
        if lower.count("e") != 1:
            raise RuntimeError(f"invalid {field_name}: too many exponent markers")
        significand_text, exponent_text = lower.split("e", 1)
        if not significand_text or not exponent_text:
            raise RuntimeError(f"invalid {field_name}: malformed exponent notation")
        exp_sign = 1
        if exponent_text[0] == "+":
            exponent_text = exponent_text[1:]
        elif exponent_text[0] == "-":
            exp_sign = -1
            exponent_text = exponent_text[1:]
        if not exponent_text or not exponent_text.isdigit():
            raise RuntimeError(f"invalid {field_name}: invalid exponent digits")
        exponent_adjust = exp_sign * int(exponent_text)
        text = significand_text

    if "." in text:
        whole_part, frac_part = text.split(".", 1)
    else:
        whole_part, frac_part = text, ""
    if not whole_part:
        whole_part = "0"
    if not whole_part.isdigit() or (frac_part and not frac_part.isdigit()):
        raise RuntimeError(f"invalid {field_name}: malformed decimal")
    digits_text = f"{whole_part}{frac_part}"
    numerator = 0 if not digits_text else int(digits_text)
    if sign < 0:
        numerator = -numerator
    scale = len(frac_part) - exponent_adjust
    if scale >= 0:
        denominator = 10 ** scale
        return Fraction(numerator, denominator)
    return Fraction(numerator * (10 ** (-scale)), 1)


def _parse_owner_funds_value(raw: Any, *, gets_is_xrp: bool) -> Amount:
    return (
        _parse_xrp_amount(raw, field_name="owner_funds")
        if gets_is_xrp
        else parse_iou_decimal(raw, field_name="owner_funds")
    )


def _amount_to_snapshot_scalar(amount: Amount) -> str:
    if isinstance(amount, XRPAmount):
        return str(amount.drops)
    return _compact_decimal_text(_amount_to_string(amount))


def _fee_units_to_decimal_text(fee_units: int) -> str:
    whole = fee_units // 100_000
    frac = fee_units % 100_000
    if frac == 0:
        return str(whole)
    return f"{whole}.{frac:05d}".rstrip("0")


def _parse_fee_units(value: Any) -> int:
    fraction = _fraction_from_decimal_text(value, field_name="fee")
    if fraction < 0 or fraction >= 1:
        raise RuntimeError("invalid fee: expected decimal fraction in [0, 1)")
    scaled = fraction * 100_000
    if scaled.denominator != 1:
        raise RuntimeError("invalid fee: must be exact in 1/100000 increments")
    fee_units = scaled.numerator
    if fee_units < 0 or fee_units > 1_000:
        raise RuntimeError("invalid fee: XRPL trading fee must be in [0, 1000]")
    return fee_units


def _parse_discounted_fee_units(
    discounted_value: Any,
    *,
    trading_fee_units: int,
) -> int:
    if trading_fee_units < 0:
        raise RuntimeError("invalid discounted_fee: missing trading_fee context")
    discounted_fraction = _fraction_from_decimal_text(discounted_value, field_name="discounted_fee")
    if discounted_fraction < 0 or discounted_fraction >= 1:
        raise RuntimeError("invalid discounted_fee: expected decimal fraction in [0, 1)")
    expected_display = Fraction(trading_fee_units, 1_000_000)
    tolerance = Fraction(1, 10**12)
    if abs(discounted_fraction - expected_display) > tolerance:
        raise RuntimeError(
            "invalid discounted_fee: expected displayed discounted fee to equal "
            "trading_fee / 10 within parquet float tolerance"
        )
    return trading_fee_units // 10


def _prebook_primary_side_for_output_currency(currency: str) -> str:
    if str(currency) == XRP:
        return "getsXRP"
    if str(currency) == TARGET_IOU_CURRENCY:
        return "getsrUSD"
    raise RuntimeError(f"unsupported output currency for snapshot side selection: {currency}")


def _snapshot_offer_to_book_offer(offer_row: dict[str, Any]) -> dict[str, Any] | None:
    taker_gets = offer_row.get("taker_gets")
    taker_pays = offer_row.get("taker_pays")
    if taker_gets is None or taker_pays is None:
        raise RuntimeError("snapshot offer row missing taker_gets/taker_pays")
    mapped: dict[str, Any] = {
        "TakerGets": taker_gets,
        "TakerPays": taker_pays,
    }
    if offer_row.get("offer_id") is not None:
        mapped["index"] = offer_row.get("offer_id")
    if offer_row.get("account") is not None:
        mapped["Account"] = offer_row.get("account")
    if offer_row.get("sequence") is not None:
        mapped["Sequence"] = offer_row.get("sequence")
    if offer_row.get("book_directory") is not None:
        mapped["BookDirectory"] = offer_row.get("book_directory")
    if bool(offer_row.get("non_fill_deleted")):
        mapped["non_fill_deleted"] = True
    raw_owner_funds = offer_row.get("owner_funds")
    if raw_owner_funds is not None:
        owner_funds_amount = _parse_owner_funds_value(
            raw_owner_funds,
            gets_is_xrp=isinstance(taker_gets, str),
        )
        if isinstance(owner_funds_amount, XRPAmount):
            if owner_funds_amount.drops < 0:
                raise RuntimeError("negative owner_funds")
        elif owner_funds_amount.mantissa < 0:
            raise RuntimeError("negative owner_funds")
        mapped["owner_funds"] = raw_owner_funds

    for src_key in ("taker_gets_funded", "taker_pays_funded"):
        raw_value = offer_row.get(src_key)
        if raw_value is None:
            continue
        try:
            _, _, amount = _parse_amount(raw_value, field_name=src_key)
        except Exception as exc:
            raise RuntimeError(f"invalid {src_key} in snapshot offer row: {exc}") from exc
        if not _positive_amount(amount):
            return None
        mapped[src_key] = raw_value
    return mapped


def _same_asset(
    left_currency: str,
    left_issuer: str | None,
    right_currency: str,
    right_issuer: str | None,
) -> bool:
    return str(left_currency) == str(right_currency) and (left_issuer or None) == (right_issuer or None)


def _amount_to_snapshot_encoding(
    amount: Amount,
    *,
    currency: str,
    issuer: str | None,
) -> Any:
    if isinstance(amount, XRPAmount):
        return str(amount.drops)
    return {
        "currency": currency,
        "issuer": issuer,
        "value": _compact_decimal_text(_amount_to_string(amount)),
    }


def _load_account_offers_snapshots(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                row = json.loads(payload)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to parse account_offers snapshot at {path}:{line_no}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                continue
            ledger_index = row.get("ledger_index")
            account = str(row.get("account") or "")
            if ledger_index is None or not account:
                raise RuntimeError(
                    f"account_offers snapshot missing ledger_index/account at {path}:{line_no}"
                )
            key = (int(ledger_index), account)
            if key in out:
                raise RuntimeError(
                    f"duplicate account_offers snapshot for ledger={ledger_index} account={account}"
                )
            out[key] = row
    return out


def _build_external_funded_caps_for_snapshot(
    *,
    side_payload: dict[str, Any],
    ledger_index: int,
    account_offers_snapshots: dict[tuple[int, str], dict[str, Any]],
) -> tuple[dict[str, Any], set[str]]:
    offers = side_payload.get("offers")
    if not isinstance(offers, list):
        raise RuntimeError("snapshot payload missing offers list")

    offers_by_account: dict[str, list[dict[str, Any]]] = {}
    for row in offers:
        if not isinstance(row, dict):
            continue
        account = str(row.get("account") or "")
        if not account:
            continue
        offers_by_account.setdefault(account, []).append(row)

    funded_caps_by_offer_id: dict[str, Any] = {}
    overridden_accounts: set[str] = set()
    for account, account_rows in offers_by_account.items():
        first_with_owner_funds = next((row for row in account_rows if row.get("owner_funds") is not None), None)
        if first_with_owner_funds is None:
            continue
        first_gets = first_with_owner_funds.get("taker_gets")
        try:
            gets_currency, gets_issuer, _available_funds = _parse_amount(
                first_gets,
                field_name="snapshot.taker_gets",
            )
            owner_funds_amount = _parse_owner_funds_value(
                first_with_owner_funds.get("owner_funds"),
                gets_is_xrp=(gets_currency == XRP),
            )
        except Exception as exc:
            raise RuntimeError(
                f"invalid snapshot owner_funds/taker_gets for ledger={ledger_index} account={account}: {exc}"
            ) from exc

        snapshot_row = account_offers_snapshots.get((ledger_index, account))
        if snapshot_row is None:
            continue
        account_offers = snapshot_row.get("offers")
        if not isinstance(account_offers, list):
            continue

        selected_sequences: set[int] = set()
        selected_offer_ids_by_sequence: dict[int, str] = {}
        selected_snapshot_gets_by_sequence: dict[int, Amount] = {}
        for row in account_rows:
            offer_id = str(row.get("offer_id") or "")
            sequence = row.get("sequence")
            if not offer_id or sequence is None:
                continue
            try:
                row_currency, row_issuer, row_gets_amount = _parse_amount(
                    row.get("taker_gets"),
                    field_name="snapshot.taker_gets",
                )
            except Exception as exc:
                raise RuntimeError(
                    f"invalid snapshot.taker_gets for ledger={ledger_index} account={account}: {exc}"
                ) from exc
            if not _same_asset(row_currency, row_issuer, gets_currency, gets_issuer):
                continue
            selected_sequences.add(int(sequence))
            selected_offer_ids_by_sequence[int(sequence)] = offer_id
            selected_snapshot_gets_by_sequence[int(sequence)] = row_gets_amount
        if not selected_sequences:
            continue

        matching_account_offers: list[tuple[int, Amount]] = []
        for offer in account_offers:
            if not isinstance(offer, dict):
                continue
            sequence = offer.get("seq")
            if sequence is None:
                continue
            try:
                offer_currency, offer_issuer, offer_gets_amount = _parse_amount(
                    offer.get("taker_gets"),
                    field_name="account_offers.taker_gets",
                )
            except Exception as exc:
                raise RuntimeError(
                    f"invalid account_offers.taker_gets for ledger={ledger_index} account={account}: {exc}"
                ) from exc
            if not _same_asset(offer_currency, offer_issuer, gets_currency, gets_issuer):
                continue
            matching_account_offers.append((int(sequence), offer_gets_amount))
        if not matching_account_offers:
            continue

        remaining_funds = owner_funds_amount
        for sequence, offer_gets_amount in sorted(matching_account_offers, key=lambda item: item[0]):
            funded_amount = offer_gets_amount if _amount_lt(offer_gets_amount, remaining_funds) else remaining_funds
            if sequence in selected_sequences:
                snapshot_gets_amount = selected_snapshot_gets_by_sequence.get(sequence)
                if snapshot_gets_amount is None:
                    continue
                if snapshot_gets_amount < funded_amount:
                    continue
                offer_id = selected_offer_ids_by_sequence[sequence]
                funded_caps_by_offer_id[offer_id] = _amount_to_snapshot_encoding(
                    funded_amount,
                    currency=gets_currency,
                    issuer=gets_issuer,
                )
                overridden_accounts.add(account)

            next_remaining = remaining_funds - funded_amount
            remaining_funds = next_remaining if _positive_amount(next_remaining) else _zero_like(remaining_funds)

    return funded_caps_by_offer_id, overridden_accounts


def _accounts_with_explicit_snapshot_funding(side_payload: dict[str, Any]) -> set[str]:
    offers = side_payload.get("offers")
    if not isinstance(offers, list):
        raise RuntimeError("snapshot payload missing offers list")
    out: set[str] = set()
    for row in offers:
        if not isinstance(row, dict):
            continue
        if row.get("taker_gets_funded") is None and row.get("taker_pays_funded") is None:
            continue
        account = str(row.get("account") or "")
        if account:
            out.add(account)
    return out


def _metadata_pre_state_field(node: dict[str, Any], key: str) -> Any:
    previous = node.get("PreviousFields")
    if isinstance(previous, dict) and key in previous:
        return previous.get(key)
    final = node.get("FinalFields")
    if isinstance(final, dict) and key in final:
        return final.get(key)
    created = node.get("NewFields")
    if isinstance(created, dict) and key in created:
        return created.get(key)
    return None


def _metadata_pre_offer_from_node(body: dict[str, Any]) -> dict[str, Any] | None:
    final_fields = body.get("FinalFields")
    previous_fields = body.get("PreviousFields")
    if not isinstance(final_fields, dict) or not isinstance(previous_fields, dict):
        return None
    pre_offer = dict(final_fields)
    pre_offer.update(previous_fields)
    if pre_offer.get("TakerGets") is None or pre_offer.get("TakerPays") is None:
        return None
    return pre_offer


def _metadata_offer_snapshot_side(pre_offer: dict[str, Any]) -> str | None:
    try:
        gets_currency, _, _ = _parse_amount(pre_offer.get("TakerGets"), field_name="Offer.TakerGets")
    except Exception:
        return None
    if gets_currency == XRP:
        return "getsXRP"
    if gets_currency == TARGET_IOU_CURRENCY:
        return "getsrUSD"
    return None


def _snapshot_row_from_metadata_offer(*, offer_id: str, pre_offer: dict[str, Any]) -> dict[str, Any] | None:
    taker_gets = pre_offer.get("TakerGets")
    taker_pays = pre_offer.get("TakerPays")
    if taker_gets is None or taker_pays is None:
        return None
    row: dict[str, Any] = {
        "offer_id": offer_id,
        "account": pre_offer.get("Account"),
        "sequence": pre_offer.get("Sequence"),
        "book_directory": pre_offer.get("BookDirectory"),
        "taker_gets": taker_gets,
        "taker_pays": taker_pays,
    }
    for key in ("owner_funds", "taker_gets_funded", "taker_pays_funded"):
        if pre_offer.get(key) is not None:
            row[key] = pre_offer.get(key)
    return row


def _metadata_current_tx_pre_offer_rows(*, result: dict[str, Any], selected_side: str) -> dict[str, dict[str, Any]]:
    meta = result.get("meta")
    if not isinstance(meta, dict):
        return {}
    affected = meta.get("AffectedNodes")
    if not isinstance(affected, list):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for wrapped in affected:
        if not isinstance(wrapped, dict):
            continue
        node_kind = None
        body = None
        for candidate in ("ModifiedNode", "DeletedNode"):
            maybe = wrapped.get(candidate)
            if isinstance(maybe, dict):
                node_kind = candidate
                body = maybe
                break
        if node_kind is None or body is None:
            continue
        if str(body.get("LedgerEntryType") or "") != "Offer":
            continue
        offer_id = str(body.get("LedgerIndex") or "")
        if not offer_id:
            continue
        pre_offer = _metadata_pre_offer_from_node(body)
        if not isinstance(pre_offer, dict):
            continue
        if _metadata_offer_snapshot_side(pre_offer) != selected_side:
            continue
        row = _snapshot_row_from_metadata_offer(offer_id=offer_id, pre_offer=pre_offer)
        if row is not None:
            out[offer_id] = row
    return out


def _overlay_current_tx_pre_offer_rows(
    *,
    side_payload: dict[str, Any],
    current_tx_pre_offer_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    if not current_tx_pre_offer_rows:
        return side_payload, 0
    offers = side_payload.get("offers")
    if not isinstance(offers, list):
        raise RuntimeError("snapshot payload missing offers list")
    patched_offers: list[Any] = []
    overlay_count = 0
    overlay_ids: set[str] = set()
    existing_ids: set[str] = set()
    for existing in offers:
        if not isinstance(existing, dict):
            patched_offers.append(existing)
            continue
        offer_id = str(existing.get("offer_id") or "")
        existing_ids.add(offer_id)
        overlay = current_tx_pre_offer_rows.get(offer_id)
        if overlay is None:
            patched_offers.append(existing)
            continue
        patched_row = dict(existing)
        patched_row.update(overlay)
        # Current-tx metadata PreviousFields is the authoritative pre-state for
        # touched offers. Snapshot funded caps are replay artifacts and can be
        # stale for same-ledger updates, so drop them unless metadata supplied
        # explicit replacements.
        for funded_key in ("taker_gets_funded", "taker_pays_funded"):
            if funded_key not in overlay:
                patched_row.pop(funded_key, None)
        patched_offers.append(patched_row)
        overlay_count += 1
        overlay_ids.add(offer_id)

    missing_rows = [
        row
        for offer_id, row in current_tx_pre_offer_rows.items()
        if offer_id and offer_id not in existing_ids
    ]
    overlay_count += len(missing_rows)
    overlay_ids.update(str(row["offer_id"]) for row in missing_rows)
    if overlay_count == 0:
        return side_payload, 0
    patched = dict(side_payload)
    patched["offers"] = [*patched_offers, *missing_rows]
    patched["current_tx_pre_offer_overlay_count"] = int(
        side_payload.get("current_tx_pre_offer_overlay_count") or 0
    ) + overlay_count
    patched["current_tx_pre_offer_overlay_ids"] = sorted(
        {
            *[
                str(x)
                for x in side_payload.get("current_tx_pre_offer_overlay_ids", [])
                if str(x)
            ],
            *overlay_ids,
        }
    )
    return patched, overlay_count


def _metadata_deleted_offer_without_fill_ids(result: dict[str, Any]) -> set[str]:
    meta = result.get("meta")
    if not isinstance(meta, dict):
        return set()
    affected = meta.get("AffectedNodes")
    if not isinstance(affected, list):
        return set()

    out: set[str] = set()
    for wrapped in affected:
        if not isinstance(wrapped, dict):
            continue
        body = wrapped.get("DeletedNode")
        if not isinstance(body, dict):
            continue
        if body.get("LedgerEntryType") != "Offer":
            continue
        previous = body.get("PreviousFields")
        if isinstance(previous, dict) and (
            "TakerGets" in previous or "TakerPays" in previous
        ):
            continue
        offer_id = str(body.get("LedgerIndex") or "")
        if offer_id:
            out.add(offer_id)
    return out


def _metadata_node_body(wrapped: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(wrapped, dict):
        return None
    for key in ("ModifiedNode", "DeletedNode", "CreatedNode"):
        maybe = wrapped.get(key)
        if isinstance(maybe, dict):
            return maybe
    return None


def _extract_overlay_owner_funds_from_metadata_result(result: dict[str, Any]) -> dict[str, str]:
    meta = result.get("meta")
    if not isinstance(meta, dict):
        return {}
    affected = meta.get("AffectedNodes")
    if not isinstance(affected, list):
        return {}

    account_root_pre: dict[str, tuple[int, int]] = {}
    ripple_pre: dict[tuple[str, str, str], IOUAmount] = {}

    for wrapped in affected:
        if not isinstance(wrapped, dict):
            continue
        body = None
        for key in ("ModifiedNode", "DeletedNode", "CreatedNode"):
            maybe = wrapped.get(key)
            if isinstance(maybe, dict):
                body = maybe
                break
        if body is None:
            continue
        ledger_entry_type = str(body.get("LedgerEntryType") or "")
        if ledger_entry_type == "AccountRoot":
            account = str(_metadata_pre_state_field(body, "Account") or "")
            balance_raw = _metadata_pre_state_field(body, "Balance")
            owner_count_raw = _metadata_pre_state_field(body, "OwnerCount")
            if not account or balance_raw is None:
                continue
            try:
                account_root_pre[account] = (
                    int(str(balance_raw)),
                    int(str(owner_count_raw)) if owner_count_raw is not None else 0,
                )
            except Exception:
                continue
            continue
        if ledger_entry_type != "RippleState":
            continue
        balance_raw = _metadata_pre_state_field(body, "Balance")
        low_limit = _metadata_pre_state_field(body, "LowLimit")
        high_limit = _metadata_pre_state_field(body, "HighLimit")
        if not isinstance(balance_raw, dict) or not isinstance(low_limit, dict) or not isinstance(high_limit, dict):
            continue
        currency = str(balance_raw.get("currency") or low_limit.get("currency") or high_limit.get("currency") or "")
        low_account = str(low_limit.get("issuer") or "")
        high_account = str(high_limit.get("issuer") or "")
        if not currency or not low_account or not high_account:
            continue
        try:
            balance = parse_iou_decimal(balance_raw.get("value"), field_name="RippleState Balance")
        except Exception:
            continue
        if balance.mantissa > 0:
            ripple_pre[(low_account, currency, high_account)] = balance
        if balance.mantissa < 0:
            ripple_pre[(high_account, currency, low_account)] = IOUAmount(-balance.mantissa, balance.exponent)

    out: dict[str, str] = {}
    for wrapped in affected:
        if not isinstance(wrapped, dict):
            continue
        body = None
        for key in ("ModifiedNode", "DeletedNode"):
            maybe = wrapped.get(key)
            if isinstance(maybe, dict):
                body = maybe
                break
        if body is None or str(body.get("LedgerEntryType") or "") != "Offer":
            continue
        offer_id = str(body.get("LedgerIndex") or "")
        if not offer_id:
            continue
        pre_offer = _metadata_pre_offer_from_node(body)
        if not isinstance(pre_offer, dict):
            continue
        account = str(pre_offer.get("Account") or "")
        if not account:
            continue
        pre_gets = pre_offer.get("TakerGets")
        if isinstance(pre_gets, str):
            state = account_root_pre.get(account)
            if state is None:
                continue
            balance_drops, owner_count = state
            reserve_drops = 1_000_000 + max(owner_count, 0) * 200_000
            spendable = balance_drops - reserve_drops
            if spendable > 0:
                out[offer_id] = str(spendable)
            continue
        if not isinstance(pre_gets, dict):
            continue
        currency = str(pre_gets.get("currency") or "")
        issuer = str(pre_gets.get("issuer") or "")
        if not currency or not issuer:
            continue
        balance = ripple_pre.get((account, currency, issuer))
        if balance is not None and balance.mantissa > 0:
            out[offer_id] = _amount_to_snapshot_scalar(balance)
    return out


def _extract_exact_amm_pre_state_from_metadata(
    *,
    result: dict[str, Any],
    expected_amm_account: str,
    amm_asset_currency: str,
    amm_asset_issuer: str | None,
    amm_asset2_currency: str,
    amm_asset2_issuer: str | None,
) -> dict[str, Any] | None:
    meta = result.get("meta")
    if not isinstance(meta, dict):
        return None
    affected = meta.get("AffectedNodes")
    if not isinstance(affected, list):
        return None

    candidate_pool_accounts: list[str] = []
    xrp_drops_by_account: dict[str, int] = {}
    for wrapped in affected:
        body = _metadata_node_body(wrapped)
        if body is None or str(body.get("LedgerEntryType") or "") != "AccountRoot":
            continue
        previous = body.get("PreviousFields") or {}
        final = body.get("FinalFields") or body.get("NewFields") or {}
        account = str(final.get("Account") or previous.get("Account") or "")
        if not account:
            continue
        if "AMMID" not in previous and "AMMID" not in final and account != expected_amm_account:
            continue
        balance_raw = _metadata_pre_state_field(body, "Balance")
        if balance_raw is None:
            continue
        try:
            xrp_drops_by_account[account] = int(str(balance_raw))
        except Exception:
            continue
        if account not in candidate_pool_accounts:
            candidate_pool_accounts.append(account)

    if not candidate_pool_accounts:
        return None

    pool_account = expected_amm_account if expected_amm_account in candidate_pool_accounts else candidate_pool_accounts[0]
    xrp_drops = xrp_drops_by_account.get(pool_account)
    if xrp_drops is None:
        return None

    target_iou_currency = amm_asset_currency if amm_asset_currency != XRP else amm_asset2_currency
    target_iou_issuer = amm_asset_issuer if amm_asset_currency != XRP else amm_asset2_issuer
    if target_iou_currency == XRP:
        return None

    iou_reserve: IOUAmount | None = None
    for wrapped in affected:
        body = _metadata_node_body(wrapped)
        if body is None or str(body.get("LedgerEntryType") or "") != "RippleState":
            continue
        balance_raw = _metadata_pre_state_field(body, "Balance")
        low_limit = _metadata_pre_state_field(body, "LowLimit")
        high_limit = _metadata_pre_state_field(body, "HighLimit")
        if not isinstance(balance_raw, dict) or not isinstance(low_limit, dict) or not isinstance(high_limit, dict):
            continue
        currency = str(balance_raw.get("currency") or low_limit.get("currency") or high_limit.get("currency") or "")
        if currency != target_iou_currency:
            continue
        low_account = str(low_limit.get("issuer") or "")
        high_account = str(high_limit.get("issuer") or "")
        if pool_account not in {low_account, high_account}:
            continue
        if target_iou_issuer and target_iou_issuer not in {low_account, high_account}:
            continue
        try:
            line_balance = parse_iou_decimal(balance_raw.get("value"), field_name="AMM RippleState Balance")
        except Exception:
            continue
        reserve = line_balance if low_account == pool_account else IOUAmount(-line_balance.mantissa, line_balance.exponent)
        if reserve.mantissa <= 0:
            continue
        iou_reserve = reserve
        break

    if iou_reserve is None:
        return None

    xrp_reserve = XRPAmount(xrp_drops)
    reserve_a = xrp_reserve if amm_asset_currency == XRP else iou_reserve
    reserve_b = xrp_reserve if amm_asset2_currency == XRP else iou_reserve
    return {
        "source": "metadata_exact_pre_state",
        "amm_account": pool_account,
        "reserve_a_before": reserve_a,
        "reserve_b_before": reserve_b,
        "xrp_reserve_drops_before": xrp_drops,
        "iou_reserve_before": iou_reserve,
        "iou_currency": target_iou_currency,
        "iou_issuer": target_iou_issuer,
    }


def _materialize_snapshot_book_offers(
    *,
    side_payload: dict[str, Any],
    derived_owner_funds_by_offer_id: dict[str, str],
    funded_caps_by_offer_id: dict[str, Any] | None = None,
    strip_owner_funds_accounts: set[str] | None = None,
) -> list[dict[str, Any]]:
    offers = side_payload.get("offers")
    if not isinstance(offers, list):
        raise RuntimeError("snapshot payload missing offers list")
    funded_caps_by_offer_id = funded_caps_by_offer_id or {}
    strip_owner_funds_accounts = strip_owner_funds_accounts or set()

    shared_owner_floor: dict[str, Amount] = {}
    derived_owner_funds_by_account: dict[str, str] = {}
    for row in offers:
        if not isinstance(row, dict):
            continue
        account = str(row.get("account") or "")
        offer_id = str(row.get("offer_id") or "")
        if account and offer_id and offer_id in derived_owner_funds_by_offer_id:
            derived_owner_funds = str(derived_owner_funds_by_offer_id[offer_id])
            prior_derived = derived_owner_funds_by_account.get(account)
            if prior_derived is not None and prior_derived != derived_owner_funds:
                raise RuntimeError(f"ambiguous metadata-derived owner_funds for account {account}")
            derived_owner_funds_by_account[account] = derived_owner_funds

        owner_funds_raw = row.get("owner_funds")
        if not account or owner_funds_raw is None:
            continue
        try:
            owner_funds = _parse_owner_funds_value(
                owner_funds_raw,
                gets_is_xrp=isinstance(row.get("taker_gets"), str),
            )
        except Exception as exc:
            raise RuntimeError(f"invalid snapshot owner_funds for account {account}: {exc}") from exc
        if not _positive_amount(owner_funds) and owner_funds != _zero_like(owner_funds):
            continue
        prior = shared_owner_floor.get(account)
        if prior is None or _amount_lt(owner_funds, prior):
            shared_owner_floor[account] = owner_funds

    materialised: list[dict[str, Any]] = []
    for row in offers:
        if not isinstance(row, dict):
            continue
        patched = dict(row)
        account = str(patched.get("account") or "")
        offer_id = str(patched.get("offer_id") or "")
        has_snapshot_funded_cap = (
            patched.get("taker_gets_funded") is not None
            or patched.get("taker_pays_funded") is not None
        )
        if account in strip_owner_funds_accounts:
            patched.pop("owner_funds", None)
        funded_cap = funded_caps_by_offer_id.get(offer_id)
        if funded_cap is not None:
            try:
                _, _, funded_amount = _parse_amount(funded_cap, field_name="taker_gets_funded")
            except Exception as exc:
                raise RuntimeError(f"invalid external funded cap for offer {offer_id}: {exc}") from exc
            if not _positive_amount(funded_amount):
                continue
            patched["taker_gets_funded"] = funded_cap
            patched.pop("taker_pays_funded", None)
        derived_owner_funds = derived_owner_funds_by_account.get(account)
        if derived_owner_funds is not None and funded_cap is None and not has_snapshot_funded_cap:
            patched["owner_funds"] = derived_owner_funds
            patched.pop("taker_gets_funded", None)
            patched.pop("taker_pays_funded", None)
        owner_floor = shared_owner_floor.get(account)
        if (
            owner_floor is not None
            and derived_owner_funds is None
            and account not in strip_owner_funds_accounts
        ):
            gets_is_xrp = isinstance(patched.get("taker_gets"), str)
            if gets_is_xrp != isinstance(owner_floor, XRPAmount):
                raise RuntimeError(
                    f"owner_funds domain mismatch for account {account}: "
                    f"gets_is_xrp={gets_is_xrp} owner_floor_type={type(owner_floor).__name__}"
                )
            patched["owner_funds"] = _amount_to_snapshot_scalar(owner_floor)
        mapped = _snapshot_offer_to_book_offer(patched)
        if mapped is not None:
            materialised.append(mapped)
    return materialised


def _apply_offercreate_cancel_sequence(
    *,
    book_offers: list[dict[str, Any]],
    intent: TxIntent,
) -> list[dict[str, Any]]:
    if str(intent.transaction_type or "").upper() != "OFFERCREATE":
        return book_offers
    cancel_sequence = intent.offer_cancel_sequence
    source_account = str(intent.source_account or "")
    if cancel_sequence is None or not source_account:
        return book_offers
    return [
        offer
        for offer in book_offers
        if not (
            str(offer.get("Account") or "") == source_account
            and offer.get("Sequence") == cancel_sequence
        )
    ]


def _filter_non_fill_deleted_offers(
    *,
    book_offers: list[dict[str, Any]],
    offer_ids: set[str],
    keep_owner: str | None = None,
) -> list[dict[str, Any]]:
    if not offer_ids:
        return book_offers
    keep_owner = str(keep_owner or "")
    out: list[dict[str, Any]] = []
    for offer in book_offers:
        offer_id = str(offer.get("index") or "")
        if offer_id not in offer_ids:
            out.append(offer)
            continue
        if keep_owner and str(offer.get("Account") or "") == keep_owner:
            # Rippled still lets self-cross-deleted tip offers influence
            # AMM synthetic-offer sizing before deleting them without fill.
            patched = dict(offer)
            patched["non_fill_deleted"] = True
            out.append(patched)
    return out


def _effective_amm_fee(
    *,
    fee_row: dict[str, Any],
    amm_in_currency: str,
    amm_in_value_raw: Any,
) -> int:
    trading_fee_raw = fee_row.get("trading_fee")
    discounted_fee_raw = fee_row.get("discounted_fee")
    trading_fee = None if pd.isna(trading_fee_raw) else _parse_fee_units(trading_fee_raw)
    discounted_fee = (
        None
        if pd.isna(discounted_fee_raw)
        else _parse_discounted_fee_units(
            discounted_fee_raw,
            trading_fee_units=trading_fee if trading_fee is not None else -1,
        )
    )
    fee_value_raw = fee_row.get("trading_fee_value")
    fee_value = None if pd.isna(fee_value_raw) else _fraction_from_decimal_text(fee_value_raw, field_name="trading_fee_value")
    fee_currency_raw = fee_row.get("trading_fee_currency")
    fee_currency = None if pd.isna(fee_currency_raw) else str(fee_currency_raw)
    amm_in_value = _fraction_from_decimal_text(amm_in_value_raw, field_name="asset_in_value")

    if trading_fee is None and discounted_fee is None:
        raise RuntimeError("AMM fee row missing both trading_fee and discounted_fee")

    if fee_value is not None:
        if amm_in_value <= 0:
            raise RuntimeError("cannot infer effective AMM fee from non-positive input amount")
        if fee_currency is not None and fee_currency != amm_in_currency:
            raise RuntimeError(
                f"AMM fee currency mismatch: fee_currency={fee_currency} input_currency={amm_in_currency}"
            )
        implied = fee_value / amm_in_value
        tolerance = Fraction(1, 10**12)
        if trading_fee is not None and abs(implied - Fraction(trading_fee, 100_000)) <= tolerance:
            return trading_fee
        if discounted_fee is not None and abs(implied - Fraction(discounted_fee, 100_000)) <= tolerance:
            return discounted_fee
        raise RuntimeError(
            "AMM fee implied by fee value does not match trading/discounted fee "
            f"(implied={implied}, trading={trading_fee}, discounted={discounted_fee})"
        )

    if trading_fee is not None:
        return trading_fee
    assert discounted_fee is not None
    return discounted_fee


def _load_sample_hashes(args: argparse.Namespace) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add_all(values: Iterable[str]) -> None:
        for value in values:
            tx_hash = _normalise_tx_hash(value)
            if not tx_hash or tx_hash.startswith("#") or tx_hash in seen:
                continue
            seen.add(tx_hash)
            out.append(tx_hash)

    add_all(args.tx_hash or [])

    should_load_samples_file = bool(args.samples_file)
    if args.tx_hash and str(args.samples_file) == str(CURATED_SAMPLES_PATH):
        should_load_samples_file = False
    if args.target_tx_file and str(args.samples_file) == str(CURATED_SAMPLES_PATH):
        should_load_samples_file = False
    if should_load_samples_file:
        samples_file = Path(args.samples_file)
        if samples_file.exists():
            add_all(samples_file.read_text(encoding="utf-8").splitlines())

    if args.target_tx_file:
        add_all(_load_hashes_from_explicit_target_file(Path(args.target_tx_file)))

    if not out:
        raise RuntimeError("no sample tx hashes provided")
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise RuntimeError("--max-samples must be positive when provided")
        out = out[: int(args.max_samples)]
    return out


def _load_hashes_from_explicit_target_file(path: Path) -> list[str]:
    if not path.exists():
        raise RuntimeError(f"target tx file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [_normalise_tx_hash(line) for line in path.read_text(encoding="utf-8").splitlines() if _normalise_tx_hash(line)]
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise RuntimeError(
            f"unsupported target tx file format: {path} "
            "(expected .csv, .parquet, or .txt)"
        )
    for column in ("transaction_hash", "tx_hash"):
        if column not in frame.columns:
            continue
        values = [_normalise_tx_hash(x) for x in frame[column].dropna().astype(str).tolist()]
        values = [x for x in values if x]
        if values:
            return values
    raise RuntimeError(
        f"target tx file missing supported hash column {['transaction_hash', 'tx_hash']}: {path}"
    )


def _load_metadata_samples(*, metadata_path: Path, target_tx_hashes: set[str]) -> dict[str, SampleInput]:
    out: dict[str, SampleInput] = {}
    progress_every = 20_000 if len(target_tx_hashes) > 10_000 else 0
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            if progress_every and line_no % progress_every == 0:
                print(
                    f"[fit] metadata_scan lines={line_no} matched={len(out)}/{len(target_tx_hashes)}",
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
            tx_hash = _normalise_tx_hash(extract_tx_hash_from_metadata_obj(obj))
            if tx_hash not in target_tx_hashes:
                continue
            result = obj.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"metadata row missing result object for tx={tx_hash}")
            intent = build_intent_from_metadata_result(result)
            if not _is_direct_pair(intent.in_cur, intent.out_cur):
                raise RuntimeError(f"sample tx is not direct rUSD/XRP pair scoped: tx={tx_hash}")
            ledger_index = result.get("ledger_index")
            tx_index = (result.get("meta") or {}).get("TransactionIndex")
            if ledger_index is None or tx_index is None:
                raise RuntimeError(f"metadata row missing ledger_index/TransactionIndex for tx={tx_hash}")
            out[tx_hash] = SampleInput(
                tx_hash=tx_hash,
                ledger_index=int(ledger_index),
                transaction_index=int(tx_index),
                result=result,
                intent=intent,
            )
            if len(out) == len(target_tx_hashes):
                break
    missing = sorted(target_tx_hashes - set(out))
    if missing:
        raise RuntimeError(f"missing metadata rows for {len(missing)} sample txs: {', '.join(missing[:5])}")
    return out


def _iter_snapshot_rows_in_sample_order(
    *,
    snapshots_path: Path,
    ordered_targets: list[tuple[str, str]],
) -> Iterable[tuple[str, dict[str, Any]]]:
    required_side_by_hash = {tx_hash: side for tx_hash, side in ordered_targets}
    buffered: dict[str, dict[str, Any]] = {}
    next_index = 0
    progress_every = 100_000 if len(ordered_targets) > 10_000 else 0

    with snapshots_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            if progress_every and line_no % progress_every == 0:
                print(
                    f"[fit] snapshot_scan lines={line_no} yielded={next_index}/{len(ordered_targets)} buffered={len(buffered)}",
                    flush=True,
                )
            try:
                row = json.loads(payload)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to parse tx_prebook snapshot at {snapshots_path}:{line_no}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                continue
            tx_hash = _normalise_tx_hash(row.get("transaction_hash") or "")
            if tx_hash not in required_side_by_hash:
                continue
            side = str(row.get("side") or "")
            if side != required_side_by_hash[tx_hash]:
                continue
            if tx_hash in buffered:
                raise RuntimeError(f"duplicate snapshot side row for tx={tx_hash} side={side}")
            buffered[tx_hash] = row

            while next_index < len(ordered_targets):
                expected_hash, _expected_side = ordered_targets[next_index]
                matched = buffered.pop(expected_hash, None)
                if matched is None:
                    break
                yield expected_hash, matched
                next_index += 1
            if next_index == len(ordered_targets):
                break

    if next_index != len(ordered_targets):
        missing = [tx_hash for tx_hash, _side in ordered_targets[next_index:] if tx_hash not in buffered]
        sample = ", ".join(missing[:5])
        raise RuntimeError(
            f"missing tx_prebook side rows for {len(missing)} sample txs after streaming snapshots: {sample}"
        )


def _load_offer_tick_size_snapshots(metadata_path: Path) -> list[tuple[int, dict[str, int | None]]]:
    tick_size_files = sorted(metadata_path.parent.glob("offer_tick_sizes_at_ledger_*.json"))
    out: list[tuple[int, dict[str, int | None]]] = []
    for path in tick_size_files:
        suffix = path.stem.removeprefix("offer_tick_sizes_at_ledger_")
        try:
            ledger_index = int(suffix)
        except ValueError as exc:
            raise RuntimeError(f"invalid offer tick-size filename: {path}") from exc
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"failed to parse offer tick-size snapshot {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise RuntimeError(f"invalid offer tick-size snapshot payload: {path}")
        payload = raw.get("tick_sizes") if isinstance(raw.get("tick_sizes"), dict) else raw
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid offer tick-size mapping payload: {path}")
        tick_sizes: dict[str, int | None] = {}
        for account, tick_size in payload.items():
            if tick_size is None:
                tick_sizes[str(account)] = None
                continue
            try:
                tick_sizes[str(account)] = int(tick_size)
            except Exception as exc:
                raise RuntimeError(f"invalid tick size for account {account} in {path}: {exc}") from exc
        out.append((ledger_index, tick_sizes))
    return out


def _lookup_offer_tick_size(
    *,
    ledger_index: int,
    account: str | None,
    tick_size_snapshots: list[tuple[int, dict[str, int | None]]],
) -> int | None:
    if not account:
        return None
    selected: int | None = None
    for snapshot_ledger, mapping in tick_size_snapshots:
        if snapshot_ledger > ledger_index:
            break
        if account in mapping:
            selected = mapping[account]
    return selected


def _effective_offer_tick_size_for_intent(
    *,
    intent: TxIntent,
    ledger_index: int,
    tick_size_snapshots: list[tuple[int, dict[str, int | None]]],
) -> int | None:
    if str(intent.transaction_type or "").upper() != "OFFERCREATE":
        return None
    candidates: list[int] = []
    for currency, issuer in (
        (intent.offer_taker_gets_currency, intent.offer_taker_gets_issuer),
        (intent.offer_taker_pays_currency, intent.offer_taker_pays_issuer),
    ):
        if not currency or _is_xrp(currency):
            continue
        tick_size = _lookup_offer_tick_size(
            ledger_index=ledger_index,
            account=issuer,
            tick_size_snapshots=tick_size_snapshots,
        )
        if tick_size is not None:
            candidates.append(tick_size)
    if not candidates:
        return None
    return min(candidates)


def _load_amm_rows(path: Path, *, tx_column: str, target_tx_hashes: set[str]) -> dict[str, list[dict[str, Any]]]:
    print(f"[fit] loading parquet {path.name} for {len(target_tx_hashes)} target txs", flush=True)
    try:
        frame = pd.read_parquet(path, filters=[(tx_column, "in", sorted(target_tx_hashes))])
    except Exception:
        frame = pd.read_parquet(path)
    out: dict[str, list[dict[str, Any]]] = {}
    for row in frame.to_dict(orient="records"):
        tx_hash = _normalise_tx_hash(row.get(tx_column) or "")
        if tx_hash not in target_tx_hashes:
            continue
        out.setdefault(tx_hash, []).append(row)
    print(f"[fit] loaded parquet {path.name}: matched_rows={sum(len(v) for v in out.values())}", flush=True)
    return out


@dataclass(slots=True)
class _RunningDecimalStats:
    count: int = 0
    total: Decimal = Decimal(0)
    total_square: Decimal = Decimal(0)

    def update(self, value: Decimal | None) -> None:
        if value is None:
            return
        self.count += 1
        self.total += value
        self.total_square += value * value

    def as_dict(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "mean": None, "sample_standard_deviation": None}
        mean = self.total / Decimal(self.count)
        if self.count < 2:
            sample_variance = Decimal(0)
        else:
            sample_variance = (self.total_square - (self.total * self.total / Decimal(self.count))) / Decimal(
                self.count - 1
            )
            if sample_variance < 0 and abs(sample_variance) < Decimal("1e-80"):
                sample_variance = Decimal(0)
        return {
            "count": self.count,
            "mean": _compact_decimal_text(format(mean, "f")),
            "sample_standard_deviation": _compact_decimal_text(format(sample_variance.sqrt(), "f")),
        }


@dataclass(slots=True)
class _EndpointTTestGroupStats:
    tx_count: int = 0
    unsigned_rel_err_buckets_by_side: dict[str, Counter[str]] = field(default_factory=dict)
    unsigned_rel_err_stats_by_side: dict[str, _RunningDecimalStats] = field(default_factory=dict)
    signed_rel_err_stats_by_side: dict[str, _RunningDecimalStats] = field(default_factory=dict)

    def update_tx(self, endpoint_rel_fields: dict[str, Decimal | None], *, sides: tuple[str, ...]) -> None:
        self.tx_count += 1
        for side in sides:
            unsigned = _endpoint_rel_err_for_side(endpoint_rel_fields, side=side, signed=False)
            signed = _endpoint_rel_err_for_side(endpoint_rel_fields, side=side, signed=True)
            if unsigned is None:
                continue
            self.unsigned_rel_err_buckets_by_side.setdefault(side, Counter())[
                _relative_error_bucket(unsigned)
            ] += 1
            self.unsigned_rel_err_stats_by_side.setdefault(side, _RunningDecimalStats()).update(unsigned)
            self.signed_rel_err_stats_by_side.setdefault(side, _RunningDecimalStats()).update(signed)

    def as_dict(self) -> dict[str, Any]:
        sides = sorted(
            set(self.unsigned_rel_err_buckets_by_side)
            | set(self.unsigned_rel_err_stats_by_side)
            | set(self.signed_rel_err_stats_by_side)
        )
        return {
            "tx_count": self.tx_count,
            "measurements": {
                side: {
                    "measurement_count": self.unsigned_rel_err_stats_by_side.get(
                        side, _RunningDecimalStats()
                    ).count,
                    "unsigned_relative_error_buckets": _ordered_bucket_counts(
                        self.unsigned_rel_err_buckets_by_side.get(side, Counter())
                    ),
                    "unsigned_relative_error": self.unsigned_rel_err_stats_by_side.get(
                        side, _RunningDecimalStats()
                    ).as_dict(),
                    "signed_relative_error": self.signed_rel_err_stats_by_side.get(
                        side, _RunningDecimalStats()
                    ).as_dict(),
                }
                for side in sides
            },
        }


def _endpoint_ttest_group_for_role(role: str) -> tuple[str, tuple[str, ...]] | None:
    if role in {
        "payment_full_output_target_input_predicted",
        "payment_partial_exact_output_target_input_predicted",
        "offer_fok_buy_output_target_input_predicted",
    }:
        return ("input_predicted", ("in",))
    if role in {
        "offer_fok_sell_input_target_output_predicted",
    }:
        return ("output_predicted", ("out",))
    if role in {
        "payment_partial_dual_constraint_both_computed",
        "offer_limit_order_crossing_both_computed",
    }:
        return ("both_computed", ("in", "out"))
    return None


@dataclass(slots=True)
class _PredictionRoleValidationStats:
    count: int = 0
    strict_exact_match_count: int = 0
    structure_match_count: int = 0
    structure_false_count: int = 0
    predicted_side_counts: Counter[str] = field(default_factory=Counter)
    constraint_side_counts: Counter[str] = field(default_factory=Counter)
    in_side_exact_count: int = 0
    out_side_exact_count: int = 0
    predicted_side_exact_count: int = 0
    constraint_side_exact_count: int = 0

    def update(self, cmp: dict[str, Any]) -> None:
        self.count += 1
        strict_exact = bool(cmp.get("strict_exact_match"))
        structure_match = bool(cmp.get("structure_match"))
        self.strict_exact_match_count += int(strict_exact)
        self.structure_match_count += int(structure_match)
        self.structure_false_count += int(not structure_match)

        predicted_side = str(cmp.get("predicted_side") or "")
        constraint_side = str(cmp.get("constraint_side") or "")
        if predicted_side:
            self.predicted_side_counts[predicted_side] += 1
        if constraint_side:
            self.constraint_side_counts[constraint_side] += 1

        in_exact = bool(cmp.get("in_side_exact_match"))
        out_exact = bool(cmp.get("out_side_exact_match"))
        self.in_side_exact_count += int(in_exact)
        self.out_side_exact_count += int(out_exact)

        side_exact = {
            "in": in_exact,
            "out": out_exact,
        }
        if predicted_side in side_exact:
            self.predicted_side_exact_count += int(side_exact[predicted_side])
        if constraint_side in side_exact:
            self.constraint_side_exact_count += int(side_exact[constraint_side])

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "strict_exact_match_count": self.strict_exact_match_count,
            "structure_match_count": self.structure_match_count,
            "structure_false_count": self.structure_false_count,
            "predicted_side_counts": dict(sorted(self.predicted_side_counts.items())),
            "constraint_side_counts": dict(sorted(self.constraint_side_counts.items())),
            "in_side_exact_count": self.in_side_exact_count,
            "out_side_exact_count": self.out_side_exact_count,
            "predicted_side_exact_count": self.predicted_side_exact_count,
            "constraint_side_exact_count": self.constraint_side_exact_count,
        }


@dataclass(slots=True)
class _AggregateStatsBuilder:
    top_relative_error_limit: int = 0
    sample_count: int = 0
    strict_exact_match_count: int = 0
    structure_match_count: int = 0
    first_non_exact_tx: dict[str, Any] | None = None
    strict_false_structure_true_count: int = 0
    structure_false_count: int = 0
    strict_false_structure_true_in_rel_err_buckets: Counter[str] = field(default_factory=Counter)
    strict_false_structure_true_out_rel_err_buckets: Counter[str] = field(default_factory=Counter)
    strict_false_structure_true_max_side_rel_err_buckets: Counter[str] = field(default_factory=Counter)
    structure_matched_in_endpoint_rel_err_buckets: Counter[str] = field(default_factory=Counter)
    structure_matched_out_endpoint_rel_err_buckets: Counter[str] = field(default_factory=Counter)
    structure_matched_max_side_endpoint_rel_err_buckets: Counter[str] = field(default_factory=Counter)
    structure_matched_predicted_side_endpoint_rel_err_buckets: Counter[str] = field(default_factory=Counter)
    structure_matched_constraint_side_endpoint_rel_err_buckets: Counter[str] = field(default_factory=Counter)
    endpoint_prediction_role_counts: Counter[str] = field(default_factory=Counter)
    endpoint_prediction_role_validation: dict[str, _PredictionRoleValidationStats] = field(default_factory=dict)
    structure_matched_unsigned_in_endpoint_rel_err_stats: _RunningDecimalStats = field(
        default_factory=_RunningDecimalStats
    )
    structure_matched_unsigned_out_endpoint_rel_err_stats: _RunningDecimalStats = field(
        default_factory=_RunningDecimalStats
    )
    structure_matched_unsigned_predicted_side_endpoint_rel_err_stats: _RunningDecimalStats = field(
        default_factory=_RunningDecimalStats
    )
    structure_matched_unsigned_constraint_side_endpoint_rel_err_stats: _RunningDecimalStats = field(
        default_factory=_RunningDecimalStats
    )
    structure_matched_signed_in_endpoint_rel_err_stats: _RunningDecimalStats = field(
        default_factory=_RunningDecimalStats
    )
    structure_matched_signed_out_endpoint_rel_err_stats: _RunningDecimalStats = field(
        default_factory=_RunningDecimalStats
    )
    structure_matched_signed_predicted_side_endpoint_rel_err_stats: _RunningDecimalStats = field(
        default_factory=_RunningDecimalStats
    )
    structure_matched_signed_constraint_side_endpoint_rel_err_stats: _RunningDecimalStats = field(
        default_factory=_RunningDecimalStats
    )
    endpoint_ttest_groups: dict[str, _EndpointTTestGroupStats] = field(default_factory=dict)
    top_relative_errors: list[dict[str, Any]] = field(default_factory=list)

    def _maybe_track_top_relative_error(
        self,
        *,
        report: dict[str, Any],
        max_side_rel_err_decimal: Decimal | None,
        endpoint_rel_fields: dict[str, Decimal | None],
    ) -> None:
        if self.top_relative_error_limit <= 0 or max_side_rel_err_decimal is None:
            return
        cmp = _comparison_with_derived_relative_fields(report)
        model_legs = report["model"]["legs"]
        real_legs = report["real"]["legs"]
        endpoint_text_fields = _endpoint_residual_relative_text_fields(endpoint_rel_fields)
        entry = {
            "_sort_key": format(max_side_rel_err_decimal, "f"),
            "tx_hash": report["tx_hash"],
            "ledger_index": report["ledger_index"],
            "transaction_index": report["transaction_index"],
            "transaction_type": report["intent"]["transaction_type"],
            "flags": report["intent"]["flags"],
            "direction": f"{report['intent']['in_cur']}->{report['intent']['out_cur']}",
            "is_sell": report["intent"].get("is_sell"),
            "is_ioc": report["intent"].get("is_ioc"),
            "is_passive": report["intent"].get("is_passive"),
            "partial_allowed": report["intent"].get("partial_allowed"),
            "model_leg_count": cmp["model_leg_count"],
            "real_leg_count": cmp["real_leg_count"],
            "model_amm_leg_count": sum(1 for leg in model_legs if str(leg.get("src") or "") == "AMM"),
            "real_amm_leg_count": sum(1 for leg in real_legs if str(leg.get("src") or "") == "AMM"),
            "model_clob_leg_count": sum(1 for leg in model_legs if str(leg.get("src") or "") == "CLOB"),
            "real_clob_leg_count": sum(1 for leg in real_legs if str(leg.get("src") or "") == "CLOB"),
            "error_metric": "endpoint_residual",
            "predicted_side": cmp.get("predicted_side"),
            "constraint_side": cmp.get("constraint_side"),
            "prediction_role": cmp.get("prediction_role"),
            "prediction_role_reason": cmp.get("prediction_role_reason"),
            "total_in_endpoint_signed_rel_err": endpoint_text_fields["total_in_endpoint_signed_rel_err"],
            "total_out_endpoint_signed_rel_err": endpoint_text_fields["total_out_endpoint_signed_rel_err"],
            "total_in_endpoint_unsigned_rel_err": endpoint_text_fields["total_in_endpoint_unsigned_rel_err"],
            "total_out_endpoint_unsigned_rel_err": endpoint_text_fields["total_out_endpoint_unsigned_rel_err"],
            "max_side_endpoint_unsigned_rel_err": endpoint_text_fields["max_side_endpoint_unsigned_rel_err"],
            "total_in_endpoint_signed_rel_err_bps": endpoint_text_fields["total_in_endpoint_signed_rel_err_bps"],
            "total_out_endpoint_signed_rel_err_bps": endpoint_text_fields["total_out_endpoint_signed_rel_err_bps"],
            "total_in_endpoint_unsigned_rel_err_bps": endpoint_text_fields["total_in_endpoint_unsigned_rel_err_bps"],
            "total_out_endpoint_unsigned_rel_err_bps": endpoint_text_fields["total_out_endpoint_unsigned_rel_err_bps"],
            "max_side_endpoint_unsigned_rel_err_bps": endpoint_text_fields["max_side_endpoint_unsigned_rel_err_bps"],
            "legacy_total_in_diff": cmp["total_in_diff"],
            "legacy_total_out_diff": cmp["total_out_diff"],
            "legacy_total_in_rel_err_bps": cmp.get("total_in_rel_err_bps"),
            "legacy_total_out_rel_err_bps": cmp.get("total_out_rel_err_bps"),
            "legacy_max_side_rel_err_bps": cmp.get("max_side_rel_err_bps"),
            "failed_side": cmp.get("failed_side"),
            "first_failed_leg_source_kind": cmp.get("first_failed_leg_source_kind"),
            "first_failed_leg_source_id": cmp.get("first_failed_leg_source_id"),
            "model_source_ids_first12": [
                (str(leg.get("src") or ""), str(leg.get("source_id") or "")[:12])
                for leg in model_legs[:12]
            ],
            "real_source_ids_first12": [
                (str(leg.get("src") or ""), str(leg.get("offer_id") or leg.get("amm_account") or "")[:12])
                for leg in real_legs[:12]
            ],
        }
        self.top_relative_errors.append(entry)
        self.top_relative_errors.sort(key=lambda item: Decimal(str(item["_sort_key"])), reverse=True)
        del self.top_relative_errors[self.top_relative_error_limit :]

    def update(self, report: dict[str, Any]) -> None:
        self.sample_count += 1
        cmp = _comparison_with_derived_relative_fields(report)
        strict_exact = bool(cmp["strict_exact_match"])
        structure_match = bool(cmp["structure_match"])
        endpoint_rel_fields: dict[str, Decimal | None] | None = None

        self.strict_exact_match_count += int(strict_exact)
        self.structure_match_count += int(structure_match)
        prediction_role = str(cmp.get("prediction_role") or "unknown")
        self.endpoint_prediction_role_validation.setdefault(
            prediction_role,
            _PredictionRoleValidationStats(),
        ).update(cmp)

        if structure_match:
            model_endpoint_legs = report["model"].get("endpoint_legs") or report["model"]["legs"]
            endpoint_rel_fields = _endpoint_residual_relative_fields(
                model_legs=model_endpoint_legs,
                real_legs=report["real"]["legs"],
            )
            total_in_rel_err_decimal = endpoint_rel_fields["total_in_endpoint_unsigned_rel_err_decimal"]
            total_out_rel_err_decimal = endpoint_rel_fields["total_out_endpoint_unsigned_rel_err_decimal"]
            max_side_rel_err_decimal = endpoint_rel_fields["max_side_endpoint_unsigned_rel_err_decimal"]
            total_in_signed_rel_err_decimal = endpoint_rel_fields["total_in_endpoint_signed_rel_err_decimal"]
            total_out_signed_rel_err_decimal = endpoint_rel_fields["total_out_endpoint_signed_rel_err_decimal"]
            predicted_side = str(cmp.get("predicted_side") or "")
            constraint_side = str(cmp.get("constraint_side") or "")
            self.endpoint_prediction_role_counts[prediction_role] += 1
            endpoint_group = _endpoint_ttest_group_for_role(prediction_role)
            if endpoint_group is not None:
                group_name, group_sides = endpoint_group
                self.endpoint_ttest_groups.setdefault(group_name, _EndpointTTestGroupStats()).update_tx(
                    endpoint_rel_fields,
                    sides=group_sides,
                )
            self.structure_matched_in_endpoint_rel_err_buckets[
                _relative_error_bucket(total_in_rel_err_decimal)
            ] += 1
            self.structure_matched_out_endpoint_rel_err_buckets[
                _relative_error_bucket(total_out_rel_err_decimal)
            ] += 1
            self.structure_matched_max_side_endpoint_rel_err_buckets[
                _relative_error_bucket(max_side_rel_err_decimal)
            ] += 1
            self.structure_matched_unsigned_in_endpoint_rel_err_stats.update(total_in_rel_err_decimal)
            self.structure_matched_unsigned_out_endpoint_rel_err_stats.update(total_out_rel_err_decimal)
            self.structure_matched_signed_in_endpoint_rel_err_stats.update(total_in_signed_rel_err_decimal)
            self.structure_matched_signed_out_endpoint_rel_err_stats.update(total_out_signed_rel_err_decimal)
            predicted_unsigned_rel_err_decimal = _endpoint_rel_err_for_side(
                endpoint_rel_fields,
                side=predicted_side,
                signed=False,
            )
            predicted_signed_rel_err_decimal = _endpoint_rel_err_for_side(
                endpoint_rel_fields,
                side=predicted_side,
                signed=True,
            )
            constraint_unsigned_rel_err_decimal = _endpoint_rel_err_for_side(
                endpoint_rel_fields,
                side=constraint_side,
                signed=False,
            )
            constraint_signed_rel_err_decimal = _endpoint_rel_err_for_side(
                endpoint_rel_fields,
                side=constraint_side,
                signed=True,
            )
            if predicted_unsigned_rel_err_decimal is not None:
                self.structure_matched_predicted_side_endpoint_rel_err_buckets[
                    _relative_error_bucket(predicted_unsigned_rel_err_decimal)
                ] += 1
                self.structure_matched_unsigned_predicted_side_endpoint_rel_err_stats.update(
                    predicted_unsigned_rel_err_decimal
                )
                self.structure_matched_signed_predicted_side_endpoint_rel_err_stats.update(
                    predicted_signed_rel_err_decimal
                )
            if constraint_unsigned_rel_err_decimal is not None:
                self.structure_matched_constraint_side_endpoint_rel_err_buckets[
                    _relative_error_bucket(constraint_unsigned_rel_err_decimal)
                ] += 1
                self.structure_matched_unsigned_constraint_side_endpoint_rel_err_stats.update(
                    constraint_unsigned_rel_err_decimal
                )
                self.structure_matched_signed_constraint_side_endpoint_rel_err_stats.update(
                    constraint_signed_rel_err_decimal
                )

        if not strict_exact and structure_match:
            assert endpoint_rel_fields is not None
            self.strict_false_structure_true_count += 1
            self._maybe_track_top_relative_error(
                report=report,
                max_side_rel_err_decimal=max_side_rel_err_decimal,
                endpoint_rel_fields=endpoint_rel_fields,
            )
            self.strict_false_structure_true_in_rel_err_buckets[
                _relative_error_bucket(total_in_rel_err_decimal)
            ] += 1
            self.strict_false_structure_true_out_rel_err_buckets[
                _relative_error_bucket(total_out_rel_err_decimal)
            ] += 1
            self.strict_false_structure_true_max_side_rel_err_buckets[
                _relative_error_bucket(max_side_rel_err_decimal)
            ] += 1
        elif not structure_match:
            self.structure_false_count += 1

        if self.first_non_exact_tx is None and not strict_exact:
            endpoint_text_fields = (
                _endpoint_residual_relative_text_fields(endpoint_rel_fields)
                if endpoint_rel_fields is not None
                else _endpoint_residual_relative_text_fields(_empty_endpoint_residual_relative_fields())
            )
            self.first_non_exact_tx = {
                "tx_hash": report["tx_hash"],
                "ledger_index": report["ledger_index"],
                "transaction_index": report["transaction_index"],
                "strict_exact_match": strict_exact,
                "structure_match": structure_match,
                "total_in_match": bool(cmp["total_in_match"]),
                "total_out_match": bool(cmp["total_out_match"]),
                "error_metric": "endpoint_residual" if structure_match else "undefined_structure_mismatch",
                "predicted_side": cmp.get("predicted_side"),
                "constraint_side": cmp.get("constraint_side"),
                "prediction_role": cmp.get("prediction_role"),
                "prediction_role_reason": cmp.get("prediction_role_reason"),
                "total_in_endpoint_signed_rel_err": endpoint_text_fields["total_in_endpoint_signed_rel_err"],
                "total_out_endpoint_signed_rel_err": endpoint_text_fields["total_out_endpoint_signed_rel_err"],
                "total_in_endpoint_unsigned_rel_err": endpoint_text_fields["total_in_endpoint_unsigned_rel_err"],
                "total_out_endpoint_unsigned_rel_err": endpoint_text_fields["total_out_endpoint_unsigned_rel_err"],
                "max_side_endpoint_unsigned_rel_err": endpoint_text_fields["max_side_endpoint_unsigned_rel_err"],
                "total_in_endpoint_signed_rel_err_bps": endpoint_text_fields["total_in_endpoint_signed_rel_err_bps"],
                "total_out_endpoint_signed_rel_err_bps": endpoint_text_fields["total_out_endpoint_signed_rel_err_bps"],
                "total_in_endpoint_unsigned_rel_err_bps": endpoint_text_fields["total_in_endpoint_unsigned_rel_err_bps"],
                "total_out_endpoint_unsigned_rel_err_bps": endpoint_text_fields["total_out_endpoint_unsigned_rel_err_bps"],
                "max_side_endpoint_unsigned_rel_err_bps": endpoint_text_fields["max_side_endpoint_unsigned_rel_err_bps"],
                "legacy_total_in_diff": cmp["total_in_diff"],
                "legacy_total_out_diff": cmp["total_out_diff"],
                "legacy_total_in_rel_err": cmp.get("total_in_rel_err"),
                "legacy_total_out_rel_err": cmp.get("total_out_rel_err"),
                "legacy_total_in_rel_err_bps": cmp.get("total_in_rel_err_bps"),
                "legacy_total_out_rel_err_bps": cmp.get("total_out_rel_err_bps"),
                "legacy_max_side_rel_err": cmp.get("max_side_rel_err"),
                "legacy_max_side_rel_err_bps": cmp.get("max_side_rel_err_bps"),
            }

    def as_dict(self) -> dict[str, Any]:
        total = self.sample_count
        if total == 0:
            return {
                "sample_count": 0,
                "strict_exact_match_count": 0,
                "structure_match_count": 0,
                "first_non_exact_tx": None,
            }
        payload = {
            "sample_count": total,
            "strict_exact_match_count": self.strict_exact_match_count,
            "strict_exact_match_rate": self.strict_exact_match_count / total,
            "structure_match_count": self.structure_match_count,
            "structure_match_rate": self.structure_match_count / total,
            "strict_false_structure_true_count": self.strict_false_structure_true_count,
            "strict_false_structure_true_rate": self.strict_false_structure_true_count / total,
            "structure_false_count": self.structure_false_count,
            "structure_false_rate": self.structure_false_count / total,
            "strict_false_structure_true_in_rel_err_buckets": _ordered_bucket_counts(
                self.strict_false_structure_true_in_rel_err_buckets
            ),
            "strict_false_structure_true_out_rel_err_buckets": _ordered_bucket_counts(
                self.strict_false_structure_true_out_rel_err_buckets
            ),
            "strict_false_structure_true_max_side_rel_err_buckets": _ordered_bucket_counts(
                self.strict_false_structure_true_max_side_rel_err_buckets
            ),
            "structure_matched_in_endpoint_rel_err_buckets": _ordered_bucket_counts(
                self.structure_matched_in_endpoint_rel_err_buckets
            ),
            "structure_matched_out_endpoint_rel_err_buckets": _ordered_bucket_counts(
                self.structure_matched_out_endpoint_rel_err_buckets
            ),
            "structure_matched_max_side_endpoint_rel_err_buckets": _ordered_bucket_counts(
                self.structure_matched_max_side_endpoint_rel_err_buckets
            ),
            "structure_matched_predicted_side_endpoint_rel_err_buckets": _ordered_bucket_counts(
                self.structure_matched_predicted_side_endpoint_rel_err_buckets
            ),
            "structure_matched_constraint_side_endpoint_rel_err_buckets": _ordered_bucket_counts(
                self.structure_matched_constraint_side_endpoint_rel_err_buckets
            ),
            "endpoint_prediction_role_counts": dict(sorted(self.endpoint_prediction_role_counts.items())),
            "endpoint_prediction_role_validation": {
                key: value.as_dict()
                for key, value in sorted(self.endpoint_prediction_role_validation.items())
            },
            "structure_matched_endpoint_ttest_inputs": {
                "unsigned_relative_error": self.structure_matched_unsigned_in_endpoint_rel_err_stats.as_dict(),
                "signed_relative_error": self.structure_matched_signed_in_endpoint_rel_err_stats.as_dict(),
            },
            "structure_matched_endpoint_ttest_outputs": {
                "unsigned_relative_error": self.structure_matched_unsigned_out_endpoint_rel_err_stats.as_dict(),
                "signed_relative_error": self.structure_matched_signed_out_endpoint_rel_err_stats.as_dict(),
            },
            "structure_matched_endpoint_ttest_predicted_side": {
                "unsigned_relative_error": self.structure_matched_unsigned_predicted_side_endpoint_rel_err_stats.as_dict(),
                "signed_relative_error": self.structure_matched_signed_predicted_side_endpoint_rel_err_stats.as_dict(),
            },
            "structure_matched_endpoint_ttest_constraint_side": {
                "unsigned_relative_error": self.structure_matched_unsigned_constraint_side_endpoint_rel_err_stats.as_dict(),
                "signed_relative_error": self.structure_matched_signed_constraint_side_endpoint_rel_err_stats.as_dict(),
            },
            "endpoint_ttest_groups": {
                key: value.as_dict() for key, value in sorted(self.endpoint_ttest_groups.items())
            },
            "first_non_exact_tx": self.first_non_exact_tx,
        }
        if self.top_relative_error_limit > 0:
            payload["top_relative_errors"] = [
                {k: v for k, v in entry.items() if k != "_sort_key"}
                for entry in self.top_relative_errors
            ]
        return payload


def _rel_to_bps(value: Decimal | None) -> Decimal | None:
    if value is None:
        return None
    return value * Decimal(10_000)


def _decimal_from_report_amount_or_none(report_amount: dict[str, Any] | None) -> Decimal | None:
    if report_amount is None:
        return None
    return _exact_report_amount_to_decimal(report_amount)


def _format_decimal_or_none(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return _compact_decimal_text(format(value, "f"))


def _percentile_decimal(values: list[Decimal], percentile: str) -> Decimal | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    numerator = Decimal(percentile) / Decimal(100)
    raw_index = (Decimal(len(values)) - Decimal(1)) * numerator
    index = int(raw_index.to_integral_value(rounding="ROUND_CEILING"))
    return values[min(index, len(values) - 1)]


def _path_shape(legs: list[dict[str, Any]]) -> str:
    has_amm = any(str(leg.get("src") or "") == "AMM" for leg in legs)
    has_clob = any(str(leg.get("src") or "") == "CLOB" for leg in legs)
    if has_amm and has_clob:
        return "AMM+CLOB"
    if has_amm:
        return "AMM_ONLY"
    if has_clob:
        return "CLOB_ONLY"
    return "NO_FILL"


def _leg_count_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 10:
        return "4-10"
    return ">10"


def _intent_mode(intent: dict[str, Any]) -> str:
    tx_type = str(intent.get("transaction_type") or "").upper()
    if tx_type == "OFFERCREATE":
        return "offer_sell" if bool(intent.get("is_sell")) else "offer_buy"
    if tx_type == "PAYMENT":
        return "payment_partial" if bool(intent.get("partial_allowed")) else "payment_full"
    return tx_type.lower() or "unknown"


def _source_funds_limited(intent: dict[str, Any]) -> bool | None:
    tx_type = str(intent.get("transaction_type") or "").upper()
    source_funds_cap = _decimal_from_report_amount_or_none(intent.get("source_funds_cap"))
    if source_funds_cap is None:
        return None
    requested_input: Decimal | None
    if tx_type == "OFFERCREATE":
        requested_input = _decimal_from_report_amount_or_none(intent.get("taker_gets"))
    elif tx_type == "PAYMENT":
        requested_input = _decimal_from_report_amount_or_none(intent.get("sendmax"))
    else:
        requested_input = None
    if requested_input is None:
        return None
    return source_funds_cap < requested_input


def _bool_label(value: Any) -> str:
    if value is None:
        return "null"
    return "true" if bool(value) else "false"


@dataclass(slots=True)
class _IntentErrorGroupStats:
    count: int = 0
    strict_exact_match_count: int = 0
    structure_match_count: int = 0
    structure_false_count: int = 0
    primary_rel_err_buckets: Counter[str] = field(default_factory=Counter)
    cost_rel_err_buckets: Counter[str] = field(default_factory=Counter)
    proceeds_rel_err_buckets: Counter[str] = field(default_factory=Counter)
    max_side_rel_err_buckets: Counter[str] = field(default_factory=Counter)
    primary_rel_err_bps_values: list[Decimal] = field(default_factory=list)
    max_primary_rel_err: Decimal | None = None
    top_primary_errors: list[dict[str, Any]] = field(default_factory=list)

    def update(
        self,
        *,
        report: dict[str, Any],
        cmp: dict[str, Any],
        primary_rel_err: Decimal | None,
        cost_rel_err: Decimal | None,
        proceeds_rel_err: Decimal | None,
        max_side_rel_err: Decimal | None,
        top_limit: int,
    ) -> None:
        self.count += 1
        strict_exact = bool(cmp["strict_exact_match"])
        structure_match = bool(cmp["structure_match"])
        self.strict_exact_match_count += int(strict_exact)
        self.structure_match_count += int(structure_match)
        self.structure_false_count += int(not structure_match)
        self.primary_rel_err_buckets[_relative_error_bucket(primary_rel_err)] += 1
        self.cost_rel_err_buckets[_relative_error_bucket(cost_rel_err)] += 1
        self.proceeds_rel_err_buckets[_relative_error_bucket(proceeds_rel_err)] += 1
        self.max_side_rel_err_buckets[_relative_error_bucket(max_side_rel_err)] += 1
        primary_bps = _rel_to_bps(primary_rel_err)
        if primary_bps is not None:
            self.primary_rel_err_bps_values.append(primary_bps)
        if primary_rel_err is not None and (
            self.max_primary_rel_err is None or primary_rel_err > self.max_primary_rel_err
        ):
            self.max_primary_rel_err = primary_rel_err
        if top_limit > 0 and primary_rel_err is not None:
            intent = report["intent"]
            model_legs = report["model"]["legs"]
            real_legs = report["real"]["legs"]
            entry = {
                "_sort_key": format(primary_rel_err, "f"),
                "tx_hash": report["tx_hash"],
                "ledger_index": report["ledger_index"],
                "transaction_index": report["transaction_index"],
                "transaction_type": intent.get("transaction_type"),
                "direction": f"{intent.get('in_cur')}->{intent.get('out_cur')}",
                "intent_mode": _intent_mode(intent),
                "primary_rel_err_bps": _format_decimal_or_none(primary_bps),
                "error_metric": "endpoint_residual",
                "predicted_side": cmp.get("predicted_side"),
                "constraint_side": cmp.get("constraint_side"),
                "prediction_role": cmp.get("prediction_role"),
                "total_in_endpoint_unsigned_rel_err_bps": cmp.get("total_in_endpoint_unsigned_rel_err_bps"),
                "total_out_endpoint_unsigned_rel_err_bps": cmp.get("total_out_endpoint_unsigned_rel_err_bps"),
                "max_side_endpoint_unsigned_rel_err_bps": cmp.get("max_side_endpoint_unsigned_rel_err_bps"),
                "legacy_total_in_rel_err_bps": cmp.get("total_in_rel_err_bps"),
                "legacy_total_out_rel_err_bps": cmp.get("total_out_rel_err_bps"),
                "legacy_max_side_rel_err_bps": cmp.get("max_side_rel_err_bps"),
                "structure_match": bool(cmp.get("structure_match")),
                "failed_side": cmp.get("failed_side"),
                "first_failed_leg_source_kind": cmp.get("first_failed_leg_source_kind"),
                "model_path_shape": _path_shape(model_legs),
                "real_path_shape": _path_shape(real_legs),
            }
            self.top_primary_errors.append(entry)
            self.top_primary_errors.sort(key=lambda item: Decimal(str(item["_sort_key"])), reverse=True)
            del self.top_primary_errors[top_limit:]

    def as_dict(self) -> dict[str, Any]:
        values = sorted(self.primary_rel_err_bps_values)
        return {
            "count": self.count,
            "strict_exact_match_count": self.strict_exact_match_count,
            "strict_exact_match_rate": self.strict_exact_match_count / self.count if self.count else None,
            "structure_match_count": self.structure_match_count,
            "structure_match_rate": self.structure_match_count / self.count if self.count else None,
            "structure_false_count": self.structure_false_count,
            "structure_false_rate": self.structure_false_count / self.count if self.count else None,
            "primary_rel_err_buckets": _ordered_bucket_counts(self.primary_rel_err_buckets),
            "cost_rel_err_buckets": _ordered_bucket_counts(self.cost_rel_err_buckets),
            "proceeds_rel_err_buckets": _ordered_bucket_counts(self.proceeds_rel_err_buckets),
            "max_side_rel_err_buckets": _ordered_bucket_counts(self.max_side_rel_err_buckets),
            "primary_rel_err_bps_p50": _format_decimal_or_none(_percentile_decimal(values, "50")),
            "primary_rel_err_bps_p95": _format_decimal_or_none(_percentile_decimal(values, "95")),
            "primary_rel_err_bps_p99": _format_decimal_or_none(_percentile_decimal(values, "99")),
            "primary_rel_err_bps_max": _format_decimal_or_none(_rel_to_bps(self.max_primary_rel_err)),
            "top_primary_errors": [
                {k: v for k, v in entry.items() if k != "_sort_key"}
                for entry in self.top_primary_errors
            ],
        }


@dataclass(slots=True)
class _IntentErrorBreakdownBuilder:
    top_limit_per_group: int = 5
    tail_limit: int = 100
    sample_count: int = 0
    dimensions: dict[str, dict[str, _IntentErrorGroupStats]] = field(default_factory=dict)
    tail_primary_errors: list[dict[str, Any]] = field(default_factory=list)

    def _group(self, dimension: str, key: str) -> _IntentErrorGroupStats:
        return self.dimensions.setdefault(dimension, {}).setdefault(key, _IntentErrorGroupStats())

    def _update_group(
        self,
        *,
        dimension: str,
        key: str,
        report: dict[str, Any],
        cmp: dict[str, Any],
        primary_rel_err: Decimal | None,
        cost_rel_err: Decimal | None,
        proceeds_rel_err: Decimal | None,
        max_side_rel_err: Decimal | None,
    ) -> None:
        self._group(dimension, key).update(
            report=report,
            cmp=cmp,
            primary_rel_err=primary_rel_err,
            cost_rel_err=cost_rel_err,
            proceeds_rel_err=proceeds_rel_err,
            max_side_rel_err=max_side_rel_err,
            top_limit=self.top_limit_per_group,
        )

    def update(self, report: dict[str, Any]) -> None:
        self.sample_count += 1
        cmp = _comparison_with_derived_relative_fields(report)
        intent = report["intent"]
        model_legs = report["model"]["legs"]
        real_legs = report["real"]["legs"]
        if bool(cmp["structure_match"]):
            endpoint_rel_fields = _endpoint_residual_relative_fields(model_legs=model_legs, real_legs=real_legs)
            total_in_rel_err = endpoint_rel_fields["total_in_endpoint_unsigned_rel_err_decimal"]
            total_out_rel_err = endpoint_rel_fields["total_out_endpoint_unsigned_rel_err_decimal"]
        else:
            total_in_rel_err = None
            total_out_rel_err = None
        max_side_rel_err = _max_defined_decimal([total_in_rel_err, total_out_rel_err])
        predicted_side = str(cmp.get("predicted_side") or "")
        if predicted_side == "in":
            primary_rel_err = total_in_rel_err
        elif predicted_side == "out":
            primary_rel_err = total_out_rel_err
        else:
            primary_rel_err = None
        cost_rel_err = total_in_rel_err
        proceeds_rel_err = total_out_rel_err

        tx_type = str(intent.get("transaction_type") or "").upper()
        direction = f"{intent.get('in_cur')}->{intent.get('out_cur')}"
        mode = _intent_mode(intent)
        source_limited = _source_funds_limited(intent)
        model_path_shape = _path_shape(model_legs)
        real_path_shape = _path_shape(real_legs)
        model_leg_bucket = _leg_count_bucket(int(cmp.get("model_leg_count") or 0))
        real_leg_bucket = _leg_count_bucket(int(cmp.get("real_leg_count") or 0))

        common_payload = {
            "report": report,
            "cmp": cmp,
            "primary_rel_err": primary_rel_err,
            "cost_rel_err": cost_rel_err,
            "proceeds_rel_err": proceeds_rel_err,
            "max_side_rel_err": max_side_rel_err,
        }
        keys: list[tuple[str, str]] = [
            ("by_intent_mode", f"{tx_type}|{mode}"),
            ("by_direction_and_mode", f"{direction}|{mode}"),
            (
                "by_intent_field_combo",
                f"{tx_type}|{direction}|{mode}|source_limited={_bool_label(source_limited)}",
            ),
            ("by_execution_shape", f"model={model_path_shape}|real={real_path_shape}"),
            ("by_execution_shape", f"structure_match={_bool_label(cmp.get('structure_match'))}"),
            (
                "by_execution_shape",
                f"model_legs={model_leg_bucket}|real_legs={real_leg_bucket}",
            ),
            ("by_execution_shape", f"first_failed={cmp.get('first_failed_leg_source_kind') or 'none'}"),
            ("by_execution_shape", f"failed_side={cmp.get('failed_side') or 'none'}"),
        ]
        if tx_type == "OFFERCREATE":
            for field_name in ("is_sell", "is_ioc", "is_fok", "is_passive"):
                keys.append(("by_restriction_flags", f"{field_name}={_bool_label(intent.get(field_name))}"))
            keys.append(("by_restriction_flags", f"has_cancel_sequence={_bool_label(intent.get('cancel_sequence') is not None)}"))
            keys.append(
                (
                    "by_intent_field_combo",
                    f"{tx_type}|{direction}|{mode}|ioc={_bool_label(intent.get('is_ioc'))}|"
                    f"fok={_bool_label(intent.get('is_fok'))}|passive={_bool_label(intent.get('is_passive'))}|"
                    f"source_limited={_bool_label(source_limited)}",
                )
            )
        elif tx_type == "PAYMENT":
            for field_name in ("partial_allowed", "no_ripple_direct", "paths_present", "paths_has_external_asset"):
                keys.append(("by_restriction_flags", f"{field_name}={_bool_label(intent.get(field_name))}"))
            keys.append(("by_restriction_flags", f"has_delivermin={_bool_label(intent.get('delivermin') is not None)}"))
            keys.append(("by_restriction_flags", f"paths_count={intent.get('paths_count') or 0}"))
            keys.append(
                (
                    "by_intent_field_combo",
                    f"{tx_type}|{direction}|{mode}|delivermin={_bool_label(intent.get('delivermin') is not None)}|"
                    f"paths={_bool_label(intent.get('paths_present'))}|"
                    f"source_limited={_bool_label(source_limited)}",
                )
            )
        keys.append(("by_restriction_flags", f"source_limited={_bool_label(source_limited)}"))

        for dimension, key in keys:
            self._update_group(dimension=dimension, key=key, **common_payload)

        if primary_rel_err is not None:
            primary_bps = _rel_to_bps(primary_rel_err)
            entry = {
                "_sort_key": format(primary_rel_err, "f"),
                "tx_hash": report["tx_hash"],
                "ledger_index": report["ledger_index"],
                "transaction_index": report["transaction_index"],
                "transaction_type": tx_type,
                "direction": direction,
                "intent_mode": mode,
                "source_limited": source_limited,
                "primary_rel_err_bps": _format_decimal_or_none(primary_bps),
                "error_metric": "endpoint_residual",
                "predicted_side": cmp.get("predicted_side"),
                "constraint_side": cmp.get("constraint_side"),
                "prediction_role": cmp.get("prediction_role"),
                "total_in_endpoint_unsigned_rel_err_bps": cmp.get("total_in_endpoint_unsigned_rel_err_bps"),
                "total_out_endpoint_unsigned_rel_err_bps": cmp.get("total_out_endpoint_unsigned_rel_err_bps"),
                "max_side_endpoint_unsigned_rel_err_bps": cmp.get("max_side_endpoint_unsigned_rel_err_bps"),
                "legacy_total_in_rel_err_bps": cmp.get("total_in_rel_err_bps"),
                "legacy_total_out_rel_err_bps": cmp.get("total_out_rel_err_bps"),
                "legacy_max_side_rel_err_bps": cmp.get("max_side_rel_err_bps"),
                "structure_match": bool(cmp.get("structure_match")),
                "model_path_shape": model_path_shape,
                "real_path_shape": real_path_shape,
                "first_failed_leg_source_kind": cmp.get("first_failed_leg_source_kind"),
                "failed_side": cmp.get("failed_side"),
            }
            self.tail_primary_errors.append(entry)
            self.tail_primary_errors.sort(key=lambda item: Decimal(str(item["_sort_key"])), reverse=True)
            del self.tail_primary_errors[self.tail_limit :]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "metric_semantics": {
                "primary_rel_err": "intent-facing endpoint residual error among structure-matched transactions; currently output-side endpoint residual for supported direct intents",
                "cost_rel_err": "input-side endpoint residual relative error among structure-matched transactions",
                "proceeds_rel_err": "output-side endpoint residual relative error among structure-matched transactions",
                "max_side_rel_err": "max(input-side endpoint residual, output-side endpoint residual) among structure-matched transactions",
            },
            "dimensions": {
                dimension: {
                    key: stats.as_dict()
                    for key, stats in sorted(groups.items(), key=lambda item: (-item[1].count, item[0]))
                }
                for dimension, groups in sorted(self.dimensions.items())
            },
            "top_primary_errors": [
                {k: v for k, v in entry.items() if k != "_sort_key"}
                for entry in self.tail_primary_errors
            ],
        }


def _write_summary_json_from_reports(
    *,
    sample_hashes: list[str],
    reports_dir: Path,
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        first = True
        for tx_hash in sample_hashes:
            report_path = reports_dir / f"{tx_hash}.json"
            if not report_path.exists():
                raise RuntimeError(f"missing report while building summary.json: {report_path}")
            if not first:
                handle.write(",\n")
            handle.write(report_path.read_text(encoding="utf-8"))
            first = False
        handle.write("\n]\n")


def _write_summary_markdown_from_reports(
    *,
    sample_hashes: list[str],
    reports_dir: Path,
    output_path: Path,
) -> None:
    lines = [
        "# Single Direct-Pair Fit Summary",
        "",
        "| tx | type | dir | model legs | real legs | strict exact | structure | endpoint in signed rel err | endpoint out signed rel err | endpoint max unsigned rel err | legacy total in diff | legacy total out diff |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for tx_hash in sample_hashes:
        report_path = reports_dir / f"{tx_hash}.json"
        if not report_path.exists():
            raise RuntimeError(f"missing report while building summary.md: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cmp = _comparison_with_derived_relative_fields(report)
        lines.append(
            "| "
            + " | ".join(
                [
                    report["tx_hash"][:12],
                    report["intent"]["transaction_type"],
                    f'{_pretty_currency(report["intent"]["in_cur"])}->{_pretty_currency(report["intent"]["out_cur"])}',
                    str(cmp["model_leg_count"]),
                    str(cmp["real_leg_count"]),
                    "yes" if cmp["strict_exact_match"] else "no",
                    "yes" if cmp["structure_match"] else "no",
                    str(cmp.get("total_in_endpoint_signed_rel_err")),
                    str(cmp.get("total_out_endpoint_signed_rel_err")),
                    str(cmp.get("max_side_endpoint_unsigned_rel_err")),
                    cmp["total_in_diff"],
                    cmp["total_out_diff"],
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


XRPL_QUALITY_ONE = 1_000_000_000
XRPL_MAX_TRANSFER_RATE = 2_000_000_000


def _issuer_transfer_rate_from_raw(value: Any, *, field_name: str) -> IOUAmount:
    raw: Any
    if isinstance(value, dict):
        if "TransferRate" in value:
            raw = value["TransferRate"]
        elif "transfer_rate" in value:
            raw = value["transfer_rate"]
        else:
            raise RuntimeError(f"{field_name} must contain raw TransferRate")
    else:
        raw = value

    if isinstance(raw, bool):
        raise RuntimeError(f"{field_name} must be a raw integer TransferRate")
    if isinstance(raw, int):
        raw_int = raw
    elif isinstance(raw, str) and raw.strip().isdigit():
        raw_int = int(raw.strip())
    else:
        raise RuntimeError(
            f"{field_name} must be a raw XRPL TransferRate integer, not a decimal rate"
        )

    if raw_int < XRPL_QUALITY_ONE or raw_int > XRPL_MAX_TRANSFER_RATE:
        raise RuntimeError(
            f"{field_name} out of XRPL AccountRoot.TransferRate range "
            f"[{XRPL_QUALITY_ONE}, {XRPL_MAX_TRANSFER_RATE}]: {raw_int}"
        )
    return IOUAmount(raw_int, -9)


def _load_issuer_transfer_rates(
    path_raw: str | None,
) -> tuple[dict[tuple[str, str], IOUAmount], dict[tuple[str, str], list[dict[str, Any]]]]:
    if not path_raw:
        return {}, {}
    path = Path(path_raw)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rates = None
    raw_ranges = None
    if isinstance(payload, dict) and (
        "issuer_transfer_rates" in payload or "issuer_transfer_rate_ranges" in payload
    ):
        raw_rates = payload.get("issuer_transfer_rates")
        raw_ranges = payload.get("issuer_transfer_rate_ranges")
    elif isinstance(payload, dict) and all(":" in str(key) for key in payload):
        raw_rates = payload
    else:
        return {}, {}
    if raw_rates is not None and not isinstance(raw_rates, dict):
        raise RuntimeError(f"issuer transfer rates JSON must be an object: {path}")
    if raw_ranges is not None and not isinstance(raw_ranges, dict):
        raise RuntimeError(f"issuer transfer rate ranges JSON must be an object: {path}")

    rates: dict[tuple[str, str], IOUAmount] = {}
    for raw_key, raw_value in (raw_rates or {}).items():
        key = str(raw_key)
        if ":" not in key:
            raise RuntimeError(f"invalid issuer transfer rate key {key!r}; expected '<currency>:<issuer>'")
        currency, issuer = key.rsplit(":", 1)
        if not currency or not issuer:
            raise RuntimeError(f"invalid issuer transfer rate key {key!r}; expected '<currency>:<issuer>'")
        rates[(currency, issuer)] = _issuer_transfer_rate_from_raw(
            raw_value,
            field_name=f"issuer_transfer_rates[{key}]",
        )

    ranges: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for raw_key, entries in (raw_ranges or {}).items():
        key = str(raw_key)
        if ":" not in key:
            raise RuntimeError(f"invalid issuer transfer rate range key {key!r}; expected '<currency>:<issuer>'")
        currency, issuer = key.rsplit(":", 1)
        if not currency or not issuer:
            raise RuntimeError(f"invalid issuer transfer rate range key {key!r}; expected '<currency>:<issuer>'")
        if not isinstance(entries, list) or not entries:
            raise RuntimeError(f"issuer_transfer_rate_ranges[{key}] must be a non-empty list")
        parsed_entries: list[dict[str, Any]] = []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise RuntimeError(f"issuer_transfer_rate_ranges[{key}][{idx}] must be an object")
            first = int(entry.get("first_ledger"))
            last = int(entry.get("last_ledger"))
            if first > last:
                raise RuntimeError(f"issuer_transfer_rate_ranges[{key}][{idx}] has first_ledger > last_ledger")
            parsed_entries.append(
                {
                    "first_ledger": first,
                    "last_ledger": last,
                    "rate": _issuer_transfer_rate_from_raw(
                        entry,
                        field_name=f"issuer_transfer_rate_ranges[{key}][{idx}]",
                    ),
                }
            )
        parsed_entries.sort(key=lambda item: item["first_ledger"])
        ranges[(currency, issuer)] = parsed_entries
    return rates, ranges


def _load_amendment_feature_flags(path_raw: str | None) -> dict[str, bool]:
    if not path_raw:
        return {}
    path = Path(path_raw)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    raw_flags = payload.get("amendment_feature_flags") or payload.get("feature_flags")
    if raw_flags is None:
        return {}
    if not isinstance(raw_flags, dict):
        raise RuntimeError(f"amendment feature flags must be an object: {path}")
    return {str(key): bool(value) for key, value in raw_flags.items()}


def _transfer_rate_for_asset(
    *,
    currency: str,
    issuer: str | None,
    issuer_transfer_rates: dict[tuple[str, str], IOUAmount],
    issuer_transfer_rate_ranges: dict[tuple[str, str], list[dict[str, Any]]],
    ledger_index: int,
) -> IOUAmount:
    if currency == XRP:
        return IOUAmount(1, 0)
    if not issuer:
        raise RuntimeError(f"missing issuer for non-XRP transfer-rate lookup: {currency}")
    key = (currency, issuer)
    for entry in issuer_transfer_rate_ranges.get(key, []):
        if entry["first_ledger"] <= ledger_index <= entry["last_ledger"]:
            return entry["rate"]
    rate = issuer_transfer_rates.get(key)
    if rate is not None:
        return rate
    raise RuntimeError(
        f"missing raw TransferRate for issuer asset {currency}:{issuer}; "
        "dataset manifest must include issuer_transfer_rates or issuer_transfer_rate_ranges "
        "with AccountRoot.TransferRate integer"
    )


def _build_model_amm(
    *,
    tx_hash: str,
    metadata_result: dict[str, Any],
    amm_rows: list[dict[str, Any]],
    fee_rows: list[dict[str, Any]],
    in_cur: str,
    out_cur: str,
) -> tuple[AMM | None, dict[str, Any] | None]:
    if not amm_rows:
        return None, None
    if len(amm_rows) != 1:
        raise RuntimeError(f"expected at most one AMM row for tx={tx_hash}, got {len(amm_rows)}")
    if len(fee_rows) != 1:
        raise RuntimeError(f"expected exactly one AMM fee row for tx={tx_hash}, got {len(fee_rows)}")

    amm_row = amm_rows[0]
    fee_row = fee_rows[0]
    amm_asset_currency = str(amm_row.get("amm_asset_currency") or "")
    amm_asset2_currency = str(amm_row.get("amm_asset2_currency") or "")
    if not amm_asset_currency or not amm_asset2_currency:
        raise RuntimeError(f"AMM row missing pool asset currencies for tx={tx_hash}")

    amm_in_currency = str(amm_row.get("asset_in_currency") or "")
    amm_out_currency = str(amm_row.get("asset_out_currency") or "")
    if amm_in_currency != in_cur or amm_out_currency != out_cur:
        raise RuntimeError(
            f"AMM row direction does not match intent for tx={tx_hash}: "
            f"row={amm_in_currency}->{amm_out_currency}, intent={in_cur}->{out_cur}"
        )

    fee_units = _effective_amm_fee(
        fee_row=fee_row,
        amm_in_currency=amm_in_currency,
        amm_in_value_raw=amm_row.get("asset_in_value"),
    )

    parquet_reserve_a_before = (
        parse_xrp_decimal(amm_row.get("amm_asset_balance_before"), field_name="amm_asset_balance_before")
        if amm_asset_currency == XRP
        else parse_iou_decimal(amm_row.get("amm_asset_balance_before"), field_name="amm_asset_balance_before")
    )
    parquet_reserve_b_before = (
        parse_xrp_decimal(amm_row.get("amm_asset2_balance_before"), field_name="amm_asset2_balance_before")
        if amm_asset2_currency == XRP
        else parse_iou_decimal(amm_row.get("amm_asset2_balance_before"), field_name="amm_asset2_balance_before")
    )
    exact_state = _extract_exact_amm_pre_state_from_metadata(
        result=metadata_result,
        expected_amm_account=str(amm_row.get("amm_account") or ""),
        amm_asset_currency=amm_asset_currency,
        amm_asset_issuer=str(amm_row.get("amm_asset_issuer") or "") or None,
        amm_asset2_currency=amm_asset2_currency,
        amm_asset2_issuer=str(amm_row.get("amm_asset2_issuer") or "") or None,
    )
    if exact_state is None:
        raise RuntimeError(f"missing metadata-exact AMM pre-state for tx={tx_hash}")
    reserve_a_before = exact_state["reserve_a_before"]
    reserve_b_before = exact_state["reserve_b_before"]
    state_source = str(exact_state["source"])
    if in_cur == amm_asset_currency and out_cur == amm_asset2_currency:
        x_reserve = reserve_a_before
        y_reserve = reserve_b_before
        x_is_xrp = _is_xrp(amm_asset_currency)
        y_is_xrp = _is_xrp(amm_asset2_currency)
    elif in_cur == amm_asset2_currency and out_cur == amm_asset_currency:
        x_reserve = reserve_b_before
        y_reserve = reserve_a_before
        x_is_xrp = _is_xrp(amm_asset2_currency)
        y_is_xrp = _is_xrp(amm_asset_currency)
    else:
        raise RuntimeError(
            f"AMM pool assets do not match intent direction for tx={tx_hash}: "
            f"pool={amm_asset_currency}/{amm_asset2_currency}, intent={in_cur}->{out_cur}"
        )

    state_report = {
        "state_source": state_source,
        "amm_account": str(amm_row.get("amm_account") or ""),
        "effective_fee": _fee_units_to_decimal_text(fee_units),
        "amm_asset_currency": amm_asset_currency,
        "amm_asset2_currency": amm_asset2_currency,
        "parquet_reserve_a_before": _amount_to_string(parquet_reserve_a_before),
        "parquet_reserve_b_before": _amount_to_string(parquet_reserve_b_before),
        "reserve_a_before_used": _amount_to_string(reserve_a_before),
        "reserve_b_before_used": _amount_to_string(reserve_b_before),
    }
    if exact_state is not None:
        state_report["metadata_exact"] = {
            "xrp_reserve_drops_before": int(exact_state["xrp_reserve_drops_before"]),
            "iou_reserve_before": _amount_to_string(exact_state["iou_reserve_before"]),
            "iou_currency": str(exact_state["iou_currency"]),
            "iou_issuer": exact_state["iou_issuer"],
        }

    return AMM(
        x_reserve=x_reserve,
        y_reserve=y_reserve,
        fee=fee_units,
        x_is_xrp=x_is_xrp,
        y_is_xrp=y_is_xrp,
    ), state_report


def _amount_report_key(report: dict[str, Any]) -> tuple[Any, ...]:
    kind = str(report.get("kind") or "")
    if kind == "XRP":
        return ("XRP", int(report["drops"]))
    return ("IOU", int(report["mantissa"]), int(report["exponent"]))


def _build_real_offer_legs(result: dict[str, Any], *, in_cur: str, out_cur: str) -> list[dict[str, Any]]:
    meta = result.get("meta")
    if not isinstance(meta, dict):
        return []
    affected = meta.get("AffectedNodes")
    if not isinstance(affected, list):
        return []

    def account_states_for_iou_node(
        body: dict[str, Any],
    ) -> list[tuple[tuple[str, str, str], IOUAmount, IOUAmount]]:
        final = body.get("FinalFields") or body.get("NewFields") or {}
        low_limit = _metadata_pre_state_field(body, "LowLimit")
        high_limit = _metadata_pre_state_field(body, "HighLimit")
        if not isinstance(low_limit, dict) or not isinstance(high_limit, dict):
            return []
        currency = str(low_limit.get("currency") or high_limit.get("currency") or "")
        low_account = str(low_limit.get("issuer") or "")
        high_account = str(high_limit.get("issuer") or "")
        if not currency or not low_account or not high_account:
            return []
        pre_balance_raw = _metadata_pre_state_field(body, "Balance")
        post_balance_raw = final.get("Balance", pre_balance_raw)
        if not isinstance(pre_balance_raw, dict) or not isinstance(post_balance_raw, dict):
            return []
        pre_balance = parse_iou_decimal(pre_balance_raw.get("value"), field_name="RippleState Previous Balance")
        post_balance = parse_iou_decimal(post_balance_raw.get("value"), field_name="RippleState Final Balance")
        return [
            ((low_account, currency, high_account), pre_balance, post_balance),
            (
                (high_account, currency, low_account),
                IOUAmount(-pre_balance.mantissa, pre_balance.exponent),
                IOUAmount(-post_balance.mantissa, post_balance.exponent),
            ),
        ]

    iou_account_states: dict[tuple[str, str, str], tuple[IOUAmount, IOUAmount]] = {}
    for wrapped in affected:
        if not isinstance(wrapped, dict):
            continue
        body = _metadata_node_body(wrapped)
        if body is None or str(body.get("LedgerEntryType") or "") != "RippleState":
            continue
        try:
            parsed_states = account_states_for_iou_node(body)
        except Exception:
            continue
        for key, pre_balance, post_balance in parsed_states:
            if pre_balance != post_balance:
                iou_account_states[key] = (pre_balance, post_balance)

    side_counts: Counter[tuple[str, str, str, str]] = Counter()
    pre_parsed_offers: list[tuple[str, dict[str, Any], dict[str, Any], str, str, str | None, str, str | None]] = []
    for wrapped in affected:
        if not isinstance(wrapped, dict):
            continue
        node_kind = None
        body = None
        for candidate in ("ModifiedNode", "DeletedNode"):
            maybe = wrapped.get(candidate)
            if isinstance(maybe, dict):
                node_kind = candidate
                body = maybe
                break
        if not isinstance(body, dict):
            continue
        if str(body.get("LedgerEntryType") or "") != "Offer":
            continue
        final_fields = body.get("FinalFields")
        if not isinstance(final_fields, dict):
            continue
        if node_kind not in {"ModifiedNode", "DeletedNode"}:
            continue
        pre_offer = _metadata_pre_offer_from_node(body)
        if not isinstance(pre_offer, dict):
            continue

        pre_gets_cur, pre_gets_issuer, pre_gets_amt = _parse_amount(
            pre_offer.get("TakerGets"),
            field_name="Offer.PreviousFields.TakerGets",
        )
        pre_pays_cur, pre_pays_issuer, pre_pays_amt = _parse_amount(
            pre_offer.get("TakerPays"),
            field_name="Offer.PreviousFields.TakerPays",
        )
        post_gets_cur, post_gets_issuer, post_gets_amt = _parse_amount(
            final_fields.get("TakerGets"),
            field_name="Offer.FinalFields.TakerGets",
        )
        post_pays_cur, post_pays_issuer, post_pays_amt = _parse_amount(
            final_fields.get("TakerPays"),
            field_name="Offer.FinalFields.TakerPays",
        )
        if (
            pre_gets_cur != post_gets_cur
            or pre_gets_issuer != post_gets_issuer
            or pre_pays_cur != post_pays_cur
            or pre_pays_issuer != post_pays_issuer
        ):
            raise RuntimeError("offer asset changed across metadata node")
        if pre_gets_cur != out_cur or pre_pays_cur != in_cur:
            continue
        owner = str(final_fields.get("Account") or "")
        pre_parsed_offers.append((node_kind, body, pre_offer, owner, pre_gets_cur, pre_gets_issuer, pre_pays_cur, pre_pays_issuer))
        if owner and pre_gets_cur != XRP and pre_gets_issuer:
            side_counts[(owner, "out", pre_gets_cur, pre_gets_issuer)] += 1
        if owner and pre_pays_cur != XRP and pre_pays_issuer:
            side_counts[(owner, "in", pre_pays_cur, pre_pays_issuer)] += 1

    offer_legs: list[dict[str, Any]] = []
    for node_kind, body, pre_offer, owner, pre_gets_cur, pre_gets_issuer, pre_pays_cur, pre_pays_issuer in pre_parsed_offers:
        final_fields = body.get("FinalFields")
        if not isinstance(final_fields, dict):
            continue
        pre_gets_cur, pre_gets_issuer, pre_gets_amt = _parse_amount(
            pre_offer.get("TakerGets"),
            field_name="Offer.PreviousFields.TakerGets",
        )
        pre_pays_cur, pre_pays_issuer, pre_pays_amt = _parse_amount(
            pre_offer.get("TakerPays"),
            field_name="Offer.PreviousFields.TakerPays",
        )
        post_gets_cur, post_gets_issuer, post_gets_amt = _parse_amount(
            final_fields.get("TakerGets"),
            field_name="Offer.FinalFields.TakerGets",
        )
        post_pays_cur, post_pays_issuer, post_pays_amt = _parse_amount(
            final_fields.get("TakerPays"),
            field_name="Offer.FinalFields.TakerPays",
        )
        if node_kind == "DeletedNode":
            # A deleted offer is no longer executable once either side reaches
            # zero. Metadata may still expose a tiny residual on the opposite
            # side in FinalFields; do not treat that deleted dust as remaining
            # live offer state for endpoint matching.
            if _positive_amount(post_gets_amt) and not _positive_amount(post_pays_amt):
                post_gets_amt = _zero_like(post_gets_amt)
            elif _positive_amount(post_pays_amt) and not _positive_amount(post_gets_amt):
                post_pays_amt = _zero_like(post_pays_amt)
        out_take = _amount_delta(pre_gets_amt, post_gets_amt)
        if not _positive_amount(out_take):
            continue
        if owner and pre_gets_cur != XRP and pre_gets_issuer:
            key = (owner, "out", pre_gets_cur, pre_gets_issuer)
            account_states = iou_account_states.get((owner, pre_gets_cur, pre_gets_issuer))
            account_delta = account_states[1] - account_states[0] if account_states is not None else None
            if side_counts[key] == 1 and isinstance(account_delta, IOUAmount) and account_delta.mantissa < 0:
                out_take = IOUAmount(-account_delta.mantissa, account_delta.exponent)
                out_report = _with_state_source(
                    _state_transition_report(
                        pre_state=account_states[0],
                        post_state=account_states[1],
                        state_change=STATE_CHANGE_DECREASE,
                        currency=pre_gets_cur,
                        issuer=pre_gets_issuer,
                    ),
                    "account_delta",
                )
            else:
                out_report = _with_state_source(
                    _state_transition_report(
                        pre_state=pre_gets_amt,
                        post_state=post_gets_amt,
                        state_change=STATE_CHANGE_DECREASE,
                        currency=pre_gets_cur,
                        issuer=pre_gets_issuer,
                    ),
                    "offer_node",
                )
        else:
            out_report = _with_state_source(
                _state_transition_report(
                    pre_state=pre_gets_amt,
                    post_state=post_gets_amt,
                    state_change=STATE_CHANGE_DECREASE,
                    currency=pre_gets_cur,
                    issuer=pre_gets_issuer,
                ),
                "offer_node",
            )
        in_take = _amount_delta(pre_pays_amt, post_pays_amt)
        if not _positive_amount(in_take):
            continue
        if owner and pre_pays_cur != XRP and pre_pays_issuer:
            key = (owner, "in", pre_pays_cur, pre_pays_issuer)
            account_states = iou_account_states.get((owner, pre_pays_cur, pre_pays_issuer))
            account_delta = account_states[1] - account_states[0] if account_states is not None else None
            if side_counts[key] == 1 and isinstance(account_delta, IOUAmount) and account_delta.mantissa > 0:
                in_take = account_delta
                in_report = _with_state_source(
                    _state_transition_report(
                        pre_state=account_states[0],
                        post_state=account_states[1],
                        state_change=STATE_CHANGE_INCREASE,
                        currency=pre_pays_cur,
                        issuer=pre_pays_issuer,
                    ),
                    "account_delta",
                )
            else:
                in_report = _with_state_source(
                    _state_transition_report(
                        pre_state=pre_pays_amt,
                        post_state=post_pays_amt,
                        state_change=STATE_CHANGE_DECREASE,
                        currency=pre_pays_cur,
                        issuer=pre_pays_issuer,
                    ),
                    "offer_node",
                )
        else:
            in_report = _with_state_source(
                _state_transition_report(
                    pre_state=pre_pays_amt,
                    post_state=post_pays_amt,
                    state_change=STATE_CHANGE_DECREASE,
                    currency=pre_pays_cur,
                    issuer=pre_pays_issuer,
                ),
                "offer_node",
            )
        offer_legs.append(
            {
                "src": "CLOB",
                "offer_id": str(body.get("LedgerIndex") or ""),
                "owner": owner,
                "out": out_report,
                "in": in_report,
                "quality": _quality_to_string(Quality.from_amounts(out_take, in_take)),
            }
        )
    return sorted(offer_legs, key=lambda leg: (leg.get("offer_id") or "", leg["out"]["value"], leg["in"]["value"]))


def _amount_from_decimal(value: Decimal, *, currency: str) -> Amount:
    if _is_xrp(currency):
        drops = int((value * Decimal(1_000_000)).to_integral_value())
        return XRPAmount(drops)
    return _decimal_to_iou_amount(value)


def _build_real_amm_legs_from_metadata(
    result: dict[str, Any],
    *,
    in_cur: str,
    out_cur: str,
) -> list[dict[str, Any]] | None:
    meta = result.get("meta")
    if not isinstance(meta, dict):
        return None
    affected = meta.get("AffectedNodes")
    if not isinstance(affected, list):
        return None

    pool_account: str | None = None
    xrp_pre: int | None = None
    xrp_post: int | None = None
    for wrapped in affected:
        body = _metadata_node_body(wrapped)
        if body is None or str(body.get("LedgerEntryType") or "") != "AccountRoot":
            continue
        previous = body.get("PreviousFields") or {}
        final = body.get("FinalFields") or body.get("NewFields") or {}
        if "AMMID" not in previous and "AMMID" not in final:
            continue
        account = str(final.get("Account") or previous.get("Account") or "")
        if not account:
            continue
        pre_balance = _metadata_pre_state_field(body, "Balance")
        post_balance = final.get("Balance")
        if pre_balance is None or post_balance is None:
            continue
        try:
            xrp_pre = int(str(pre_balance))
            xrp_post = int(str(post_balance))
        except Exception:
            continue
        pool_account = account
        break

    if pool_account is None or xrp_pre is None or xrp_post is None:
        return None

    iou_pre: Decimal | None = None
    iou_post: Decimal | None = None
    for wrapped in affected:
        body = _metadata_node_body(wrapped)
        if body is None or str(body.get("LedgerEntryType") or "") != "RippleState":
            continue
        low_limit = _metadata_pre_state_field(body, "LowLimit")
        high_limit = _metadata_pre_state_field(body, "HighLimit")
        pre_balance = _metadata_pre_state_field(body, "Balance")
        final = body.get("FinalFields") or body.get("NewFields") or {}
        post_balance = final.get("Balance", pre_balance)
        if not isinstance(low_limit, dict) or not isinstance(high_limit, dict):
            continue
        if not isinstance(pre_balance, dict) or not isinstance(post_balance, dict):
            continue
        low_account = str(low_limit.get("issuer") or "")
        high_account = str(high_limit.get("issuer") or "")
        if pool_account not in {low_account, high_account}:
            continue
        try:
            raw_pre = Decimal(str(pre_balance.get("value")))
            raw_post = Decimal(str(post_balance.get("value")))
        except Exception:
            continue
        reserve_pre = raw_pre if low_account == pool_account else -raw_pre
        reserve_post = raw_post if low_account == pool_account else -raw_post
        if reserve_pre <= 0 or reserve_post <= 0:
            continue
        iou_pre = reserve_pre
        iou_post = reserve_post
        break

    if iou_pre is None or iou_post is None:
        return None

    iou_currency = out_cur if not _is_xrp(out_cur) else in_cur
    xrp_pre_amt = XRPAmount(xrp_pre)
    xrp_post_amt = XRPAmount(xrp_post)
    iou_pre_amt = _amount_from_decimal(iou_pre, currency=iou_currency)
    iou_post_amt = _amount_from_decimal(iou_post, currency=iou_currency)
    xrp_delta = _state_delta_amount(
        pre_state=xrp_pre_amt,
        post_state=xrp_post_amt,
        state_change=STATE_CHANGE_INCREASE if xrp_post > xrp_pre else STATE_CHANGE_DECREASE,
    )
    iou_delta = _state_delta_amount(
        pre_state=iou_pre_amt,
        post_state=iou_post_amt,
        state_change=STATE_CHANGE_INCREASE if iou_post > iou_pre else STATE_CHANGE_DECREASE,
    )
    if not _positive_amount(xrp_delta) or not _positive_amount(iou_delta):
        return None

    if _is_xrp(in_cur) and not _is_xrp(out_cur):
        in_take = xrp_delta
        out_take = iou_delta
        out_report = _state_transition_report(
            pre_state=iou_pre_amt,
            post_state=iou_post_amt,
            state_change=STATE_CHANGE_DECREASE,
            currency=out_cur,
            issuer=None,
        )
        in_report = _state_transition_report(
            pre_state=xrp_pre_amt,
            post_state=xrp_post_amt,
            state_change=STATE_CHANGE_INCREASE,
            currency=in_cur,
            issuer=None,
        )
    elif not _is_xrp(in_cur) and _is_xrp(out_cur):
        in_take = iou_delta
        out_take = xrp_delta
        in_report = _state_transition_report(
            pre_state=iou_pre_amt,
            post_state=iou_post_amt,
            state_change=STATE_CHANGE_INCREASE,
            currency=in_cur,
            issuer=None,
        )
        out_report = _state_transition_report(
            pre_state=xrp_pre_amt,
            post_state=xrp_post_amt,
            state_change=STATE_CHANGE_DECREASE,
            currency=out_cur,
            issuer=None,
        )
    else:
        return None

    if not _positive_amount(in_take) or not _positive_amount(out_take):
        return None

    return [
        {
            "src": "AMM",
            "amm_account": pool_account,
            "out": out_report,
            "in": in_report,
            "quality": _quality_to_string(Quality.from_amounts(out_take, in_take)),
            "source": "metadata_exact_delta",
        }
    ]


def _sum_legs(legs: list[dict[str, Any]], *, in_cur: str, out_cur: str) -> tuple[Amount, Amount]:
    total_out: Amount | None = None
    total_in: Amount | None = None
    for leg in legs:
        out_amount = _amount_from_report_dict(leg["out"])
        in_amount = _amount_from_report_dict(leg["in"])
        total_out = out_amount if total_out is None else total_out + out_amount
        total_in = in_amount if total_in is None else total_in + in_amount
    if total_out is None:
        total_out = _zero_like(XRPAmount(0) if _is_xrp(out_cur) else IOUAmount(0, 0))
    if total_in is None:
        total_in = _zero_like(XRPAmount(0) if _is_xrp(in_cur) else IOUAmount(0, 0))
    return total_out, total_in


def _amount_from_report_dict(report: dict[str, Any]) -> Amount:
    kind = str(report.get("kind") or "")
    if kind == "XRP":
        return XRPAmount(int(report["drops"]))
    return IOUAmount(int(report["mantissa"]), int(report["exponent"]))


def _exact_report_amount_key(report: dict[str, Any]) -> tuple[str, int, int]:
    kind = str(report.get("kind") or "")
    if kind == "XRP":
        return ("XRP", int(report["drops"]), -6)
    return ("IOU", int(report["mantissa"]), int(report["exponent"]))


def _exact_report_amount_to_decimal(report: dict[str, Any]) -> Decimal:
    kind, units, exponent = _exact_report_amount_key(report)
    if kind == "XRP":
        return Decimal(units) / Decimal(1_000_000)
    return Decimal(units) * (Decimal(10) ** exponent)


def _sum_report_amounts_decimal(reports: list[dict[str, Any]]) -> Decimal:
    total = Decimal(0)
    for report in reports:
        total += _exact_report_amount_to_decimal(report)
    return total


def _exact_report_totals_decimal(legs: list[dict[str, Any]]) -> tuple[Decimal, Decimal]:
    if not legs:
        return Decimal(0), Decimal(0)
    return (
        _sum_report_amounts_decimal([leg["out"] for leg in legs]),
        _sum_report_amounts_decimal([leg["in"] for leg in legs]),
    )


def _fit_leg_identity(leg: dict[str, Any]) -> tuple[str, str]:
    src = str(leg.get("src") or "")
    if src == "CLOB":
        source_id = str(leg.get("source_id") or leg.get("offer_id") or "")
        return src, source_id
    return src, "AMM"


def _state_report_sort_key(report: dict[str, Any]) -> tuple[Any, ...]:
    pre_state = report.get("pre_state")
    post_state = report.get("post_state")
    if not isinstance(pre_state, dict) or not isinstance(post_state, dict):
        raise RuntimeError("missing canonical pre_state/post_state in real-side report")
    return (
        str(report.get("state_change") or ""),
        _amount_report_key(pre_state),
        _amount_report_key(post_state),
    )


def _project_model_post_state(
    *,
    pre_state: Amount,
    model_amount: Amount,
    state_change: str,
) -> Amount:
    if state_change == STATE_CHANGE_DECREASE:
        return amount_sub_ripple_number(pre_state, model_amount)
    if state_change == STATE_CHANGE_INCREASE:
        return amount_add_ripple_number(pre_state, model_amount)
    raise RuntimeError(f"unsupported state_change={state_change}")


def _state_transition_matches_model(
    *,
    model_amount_report: dict[str, Any],
    real_amount_report: dict[str, Any],
) -> bool:
    pre_state_report = real_amount_report.get("pre_state")
    post_state_report = real_amount_report.get("post_state")
    state_change = str(real_amount_report.get("state_change") or "")
    if not isinstance(pre_state_report, dict) or not isinstance(post_state_report, dict):
        return False
    pre_state = _amount_from_report_dict(pre_state_report)
    post_state = _amount_from_report_dict(post_state_report)
    model_amount = _amount_from_report_dict(model_amount_report)
    return (
        _project_model_post_state(
            pre_state=pre_state,
            model_amount=model_amount,
            state_change=state_change,
        )
        == post_state
    )


def _state_transition_failure_detail(
    *,
    model_amount_report: dict[str, Any],
    real_amount_report: dict[str, Any],
) -> dict[str, Any]:
    pre_state_report = real_amount_report.get("pre_state")
    post_state_report = real_amount_report.get("post_state")
    state_change = str(real_amount_report.get("state_change") or "")
    if not isinstance(pre_state_report, dict) or not isinstance(post_state_report, dict):
        raise RuntimeError("missing canonical pre_state/post_state in real-side report")
    pre_state = _amount_from_report_dict(pre_state_report)
    post_state = _amount_from_report_dict(post_state_report)
    model_amount = _amount_from_report_dict(model_amount_report)
    pre_value = _amount_to_decimal(pre_state)
    post_value = _amount_to_decimal(post_state)
    model_value = _amount_to_decimal(model_amount)
    if state_change == STATE_CHANGE_DECREASE:
        projected_post = _project_model_post_state(
            pre_state=pre_state,
            model_amount=model_amount,
            state_change=state_change,
        )
    elif state_change == STATE_CHANGE_INCREASE:
        projected_post = _project_model_post_state(
            pre_state=pre_state,
            model_amount=model_amount,
            state_change=state_change,
        )
    else:
        raise RuntimeError(f"unsupported state_change={state_change}")
    residual = _amount_to_decimal(projected_post) - post_value
    return {
        "state_change": state_change,
        "pre_state": _amount_to_report(pre_state, currency=str(pre_state_report["currency"]), issuer=pre_state_report.get("issuer")),
        "model_amount": dict(model_amount_report),
        "projected_post_state": _amount_to_report(
            projected_post,
            currency=str(post_state_report["currency"]),
            issuer=post_state_report.get("issuer"),
        ),
        "post_state": _amount_to_report(post_state, currency=str(post_state_report["currency"]), issuer=post_state_report.get("issuer")),
        "equation_residual": _compact_decimal_text(format(residual, "f")),
    }


def _sum_model_amount_reports(model_amount_reports: list[dict[str, Any]]) -> Amount:
    total: Amount | None = None
    for report in model_amount_reports:
        amount = _amount_from_report_dict(report)
        total = amount if total is None else total + amount
    if total is None:
        raise RuntimeError("empty model amount sequence")
    return total


def _project_model_sequence_post_state(
    *,
    pre_state: Amount,
    model_amount_reports: list[dict[str, Any]],
    state_change: str,
) -> Amount:
    projected = pre_state
    for report in model_amount_reports:
        projected = _project_model_post_state(
            pre_state=projected,
            model_amount=_amount_from_report_dict(report),
            state_change=state_change,
        )
    return projected


def _state_transition_matches_model_sequence(
    *,
    model_amount_reports: list[dict[str, Any]],
    real_amount_report: dict[str, Any],
) -> bool:
    pre_state_report = real_amount_report.get("pre_state")
    post_state_report = real_amount_report.get("post_state")
    state_change = str(real_amount_report.get("state_change") or "")
    if not isinstance(pre_state_report, dict) or not isinstance(post_state_report, dict):
        return False
    pre_state = _amount_from_report_dict(pre_state_report)
    post_state = _amount_from_report_dict(post_state_report)
    return (
        _project_model_sequence_post_state(
            pre_state=pre_state,
            model_amount_reports=model_amount_reports,
            state_change=state_change,
        )
        == post_state
    )


def _state_transition_failure_detail_sequence(
    *,
    model_amount_reports: list[dict[str, Any]],
    real_amount_report: dict[str, Any],
) -> dict[str, Any]:
    pre_state_report = real_amount_report.get("pre_state")
    post_state_report = real_amount_report.get("post_state")
    state_change = str(real_amount_report.get("state_change") or "")
    if not isinstance(pre_state_report, dict) or not isinstance(post_state_report, dict):
        raise RuntimeError("missing canonical pre_state/post_state in real-side report")
    pre_state = _amount_from_report_dict(pre_state_report)
    post_state = _amount_from_report_dict(post_state_report)
    projected_post = _project_model_sequence_post_state(
        pre_state=pre_state,
        model_amount_reports=model_amount_reports,
        state_change=state_change,
    )
    residual = _amount_to_decimal(projected_post) - _amount_to_decimal(post_state)
    model_total = _sum_model_amount_reports(model_amount_reports)
    return {
        "state_change": state_change,
        "pre_state": _amount_to_report(pre_state, currency=str(pre_state_report["currency"]), issuer=pre_state_report.get("issuer")),
        "model_amount": _amount_to_report(
            model_total,
            currency=str(model_amount_reports[0].get("currency") or real_amount_report.get("currency")),
            issuer=model_amount_reports[0].get("issuer") or real_amount_report.get("issuer"),
        ),
        "model_amount_sequence": [dict(report) for report in model_amount_reports],
        "model_transition_count": len(model_amount_reports),
        "projected_post_state": _amount_to_report(
            projected_post,
            currency=str(post_state_report["currency"]),
            issuer=post_state_report.get("issuer"),
        ),
        "post_state": _amount_to_report(post_state, currency=str(post_state_report["currency"]), issuer=post_state_report.get("issuer")),
        "equation_residual": _compact_decimal_text(format(residual, "f")),
    }


def _endpoint_model_groups_for_real_group(
    *,
    model_group: list[dict[str, Any]],
    real_group: list[dict[str, Any]],
) -> list[tuple[list[dict[str, Any]], dict[str, Any]]] | None:
    if not model_group or not real_group:
        return None
    if len(real_group) == 1:
        return [(model_group, real_group[0])]
    if len(model_group) != len(real_group):
        return None
    sorted_model_group = sorted(
        model_group,
        key=lambda leg: (_amount_report_key(leg["out"]), _amount_report_key(leg["in"])),
    )
    return [([model_leg], real_leg) for model_leg, real_leg in zip(sorted_model_group, real_group)]


def _model_amount_reports_for_real_side(
    *,
    model_sequence: list[dict[str, Any]],
    real_amount_report: dict[str, Any],
    side: str,
) -> list[dict[str, Any]]:
    if side == "out" and str(real_amount_report.get("state_source") or "") == "account_delta":
        return [leg.get("owner_out") or leg["out"] for leg in model_sequence]
    return [leg[side] for leg in model_sequence]


def _endpoint_residual_relative_fields(
    *,
    model_legs: list[dict[str, Any]],
    real_legs: list[dict[str, Any]],
) -> dict[str, Decimal | None]:
    model_by_identity: dict[tuple[str, str], list[dict[str, Any]]] = {}
    real_by_identity: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for leg in model_legs:
        model_by_identity.setdefault(_fit_leg_identity(leg), []).append(leg)
    for leg in real_legs:
        real_by_identity.setdefault(_fit_leg_identity(leg), []).append(leg)
    if set(model_by_identity) != set(real_by_identity):
        return {
            "total_in_endpoint_signed_rel_err_decimal": None,
            "total_out_endpoint_signed_rel_err_decimal": None,
            "total_in_endpoint_unsigned_rel_err_decimal": None,
            "total_out_endpoint_unsigned_rel_err_decimal": None,
            "max_side_endpoint_unsigned_rel_err_decimal": None,
        }

    total_in_residual = Decimal(0)
    total_out_residual = Decimal(0)
    for identity in sorted(model_by_identity):
        model_group = model_by_identity[identity]
        real_group = sorted(
            real_by_identity[identity],
            key=lambda leg: (_state_report_sort_key(leg["out"]), _state_report_sort_key(leg["in"])),
        )
        grouped = _endpoint_model_groups_for_real_group(
            model_group=model_group,
            real_group=real_group,
        )
        if grouped is None:
            return {
                "total_in_endpoint_signed_rel_err_decimal": None,
                "total_out_endpoint_signed_rel_err_decimal": None,
                "total_in_endpoint_unsigned_rel_err_decimal": None,
                "total_out_endpoint_unsigned_rel_err_decimal": None,
                "max_side_endpoint_unsigned_rel_err_decimal": None,
            }
        for model_sequence, real_leg in grouped:
            for side in ("in", "out"):
                real_amount_report = real_leg[side]
                pre_state_report = real_amount_report.get("pre_state")
                post_state_report = real_amount_report.get("post_state")
                if not isinstance(pre_state_report, dict) or not isinstance(post_state_report, dict):
                    raise RuntimeError("missing canonical pre_state/post_state in real-side report")
                projected_post = _project_model_sequence_post_state(
                    pre_state=_amount_from_report_dict(pre_state_report),
                    model_amount_reports=_model_amount_reports_for_real_side(
                        model_sequence=model_sequence,
                        real_amount_report=real_amount_report,
                        side=side,
                    ),
                    state_change=str(real_amount_report.get("state_change") or ""),
                )
                post_state = _amount_from_report_dict(post_state_report)
                residual = _amount_to_decimal(projected_post) - _amount_to_decimal(post_state)
                if side == "in":
                    total_in_residual += residual
                else:
                    total_out_residual += residual

    _, real_total_in_decimal = _exact_report_totals_decimal(real_legs)
    real_total_out_decimal, _ = _exact_report_totals_decimal(real_legs)

    def rel(residual: Decimal, denominator: Decimal) -> Decimal | None:
        abs_denominator = abs(denominator)
        if abs_denominator == 0:
            return Decimal(0) if residual == 0 else None
        return residual / abs_denominator

    in_signed = rel(total_in_residual, real_total_in_decimal)
    out_signed = rel(total_out_residual, real_total_out_decimal)
    in_unsigned = abs(in_signed) if in_signed is not None else None
    out_unsigned = abs(out_signed) if out_signed is not None else None
    return {
        "total_in_endpoint_signed_rel_err_decimal": in_signed,
        "total_out_endpoint_signed_rel_err_decimal": out_signed,
        "total_in_endpoint_unsigned_rel_err_decimal": in_unsigned,
        "total_out_endpoint_unsigned_rel_err_decimal": out_unsigned,
        "max_side_endpoint_unsigned_rel_err_decimal": _max_defined_decimal([in_unsigned, out_unsigned]),
    }


def _empty_endpoint_residual_relative_fields() -> dict[str, Decimal | None]:
    return {
        "total_in_endpoint_signed_rel_err_decimal": None,
        "total_out_endpoint_signed_rel_err_decimal": None,
        "total_in_endpoint_unsigned_rel_err_decimal": None,
        "total_out_endpoint_unsigned_rel_err_decimal": None,
        "max_side_endpoint_unsigned_rel_err_decimal": None,
    }


def _endpoint_residual_relative_text_fields(
    fields: dict[str, Decimal | None],
) -> dict[str, str | None]:
    total_in_signed = fields["total_in_endpoint_signed_rel_err_decimal"]
    total_out_signed = fields["total_out_endpoint_signed_rel_err_decimal"]
    total_in_unsigned = fields["total_in_endpoint_unsigned_rel_err_decimal"]
    total_out_unsigned = fields["total_out_endpoint_unsigned_rel_err_decimal"]
    max_side_unsigned = fields["max_side_endpoint_unsigned_rel_err_decimal"]
    return {
        "total_in_endpoint_signed_rel_err": _format_decimal_or_none(total_in_signed),
        "total_out_endpoint_signed_rel_err": _format_decimal_or_none(total_out_signed),
        "total_in_endpoint_unsigned_rel_err": _format_decimal_or_none(total_in_unsigned),
        "total_out_endpoint_unsigned_rel_err": _format_decimal_or_none(total_out_unsigned),
        "max_side_endpoint_unsigned_rel_err": _format_decimal_or_none(max_side_unsigned),
        "total_in_endpoint_signed_rel_err_bps": _format_decimal_or_none(_rel_to_bps(total_in_signed)),
        "total_out_endpoint_signed_rel_err_bps": _format_decimal_or_none(_rel_to_bps(total_out_signed)),
        "total_in_endpoint_unsigned_rel_err_bps": _format_decimal_or_none(_rel_to_bps(total_in_unsigned)),
        "total_out_endpoint_unsigned_rel_err_bps": _format_decimal_or_none(_rel_to_bps(total_out_unsigned)),
        "max_side_endpoint_unsigned_rel_err_bps": _format_decimal_or_none(_rel_to_bps(max_side_unsigned)),
    }


def _endpoint_rel_err_for_side(
    fields: dict[str, Decimal | None],
    *,
    side: str,
    signed: bool,
) -> Decimal | None:
    if side == "in":
        key = (
            "total_in_endpoint_signed_rel_err_decimal"
            if signed
            else "total_in_endpoint_unsigned_rel_err_decimal"
        )
        return fields[key]
    if side == "out":
        key = (
            "total_out_endpoint_signed_rel_err_decimal"
            if signed
            else "total_out_endpoint_unsigned_rel_err_decimal"
        )
        return fields[key]
    return None


def _amount_from_report_or_none(report_amount: dict[str, Any] | None) -> Amount | None:
    if not isinstance(report_amount, dict):
        return None
    return _amount_from_report_dict(report_amount)


def _amount_equal_from_reports(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    left_amount = _amount_from_report_or_none(left)
    right_amount = _amount_from_report_or_none(right)
    if left_amount is None or right_amount is None:
        return False
    if type(left_amount) is not type(right_amount):
        return False
    return left_amount == right_amount


def _endpoint_prediction_role_fields(report: dict[str, Any]) -> dict[str, str | None]:
    """Classify the endpoint prediction role from intent semantics only.

    This is deliberately not an observed-outcome classifier: real metadata is
    used later to validate the semantic rule, not to choose the rule.
    """
    intent = report["intent"]
    tx_type = str(intent.get("transaction_type") or "").upper()
    if tx_type == "OFFERCREATE":
        if bool(intent.get("is_fok")) and bool(intent.get("is_sell")):
            return {
                "predicted_side": "out",
                "constraint_side": "in",
                "prediction_role": "offer_fok_sell_input_target_output_predicted",
                "prediction_role_reason": "OfferCreate tfFillOrKill+tfSell succeeds only if the sell-side TakerGets amount is fully consumed; output proceeds are predicted.",
            }
        if bool(intent.get("is_fok")):
            return {
                "predicted_side": "in",
                "constraint_side": "out",
                "prediction_role": "offer_fok_buy_output_target_input_predicted",
                "prediction_role_reason": "OfferCreate tfFillOrKill without tfSell succeeds only if the buy-side TakerPays amount is fully acquired; required input is predicted.",
            }
        return {
            "predicted_side": None,
            "constraint_side": None,
            "prediction_role": "offer_limit_order_crossing_both_computed",
            "prediction_role_reason": "OfferCreate crossing is bounded by offer size and limit quality, but non-FOK offers can partially cross and post or cancel the remainder; both crossing endpoints are computed execution results.",
        }

    if tx_type == "PAYMENT":
        if not bool(intent.get("partial_allowed")):
            return {
                "predicted_side": "in",
                "constraint_side": "out",
                "prediction_role": "payment_full_output_target_input_predicted",
                "prediction_role_reason": "Non-partial Payment targets the delivered output Amount/DeliverMax and predicts required input.",
            }

        delivermax = intent.get("delivermax")
        delivermin = intent.get("delivermin")
        if _amount_equal_from_reports(delivermin, delivermax):
            return {
                "predicted_side": "in",
                "constraint_side": "out",
                "prediction_role": "payment_partial_exact_output_target_input_predicted",
                "prediction_role_reason": "Partial Payment has DeliverMin equal to DeliverMax, so successful execution fixes the delivered output and predicts required input.",
            }
        return {
            "predicted_side": None,
            "constraint_side": None,
            "prediction_role": "payment_partial_dual_constraint_both_computed",
            "prediction_role_reason": "Partial Payment with DeliverMin below DeliverMax is bounded by output cap, input cap, quality, and liquidity; both endpoints are computed execution results.",
        }

    return {
        "predicted_side": None,
        "constraint_side": None,
        "prediction_role": "unsupported_or_unknown_intent",
        "prediction_role_reason": "Unsupported transaction type for predicted-side statistics.",
    }


def _strict_exact_match_details(
    *,
    model_legs: list[dict[str, Any]],
    real_legs: list[dict[str, Any]],
) -> dict[str, Any]:
    model_by_identity: dict[tuple[str, str], list[dict[str, Any]]] = {}
    real_by_identity: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for leg in model_legs:
        model_by_identity.setdefault(_fit_leg_identity(leg), []).append(leg)
    for leg in real_legs:
        real_by_identity.setdefault(_fit_leg_identity(leg), []).append(leg)
    if set(model_by_identity) != set(real_by_identity):
        return {
            "strict_exact_match": False,
            "first_failed_leg_index": None,
            "first_failed_leg_source_kind": None,
            "first_failed_leg_source_id": None,
            "failed_side": None,
            "state_change": None,
            "pre_state": None,
            "model_amount": None,
            "projected_post_state": None,
            "post_state": None,
            "equation_residual": None,
        }
    leg_index = 0
    for identity in sorted(model_by_identity):
        model_group = model_by_identity[identity]
        real_group = sorted(
            real_by_identity[identity],
            key=lambda leg: (_state_report_sort_key(leg["out"]), _state_report_sort_key(leg["in"])),
        )
        grouped = _endpoint_model_groups_for_real_group(
            model_group=model_group,
            real_group=real_group,
        )
        if grouped is None:
            return {
                "strict_exact_match": False,
                "first_failed_leg_index": None,
                "first_failed_leg_source_kind": None,
                "first_failed_leg_source_id": None,
                "failed_side": None,
                "state_change": None,
                "pre_state": None,
                "model_amount": None,
                "projected_post_state": None,
                "post_state": None,
                "equation_residual": None,
            }
        for model_sequence, real_leg in grouped:
            out_model_reports = _model_amount_reports_for_real_side(
                model_sequence=model_sequence,
                real_amount_report=real_leg["out"],
                side="out",
            )
            if not _state_transition_matches_model_sequence(
                model_amount_reports=out_model_reports,
                real_amount_report=real_leg["out"],
            ):
                detail = _state_transition_failure_detail_sequence(
                    model_amount_reports=out_model_reports,
                    real_amount_report=real_leg["out"],
                )
                return {
                    "strict_exact_match": False,
                    "first_failed_leg_index": leg_index,
                    "first_failed_leg_source_kind": str(real_leg["src"]),
                    "first_failed_leg_source_id": identity[1],
                    "failed_side": "out",
                    **detail,
                }
            in_model_reports = _model_amount_reports_for_real_side(
                model_sequence=model_sequence,
                real_amount_report=real_leg["in"],
                side="in",
            )
            if not _state_transition_matches_model_sequence(
                model_amount_reports=in_model_reports,
                real_amount_report=real_leg["in"],
            ):
                detail = _state_transition_failure_detail_sequence(
                    model_amount_reports=in_model_reports,
                    real_amount_report=real_leg["in"],
                )
                return {
                    "strict_exact_match": False,
                    "first_failed_leg_index": leg_index,
                    "first_failed_leg_source_kind": str(real_leg["src"]),
                    "first_failed_leg_source_id": identity[1],
                    "failed_side": "in",
                    **detail,
                }
            leg_index += 1
    return {
        "strict_exact_match": True,
        "first_failed_leg_index": None,
        "first_failed_leg_source_kind": None,
        "first_failed_leg_source_id": None,
        "failed_side": None,
        "state_change": None,
        "pre_state": None,
        "model_amount": None,
        "projected_post_state": None,
        "post_state": None,
        "equation_residual": None,
    }


def _side_exact_match(
    *,
    model_legs: list[dict[str, Any]],
    real_legs: list[dict[str, Any]],
    side: str,
) -> bool:
    model_by_identity: dict[tuple[str, str], list[dict[str, Any]]] = {}
    real_by_identity: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for leg in model_legs:
        model_by_identity.setdefault(_fit_leg_identity(leg), []).append(leg)
    for leg in real_legs:
        real_by_identity.setdefault(_fit_leg_identity(leg), []).append(leg)
    if set(model_by_identity) != set(real_by_identity):
        return False

    for identity in sorted(model_by_identity):
        model_group = model_by_identity[identity]
        real_group = sorted(
            real_by_identity[identity],
            key=lambda leg: (_state_report_sort_key(leg["out"]), _state_report_sort_key(leg["in"])),
        )
        grouped = _endpoint_model_groups_for_real_group(
            model_group=model_group,
            real_group=real_group,
        )
        if grouped is None:
            return False
        for model_sequence, real_leg in grouped:
            model_reports = _model_amount_reports_for_real_side(
                model_sequence=model_sequence,
                real_amount_report=real_leg[side],
                side=side,
            )
            if not _state_transition_matches_model_sequence(
                model_amount_reports=model_reports,
                real_amount_report=real_leg[side],
            ):
                return False
    return True


def _classify_fit_bucket(
    *,
    model: dict[str, Any],
    real: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    if bool(comparison["strict_exact_match"]):
        return FIT_BUCKET_STRICT_EXACT
    if bool(comparison["structure_match"]):
        return FIT_BUCKET_STRICT_FAILED_BUT_STRUCTURE_MATCHED
    return FIT_BUCKET_STRUCTURE_MISMATCH


def _intent_report(intent: TxIntent) -> dict[str, Any]:
    base: dict[str, Any] = {
        "transaction_type": intent.transaction_type,
        "flags": intent.flags,
        "source_account": intent.source_account,
        "in_cur": intent.in_cur,
        "out_cur": intent.out_cur,
        "source_funds_cap": (
            _amount_to_report(intent.source_funds_cap_amt, currency=intent.in_cur)
            if intent.source_funds_cap_amt is not None
            else None
        ),
    }
    if intent.transaction_type.upper() == "PAYMENT":
        base.update(
            {
                "sendmax": _amount_to_report(intent.payment_sendmax_amt, currency=intent.in_cur)
                if intent.payment_sendmax_amt is not None
                else None,
                "delivermax": _amount_to_report(intent.payment_delivermax_amt, currency=intent.out_cur)
                if intent.payment_delivermax_amt is not None
                else None,
                "destination_account": intent.payment_destination_account,
                "delivermin": _amount_to_report(intent.payment_delivermin_amt, currency=intent.out_cur)
                if intent.payment_delivermin_amt is not None
                else None,
                "partial_allowed": intent.payment_partial_allowed,
                "no_ripple_direct": intent.payment_no_ripple_direct,
                "limit_quality": _quality_to_string(intent.payment_limit_quality),
                "paths_present": intent.payment_paths_present,
                "paths_count": intent.payment_paths_count,
                "paths_has_external_asset": intent.payment_paths_has_external_asset,
            }
        )
        return base

    base.update(
        {
            "taker_gets": _amount_to_report(intent.offer_taker_gets_amt, currency=intent.in_cur)
            if intent.offer_taker_gets_amt is not None
            else None,
            "taker_pays": _amount_to_report(intent.offer_taker_pays_amt, currency=intent.out_cur)
            if intent.offer_taker_pays_amt is not None
            else None,
            "limit_quality": _quality_to_string(intent.offer_limit_quality),
            "is_sell": intent.offer_is_sell,
            "is_ioc": intent.offer_is_ioc,
            "is_fok": intent.offer_is_fok,
            "is_passive": intent.offer_is_passive,
            "cancel_sequence": intent.offer_cancel_sequence,
        }
    )
    return base


def _model_report_from_quote(
    quote: dict[str, Any],
    *,
    in_cur: str,
    out_cur: str,
    amm_account: str | None = None,
) -> dict[str, Any]:
    summary = quote["summary"]
    def build_legs(slice_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        legs: list[dict[str, Any]] = []
        for slice_row in slice_rows:
            if str(slice_row.get("src") or "") == "CLOB_DELETE":
                continue
            if slice_row.get("out_take") is None or slice_row.get("in_take") is None:
                continue
            out_take = slice_row["out_take"]
            in_take = slice_row["in_take"]
            source_id = slice_row.get("source_id")
            if source_id is None and slice_row["src"] == "AMM":
                source_id = amm_account
            leg = {
                "src": slice_row["src"],
                "source_id": source_id,
                "out": _amount_to_report(out_take, currency=out_cur),
                "in": _amount_to_report(in_take, currency=in_cur),
                "quality": _quality_to_string(slice_row.get("avg_quality")),
            }
            owner_out_take = slice_row.get("owner_out_take")
            if owner_out_take is not None:
                leg["owner_out"] = _amount_to_report(owner_out_take, currency=out_cur)
            legs.append(leg)
        return legs

    model_legs = build_legs(list(quote.get("slices", []) or []))
    endpoint_legs = build_legs(list(quote.get("endpoint_slices", quote.get("slices", [])) or []))
    total_out, total_in = _sum_legs(model_legs, in_cur=in_cur, out_cur=out_cur)
    return {
        "legs": model_legs,
        "endpoint_legs": endpoint_legs,
        "summary": {
            "total_out": _amount_to_report(total_out, currency=out_cur),
            "total_in": _amount_to_report(total_in, currency=in_cur),
            "avg_quality": _quality_to_string(summary.get("avg_quality")),
            "is_partial": bool(summary.get("is_partial")),
            "fill_ratio": summary.get("fill_ratio"),
        },
    }


def _model_error_report(*, error: Exception, in_cur: str, out_cur: str) -> dict[str, Any]:
    zero_out = XRPAmount(0) if _is_xrp(out_cur) else IOUAmount(0, 0)
    zero_in = XRPAmount(0) if _is_xrp(in_cur) else IOUAmount(0, 0)
    return {
        "legs": [],
        "summary": {
            "total_out": _amount_to_report(zero_out, currency=out_cur),
            "total_in": _amount_to_report(zero_in, currency=in_cur),
            "avg_quality": None,
            "is_partial": True,
            "fill_ratio": 0,
            "model_error": str(error),
        },
    }


def _compare_reports(
    *,
    model: dict[str, Any],
    real: dict[str, Any],
) -> dict[str, Any]:
    model_legs = model["legs"]
    model_endpoint_legs = model.get("endpoint_legs") or model_legs
    real_legs = real["legs"]
    strict = _strict_exact_match_details(model_legs=model_endpoint_legs, real_legs=real_legs)
    model_clob_legs = [leg for leg in model_legs if str(leg.get("src") or "") == "CLOB"]
    real_clob_legs = [leg for leg in real_legs if str(leg.get("src") or "") == "CLOB"]
    model_sources = Counter((leg["src"], leg.get("source_id")) for leg in model_clob_legs)
    real_sources = Counter((leg["src"], leg.get("offer_id")) for leg in real_clob_legs)
    clob_offer_multiset_match = model_sources == real_sources
    amm_participation_match = (
        sum(1 for leg in model_legs if str(leg.get("src") or "") == "AMM")
        == sum(1 for leg in real_legs if str(leg.get("src") or "") == "AMM")
    )
    structure_match = clob_offer_multiset_match and amm_participation_match

    model_total_out_decimal, model_total_in_decimal = _exact_report_totals_decimal(model_legs)
    real_total_out_decimal, real_total_in_decimal = _exact_report_totals_decimal(real_legs)
    rel_fields = _relative_error_fields_from_decimals(
        model_total_out_decimal=model_total_out_decimal,
        model_total_in_decimal=model_total_in_decimal,
        real_total_out_decimal=real_total_out_decimal,
        real_total_in_decimal=real_total_in_decimal,
    )
    endpoint_rel_fields = _endpoint_residual_relative_fields(model_legs=model_endpoint_legs, real_legs=real_legs)
    endpoint_rel_text_fields = _endpoint_residual_relative_text_fields(endpoint_rel_fields)
    out_side_exact = _side_exact_match(model_legs=model_endpoint_legs, real_legs=real_legs, side="out")
    in_side_exact = _side_exact_match(model_legs=model_endpoint_legs, real_legs=real_legs, side="in")

    return {
        "strict_exact_match": bool(strict["strict_exact_match"]),
        "structure_match": structure_match,
        "clob_offer_multiset_match": clob_offer_multiset_match,
        "amm_participation_match": amm_participation_match,
        "model_leg_count": len(model_legs),
        "model_endpoint_leg_count": len(model_endpoint_legs),
        "real_leg_count": len(real_legs),
        "out_side_exact_match": out_side_exact,
        "in_side_exact_match": in_side_exact,
        "total_out_match": model_total_out_decimal == real_total_out_decimal,
        "total_in_match": model_total_in_decimal == real_total_in_decimal,
        "total_out_diff": _compact_decimal_text(format(model_total_out_decimal - real_total_out_decimal, "f")),
        "total_in_diff": _compact_decimal_text(format(model_total_in_decimal - real_total_in_decimal, "f")),
        "total_out_rel_err": rel_fields["total_out_rel_err"],
        "total_in_rel_err": rel_fields["total_in_rel_err"],
        "max_side_rel_err": rel_fields["max_side_rel_err"],
        "total_out_rel_err_bps": rel_fields["total_out_rel_err_bps"],
        "total_in_rel_err_bps": rel_fields["total_in_rel_err_bps"],
        "max_side_rel_err_bps": rel_fields["max_side_rel_err_bps"],
        **endpoint_rel_text_fields,
        "first_failed_leg_index": strict["first_failed_leg_index"],
        "first_failed_leg_source_kind": strict["first_failed_leg_source_kind"],
        "first_failed_leg_source_id": strict["first_failed_leg_source_id"],
        "failed_side": strict["failed_side"],
        "state_change": strict["state_change"],
        "pre_state": strict["pre_state"],
        "model_amount": strict["model_amount"],
        "projected_post_state": strict["projected_post_state"],
        "post_state": strict["post_state"],
        "equation_residual": strict["equation_residual"],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    global TARGET_IOU_CURRENCY, TARGET_IOU_LABEL
    args = _parse_args()
    TARGET_IOU_CURRENCY = str(args.base_currency)
    TARGET_IOU_LABEL = str(args.base_label)
    root = Path(args.root)
    metadata_path = Path(args.metadata_ndjson or root / "tx_metadata_full_merged_cut.ndjson")
    snapshots_path = Path(args.tx_prebook_snapshots or root / "tx_prebook_snapshots_cut.ndjson")
    account_offers_snapshots_path = (
        Path(args.account_offers_snapshots) if args.account_offers_snapshots else None
    )
    amm_swaps_path = Path(args.amm_swaps or root / "amm_swaps_cut.parquet")
    amm_fees_path = Path(args.amm_fees or root / "amm_fees_cut.parquet")
    output_dir = Path(args.output_dir or ".tmp/single_direct_fit_latest")
    reports_dir = output_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_any_reports = (
        (not args.aggregate_only)
        or args.write_structure_false_reports
        or args.write_strict_false_reports
    )
    if write_any_reports:
        reports_dir.mkdir(parents=True, exist_ok=True)

    sample_hashes = _load_sample_hashes(args)
    target_set = set(sample_hashes)
    samples = _load_metadata_samples(metadata_path=metadata_path, target_tx_hashes=target_set)
    account_offers_snapshots = (
        _load_account_offers_snapshots(account_offers_snapshots_path)
        if account_offers_snapshots_path is not None
        else {}
    )
    tick_size_snapshots = _load_offer_tick_size_snapshots(metadata_path)
    amm_rows = _load_amm_rows(amm_swaps_path, tx_column="transaction_hash", target_tx_hashes=target_set)
    fee_rows = _load_amm_rows(amm_fees_path, tx_column="transaction_hash", target_tx_hashes=target_set)
    ordered_snapshot_targets = [
        (tx_hash, _prebook_primary_side_for_output_currency(samples[tx_hash].intent.out_cur))
        for tx_hash in sample_hashes
    ]

    issuer_transfer_rates, issuer_transfer_rate_ranges = _load_issuer_transfer_rates(args.issuer_transfer_rates_json)
    amendment_feature_flags = _load_amendment_feature_flags(args.issuer_transfer_rates_json)
    fix_amm_v1_1_enabled = amendment_feature_flags.get(
        "fixAMMv1_1",
        amendment_feature_flags.get("AMMv1_1", True),
    )

    aggregate_builder = _AggregateStatsBuilder(top_relative_error_limit=max(0, int(args.top_relative_errors)))
    intent_error_builder = _IntentErrorBreakdownBuilder()
    processed_txs: list[str] = []
    progress_every = 1000 if len(sample_hashes) > 10000 else 250

    for processed_count, (tx_hash, snapshot_payload) in enumerate(
        _iter_snapshot_rows_in_sample_order(
            snapshots_path=snapshots_path,
            ordered_targets=ordered_snapshot_targets,
        ),
        start=1,
    ):
        sample = samples[tx_hash]
        tick_size = _effective_offer_tick_size_for_intent(
            intent=sample.intent,
            ledger_index=sample.ledger_index,
            tick_size_snapshots=tick_size_snapshots,
        )
        intent = apply_offer_tick_size_to_intent(sample.intent, tick_size)
        tx_in_transfer_rate = _transfer_rate_for_asset(
            currency=intent.in_cur,
            issuer=(
                intent.payment_sendmax_issuer
                if str(intent.transaction_type or "").upper() == "PAYMENT"
                else intent.offer_taker_gets_issuer
            ),
            issuer_transfer_rates=issuer_transfer_rates,
            issuer_transfer_rate_ranges=issuer_transfer_rate_ranges,
            ledger_index=sample.ledger_index,
        )
        tx_out_transfer_rate = _transfer_rate_for_asset(
            currency=intent.out_cur,
            issuer=(
                intent.payment_delivermax_issuer
                if str(intent.transaction_type or "").upper() == "PAYMENT"
                else intent.offer_taker_pays_issuer
            ),
            issuer_transfer_rates=issuer_transfer_rates,
            issuer_transfer_rate_ranges=issuer_transfer_rate_ranges,
            ledger_index=sample.ledger_index,
        )
        selected_side = ordered_snapshot_targets[processed_count - 1][1]

        owner_funds_overlay = _extract_overlay_owner_funds_from_metadata_result(sample.result)
        current_tx_pre_overlay_rows = _metadata_current_tx_pre_offer_rows(
            result=sample.result,
            selected_side=selected_side,
        )
        snapshot_payload, current_tx_pre_overlay_count = _overlay_current_tx_pre_offer_rows(
            side_payload=snapshot_payload,
            current_tx_pre_offer_rows=current_tx_pre_overlay_rows,
        )
        snapshot_ledger_index = snapshot_payload.get("used_prebook_ledger")
        if snapshot_ledger_index is None:
            raise RuntimeError(f"snapshot missing used_prebook_ledger for tx={tx_hash}")
        raw_funded_caps_by_offer_id, raw_overridden_accounts = _build_external_funded_caps_for_snapshot(
            side_payload=snapshot_payload,
            ledger_index=int(snapshot_ledger_index),
            account_offers_snapshots=account_offers_snapshots,
        )
        explicit_funded_accounts = _accounts_with_explicit_snapshot_funding(snapshot_payload)
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
        non_fill_deleted_offer_ids = _metadata_deleted_offer_without_fill_ids(sample.result)
        book_offers = _materialize_snapshot_book_offers(
            side_payload=snapshot_payload,
            derived_owner_funds_by_offer_id=owner_funds_overlay,
            funded_caps_by_offer_id=funded_caps_by_offer_id,
            strip_owner_funds_accounts=overridden_accounts,
        )
        materialized_before_non_fill_deleted_filter = len(book_offers)
        book_offers = _filter_non_fill_deleted_offers(
            book_offers=book_offers,
            offer_ids=non_fill_deleted_offer_ids,
            keep_owner=str(intent.source_account or "")
            if str(intent.transaction_type or "").upper() == "OFFERCREATE"
            else None,
        )
        book_offers = _apply_offercreate_cancel_sequence(
            book_offers=book_offers,
            intent=intent,
        )
        clob_segments = build_clob_segments_from_offers(
            book_offers,
            in_cur=intent.in_cur,
            out_cur=intent.out_cur,
            iou_in_transfer_rate=tx_in_transfer_rate,
            iou_out_transfer_rate=tx_out_transfer_rate,
            owner_pays_transfer_fee=str(intent.transaction_type or "").upper() == "OFFERCREATE",
            exclude_account=None,
            preserve_book_visible_offers=str(intent.transaction_type or "").upper() == "OFFERCREATE",
        )
        amm, amm_state_report = _build_model_amm(
            tx_hash=tx_hash,
            metadata_result=sample.result,
            amm_rows=amm_rows.get(tx_hash, []),
            fee_rows=fee_rows.get(tx_hash, []),
            in_cur=intent.in_cur,
            out_cur=intent.out_cur,
        )
        model_error = None
        try:
            quote, meta = execute_single_path_intent(
                intent=intent,
                amm=amm,
                segs=clob_segments,
                in_cur=intent.in_cur,
                out_cur=intent.out_cur,
                book_in_transfer_rate=tx_in_transfer_rate,
                book_out_transfer_rate=tx_out_transfer_rate,
                fix_amm_v1_1_enabled=fix_amm_v1_1_enabled,
            )
            model_report = _model_report_from_quote(
                quote,
                in_cur=intent.in_cur,
                out_cur=intent.out_cur,
                amm_account=amm_state_report.get("amm_account") if amm_state_report is not None else None,
            )
        except RuntimeError as exc:
            model_error = str(exc)
            quote = {"summary": {}, "slices": []}
            meta = {"mode": "model-error"}
            model_report = _model_error_report(error=exc, in_cur=intent.in_cur, out_cur=intent.out_cur)

        real_legs = _build_real_offer_legs(sample.result, in_cur=intent.in_cur, out_cur=intent.out_cur)
        real_amm_legs = _build_real_amm_legs_from_metadata(
            sample.result,
            in_cur=intent.in_cur,
            out_cur=intent.out_cur,
        )
        if real_amm_legs is None:
            if amm_rows.get(tx_hash, []):
                raise RuntimeError(f"missing metadata-exact real AMM legs for tx={tx_hash}")
            real_amm_legs = []
        real_legs.extend(real_amm_legs)
        real_total_out, real_total_in = _sum_legs(real_legs, in_cur=intent.in_cur, out_cur=intent.out_cur)
        real_report = {
            "legs": real_legs,
            "summary": {
                "total_out": _amount_to_report(real_total_out, currency=intent.out_cur),
                "total_in": _amount_to_report(real_total_in, currency=intent.in_cur),
            },
        }

        comparison = _compare_reports(model=model_report, real=real_report)
        comparison["fit_bucket"] = _classify_fit_bucket(
            model=model_report,
            real=real_report,
            comparison=comparison,
        )
        report = {
            "tx_hash": tx_hash,
            "ledger_index": sample.ledger_index,
            "transaction_index": sample.transaction_index,
            "intent": _intent_report(intent),
            "snapshot": {
                "selected_side": selected_side,
                "used_prebook_ledger": snapshot_payload.get("used_prebook_ledger"),
                "offers_count_visible": snapshot_payload.get("offers_count_visible"),
                "offers_count_emitted": snapshot_payload.get("offers_count_emitted"),
                "pre_overlay_offer_count": snapshot_payload.get("pre_overlay_offer_count"),
                "current_tx_pre_offer_overlay_count": snapshot_payload.get("current_tx_pre_offer_overlay_count"),
                "current_tx_pre_offer_overlay_ids": snapshot_payload.get("current_tx_pre_offer_overlay_ids"),
                "fit_time_current_tx_pre_offer_overlay_count": current_tx_pre_overlay_count,
                "materialized_offer_count": len(book_offers),
                "materialized_before_non_fill_deleted_filter": materialized_before_non_fill_deleted_filter,
                "non_fill_deleted_offer_count": len(non_fill_deleted_offer_ids),
                "non_fill_deleted_offer_ids": sorted(non_fill_deleted_offer_ids),
                "clob_segment_count": len(clob_segments),
                "offer_tick_size": tick_size,
                "account_offers_explicit_funded_accounts": len(explicit_funded_accounts),
                "account_offers_raw_override_accounts": len(raw_overridden_accounts),
                "account_offers_raw_funded_caps": len(raw_funded_caps_by_offer_id),
                "account_offers_override_accounts": len(overridden_accounts),
                "account_offers_funded_caps": len(funded_caps_by_offer_id),
                "fix_amm_v1_1_enabled": fix_amm_v1_1_enabled,
            },
            "amm_state": amm_state_report,
            "model": model_report,
            "real": real_report,
            "comparison": comparison,
            "execution_meta": meta,
        }
        comparison.update(_endpoint_prediction_role_fields(report))
        if model_error is not None:
            report["model_error"] = model_error
        processed_txs.append(tx_hash)
        aggregate_builder.update(report)
        intent_error_builder.update(report)
        if (
            (not args.aggregate_only)
            or (args.write_structure_false_reports and not comparison["structure_match"])
            or (args.write_strict_false_reports and not comparison["strict_exact_match"])
        ):
            _write_json(reports_dir / f"{tx_hash}.json", report)
        if processed_count % progress_every == 0 or processed_count == len(sample_hashes):
            print(
                f"processed={processed_count}/{len(sample_hashes)} "
                f"latest={tx_hash} strict={aggregate_builder.strict_exact_match_count}",
                flush=True,
            )

    if processed_txs != sample_hashes:
        raise RuntimeError(
            "processed tx order/count mismatch after streaming fit: "
            f"processed={len(processed_txs)} expected={len(sample_hashes)}"
        )

    summary = {
        "root": str(root),
        "metadata_ndjson": str(metadata_path),
        "tx_prebook_snapshots": str(snapshots_path),
        "account_offers_snapshots": str(account_offers_snapshots_path) if account_offers_snapshots_path else None,
        "amm_swaps": str(amm_swaps_path),
        "amm_fees": str(amm_fees_path),
        "sample_count": len(processed_txs),
        "samples": sample_hashes,
        "aggregate": aggregate_builder.as_dict(),
    }
    _write_json(output_dir / "manifest.json", summary)
    _write_json(output_dir / "aggregate.json", summary["aggregate"])
    _write_json(output_dir / "intent_error_breakdown.json", intent_error_builder.as_dict())
    if not args.aggregate_only:
        _write_summary_json_from_reports(
            sample_hashes=sample_hashes,
            reports_dir=reports_dir,
            output_path=output_dir / "summary.json",
        )
        _write_summary_markdown_from_reports(
            sample_hashes=sample_hashes,
            reports_dir=reports_dir,
            output_path=output_dir / "summary.md",
        )
        print(f"wrote {len(processed_txs)} sample reports to {output_dir}")
    else:
        print(f"wrote aggregate-only fit for {len(processed_txs)} samples to {output_dir}")


if __name__ == "__main__":
    main()
