from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterator, Optional, Set, Tuple

from .core import Amount, IOUAmount, Quality, XRPAmount, parse_iou_decimal, parse_xrp_drops

XRP = "XRP"
ST_MANTISSA_MIN = 10**15
ST_MANTISSA_MAX = (10**16) - 1
ST_NATIVE_MAX = 100_000_000_000_000_000
QUALITY_MANTISSA_MASK = 0x00FFFFFFFFFFFFFF
QUALITY_MAX_TICK_SIZE = 16
QUALITY_MIN_TICK_SIZE = 3
XRPL_ACCOUNT_RESERVE_BASE_DROPS = 1_000_000
XRPL_ACCOUNT_RESERVE_INC_DROPS = 200_000

# XRPL tx flag bits (contextual by transaction type).
TF_PARTIAL_PAYMENT = 0x00020000
TF_NO_RIPPLE_DIRECT = 0x00010000
TF_LIMIT_QUALITY = 0x00040000
TF_PASSIVE = 0x00010000
TF_IMMEDIATE_OR_CANCEL = 0x00020000
TF_FILL_OR_KILL = 0x00040000
TF_SELL = 0x00080000
TF_FULLY_CANONICAL_SIG = 0x80000000


@dataclass(frozen=True, slots=True)
class TxIntent:
    transaction_type: str
    flags: int
    in_cur: str
    out_cur: str
    source_account: Optional[str] = None
    # Payment
    payment_sendmax_amt: Optional[Amount] = None
    payment_sendmax_currency: Optional[str] = None
    payment_sendmax_issuer: Optional[str] = None
    payment_delivermax_amt: Optional[Amount] = None
    payment_delivermax_currency: Optional[str] = None
    payment_delivermax_issuer: Optional[str] = None
    payment_destination_account: Optional[str] = None
    payment_delivermin_amt: Optional[Amount] = None
    payment_partial_allowed: bool = False
    payment_no_ripple_direct: bool = False
    payment_limit_quality: Optional[Quality] = None
    payment_paths_present: bool = False
    payment_paths_count: int = 0
    payment_paths_has_external_asset: bool = False
    source_funds_cap_amt: Optional[Amount] = None
    # OfferCreate
    offer_taker_gets_amt: Optional[Amount] = None
    offer_taker_gets_currency: Optional[str] = None
    offer_taker_gets_issuer: Optional[str] = None
    offer_taker_pays_amt: Optional[Amount] = None
    offer_taker_pays_currency: Optional[str] = None
    offer_taker_pays_issuer: Optional[str] = None
    offer_limit_quality: Optional[Quality] = None
    offer_is_sell: bool = False
    offer_is_ioc: bool = False
    offer_is_fok: bool = False
    offer_is_passive: bool = False
    offer_tick_size: Optional[int] = None
    offer_cancel_sequence: Optional[int] = None

    @property
    def payment_currency(self) -> Optional[str]:
        return self.payment_delivermax_currency

    @property
    def payment_issuer(self) -> Optional[str]:
        return self.payment_delivermax_issuer


def _parse_amount_field(raw: Any, *, field_name: str) -> Tuple[str, Optional[str], Amount]:
    if isinstance(raw, str):
        try:
            amount = parse_xrp_drops(raw, field_name=field_name)
        except Exception as e:
            raise RuntimeError(str(e)) from e
        if amount.drops <= 0:
            raise RuntimeError(f"invalid {field_name}: non-positive amount")
        return XRP, None, amount

    if not isinstance(raw, dict):
        raise RuntimeError(f"invalid {field_name}: expected XRP string or IOU object")

    currency_raw = raw.get("currency")
    issuer_raw = raw.get("issuer")
    value_raw = raw.get("value")
    if currency_raw is None or str(currency_raw) == "":
        raise RuntimeError(f"invalid {field_name}: missing currency")

    currency = str(currency_raw)
    issuer = None if issuer_raw is None else str(issuer_raw)
    if currency == XRP:
        raise RuntimeError(f"invalid {field_name}: XRP must use string drops encoding")
    if issuer is None or issuer == "":
        raise RuntimeError(f"invalid {field_name}: missing issuer for non-XRP amount")

    try:
        amount = parse_iou_decimal(value_raw, field_name=field_name)
    except Exception as e:
        raise RuntimeError(str(e)) from e
    if amount.mantissa <= 0:
        raise RuntimeError(f"invalid {field_name}: non-positive amount")
    return currency, issuer, amount


def _asset_key(currency: str, issuer: Optional[str]) -> Tuple[str, Optional[str]]:
    if currency == XRP:
        return XRP, None
    return currency, issuer


def _same_asset(
    left_currency: str,
    left_issuer: Optional[str],
    right_currency: str,
    right_issuer: Optional[str],
) -> bool:
    return _asset_key(left_currency, left_issuer) == _asset_key(right_currency, right_issuer)


def _copy_amount(amount: Amount) -> Amount:
    if isinstance(amount, XRPAmount):
        return XRPAmount(amount.drops)
    return IOUAmount(amount.mantissa, amount.exponent)


def _zero_like(amount: Amount) -> Amount:
    if isinstance(amount, XRPAmount):
        return XRPAmount(0)
    return IOUAmount(0, 0)


def _positive_amount_for_quality(amount: Amount) -> None:
    if isinstance(amount, XRPAmount):
        if amount.drops <= 0:
            raise RuntimeError("quality requires strictly positive XRP amount")
        return
    if amount.mantissa <= 0:
        raise RuntimeError("quality requires strictly positive IOU amount")


def _quality_from_amounts(*, out_amount: Amount, in_amount: Amount) -> Quality:
    _positive_amount_for_quality(out_amount)
    _positive_amount_for_quality(in_amount)
    return Quality.from_amounts(out_amount, in_amount)


def _working_components(amount: Amount) -> Tuple[int, int, bool]:
    if isinstance(amount, XRPAmount):
        value = abs(amount.drops)
        exponent = 0
        integral = True
    else:
        value = abs(amount.mantissa)
        exponent = amount.exponent
        integral = False

    if value == 0:
        return 0, exponent, integral

    if integral:
        while value < ST_MANTISSA_MIN:
            value *= 10
            exponent -= 1

    return value, exponent, integral


def _canonicalize_native_amount(value: int, exponent: int) -> XRPAmount:
    if value == 0:
        return XRPAmount(0)

    negative = value < 0
    if negative:
        value = -value

    if exponent < 0:
        divisor = 10 ** (-exponent)
        quotient, remainder = divmod(value, divisor)
        doubled_remainder = remainder * 2
        if doubled_remainder > divisor or (doubled_remainder == divisor and quotient % 2 == 1):
            quotient += 1
        value = quotient
        exponent = 0

    while exponent < 0:
        value //= 10
        exponent += 1

    while exponent > 0:
        if value > ST_NATIVE_MAX:
            raise RuntimeError("native amount out of range")
        value *= 10
        exponent -= 1

    if value > ST_NATIVE_MAX:
        raise RuntimeError("native amount out of range")
    return XRPAmount(-value if negative else value)


def _amount_from_math_value(*, is_xrp: bool, value: int, exponent: int) -> Amount:
    if value == 0:
        return XRPAmount(0) if is_xrp else IOUAmount(0, 0)
    if is_xrp:
        return _canonicalize_native_amount(value, exponent)
    try:
        return IOUAmount(value, exponent)
    except Exception as e:
        raise RuntimeError(f"failed to canonicalize IOU amount: {e}") from e


def _divide_amount_by_rate(amount: Amount, rate: IOUAmount, *, result_is_xrp: bool) -> Amount:
    num_value, num_exponent, _ = _working_components(amount)
    den_value, den_exponent, _ = _working_components(rate)
    if num_value == 0:
        return XRPAmount(0) if result_is_xrp else IOUAmount(0, 0)
    if den_value == 0:
        raise RuntimeError("tick-size divide by zero quality rate")
    quotient = ((num_value * (10**17)) // den_value) + 5
    exponent = num_exponent - den_exponent - 17
    return _amount_from_math_value(is_xrp=result_is_xrp, value=quotient, exponent=exponent)


def _multiply_amount_by_rate(amount: Amount, rate: IOUAmount, *, result_is_xrp: bool) -> Amount:
    if result_is_xrp:
        product = amount.as_fraction() * rate.as_fraction()
        if product == 0:
            return XRPAmount(0)
        negative = product < 0
        if negative:
            product = -product
        quotient, remainder = divmod(product.numerator, product.denominator)
        doubled_remainder = remainder * 2
        if doubled_remainder > product.denominator or (
            doubled_remainder == product.denominator and quotient % 2 == 1
        ):
            quotient += 1
        drops = -quotient if negative else quotient
        return _canonicalize_native_amount(drops, 0)

    left_value, left_exponent, _ = _working_components(amount)
    right_value, right_exponent, _ = _working_components(rate)
    if left_value == 0 or right_value == 0:
        return XRPAmount(0) if result_is_xrp else IOUAmount(0, 0)
    product = ((left_value * right_value) // (10**14)) + 7
    exponent = left_exponent + right_exponent + 14
    return _amount_from_math_value(is_xrp=result_is_xrp, value=product, exponent=exponent)


def _round_quality_to_tick_size(quality: Quality, tick_size: int) -> Quality:
    if tick_size < QUALITY_MIN_TICK_SIZE or tick_size >= QUALITY_MAX_TICK_SIZE:
        raise RuntimeError(f"invalid offer tick size: {tick_size}")
    if quality.is_zero():
        return quality

    modulus = 10 ** (QUALITY_MAX_TICK_SIZE - tick_size)
    exponent = quality.value >> 56
    mantissa = quality.value & QUALITY_MANTISSA_MASK
    mantissa += modulus - 1
    mantissa -= mantissa % modulus
    return Quality((exponent << 56) | mantissa)


def apply_offer_tick_size_to_intent(intent: TxIntent, tick_size: Optional[int]) -> TxIntent:
    """Apply rippled-style OfferCreate tick-size preprocessing when available."""
    if str(intent.transaction_type or "").upper() != "OFFERCREATE":
        return intent
    if tick_size is None or tick_size >= QUALITY_MAX_TICK_SIZE:
        return intent
    if intent.offer_taker_gets_amt is None or intent.offer_taker_pays_amt is None:
        raise RuntimeError("OfferCreate intent missing TakerGets/TakerPays for tick-size preprocessing")
    if intent.offer_limit_quality is None:
        raise RuntimeError("OfferCreate intent missing limit quality for tick-size preprocessing")

    # Rippled rounds the placed-offer quality, where:
    # - out = TakerGets
    # - in = TakerPays
    book_quality = _quality_from_amounts(
        out_amount=intent.offer_taker_gets_amt,
        in_amount=intent.offer_taker_pays_amt,
    )
    rounded_quality = _round_quality_to_tick_size(book_quality, tick_size)
    rounded_rate = rounded_quality.rate()
    rounded_taker_gets = intent.offer_taker_gets_amt
    rounded_taker_pays = intent.offer_taker_pays_amt

    if intent.offer_is_sell:
        # CreateOffer::doApply() calls multiply(saTakerGets, rate, issue) here.
        # That is STAmount's default multiply path, not the book-offer helper.
        rounded_taker_pays = _multiply_amount_by_rate(
            intent.offer_taker_gets_amt,
            rounded_rate,
            result_is_xrp=isinstance(intent.offer_taker_pays_amt, XRPAmount),
        )
    else:
        # Rippled buy-side preprocessing calls divide(), whose current
        # implementation still follows the classic "+5" STAmount path rather
        # than the round-down helper we use in book replay code.
        rounded_taker_gets = _divide_amount_by_rate(
            intent.offer_taker_pays_amt,
            rounded_rate,
            result_is_xrp=isinstance(intent.offer_taker_gets_amt, XRPAmount),
        )

    if rounded_taker_gets == _zero_like(rounded_taker_gets) or rounded_taker_pays == _zero_like(rounded_taker_pays):
        return replace(
            intent,
            offer_taker_gets_amt=rounded_taker_gets,
            offer_taker_pays_amt=rounded_taker_pays,
            offer_limit_quality=rounded_quality,
            offer_tick_size=tick_size,
        )

    return replace(
        intent,
        offer_taker_gets_amt=rounded_taker_gets,
        offer_taker_pays_amt=rounded_taker_pays,
        offer_limit_quality=_quality_from_amounts(
            out_amount=rounded_taker_pays,
            in_amount=rounded_taker_gets,
        ),
        offer_tick_size=tick_size,
    )


def _iter_meta_nodes(result: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    meta = result.get("meta")
    if not isinstance(meta, dict):
        return
    nodes = meta.get("AffectedNodes")
    if not isinstance(nodes, list):
        return
    for wrapped in nodes:
        if not isinstance(wrapped, dict):
            continue
        node = wrapped.get("ModifiedNode") or wrapped.get("DeletedNode") or wrapped.get("CreatedNode")
        if isinstance(node, dict):
            yield node


def _fields_for_read(node: Dict[str, Any], key: str) -> Any:
    previous = node.get("PreviousFields")
    if isinstance(previous, dict) and key in previous:
        return previous.get(key)
    final = node.get("FinalFields")
    if isinstance(final, dict) and key in final:
        return final.get(key)
    new = node.get("NewFields")
    if isinstance(new, dict) and key in new:
        return new.get(key)
    return None


def _extract_source_funds_cap_from_metadata(
    *,
    result: Dict[str, Any],
    account: str,
    in_cur: str,
    in_issuer: Optional[str],
) -> Optional[Amount]:
    if not account:
        return None

    if in_cur == XRP:
        fee_raw = result.get("Fee")
        fee_drops = int(str(fee_raw)) if fee_raw is not None else 0
        for node in _iter_meta_nodes(result):
            if str(node.get("LedgerEntryType") or "") != "AccountRoot":
                continue
            account_raw = _fields_for_read(node, "Account")
            if str(account_raw or "") != account:
                continue
            balance_raw = _fields_for_read(node, "Balance")
            if balance_raw is None:
                continue
            owner_count_raw = _fields_for_read(node, "OwnerCount")
            try:
                balance_drops = int(str(balance_raw))
                owner_count = int(str(owner_count_raw)) if owner_count_raw is not None else 0
            except Exception:
                continue
            if result.get("TicketSequence") is not None and str(result.get("Account") or "") == account:
                # Rippled consumes the Ticket before entering the payment /
                # offer-crossing execution path, so spendable XRP sees one
                # owner reserve released even when metadata only reports
                # TicketCount, not OwnerCount, in PreviousFields.
                owner_count = max(owner_count - 1, 0)
            reserve_drops = XRPL_ACCOUNT_RESERVE_BASE_DROPS + max(owner_count, 0) * XRPL_ACCOUNT_RESERVE_INC_DROPS
            spendable = balance_drops - reserve_drops - fee_drops
            if spendable <= 0:
                return None
            return XRPAmount(spendable)
        return None

    if in_issuer is None:
        return None

    for node in _iter_meta_nodes(result):
        if str(node.get("LedgerEntryType") or "") != "RippleState":
            continue
        low_limit = _fields_for_read(node, "LowLimit")
        high_limit = _fields_for_read(node, "HighLimit")
        if not isinstance(low_limit, dict) or not isinstance(high_limit, dict):
            continue

        currency = str(low_limit.get("currency") or high_limit.get("currency") or "")
        if currency != in_cur:
            continue

        low_account = str(low_limit.get("issuer") or "")
        high_account = str(high_limit.get("issuer") or "")
        if account not in {low_account, high_account}:
            continue
        if in_issuer not in {low_account, high_account}:
            continue

        balance_raw = _fields_for_read(node, "Balance")
        if not isinstance(balance_raw, dict):
            continue
        try:
            balance_amount = parse_iou_decimal(
                balance_raw.get("value"),
                field_name="metadata RippleState Balance",
            )
        except Exception:
            continue

        account_balance = (
            balance_amount
            if account == low_account
            else IOUAmount(-balance_amount.mantissa, balance_amount.exponent)
        )
        if account_balance.mantissa <= 0:
            return None
        return account_balance

    return None


def _validate_supported_flags(*, tx_type: str, flags: int) -> None:
    tx_type_upper = str(tx_type or "").upper()
    if tx_type_upper == "PAYMENT":
        supported = TF_PARTIAL_PAYMENT | TF_NO_RIPPLE_DIRECT | TF_LIMIT_QUALITY | TF_FULLY_CANONICAL_SIG
    elif tx_type_upper == "OFFERCREATE":
        supported = TF_PASSIVE | TF_IMMEDIATE_OR_CANCEL | TF_FILL_OR_KILL | TF_SELL | TF_FULLY_CANONICAL_SIG
        if (flags & TF_IMMEDIATE_OR_CANCEL) and (flags & TF_FILL_OR_KILL):
            raise RuntimeError("invalid OFFERCREATE flags: both tfImmediateOrCancel and tfFillOrKill are set")
    else:
        raise RuntimeError(f"unsupported transaction type for flag validation: {tx_type_upper or 'unknown'}")

    unknown = int(flags) & ~int(supported)
    if unknown:
        raise RuntimeError(
            f"unsupported {tx_type_upper} flags bits: flags=0x{int(flags):X} unknown=0x{int(unknown):X}"
        )


def _payment_paths_scope_flags(
    *,
    result: Dict[str, Any],
    in_cur: str,
    out_cur: str,
) -> Tuple[bool, int, bool]:
    paths_raw = result.get("Paths")
    if paths_raw is None:
        return False, 0, False
    if not isinstance(paths_raw, list):
        raise RuntimeError(f"invalid Payment.Paths: expected list, got {type(paths_raw).__name__}")

    allowed = {str(in_cur), str(out_cur), XRP, "0000000000000000000000000000000000000000"}
    has_external = False
    for i, path in enumerate(paths_raw):
        if not isinstance(path, list):
            raise RuntimeError(f"invalid Payment.Paths[{i}]: expected list, got {type(path).__name__}")
        for j, step in enumerate(path):
            if not isinstance(step, dict):
                raise RuntimeError(
                    f"invalid Payment.Paths[{i}][{j}]: expected object, got {type(step).__name__}"
                )
            currency_raw = step.get("currency")
            if currency_raw is None:
                continue
            currency = str(currency_raw)
            if currency not in allowed:
                has_external = True
    return True, len(paths_raw), has_external


def extract_tx_hash_from_metadata_obj(obj: Dict[str, Any]) -> str:
    result = obj.get("result") if isinstance(obj.get("result"), dict) else {}
    return str(obj.get("tx_hash") or result.get("hash") or "").upper()


def _payment_deliver_field(result: Dict[str, Any]) -> Tuple[str, Any]:
    if result.get("DeliverMax") is not None:
        return "Payment.DeliverMax", result.get("DeliverMax")
    if result.get("Amount") is not None:
        return "Payment.Amount", result.get("Amount")
    raise RuntimeError("Payment missing required DeliverMax/Amount")


def _derive_payment_sendmax(
    *,
    account: str,
    deliver_currency: str,
    deliver_amount: Amount,
) -> Tuple[str, Optional[str], Amount]:
    if deliver_currency == XRP:
        return XRP, None, _copy_amount(deliver_amount)
    if not account:
        raise RuntimeError("Payment missing Account required to derive SendMax")
    return deliver_currency, account, _copy_amount(deliver_amount)


def build_intent_from_metadata_result(result: Dict[str, Any]) -> TxIntent:
    tx_type = str(result.get("TransactionType") or "").upper()
    if tx_type not in {"PAYMENT", "OFFERCREATE"}:
        raise RuntimeError(f"unsupported transaction type for rolling compare: {tx_type or 'unknown'}")

    flags_raw = result.get("Flags")
    try:
        flags = int(flags_raw) if flags_raw is not None else 0
    except Exception:
        flags = 0
    _validate_supported_flags(tx_type=tx_type, flags=flags)

    account = str(result.get("Account") or "") or None

    if tx_type == "PAYMENT":
        deliver_field_name, deliver_raw = _payment_deliver_field(result)
        deliver_currency, deliver_issuer, deliver_amount = _parse_amount_field(
            deliver_raw,
            field_name=deliver_field_name,
        )

        sendmax_raw = result.get("SendMax")
        if sendmax_raw is None:
            sendmax_currency, sendmax_issuer, sendmax_amount = _derive_payment_sendmax(
                account=account or "",
                deliver_currency=deliver_currency,
                deliver_amount=deliver_amount,
            )
        else:
            sendmax_currency, sendmax_issuer, sendmax_amount = _parse_amount_field(
                sendmax_raw,
                field_name="Payment.SendMax",
            )

        deliver_min_amount: Optional[Amount] = None
        if result.get("DeliverMin") is not None:
            if not bool(flags & TF_PARTIAL_PAYMENT):
                raise RuntimeError("Payment DeliverMin requires tfPartialPayment")
            dmin_currency, dmin_issuer, dmin_amount = _parse_amount_field(
                result.get("DeliverMin"),
                field_name="Payment.DeliverMin",
            )
            if not _same_asset(dmin_currency, dmin_issuer, deliver_currency, deliver_issuer):
                raise RuntimeError("Payment DeliverMin asset mismatch with DeliverMax/Amount")
            if dmin_amount > deliver_amount:
                raise RuntimeError("Payment DeliverMin exceeds DeliverMax/Amount")
            deliver_min_amount = dmin_amount

        source_funds_cap = _extract_source_funds_cap_from_metadata(
            result=result,
            account=account or "",
            in_cur=sendmax_currency,
            in_issuer=sendmax_issuer,
        )
        paths_present, paths_count, paths_has_external = _payment_paths_scope_flags(
            result=result,
            in_cur=sendmax_currency,
            out_cur=deliver_currency,
        )
        payment_limit_quality = (
            _quality_from_amounts(out_amount=deliver_amount, in_amount=sendmax_amount)
            if bool(flags & TF_LIMIT_QUALITY)
            else None
        )

        return TxIntent(
            transaction_type=tx_type,
            flags=flags,
            source_account=account,
            in_cur=sendmax_currency,
            out_cur=deliver_currency,
            payment_sendmax_amt=sendmax_amount,
            payment_sendmax_currency=sendmax_currency,
            payment_sendmax_issuer=sendmax_issuer,
            payment_delivermax_amt=deliver_amount,
            payment_delivermax_currency=deliver_currency,
            payment_delivermax_issuer=deliver_issuer,
            payment_destination_account=str(result.get("Destination") or "") or None,
            payment_delivermin_amt=deliver_min_amount,
            payment_partial_allowed=bool(flags & TF_PARTIAL_PAYMENT),
            payment_no_ripple_direct=bool(flags & TF_NO_RIPPLE_DIRECT),
            payment_limit_quality=payment_limit_quality,
            payment_paths_present=paths_present,
            payment_paths_count=paths_count,
            payment_paths_has_external_asset=paths_has_external,
            source_funds_cap_amt=source_funds_cap,
        )

    taker_gets_currency, taker_gets_issuer, taker_gets_amount = _parse_amount_field(
        result.get("TakerGets"),
        field_name="OfferCreate.TakerGets",
    )
    taker_pays_currency, taker_pays_issuer, taker_pays_amount = _parse_amount_field(
        result.get("TakerPays"),
        field_name="OfferCreate.TakerPays",
    )

    if _same_asset(
        taker_gets_currency,
        taker_gets_issuer,
        taker_pays_currency,
        taker_pays_issuer,
    ):
        raise RuntimeError("OfferCreate cannot exchange an asset for itself")

    cancel_sequence_raw = result.get("OfferSequence")
    offer_cancel_sequence = None
    if cancel_sequence_raw is not None:
        try:
            offer_cancel_sequence = int(cancel_sequence_raw)
        except Exception as e:
            raise RuntimeError(f"invalid OfferCreate.OfferSequence: {e}") from e
        if offer_cancel_sequence <= 0:
            raise RuntimeError("invalid OfferCreate.OfferSequence: must be positive")

    source_funds_cap = _extract_source_funds_cap_from_metadata(
        result=result,
        account=account or "",
        in_cur=taker_gets_currency,
        in_issuer=taker_gets_issuer,
    )

    return TxIntent(
        transaction_type=tx_type,
        flags=flags,
        source_account=account,
        in_cur=taker_gets_currency,
        out_cur=taker_pays_currency,
        source_funds_cap_amt=source_funds_cap,
        offer_taker_gets_amt=taker_gets_amount,
        offer_taker_gets_currency=taker_gets_currency,
        offer_taker_gets_issuer=taker_gets_issuer,
        offer_taker_pays_amt=taker_pays_amount,
        offer_taker_pays_currency=taker_pays_currency,
        offer_taker_pays_issuer=taker_pays_issuer,
        offer_limit_quality=_quality_from_amounts(out_amount=taker_pays_amount, in_amount=taker_gets_amount),
        offer_is_sell=bool(flags & TF_SELL),
        offer_is_ioc=bool(flags & TF_IMMEDIATE_OR_CANCEL),
        offer_is_fok=bool(flags & TF_FILL_OR_KILL),
        offer_is_passive=bool(flags & TF_PASSIVE),
        offer_cancel_sequence=offer_cancel_sequence,
    )


def load_tx_intents_from_metadata(
    *,
    metadata_ndjson: str,
    target_tx_hashes: Set[str],
) -> Dict[str, TxIntent]:
    out: Dict[str, TxIntent] = {}
    with open(metadata_ndjson, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                obj = json.loads(payload)
            except Exception as e:
                raise RuntimeError(
                    f"metadata parse failed at {metadata_ndjson}:{line_no}: {type(e).__name__}: {e}"
                ) from e
            if not isinstance(obj, dict):
                continue

            tx_hash = extract_tx_hash_from_metadata_obj(obj)
            if not tx_hash or tx_hash not in target_tx_hashes:
                continue

            result = obj.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(
                    f"metadata row missing result object for tx={tx_hash} at {metadata_ndjson}:{line_no}"
                )
            if tx_hash in out:
                raise RuntimeError(f"duplicate metadata rows found for tx={tx_hash}")

            try:
                out[tx_hash] = build_intent_from_metadata_result(result)
            except Exception as e:
                raise RuntimeError(f"metadata intent parse failed for tx={tx_hash}: {type(e).__name__}: {e}") from e

    missing = sorted(target_tx_hashes - set(out))
    if missing:
        sample = ", ".join(missing[:8])
        raise RuntimeError(
            "metadata missing tx intents for rolling window target set: "
            f"missing={len(missing)} sample=[{sample}]"
        )
    return out


__all__ = [
    "TxIntent",
    "TF_PARTIAL_PAYMENT",
    "TF_NO_RIPPLE_DIRECT",
    "TF_LIMIT_QUALITY",
    "TF_PASSIVE",
    "TF_IMMEDIATE_OR_CANCEL",
    "TF_FILL_OR_KILL",
    "TF_SELL",
    "TF_FULLY_CANONICAL_SIG",
    "apply_offer_tick_size_to_intent",
    "extract_tx_hash_from_metadata_obj",
    "build_intent_from_metadata_result",
    "load_tx_intents_from_metadata",
]
