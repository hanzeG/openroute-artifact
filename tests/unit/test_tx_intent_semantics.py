from __future__ import annotations

from decimal import Decimal

import pytest

from xrpl_router.tx_intent import (
    TF_FILL_OR_KILL,
    TF_IMMEDIATE_OR_CANCEL,
    TF_LIMIT_QUALITY,
    TF_NO_RIPPLE_DIRECT,
    TF_PASSIVE,
    TF_PARTIAL_PAYMENT,
    TF_SELL,
    apply_offer_tick_size_to_intent,
    build_intent_from_metadata_result,
)


def test_build_payment_intent_required_fields_and_partial_flag() -> None:
    result = {
        "TransactionType": "Payment",
        "Flags": TF_PARTIAL_PAYMENT,
        "SendMax": "1500000",  # 1.5 XRP
        "DeliverMax": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1.2",
        },
        "DeliverMin": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1.0",
        },
    }

    intent = build_intent_from_metadata_result(result)

    assert intent.transaction_type == "PAYMENT"
    assert intent.in_cur == "XRP"
    assert intent.out_cur == "USD"
    assert intent.payment_partial_allowed is True
    assert intent.payment_sendmax_currency == "XRP"
    assert intent.payment_currency == "USD"
    assert intent.payment_delivermin_amt is not None


def test_build_payment_intent_delivermin_currency_mismatch_fails() -> None:
    result = {
        "TransactionType": "Payment",
        "SendMax": "1000000",
        "DeliverMax": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1",
        },
        "DeliverMin": {
            "currency": "EUR",
            "issuer": "rIssuer",
            "value": "1",
        },
    }

    with pytest.raises(RuntimeError, match="DeliverMin currency mismatch"):
        build_intent_from_metadata_result(result)


def test_build_payment_intent_rejects_unsupported_flags_bits() -> None:
    result = {
        "TransactionType": "Payment",
        "Flags": 0x00000020,  # unsupported in current single-path implementation
        "SendMax": "1000000",
        "DeliverMax": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1",
        },
    }

    with pytest.raises(RuntimeError, match="unsupported PAYMENT flags bits"):
        build_intent_from_metadata_result(result)


def test_build_payment_intent_marks_external_path_assets_scope() -> None:
    result = {
        "TransactionType": "Payment",
        "SendMax": "1000000",
        "DeliverMax": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1",
        },
        "Paths": [[{"currency": "EUR", "issuer": "rOther", "type": 48}]],
    }

    intent = build_intent_from_metadata_result(result)
    assert intent.payment_paths_present is True
    assert intent.payment_paths_count == 1
    assert intent.payment_paths_has_external_asset is True


def test_build_payment_intent_limit_quality_from_sendmax_delivermax() -> None:
    result = {
        "TransactionType": "Payment",
        "Flags": TF_LIMIT_QUALITY,
        "SendMax": "2000000",  # 2 XRP, router in-units for XRP are drops
        "DeliverMax": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1.5",
        },
    }

    intent = build_intent_from_metadata_result(result)
    assert intent.payment_limit_quality is not None
    q_dec = Decimal(intent.payment_limit_quality.rate.to_decimal())
    expected = Decimal("1.5") / Decimal("2000000")
    assert abs(q_dec - expected) < Decimal("1e-15")


def test_build_payment_intent_marks_no_ripple_direct_flag() -> None:
    result = {
        "TransactionType": "Payment",
        "Flags": TF_NO_RIPPLE_DIRECT,
        "SendMax": "1000000",
        "DeliverMax": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1",
        },
        "Paths": [[{"currency": "USD", "issuer": "rIssuer"}]],
    }

    intent = build_intent_from_metadata_result(result)
    assert intent.payment_no_ripple_direct is True
    assert intent.payment_paths_present is True


def test_build_offercreate_intent_sell_vs_buy_mapping() -> None:
    sell_result = {
        "TransactionType": "OfferCreate",
        "Flags": TF_SELL,
        "TakerGets": "2000000",  # sell XRP
        "TakerPays": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1.9",
        },
    }
    sell_intent = build_intent_from_metadata_result(sell_result)
    assert sell_intent.transaction_type == "OFFERCREATE"
    assert sell_intent.offer_is_sell is True
    assert sell_intent.in_cur == "XRP"
    assert sell_intent.out_cur == "USD"
    assert sell_intent.offer_limit_quality is not None

    buy_result = {
        "TransactionType": "OfferCreate",
        "Flags": TF_FILL_OR_KILL,
        "TakerGets": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "2",
        },
        "TakerPays": "2000000",  # pay XRP
    }
    buy_intent = build_intent_from_metadata_result(buy_result)
    assert buy_intent.transaction_type == "OFFERCREATE"
    assert buy_intent.offer_is_sell is False
    assert buy_intent.offer_is_fok is True
    assert buy_intent.in_cur == "USD"
    assert buy_intent.out_cur == "XRP"
    assert buy_intent.offer_limit_quality is not None

    # sanity: quality is positive
    assert buy_intent.offer_limit_quality.rate.mantissa > 0
    assert Decimal(buy_intent.offer_limit_quality.rate.to_decimal()) > 0


def test_build_offercreate_intent_marks_passive_and_ioc_flags() -> None:
    result = {
        "TransactionType": "OfferCreate",
        "Flags": TF_PASSIVE | TF_IMMEDIATE_OR_CANCEL,
        "OfferSequence": 123,
        "TakerGets": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "2",
        },
        "TakerPays": "2000000",
    }

    intent = build_intent_from_metadata_result(result)
    assert intent.offer_is_passive is True
    assert intent.offer_is_ioc is True
    assert intent.offer_cancel_sequence == 123


def test_build_offercreate_intent_rejects_unsupported_flags_bits() -> None:
    result = {
        "TransactionType": "OfferCreate",
        "Flags": 0x00000020,
        "TakerGets": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "2",
        },
        "TakerPays": "2000000",
    }

    with pytest.raises(RuntimeError, match="unsupported OFFERCREATE flags bits"):
        build_intent_from_metadata_result(result)


def test_build_offercreate_limit_quality_keeps_router_domain_precision_for_tiny_amounts() -> None:
    # Regression: very small TakerGets can over-tighten limit quality if we
    # derive it via coarse STAmount-like divide.
    result = {
        "TransactionType": "OfferCreate",
        "Flags": 0,
        "TakerGets": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "0.000589",
        },
        "TakerPays": "288",  # 288 drops
    }

    intent = build_intent_from_metadata_result(result)
    assert intent.offer_limit_quality is not None

    # Router quality domain for out=XRP is drops-per-IOU.
    q_dec = Decimal(intent.offer_limit_quality.rate.to_decimal())
    expected = Decimal("288") / Decimal("0.000589")
    assert abs(q_dec - expected) < Decimal("1e-6")

    # Guard against the old over-tightened mapping (~489400 for this case).
    assert q_dec < Decimal("489000")


def test_apply_offer_tick_size_to_buy_offer_matches_rippled_rounding() -> None:
    result = {
        "TransactionType": "OfferCreate",
        "Flags": TF_PASSIVE,
        "TakerGets": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1910.99693",
        },
        "TakerPays": "932043590",
    }

    intent = build_intent_from_metadata_result(result)
    rounded = apply_offer_tick_size_to_intent(intent, 7)

    assert rounded.offer_tick_size == 7
    assert rounded.offer_taker_gets_amt is not None
    assert rounded.offer_taker_pays_amt is not None
    assert rounded.offer_limit_quality is not None
    assert Decimal(rounded.offer_taker_gets_amt.to_decimal()) == Decimal("1910.996800665291")
    assert Decimal(rounded.offer_taker_pays_amt.to_decimal()) == Decimal("932.04359")
    assert Decimal(rounded.offer_limit_quality.rate.to_decimal()) == Decimal("487726.4")


def test_build_offercreate_intent_extracts_source_funds_cap_from_trustline_prestate() -> None:
    result = {
        "TransactionType": "OfferCreate",
        "Flags": 0,
        "Account": "rUser",
        "TakerGets": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1000",
        },
        "TakerPays": "1000000000",
        "meta": {
            "AffectedNodes": [
                {
                    "ModifiedNode": {
                        "LedgerEntryType": "RippleState",
                        "FinalFields": {
                            "LowLimit": {"currency": "USD", "issuer": "rUser", "value": "1000"},
                            "HighLimit": {"currency": "USD", "issuer": "rIssuer", "value": "0"},
                            "Balance": {"currency": "USD", "issuer": "rrrrrrrrrrrrrrrrrrrrBZbvji", "value": "2"},
                        },
                        "PreviousFields": {
                            "Balance": {"currency": "USD", "issuer": "rrrrrrrrrrrrrrrrrrrrBZbvji", "value": "7.5"},
                        },
                    }
                }
            ]
        },
    }

    intent = build_intent_from_metadata_result(result)
    assert intent.source_funds_cap_amt is not None
    assert Decimal(intent.source_funds_cap_amt.to_decimal()) == Decimal("7.5")


def test_build_offercreate_intent_extracts_xrp_source_funds_cap_minus_reserve_and_fee() -> None:
    result = {
        "TransactionType": "OfferCreate",
        "Flags": TF_SELL | TF_PASSIVE,
        "Account": "rUser",
        "Fee": "50",
        "TakerGets": "20000000",
        "TakerPays": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "41.201536",
        },
        "meta": {
            "AffectedNodes": [
                {
                    "ModifiedNode": {
                        "LedgerEntryType": "AccountRoot",
                        "FinalFields": {
                            "Account": "rUser",
                            "Balance": "2400000",
                            "OwnerCount": 7,
                            "Sequence": 2,
                        },
                        "PreviousFields": {
                            "Balance": "4091414",
                            "Sequence": 1,
                        },
                    }
                }
            ]
        },
    }

    intent = build_intent_from_metadata_result(result)

    assert intent.source_funds_cap_amt is not None
    assert Decimal(intent.source_funds_cap_amt.to_decimal()) == Decimal("1.691364")
