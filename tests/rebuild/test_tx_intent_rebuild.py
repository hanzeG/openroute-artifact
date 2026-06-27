from __future__ import annotations

import pytest

from xrpl_router import IOUAmount, Quality, XRPAmount
from xrpl_router.tx_intent import (
    TF_FILL_OR_KILL,
    TF_IMMEDIATE_OR_CANCEL,
    TF_LIMIT_QUALITY,
    TF_NO_RIPPLE_DIRECT,
    TF_PARTIAL_PAYMENT,
    TF_PASSIVE,
    TF_SELL,
    apply_offer_tick_size_to_intent,
    build_intent_from_metadata_result,
)


def test_payment_with_explicit_sendmax_and_delivermin_parses_flags_and_amounts() -> None:
    result = {
        "TransactionType": "Payment",
        "Account": "rSrc",
        "Flags": TF_PARTIAL_PAYMENT | TF_NO_RIPPLE_DIRECT,
        "SendMax": "1500000",
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
        "Paths": [[{"currency": "USD", "issuer": "rIssuer"}]],
    }

    intent = build_intent_from_metadata_result(result)

    assert intent.transaction_type == "PAYMENT"
    assert intent.in_cur == "XRP"
    assert intent.out_cur == "USD"
    assert intent.payment_partial_allowed is True
    assert intent.payment_no_ripple_direct is True
    assert intent.payment_sendmax_amt == XRPAmount(1_500_000)
    assert intent.payment_delivermax_amt == IOUAmount(1200000000000000, -15)
    assert intent.payment_delivermin_amt == IOUAmount(1000000000000000, -15)
    assert intent.payment_paths_present is True
    assert intent.payment_paths_count == 1
    assert intent.payment_paths_has_external_asset is False


def test_payment_without_sendmax_derives_max_source_amount_like_rippled() -> None:
    result = {
        "TransactionType": "Payment",
        "Account": "rSrc",
        "Flags": TF_LIMIT_QUALITY,
        "Amount": {
            "currency": "USD",
            "issuer": "rGateway",
            "value": "1.2",
        },
    }

    intent = build_intent_from_metadata_result(result)

    assert intent.in_cur == "USD"
    assert intent.out_cur == "USD"
    assert intent.payment_sendmax_currency == "USD"
    assert intent.payment_sendmax_issuer == "rSrc"
    assert intent.payment_sendmax_amt == IOUAmount(1200000000000000, -15)
    assert intent.payment_delivermax_amt == IOUAmount(1200000000000000, -15)
    assert intent.payment_limit_quality == Quality.from_amounts(
        IOUAmount(1200000000000000, -15),
        IOUAmount(1200000000000000, -15),
    )


def test_payment_rejects_delivermin_without_partial_payment_flag() -> None:
    result = {
        "TransactionType": "Payment",
        "Account": "rSrc",
        "SendMax": "1000000",
        "DeliverMax": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1",
        },
        "DeliverMin": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1",
        },
    }

    with pytest.raises(RuntimeError, match="DeliverMin requires tfPartialPayment"):
        build_intent_from_metadata_result(result)


def test_payment_marks_external_path_assets_for_single_path_scope_checks() -> None:
    result = {
        "TransactionType": "Payment",
        "Account": "rSrc",
        "SendMax": "1000000",
        "DeliverMax": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1",
        },
        "Paths": [[{"currency": "EUR", "issuer": "rOther"}]],
    }

    intent = build_intent_from_metadata_result(result)

    assert intent.payment_paths_present is True
    assert intent.payment_paths_count == 1
    assert intent.payment_paths_has_external_asset is True


def test_offercreate_uses_takergets_to_takerpays_execution_direction_for_both_sell_and_buy() -> None:
    sell_result = {
        "TransactionType": "OfferCreate",
        "Flags": TF_SELL,
        "TakerGets": "2000000",
        "TakerPays": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1.9",
        },
    }
    buy_result = {
        "TransactionType": "OfferCreate",
        "Flags": TF_FILL_OR_KILL,
        "TakerGets": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "2",
        },
        "TakerPays": "2000000",
    }

    sell_intent = build_intent_from_metadata_result(sell_result)
    buy_intent = build_intent_from_metadata_result(buy_result)

    assert sell_intent.in_cur == "XRP"
    assert sell_intent.out_cur == "USD"
    assert sell_intent.offer_is_sell is True
    assert buy_intent.in_cur == "USD"
    assert buy_intent.out_cur == "XRP"
    assert buy_intent.offer_is_fok is True


def test_offercreate_limit_quality_uses_new_core_quality_semantics() -> None:
    result = {
        "TransactionType": "OfferCreate",
        "Flags": 0,
        "TakerGets": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "0.000589",
        },
        "TakerPays": "288",
    }

    intent = build_intent_from_metadata_result(result)
    assert intent.offer_limit_quality == Quality.from_amounts(
        XRPAmount(288),
        IOUAmount(5890000000000000, -19),
    )


def test_offercreate_rejects_ioc_and_fok_together_like_rippled_preflight() -> None:
    result = {
        "TransactionType": "OfferCreate",
        "Flags": TF_IMMEDIATE_OR_CANCEL | TF_FILL_OR_KILL,
        "TakerGets": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "2",
        },
        "TakerPays": "2000000",
    }

    with pytest.raises(RuntimeError, match="tfImmediateOrCancel and tfFillOrKill"):
        build_intent_from_metadata_result(result)


def test_offercreate_rejects_redundant_same_asset_exchange() -> None:
    result = {
        "TransactionType": "OfferCreate",
        "Flags": 0,
        "TakerGets": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1",
        },
        "TakerPays": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "2",
        },
    }

    with pytest.raises(RuntimeError, match="cannot exchange an asset for itself"):
        build_intent_from_metadata_result(result)


def test_offercreate_tick_size_rounding_matches_rippled_preprocessing_shape() -> None:
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
    assert rounded.offer_taker_pays_amt == XRPAmount(932043590)
    assert rounded.offer_taker_gets_amt == IOUAmount(1910996800665291, -12)
    assert Quality.from_amounts(
        rounded.offer_taker_gets_amt,
        rounded.offer_taker_pays_amt,
    ).rate() == IOUAmount(4877264000000000, -10)


def test_offercreate_tick_size_rounding_sell_side_matches_rippled_preprocessing_shape() -> None:
    result = {
        "TransactionType": "OfferCreate",
        "Flags": TF_PASSIVE | TF_SELL,
        "TakerGets": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "29.802033",
        },
        "TakerPays": "14285661",
    }

    intent = build_intent_from_metadata_result(result)
    rounded = apply_offer_tick_size_to_intent(intent, 7)

    assert rounded.offer_tick_size == 7
    assert rounded.offer_taker_gets_amt == IOUAmount(2980203300000000, -14)
    assert rounded.offer_taker_pays_amt == XRPAmount(14285661)
    assert rounded.offer_limit_quality == Quality.from_amounts(
        rounded.offer_taker_pays_amt,
        rounded.offer_taker_gets_amt,
    )


def test_offercreate_tick_size_rounding_sell_side_native_result_uses_nearest_even() -> None:
    floor_result = {
        "TransactionType": "OfferCreate",
        "Flags": TF_IMMEDIATE_OR_CANCEL | TF_SELL,
        "TakerGets": {
            "currency": "524C555344000000000000000000000000000000",
            "issuer": "rIssuer",
            "value": "11.909165",
        },
        "TakerPays": "5974537",
    }
    ceil_result = {
        "TransactionType": "OfferCreate",
        "Flags": TF_PASSIVE | TF_SELL,
        "TakerGets": {
            "currency": "524C555344000000000000000000000000000000",
            "issuer": "rIssuer",
            "value": "24.916066",
        },
        "TakerPays": "12474775",
    }

    rounded_floor = apply_offer_tick_size_to_intent(
        build_intent_from_metadata_result(floor_result),
        7,
    )
    rounded_ceil = apply_offer_tick_size_to_intent(
        build_intent_from_metadata_result(ceil_result),
        7,
    )

    assert rounded_floor.offer_taker_pays_amt == XRPAmount(5974537)
    assert rounded_ceil.offer_taker_pays_amt == XRPAmount(12474777)


def test_offercreate_extracts_source_funds_cap_from_metadata_prestate() -> None:
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
    assert intent.source_funds_cap_amt == IOUAmount(7500000000000000, -15)


def test_real_payment_sample_with_no_ripple_direct_and_external_paths_parses_as_expected() -> None:
    # Real tx from two-week dataset:
    # EC24199175BCE94DF4D6CE59F21470CFF707E69A700EE7FD7BACCB72FB825C5E
    result = {
        "TransactionType": "Payment",
        "Flags": 196608,
        "Account": "rnKh83nJyzmtpf3huo8yaAV2NyAgtHPPsd",
        "SendMax": "137102",
        "DeliverMax": {
            "currency": "589",
            "issuer": "rfcasq9uRbvwcmLFvc4ti3j8Qt1CYCGgHz",
            "value": "1000000000000000",
        },
        "Amount": {
            "currency": "589",
            "issuer": "rfcasq9uRbvwcmLFvc4ti3j8Qt1CYCGgHz",
            "value": "1000000000000000",
        },
        "DeliverMin": {
            "currency": "589",
            "issuer": "rfcasq9uRbvwcmLFvc4ti3j8Qt1CYCGgHz",
            "value": "4728.64527442193",
        },
        "Paths": [[
            {"currency": "5265646479000000000000000000000000000000", "issuer": "rw5Kdcm1vW4xVefFafjMbevs5SL9fYP36U", "type": 48},
            {"currency": "524C555344000000000000000000000000000000", "issuer": "rMxCKbEDwqr76QuheSUMdEGf4B9xJ8m5De", "type": 48},
        ]],
    }

    intent = build_intent_from_metadata_result(result)

    assert intent.in_cur == "XRP"
    assert intent.out_cur == "589"
    assert intent.payment_sendmax_amt == XRPAmount(137102)
    assert intent.payment_delivermax_amt == IOUAmount(1000000000000000, 0)
    assert intent.payment_delivermin_amt == IOUAmount(4728645274421930, -12)
    assert intent.payment_partial_allowed is True
    assert intent.payment_no_ripple_direct is True
    assert intent.payment_paths_present is True
    assert intent.payment_paths_count == 1
    assert intent.payment_paths_has_external_asset is True


def test_real_payment_sample_without_sendmax_derives_sendmax_like_rippled() -> None:
    # Real tx from two-week dataset:
    # 13993C112CC47525C1A19C6C83EF8305A3FF1C1E21C2514DBDEB2291544B6BA1
    result = {
        "TransactionType": "Payment",
        "Flags": 0,
        "Account": "rnXZ876yGEhoATQSYegtD8bg8wpA8TTX5a",
        "DeliverMax": "1",
        "Amount": "1",
    }

    intent = build_intent_from_metadata_result(result)

    assert intent.in_cur == "XRP"
    assert intent.out_cur == "XRP"
    assert intent.payment_sendmax_currency == "XRP"
    assert intent.payment_sendmax_issuer is None
    assert intent.payment_sendmax_amt == XRPAmount(1)
    assert intent.payment_delivermax_amt == XRPAmount(1)


def test_real_payment_sample_with_limit_quality_parses_core_quality() -> None:
    # Real tx from two-week dataset:
    # 5BC8E9EF53BD3A2B120364424DAC79F5A249BCA573C404F4487D39CE4FB6D0B8
    result = {
        "TransactionType": "Payment",
        "Flags": 393216,
        "Account": "rghL9q8iPW6P4ZqG53nv3VNkVBKWWngdd",
        "SendMax": {
            "currency": "CNY",
            "issuer": "rKiCet8SdvWxPXnAgYarFUXMh1zCPz432Y",
            "value": "693.2663783999999",
        },
        "DeliverMax": {
            "currency": "XLM",
            "issuer": "rKiCet8SdvWxPXnAgYarFUXMh1zCPz432Y",
            "value": "408",
        },
        "Amount": {
            "currency": "XLM",
            "issuer": "rKiCet8SdvWxPXnAgYarFUXMh1zCPz432Y",
            "value": "408",
        },
        "Paths": [
            [{"currency": "XLM", "issuer": "rKiCet8SdvWxPXnAgYarFUXMh1zCPz432Y", "type": 48}],
            [{"currency": "XRP", "type": 16}, {"currency": "XLM", "issuer": "rKiCet8SdvWxPXnAgYarFUXMh1zCPz432Y", "type": 48}],
            [{"currency": "XAG", "issuer": "rpG9E7B3ocgaKqG7vmrsu3jmGwex8W4xAG", "type": 48}, {"currency": "XLM", "issuer": "rKiCet8SdvWxPXnAgYarFUXMh1zCPz432Y", "type": 48}],
        ],
    }

    intent = build_intent_from_metadata_result(result)

    assert intent.in_cur == "CNY"
    assert intent.out_cur == "XLM"
    assert intent.payment_partial_allowed is True
    assert intent.payment_limit_quality is not None
    assert intent.payment_limit_quality.rate() == IOUAmount(1699182300000000, -15)
    assert intent.payment_paths_present is True
    assert intent.payment_paths_has_external_asset is True


def test_real_offercreate_samples_cover_ioc_fok_sell_and_passive_cancel() -> None:
    ioc_buy = {
        "TransactionType": "OfferCreate",
        "Flags": 131072,
        "Account": "r9hNR8G9nARPL7LSAQoLv5EZdVjHBiXr7Q",
        "TakerGets": {"currency": "USD", "issuer": "rKiCet8SdvWxPXnAgYarFUXMh1zCPz432Y", "value": "161.1215647693479"},
        "TakerPays": {"currency": "CNY", "issuer": "rKiCet8SdvWxPXnAgYarFUXMh1zCPz432Y", "value": "1133.728951500459"},
    }
    fok_buy = {
        "TransactionType": "OfferCreate",
        "Flags": 262144,
        "Account": "rKv64MDJ5ciUYQ8rzPUZDNiJRzSnyyv6nY",
        "TakerGets": {"currency": "666", "issuer": "rhvf9fe6PP3GC8Bku2Ug7iQPjPDxYZfrxN", "value": "12.302491"},
        "TakerPays": "1035797",
    }
    sell = {
        "TransactionType": "OfferCreate",
        "Flags": 524288,
        "Account": "rGidge2jZYXk2vMy9yjEXyEx6JipQpaxek",
        "TakerGets": "7818465",
        "TakerPays": {"currency": "524C555344000000000000000000000000000000", "issuer": "rMxCKbEDwqr76QuheSUMdEGf4B9xJ8m5De", "value": "16.0941131785493"},
    }
    passive_cancel = {
        "TransactionType": "OfferCreate",
        "Flags": 65536,
        "Account": "rpiFwLYi6Gb1ESHYorn2QG1WU5vw2u4exQ",
        "TakerGets": "2338947330",
        "TakerPays": {"currency": "524C555344000000000000000000000000000000", "issuer": "rMxCKbEDwqr76QuheSUMdEGf4B9xJ8m5De", "value": "4801.39108"},
        "OfferSequence": 110501033,
    }

    ioc_intent = build_intent_from_metadata_result(ioc_buy)
    fok_intent = build_intent_from_metadata_result(fok_buy)
    sell_intent = build_intent_from_metadata_result(sell)
    passive_intent = build_intent_from_metadata_result(passive_cancel)

    assert ioc_intent.offer_is_ioc is True
    assert ioc_intent.in_cur == "USD"
    assert ioc_intent.out_cur == "CNY"
    assert ioc_intent.offer_limit_quality is not None

    assert fok_intent.offer_is_fok is True
    assert fok_intent.in_cur == "666"
    assert fok_intent.out_cur == "XRP"
    assert fok_intent.offer_limit_quality is not None

    assert sell_intent.offer_is_sell is True
    assert sell_intent.in_cur == "XRP"
    assert sell_intent.out_cur == "524C555344000000000000000000000000000000"
    assert sell_intent.offer_limit_quality is not None

    assert passive_intent.offer_is_passive is True
    assert passive_intent.offer_cancel_sequence == 110501033
    assert passive_intent.in_cur == "XRP"
    assert passive_intent.out_cur == "524C555344000000000000000000000000000000"


def test_real_offercreate_sample_extracts_xrp_source_cap_from_metadata_prestate() -> None:
    # Real tx from two-week dataset:
    # 86E9D8C7D530950D82EFBBB7ADAE2D1506559DFEE9F53D0B1E457EBCAD7093DE
    result = {
        "TransactionType": "OfferCreate",
        "Flags": 0,
        "Account": "rw22L9LYRbEKQ5h7YiHJnxUAmYGERjCSmK",
        "Fee": "12",
        "TakerGets": "8539120",
        "TakerPays": {
            "currency": "3352444559450000000000000000000000000000",
            "issuer": "rHjyBqFM5oQvXu1soWtATC4r1V6GBnhCQQ",
            "value": "84629.4786917674",
        },
        "OfferSequence": 97631360,
        "meta": {
            "AffectedNodes": [
                {
                    "ModifiedNode": {
                        "LedgerEntryType": "AccountRoot",
                        "FinalFields": {
                            "Account": "rw22L9LYRbEKQ5h7YiHJnxUAmYGERjCSmK",
                            "Balance": "661076280",
                            "OwnerCount": 79,
                        },
                        "PreviousFields": {
                            "Balance": "661076292",
                            "OwnerCount": 80,
                        },
                    }
                }
            ]
        },
    }

    intent = build_intent_from_metadata_result(result)

    assert intent.offer_cancel_sequence == 97631360
    assert intent.source_funds_cap_amt == XRPAmount(644076280)


def test_ticket_payment_source_cap_uses_owner_count_after_ticket_consumption() -> None:
    result = {
        "TransactionType": "Payment",
        "Account": "rUser",
        "Sequence": 0,
        "TicketSequence": 123,
        "Fee": "16",
        "Flags": 0x00020000,
        "SendMax": "909479630",
        "Amount": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "10953.094651914",
        },
        "DeliverMin": {
            "currency": "USD",
            "issuer": "rIssuer",
            "value": "1825.49752016125",
        },
        "meta": {
            "AffectedNodes": [
                {
                    "ModifiedNode": {
                        "LedgerEntryType": "AccountRoot",
                        "FinalFields": {
                            "Account": "rUser",
                            "Balance": "35600000",
                            "OwnerCount": 173,
                            "TicketCount": 43,
                        },
                        "PreviousFields": {
                            "Balance": "943448222",
                            "OwnerCount": 174,
                            "TicketCount": 44,
                        },
                    }
                }
            ]
        },
    }

    intent = build_intent_from_metadata_result(result)

    assert intent.source_funds_cap_amt == XRPAmount(907848206)
