from __future__ import annotations

import importlib.util
from pathlib import Path
from decimal import Decimal

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "empirical"
    / "scripts"
    / "empirical_compare_rolling.py"
)
_SPEC = importlib.util.spec_from_file_location("empirical_compare_rolling", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_extract_overlay_owner_funds_from_metadata_result = _MODULE._extract_overlay_owner_funds_from_metadata_result
_materialize_snapshot_book_offers = _MODULE._materialize_snapshot_book_offers
_prebook_primary_side_for_output_currency = _MODULE._prebook_primary_side_for_output_currency
_best_offer_quality_for_account = _MODULE.best_offer_quality_for_account


def test_prebook_primary_side_uses_output_currency() -> None:
    assert _prebook_primary_side_for_output_currency("XRP") == "getsXRP"
    assert _prebook_primary_side_for_output_currency("524C555344000000000000000000000000000000") == "getsrUSD"


def test_materialize_snapshot_book_offers_uses_full_snapshot_without_overlay_ids() -> None:
    payload = {
        "offers": [
            {
                "account": "rMaker",
                "offer_id": "offer-1",
                "sequence": 1,
                "taker_gets": "1000000",
                "taker_pays": {
                    "currency": "524C555344000000000000000000000000000000",
                    "issuer": "rIssuer",
                    "value": "2",
                },
                "quality_out_over_in": "0.000002",
            }
        ],
        "pre_overlay_offer_ids": [],
    }

    offers, mode = _materialize_snapshot_book_offers(
        side_payload=payload,
        side_name="getsXRP",
        tx_hash="ABC",
    )

    assert mode == "all_offers"
    assert len(offers) == 1
    assert offers[0]["Account"] == "rMaker"
    assert offers[0]["index"] == "offer-1"


def test_materialize_snapshot_book_offers_prefers_overlay_rows_when_present() -> None:
    payload = {
        "offers": [
            {
                "account": "rMaker1",
                "offer_id": "offer-1",
                "sequence": 1,
                "taker_gets": "1000000",
                "taker_pays": {
                    "currency": "524C555344000000000000000000000000000000",
                    "issuer": "rIssuer",
                    "value": "2",
                },
                "quality_out_over_in": "0.000002",
            },
            {
                "account": "rMaker2",
                "offer_id": "offer-2",
                "sequence": 2,
                "taker_gets": "2000000",
                "taker_pays": {
                    "currency": "524C555344000000000000000000000000000000",
                    "issuer": "rIssuer",
                    "value": "5",
                },
                "quality_out_over_in": "0.0000025",
            },
        ],
        "pre_overlay_offer_ids": ["offer-2"],
    }

    offers, mode = _materialize_snapshot_book_offers(
        side_payload=payload,
        side_name="getsXRP",
        tx_hash="ABC",
    )

    assert mode == "pre_overlay_only"
    assert [offer["index"] for offer in offers] == ["offer-2"]


def test_materialize_snapshot_book_offers_enriches_overlay_owner_funds_from_metadata_prestate() -> None:
    payload = {
        "offers": [
            {
                "account": "rMaker",
                "offer_id": "offer-1",
                "sequence": 1,
                "taker_gets": {
                    "currency": "524C555344000000000000000000000000000000",
                    "issuer": "rIssuer",
                    "value": "4.5",
                },
                "taker_pays": "2200000",
                "quality_out_over_in": "2.0454545454545454",
            }
        ],
        "pre_overlay_offer_ids": ["offer-1"],
    }

    offers, _mode = _materialize_snapshot_book_offers(
        side_payload=payload,
        side_name="getsrUSD",
        tx_hash="ABC",
        derived_owner_funds_by_offer_id={"offer-1": "4.5"},
    )

    assert offers[0]["owner_funds"] == "4.5"


def test_extract_overlay_owner_funds_from_metadata_result_uses_iou_pre_balance() -> None:
    result = {
        "hash": "ABC",
        "meta": {
            "AffectedNodes": [
                {
                    "DeletedNode": {
                        "LedgerEntryType": "Offer",
                        "LedgerIndex": "offer-1",
                        "FinalFields": {
                            "Account": "rMaker",
                            "TakerGets": {
                                "currency": "524C555344000000000000000000000000000000",
                                "issuer": "rIssuer",
                                "value": "0",
                            },
                            "TakerPays": "0",
                        },
                        "PreviousFields": {
                            "TakerGets": {
                                "currency": "524C555344000000000000000000000000000000",
                                "issuer": "rIssuer",
                                "value": "4.878455606343047",
                            },
                            "TakerPays": "2388065",
                        },
                    }
                },
                {
                    "ModifiedNode": {
                        "LedgerEntryType": "RippleState",
                        "FinalFields": {
                            "Balance": {
                                "currency": "524C555344000000000000000000000000000000",
                                "issuer": "rrrrrrrrrrrrrrrrrrrrBZbvji",
                                "value": "17.125",
                            },
                            "LowLimit": {
                                "currency": "524C555344000000000000000000000000000000",
                                "issuer": "rMaker",
                                "value": "1000",
                            },
                            "HighLimit": {
                                "currency": "524C555344000000000000000000000000000000",
                                "issuer": "rIssuer",
                                "value": "0",
                            },
                        },
                        "PreviousFields": {
                            "Balance": {
                                "currency": "524C555344000000000000000000000000000000",
                                "issuer": "rrrrrrrrrrrrrrrrrrrrBZbvji",
                                "value": "21.375",
                            }
                        },
                    }
                }
            ]
        },
    }

    out = _extract_overlay_owner_funds_from_metadata_result(result)

    assert out == {"offer-1": "21.375"}


def test_extract_overlay_owner_funds_from_metadata_result_uses_xrp_pre_balance_minus_reserve() -> None:
    result = {
        "hash": "ABC",
        "meta": {
            "AffectedNodes": [
                {
                    "ModifiedNode": {
                        "LedgerEntryType": "Offer",
                        "LedgerIndex": "offer-1",
                        "FinalFields": {
                            "Account": "rMaker",
                            "TakerGets": "1000000",
                            "TakerPays": {
                                "currency": "524C555344000000000000000000000000000000",
                                "issuer": "rIssuer",
                                "value": "2",
                            },
                        },
                        "PreviousFields": {
                            "TakerGets": "2500000",
                            "TakerPays": {
                                "currency": "524C555344000000000000000000000000000000",
                                "issuer": "rIssuer",
                                "value": "5",
                            },
                        },
                    }
                },
                {
                    "ModifiedNode": {
                        "LedgerEntryType": "AccountRoot",
                        "FinalFields": {
                            "Account": "rMaker",
                            "Balance": "12299999",
                            "OwnerCount": 4,
                        },
                        "PreviousFields": {
                            "Balance": "12345678",
                        },
                    }
                },
            ]
        },
    }

    out = _extract_overlay_owner_funds_from_metadata_result(result)

    assert out == {"offer-1": "10545678"}


def test_best_offer_quality_for_account_keeps_hidden_self_boundary() -> None:
    offers = [
        {
            "Account": "rSelf",
            "index": "self-best",
            "Sequence": 1,
            "TakerGets": "918178251",
            "TakerPays": {
                "currency": "524C555344000000000000000000000000000000",
                "issuer": "rIssuer",
                "value": "1913.67721",
            },
            "quality": "0.4797978709272500559276660874",
        },
        {
            "Account": "rExt",
            "index": "ext-best",
            "Sequence": 2,
            "TakerGets": "4799765850",
            "TakerPays": {
                "currency": "524C555344000000000000000000000000000000",
                "issuer": "rIssuer",
                "value": "10007.51179725",
            },
            "quality": "0.4796163069544364508393285372",
        },
    ]

    q = _best_offer_quality_for_account(
        offers,
        account="rSelf",
        in_cur="524C555344000000000000000000000000000000",
        out_cur="XRP",
    )

    assert q is not None
    assert q.rate > _best_offer_quality_for_account(
        offers,
        account="rExt",
        in_cur="524C555344000000000000000000000000000000",
        out_cur="XRP",
    ).rate


def test_best_offer_quality_for_account_ignores_explicitly_cancelled_offer_sequence() -> None:
    offers = [
        {
            "Account": "rSelf",
            "index": "self-cancelled",
            "Sequence": 417,
            "TakerGets": "924563704",
            "TakerPays": {
                "currency": "524C555344000000000000000000000000000000",
                "issuer": "rIssuer",
                "value": "1911.28675",
            },
            "quality": "0.4837388759169705958564302295",
        },
        {
            "Account": "rSelf",
            "index": "self-live",
            "Sequence": 500,
            "TakerGets": "918178251",
            "TakerPays": {
                "currency": "524C555344000000000000000000000000000000",
                "issuer": "rIssuer",
                "value": "1913.67721",
            },
            "quality": "0.4797978709272500559276660874",
        },
    ]

    q = _best_offer_quality_for_account(
        offers,
        account="rSelf",
        exclude_sequence=417,
        in_cur="524C555344000000000000000000000000000000",
        out_cur="XRP",
    )

    assert q is not None
    assert abs(q.rate.to_decimal() - Decimal("479797.87097")) < Decimal("1e-9")
