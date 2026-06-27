from __future__ import annotations

import importlib.util
import json
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from xrpl_router.core import IOUAmount, Quality, XRPAmount
from xrpl_router.tx_intent import TxIntent, apply_offer_tick_size_to_intent

RUSD = "524C555344000000000000000000000000000000"
XRP = "XRP"


def _load_fit_module():
    path = Path("empirical/scripts/empirical_fit_single_direct.py")
    spec = importlib.util.spec_from_file_location("empirical_fit_single_direct_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _state_report(fit, *, currency: str, pre, post, state_change: str):
    return fit._state_transition_report(
        pre_state=pre,
        post_state=post,
        state_change=state_change,
        currency=currency,
        issuer=None,
    )


def test_real_offer_legs_skip_deleted_offer_without_recoverable_previous_fields() -> None:
    fit = _load_fit_module()
    result = {
        "meta": {
            "AffectedNodes": [
                {
                    "DeletedNode": {
                        "LedgerEntryType": "Offer",
                        "LedgerIndex": "DEADBEEF",
                        "FinalFields": {
                            "Account": "rOwner",
                            "TakerGets": {"currency": RUSD, "issuer": "rIssuer", "value": "0.5"},
                            "TakerPays": "287",
                        },
                    }
                }
            ]
        }
    }

    legs = fit._build_real_offer_legs(result, in_cur=XRP, out_cur=RUSD)

    assert legs == []


def test_real_offer_legs_reconstruct_canonical_endpoint_state_reports() -> None:
    fit = _load_fit_module()
    result = {
        "meta": {
            "AffectedNodes": [
                {
                    "ModifiedNode": {
                        "LedgerEntryType": "Offer",
                        "LedgerIndex": "F501",
                        "FinalFields": {
                            "Account": "rOwner",
                            "TakerGets": "3001881208",
                            "TakerPays": {"currency": RUSD, "issuer": "rIssuer", "value": "6093.817883868158"},
                        },
                        "PreviousFields": {
                            "TakerGets": "7920977698",
                            "TakerPays": {"currency": RUSD, "issuer": "rIssuer", "value": "16079.58221907563"},
                        },
                    }
                }
            ]
        }
    }

    legs = fit._build_real_offer_legs(result, in_cur=RUSD, out_cur=XRP)

    assert len(legs) == 1
    leg = legs[0]
    assert leg["out"]["drops"] == 4_919_096_490
    assert leg["in"]["value"] == "9985.764335207480"
    assert (leg["in"]["mantissa"], leg["in"]["exponent"]) == (9985764335207480, -12)
    assert leg["in"]["state_change"] == fit.STATE_CHANGE_DECREASE
    assert leg["in"]["pre_state"]["value"] == "16079.58221907563"
    assert leg["in"]["post_state"]["value"] == "6093.817883868158"


def test_strict_compare_uses_endpoint_state_equality() -> None:
    fit = _load_fit_module()
    pre_iou = fit._decimal_to_iou_amount(Decimal("16079.58221907563"))
    model_iou = fit._decimal_to_iou_amount(Decimal("9985.764335207480"))
    post_iou = pre_iou - model_iou
    model = {
        "legs": [
            {
                "src": "CLOB",
                "source_id": "F501",
                "out": {"kind": "XRP", "drops": 4_919_096_490, "value": "4919.09649"},
                "in": {
                    "kind": "IOU",
                    "mantissa": model_iou.mantissa,
                    "exponent": model_iou.exponent,
                    "value": "9985.764335207480",
                },
            }
        ]
    }
    real = {
        "legs": [
            {
                "src": "CLOB",
                "offer_id": "F501",
                "out": _state_report(
                    fit,
                    currency=XRP,
                    pre=XRPAmount(7_920_977_698),
                    post=XRPAmount(3_001_881_208),
                    state_change=fit.STATE_CHANGE_DECREASE,
                ),
                "in": _state_report(
                    fit,
                    currency=RUSD,
                    pre=pre_iou,
                    post=post_iou,
                    state_change=fit.STATE_CHANGE_DECREASE,
                ),
            }
        ]
    }

    comparison = fit._compare_reports(model=model, real=real)

    assert comparison["strict_exact_match"] is True
    assert comparison["structure_match"] is True
    assert comparison["total_in_match"] is True
    assert comparison["total_in_diff"] == "0"


def test_strict_compare_uses_rippled_number_projection_for_iou_endpoint() -> None:
    fit = _load_fit_module()
    model = {
        "legs": [
            {
                "src": "CLOB",
                "source_id": "8EF7",
                "out": {"kind": "XRP", "drops": 9_075_992, "value": "9.075992"},
                "in": {
                    "kind": "IOU",
                    "mantissa": 1861431502790658,
                    "exponent": -14,
                    "value": "18.61431502790658",
                },
            }
        ]
    }
    real = {
        "legs": [
            {
                "src": "CLOB",
                "offer_id": "8EF7",
                "out": _state_report(
                    fit,
                    currency=XRP,
                    pre=XRPAmount(9_075_992),
                    post=XRPAmount(0),
                    state_change=fit.STATE_CHANGE_DECREASE,
                ),
                "in": _state_report(
                    fit,
                    currency=RUSD,
                    pre=fit._decimal_to_iou_amount(Decimal("1894.838105600513")),
                    post=fit._decimal_to_iou_amount(Decimal("1876.223790572606")),
                    state_change=fit.STATE_CHANGE_DECREASE,
                ),
            }
        ]
    }

    comparison = fit._compare_reports(model=model, real=real)

    assert comparison["strict_exact_match"] is True
    assert comparison["structure_match"] is True
    assert comparison["total_in_match"] is False
    assert Decimal(comparison["total_in_diff"]) == Decimal("-0.00000000000042")
    assert comparison["total_in_endpoint_signed_rel_err"] == "0"
    assert comparison["max_side_endpoint_unsigned_rel_err"] == "0"


def test_aggregate_predicted_side_stats_follow_intent_role() -> None:
    fit = _load_fit_module()
    xrp_in = XRPAmount(1_000)
    iou_out = fit._decimal_to_iou_amount(Decimal("2"))
    model = {
        "legs": [
            {
                "src": "CLOB",
                "source_id": "OFFER1",
                "in": {"kind": "XRP", "drops": xrp_in.drops, "value": "0.001"},
                "out": {
                    "kind": "IOU",
                    "mantissa": iou_out.mantissa,
                    "exponent": iou_out.exponent,
                    "value": "2",
                },
            }
        ],
        "summary": {
            "total_in": fit._amount_to_report(xrp_in, currency=XRP),
            "total_out": fit._amount_to_report(iou_out, currency=RUSD),
        },
    }
    real = {
        "legs": [
            {
                "src": "CLOB",
                "offer_id": "OFFER1",
                "in": _state_report(
                    fit,
                    currency=XRP,
                    pre=XRPAmount(10_000),
                    post=XRPAmount(9_000),
                    state_change=fit.STATE_CHANGE_DECREASE,
                ),
                "out": _state_report(
                    fit,
                    currency=RUSD,
                    pre=fit._decimal_to_iou_amount(Decimal("7")),
                    post=fit._decimal_to_iou_amount(Decimal("5")),
                    state_change=fit.STATE_CHANGE_DECREASE,
                ),
            }
        ],
        "summary": {
            "total_in": fit._amount_to_report(xrp_in, currency=XRP),
            "total_out": fit._amount_to_report(iou_out, currency=RUSD),
        },
    }
    comparison = fit._compare_reports(model=model, real=real)
    report = {
        "tx_hash": "ABC",
        "ledger_index": 1,
        "transaction_index": 0,
        "intent": {
            "transaction_type": "OfferCreate",
            "is_sell": True,
            "in_cur": XRP,
            "out_cur": RUSD,
            "taker_gets": fit._amount_to_report(xrp_in, currency=XRP),
            "taker_pays": fit._amount_to_report(iou_out, currency=RUSD),
            "source_funds_cap": None,
        },
        "model": model,
        "real": real,
        "comparison": comparison,
    }
    comparison.update(fit._endpoint_prediction_role_fields(report))

    builder = fit._AggregateStatsBuilder()
    builder.update(report)
    aggregate = builder.as_dict()

    assert comparison["predicted_side"] == "out"
    assert comparison["constraint_side"] == "in"
    assert aggregate["endpoint_prediction_role_counts"] == {
        "offer_sell_input_bound_output_predicted": 1,
    }
    assert aggregate["structure_matched_endpoint_ttest_predicted_side"]["unsigned_relative_error"]["count"] == 1
    assert aggregate["structure_matched_endpoint_ttest_constraint_side"]["unsigned_relative_error"]["count"] == 1


def test_strict_compare_uses_rippled_number_projection_for_amm_iou_increase() -> None:
    fit = _load_fit_module()
    pre_iou = IOUAmount(1437008427327348, -9)
    model_iou = IOUAmount(5882305137800608, -19)
    post_iou = IOUAmount(1437008427915579, -9)
    model = {
        "legs": [
            {
                "src": "AMM",
                "source_id": "AMM",
                "out": {"kind": "XRP", "drops": 282, "value": "0.000282"},
                "in": {
                    "kind": "IOU",
                    "mantissa": model_iou.mantissa,
                    "exponent": model_iou.exponent,
                    "value": "0.0005882305137800608",
                },
            }
        ]
    }
    real = {
        "legs": [
            {
                "src": "AMM",
                "offer_id": "AMM",
                "out": _state_report(
                    fit,
                    currency=XRP,
                    pre=XRPAmount(691167997456),
                    post=XRPAmount(691167997174),
                    state_change=fit.STATE_CHANGE_DECREASE,
                ),
                "in": _state_report(
                    fit,
                    currency=RUSD,
                    pre=pre_iou,
                    post=post_iou,
                    state_change=fit.STATE_CHANGE_INCREASE,
                ),
            }
        ]
    }

    comparison = fit._compare_reports(model=model, real=real)

    assert comparison["strict_exact_match"] is True
    assert comparison["structure_match"] is True
    assert comparison["total_in_match"] is False
    assert Decimal(comparison["total_in_diff"]) == Decimal("-0.0000000004862199392")
    assert comparison["total_in_endpoint_signed_rel_err"] == "0"
    assert comparison["max_side_endpoint_unsigned_rel_err"] == "0"


def test_fit_bucket_classifies_benchmark_exact_match() -> None:
    fit = _load_fit_module()
    pre_iou = fit._decimal_to_iou_amount(Decimal("16079.58221907563"))
    model_iou = fit._decimal_to_iou_amount(Decimal("9985.764335207480"))
    post_iou = pre_iou - model_iou
    model = {
        "legs": [
            {
                "src": "CLOB",
                "source_id": "F501",
                "out": {"kind": "XRP", "drops": 4_919_096_490, "value": "4919.09649"},
                "in": {
                    "kind": "IOU",
                    "mantissa": model_iou.mantissa,
                    "exponent": model_iou.exponent,
                    "value": "9985.764335207480",
                },
            }
        ]
    }
    real = {
        "legs": [
            {
                "src": "CLOB",
                "offer_id": "F501",
                "out": _state_report(
                    fit,
                    currency=XRP,
                    pre=XRPAmount(7_920_977_698),
                    post=XRPAmount(3_001_881_208),
                    state_change=fit.STATE_CHANGE_DECREASE,
                ),
                "in": _state_report(
                    fit,
                    currency=RUSD,
                    pre=pre_iou,
                    post=post_iou,
                    state_change=fit.STATE_CHANGE_DECREASE,
                ),
            }
        ]
    }

    comparison = fit._compare_reports(model=model, real=real)
    bucket = fit._classify_fit_bucket(model=model, real=real, comparison=comparison)

    assert bucket == fit.FIT_BUCKET_STRICT_EXACT


def test_fraction_parser_accepts_scientific_notation() -> None:
    fit = _load_fit_module()
    assert fit._fraction_from_decimal_text("1.5800000000000001e-06", field_name="v") == Fraction(
        15800000000000001,
        10**22,
    )


def test_effective_amm_fee_derives_discounted_units_from_trading_fee() -> None:
    fit = _load_fit_module()
    fee_units = fit._effective_amm_fee(
        fee_row={
            "trading_fee": 0.00158,
            "discounted_fee": 0.00015800000000000002,
            "trading_fee_value": 0.0015,
            "trading_fee_currency": "XRP",
        },
        amm_in_currency="XRP",
        amm_in_value_raw="10",
    )

    assert fee_units == 15


def test_fit_bucket_classifies_structure_matched_amount_unresolved() -> None:
    fit = _load_fit_module()
    model = {
        "legs": [
            {
                "src": "CLOB",
                "source_id": "OFFER1",
                "out": {
                    "kind": "IOU",
                    "mantissa": 6506040167199990,
                    "exponent": -16,
                    "value": "0.6506040167199990",
                },
                "in": {"kind": "XRP", "drops": 351752, "value": "0.351752"},
            }
        ]
    }
    real = {
        "legs": [
            {
                "src": "CLOB",
                "offer_id": "OFFER1",
                "out": _state_report(
                    fit,
                    currency=RUSD,
                    pre=fit._decimal_to_iou_amount(Decimal("1.6506040167200000")),
                    post=fit._decimal_to_iou_amount(Decimal("1.0000000000000000")),
                    state_change=fit.STATE_CHANGE_DECREASE,
                ),
                "in": _state_report(
                    fit,
                    currency=XRP,
                    pre=XRPAmount(351_752),
                    post=XRPAmount(0),
                    state_change=fit.STATE_CHANGE_DECREASE,
                ),
            }
        ]
    }

    comparison = fit._compare_reports(model=model, real=real)
    bucket = fit._classify_fit_bucket(model=model, real=real, comparison=comparison)

    assert comparison["structure_match"] is True
    assert bucket == fit.FIT_BUCKET_STRICT_FAILED_BUT_STRUCTURE_MATCHED


def test_fit_bucket_classifies_path_kind_matched_offer_mismatch() -> None:
    fit = _load_fit_module()
    model = {
        "legs": [
            {
                "src": "CLOB",
                "source_id": "MODEL1",
                "out": {"kind": "XRP", "drops": 100, "value": "0.0001"},
                "in": {
                    "kind": "IOU",
                    "mantissa": 1000000000000000,
                    "exponent": -14,
                    "value": "10",
                },
            }
        ]
    }
    real = {
        "legs": [
            {
                "src": "CLOB",
                "offer_id": "REAL1",
                "out": _state_report(
                    fit,
                    currency=XRP,
                    pre=XRPAmount(100),
                    post=XRPAmount(0),
                    state_change=fit.STATE_CHANGE_DECREASE,
                ),
                "in": _state_report(
                    fit,
                    currency=RUSD,
                    pre=fit._decimal_to_iou_amount(Decimal("20")),
                    post=fit._decimal_to_iou_amount(Decimal("10")),
                    state_change=fit.STATE_CHANGE_DECREASE,
                ),
            }
        ]
    }

    comparison = fit._compare_reports(model=model, real=real)
    bucket = fit._classify_fit_bucket(model=model, real=real, comparison=comparison)

    assert comparison["structure_match"] is False
    assert bucket == fit.FIT_BUCKET_STRUCTURE_MISMATCH


def test_fit_bucket_classifies_path_mismatch() -> None:
    fit = _load_fit_module()
    model = {"legs": []}
    real = {
        "legs": [
            {
                "src": "CLOB",
                "offer_id": "REAL1",
                "out": _state_report(
                    fit,
                    currency=RUSD,
                    pre=fit._decimal_to_iou_amount(Decimal("20")),
                    post=fit._decimal_to_iou_amount(Decimal("6.10894565417")),
                    state_change=fit.STATE_CHANGE_DECREASE,
                ),
                "in": _state_report(
                    fit,
                    currency=XRP,
                    pre=XRPAmount(6_921_613),
                    post=XRPAmount(0),
                    state_change=fit.STATE_CHANGE_DECREASE,
                ),
            }
        ]
    }

    comparison = fit._compare_reports(model=model, real=real)
    bucket = fit._classify_fit_bucket(model=model, real=real, comparison=comparison)

    assert comparison["structure_match"] is False
    assert bucket == fit.FIT_BUCKET_STRUCTURE_MISMATCH


def test_report_relative_error_uses_leg_exact_totals_not_summary_projection() -> None:
    fit = _load_fit_module()
    report = {
        "model": {
            "legs": [
                {
                    "out": {"kind": "XRP", "drops": 1, "value": "0.000001"},
                    "in": {
                        "kind": "IOU",
                        "mantissa": 1000000000000000,
                        "exponent": -15,
                        "value": "1",
                    },
                }
            ],
            "summary": {
                "total_out": {"kind": "XRP", "drops": 1, "value": "0.000001"},
                "total_in": {
                    "kind": "IOU",
                    "mantissa": 1000000000000000,
                    "exponent": -15,
                    "value": "1",
                },
            },
        },
        "real": {
            "legs": [
                {
                    "out": {"kind": "XRP", "drops": 1, "value": "0.000001"},
                    "in": {
                        "kind": "IOU",
                        "mantissa": 1000000000000001,
                        "exponent": -15,
                        "value": "1.000000000000001",
                    },
                }
            ],
            "summary": {
                "total_out": {"kind": "XRP", "drops": 1, "value": "0.000001"},
                "total_in": {
                    "kind": "IOU",
                    "mantissa": 1000000000000000,
                    "exponent": -15,
                    "value": "1",
                },
            },
        },
    }

    rel = fit._report_total_relative_error_fields(report)

    assert rel["total_out_rel_err"] == "0"
    assert Decimal(rel["total_in_rel_err"]) > Decimal(0)


def test_model_report_assigns_amm_account_identity_to_amm_leg() -> None:
    fit = _load_fit_module()
    quote = {
        "slices": [
            {
                "src": "AMM",
                "source_id": None,
                "out_take": fit._decimal_to_iou_amount(Decimal("387.9228342064362")),
                "in_take": XRPAmount(196_713_463),
                "avg_quality": Quality.from_amounts(
                    fit._decimal_to_iou_amount(Decimal("387.9228342064362")),
                    XRPAmount(196_713_463),
                ),
            }
        ],
        "summary": {
            "avg_quality": Quality.from_amounts(
                fit._decimal_to_iou_amount(Decimal("387.9228342064362")),
                XRPAmount(196_713_463),
            ),
            "is_partial": False,
            "fill_ratio": 1.0,
        },
    }

    report = fit._model_report_from_quote(
        quote,
        in_cur=XRP,
        out_cur=RUSD,
        amm_account="rhWTXC2m2gGGA9WozUaoMm6kLAVPb1tcS3",
    )

    assert report["legs"][0]["source_id"] == "rhWTXC2m2gGGA9WozUaoMm6kLAVPb1tcS3"


def test_offer_tick_size_lookup_uses_latest_snapshot_at_or_before_tx() -> None:
    fit = _load_fit_module()
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur=RUSD,
        out_cur=XRP,
        offer_taker_gets_amt=fit._decimal_to_iou_amount(fit.Decimal("28.992074")),
        offer_taker_gets_currency=RUSD,
        offer_taker_gets_issuer="rIssuer",
        offer_taker_pays_amt=XRPAmount(13_879_489),
        offer_taker_pays_currency=XRP,
        offer_taker_pays_issuer=None,
        offer_limit_quality=Quality.from_amounts(XRPAmount(13_879_489), fit._decimal_to_iou_amount(fit.Decimal("28.992074"))),
        offer_is_sell=True,
    )

    tick = fit._effective_offer_tick_size_for_intent(
        intent=intent,
        ledger_index=100746156,
        tick_size_snapshots=[
            (100725835, {"rIssuer": 8}),
            (100726910, {"rIssuer": 7}),
            (100800000, {"rIssuer": 6}),
        ],
    )
    rounded = apply_offer_tick_size_to_intent(intent, tick)

    assert tick == 7
    assert rounded.offer_taker_pays_amt == XRPAmount(13_879_492)


def test_offer_tick_size_lookup_respects_later_snapshot_removal() -> None:
    fit = _load_fit_module()
    intent = TxIntent(
        transaction_type="OFFERCREATE",
        flags=0,
        in_cur=RUSD,
        out_cur=XRP,
        offer_taker_gets_amt=fit._decimal_to_iou_amount(fit.Decimal("28.992074")),
        offer_taker_gets_currency=RUSD,
        offer_taker_gets_issuer="rIssuer",
        offer_taker_pays_amt=XRPAmount(13_879_489),
        offer_taker_pays_currency=XRP,
        offer_taker_pays_issuer=None,
        offer_limit_quality=Quality.from_amounts(
            XRPAmount(13_879_489),
            fit._decimal_to_iou_amount(fit.Decimal("28.992074")),
        ),
        offer_is_sell=True,
    )

    tick = fit._effective_offer_tick_size_for_intent(
        intent=intent,
        ledger_index=100746156,
        tick_size_snapshots=[
            (100725835, {"rIssuer": 8}),
            (100726910, {"rIssuer": 7}),
            (100730000, {"rIssuer": None}),
        ],
    )

    assert tick is None


def test_load_offer_tick_size_snapshots_accepts_real_file_shape(tmp_path: Path) -> None:
    fit = _load_fit_module()
    metadata_path = tmp_path / "tx_metadata_full_merged.ndjson"
    metadata_path.write_text("", encoding="utf-8")
    snapshot_path = tmp_path / "offer_tick_sizes_at_ledger_100725835.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "ledger_index": 100725835,
                "rpc_url": "https://s1.ripple.com:51234/",
                "tick_sizes": {
                    "rIssuerA": 8,
                    "rIssuerB": 7,
                },
            }
        ),
        encoding="utf-8",
    )

    snapshots = fit._load_offer_tick_size_snapshots(metadata_path)

    assert snapshots == [(100725835, {"rIssuerA": 8, "rIssuerB": 7})]


def test_load_offer_tick_size_snapshots_accepts_explicit_null_removal(tmp_path: Path) -> None:
    fit = _load_fit_module()
    metadata_path = tmp_path / "tx_metadata_full_merged.ndjson"
    metadata_path.write_text("", encoding="utf-8")
    snapshot_path = tmp_path / "offer_tick_sizes_at_ledger_100725835.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "ledger_index": 100725835,
                "rpc_url": "https://s1.ripple.com:51234/",
                "tick_sizes": {
                    "rIssuerA": 8,
                    "rIssuerB": None,
                },
            }
        ),
        encoding="utf-8",
    )

    snapshots = fit._load_offer_tick_size_snapshots(metadata_path)

    assert snapshots == [(100725835, {"rIssuerA": 8, "rIssuerB": None})]
