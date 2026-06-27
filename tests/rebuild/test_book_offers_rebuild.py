from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from xrpl_router import IOUAmount, Quality, XRPAmount
from xrpl_router.book_offers import BookOfferSegment, build_clob_segments_from_offers
from xrpl_router.book_step import _limit_segment_out
from xrpl_router.tx_intent import TxIntent

RUSD = "524C555344000000000000000000000000000000"
XRP = "XRP"
ISSUER = "rIssuer"

_EMPIRICAL_FIT_SINGLE_DIRECT = (
    Path(__file__).resolve().parents[2] / "empirical/scripts/empirical_fit_single_direct.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "empirical_fit_single_direct_test_module",
    _EMPIRICAL_FIT_SINGLE_DIRECT,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("failed to load empirical_fit_single_direct.py for tests")
_EMPIRICAL_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _EMPIRICAL_MODULE
_SPEC.loader.exec_module(_EMPIRICAL_MODULE)
_snapshot_offer_to_book_offer = _EMPIRICAL_MODULE._snapshot_offer_to_book_offer
_apply_offercreate_cancel_sequence = _EMPIRICAL_MODULE._apply_offercreate_cancel_sequence
_materialize_snapshot_book_offers = _EMPIRICAL_MODULE._materialize_snapshot_book_offers
_build_external_funded_caps_for_snapshot = _EMPIRICAL_MODULE._build_external_funded_caps_for_snapshot
_accounts_with_explicit_snapshot_funding = _EMPIRICAL_MODULE._accounts_with_explicit_snapshot_funding
_materialize_snapshot_book_offers = _EMPIRICAL_MODULE._materialize_snapshot_book_offers


def _decimal_to_iou_amount(value: str) -> IOUAmount:
    dec = Decimal(str(value))
    sign, digits, exponent = dec.as_tuple()
    mantissa = 0 if not digits else int("".join(str(d) for d in digits))
    if sign:
        mantissa = -mantissa
    return IOUAmount(mantissa, exponent)


def _parse_amount(raw: object) -> XRPAmount | IOUAmount:
    if isinstance(raw, str):
        return XRPAmount(int(raw))
    assert isinstance(raw, dict)
    return _decimal_to_iou_amount(str(raw["value"]))


def _book_directory_for_offer(offer: dict[str, object]) -> str:
    quality = Quality.from_amounts(
        _parse_amount(offer["TakerGets"]),
        _parse_amount(offer["TakerPays"]),
    )
    return ("0" * 48) + f"{quality.value:016X}"


def _with_book_directories(offers: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for offer in offers:
        patched = dict(offer)
        if patched.get("BookDirectory") is None and patched.get("book_directory") is None:
            patched["BookDirectory"] = _book_directory_for_offer(patched)
        out.append(patched)
    return out


def _snapshot_book_directory_for_offer(offer: dict[str, object]) -> str:
    return _book_directory_for_offer(
        {
            "TakerGets": offer["taker_gets"],
            "TakerPays": offer["taker_pays"],
        }
    )


def _build_segments(
    offers: list[dict[str, object]],
    *,
    in_cur: str,
    out_cur: str,
    exclude_account: str | None = None,
    iou_in_transfer_rate: Decimal = Decimal("1"),
    iou_out_transfer_rate: Decimal = Decimal("1"),
) -> list[BookOfferSegment]:
    return build_clob_segments_from_offers(
        _with_book_directories(offers),
        in_cur=in_cur,
        out_cur=out_cur,
        exclude_account=exclude_account,
        iou_in_transfer_rate=iou_in_transfer_rate,
        iou_out_transfer_rate=iou_out_transfer_rate,
    )


def test_book_offer_builder_returns_strict_adapter_rows() -> None:
    offers = [
        {
            "index": "FULLY_FUNDED",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
            "TakerPays": "5000000",
            "quality": "0.5",
        }
    ]

    rows = _build_segments(offers, in_cur=XRP, out_cur=RUSD)

    assert len(rows) == 1
    assert isinstance(rows[0], BookOfferSegment)
    assert rows[0].offer_id == "FULLY_FUNDED"
    assert rows[0].quoted_quality == Quality.from_amounts(
        IOUAmount(1000000000000000, -14),
        XRPAmount(5_000_000),
    )
    assert rows[0].quality == rows[0].quoted_quality
    assert rows[0].out_max == IOUAmount(1000000000000000, -14)
    assert rows[0].in_at_out_max == XRPAmount(5_000_000)
    assert rows[0].owner == "rMaker"
    assert rows[0].owner_funds is None


def test_book_offer_builder_partial_funding_preserves_authoritative_offer_quality() -> None:
    offers = [
        {
            "index": "0D83B085A467D5E7FDED66C96E5C25788B3CAEF6F92DD8BF8FA2163E5403F1A3",
            "Account": "rpiFwLYi6Gb1ESHYorn2QG1WU5vw2u4exQ",
            "Sequence": 110501058,
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "1911.000059560127"},
            "TakerPays": "933680040",
            "owner_funds": "1207.921052746059",
            "quality": "0.4885819",
        }
    ]

    rows = _build_segments(offers, in_cur=XRP, out_cur=RUSD)

    assert len(rows) == 1
    assert rows[0].out_max == IOUAmount(1207921052746059, -12)
    assert rows[0].in_at_out_max == XRPAmount(590_168_363)
    assert rows[0].quality == rows[0].quoted_quality
    assert rows[0].owner_funds == IOUAmount(1207921052746059, -12)


def test_book_offer_builder_owner_funds_cap_preserves_strict_drop_floor_on_xrp_input() -> None:
    offers = [
        {
            "index": "75AC3CB7B414B68761F4A255AADEA8A4BD3AF5E0A8410D6E4E709A8E19F8A5DB",
            "Account": "rUiiCirAGyGfEhFFyt24cQ6ATJpx53JaqB",
            "Sequence": 100684586,
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "87.53507414032832"},
            "TakerPays": "45130026",
            "owner_funds": "0.05995548792832",
            "book_directory": "CFEC8953D22B1B50F20F46ECBCD51990A26BDB71951ED8265A125109744AF0E4",
            "quality": "515565.0628415716",
        }
    ]

    rows = _build_segments(offers, in_cur=XRP, out_cur=RUSD)

    assert len(rows) == 1
    assert rows[0].out_max == IOUAmount(5995548792832000, -17)
    assert rows[0].in_at_out_max == XRPAmount(30_910)


def test_book_offer_builder_owner_funds_transfer_rate_uses_rippled_mul_ratio() -> None:
    offers = [
        {
            "index": "E0E44FC4EF0913DDF93F15C27DF6E5EC94F56A99436F795D23CD63ED5D5F038F",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "3372.663632179002"},
            "TakerPays": "290049072",
            "owner_funds": "3372.643629853395",
            "book_directory": "5C8970D155D65DB8FF49B291D7EFFA4A09F9E8A68D9974B2591E8DA7886B4814",
        }
    ]

    rows = _build_segments(
        offers,
        in_cur=XRP,
        out_cur=RUSD,
        iou_out_transfer_rate=Decimal("1.0001"),
    )

    assert len(rows) == 1
    assert rows[0].out_max == IOUAmount(3372306399213474, -12)
    assert rows[0].in_at_out_max == XRPAmount(290_018_349)


def test_book_offer_builder_prefers_exact_book_directory_quality_over_decimal_snapshot_ratio() -> None:
    offers = [
        {
            "index": "C23BECB5B7A46FB19E779A106EF0ABC1787E08FBF4470B7BB94173200733A9B3",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "25.64325"},
            "TakerPays": "13500000",
            "book_directory": "CFEC8953D22B1B50F20F46ECBCD51990A26BDB71951ED8265A12B413015A5A2A",
            "quality": "0.5264543300868649644643327191",
        }
    ]

    rows = _build_segments(offers, in_cur=XRP, out_cur=RUSD)

    assert len(rows) == 1
    assert rows[0].quoted_quality == Quality(int("5A12B413015A5A2A", 16))
    assert rows[0].quoted_quality.rate() == IOUAmount(5264543300868650, -10)
    assert rows[0].quality == rows[0].quoted_quality


def test_book_offer_builder_uses_original_book_directory_quality_for_partial_fill_math() -> None:
    offers = [
        {
            "Account": "rpiFwLYi6Gb1ESHYorn2QG1WU5vw2u4exQ",
            "Sequence": 110501020,
            "TakerGets": "923887635",
            "TakerPays": {
                "currency": RUSD,
                "issuer": "rMxCKbEDwqr76QuheSUMdEGf4B9xJ8m5De",
                "value": "1894.838105600513",
            },
            "BookDirectory": "54123B870896F4964184D7042DB75D78913D3E377BFC4ECA4F074951AE2027A0",
            "quality": "0.000002050939999496096",
        }
    ]

    rows = _build_segments(offers, in_cur=RUSD, out_cur=XRP)

    assert len(rows) == 1
    row = rows[0]
    assert row.quoted_quality.rate() == IOUAmount(2050939999496096, -21)
    assert row.quality == row.quoted_quality

    limited = _limit_segment_out(row.segment, XRPAmount(9_075_992), round_up=True)

    assert limited.in_at_out_max == IOUAmount(1861431502790658, -14)


def test_book_offer_builder_sorts_best_quality_first_and_keeps_input_order_on_ties() -> None:
    offers = [
        {
            "index": "AAAA",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
            "TakerPays": "5000000",
            "quality": "0.5",
        },
        {
            "index": "BBBB",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "8"},
            "TakerPays": "4000000",
            "quality": "0.5",
        },
        {
            "index": "CCCC",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "9"},
            "TakerPays": "5000000",
            "quality": "0.5555555555555556",
        },
    ]

    rows = _build_segments(offers, in_cur=XRP, out_cur=RUSD)

    assert [row.offer_id for row in rows] == ["AAAA", "BBBB", "CCCC"]


def test_book_offer_builder_orders_best_quality_first() -> None:
    offers = [
        {
            "index": "WORSE",
            "TakerGets": "1000000",
            "TakerPays": {"currency": RUSD, "issuer": ISSUER, "value": "2.1"},
            "quality": "2.1",
        },
        {
            "index": "BETTER",
            "TakerGets": "1000000",
            "TakerPays": {"currency": RUSD, "issuer": ISSUER, "value": "2.0"},
            "quality": "2.0",
        },
    ]

    rows = _build_segments(offers, in_cur=RUSD, out_cur=XRP)

    assert [row.offer_id for row in rows] == ["BETTER", "WORSE"]


def test_book_offer_builder_propagates_sparse_owner_funds_across_same_account() -> None:
    offers = [
        {
            "index": "FIRST",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "20"},
            "TakerPays": "10000000",
            "quality": "0.5",
        },
        {
            "index": "SECOND",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "14"},
            "TakerPays": "7000000",
            "owner_funds": "13.5",
            "quality": "0.5",
        },
    ]

    rows = _build_segments(offers, in_cur=XRP, out_cur=RUSD)

    assert [row.offer_id for row in rows] == ["FIRST"]
    assert rows[0].out_max == IOUAmount(1350000000000000, -14)
    assert rows[0].in_at_out_max == XRPAmount(6_750_000)
    assert rows[0].owner_funds == IOUAmount(1350000000000000, -14)


def test_book_offer_builder_allocates_shared_owner_funds_in_quality_order() -> None:
    offers = [
        {
            "index": "WORSE_FIRST_IN_SNAPSHOT",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "20"},
            "TakerPays": "10400000",
            "owner_funds": "13.5",
            "quality": "0.52",
        },
        {
            "index": "BETTER_LATER_IN_SNAPSHOT",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "20"},
            "TakerPays": "10000000",
            "owner_funds": "13.5",
            "quality": "0.5",
        },
    ]

    rows = _build_segments(offers, in_cur=XRP, out_cur=RUSD)

    assert [row.offer_id for row in rows] == ["BETTER_LATER_IN_SNAPSHOT"]
    assert rows[0].out_max == IOUAmount(1350000000000000, -14)
    assert rows[0].in_at_out_max == XRPAmount(6_750_000)


def test_book_offer_builder_allocates_shared_owner_funds_by_owner_pays_with_output_transfer_fee() -> None:
    offers = [
        {
            "index": "FIRST",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
            "TakerPays": "5000000",
            "owner_funds": "12",
            "quality": "0.5",
        },
        {
            "index": "SECOND",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
            "TakerPays": "5000000",
            "owner_funds": "12",
            "quality": "0.5",
        },
    ]

    rows = _build_segments(
        offers,
        in_cur=XRP,
        out_cur=RUSD,
        iou_out_transfer_rate=Decimal("1.25"),
    )

    assert [row.offer_id for row in rows] == ["FIRST"]
    assert rows[0].out_max == IOUAmount(9600000000000000, -15)
    assert rows[0].owner_funds == IOUAmount(1200000000000000, -14)


def test_snapshot_materialization_keeps_metadata_derived_owner_funds() -> None:
    offers = [
        {
            "offer_id": "TARGET",
            "account": "rMaker",
            "taker_gets": "44144273",
            "taker_pays": {"currency": RUSD, "issuer": ISSUER, "value": "85.56165696738809"},
            "owner_funds": "30475635",
            "quality": "0.000001938227796103654",
        },
        {
            "offer_id": "OTHER",
            "account": "rMaker",
            "taker_gets": "1000",
            "taker_pays": {"currency": RUSD, "issuer": ISSUER, "value": "0.001938227796103654"},
            "owner_funds": "30475635",
            "quality": "0.000001938227796103654",
        },
    ]

    rows = _materialize_snapshot_book_offers(
        side_payload={"offers": offers},
        derived_owner_funds_by_offer_id={"TARGET": "30675623"},
    )

    target = next(row for row in rows if row["index"] == "TARGET")
    other = next(row for row in rows if row["index"] == "OTHER")
    assert target["owner_funds"] == "30675623"
    assert other["owner_funds"] == "30675623"


def test_book_offer_builder_rejects_conflicting_owner_funds_for_same_account() -> None:
    offers = [
        {
            "index": "FIRST",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "20"},
            "TakerPays": "10000000",
            "owner_funds": "13.5",
            "quality": "0.5",
        },
        {
            "index": "SECOND",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "14"},
            "TakerPays": "7000000",
            "owner_funds": "12.0",
            "quality": "0.5",
        },
    ]

    with pytest.raises(RuntimeError, match="ambiguous owner_funds"):
        build_clob_segments_from_offers(
            _with_book_directories(offers),
            in_cur=XRP,
            out_cur=RUSD,
        )


def test_book_offer_builder_requires_book_directory_on_direction_aligned_offer() -> None:
    offers = [
        {
            "index": "NOQUALITY",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
            "TakerPays": "5000000",
        }
    ]

    with pytest.raises(RuntimeError, match="offer\\.BookDirectory"):
        build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD)


def test_book_offer_builder_excludes_account_and_requires_account_when_filtering() -> None:
    offers = [
        {
            "index": "SELF",
            "Account": "rSelf",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
            "TakerPays": "5000000",
            "quality": "0.5",
        },
        {
            "index": "OTHER",
            "Account": "rOther",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "8"},
            "TakerPays": "4000000",
            "quality": "0.5",
        },
    ]

    rows = _build_segments(
        offers,
        in_cur=XRP,
        out_cur=RUSD,
        exclude_account="rSelf",
    )

    assert [row.offer_id for row in rows] == ["OTHER"]

    with pytest.raises(RuntimeError, match="missing offer.Account"):
        build_clob_segments_from_offers(
            _with_book_directories(
                [
                    {
                        "index": "NO_ACCOUNT",
                        "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                        "TakerPays": "5000000",
                        "quality": "0.5",
                    }
                ]
            ),
            in_cur=XRP,
            out_cur=RUSD,
            exclude_account="rSelf",
        )


def test_book_offer_builder_applies_input_side_iou_transfer_rate_to_segment_amounts() -> None:
    offers = [
        {
            "index": "IOU_INPUT",
            "TakerGets": "1000000",
            "TakerPays": {"currency": RUSD, "issuer": ISSUER, "value": "2"},
            "quality": "2",
        }
    ]

    rows = _build_segments(
        offers,
        in_cur=RUSD,
        out_cur=XRP,
        iou_in_transfer_rate=Decimal("1.25"),
    )

    assert len(rows) == 1
    assert rows[0].quoted_quality == Quality.from_amounts(
        XRPAmount(1_000_000),
        IOUAmount(2_000_000_000_000_000, -15),
    )
    assert rows[0].out_max == XRPAmount(1_000_000)
    assert rows[0].in_at_out_max == IOUAmount(2_000_000_000_000_000, -15)
    assert rows[0].quality == rows[0].quoted_quality
    assert rows[0].in_transfer_rate == IOUAmount(1250000000000000, -15)


def test_book_offer_builder_rejects_taker_pays_funded_without_matching_output_cap() -> None:
    offers = [
        {
            "index": "BAD_FUNDED",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
            "TakerPays": "5000000",
            "taker_pays_funded": "1000000",
            "quality": "0.5",
        }
    ]

    with pytest.raises(RuntimeError, match="taker_pays_funded without taker_gets_funded"):
        build_clob_segments_from_offers(
            _with_book_directories(offers),
            in_cur=XRP,
            out_cur=RUSD,
        )


def test_book_offer_builder_accepts_snapshot_funded_pair_with_one_drop_input_gap() -> None:
    offers = [
        {
            "index": "FUNDED_PAIR_ONE_DROP",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "15362.1206426"},
            "TakerPays": "8303848996",
            "taker_gets_funded": {"currency": RUSD, "issuer": ISSUER, "value": "3890.00000079826"},
            "taker_pays_funded": "2102702703",
            "quality": "0.5405405405405405405405405405",
        }
    ]

    rows = _build_segments(offers, in_cur=XRP, out_cur=RUSD)

    assert len(rows) == 1
    assert rows[0].out_max == IOUAmount(3890000000798260, -12)
    assert rows[0].in_at_out_max == XRPAmount(2_102_702_703)


def test_single_direct_snapshot_adapter_preserves_scalar_owner_funds() -> None:
    mapped = _snapshot_offer_to_book_offer(
        {
            "offer_id": "SNAPSHOT_OWNER_FUNDS",
            "account": "rMaker",
            "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "162.226008584139"},
            "taker_pays": "84027962",
            "book_directory": _snapshot_book_directory_for_offer(
                {
                    "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "162.226008584139"},
                    "taker_pays": "84027962",
                }
            ),
            "owner_funds": "81.12111281206936",
        }
    )

    assert mapped is not None
    assert mapped["BookDirectory"] is not None
    rows = _build_segments([mapped], in_cur=XRP, out_cur=RUSD)
    assert len(rows) == 1
    assert rows[0].owner_funds == IOUAmount(8112111281206936, -14)
    assert rows[0].out_max == IOUAmount(8112111281206936, -14)
    assert rows[0].in_at_out_max == XRPAmount(42_018_180)


def test_single_direct_snapshot_adapter_preserves_zero_owner_funds_and_builder_skips_unfunded_offer() -> None:
    mapped = _snapshot_offer_to_book_offer(
        {
            "offer_id": "SNAPSHOT_ZERO_OWNER_FUNDS",
            "account": "rMaker",
            "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "1915.1"},
            "taker_pays": "1000000000",
            "book_directory": _snapshot_book_directory_for_offer(
                {
                    "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "1915.1"},
                    "taker_pays": "1000000000",
                }
            ),
            "owner_funds": "0",
        }
    )

    assert mapped is not None
    assert mapped["owner_funds"] == "0"
    rows = _build_segments([mapped], in_cur=XRP, out_cur=RUSD)
    assert rows == []


def test_single_direct_snapshot_adapter_drops_zero_taker_gets_funded() -> None:
    mapped = _snapshot_offer_to_book_offer(
        {
            "offer_id": "ZERO_FUNDED",
            "account": "rMaker",
            "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
            "taker_pays": "5000000",
            "book_directory": _snapshot_book_directory_for_offer(
                {
                    "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                    "taker_pays": "5000000",
                }
            ),
            "taker_gets_funded": {"currency": RUSD, "issuer": ISSUER, "value": "0"},
        }
    )

    assert mapped is None


def test_external_account_offers_funded_caps_reserve_liquidity_for_earlier_cross_book_offers() -> None:
    funded_caps_by_offer_id, overridden_accounts = _build_external_funded_caps_for_snapshot(
        side_payload={
            "offers": [
                {
                    "offer_id": "BOOK_A",
                    "account": "rMaker",
                    "sequence": 30,
                    "owner_funds": "12",
                    "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                    "taker_pays": "5000000",
                },
                {
                    "offer_id": "BOOK_B",
                    "account": "rMaker",
                    "sequence": 50,
                    "owner_funds": "12",
                    "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                    "taker_pays": "5000000",
                },
            ]
        },
        ledger_index=123,
        account_offers_snapshots={
            (123, "rMaker"): {
                "ledger_index": 123,
                "account": "rMaker",
                "offers": [
                    {
                        "seq": 10,
                        "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "5"},
                        "taker_pays": "1",
                    },
                    {
                        "seq": 30,
                        "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                        "taker_pays": "2",
                    },
                    {
                        "seq": 40,
                        "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "4"},
                        "taker_pays": "3",
                    },
                    {
                        "seq": 50,
                        "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                        "taker_pays": "4",
                    },
                ],
            }
        },
    )

    assert overridden_accounts == {"rMaker"}
    assert funded_caps_by_offer_id == {
        "BOOK_A": {"currency": RUSD, "issuer": ISSUER, "value": "7"},
        "BOOK_B": {"currency": RUSD, "issuer": ISSUER, "value": "0"},
    }


def test_external_account_offers_funded_caps_use_prebook_ledger_snapshot_key() -> None:
    funded_caps_by_offer_id, overridden_accounts = _build_external_funded_caps_for_snapshot(
        side_payload={
            "offers": [
                {
                    "offer_id": "BOOK_A",
                    "account": "rMaker",
                    "sequence": 30,
                    "owner_funds": "12",
                    "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                    "taker_pays": "5000000",
                }
            ]
        },
        ledger_index=122,
        account_offers_snapshots={
            (123, "rMaker"): {
                "ledger_index": 123,
                "account": "rMaker",
                "offers": [
                    {
                        "seq": 30,
                        "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "1"},
                        "taker_pays": "1",
                    }
                ],
            },
            (122, "rMaker"): {
                "ledger_index": 122,
                "account": "rMaker",
                "offers": [
                    {
                        "seq": 30,
                        "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                        "taker_pays": "2",
                    }
                ],
            },
        },
    )

    assert overridden_accounts == {"rMaker"}
    assert funded_caps_by_offer_id == {
        "BOOK_A": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
    }


def test_external_account_offers_funded_caps_skip_when_account_offers_amount_exceeds_snapshot_amount() -> None:
    funded_caps_by_offer_id, overridden_accounts = _build_external_funded_caps_for_snapshot(
        side_payload={
            "offers": [
                {
                    "offer_id": "BOOK_A",
                    "account": "rMaker",
                    "sequence": 30,
                    "owner_funds": "12",
                    "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "7"},
                    "taker_pays": "3500000",
                }
            ]
        },
        ledger_index=123,
        account_offers_snapshots={
            (123, "rMaker"): {
                "ledger_index": 123,
                "account": "rMaker",
                "offers": [
                    {
                        "seq": 30,
                        "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                        "taker_pays": "5000000",
                    }
                ],
            }
        },
    )

    assert overridden_accounts == set()
    assert funded_caps_by_offer_id == {}


def test_external_account_offers_funded_caps_do_not_consume_balance_on_skipped_selected_sequence() -> None:
    funded_caps_by_offer_id, overridden_accounts = _build_external_funded_caps_for_snapshot(
        side_payload={
            "offers": [
                {
                    "offer_id": "BOOK_A",
                    "account": "rMaker",
                    "sequence": 10,
                    "owner_funds": "16",
                    "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                    "taker_pays": "5",
                },
                {
                    "offer_id": "BOOK_B",
                    "account": "rMaker",
                    "sequence": 20,
                    "owner_funds": "16",
                    "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                    "taker_pays": "6",
                },
            ]
        },
        ledger_index=123,
        account_offers_snapshots={
            (123, "rMaker"): {
                "ledger_index": 123,
                "account": "rMaker",
                "offers": [
                    {
                        "seq": 10,
                        "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "12"},
                        "taker_pays": "6",
                    },
                    {
                        "seq": 20,
                        "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                        "taker_pays": "5",
                    },
                ],
            }
        },
    )

    assert overridden_accounts == {"rMaker"}
    assert funded_caps_by_offer_id == {
        "BOOK_B": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
    }


def test_accounts_with_explicit_snapshot_funding_selects_only_accounts_with_funded_fields() -> None:
    accounts = _accounts_with_explicit_snapshot_funding(
        {
            "offers": [
                {
                    "offer_id": "FUNDED",
                    "account": "rFunded",
                    "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                    "taker_pays": "5000000",
                    "taker_gets_funded": {"currency": RUSD, "issuer": ISSUER, "value": "7"},
                },
                {
                    "offer_id": "UNFUNDED",
                    "account": "rUnfunded",
                    "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                    "taker_pays": "5000000",
                },
                {
                    "offer_id": "ZERO_FUNDED",
                    "account": "rZero",
                    "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                    "taker_pays": "5000000",
                    "taker_pays_funded": "0",
                },
            ]
        }
    )

    assert accounts == {"rFunded", "rZero"}


def test_snapshot_materialization_applies_external_funded_caps_without_shared_owner_funds() -> None:
    rows = _materialize_snapshot_book_offers(
        side_payload={
            "offers": [
                {
                    "offer_id": "BOOK_A",
                    "account": "rMaker",
                    "sequence": 30,
                    "owner_funds": "12",
                    "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                    "taker_pays": "5000000",
                    "quality": "0.5",
                },
                {
                    "offer_id": "BOOK_B",
                    "account": "rMaker",
                    "sequence": 50,
                    "owner_funds": "12",
                    "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                    "taker_pays": "5000000",
                    "quality": "0.5",
                },
            ]
        },
        derived_owner_funds_by_offer_id={},
        funded_caps_by_offer_id={
            "BOOK_A": {"currency": RUSD, "issuer": ISSUER, "value": "7"},
            "BOOK_B": {"currency": RUSD, "issuer": ISSUER, "value": "0"},
        },
        strip_owner_funds_accounts={"rMaker"},
    )

    first = next(row for row in rows if row["index"] == "BOOK_A")
    assert "owner_funds" not in first
    assert first["taker_gets_funded"] == {"currency": RUSD, "issuer": ISSUER, "value": "7"}
    assert all(row["index"] != "BOOK_B" for row in rows)

    built = _build_segments(rows, in_cur=XRP, out_cur=RUSD)
    assert len(built) == 1
    assert built[0].offer_id == "BOOK_A"
    assert built[0].out_max == IOUAmount(7000000000000000, -15)


def test_snapshot_materialization_external_caps_can_apply_without_preexisting_snapshot_funded_fields() -> None:
    side_payload = {
        "offers": [
            {
                "offer_id": "BOOK_A",
                "account": "rMaker",
                "sequence": 30,
                "owner_funds": "12",
                "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                "taker_pays": "5000000",
                "quality": "0.5",
            },
            {
                "offer_id": "BOOK_B",
                "account": "rMaker",
                "sequence": 50,
                "owner_funds": "12",
                "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                "taker_pays": "5000000",
                "quality": "0.5",
            },
        ]
    }

    funded_caps_by_offer_id, raw_overridden_accounts = _build_external_funded_caps_for_snapshot(
        side_payload=side_payload,
        ledger_index=123,
        account_offers_snapshots={
            (123, "rMaker"): {
                "ledger_index": 123,
                "account": "rMaker",
                "offers": [
                    {
                        "seq": 10,
                        "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "5"},
                        "taker_pays": "1",
                    },
                    {
                        "seq": 30,
                        "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                        "taker_pays": "2",
                    },
                    {
                        "seq": 40,
                        "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "4"},
                        "taker_pays": "3",
                    },
                    {
                        "seq": 50,
                        "taker_gets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
                        "taker_pays": "4",
                    },
                ],
            }
        },
    )
    assert raw_overridden_accounts == {"rMaker"}
    assert _accounts_with_explicit_snapshot_funding(side_payload) == set()

    rows = _materialize_snapshot_book_offers(
        side_payload=side_payload,
        derived_owner_funds_by_offer_id={},
        funded_caps_by_offer_id=funded_caps_by_offer_id,
        strip_owner_funds_accounts=raw_overridden_accounts,
    )

    first = next(row for row in rows if row["index"] == "BOOK_A")
    assert "owner_funds" not in first
    assert first["taker_gets_funded"] == {"currency": RUSD, "issuer": ISSUER, "value": "7"}
    assert all(row["index"] != "BOOK_B" for row in rows)


def test_book_offer_builder_zero_owner_funds_propagates_across_same_account_as_zero_liquidity() -> None:
    offers = [
        {
            "index": "ZERO_A",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
            "TakerPays": "5000000",
            "quality": "0.5",
        },
        {
            "index": "ZERO_B",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "1"},
            "TakerPays": "500000",
            "owner_funds": "0",
            "quality": "0.5",
        },
    ]

    rows = _build_segments(offers, in_cur=XRP, out_cur=RUSD)

    assert rows == []


def test_book_offer_builder_preserves_raw_row_and_stores_output_transfer_rate_metadata() -> None:
    offers = [
        {
            "index": "OUTPUT_FEE",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
            "TakerPays": "5000000",
            "quality": "0.5",
        }
    ]

    rows = _build_segments(
        offers,
        in_cur=XRP,
        out_cur=RUSD,
        iou_out_transfer_rate=Decimal("1.25"),
    )

    assert len(rows) == 1
    assert rows[0].out_max == IOUAmount(1000000000000000, -14)
    assert rows[0].in_at_out_max == XRPAmount(5_000_000)
    assert rows[0].out_transfer_rate == IOUAmount(1250000000000000, -15)


def test_single_direct_snapshot_adapter_applies_offercreate_cancel_sequence_before_building_book() -> None:
    intent = TxIntent(
        transaction_type="OfferCreate",
        flags=0,
        in_cur=XRP,
        out_cur=RUSD,
        source_account="rSource",
        offer_cancel_sequence=101,
    )
    offers = [
        {
            "index": "CANCELLED_SELF",
            "Account": "rSource",
            "Sequence": 101,
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "10"},
            "TakerPays": "5000000",
            "quality": "0.5",
        },
        {
            "index": "KEEP_SELF",
            "Account": "rSource",
            "Sequence": 102,
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "8"},
            "TakerPays": "4000000",
            "quality": "0.5",
        },
        {
            "index": "KEEP_OTHER",
            "Account": "rOther",
            "Sequence": 101,
            "TakerGets": {"currency": RUSD, "issuer": ISSUER, "value": "7"},
            "TakerPays": "3500000",
            "quality": "0.5",
        },
    ]

    filtered = _apply_offercreate_cancel_sequence(book_offers=offers, intent=intent)

    assert [offer["index"] for offer in filtered] == ["KEEP_SELF", "KEEP_OTHER"]
