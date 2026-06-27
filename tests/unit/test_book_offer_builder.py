from decimal import Decimal
import pytest

from xrpl_router.core.fmt import amount_to_decimal
from xrpl_router.book_offers import build_clob_segments_from_offers

RUSD_HEX = "524C555344000000000000000000000000000000"
XRP = "XRP"


def test_builder_keeps_in_amount_without_quality_roundtrip_drift():
    # IN is exactly 4.336217 XRP (4336217 drops).
    offers = [
        {
            "index": "ABCDEF0000000000000000000000000000009075",
            "TakerGets": {"currency": RUSD_HEX, "value": "8.412263"},
            "TakerPays": "4336217",
            "owner_funds": "8.412263",
            "quality": "1.94",
        }
    ]

    segs = build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD_HEX)

    assert len(segs) == 1
    assert amount_to_decimal(segs[0].out_max) == Decimal("8.412263")
    assert amount_to_decimal(segs[0].in_at_out_max) == Decimal("4.336217")


def test_builder_stable_sort_keeps_source_id_alignment_on_ties():
    offers = [
        {
            "index": "000000000000000000000000000000000000AAAA",
            "TakerGets": {"currency": RUSD_HEX, "value": "10"},
            "TakerPays": "5000000",
            "owner_funds": "10",
            "quality": "2",
        },
        {
            "index": "000000000000000000000000000000000000BBBB",
            "TakerGets": {"currency": RUSD_HEX, "value": "8"},
            "TakerPays": "4000000",
            "owner_funds": "8",
            "quality": "2",
        },
        {
            "index": "000000000000000000000000000000000000CCCC",
            "TakerGets": {"currency": RUSD_HEX, "value": "9"},
            "TakerPays": "5000000",
            "owner_funds": "9",
            "quality": "1.8",
        },
    ]

    segs = build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD_HEX)
    ids = [str(s.source_id)[-4:] for s in segs]

    # AAAA and BBBB have equal quality; stable sort should preserve input order.
    assert ids == ["AAAA", "BBBB", "CCCC"]
    assert amount_to_decimal(segs[0].out_max) == Decimal("10")
    assert amount_to_decimal(segs[1].out_max) == Decimal("8")


def test_builder_keeps_rows_without_funding_signals():
    offers = [
        {
            "index": "NOFUNDS",
            "TakerGets": {"currency": RUSD_HEX, "value": "8.412263"},
            "TakerPays": "4336217",
            "quality": "1.94",
        }
    ]

    segs = build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD_HEX)
    assert len(segs) == 1


def test_builder_allows_partial_funded_presence_with_fallback():
    offers = [
        {
            "index": "HALFFUNDED",
            "TakerGets": {"currency": RUSD_HEX, "value": "8.412263"},
            "TakerPays": "4336217",
            "taker_gets_funded": {"currency": RUSD_HEX, "value": "8.412263"},
            "quality": "1.94",
        }
    ]

    segs = build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD_HEX)
    assert len(segs) == 1


def test_builder_owner_funds_caps_xrp_gets_amounts():
    offers = [
        {
            "index": "OWNERFUNDS_XRP_CAP",
            "TakerGets": "10000000",  # 10 XRP
            "TakerPays": {"currency": RUSD_HEX, "value": "20"},
            "owner_funds": "3000000",  # 3 XRP funded
            "quality": "2",
        }
    ]

    segs = build_clob_segments_from_offers(offers, in_cur=RUSD_HEX, out_cur=XRP)
    assert len(segs) == 1
    assert amount_to_decimal(segs[0].out_max) == Decimal("3")
    assert amount_to_decimal(segs[0].in_at_out_max) == Decimal("6")


def test_builder_owner_funds_caps_iou_gets_amounts():
    offers = [
        {
            "index": "OWNERFUNDS_IOU_CAP",
            "TakerGets": {"currency": RUSD_HEX, "value": "20"},
            "TakerPays": "10000000",  # 10 XRP
            "owner_funds": "4",  # 4 rUSD funded
            "quality": "0.5",
        }
    ]

    segs = build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD_HEX)
    assert len(segs) == 1
    assert amount_to_decimal(segs[0].out_max) == Decimal("4")
    assert amount_to_decimal(segs[0].in_at_out_max) == Decimal("2")


def test_builder_partial_funded_offer_ignores_tiny_rounding_noise_near_drop_boundary() -> None:
    offers = [
        {
            "index": "223B76C68E23D96482EEB713B61F26A1788F687DB64FA3A65547E5A5F3A3EAED",
            "Account": "rfmdBKhtJw2J22rw1JxQcchQTM68qzE4N2",
            "Sequence": 117327390,
            "TakerGets": {"currency": RUSD_HEX, "value": "9998.467046823989"},
            "TakerPays": "4813759960",
            "owner_funds": "20.770597474545",
            "quality": "2.0770597474544594865922645632",
        }
    ]

    segs = build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD_HEX)
    assert len(segs) == 1
    assert amount_to_decimal(segs[0].out_max) == Decimal("20.770597474545")
    assert amount_to_decimal(segs[0].in_at_out_max) == Decimal("10")


def test_builder_partial_funded_offer_rounds_input_up_when_materially_above_drop_boundary() -> None:
    offers = [
        {
            "index": "0D83B085A467D5E7FDED66C96E5C25788B3CAEF6F92DD8BF8FA2163E5403F1A3",
            "Account": "rpiFwLYi6Gb1ESHYorn2QG1WU5vw2u4exQ",
            "Sequence": 110501058,
            "TakerGets": {"currency": RUSD_HEX, "value": "1911.000059560127"},
            "TakerPays": "933680040",
            "owner_funds": "1207.921052746059",
            "quality": "2.0467397584724280921759878256",
        }
    ]

    segs = build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD_HEX)
    assert len(segs) == 1
    assert amount_to_decimal(segs[0].out_max) == Decimal("1207.921052746059")
    assert amount_to_decimal(segs[0].in_at_out_max) == Decimal("590.168364")


def test_builder_partial_funded_offer_rounds_up_one_drop_when_ratio_is_material() -> None:
    offers = [
        {
            "index": "63A7E711A39B13772ED84BDC68A694D5C70AF98DAFAE7391DD0F136917F83B54",
            "Account": "rE9m7LZKMSQfsgrUj9vSzhX4FsP4BUykkS",
            "Sequence": 64683659,
            "TakerGets": {"currency": RUSD_HEX, "value": "217.126414976478"},
            "TakerPays": "106223127",
            "owner_funds": "7.2875164163545",
            "quality": "0.4887275002422678434314059049",
        }
    ]

    segs = build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD_HEX)
    assert len(segs) == 1
    assert amount_to_decimal(segs[0].out_max) == Decimal("7.2875164163545")
    assert amount_to_decimal(segs[0].in_at_out_max) == Decimal("3.565217")


def test_builder_distributes_owner_funds_across_same_account_in_book_order():
    offers = [
        {
            "index": "FIRST",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD_HEX, "value": "20"},
            "TakerPays": "10000000",  # 10 XRP
            "quality": "0.5",
        },
        {
            "index": "SECOND",
            "Account": "rMaker",
            "TakerGets": {"currency": RUSD_HEX, "value": "14"},
            "TakerPays": "7000000",  # 7 XRP
            "owner_funds": "13.5",
            "quality": "0.49",
        },
    ]

    segs = build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD_HEX)
    assert [str(s.source_id) for s in segs] == ["FIRST"]
    assert amount_to_decimal(segs[0].out_max) == Decimal("13.5")
    assert amount_to_decimal(segs[0].in_at_out_max) == Decimal("6.75")


def test_builder_taker_gets_funded_scales_taker_pays_proportionally():
    offers = [
        {
            "index": "FUNDED_RATIO_CAP",
            "TakerGets": {"currency": RUSD_HEX, "value": "10"},
            "TakerPays": "5000000",  # 5 XRP
            "taker_gets_funded": {"currency": RUSD_HEX, "value": "4"},
            "quality": "0.5",
        }
    ]

    segs = build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD_HEX)
    assert len(segs) == 1
    assert amount_to_decimal(segs[0].out_max) == Decimal("4")
    assert amount_to_decimal(segs[0].in_at_out_max) == Decimal("2")


def test_builder_drops_row_when_present_funded_is_non_positive():
    offers = [
        {
            "index": "FUNDEDZERO",
            "TakerGets": {"currency": RUSD_HEX, "value": "8.412263"},
            "TakerPays": "4336217",
            "taker_gets_funded": {"currency": RUSD_HEX, "value": "0"},
            "quality": "1.94",
        }
    ]

    segs = build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD_HEX)
    assert segs == []


def test_builder_raises_when_quality_missing():
    offers = [
        {
            "index": "NOQUALITY",
            "TakerGets": {"currency": RUSD_HEX, "value": "8.412263"},
            "TakerPays": "4336217",
        }
    ]
    with pytest.raises(RuntimeError, match="quality"):
        build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD_HEX)


def test_builder_sorts_by_router_quality_from_amounts_not_raw_quality_field():
    offers = [
        {
            "index": "LOW_AMT_HIGH_RAW",
            "TakerGets": {"currency": RUSD_HEX, "value": "1"},
            "TakerPays": "1000000",
            "owner_funds": "1",
            "quality": "9",
        },
        {
            "index": "HIGH_AMT_LOW_RAW",
            "TakerGets": {"currency": RUSD_HEX, "value": "10"},
            "TakerPays": "1000000",
            "owner_funds": "10",
            "quality": "1",
        },
    ]

    segs = build_clob_segments_from_offers(offers, in_cur=XRP, out_cur=RUSD_HEX)
    ids = [str(s.source_id) for s in segs]
    assert ids == ["HIGH_AMT_LOW_RAW", "LOW_AMT_HIGH_RAW"]


def test_builder_excludes_self_account_rows_when_requested():
    offers = [
        {
            "index": "SELF",
            "Account": "rSelf",
            "TakerGets": {"currency": RUSD_HEX, "value": "10"},
            "TakerPays": "5000000",
            "quality": "2",
        },
        {
            "index": "OTHER",
            "Account": "rOther",
            "TakerGets": {"currency": RUSD_HEX, "value": "8"},
            "TakerPays": "4000000",
            "quality": "2",
        },
    ]
    segs = build_clob_segments_from_offers(
        offers,
        in_cur=XRP,
        out_cur=RUSD_HEX,
        exclude_account="rSelf",
    )
    assert [str(s.source_id) for s in segs] == ["OTHER"]


def test_builder_fails_fast_when_excluding_account_but_offer_missing_account():
    offers = [
        {
            "index": "NO_ACCOUNT",
            "TakerGets": {"currency": RUSD_HEX, "value": "10"},
            "TakerPays": "5000000",
            "quality": "2",
        }
    ]
    with pytest.raises(RuntimeError, match="missing offer.Account"):
        build_clob_segments_from_offers(
            offers,
            in_cur=XRP,
            out_cur=RUSD_HEX,
            exclude_account="rSelf",
        )
