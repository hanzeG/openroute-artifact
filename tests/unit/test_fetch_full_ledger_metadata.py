from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    mod_path = (
        Path(__file__)
        .resolve()
        .parents[2]
        / "empirical/scripts/empirical_fetch_full_ledger_metadata.py"
    )
    spec = importlib.util.spec_from_file_location("empirical_fetch_full_ledger_metadata", mod_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_ledger_result_sorts_and_normalizes_meta() -> None:
    mod = _load_module()
    rows = mod.normalize_ledger_result(
        {
            "ledger": {
                "ledger_index": 123,
                "transactions": [
                    {
                        "hash": "bb",
                        "metaData": {"TransactionIndex": 9},
                        "TransactionType": "Payment",
                    },
                    {
                        "hash": "aa",
                        "meta": {"TransactionIndex": 3},
                        "TransactionType": "OfferCreate",
                    },
                ],
            },
            "validated": True,
        }
    )

    assert [r["tx_hash"] for r in rows] == ["AA", "BB"]
    assert rows[0]["result"]["ledger_index"] == 123
    assert rows[0]["result"]["inLedger"] == 123
    assert rows[0]["result"]["validated"] is True
    assert rows[1]["result"]["meta"]["TransactionIndex"] == 9
    assert "metaData" not in rows[1]["result"]


def test_normalize_ledger_result_skips_rows_without_hash() -> None:
    mod = _load_module()
    rows = mod.normalize_ledger_result(
        {
            "transactions": [
                {"meta": {"TransactionIndex": 1}},
                {"hash": "cc", "meta": {"TransactionIndex": 2}},
            ],
            "ledger_index": 456,
            "validated": False,
        }
    )

    assert len(rows) == 1
    assert rows[0]["tx_hash"] == "CC"
    assert rows[0]["result"]["ledger_index"] == 456


def test_load_required_hashes_by_ledger_groups_hashes(tmp_path) -> None:
    mod = _load_module()
    csv_path = tmp_path / "required.csv"
    csv_path.write_text(
        "ledger_index,transaction_hash\n"
        "100,A1\n"
        "100,a2\n"
        "101,B1\n",
        encoding="utf-8",
    )

    out = mod.load_required_hashes_by_ledger(csv_path)

    assert out == {
        100: {"A1", "A2"},
        101: {"B1"},
    }
