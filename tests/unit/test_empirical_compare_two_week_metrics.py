from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "empirical"
    / "scripts"
    / "empirical_compare_two_week_metrics.py"
)
_SPEC = importlib.util.spec_from_file_location("empirical_compare_two_week_metrics", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_strict_segment_signature = _MODULE._strict_segment_signature


def test_strict_segment_signature_sorts_clob_ids_and_collapses_amm_to_one_segment() -> None:
    segs = [
        {"segment_kind": "AMM", "source_id": "pool-a"},
        {"segment_kind": "CLOB", "source_id": "offer-2"},
        {"segment_kind": "AMM", "source_id": "pool-a"},
        {"segment_kind": "CLOB", "source_id": "offer-1"},
    ]

    assert _strict_segment_signature(segs) == [
        ("AMM", None),
        ("CLOB", "offer-1"),
        ("CLOB", "offer-2"),
    ]


def test_strict_segment_signature_preserves_duplicate_clob_segments() -> None:
    segs = [
        {"segment_kind": "CLOB", "source_id": "offer-1"},
        {"segment_kind": "CLOB", "source_id": "offer-1"},
        {"segment_kind": "AMM", "source_id": "pool-a"},
    ]

    assert _strict_segment_signature(segs) == [
        ("AMM", None),
        ("CLOB", "offer-1"),
        ("CLOB", "offer-1"),
    ]
