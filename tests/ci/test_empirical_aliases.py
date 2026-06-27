from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_run_empirical_module():
    path = Path("apps/run_empirical.py").resolve()
    spec = importlib.util.spec_from_file_location("run_empirical_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_aliases_map_to_canonical_aliases() -> None:
    module = _load_run_empirical_module()
    for legacy, canonical in module.LEGACY_ALIAS_MAP.items():
        assert canonical in module.SCRIPT_MAP, f"{legacy} -> {canonical} must exist in SCRIPT_MAP"


def test_canonical_alias_prefixes_are_consistent() -> None:
    module = _load_run_empirical_module()
    allowed_prefixes = ("delta-sharing-", "empirical-")
    for alias in module.SCRIPT_MAP:
        assert alias.startswith(allowed_prefixes), f"non-unified alias name: {alias}"
