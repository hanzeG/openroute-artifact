#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print shell exports for one resolved paper pair.")
    parser.add_argument("--pair-config", required=True)
    parser.add_argument("--pair", required=True)
    return parser.parse_args()


def _find_pair(payload: dict[str, Any], pair: str) -> dict[str, Any]:
    for row in payload.get("pairs") or []:
        if str(row.get("pair")) == pair:
            return row
    raise RuntimeError(f"pair not found in resolved config: {pair}")


def _emit(name: str, value: Any) -> None:
    print(f"export {name}={shlex.quote(str(value))}")


def main() -> None:
    args = _parse_args()
    payload = json.loads(Path(args.pair_config).read_text(encoding="utf-8"))
    row = _find_pair(payload, args.pair)
    _emit("PAIR", row["pair"])
    _emit("PAIR_LABEL", row["pair_label"])
    _emit("BASE_LABEL", row["base_label"])
    _emit("CURRENCY_HEX", row["base_currency"])
    _emit("ISSUER", row["base_issuer"])
    _emit("COUNTER_CURRENCY", row["counter_currency"])
    _emit("COUNTER_ISSUER", row["counter_issuer"])
    _emit("LEDGER_START", row["ledger_start"])
    _emit("LEDGER_END", row["ledger_end"])
    _emit("WINDOW", row["window"])


if __name__ == "__main__":
    main()
