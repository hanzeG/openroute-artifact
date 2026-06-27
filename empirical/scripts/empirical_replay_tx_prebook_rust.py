#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    manifest = root / "empirical/rust/tx_prebook_replay/Cargo.toml"
    cmd = [
        "cargo",
        "run",
        "--release",
        "--manifest-path",
        str(manifest),
        "--",
        *sys.argv[1:],
    ]
    print("[rust] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

