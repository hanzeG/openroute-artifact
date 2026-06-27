#!/usr/bin/env python3
"""Quick smoke test for a Delta Sharing profile/table."""

from __future__ import annotations

import argparse
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test Delta Sharing profile access.")
    p.add_argument(
        "--profile",
        default=os.environ.get("XRPL_SHARE_PROFILE", "data/config.share"),
        help="Path to .share profile",
    )
    p.add_argument(
        "--table",
        default="ripple-ubri-share.ripplex.offers_fact_tx",
        help="Share.schema.table name",
    )
    p.add_argument("--limit", type=int, default=1, help="Rows to fetch (default: 1)")
    return p.parse_args()


def check_profile_expiry(profile_path: Path) -> None:
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] Could not parse profile JSON: {exc}")
        return

    exp_raw = payload.get("expirationTime")
    if not exp_raw:
        print("[info] Profile has no expirationTime field.")
        return

    try:
        exp = datetime.fromisoformat(exp_raw.replace("Z", "+00:00"))
    except Exception:
        print(f"[warn] Unparseable expirationTime: {exp_raw}")
        return

    now = datetime.now(timezone.utc)
    status = "EXPIRED" if exp <= now else "VALID"
    print(f"[info] expirationTime: {exp.isoformat()} (UTC) -> {status}")


def check_profile_permissions(profile_path: Path) -> None:
    mode = stat.S_IMODE(profile_path.stat().st_mode)
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        print(
            f"[warn] {profile_path} is group/other-readable "
            f"(mode {oct(mode)}). Recommended: chmod 600 {profile_path}"
        )


def main() -> int:
    args = parse_args()
    profile_path = Path(args.profile)
    if not profile_path.exists():
        print(f"[error] Profile not found: {profile_path}")
        return 2

    print(f"[info] profile: {profile_path}")
    check_profile_expiry(profile_path)
    check_profile_permissions(profile_path)

    table_url = f"{profile_path}#{args.table}"
    print(f"[info] testing table read: {table_url}")

    try:
        import delta_sharing
    except Exception as exc:
        print(f"[error] delta_sharing import failed: {exc}")
        return 3

    try:
        df = delta_sharing.load_as_pandas(table_url, limit=args.limit)
        print(f"[ok] Read succeeded. rows={len(df)}")
        if len(df) > 0:
            print(f"[ok] columns={list(df.columns)}")
        return 0
    except Exception as exc:
        print(f"[fail] Read failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
