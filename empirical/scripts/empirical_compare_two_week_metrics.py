#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

getcontext().prec = 40

RUSD_HEX = "524C555344000000000000000000000000000000"
XRP = "XRP"
XRP_QUANTUM = Decimal("0.000001")
REAL_METRICS_CACHE_VERSION = 3


@dataclass
class ModelMetrics:
    model_in: Decimal
    model_out: Decimal
    model_amm_in: Decimal
    model_amm_out: Decimal
    model_clob_in: Decimal
    model_clob_out: Decimal
    model_segment_count: int
    model_clob_segment_count: int
    model_amm_segment_count_premerge: int
    model_amm_segment_count_postmerge: int
    model_offer_ids: List[str]
    model_offer_unique: bool


@dataclass
class RealMetrics:
    real_in: Decimal
    real_out: Decimal
    real_amm_in: Decimal
    real_amm_out: Decimal
    real_clob_in: Decimal
    real_clob_out: Decimal
    real_segment_count: int
    real_clob_segment_count: int
    real_amm_segment_count: int
    real_offer_ids: List[str]
    real_offer_unique: bool


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run rolling compare for a two-week ledger window and persist notebook-aligned metrics at "
            "tx / ledger / full-window levels."
        )
    )
    p.add_argument("--pair", default="rlusd_xrp")
    p.add_argument("--ledger-start", type=int, required=True)
    p.add_argument("--ledger-end", type=int, required=True)

    p.add_argument("--amm-swaps", required=True, help="amm_swaps parquet path")
    p.add_argument("--amm-fees", required=True, help="amm_fees parquet path")
    p.add_argument("--clob-legs", required=True, help="clob_legs_with_idx parquet path")

    p.add_argument(
        "--tx-prebook-snapshots",
        required=True,
        help="Required tx-level prebook snapshots ndjson (from replay tool).",
    )

    p.add_argument(
        "--metadata-ndjson",
        required=True,
        help="tx_metadata_full_merged.ndjson path, used to reconstruct real trajectory metrics",
    )
    p.add_argument(
        "--real-metrics-cache",
        default=None,
        help=(
            "Optional cache NDJSON for metadata-derived real tx metrics. "
            "If omitted, defaults to <metadata_dir>/real_metrics_cache_<pair>_<start>_<end>_v1.ndjson"
        ),
    )

    p.add_argument(
        "--rolling-output-dir",
        default=None,
        help="Directory for rolling compare output (default: artifacts/compare/<pair>/ledger_<start>_<end>)",
    )
    p.add_argument(
        "--rolling-output-name",
        default=None,
        help="Rolling parquet filename (default: compare_rolling_<pair>_ledger_<start>_<end>_v1.parquet)",
    )
    p.add_argument(
        "--clob-roll-intra-ledger",
        type=int,
        default=1,
        help="Pass-through to empirical_compare_rolling.py: 1=roll CLOB tiers tx-by-tx within ledger",
    )
    p.add_argument(
        "--state-mode",
        choices=["ledger_local", "window_linked"],
        default="ledger_local",
        help=(
            "Pass-through to empirical_compare_rolling.py: "
            "ledger_local resets model state at ledger boundaries; "
            "window_linked rolls model state continuously across ledgers"
        ),
    )
    p.add_argument(
        "--amm-roll-mode",
        choices=["model_linked", "real_pre_tx"],
        default="model_linked",
        help=(
            "Pass-through to empirical_compare_rolling.py: "
            "model_linked advances AMM with model execution for pair-direct tx; "
            "real_pre_tx re-anchors each tx to the real pre-tx AMM state"
        ),
    )
    p.add_argument(
        "--force-rerun-rolling",
        action="store_true",
        help="Always rerun rolling compare even if rolling parquet + slices already exist",
    )
    p.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write manifest sidecar files (manifest.json / metrics_manifest.json)",
    )

    p.add_argument(
        "--output-dir",
        default=None,
        help="Metrics output dir (default: same as rolling output dir)",
    )
    p.add_argument(
        "--output-ndjson-name",
        default=None,
        help="Unified NDJSON output filename (default: metrics_<pair>_ledger_<start>_<end>_v1.ndjson)",
    )
    p.add_argument(
        "--tx-metrics-name",
        default=None,
        help="Tx metrics parquet name (only used with --also-write-parquet)",
    )
    p.add_argument(
        "--ledger-metrics-name",
        default=None,
        help="Ledger metrics parquet name (only used with --also-write-parquet)",
    )
    p.add_argument(
        "--window-metrics-name",
        default=None,
        help="Window metrics json name (only used with --also-write-parquet)",
    )
    p.add_argument(
        "--also-write-parquet",
        action="store_true",
        help="Also write parquet/json side outputs in addition to the unified NDJSON",
    )
    p.add_argument(
        "--also-write-csv",
        action="store_true",
        help="Also write CSV sidecars for tx/ledger metrics (implies --also-write-parquet)",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Backward-compatible flag; disables CSV sidecars",
    )
    return p.parse_args()


def _d(v: Any, default: Decimal = Decimal("0")) -> Decimal:
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    try:
        return Decimal(str(v))
    except (InvalidOperation, ValueError, TypeError):
        return default


def _d_opt(v: Any) -> Optional[Decimal]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return Decimal(str(v))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _compare_raw_amount_decimal(row: pd.Series, prefix: str) -> Optional[Decimal]:
    kind = row.get(f"{prefix}_kind")
    if kind is None:
        return None
    try:
        if pd.isna(kind):
            return None
    except Exception:
        pass
    kind_s = str(kind)
    if kind_s == "XRP":
        drops = row.get(f"{prefix}_drops")
        if drops is None:
            return None
        try:
            if pd.isna(drops):
                return None
        except Exception:
            pass
        try:
            return Decimal(int(drops)) * XRP_QUANTUM
        except Exception:
            return None
    if kind_s == "IOU":
        m = row.get(f"{prefix}_mantissa")
        e = row.get(f"{prefix}_exponent")
        if m is None or e is None:
            return None
        try:
            if pd.isna(m) or pd.isna(e):
                return None
        except Exception:
            pass
        try:
            return Decimal(int(m)) * (Decimal(10) ** int(e))
        except Exception:
            return None
    return None


def _compare_raw_quality_decimal(row: pd.Series, prefix: str) -> Optional[Decimal]:
    m = row.get(f"{prefix}_mantissa")
    e = row.get(f"{prefix}_exponent")
    if m is None or e is None:
        return None
    try:
        if pd.isna(m) or pd.isna(e):
            return None
    except Exception:
        pass
    try:
        return Decimal(int(m)) * (Decimal(10) ** int(e))
    except Exception:
        return None


def _f(v: Optional[Decimal]) -> Optional[float]:
    if v is None:
        return None
    return float(v)


def _jsonable(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (str, int, bool)):
        return v
    if isinstance(v, float):
        if pd.isna(v):
            return None
        return v
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.isoformat()
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def _print_progress(prefix: str, message: str) -> None:
    print(f"\r{prefix} {message}", end="", flush=True)


def _write_ndjson(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            obj = {k: _jsonable(v) for k, v in r.items()}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _bool_is_ok_error(err: Any) -> bool:
    if err is None:
        return True
    try:
        if pd.isna(err):
            return True
    except Exception:
        pass
    s = str(err).strip().lower()
    return s in {"", "none", "nan"}


def _direction_to_currencies(direction: str) -> Tuple[str, str]:
    d = str(direction or "")
    if "XRP->rUSD" in d:
        return XRP, RUSD_HEX
    if "rUSD->XRP" in d:
        return RUSD_HEX, XRP
    raise RuntimeError(f"Unexpected direction label: {direction}")


def _derive_slices_name(compare_name: str) -> str:
    if compare_name.startswith("compare_rolling_"):
        return f"compare_rolling_slices_{compare_name[len('compare_rolling_'):]}"
    stem = compare_name
    if stem.endswith(".parquet"):
        stem = stem[:-8]
    return f"{stem}_slices.parquet"


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _resolve_real_metrics_cache_path(
    args: argparse.Namespace,
    metadata_ndjson: Path,
    ledger_start: int,
    ledger_end: int,
) -> Path:
    if args.real_metrics_cache:
        return Path(args.real_metrics_cache).resolve()
    pair_safe = str(args.pair).replace("/", "_")
    return metadata_ndjson.parent / f"real_metrics_cache_{pair_safe}_{ledger_start}_{ledger_end}_v1.ndjson"


def _metadata_signature(metadata_ndjson: Path) -> Dict[str, Any]:
    st = metadata_ndjson.stat()
    return {
        "metadata_path": str(metadata_ndjson.resolve()),
        "metadata_size": int(st.st_size),
        "metadata_mtime_ns": int(st.st_mtime_ns),
    }


def _cache_meta_matches(cache_meta: Dict[str, Any], expected_sig: Dict[str, Any]) -> bool:
    try:
        if int(cache_meta.get("version")) != int(REAL_METRICS_CACHE_VERSION):
            return False
    except Exception:
        return False
    if str(cache_meta.get("metadata_path") or "") != str(expected_sig["metadata_path"]):
        return False
    try:
        if int(cache_meta.get("metadata_size")) != int(expected_sig["metadata_size"]):
            return False
        if int(cache_meta.get("metadata_mtime_ns")) != int(expected_sig["metadata_mtime_ns"]):
            return False
    except Exception:
        return False
    return True


def _real_metrics_to_cache_row(tx_hash: str, m: RealMetrics) -> Dict[str, Any]:
    return {
        "scope": "tx",
        "transaction_hash": tx_hash,
        "real_in": str(m.real_in),
        "real_out": str(m.real_out),
        "real_amm_in": str(m.real_amm_in),
        "real_amm_out": str(m.real_amm_out),
        "real_clob_in": str(m.real_clob_in),
        "real_clob_out": str(m.real_clob_out),
        "real_segment_count": int(m.real_segment_count),
        "real_clob_segment_count": int(m.real_clob_segment_count),
        "real_amm_segment_count": int(m.real_amm_segment_count),
        "real_offer_ids": [str(x) for x in (m.real_offer_ids or [])],
        "real_offer_unique": bool(m.real_offer_unique),
    }


def _real_metrics_from_cache_row(obj: Dict[str, Any]) -> Optional[Tuple[str, RealMetrics]]:
    txh = str(obj.get("transaction_hash") or "").upper()
    if not txh:
        return None
    offer_ids_raw = obj.get("real_offer_ids") or []
    if isinstance(offer_ids_raw, list):
        offer_ids = [str(x) for x in offer_ids_raw if x is not None]
    else:
        offer_ids = [str(offer_ids_raw)]
    return txh, RealMetrics(
        real_in=_d(obj.get("real_in")),
        real_out=_d(obj.get("real_out")),
        real_amm_in=_d(obj.get("real_amm_in")),
        real_amm_out=_d(obj.get("real_amm_out")),
        real_clob_in=_d(obj.get("real_clob_in")),
        real_clob_out=_d(obj.get("real_clob_out")),
        real_segment_count=int(_d(obj.get("real_segment_count"))),
        real_clob_segment_count=int(_d(obj.get("real_clob_segment_count"))),
        real_amm_segment_count=int(_d(obj.get("real_amm_segment_count"))),
        real_offer_ids=offer_ids,
        real_offer_unique=bool(obj.get("real_offer_unique", True)),
    )


def _real_segment_to_cache_row(tx_hash: str, seg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scope": "tx_segment",
        "side": "real",
        "transaction_hash": tx_hash,
        "segment_no": int(_d(seg.get("segment_no"))),
        "segment_kind": str(seg.get("segment_kind") or ""),
        "node_type": seg.get("node_type"),
        "source_id": seg.get("source_id"),
        "step_idx": (None if seg.get("step_idx") is None else int(_d(seg.get("step_idx")))),
        "out_take": str(_d(seg.get("out_take"))),
        "in_take": str(_d(seg.get("in_take"))),
        "avg_quality": (str(_d(seg.get("avg_quality"))) if seg.get("avg_quality") is not None else None),
        "clob_offer_id": seg.get("clob_offer_id"),
    }


def _real_segment_from_cache_row(obj: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    txh = str(obj.get("transaction_hash") or "").upper()
    if not txh:
        return None
    out_take = _d(obj.get("out_take"))
    in_take = _d(obj.get("in_take"))
    if out_take <= 0 or in_take <= 0:
        return None
    avg_quality = _d_opt(obj.get("avg_quality"))
    if avg_quality is None and in_take > 0:
        avg_quality = out_take / in_take
    return txh, {
        "segment_no": int(_d(obj.get("segment_no"))),
        "segment_kind": str(obj.get("segment_kind") or ""),
        "node_type": obj.get("node_type"),
        "source_id": obj.get("source_id"),
        "clob_offer_id": obj.get("clob_offer_id"),
        "step_idx": (None if obj.get("step_idx") is None else int(_d(obj.get("step_idx")))),
        "out_take": out_take,
        "in_take": in_take,
        "avg_quality": avg_quality,
    }


def _load_real_metrics_cache(
    cache_path: Path,
    metadata_ndjson: Path,
    wanted_tx: Optional[set[str]] = None,
) -> Tuple[Dict[str, RealMetrics], Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    if not cache_path.exists():
        return {}, {}, {"status": "miss", "reason": "cache_not_found", "loaded_tx": 0}

    expected_sig = _metadata_signature(metadata_ndjson)
    out: Dict[str, RealMetrics] = {}
    seg_out: Dict[str, List[Dict[str, Any]]] = {}
    seen_lines = 0
    cache_meta: Optional[Dict[str, Any]] = None

    try:
        with cache_path.open("r", encoding="utf-8") as f:
            for ln in f:
                seen_lines += 1
                s = ln.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue

                scope = str(obj.get("scope") or "")
                if scope == "meta" and cache_meta is None:
                    cache_meta = obj
                    continue
                if scope == "tx_segment":
                    parsed_seg = _real_segment_from_cache_row(obj)
                    if parsed_seg is None:
                        continue
                    txh, seg = parsed_seg
                    if wanted_tx is not None and txh not in wanted_tx:
                        continue
                    seg_out.setdefault(txh, []).append(seg)
                    continue

                parsed = _real_metrics_from_cache_row(obj)
                if parsed is None:
                    continue
                txh, m = parsed
                if wanted_tx is not None and txh not in wanted_tx:
                    continue
                out[txh] = m
    except Exception as e:
        return {}, {}, {"status": "miss", "reason": f"cache_read_error:{type(e).__name__}", "loaded_tx": 0}

    if cache_meta is None:
        return {}, {}, {
            "status": "miss",
            "reason": "cache_missing_meta_header",
            "loaded_tx": 0,
            "seen_lines": int(seen_lines),
        }

    if not _cache_meta_matches(cache_meta, expected_sig):
        return {}, {}, {
            "status": "miss",
            "reason": "cache_signature_mismatch",
            "loaded_tx": 0,
            "seen_lines": int(seen_lines),
        }

    for txh in out.keys():
        seg_out.setdefault(txh, [])
    for txh, segs in seg_out.items():
        segs.sort(
            key=lambda s: (
                int(_d(s.get("segment_no"))),
                str(s.get("segment_kind") or ""),
                str(s.get("source_id") or ""),
            )
        )
        for i, seg in enumerate(segs, start=1):
            seg["segment_no"] = int(i)
            if seg.get("clob_offer_id") is None and str(seg.get("segment_kind") or "").upper() == "CLOB":
                seg["clob_offer_id"] = _short_offer_id(seg.get("source_id"))

    return out, seg_out, {"status": "hit", "reason": "ok", "loaded_tx": int(len(out)), "seen_lines": int(seen_lines)}


def _write_real_metrics_cache(
    cache_path: Path,
    metadata_ndjson: Path,
    pair: str,
    ledger_start: int,
    ledger_end: int,
    real_map: Dict[str, RealMetrics],
    real_segments_map: Dict[str, List[Dict[str, Any]]],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sig = _metadata_signature(metadata_ndjson)
    meta_row = {
        "scope": "meta",
        "version": int(REAL_METRICS_CACHE_VERSION),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair": str(pair),
        "ledger_start": int(ledger_start),
        "ledger_end": int(ledger_end),
        **sig,
        "tx_count": int(len(real_map)),
    }

    with cache_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta_row, ensure_ascii=False) + "\n")
        for txh in sorted(real_map.keys()):
            f.write(json.dumps(_real_metrics_to_cache_row(txh, real_map[txh]), ensure_ascii=False) + "\n")
        for txh in sorted(real_segments_map.keys()):
            for seg in real_segments_map.get(txh, []):
                f.write(json.dumps(_real_segment_to_cache_row(txh, seg), ensure_ascii=False) + "\n")


def _run_rolling_compare(args: argparse.Namespace, rolling_out_dir: Path, rolling_out_name: str) -> Tuple[Path, Path]:
    rolling_out_dir.mkdir(parents=True, exist_ok=True)
    compare_path = rolling_out_dir / rolling_out_name
    slices_path = rolling_out_dir / _derive_slices_name(rolling_out_name)

    need_run = args.force_rerun_rolling or (not compare_path.exists()) or (not slices_path.exists())
    expect_tx_prebook = str(Path(args.tx_prebook_snapshots).resolve())
    expect_metadata = str(Path(args.metadata_ndjson).resolve())
    if (not need_run) and (not args.force_rerun_rolling):
        manifest_path = rolling_out_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
            manifest_mode = str(manifest.get("state_mode") or "")
            manifest_output = str(manifest.get("output_parquet") or "")
            manifest_slices = str(manifest.get("output_slices_parquet") or "")
            if manifest_mode != str(args.state_mode):
                print(
                    "[rolling] existing manifest state_mode mismatch: "
                    f"found={manifest_mode or 'unknown'} expected={args.state_mode}; keep reuse by filename."
                )
            elif manifest_output:
                try:
                    if Path(manifest_output).resolve() != compare_path.resolve():
                        print("[rolling] existing manifest output_parquet mismatch; keep reuse by filename.")
                except Exception:
                    pass
            if (not need_run) and manifest_slices:
                try:
                    if Path(manifest_slices).resolve() != slices_path.resolve():
                        print("[rolling] existing manifest output_slices_parquet mismatch; keep reuse by filename.")
                except Exception:
                    pass
            manifest_inputs = manifest.get("inputs") if isinstance(manifest, dict) else {}
            manifest_tx_prebook = ""
            manifest_metadata = ""
            if isinstance(manifest_inputs, dict):
                manifest_tx_prebook = str(manifest_inputs.get("tx_prebook_snapshots") or "")
                manifest_metadata = str(manifest_inputs.get("metadata_ndjson") or "")
            if manifest_tx_prebook != expect_tx_prebook:
                print(
                    "[rolling] existing manifest tx_prebook_snapshots mismatch: "
                    f"found={manifest_tx_prebook or 'none'} expected={expect_tx_prebook or 'none'}; rerun rolling."
                )
                need_run = True
            if manifest_metadata != expect_metadata:
                print(
                    "[rolling] existing manifest metadata_ndjson mismatch: "
                    f"found={manifest_metadata or 'none'} expected={expect_metadata or 'none'}; rerun rolling."
                )
                need_run = True
        elif expect_tx_prebook:
            print("[rolling] manifest missing; cannot verify prebook input mode. rerun rolling.")
            need_run = True

    if not need_run:
        print(f"[rolling] reuse existing compare output: {compare_path}")
        print(f"[rolling] reuse existing slices output : {slices_path}")
        return compare_path, slices_path

    script_path = Path(__file__).resolve().parent / "empirical_compare_rolling.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find rolling script: {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--pair",
        str(args.pair),
        "--ledger-start",
        str(int(args.ledger_start)),
        "--ledger-end",
        str(int(args.ledger_end)),
        "--amm-swaps",
        str(Path(args.amm_swaps).resolve()),
        "--amm-fees",
        str(Path(args.amm_fees).resolve()),
        "--clob-legs",
        str(Path(args.clob_legs).resolve()),
        "--metadata-ndjson",
        str(Path(args.metadata_ndjson).resolve()),
        "--tx-prebook-snapshots",
        str(Path(args.tx_prebook_snapshots).resolve()),
        "--clob-roll-intra-ledger",
        str(int(bool(args.clob_roll_intra_ledger))),
        "--state-mode",
        str(args.state_mode),
        "--amm-roll-mode",
        str(args.amm_roll_mode),
        "--output-dir",
        str(rolling_out_dir),
        "--output-name",
        rolling_out_name,
    ]
    if args.write_manifest:
        cmd.append("--write-manifest")

    print("[rolling] running:")
    print(" ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PRINT_PER_TX", "0")
    env.setdefault("DEBUG_DIR_RUSD_TO_XRP", "0")
    proc = subprocess.run(cmd, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Rolling compare failed with exit code {proc.returncode}")

    _ensure_exists(compare_path, "rolling compare parquet")
    _ensure_exists(slices_path, "rolling slices parquet")
    return compare_path, slices_path


def _parse_amt_to_decimal_for_currency(v: Any, target_cur: str) -> Optional[Decimal]:
    if v is None:
        return None

    if isinstance(v, str):
        if target_cur != XRP:
            return None
        try:
            return Decimal(v) / Decimal("1000000")
        except (InvalidOperation, ValueError, TypeError):
            return None

    if isinstance(v, dict):
        cur = v.get("currency")
        if target_cur == XRP:
            if cur is not None and str(cur) != XRP:
                return None
            val = v.get("value")
            if val is None:
                return None
            try:
                return Decimal(str(val))
            except (InvalidOperation, ValueError, TypeError):
                return None

        if str(cur) != target_cur:
            return None
        val = v.get("value")
        if val is None:
            return None
        try:
            return Decimal(str(val))
        except (InvalidOperation, ValueError, TypeError):
            return None

    return None


def _offer_order_key(body: Dict[str, Any]) -> str:
    ledger_index = body.get("LedgerIndex")
    if ledger_index:
        return str(ledger_index)

    final = body.get("FinalFields") or body.get("NewFields") or {}
    account = final.get("Account")
    seq = final.get("Sequence")
    if account is not None and seq is not None:
        return f"{account}:{seq}"

    return "unknown-offer"


def _amount_obj_to_decimal(v: Any) -> Optional[Decimal]:
    if v is None:
        return None
    if isinstance(v, str):
        try:
            return Decimal(v) / Decimal("1000000")
        except (InvalidOperation, ValueError, TypeError):
            return None
    if isinstance(v, dict):
        val = v.get("value")
        if val is None:
            return None
        try:
            return Decimal(str(val))
        except (InvalidOperation, ValueError, TypeError):
            return None
    return None


def _short_offer_id(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v)
    if not s:
        return None
    return s[:4]


def _sort_segments_for_notebook_display(segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seg in segs:
        kind = str(seg.get("segment_kind") or "").upper()
        source_id = seg.get("source_id")
        clob_offer_id = _short_offer_id(source_id) if (kind == "CLOB" and source_id is not None) else None
        avg_quality = seg.get("avg_quality")
        if avg_quality is None:
            avg_q_sort = Decimal("-Infinity")
        else:
            avg_q_sort = _d(avg_quality)
        rows.append(
            {
                "segment_no": int(_d(seg.get("segment_no"))),
                "segment_kind": kind,
                "node_type": seg.get("node_type"),
                "source_id": source_id,
                "clob_offer_id": clob_offer_id,
                "out_take": _d(seg.get("out_take")),
                "in_take": _d(seg.get("in_take")),
                "avg_quality": (_d(avg_quality) if avg_quality is not None else None),
                "_avg_q_sort": avg_q_sort,
                "_offer_sort": (str(clob_offer_id).upper() if clob_offer_id is not None else ""),
            }
        )

    rows.sort(key=lambda x: x["_offer_sort"])
    rows.sort(key=lambda x: x["_avg_q_sort"], reverse=True)
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows, start=1):
        out.append(
            {
                "segment_no": int(i),
                "segment_kind": r["segment_kind"],
                "node_type": r.get("node_type"),
                "source_id": r.get("source_id"),
                "clob_offer_id": r.get("clob_offer_id"),
                "out_take": r.get("out_take"),
                "in_take": r.get("in_take"),
                "avg_quality": r.get("avg_quality"),
            }
        )
    return out


def _build_model_notebook_views(
    model_exec_segs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    premerge = [dict(s) for s in model_exec_segs]
    premerge_display = _sort_segments_for_notebook_display(premerge)

    amm_rows = [s for s in premerge if str(s.get("segment_kind") or "").upper() == "AMM"]
    other_rows = [s for s in premerge if str(s.get("segment_kind") or "").upper() != "AMM"]
    if amm_rows:
        amm_out_sum = sum((_d(s.get("out_take")) for s in amm_rows), Decimal("0"))
        amm_in_sum = sum((_d(s.get("in_take")) for s in amm_rows), Decimal("0"))
        merged_amm_row = {
            "segment_no": 0,
            "segment_kind": "AMM",
            "node_type": "ModifiedNode(model-AMM)",
            "source_id": None,
            "out_take": amm_out_sum,
            "in_take": amm_in_sum,
            "avg_quality": ((amm_out_sum / amm_in_sum) if amm_in_sum > 0 else None),
        }
        merged_input = other_rows + [merged_amm_row]
    else:
        merged_input = premerge

    merged_display = _sort_segments_for_notebook_display(merged_input)
    premerge_out = premerge_display if len(amm_rows) > 1 else None
    return merged_display, premerge_out


def _compute_amm_trade_from_metadata(tx_response: Dict[str, Any], in_cur: str, out_cur: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    res = tx_response.get("result") or {}
    meta = res.get("meta") or {}
    affected = meta.get("AffectedNodes") or []

    pool_accounts: set[str] = set()
    for n in affected:
        node_kind = next((k for k in ("CreatedNode", "ModifiedNode", "DeletedNode") if k in n), None)
        if node_kind is None:
            continue
        body = n[node_kind]
        if body.get("LedgerEntryType") != "AccountRoot":
            continue
        prev = body.get("PreviousFields") or {}
        fin = body.get("FinalFields") or body.get("NewFields") or {}
        if ("AMMID" in prev) or ("AMMID" in fin):
            acct = fin.get("Account") or prev.get("Account")
            if acct:
                pool_accounts.add(str(acct))

    if not pool_accounts:
        return None, None

    deltas: Dict[str, Decimal] = {}

    for n in affected:
        node_kind = next((k for k in ("CreatedNode", "ModifiedNode", "DeletedNode") if k in n), None)
        if node_kind is None:
            continue
        body = n[node_kind]
        prev = body.get("PreviousFields") or {}
        fin = body.get("FinalFields") or body.get("NewFields") or {}
        le = body.get("LedgerEntryType")

        if le == "AccountRoot":
            acct = fin.get("Account") or prev.get("Account")
            if acct in pool_accounts:
                vb = _amount_obj_to_decimal(prev.get("Balance"))
                va = _amount_obj_to_decimal(fin.get("Balance"))
                if vb is not None and va is not None:
                    deltas[XRP] = deltas.get(XRP, Decimal("0")) + (va - vb)

        elif le == "RippleState":
            hb = (fin.get("HighLimit") or prev.get("HighLimit") or {})
            lb = (fin.get("LowLimit") or prev.get("LowLimit") or {})
            high = hb.get("issuer")
            low = lb.get("issuer")

            pool_side = None
            if high in pool_accounts:
                pool_side = "high"
            elif low in pool_accounts:
                pool_side = "low"
            if pool_side is None:
                continue

            bal_b = _amount_obj_to_decimal(prev.get("Balance"))
            bal_a = _amount_obj_to_decimal(fin.get("Balance"))
            if bal_b is None or bal_a is None:
                continue

            ccy = (fin.get("Balance") or prev.get("Balance") or {}).get("currency")
            if ccy is None:
                continue
            ccy = str(ccy)

            if pool_side == "low":
                rb, ra = bal_b, bal_a
            else:
                rb, ra = -bal_b, -bal_a

            deltas[ccy] = deltas.get(ccy, Decimal("0")) + (ra - rb)

    d_in = deltas.get(in_cur)
    d_out = deltas.get(out_cur)

    in_take = max(d_in, Decimal("0")) if d_in is not None else None
    out_take = max(-d_out, Decimal("0")) if d_out is not None else None

    if in_take is None and out_take is None:
        return None, None
    return out_take, in_take


def _extract_real_raw_steps(tx_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    res = tx_response.get("result") or {}
    meta = res.get("meta") or {}
    affected = meta.get("AffectedNodes") or []

    out: List[Dict[str, Any]] = []
    for step_idx, node in enumerate(affected):
        node_kind = next((k for k in ("CreatedNode", "ModifiedNode", "DeletedNode") if k in node), None)
        if node_kind is None:
            continue

        body = node[node_kind]
        entry = body.get("LedgerEntryType")
        prev = body.get("PreviousFields") or {}
        fin = body.get("FinalFields") or body.get("NewFields") or {}

        if entry == "AMM":
            source_id = str(body.get("LedgerIndex") or fin.get("Account") or "amm-pool")
            out.append(
                {
                    "step_idx": int(step_idx),
                    "segment_kind": "AMM",
                    "source_id": source_id,
                    "node_type": str(node_kind),
                    "offer_gets_before": None,
                    "offer_gets_after": None,
                    "offer_pays_before": None,
                    "offer_pays_after": None,
                }
            )
            continue

        if entry == "AccountRoot" and (("AMMID" in fin) or ("AMMID" in prev)):
            source_id = str(fin.get("AMMID") or prev.get("AMMID") or fin.get("Account") or "amm-accountroot")
            out.append(
                {
                    "step_idx": int(step_idx),
                    "segment_kind": "AMM",
                    "source_id": source_id,
                    "node_type": str(node_kind),
                    "offer_gets_before": None,
                    "offer_gets_after": None,
                    "offer_pays_before": None,
                    "offer_pays_after": None,
                }
            )
            continue

        if entry != "Offer":
            continue

        gets_before = prev.get("TakerGets")
        gets_after = fin.get("TakerGets")
        pays_before = prev.get("TakerPays")
        pays_after = fin.get("TakerPays")

        changed_gets = (gets_before is not None) and (gets_after is not None) and (gets_before != gets_after)
        changed_pays = (pays_before is not None) and (pays_after is not None) and (pays_before != pays_after)

        # Only count Offer nodes whose before/after delta is explicit in metadata.
        # A bare DeletedNode without PreviousFields is a touched/removed node, not a
        # provable executed leg for path reconstruction.
        if not (changed_gets or changed_pays):
            continue

        out.append(
            {
                "step_idx": int(step_idx),
                "segment_kind": "CLOB",
                "source_id": _offer_order_key(body),
                "node_type": str(node_kind),
                "offer_gets_before": gets_before,
                "offer_gets_after": gets_after,
                "offer_pays_before": pays_before,
                "offer_pays_after": pays_after,
            }
        )

    return out


def _build_real_tx_metrics_and_segments(
    tx_response: Dict[str, Any],
    in_cur: str,
    out_cur: str,
) -> Tuple[RealMetrics, List[Dict[str, Any]]]:
    raw_steps = _extract_real_raw_steps(tx_response)
    amm_out_tx, amm_in_tx = _compute_amm_trade_from_metadata(tx_response, in_cur=in_cur, out_cur=out_cur)
    amm_filled = False

    segs: List[Dict[str, Any]] = []
    clob_offer_ids: List[str] = []

    for s in raw_steps:
        kind = str(s.get("segment_kind") or "").upper()
        out_d: Optional[Decimal] = None
        in_d: Optional[Decimal] = None

        if kind == "CLOB":
            gets_before = _parse_amt_to_decimal_for_currency(s.get("offer_gets_before"), out_cur)
            gets_after = _parse_amt_to_decimal_for_currency(s.get("offer_gets_after"), out_cur)
            pays_before = _parse_amt_to_decimal_for_currency(s.get("offer_pays_before"), in_cur)
            pays_after = _parse_amt_to_decimal_for_currency(s.get("offer_pays_after"), in_cur)

            if None not in (gets_before, gets_after, pays_before, pays_after):
                out_cand = gets_before - gets_after
                in_cand = pays_before - pays_after
                if out_cand > 0 and in_cand > 0:
                    out_d, in_d = out_cand, in_cand

            if out_d is None or in_d is None:
                pays_before_out = _parse_amt_to_decimal_for_currency(s.get("offer_pays_before"), out_cur)
                pays_after_out = _parse_amt_to_decimal_for_currency(s.get("offer_pays_after"), out_cur)
                gets_before_in = _parse_amt_to_decimal_for_currency(s.get("offer_gets_before"), in_cur)
                gets_after_in = _parse_amt_to_decimal_for_currency(s.get("offer_gets_after"), in_cur)

                if None not in (pays_before_out, pays_after_out, gets_before_in, gets_after_in):
                    out_cand = pays_before_out - pays_after_out
                    in_cand = gets_before_in - gets_after_in
                    if out_cand > 0 and in_cand > 0:
                        out_d, in_d = out_cand, in_cand

        elif kind == "AMM":
            if (not amm_filled) and (amm_out_tx is not None) and (amm_in_tx is not None):
                out_d, in_d = amm_out_tx, amm_in_tx
                amm_filled = True

        if out_d is None or in_d is None or out_d <= 0 or in_d <= 0:
            continue

        if kind == "CLOB":
            sid = str(s.get("source_id") or "")
            if sid:
                clob_offer_ids.append(sid)

        avg_quality = out_d / in_d if in_d > 0 else None
        clob_offer_id = (_short_offer_id(s.get("source_id")) if kind == "CLOB" else None)
        segs.append(
            {
                "segment_no": int(len(segs) + 1),
                "segment_kind": kind,
                "node_type": s.get("node_type"),
                "source_id": s.get("source_id"),
                "clob_offer_id": clob_offer_id,
                "step_idx": s.get("step_idx"),
                "out_take": out_d,
                "in_take": in_d,
                "avg_quality": avg_quality,
            }
        )

    real_amm_in = sum((x["in_take"] for x in segs if str(x.get("segment_kind")).upper() == "AMM"), Decimal("0"))
    real_amm_out = sum((x["out_take"] for x in segs if str(x.get("segment_kind")).upper() == "AMM"), Decimal("0"))
    real_clob_in = sum((x["in_take"] for x in segs if str(x.get("segment_kind")).upper() == "CLOB"), Decimal("0"))
    real_clob_out = sum((x["out_take"] for x in segs if str(x.get("segment_kind")).upper() == "CLOB"), Decimal("0"))

    real_clob_segment_count = int(sum(1 for x in segs if str(x.get("segment_kind")).upper() == "CLOB"))
    real_amm_segment_count = int(sum(1 for x in segs if str(x.get("segment_kind")).upper() == "AMM"))
    real_offer_unique = len(clob_offer_ids) == len(set(clob_offer_ids))

    metrics = RealMetrics(
        real_in=real_amm_in + real_clob_in,
        real_out=real_amm_out + real_clob_out,
        real_amm_in=real_amm_in,
        real_amm_out=real_amm_out,
        real_clob_in=real_clob_in,
        real_clob_out=real_clob_out,
        real_segment_count=int(len(segs)),
        real_clob_segment_count=real_clob_segment_count,
        real_amm_segment_count=real_amm_segment_count,
        real_offer_ids=clob_offer_ids,
        real_offer_unique=bool(real_offer_unique),
    )
    return metrics, segs


def _build_model_maps(df_slices: pd.DataFrame) -> Tuple[Dict[str, ModelMetrics], Dict[str, List[Dict[str, Any]]]]:
    if df_slices.empty:
        return {}, {}

    df = df_slices.copy()
    required_raw_cols = {
        "out_take_kind",
        "out_take_drops",
        "out_take_mantissa",
        "out_take_exponent",
        "in_take_kind",
        "in_take_drops",
        "in_take_mantissa",
        "in_take_exponent",
    }
    missing_raw = sorted(required_raw_cols - set(df.columns))
    if missing_raw:
        raise RuntimeError(
            "Model slices missing canonical raw columns: "
            f"{missing_raw}"
        )
    df["transaction_hash"] = df["transaction_hash"].astype(str).str.upper()
    if "segment_no" in df.columns:
        df = df.sort_values(["transaction_hash", "segment_no"], na_position="last")

    out: Dict[str, ModelMetrics] = {}
    seg_out: Dict[str, List[Dict[str, Any]]] = {}

    for txh, g in df.groupby("transaction_hash", sort=False):
        model_amm_in = Decimal("0")
        model_amm_out = Decimal("0")
        model_clob_in = Decimal("0")
        model_clob_out = Decimal("0")
        model_clob_segment_count = 0
        model_amm_segment_count_premerge = 0
        model_offer_ids: List[str] = []
        model_segments: List[Dict[str, Any]] = []

        for _, r in g.iterrows():
            kind = str(r.get("segment_kind") or "").upper()
            in_d = _compare_raw_amount_decimal(r, "in_take")
            out_d = _compare_raw_amount_decimal(r, "out_take")
            if in_d is None or out_d is None or in_d <= 0 or out_d <= 0:
                continue

            sid = r.get("source_id")
            sid_str = None if sid is None or (isinstance(sid, float) and pd.isna(sid)) else str(sid)

            if kind == "AMM":
                model_amm_segment_count_premerge += 1
                model_amm_in += in_d
                model_amm_out += out_d
            elif kind == "CLOB":
                model_clob_segment_count += 1
                model_clob_in += in_d
                model_clob_out += out_d
                if sid_str:
                    model_offer_ids.append(sid_str)

            avg_quality = _compare_raw_quality_decimal(r, "avg_quality")
            if avg_quality is None and in_d > 0:
                avg_quality = out_d / in_d
            seg_no_raw = r.get("segment_no")
            if seg_no_raw is None or (isinstance(seg_no_raw, float) and pd.isna(seg_no_raw)):
                seg_no = int(len(model_segments) + 1)
            else:
                seg_no = int(_d(seg_no_raw))
            model_segments.append(
                {
                    "segment_no": seg_no,
                    "segment_kind": kind,
                    "node_type": r.get("node_type"),
                    "source_id": sid_str,
                    "clob_offer_id": (_short_offer_id(sid_str) if kind == "CLOB" and sid_str else None),
                    "step_idx": None,
                    "out_take": out_d,
                    "in_take": in_d,
                    "avg_quality": avg_quality,
                }
            )

        model_amm_segment_count_postmerge = 1 if model_amm_segment_count_premerge > 0 else 0
        model_segment_count = model_clob_segment_count + model_amm_segment_count_postmerge

        model_offer_unique = len(model_offer_ids) == len(set(model_offer_ids))

        out[str(txh)] = ModelMetrics(
            model_in=model_amm_in + model_clob_in,
            model_out=model_amm_out + model_clob_out,
            model_amm_in=model_amm_in,
            model_amm_out=model_amm_out,
            model_clob_in=model_clob_in,
            model_clob_out=model_clob_out,
            model_segment_count=int(model_segment_count),
            model_clob_segment_count=int(model_clob_segment_count),
            model_amm_segment_count_premerge=int(model_amm_segment_count_premerge),
            model_amm_segment_count_postmerge=int(model_amm_segment_count_postmerge),
            model_offer_ids=model_offer_ids,
            model_offer_unique=bool(model_offer_unique),
        )
        model_segments.sort(
            key=lambda s: (
                int(_d(s.get("segment_no"))),
                str(s.get("segment_kind") or ""),
                str(s.get("source_id") or ""),
            )
        )
        for i, seg in enumerate(model_segments, start=1):
            seg["segment_no"] = int(i)
        seg_out[str(txh)] = model_segments

    return out, seg_out


def _build_real_metrics_map(
    metadata_ndjson: Path,
    tx_dir_map: Dict[str, Tuple[str, str]],
) -> Tuple[Dict[str, RealMetrics], Dict[str, List[Dict[str, Any]]]]:
    out: Dict[str, RealMetrics] = {}
    seg_out: Dict[str, List[Dict[str, Any]]] = {}
    wanted = set(tx_dir_map.keys())

    seen = 0
    matched_rows = 0
    total_bytes = int(metadata_ndjson.stat().st_size)
    bytes_read = 0
    last_progress_ts = 0.0

    with metadata_ndjson.open("r", encoding="utf-8") as f:
        for ln in f:
            seen += 1
            bytes_read += len(ln.encode("utf-8"))
            now = time.perf_counter()
            if (now - last_progress_ts) >= 0.5:
                pct = ((bytes_read / total_bytes) * 100.0) if total_bytes > 0 else 0.0
                _print_progress(
                    "[metadata][progress]",
                    f"{pct:6.2f}% | lines={seen} | matched_unique_tx={len(out)}/{len(wanted)}",
                )
                last_progress_ts = now
            s = ln.strip()
            if not s:
                continue

            try:
                obj = json.loads(s)
            except Exception:
                continue

            txh = str(obj.get("tx_hash") or (obj.get("result") or {}).get("hash") or "").upper()
            if txh not in wanted:
                continue

            matched_rows += 1
            in_cur, out_cur = tx_dir_map[txh]

            metrics, segs = _build_real_tx_metrics_and_segments(obj, in_cur=in_cur, out_cur=out_cur)
            out[txh] = metrics
            seg_out[txh] = segs

            if len(out) >= len(wanted):
                break

    _print_progress(
        "[metadata][progress]",
        f"100.00% | lines={seen} | matched_unique_tx={len(out)}/{len(wanted)}",
    )
    print()
    print(
        f"[metadata] scanned lines={seen} | matched rows={matched_rows} "
        f"| matched unique tx={len(out)}/{len(wanted)}"
    )
    return out, seg_out


def _pct(part: Decimal, total: Decimal) -> Optional[Decimal]:
    if total == 0:
        return None
    return (part / total) * Decimal("100")


def _cheaper_metrics(real_in: Decimal, model_in: Decimal, tol: Decimal = Decimal("1e-18")) -> Tuple[str, Decimal, Optional[Decimal]]:
    in_saving = real_in - model_in
    if in_saving > tol:
        cheaper_side = "MODEL"
    elif in_saving < -tol:
        cheaper_side = "REAL"
    else:
        cheaper_side = "TIE"
    if real_in == 0:
        return cheaper_side, in_saving, None
    return cheaper_side, in_saving, (in_saving / real_in) * Decimal("10000")


def _strict_segment_signature(segs: List[Dict[str, Any]]) -> List[Tuple[str, Optional[str]]]:
    sig: List[Tuple[str, Optional[str]]] = []
    amm_present = False
    for seg in segs or []:
        kind = str(seg.get("segment_kind") or "").upper()
        if kind == "CLOB":
            sid = seg.get("source_id")
            sid_str = None if sid is None else str(sid)
            sig.append(("CLOB", sid_str))
        elif kind == "AMM":
            amm_present = True
    if amm_present:
        sig.append(("AMM", None))
    sig.sort()
    return sig


def _build_tx_metrics(
    df_compare: pd.DataFrame,
    model_map: Dict[str, ModelMetrics],
    real_map: Dict[str, RealMetrics],
    model_segments_map: Dict[str, List[Dict[str, Any]]],
    real_segments_map: Dict[str, List[Dict[str, Any]]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total_tx = int(len(df_compare))
    last_progress_ts = 0.0

    for i, (_, r) in enumerate(df_compare.iterrows(), start=1):
        now = time.perf_counter()
        if (now - last_progress_ts) >= 0.5:
            pct = (float(i) / float(total_tx) * 100.0) if total_tx > 0 else 0.0
            _print_progress("[tx-metrics][progress]", f"{pct:6.2f}% | tx={i}/{total_tx}")
            last_progress_ts = now
        txh = str(r.get("transaction_hash") or "").upper()

        model = model_map.get(txh)
        real = real_map.get(txh)
        if real is None:
            raise RuntimeError(
                "Strict metadata-only violated: missing metadata-derived real metrics for "
                f"tx={txh}"
            )

        real_in = real.real_in
        real_out = real.real_out

        compare_model_in_raw = _compare_raw_amount_decimal(r, "model_in")
        compare_model_out_raw = _compare_raw_amount_decimal(r, "model_out")
        compare_model_ok = (compare_model_in_raw is not None) and (compare_model_out_raw is not None)

        if model is not None:
            model_in: Optional[Decimal] = model.model_in
            model_out: Optional[Decimal] = model.model_out
        else:
            if compare_model_ok:
                raise RuntimeError(
                    "Strict mode violated: compare row has model totals but slices-derived model "
                    f"metrics are missing for tx={txh}"
                )
            model_in = None
            model_out = None

        diff_in: Optional[Decimal]
        diff_out: Optional[Decimal]
        in_gap_pct: Optional[Decimal]
        out_gap_pct: Optional[Decimal]
        if model_in is None or model_out is None:
            diff_in = None
            diff_out = None
            in_gap_pct = None
            out_gap_pct = None
            cheaper_side = None
            in_saving = None
            in_saving_bps_vs_real = None
        else:
            diff_in = model_in - real_in
            diff_out = model_out - real_out
            in_gap_pct = _pct(diff_in, real_in)
            out_gap_pct = _pct(diff_out, real_out)
            cheaper_side, in_saving, in_saving_bps_vs_real = _cheaper_metrics(real_in, model_in)

        real_avg_quality: Optional[Decimal] = None
        if real_in > 0:
            real_avg_quality = real_out / real_in

        model_avg_quality: Optional[Decimal] = None
        if model_in is not None and model_out is not None and model_in > 0:
            model_avg_quality = model_out / model_in

        avg_quality_diff: Optional[Decimal] = None
        if real_avg_quality is not None and model_avg_quality is not None:
            avg_quality_diff = model_avg_quality - real_avg_quality

        real_amm_in = real.real_amm_in
        real_amm_out = real.real_amm_out
        real_clob_in = real.real_clob_in
        real_clob_out = real.real_clob_out
        real_segment_count = real.real_segment_count
        real_clob_segment_count = real.real_clob_segment_count
        real_amm_segment_count = real.real_amm_segment_count
        real_offer_ids = real.real_offer_ids
        real_offer_unique = real.real_offer_unique

        if model is not None:
            model_amm_in = model.model_amm_in
            model_amm_out = model.model_amm_out
            model_clob_in = model.model_clob_in
            model_clob_out = model.model_clob_out
            model_segment_count = model.model_segment_count
            model_clob_segment_count = model.model_clob_segment_count
            model_amm_segment_count_premerge = model.model_amm_segment_count_premerge
            model_amm_segment_count_postmerge = model.model_amm_segment_count_postmerge
            model_offer_ids = model.model_offer_ids
            model_offer_unique = model.model_offer_unique
        else:
            model_amm_in = None
            model_amm_out = None
            model_clob_in = None
            model_clob_out = None
            model_segment_count = 0
            model_clob_segment_count = int(_d(r.get("model_clob_segs")))
            model_amm_segment_count_premerge = 0
            model_amm_segment_count_postmerge = 0
            model_offer_ids = []
            model_offer_unique = True

        overlap_count = len(set(real_offer_ids) & set(model_offer_ids))

        err = r.get("error")
        err_text = None if _bool_is_ok_error(err) else str(err)
        is_ok = (model_in is not None and model_out is not None)
        has_warning = bool(is_ok and err_text)
        path_sig_real = r.get("path_sig_real")
        path_sig_model = r.get("path_sig_model")
        real_seg_sig = _strict_segment_signature(real_segments_map.get(txh, []))
        model_seg_sig = _strict_segment_signature(model_segments_map.get(txh, []))
        path_match = bool(real_seg_sig == model_seg_sig) if is_ok else False

        rows.append(
            {
                "transaction_hash": txh,
                "ledger_index": int(_d(r.get("ledger_index"))),
                "transaction_index": int(_d(r.get("transaction_index"), default=Decimal("-1"))),
                "direction": r.get("direction"),
                "used_prebook_ledger": (int(_d(r.get("used_prebook_ledger"))) if not pd.isna(r.get("used_prebook_ledger")) else None),
                "path_sig_real": path_sig_real,
                "path_sig_model": path_sig_model,
                "path_match": path_match,
                "is_ok": is_ok,
                "has_warning": has_warning,
                "warning": (err_text if has_warning else None),
                "error": (None if is_ok else err_text),
                "real_in": _f(real_in),
                "model_in": _f(model_in),
                "diff_in": _f(diff_in),
                "in_gap_pct_vs_real": _f(in_gap_pct),
                "cheaper_side": cheaper_side,
                "in_saving": _f(in_saving),
                "in_saving_bps_vs_real": _f(in_saving_bps_vs_real),
                "real_out": _f(real_out),
                "model_out": _f(model_out),
                "diff_out": _f(diff_out),
                "out_gap_pct_vs_real": _f(out_gap_pct),
                "real_avg_quality": _f(real_avg_quality),
                "model_avg_quality": _f(model_avg_quality),
                "avg_quality_diff": _f(avg_quality_diff),
                "real_amm_in": _f(real_amm_in),
                "real_amm_out": _f(real_amm_out),
                "real_clob_in": _f(real_clob_in),
                "real_clob_out": _f(real_clob_out),
                "model_amm_in": _f(model_amm_in),
                "model_amm_out": _f(model_amm_out),
                "model_clob_in": _f(model_clob_in),
                "model_clob_out": _f(model_clob_out),
                "real_amm_in_pct": _f(_pct(real_amm_in, real_in)),
                "real_clob_in_pct": _f(_pct(real_clob_in, real_in)),
                "model_amm_in_pct": _f(_pct(model_amm_in, model_in)) if (model_amm_in is not None and model_in is not None) else None,
                "model_clob_in_pct": _f(_pct(model_clob_in, model_in)) if (model_clob_in is not None and model_in is not None) else None,
                "real_segment_count": int(real_segment_count),
                "real_clob_segment_count": int(real_clob_segment_count),
                "real_amm_segment_count": int(real_amm_segment_count),
                "model_segment_count": int(model_segment_count),
                "model_clob_segment_count": int(model_clob_segment_count),
                "model_amm_segment_count_premerge": int(model_amm_segment_count_premerge),
                "model_amm_segment_count_postmerge": int(model_amm_segment_count_postmerge),
                "real_offer_count": int(len(real_offer_ids)),
                "model_offer_count": int(len(model_offer_ids)),
                "real_offer_unique": bool(real_offer_unique),
                "model_offer_unique": bool(model_offer_unique),
                "model_overlap_with_real_offer_count": int(overlap_count),
            }
        )

    _print_progress("[tx-metrics][progress]", "100.00% | tx={}/{}".format(total_tx, total_tx))
    print()
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df

    out_df = out_df.sort_values(["ledger_index", "transaction_index", "transaction_hash"], kind="mergesort").reset_index(drop=True)
    return out_df


def _agg_ledger(df_tx: pd.DataFrame) -> pd.DataFrame:
    if df_tx.empty:
        return pd.DataFrame()

    g = df_tx.groupby("ledger_index", as_index=False)

    df_ledger = g.agg(
        tx_count=("transaction_hash", "count"),
        ok_tx_count=("is_ok", "sum"),
        warning_tx_count=("has_warning", "sum"),
        path_match_count=("path_match", "sum"),
        real_in_sum=("real_in", "sum"),
        model_in_sum=("model_in", "sum"),
        diff_in_sum=("diff_in", "sum"),
        real_out_sum=("real_out", "sum"),
        model_out_sum=("model_out", "sum"),
        diff_out_sum=("diff_out", "sum"),
        real_amm_in_sum=("real_amm_in", "sum"),
        real_clob_in_sum=("real_clob_in", "sum"),
        model_amm_in_sum=("model_amm_in", "sum"),
        model_clob_in_sum=("model_clob_in", "sum"),
        real_segment_count_sum=("real_segment_count", "sum"),
        model_segment_count_sum=("model_segment_count", "sum"),
        model_overlap_with_real_offer_count_sum=("model_overlap_with_real_offer_count", "sum"),
        in_gap_pct_mean=("in_gap_pct_vs_real", "mean"),
        out_gap_pct_mean=("out_gap_pct_vs_real", "mean"),
        real_avg_quality_mean_tx=("real_avg_quality", "mean"),
        model_avg_quality_mean_tx=("model_avg_quality", "mean"),
        avg_quality_diff_mean_tx=("avg_quality_diff", "mean"),
    )

    df_ledger["fail_tx_count"] = df_ledger["tx_count"] - df_ledger["ok_tx_count"]
    df_ledger["path_match_rate"] = df_ledger.apply(
        lambda x: (float(x["path_match_count"]) / float(x["ok_tx_count"])) if x["ok_tx_count"] > 0 else None,
        axis=1,
    )

    def _safe_pct(num_col: str, den_col: str) -> pd.Series:
        out = []
        for _, r in df_ledger.iterrows():
            den = r[den_col]
            num = r[num_col]
            if den is None or den == 0:
                out.append(None)
            else:
                out.append((float(num) / float(den)) * 100.0)
        return pd.Series(out)

    df_ledger["in_gap_pct_vs_real"] = _safe_pct("diff_in_sum", "real_in_sum")
    df_ledger["out_gap_pct_vs_real"] = _safe_pct("diff_out_sum", "real_out_sum")
    df_ledger["real_amm_in_pct"] = _safe_pct("real_amm_in_sum", "real_in_sum")
    df_ledger["real_clob_in_pct"] = _safe_pct("real_clob_in_sum", "real_in_sum")
    df_ledger["model_amm_in_pct"] = _safe_pct("model_amm_in_sum", "model_in_sum")
    df_ledger["model_clob_in_pct"] = _safe_pct("model_clob_in_sum", "model_in_sum")

    def _safe_div(num_col: str, den_col: str) -> pd.Series:
        out = []
        for _, r in df_ledger.iterrows():
            den = r[den_col]
            num = r[num_col]
            if den is None or den == 0:
                out.append(None)
            else:
                out.append(float(num) / float(den))
        return pd.Series(out)

    df_ledger["real_avg_quality_from_sums"] = _safe_div("real_out_sum", "real_in_sum")
    df_ledger["model_avg_quality_from_sums"] = _safe_div("model_out_sum", "model_in_sum")
    df_ledger["avg_quality_diff_from_sums"] = df_ledger.apply(
        lambda x: (
            float(x["model_avg_quality_from_sums"] - x["real_avg_quality_from_sums"])
            if x["model_avg_quality_from_sums"] is not None and x["real_avg_quality_from_sums"] is not None
            else None
        ),
        axis=1,
    )

    col_order = [
        "ledger_index",
        "tx_count",
        "ok_tx_count",
        "fail_tx_count",
        "warning_tx_count",
        "path_match_count",
        "path_match_rate",
        "real_in_sum",
        "model_in_sum",
        "diff_in_sum",
        "in_gap_pct_vs_real",
        "real_out_sum",
        "model_out_sum",
        "diff_out_sum",
        "out_gap_pct_vs_real",
        "real_avg_quality_mean_tx",
        "model_avg_quality_mean_tx",
        "avg_quality_diff_mean_tx",
        "real_avg_quality_from_sums",
        "model_avg_quality_from_sums",
        "avg_quality_diff_from_sums",
        "real_amm_in_sum",
        "real_clob_in_sum",
        "model_amm_in_sum",
        "model_clob_in_sum",
        "real_amm_in_pct",
        "real_clob_in_pct",
        "model_amm_in_pct",
        "model_clob_in_pct",
        "real_segment_count_sum",
        "model_segment_count_sum",
        "model_overlap_with_real_offer_count_sum",
        "in_gap_pct_mean",
        "out_gap_pct_mean",
    ]
    df_ledger = df_ledger[col_order].sort_values(["ledger_index"]).reset_index(drop=True)
    return df_ledger


def _agg_window(
    df_tx: pd.DataFrame,
    args: argparse.Namespace,
    compare_path: Path,
    slices_path: Path,
    strict_metadata_only: bool = True,
) -> Dict[str, Any]:
    if df_tx.empty:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pair": args.pair,
            "state_mode": args.state_mode,
            "strict_metadata_only": bool(strict_metadata_only),
            "ledger_start": int(args.ledger_start),
            "ledger_end": int(args.ledger_end),
            "tx_count": 0,
            "ok_tx_count": 0,
            "fail_tx_count": 0,
            "warning_tx_count": 0,
            "inputs": {
                "amm_swaps": str(Path(args.amm_swaps).resolve()),
                "amm_fees": str(Path(args.amm_fees).resolve()),
                "clob_legs": str(Path(args.clob_legs).resolve()),
                "tx_prebook_snapshots": str(Path(args.tx_prebook_snapshots).resolve()),
                "metadata_ndjson": str(Path(args.metadata_ndjson).resolve()),
                "rolling_compare_parquet": str(compare_path),
                "rolling_slices_parquet": str(slices_path),
            },
        }

    tx_count = int(len(df_tx))
    ok_tx_count = int(df_tx["is_ok"].sum())
    fail_tx_count = int(tx_count - ok_tx_count)
    warning_tx_count = int(df_tx["has_warning"].sum())

    real_in_sum = float(df_tx["real_in"].sum())
    model_in_sum = float(df_tx["model_in"].sum())
    diff_in_sum = float(df_tx["diff_in"].sum())
    real_out_sum = float(df_tx["real_out"].sum())
    model_out_sum = float(df_tx["model_out"].sum())
    diff_out_sum = float(df_tx["diff_out"].sum())

    path_match_count = int(df_tx["path_match"].sum())
    path_match_rate = (float(path_match_count) / float(ok_tx_count)) if ok_tx_count > 0 else None

    def _safe_ratio(num: float, den: float) -> Optional[float]:
        if den == 0:
            return None
        return (num / den) * 100.0

    def _safe_div(num: float, den: float) -> Optional[float]:
        if den == 0:
            return None
        return num / den

    real_avg_quality_from_sums = _safe_div(real_out_sum, real_in_sum)
    model_avg_quality_from_sums = _safe_div(model_out_sum, model_in_sum)
    avg_quality_diff_from_sums = (
        model_avg_quality_from_sums - real_avg_quality_from_sums
        if model_avg_quality_from_sums is not None and real_avg_quality_from_sums is not None
        else None
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair": args.pair,
        "state_mode": args.state_mode,
        "strict_metadata_only": bool(strict_metadata_only),
        "ledger_start": int(args.ledger_start),
        "ledger_end": int(args.ledger_end),
        "tx_count": tx_count,
        "ok_tx_count": ok_tx_count,
        "fail_tx_count": fail_tx_count,
        "warning_tx_count": warning_tx_count,
        "path_match_count": path_match_count,
        "path_match_rate": path_match_rate,
        "real_in_sum": real_in_sum,
        "model_in_sum": model_in_sum,
        "diff_in_sum": diff_in_sum,
        "in_gap_pct_vs_real": _safe_ratio(diff_in_sum, real_in_sum),
        "real_out_sum": real_out_sum,
        "model_out_sum": model_out_sum,
        "diff_out_sum": diff_out_sum,
        "out_gap_pct_vs_real": _safe_ratio(diff_out_sum, real_out_sum),
        "real_avg_quality_mean_tx": float(df_tx["real_avg_quality"].mean()) if not df_tx["real_avg_quality"].dropna().empty else None,
        "model_avg_quality_mean_tx": float(df_tx["model_avg_quality"].mean()) if not df_tx["model_avg_quality"].dropna().empty else None,
        "avg_quality_diff_mean_tx": float(df_tx["avg_quality_diff"].mean()) if not df_tx["avg_quality_diff"].dropna().empty else None,
        "real_avg_quality_from_sums": real_avg_quality_from_sums,
        "model_avg_quality_from_sums": model_avg_quality_from_sums,
        "avg_quality_diff_from_sums": avg_quality_diff_from_sums,
        "real_amm_in_sum": float(df_tx["real_amm_in"].sum()),
        "real_clob_in_sum": float(df_tx["real_clob_in"].sum()),
        "model_amm_in_sum": float(df_tx["model_amm_in"].sum()),
        "model_clob_in_sum": float(df_tx["model_clob_in"].sum()),
        "real_amm_in_pct": _safe_ratio(float(df_tx["real_amm_in"].sum()), real_in_sum),
        "real_clob_in_pct": _safe_ratio(float(df_tx["real_clob_in"].sum()), real_in_sum),
        "model_amm_in_pct": _safe_ratio(float(df_tx["model_amm_in"].sum()), model_in_sum),
        "model_clob_in_pct": _safe_ratio(float(df_tx["model_clob_in"].sum()), model_in_sum),
        "real_segment_count_sum": int(df_tx["real_segment_count"].sum()),
        "model_segment_count_sum": int(df_tx["model_segment_count"].sum()),
        "model_overlap_with_real_offer_count_sum": int(df_tx["model_overlap_with_real_offer_count"].sum()),
        "avg_in_gap_pct_vs_real": float(df_tx["in_gap_pct_vs_real"].mean()) if not df_tx["in_gap_pct_vs_real"].dropna().empty else None,
        "avg_out_gap_pct_vs_real": float(df_tx["out_gap_pct_vs_real"].mean()) if not df_tx["out_gap_pct_vs_real"].dropna().empty else None,
        "inputs": {
            "amm_swaps": str(Path(args.amm_swaps).resolve()),
            "amm_fees": str(Path(args.amm_fees).resolve()),
            "clob_legs": str(Path(args.clob_legs).resolve()),
            "tx_prebook_snapshots": str(Path(args.tx_prebook_snapshots).resolve()),
            "metadata_ndjson": str(Path(args.metadata_ndjson).resolve()),
            "rolling_compare_parquet": str(compare_path),
            "rolling_slices_parquet": str(slices_path),
        },
    }
    return summary


def main() -> None:
    t_start_all = time.perf_counter()
    args = _parse_args()
    strict_metadata_only = True
    print(f"[mode] strict_metadata_only={strict_metadata_only}")

    ledger_start = int(args.ledger_start)
    ledger_end = int(args.ledger_end)
    if ledger_end < ledger_start:
        raise ValueError("--ledger-end must be >= --ledger-start")
    metadata_path = Path(args.metadata_ndjson).resolve()
    real_metrics_cache_path = _resolve_real_metrics_cache_path(
        args=args,
        metadata_ndjson=metadata_path,
        ledger_start=ledger_start,
        ledger_end=ledger_end,
    )

    rolling_out_dir = (
        Path(args.rolling_output_dir).resolve()
        if args.rolling_output_dir
        else Path("artifacts") / "compare" / str(args.pair) / f"ledger_{ledger_start}_{ledger_end}"
    )

    rolling_out_name = (
        args.rolling_output_name
        if args.rolling_output_name
        else f"compare_rolling_{args.pair}_{args.state_mode}_ledger_{ledger_start}_{ledger_end}_v1.parquet"
    )

    _ensure_exists(Path(args.amm_swaps).resolve(), "amm_swaps parquet")
    _ensure_exists(Path(args.amm_fees).resolve(), "amm_fees parquet")
    _ensure_exists(Path(args.clob_legs).resolve(), "clob legs parquet")
    _ensure_exists(metadata_path, "metadata ndjson")
    _ensure_exists(Path(args.tx_prebook_snapshots).resolve(), "tx prebook snapshots ndjson")

    t0 = time.perf_counter()
    compare_path, slices_path = _run_rolling_compare(args, rolling_out_dir=rolling_out_dir, rolling_out_name=rolling_out_name)
    print(f"[time] rolling ready in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    print(f"[load] compare parquet: {compare_path}")
    df_compare = pd.read_parquet(compare_path)
    if df_compare.empty:
        raise RuntimeError("Rolling compare output is empty; nothing to aggregate")

    df_compare = df_compare.copy()
    df_compare["transaction_hash"] = df_compare["transaction_hash"].astype(str).str.upper()
    df_compare = df_compare[
        (df_compare["ledger_index"] >= ledger_start) & (df_compare["ledger_index"] <= ledger_end)
    ].copy()

    if df_compare.empty:
        raise RuntimeError("No compare rows in requested ledger range after filtering")

    if not slices_path.exists():
        raise RuntimeError(
            "Strict mode violated: rolling slices parquet missing. "
            f"expected={slices_path}"
        )
    print(f"[load] slices parquet : {slices_path}")
    df_slices = pd.read_parquet(slices_path)
    print(f"[time] load compare+slices in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    tx_hashes_all = {
        str(x).upper()
        for x in df_compare["transaction_hash"].astype(str).tolist()
        if str(x).strip() != ""
    }
    tx_dir_map: Dict[str, Tuple[str, str]] = {}
    direction_parse_fail: List[str] = []
    for txh_raw, direction_raw in df_compare[["transaction_hash", "direction"]].itertuples(index=False, name=None):
        txh = str(txh_raw or "").upper()
        if not txh:
            continue
        direction = str(direction_raw or "")
        try:
            tx_dir_map[txh] = _direction_to_currencies(direction)
        except Exception:
            direction_parse_fail.append(txh)
            continue
    if direction_parse_fail:
        sample = ", ".join(sorted(set(direction_parse_fail))[:8])
        raise RuntimeError(
            "Strict metadata-only violated: direction parse failed for compare rows, "
            f"cannot reconstruct metadata path. failed_tx={len(set(direction_parse_fail))}; sample=[{sample}]"
        )
    missing_dir = sorted(tx_hashes_all - set(tx_dir_map.keys()))
    if missing_dir:
        sample = ", ".join(missing_dir[:8])
        raise RuntimeError(
            "Strict metadata-only violated: missing tx direction mapping for compare rows. "
            f"missing_tx={len(missing_dir)}; sample=[{sample}]"
        )
    print(f"[time] build tx direction map in {time.perf_counter() - t0:.2f}s | tx={len(tx_dir_map)}")

    t0 = time.perf_counter()
    print(f"[build] model metrics map from slices: tx={df_slices['transaction_hash'].nunique() if not df_slices.empty else 0}")
    model_map, model_segments_map = _build_model_maps(df_slices)
    print(f"[time] build model metrics map in {time.perf_counter() - t0:.2f}s | tx={len(model_map)}")

    print(f"[cache] real metrics cache: {real_metrics_cache_path}")
    t0 = time.perf_counter()
    cached_real_map, cached_real_segments_map, cache_info = _load_real_metrics_cache(
        cache_path=real_metrics_cache_path,
        metadata_ndjson=metadata_path,
        wanted_tx=set(tx_dir_map.keys()),
    )
    print(
        f"[cache] status={cache_info.get('status')} reason={cache_info.get('reason')} "
        f"loaded_tx={cache_info.get('loaded_tx')}"
    )

    missing_tx_dir_map = {txh: tx_dir_map[txh] for txh in tx_dir_map.keys() if txh not in cached_real_map}
    real_map = dict(cached_real_map)
    real_segments_map = dict(cached_real_segments_map)
    if missing_tx_dir_map:
        print(f"[build] real metrics map from metadata: miss tx={len(missing_tx_dir_map)}")
        t_build_real = time.perf_counter()
        built_real, built_real_segments = _build_real_metrics_map(metadata_path, tx_dir_map=missing_tx_dir_map)
        print(f"[time] metadata scan+build in {time.perf_counter() - t_build_real:.2f}s")
        real_map.update(built_real)
        real_segments_map.update(built_real_segments)
        still_missing_after_scan = sorted(set(missing_tx_dir_map.keys()) - set(real_map.keys()))
        if still_missing_after_scan:
            sample = ", ".join(still_missing_after_scan[:8])
            raise RuntimeError(
                "Strict metadata-only violated: missing metadata-derived real metrics after metadata scan. "
                f"missing_tx={len(still_missing_after_scan)}; sample=[{sample}]"
            )
        t_write_cache = time.perf_counter()
        _write_real_metrics_cache(
            cache_path=real_metrics_cache_path,
            metadata_ndjson=metadata_path,
            pair=str(args.pair),
            ledger_start=ledger_start,
            ledger_end=ledger_end,
            real_map=real_map,
            real_segments_map=real_segments_map,
        )
        print(f"[cache] refreshed cache: tx={len(real_map)} | write {time.perf_counter() - t_write_cache:.2f}s")
    else:
        print("[build] real metrics map fully served from cache")
    print(f"[time] real metrics ready in {time.perf_counter() - t0:.2f}s | tx={len(real_map)}")

    missing_real = sorted(set(tx_dir_map.keys()) - set(real_map.keys()))
    if missing_real:
        sample = ", ".join(missing_real[:8])
        raise RuntimeError(
            "Strict metadata-only violated: metadata-derived real metrics missing for compare rows. "
            f"missing_tx={len(missing_real)}; sample=[{sample}]"
        )
    for txh in tx_dir_map.keys():
        real_segments_map.setdefault(txh, [])
        model_segments_map.setdefault(txh, [])

    t0 = time.perf_counter()
    print("[build] tx metrics table")
    df_tx = _build_tx_metrics(
        df_compare=df_compare,
        model_map=model_map,
        real_map=real_map,
        model_segments_map=model_segments_map,
        real_segments_map=real_segments_map,
    )
    print(f"[time] build tx metrics in {time.perf_counter() - t0:.2f}s | rows={len(df_tx)}")

    t0 = time.perf_counter()
    print("[build] ledger metrics table")
    df_ledger = _agg_ledger(df_tx)
    print(f"[time] build ledger metrics in {time.perf_counter() - t0:.2f}s | rows={len(df_ledger)}")

    t0 = time.perf_counter()
    print("[build] window metrics summary")
    window_summary = _agg_window(
        df_tx,
        args=args,
        compare_path=compare_path,
        slices_path=slices_path,
        strict_metadata_only=strict_metadata_only,
    )
    print(f"[time] build window summary in {time.perf_counter() - t0:.2f}s")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else rolling_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ndjson_name = (
        args.output_ndjson_name
        if args.output_ndjson_name
        else f"metrics_{args.pair}_{args.state_mode}_ledger_{ledger_start}_{ledger_end}_v1.ndjson"
    )
    ndjson_path = out_dir / ndjson_name

    ndjson_rows: List[Dict[str, Any]] = []
    ndjson_rows.append({"scope": "window", **window_summary})
    for row in df_ledger.to_dict(orient="records"):
        ndjson_rows.append({"scope": "ledger", **row})
    for row in df_tx.to_dict(orient="records"):
        ndjson_rows.append({"scope": "tx", **row})

    tx_ctx_cols = ["transaction_hash", "ledger_index", "transaction_index", "direction", "used_prebook_ledger"]
    tx_ctx_map: Dict[str, Dict[str, Any]] = {}
    for row in df_tx[tx_ctx_cols].to_dict(orient="records"):
        tx_ctx_map[str(row["transaction_hash"]).upper()] = row

    for txh in df_tx["transaction_hash"].astype(str).str.upper().tolist():
        tx_ctx = tx_ctx_map.get(txh, {})
        for seg in real_segments_map.get(txh, []):
            ndjson_rows.append(
                {
                    "scope": "tx_segment",
                    "segment_view": "execution",
                    "side": "real",
                    "transaction_hash": txh,
                    "ledger_index": tx_ctx.get("ledger_index"),
                    "transaction_index": tx_ctx.get("transaction_index"),
                    "direction": tx_ctx.get("direction"),
                    "used_prebook_ledger": tx_ctx.get("used_prebook_ledger"),
                    "segment_no": int(_d(seg.get("segment_no"))),
                    "segment_kind": str(seg.get("segment_kind") or ""),
                    "node_type": seg.get("node_type"),
                    "source_id": seg.get("source_id"),
                    "step_idx": seg.get("step_idx"),
                    "out_take": _f(seg.get("out_take")),
                    "in_take": _f(seg.get("in_take")),
                    "avg_quality": _f(seg.get("avg_quality")),
                    "clob_offer_id": (_short_offer_id(seg.get("source_id")) if str(seg.get("segment_kind") or "").upper() == "CLOB" else None),
                }
            )
        for seg in model_segments_map.get(txh, []):
            ndjson_rows.append(
                {
                    "scope": "tx_segment",
                    "segment_view": "execution",
                    "side": "model",
                    "transaction_hash": txh,
                    "ledger_index": tx_ctx.get("ledger_index"),
                    "transaction_index": tx_ctx.get("transaction_index"),
                    "direction": tx_ctx.get("direction"),
                    "used_prebook_ledger": tx_ctx.get("used_prebook_ledger"),
                    "segment_no": int(_d(seg.get("segment_no"))),
                    "segment_kind": str(seg.get("segment_kind") or ""),
                    "node_type": seg.get("node_type"),
                    "source_id": seg.get("source_id"),
                    "step_idx": seg.get("step_idx"),
                    "out_take": _f(seg.get("out_take")),
                    "in_take": _f(seg.get("in_take")),
                    "avg_quality": _f(seg.get("avg_quality")),
                    "clob_offer_id": (_short_offer_id(seg.get("source_id")) if str(seg.get("segment_kind") or "").upper() == "CLOB" else None),
                }
            )

        real_display = _sort_segments_for_notebook_display(real_segments_map.get(txh, []))
        for seg in real_display:
            ndjson_rows.append(
                {
                    "scope": "tx_segment",
                    "segment_view": "notebook_display",
                    "side": "real",
                    "transaction_hash": txh,
                    "ledger_index": tx_ctx.get("ledger_index"),
                    "transaction_index": tx_ctx.get("transaction_index"),
                    "direction": tx_ctx.get("direction"),
                    "used_prebook_ledger": tx_ctx.get("used_prebook_ledger"),
                    "segment_no": int(_d(seg.get("segment_no"))),
                    "segment_kind": str(seg.get("segment_kind") or ""),
                    "node_type": seg.get("node_type"),
                    "source_id": seg.get("source_id"),
                    "step_idx": None,
                    "out_take": _f(seg.get("out_take")),
                    "in_take": _f(seg.get("in_take")),
                    "avg_quality": _f(seg.get("avg_quality")),
                    "clob_offer_id": seg.get("clob_offer_id"),
                }
            )

        model_display, model_premerge_display = _build_model_notebook_views(model_segments_map.get(txh, []))
        for seg in model_display:
            ndjson_rows.append(
                {
                    "scope": "tx_segment",
                    "segment_view": "notebook_display",
                    "side": "model",
                    "transaction_hash": txh,
                    "ledger_index": tx_ctx.get("ledger_index"),
                    "transaction_index": tx_ctx.get("transaction_index"),
                    "direction": tx_ctx.get("direction"),
                    "used_prebook_ledger": tx_ctx.get("used_prebook_ledger"),
                    "segment_no": int(_d(seg.get("segment_no"))),
                    "segment_kind": str(seg.get("segment_kind") or ""),
                    "node_type": seg.get("node_type"),
                    "source_id": seg.get("source_id"),
                    "step_idx": None,
                    "out_take": _f(seg.get("out_take")),
                    "in_take": _f(seg.get("in_take")),
                    "avg_quality": _f(seg.get("avg_quality")),
                    "clob_offer_id": seg.get("clob_offer_id"),
                }
            )

        if model_premerge_display is not None:
            for seg in model_premerge_display:
                ndjson_rows.append(
                    {
                        "scope": "tx_segment",
                        "segment_view": "notebook_premerge",
                        "side": "model",
                        "transaction_hash": txh,
                        "ledger_index": tx_ctx.get("ledger_index"),
                        "transaction_index": tx_ctx.get("transaction_index"),
                        "direction": tx_ctx.get("direction"),
                        "used_prebook_ledger": tx_ctx.get("used_prebook_ledger"),
                        "segment_no": int(_d(seg.get("segment_no"))),
                        "segment_kind": str(seg.get("segment_kind") or ""),
                        "node_type": seg.get("node_type"),
                        "source_id": seg.get("source_id"),
                        "step_idx": None,
                        "out_take": _f(seg.get("out_take")),
                        "in_take": _f(seg.get("in_take")),
                        "avg_quality": _f(seg.get("avg_quality")),
                        "clob_offer_id": seg.get("clob_offer_id"),
                    }
                )
    _write_ndjson(ndjson_path, ndjson_rows)

    parquet_outputs: Dict[str, str] = {}
    csv_outputs: Dict[str, str] = {}

    tx_name = (
        args.tx_metrics_name
        if args.tx_metrics_name
        else f"tx_metrics_{args.pair}_{args.state_mode}_ledger_{ledger_start}_{ledger_end}_v1.parquet"
    )
    ledger_name = (
        args.ledger_metrics_name
        if args.ledger_metrics_name
        else f"ledger_metrics_{args.pair}_{args.state_mode}_ledger_{ledger_start}_{ledger_end}_v1.parquet"
    )
    window_name = (
        args.window_metrics_name
        if args.window_metrics_name
        else f"window_metrics_{args.pair}_{args.state_mode}_ledger_{ledger_start}_{ledger_end}_v1.json"
    )

    write_parquet_side_outputs = bool(args.also_write_parquet or args.also_write_csv)
    if write_parquet_side_outputs:
        tx_path = out_dir / tx_name
        ledger_path = out_dir / ledger_name
        window_path = out_dir / window_name

        df_tx.to_parquet(tx_path, index=False)
        df_ledger.to_parquet(ledger_path, index=False)
        window_path.write_text(json.dumps(window_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        parquet_outputs = {
            "tx_metrics_output": str(tx_path),
            "ledger_metrics_output": str(ledger_path),
            "window_metrics_output": str(window_path),
        }

        if args.also_write_csv and (not args.no_csv):
            tx_csv = tx_path.with_suffix(".csv")
            ledger_csv = ledger_path.with_suffix(".csv")
            df_tx.to_csv(tx_csv, index=False)
            df_ledger.to_csv(ledger_csv, index=False)
            csv_outputs = {
                "tx_metrics_csv_output": str(tx_csv),
                "ledger_metrics_csv_output": str(ledger_csv),
            }

    manifest_path: Optional[Path] = None
    if args.write_manifest:
        manifest = {
            "script": "empirical_compare_two_week_metrics.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pair": args.pair,
            "state_mode": args.state_mode,
            "strict_metadata_only": bool(strict_metadata_only),
            "window": {"ledger_start": ledger_start, "ledger_end": ledger_end},
            "rolling_compare_output": str(compare_path),
            "rolling_slices_output": str(slices_path),
            "real_metrics_cache": str(real_metrics_cache_path),
            "cache": {
                "status": cache_info.get("status"),
                "reason": cache_info.get("reason"),
                "loaded_tx": int(cache_info.get("loaded_tx") or 0),
                "metadata_scan_tx": int(len(missing_tx_dir_map)),
            },
            "unified_ndjson_output": str(ndjson_path),
            **parquet_outputs,
            **csv_outputs,
            "counts": {
                "tx": int(len(df_tx)),
                "ledger": int(len(df_ledger)),
                "window": 1,
                "tx_segment_real": int(sum(len(v) for v in real_segments_map.values())),
                "tx_segment_model": int(sum(len(v) for v in model_segments_map.values())),
                "tx_segment_notebook_display_real": int(sum(len(_sort_segments_for_notebook_display(v)) for v in real_segments_map.values())),
                "tx_segment_notebook_display_model": int(sum(len(_build_model_notebook_views(v)[0]) for v in model_segments_map.values())),
                "tx_segment_notebook_premerge_model": int(
                    sum((len(pm) if pm is not None else 0) for _, pm in (_build_model_notebook_views(v) for v in model_segments_map.values()))
                ),
                "metadata_matched_tx": int(len(real_map)),
                "model_slices_tx": int(len(model_map)),
            },
        }
        manifest_path = out_dir / "metrics_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[OK] outputs written")
    print(f"metrics ndjson: {ndjson_path}")
    if parquet_outputs:
        print(f"tx metrics    : {parquet_outputs['tx_metrics_output']}")
        print(f"ledger metrics: {parquet_outputs['ledger_metrics_output']}")
        print(f"window metrics: {parquet_outputs['window_metrics_output']}")
    if csv_outputs:
        print(f"tx metrics csv: {csv_outputs['tx_metrics_csv_output']}")
        print(f"ledger csv    : {csv_outputs['ledger_metrics_csv_output']}")
    if manifest_path is not None:
        print(f"manifest      : {manifest_path}")
    print(f"[time] total elapsed: {time.perf_counter() - t_start_all:.2f}s")


if __name__ == "__main__":
    main()
