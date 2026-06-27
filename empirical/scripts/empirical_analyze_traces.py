#!/usr/bin/env python3
# analyse_traces.py
#
# Read tx_traces.ndjson produced by compare_rolling.py and compute statistics
# describing divergences between MODEL trace and REAL (AMM summary + CLOB legs).
#
# Also optionally loads CLOB book snapshots (getsXRP / getsrUSD) and compares
# REAL legs and MODEL CLOB slices to the snapshot tier ladder around the tx's
# book_snapshot_ledger (L-1 or fallback), to quantify how "explainable" each is
# by the visible book.
#
# Usage:
#   python3 analyse_traces.py \
#     --traces rlusd_xrp_100891230_100913230_1d/tx_traces.ndjson \
#     --book-getsxrp rlusd_xrp_100891230_100913230_1d/book_rusd_xrp_getsXRP.ndjson \
#     --book-getsrusd rlusd_xrp_100891230_100913230_1d/book_rusd_xrp_getsrUSD.ndjson
#
# Optional:
#   --max-rows 200000
#   --topn 200
#   --no-snapshot
#
# Output: prints a compact report to stdout.

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple

getcontext().prec = 50

RUSD_HEX = "524C555344000000000000000000000000000000"
XRP = "XRP"

XRP_QUANTUM = Decimal("0.000001")
IOU_QUANTUM = Decimal("0.000000000000001")  # 1e-15


def D(x: Any) -> Decimal:
    if x is None:
        return Decimal("0")
    return Decimal(str(x))


def safe_div(a: Decimal, b: Decimal) -> Optional[Decimal]:
    if b == 0:
        return None
    return a / b


def is_xrp(cur: str) -> bool:
    return cur == XRP


def pretty_cur(c: str) -> str:
    return "rUSD" if c == RUSD_HEX else c


def quantize_down(v: Decimal, q: Decimal) -> Decimal:
    if q == 0:
        return v
    return (v // q) * q


def quantize_up(v: Decimal, q: Decimal) -> Decimal:
    if q == 0:
        return v
    n = (v / q).to_integral_value(rounding="ROUND_CEILING")
    return n * q


def pct(x: Decimal) -> str:
    return f"{(x * Decimal('100')):.4f}%"


def fmt(x: Optional[Decimal], dp: int = 10) -> str:
    if x is None:
        return "None"
    q = Decimal(1).scaleb(-dp)
    try:
        return str(x.quantize(q))
    except Exception:
        return str(x)


def percentile(sorted_vals: List[Decimal], p: float) -> Optional[Decimal]:
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * Decimal(str(c - k))
    d1 = sorted_vals[c] * Decimal(str(k - f))
    return d0 + d1


def summarise_dist(vals: List[Decimal], name: str, dp: int = 10) -> str:
    if not vals:
        return f"{name}: (n=0)"
    s = sorted(vals)
    n = len(s)
    mean = sum(s, start=Decimal("0")) / Decimal(n)
    p50 = percentile(s, 50)
    p90 = percentile(s, 90)
    p95 = percentile(s, 95)
    p99 = percentile(s, 99)
    p999 = percentile(s, 99.9)
    mn = s[0]
    mx = s[-1]
    return (
        f"{name}: n={n} mean={fmt(mean,dp)} "
        f"p50={fmt(p50,dp)} p90={fmt(p90,dp)} p95={fmt(p95,dp)} "
        f"p99={fmt(p99,dp)} p99.9={fmt(p999,dp)} min={fmt(mn,dp)} max={fmt(mx,dp)}"
    )


def parse_offer_amount(a: Any) -> Tuple[str, Optional[str], Decimal]:
    # XRP amounts often come as string drops (int), IOU as dict {currency,issuer,value}
    if isinstance(a, str):
        # drops -> XRP
        return (XRP, None, Decimal(a) * XRP_QUANTUM)
    if isinstance(a, dict):
        return (a.get("currency"), a.get("issuer"), Decimal(str(a.get("value"))))
    raise ValueError(f"Bad amount: {a!r}")


@dataclass
class Tier:
    px_in_per_out: Decimal  # input per 1 output
    out_max: Decimal
    in_at_out_max: Decimal


def build_tiers_from_snapshot_offers(
    offers: List[Dict[str, Any]],
    in_cur: str,
    out_cur: str,
    topn: int,
) -> List[Tier]:
    tiers: List[Tier] = []

    out_q = XRP_QUANTUM if is_xrp(out_cur) else IOU_QUANTUM
    in_q = XRP_QUANTUM if is_xrp(in_cur) else IOU_QUANTUM

    for o in offers:
        try:
            gets_cur, _, gets_val = parse_offer_amount(o.get("TakerGets"))
            pays_cur, _, pays_val = parse_offer_amount(o.get("TakerPays"))
        except Exception:
            continue

        # funded overrides if present
        if "taker_gets_funded" in o:
            try:
                fg_cur, _, fg_val = parse_offer_amount(o.get("taker_gets_funded"))
                if fg_cur == gets_cur:
                    gets_val = fg_val
            except Exception:
                pass
        if "taker_pays_funded" in o:
            try:
                fp_cur, _, fp_val = parse_offer_amount(o.get("taker_pays_funded"))
                if fp_cur == pays_cur:
                    pays_val = fp_val
            except Exception:
                pass

        # align to direction in_cur -> out_cur
        if gets_cur == out_cur and pays_cur == in_cur:
            out_max = gets_val
            in_need = pays_val
        elif pays_cur == out_cur and gets_cur == in_cur:
            out_max = pays_val
            in_need = gets_val
        else:
            continue

        if out_max <= 0 or in_need <= 0:
            continue

        out_max = quantize_down(out_max, out_q)
        in_need = quantize_up(in_need, in_q)

        if out_max <= 0 or in_need <= 0:
            continue

        px = in_need / out_max
        tiers.append(Tier(px_in_per_out=px, out_max=out_max, in_at_out_max=in_need))

    tiers.sort(key=lambda t: t.px_in_per_out)  # ascending = cheaper
    if topn and len(tiers) > topn:
        tiers = tiers[:topn]
    return tiers


def nearest_tier_px_err(px: Decimal, tiers: List[Tier]) -> Optional[Decimal]:
    if not tiers:
        return None
    # tiers sorted by px
    lo, hi = 0, len(tiers) - 1
    best = None
    best_err = None
    while lo <= hi:
        mid = (lo + hi) // 2
        mpx = tiers[mid].px_in_per_out
        err = abs(mpx - px)
        if best_err is None or err < best_err:
            best_err = err
            best = mid
        if mpx < px:
            lo = mid + 1
        elif mpx > px:
            hi = mid - 1
        else:
            return Decimal("0")
    return best_err


def load_book_ndjson(path: str) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            li = obj.get("ledger_index")
            if li is None:
                li = obj.get("ledger") or obj.get("ledger_current_index")
            if li is None:
                continue
            offers = obj.get("offers")
            if offers is None:
                offers = obj.get("result", {}).get("offers", [])
            out[int(li)] = offers or []
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="tx_traces.ndjson")
    ap.add_argument("--book-getsxrp", required=False, help="book_rusd_xrp_getsXRP.ndjson")
    ap.add_argument("--book-getsrusd", required=False, help="book_rusd_xrp_getsrUSD.ndjson")
    ap.add_argument("--topn", type=int, default=200, help="top N tiers to build per snapshot")
    ap.add_argument("--max-rows", type=int, default=0, help="stop after N rows (0=all)")
    ap.add_argument("--no-snapshot", action="store_true", help="skip snapshot-tier comparison")
    # New CLI arguments for tolerant same-path evaluation
    ap.add_argument("--dust-out-min", type=float, default=1e-9, help="ignore MODEL CLOB slices with out_take < dust_out_min for path/tier comparison")
    ap.add_argument("--tier-eps", type=float, default=5e-7, help="max abs px diff to match as fallback if tier mapping fails")
    ap.add_argument("--tier-f1-min", type=float, default=0.9, help="min F1 score between REAL/MODEL tier multisets for approx same")
    ap.add_argument("--tier-vol-rel", type=float, default=0.01, help="relative tolerance for per-tier volume comparison (1%%)")
    ap.add_argument("--tier-vol-abs", type=float, default=1e-9, help="absolute tolerance for per-tier volume comparison")
    ap.add_argument("--same-examples", type=int, default=10, help="number of example tx hashes to print for matches/mismatches")
    args = ap.parse_args()

    # Convert CLI floats to Decimal once
    dust_out_min = Decimal(str(args.dust_out_min))
    eps_px = Decimal(str(args.tier_eps))
    tier_f1_min = Decimal(str(args.tier_f1_min))
    tier_vol_rel = Decimal(str(args.tier_vol_rel))
    tier_vol_abs = Decimal(str(args.tier_vol_abs))
    same_examples_n = args.same_examples

    # Load books (optional)
    book_xrp = {}
    book_rusd = {}
    if not args.no_snapshot:
        if not (args.book_getsxrp and args.book_getsrusd):
            raise SystemExit("Need both --book-getsxrp and --book-getsrusd unless --no-snapshot")
        book_xrp = load_book_ndjson(args.book_getsxrp)
        book_rusd = load_book_ndjson(args.book_getsrusd)

    # Accumulators
    diffs_in = {"rUSD->XRP": [], "XRP->rUSD": [], "ALL": []}
    diffs_out = {"rUSD->XRP": [], "XRP->rUSD": [], "ALL": []}
    rel_in = {"rUSD->XRP": [], "XRP->rUSD": [], "ALL": []}

    # Path structure stats
    switches = {"rUSD->XRP": [], "XRP->rUSD": [], "ALL": []}
    segs_total = {"rUSD->XRP": [], "XRP->rUSD": [], "ALL": []}
    clob_slices_n = {"rUSD->XRP": [], "XRP->rUSD": [], "ALL": []}
    amm_slices_n = {"rUSD->XRP": [], "XRP->rUSD": [], "ALL": []}
    real_legs_n = {"rUSD->XRP": [], "XRP->rUSD": [], "ALL": []}

    # Price stats
    real_amm_px = {"rUSD->XRP": [], "XRP->rUSD": [], "ALL": []}
    model_amm_px = {"rUSD->XRP": [], "XRP->rUSD": [], "ALL": []}
    real_leg_px = {"rUSD->XRP": [], "XRP->rUSD": [], "ALL": []}
    model_clob_slice_px = {"rUSD->XRP": [], "XRP->rUSD": [], "ALL": []}

    # Snapshot alignment errors
    real_leg_px_err = {"rUSD->XRP": [], "XRP->rUSD": [], "ALL": []}
    model_clob_px_err = {"rUSD->XRP": [], "XRP->rUSD": [], "ALL": []}
    snap_missing = Counter()

    # Presence signature mismatch
    sig_mismatch = Counter()
    sig_counts = Counter()

    # Helper
    def add_all(dct: Dict[str, List[Decimal]], k: str, v: Decimal):
        dct[k].append(v)
        dct["ALL"].append(v)

    def dir_key(rec: Dict[str, Any]) -> str:
        return rec.get("direction", "UNK")

    # -- Helper functions for tolerant same-path evaluation --
    def f1_score_multiset(a: Counter, b: Counter) -> float:
        keys = set(a.keys()) | set(b.keys())
        tp = sum(min(a[k], b[k]) for k in keys)
        pa = sum(a.values())
        pb = sum(b.values())
        precision = tp / pb if pb else 0.0
        recall = tp / pa if pa else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def map_px_to_tier_index(px: Decimal, tiers: List[Tier]) -> Optional[int]:
        if not tiers:
            return None
        # Binary search for nearest px, return argmin
        lo, hi = 0, len(tiers) - 1
        best = None
        best_err = None
        while lo <= hi:
            mid = (lo + hi) // 2
            mpx = tiers[mid].px_in_per_out
            err = abs(mpx - px)
            if best_err is None or err < best_err:
                best_err = err
                best = mid
            if mpx < px:
                lo = mid + 1
            elif mpx > px:
                hi = mid - 1
            else:
                return mid
        return best

    def map_items_to_tier_multiset(pxs: List[Decimal], tiers: List[Tier], eps_px: Decimal) -> Tuple[Counter, int]:
        ctr = Counter()
        unmapped = 0
        for px in pxs:
            idx = map_px_to_tier_index(px, tiers)
            if idx is not None:
                abs_err = abs(tiers[idx].px_in_per_out - px)
                if abs_err <= eps_px:
                    ctr[idx] += 1
                else:
                    unmapped += 1
            else:
                unmapped += 1
        return ctr, unmapped

    # New accumulators for tolerant same-path stats
    same_presence = {"ALL": 0, "rUSD->XRP": 0, "XRP->rUSD": 0}
    same_approx_tiers = {"ALL": 0, "rUSD->XRP": 0, "XRP->rUSD": 0}
    same_approx_path = {"ALL": 0, "rUSD->XRP": 0, "XRP->rUSD": 0}
    same_examples = {"match": [], "mismatch": []}
    total_ok_by_dir = {"ALL": 0, "rUSD->XRP": 0, "XRP->rUSD": 0}

    # Read traces
    n = 0
    with open(args.traces, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1
            if args.max_rows and n > args.max_rows:
                break

            d = dir_key(rec)
            if d not in ("rUSD->XRP", "XRP->rUSD"):
                # keep but bucket as ALL only
                d = "ALL"

            if rec.get("error"):
                continue

            total_ok_by_dir[d] = total_ok_by_dir.get(d, 0) + 1
            total_ok_by_dir["ALL"] = total_ok_by_dir.get("ALL", 0) + 1

            # diffs
            diff = rec.get("diff") or {}
            din = diff.get("in")
            dout = diff.get("out")
            if din is not None:
                add_all(diffs_in, d, D(din))
            if dout is not None:
                add_all(diffs_out, d, D(dout))

            real_tot_in = D(((rec.get("real") or {}).get("tot") or {}).get("in"))
            model_tot_in = D(((rec.get("model") or {}).get("tot") or {}).get("in"))
            if real_tot_in != 0 and model_tot_in != 0:
                add_all(rel_in, d, (model_tot_in - real_tot_in) / real_tot_in)

            # path structure from model trace
            m = rec.get("model") or {}
            trace = m.get("trace") or []
            norm = []
            for s in trace:
                src = (s.get("src") or "").upper()
                if "AMM" in src:
                    norm.append("AMM")
                elif "CLOB" in src or "BOOK" in src or "ORDER" in src:
                    norm.append("CLOB")
            collapsed = []
            for x in norm:
                if not collapsed or collapsed[-1] != x:
                    collapsed.append(x)
            sw = max(0, len(collapsed) - 1)
            add_all(switches, d, Decimal(sw))
            add_all(segs_total, d, Decimal(len(collapsed)))
            add_all(clob_slices_n, d, Decimal(sum(1 for x in collapsed if x == "CLOB")))
            add_all(amm_slices_n, d, Decimal(sum(1 for x in collapsed if x == "AMM")))

            # real legs count
            real_clob = (rec.get("real") or {}).get("clob") or {}
            legs = real_clob.get("legs") or []
            add_all(real_legs_n, d, Decimal(len(legs)))

            # prices
            ramm = (rec.get("real") or {}).get("amm") or {}
            mamp = (rec.get("model") or {}).get("amm") or {}
            rpx = ramm.get("avg_px_in_per_out")
            mpx = mamp.get("avg_px_in_per_out")
            if rpx is not None:
                add_all(real_amm_px, d, D(rpx))
            if mpx is not None:
                add_all(model_amm_px, d, D(mpx))

            for lg in legs:
                px = lg.get("avg_px_in_per_out")
                if px is not None:
                    add_all(real_leg_px, d, D(px))

            for s in trace:
                src = (s.get("src") or "").upper()
                if "CLOB" in src or "BOOK" in src or "ORDER" in src:
                    px = s.get("avg_px_in_per_out")
                    if px is not None:
                        add_all(model_clob_slice_px, d, D(px))

            # presence signature mismatch (coarse but useful)
            real_sig = "AMM+CLOB" if len(legs) > 0 else "AMM"
            uses_clob_model = any(x == "CLOB" for x in collapsed)
            model_sig = "AMM+CLOB" if uses_clob_model else "AMM"
            sig_counts[(d, real_sig, model_sig)] += 1
            if real_sig != model_sig:
                sig_mismatch[(d, real_sig, model_sig)] += 1

            # Approx same-path tolerant evaluation (only if snapshot available)
            # Only if snapshot and tiers available
            if args.no_snapshot:
                continue

            snap_li = rec.get("book_snapshot_ledger")
            if snap_li is None:
                snap_missing[(d, "no_snapshot_ledger")] += 1
                continue
            snap_li = int(snap_li)

            # choose snapshot file based on direction (same convention as compare_rolling.py)
            in_cur = rec.get("in_currency")
            out_cur = rec.get("out_currency")
            # stored pretty names in traces; map back
            in_cur_raw = XRP if in_cur == "XRP" else RUSD_HEX
            out_cur_raw = XRP if out_cur == "XRP" else RUSD_HEX

            if in_cur_raw == XRP and out_cur_raw == RUSD_HEX:
                offers = book_rusd.get(snap_li)
            elif in_cur_raw == RUSD_HEX and out_cur_raw == XRP:
                offers = book_xrp.get(snap_li)
            else:
                offers = None

            if not offers:
                snap_missing[(d, "missing_offers_for_ledger")] += 1
                continue

            tiers = build_tiers_from_snapshot_offers(offers, in_cur_raw, out_cur_raw, topn=args.topn)
            if not tiers:
                snap_missing[(d, "no_tiers")] += 1
                continue

            # --- tolerant same-path logic ---
            # 1. Presence (REAL legs > 0 vs MODEL path contains CLOB)
            real_has_clob = len(legs) > 0
            model_has_clob = any(normalise_src((s.get("src") or "")) == "CLOB" for s in trace)
            if real_has_clob == model_has_clob:
                same_presence[d] += 1
                same_presence["ALL"] += 1

            # 2. Tolerant tier/path matching
            # -- Build REAL px and out_amt lists from legs
            real_px_list = []
            real_out_amt_list = []
            for lg in legs:
                px = lg.get("avg_px_in_per_out")
                out_amt = lg.get("out_amount") if "out_amount" in lg else lg.get("out") if "out" in lg else None
                if px is not None:
                    real_px_list.append(D(px))
                if out_amt is not None:
                    real_out_amt_list.append(D(out_amt))
            # -- Build MODEL CLOB px and out_take lists (dust filter)
            model_px_list = []
            model_out_amt_list = []
            for s in trace:
                if normalise_src((s.get("src") or "")) != "CLOB":
                    continue
                out_take = s.get("out_take")
                if out_take is None:
                    continue
                out_take_dec = D(out_take)
                if out_take_dec < dust_out_min:
                    continue
                px = s.get("avg_px_in_per_out")
                if px is not None:
                    model_px_list.append(D(px))
                    model_out_amt_list.append(out_take_dec)

            # Map to tier-index multisets
            real_ctr, real_unmapped = map_items_to_tier_multiset(real_px_list, tiers, eps_px)
            model_ctr, model_unmapped = map_items_to_tier_multiset(model_px_list, tiers, eps_px)
            f1 = f1_score_multiset(real_ctr, model_ctr)
            total_model = sum(model_ctr.values()) + model_unmapped
            # approx_same_tiers
            approx_same_tiers = (
                (Decimal(str(f1)) >= tier_f1_min)
                and (real_unmapped == 0)
                and (model_unmapped / max(1, total_model) <= Decimal("0.1"))
            )
            if approx_same_tiers:
                same_approx_tiers[d] += 1
                same_approx_tiers["ALL"] += 1

            # Per-tier volume comparison (if both sides have volumes)
            approx_same_vol = None
            if real_px_list and model_px_list and real_out_amt_list and model_out_amt_list:
                # Aggregate per tier: sum out volumes by tier for REAL and MODEL
                real_vols = defaultdict(Decimal)
                for i, px in enumerate(real_px_list):
                    idx = map_px_to_tier_index(px, tiers)
                    if idx is not None and abs(tiers[idx].px_in_per_out - px) <= eps_px:
                        real_vols[idx] += real_out_amt_list[i] if i < len(real_out_amt_list) else Decimal("0")
                model_vols = defaultdict(Decimal)
                for i, px in enumerate(model_px_list):
                    idx = map_px_to_tier_index(px, tiers)
                    if idx is not None and abs(tiers[idx].px_in_per_out - px) <= eps_px:
                        model_vols[idx] += model_out_amt_list[i] if i < len(model_out_amt_list) else Decimal("0")
                # For each tier present in either, compare
                all_tiers = set(real_vols.keys()) | set(model_vols.keys())
                all_pass = True
                for tidx in all_tiers:
                    a = real_vols.get(tidx, Decimal("0"))
                    b = model_vols.get(tidx, Decimal("0"))
                    mx = max(a, b)
                    if abs(a - b) > max(tier_vol_abs, tier_vol_rel * mx):
                        all_pass = False
                        break
                approx_same_vol = all_pass
            # else, if either side lacks volume data, approx_same_vol = None

            approx_same_path = approx_same_tiers and (approx_same_vol is True or approx_same_vol is None)
            if approx_same_path:
                same_approx_path[d] += 1
                same_approx_path["ALL"] += 1

            # Collect up to N examples for matches/mismatches
            # Each: ledger_index, transaction_index, tx_hash, direction, real_legs_count, model_collapsed, f1, real_unmapped, model_unmapped
            if len(same_examples["match"]) < same_examples_n and approx_same_path:
                same_examples["match"].append({
                    "ledger_index": rec.get("ledger_index"),
                    "transaction_index": rec.get("transaction_index"),
                    "tx_hash": rec.get("tx_hash"),
                    "direction": d,
                    "real_legs_count": len(legs),
                    "model_collapsed": " + ".join(collapsed),
                    "f1": float(f1),
                    "real_unmapped": real_unmapped,
                    "model_unmapped": model_unmapped,
                })
            if len(same_examples["mismatch"]) < same_examples_n and not approx_same_path:
                same_examples["mismatch"].append({
                    "ledger_index": rec.get("ledger_index"),
                    "transaction_index": rec.get("transaction_index"),
                    "tx_hash": rec.get("tx_hash"),
                    "direction": d,
                    "real_legs_count": len(legs),
                    "model_collapsed": " + ".join(collapsed),
                    "f1": float(f1),
                    "real_unmapped": real_unmapped,
                    "model_unmapped": model_unmapped,
                })

    # Report
    print("\n=== analyse_traces report ===")
    print(f"rows read: {n}")
    print("\n--- Amount diffs ---")
    print(summarise_dist(diffs_in["ALL"], "diff_in (model-real)", dp=10))
    print(summarise_dist(diffs_out["ALL"], "diff_out (model-real)", dp=10))
    print(summarise_dist(rel_in["ALL"], "rel_err_in", dp=10))

    for d in ("rUSD->XRP", "XRP->rUSD"):
        print(f"\n--- Amount diffs ({d}) ---")
        print(summarise_dist(diffs_in[d], "diff_in", dp=10))
        print(summarise_dist(diffs_out[d], "diff_out", dp=10))
        print(summarise_dist(rel_in[d], "rel_err_in", dp=10))

    print("\n--- Path structure (MODEL) ---")
    print(summarise_dist(switches["ALL"], "switches(collapsed)", dp=4))
    print(summarise_dist(segs_total["ALL"], "segments(collapsed)", dp=4))
    print(summarise_dist(amm_slices_n["ALL"], "AMM segments(collapsed)", dp=4))
    print(summarise_dist(clob_slices_n["ALL"], "CLOB segments(collapsed)", dp=4))
    print(summarise_dist(real_legs_n["ALL"], "REAL CLOB legs (count)", dp=4))

    for d in ("rUSD->XRP", "XRP->rUSD"):
        print(f"\n--- Path structure ({d}) ---")
        print(summarise_dist(switches[d], "switches", dp=4))
        print(summarise_dist(segs_total[d], "segments", dp=4))
        print(summarise_dist(amm_slices_n[d], "AMM segments", dp=4))
        print(summarise_dist(clob_slices_n[d], "CLOB segments", dp=4))
        print(summarise_dist(real_legs_n[d], "REAL CLOB legs", dp=4))

    print("\n--- Prices ---")
    print(summarise_dist(real_amm_px["ALL"], "REAL AMM avg_px (in/out)", dp=12))
    print(summarise_dist(model_amm_px["ALL"], "MODEL AMM avg_px (in/out)", dp=12))
    print(summarise_dist(real_leg_px["ALL"], "REAL CLOB leg avg_px", dp=12))
    print(summarise_dist(model_clob_slice_px["ALL"], "MODEL CLOB slice avg_px", dp=12))

    for d in ("rUSD->XRP", "XRP->rUSD"):
        print(f"\n--- Prices ({d}) ---")
        print(summarise_dist(real_amm_px[d], "REAL AMM avg_px", dp=12))
        print(summarise_dist(model_amm_px[d], "MODEL AMM avg_px", dp=12))
        print(summarise_dist(real_leg_px[d], "REAL CLOB leg avg_px", dp=12))
        print(summarise_dist(model_clob_slice_px[d], "MODEL CLOB slice avg_px", dp=12))

    print("\n--- Presence signature mismatches (coarse) ---")
    total_sig = sum(sig_counts.values())
    total_mis = sum(sig_mismatch.values())
    print(f"total comparable txs: {total_sig} | mismatches: {total_mis}")
    if total_sig:
        print(f"mismatch rate: {pct(Decimal(total_mis) / Decimal(total_sig))}")
    # top mismatch combos
    for (d, rs, ms), c in sig_mismatch.most_common(10):
        print(f"  {d}: REAL={rs} MODEL={ms} count={c}")

    if not args.no_snapshot:
        print("\n--- Snapshot tier alignment (px error to nearest tier) ---")
        print(summarise_dist(real_leg_px_err["ALL"], "REAL leg nearest-tier |px_err|", dp=12))
        print(summarise_dist(model_clob_px_err["ALL"], "MODEL CLOB slice nearest-tier |px_err|", dp=12))

        for d in ("rUSD->XRP", "XRP->rUSD"):
            print(f"\n--- Snapshot tier alignment ({d}) ---")
            print(summarise_dist(real_leg_px_err[d], "REAL leg |px_err|", dp=12))
            print(summarise_dist(model_clob_px_err[d], "MODEL CLOB slice |px_err|", dp=12))

        if snap_missing:
            print("\nSnapshot comparison missing reasons (top 10):")
            for (d, reason), c in snap_missing.most_common(10):
                print(f"  {d}: {reason} -> {c}")

        # --- Approx same-path tolerant tier matching report ---
        print("\n--- Approx same-path (tolerant tier matching) ---")
        print("Totals per direction (ok txs):")
        for k in ("rUSD->XRP", "XRP->rUSD", "ALL"):
            tot = total_ok_by_dir.get(k, 0)
            print(f"  {k}: total_ok={tot}")
        print("Presence signature (REAL legs>0 vs MODEL CLOB):")
        for k in ("rUSD->XRP", "XRP->rUSD", "ALL"):
            tot = total_ok_by_dir.get(k, 0)
            matched = same_presence.get(k, 0)
            rate = (Decimal(matched) / Decimal(tot)) if tot else Decimal("0")
            print(f"  {k}: {matched}/{tot}  ({pct(rate)})")
        print("Approx same tiers (F1>=thresh, all REAL mapped, <=10% MODEL unmapped):")
        for k in ("rUSD->XRP", "XRP->rUSD", "ALL"):
            tot = total_ok_by_dir.get(k, 0)
            matched = same_approx_tiers.get(k, 0)
            rate = (Decimal(matched) / Decimal(tot)) if tot else Decimal("0")
            print(f"  {k}: {matched}/{tot}  ({pct(rate)})")
        print("Approx same path (tiers + per-tier vol tolerance):")
        for k in ("rUSD->XRP", "XRP->rUSD", "ALL"):
            tot = total_ok_by_dir.get(k, 0)
            matched = same_approx_path.get(k, 0)
            rate = (Decimal(matched) / Decimal(tot)) if tot else Decimal("0")
            print(f"  {k}: {matched}/{tot}  ({pct(rate)})")
        # Print example matches and mismatches
        print(f"\nExample approx same-path matches (up to {same_examples_n}):")
        for ex in same_examples["match"]:
            print(json.dumps(ex))
        print(f"\nExample approx same-path mismatches (up to {same_examples_n}):")
        for ex in same_examples["mismatch"]:
            print(json.dumps(ex))

    print("\n[done]")


def normalise_src(src: str) -> str:
    s = (src or "").upper()
    if "AMM" in s:
        return "AMM"
    if "CLOB" in s or "BOOK" in s or "ORDER" in s:
        return "CLOB"
    return "?"

if __name__ == "__main__":
    main()
