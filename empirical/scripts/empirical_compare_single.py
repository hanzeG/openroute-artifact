# compare_single.py
# Compare model vs real execution for rUSD-XRP over a ledger window.
#
# Assumptions:
# - Pre CLOB snapshot for ledger L is book_offers from ledger (L-1),
#   but if (L-1) snapshot is missing, we fallback to the latest snapshot < L.
# - AMM pre state for each tx comes from amm_swaps rows (*_before reserves + fee from amm_fees)
# - Model uses RouterQuoteView.preview_out (exact-out); if partial => recorded as error
# - Real path: AMM leg from amm_swaps + optional CLOB legs from clob legs parquet
#
# Output:
# - prints per-tx comparisons
# - writes parquet: {ROOT}/compare_results.parquet
# - prints direction-bucket summaries
# - prints compact summary blocks
#
# New in this version:
# - Optional IOU transfer fee scaling on IOU inputs when building CLOB tiers:
#     export IOU_IN_TRANSFER_RATE=1.10
# - Extra diagnostics for rUSD->XRP direction
# - Replace mismatch report with "perfect same execution path" stats (path only, ignore diffs)

import json
import os
import glob
import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, getcontext
from typing import Any, Dict, List, Optional, Tuple

from pyspark.sql import SparkSession, functions as F, types as T

from xrpl_router.amm import AMM
from xrpl_router.book_step import RouterQuoteView
from xrpl_router.core import Quality, IOUAmount, XRPAmount
from xrpl_router.core.quality import QUALITY_MANTISSA_MASK
from xrpl_router.core.datatypes import Segment
from xrpl_router.core.constants import XRP_QUANTUM, IOU_QUANTUM
from xrpl_router.core.fmt import amount_to_decimal, quantize_down
from xrpl_router.core.exc import InsufficientLiquidityError

getcontext().prec = 40

# -----------------------------
# Window config (edit if needed)
# -----------------------------
ROOT = "ledger_100893230_to_100895416"
LEDGER_MIN = 100893230
LEDGER_MAX = 100895416

AMM_SWAPS_PARQUET = f"{ROOT}/amm_rusd_xrp_100893230_100895416"
AMM_FEES_PARQUET  = f"{ROOT}/amm_fees_100893230_100895416"
CLOB_LEGS_PARQUET = f"{ROOT}/clob_rusd_xrp_100893230_100895416_with_idx"

BOOK_GETS_XRP_NDJSON  = f"{ROOT}/book_rusd_xrp_getsXRP.ndjson"
BOOK_GETS_RUSD_NDJSON = f"{ROOT}/book_rusd_xrp_getsrUSD.ndjson"
OUT_PARQUET = f"{ROOT}/compare_results.parquet"

# currencies
RUSD_HEX = "524C555344000000000000000000000000000000"
XRP = "XRP"

PRINT_DP: Optional[int] = 6

# Debug toggles
DEBUG_QUOTE = bool(int(os.environ.get("DEBUG_QUOTE", "0")))
DEBUG_BOOK  = bool(int(os.environ.get("DEBUG_BOOK", "0")))

# Optional issuer transfer rate for IOU inputs (e.g. rUSD may have a transfer fee).
IOU_IN_TRANSFER_RATE = Decimal(os.environ.get("IOU_IN_TRANSFER_RATE", "1"))

# Extra diagnostics for rUSD->XRP direction
DEBUG_DIR_RUSD_TO_XRP = bool(int(os.environ.get("DEBUG_DIR_RUSD_TO_XRP", "1")))

# kept for parity (unused)
XRPL_RPC_URL = os.environ.get("XRPL_RPC_URL", "")


def _quality_from_book_directory(raw_book_directory: Any) -> Quality:
    if raw_book_directory is None:
        raise RuntimeError("missing required offer.BookDirectory in snapshot row")
    book_directory = str(raw_book_directory).strip()
    if len(book_directory) < 16:
        raise RuntimeError("invalid offer.BookDirectory: expected at least 16 hex chars")
    try:
        encoded = int(book_directory[-16:], 16)
    except ValueError as exc:
        raise RuntimeError(f"invalid offer.BookDirectory: bad hex value: {exc}") from exc
    encoded &= (0xFF << 56) | QUALITY_MANTISSA_MASK
    return Quality(encoded)


def _pick_one(pattern: str, what: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {what} matched pattern: {pattern}")
    return matches[0]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Research single-window compare (non-rolling).")
    p.add_argument("--root", default=ROOT, help="Dataset root folder")
    p.add_argument("--pair", default="rlusd_xrp", help="Pair key used in output naming")
    p.add_argument("--ledger-start", "--ledger-min", dest="ledger_start", type=int, default=LEDGER_MIN, help="Ledger lower bound (inclusive)")
    p.add_argument("--ledger-end", "--ledger-max", dest="ledger_end", type=int, default=LEDGER_MAX, help="Ledger upper bound (inclusive)")
    p.add_argument("--amm-swaps", default=None, help="AMM swaps parquet path (optional)")
    p.add_argument("--amm-fees", default=None, help="AMM fees parquet path (optional)")
    p.add_argument("--clob-legs", default=None, help="CLOB legs parquet path (optional)")
    p.add_argument("--book-gets-xrp", default=None, help="Book snapshot getsXRP ndjson path (optional)")
    p.add_argument("--book-gets-rusd", default=None, help="Book snapshot getsrUSD ndjson path (optional)")
    p.add_argument("--output-dir", default=None, help="Output directory")
    p.add_argument("--output-name", default=None, help="Output parquet filename")
    return p.parse_args()


def _configure_from_args(args: argparse.Namespace) -> None:
    global ROOT, LEDGER_MIN, LEDGER_MAX
    global AMM_SWAPS_PARQUET, AMM_FEES_PARQUET, CLOB_LEGS_PARQUET
    global BOOK_GETS_XRP_NDJSON, BOOK_GETS_RUSD_NDJSON, OUT_PARQUET

    ROOT = args.root
    LEDGER_MIN = int(args.ledger_start)
    LEDGER_MAX = int(args.ledger_end)

    AMM_SWAPS_PARQUET = args.amm_swaps or _pick_one(f"{ROOT}/amm_rusd_xrp_*", "AMM swaps parquet")
    AMM_FEES_PARQUET = args.amm_fees or _pick_one(f"{ROOT}/amm_fees_*", "AMM fees parquet")
    CLOB_LEGS_PARQUET = args.clob_legs or _pick_one(f"{ROOT}/clob_rusd_xrp*_with_idx", "CLOB legs parquet")
    BOOK_GETS_XRP_NDJSON = args.book_gets_xrp or f"{ROOT}/book_rusd_xrp_getsXRP.ndjson"
    BOOK_GETS_RUSD_NDJSON = args.book_gets_rusd or f"{ROOT}/book_rusd_xrp_getsrUSD.ndjson"
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join("artifacts", "compare", args.pair, f"ledger_{LEDGER_MIN}_{LEDGER_MAX}")
    os.makedirs(out_dir, exist_ok=True)
    output_name = args.output_name or f"compare_single_{args.pair}_ledger_{LEDGER_MIN}_{LEDGER_MAX}_v1.parquet"
    OUT_PARQUET = os.path.join(out_dir, output_name)


def _write_manifest(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# -----------------------------
# Utility formatting
# -----------------------------
def fmt(v: Any, dp: Optional[int] = None) -> str:
    if v is None:
        return "None"
    if dp is None:
        dp = PRINT_DP
    try:
        d = v if isinstance(v, Decimal) else Decimal(str(v))
    except Exception:
        return str(v)
    if dp is None:
        return str(d)
    q = Decimal(1).scaleb(-int(dp))
    try:
        return str(d.quantize(q))
    except Exception:
        return str(d)


def assert_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")


def D(x: Any) -> Decimal:
    if x is None:
        return Decimal("0")
    return Decimal(str(x))


def pretty_cur(c: str) -> str:
    return "rUSD" if c == RUSD_HEX else c


def is_xrp(cur: str) -> bool:
    return cur == XRP


def direction_label(in_cur: str, out_cur: str) -> str:
    return f"{pretty_cur(in_cur)}->{pretty_cur(out_cur)}"


# -----------------------------
# XRPL amount parsing
# -----------------------------
def parse_amount(a: Any) -> Tuple[str, Optional[str], Decimal]:
    if isinstance(a, str):
        return (XRP, None, Decimal(a) * XRP_QUANTUM)  # drops -> XRP
    if isinstance(a, dict):
        return (a.get("currency"), a.get("issuer"), Decimal(str(a.get("value"))))
    raise ValueError(f"Unrecognised amount: {a!r}")


def amt_floor(cur: str, v: Decimal):
    if is_xrp(cur):
        drops = int((v / XRP_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
        return XRPAmount(value=drops)
    units = int((v / IOU_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
    return IOUAmount.from_components(units, -15)


# -----------------------------
# Book snapshot loading
# -----------------------------
def load_book_ndjson(path: str) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            li = obj.get("ledger_index", None)
            if li is None:
                li = obj.get("ledger", None) or obj.get("ledger_current_index", None)
            if li is None:
                raise ValueError(f"NDJSON line {line_no} missing ledger_index: {path}")
            li = int(li)

            offers = obj.get("offers", None)
            if offers is None:
                offers = obj.get("result", {}).get("offers", [])
            if offers is None:
                offers = []
            out[li] = offers
    return out


# -----------------------------
# Pre-book selection with fallback
# -----------------------------
def build_sorted_keys(m: Dict[int, Any]) -> List[int]:
    return sorted(m.keys())


def latest_snapshot_before(keys: List[int], ledger_index: int) -> Optional[int]:
    lo, hi = 0, len(keys) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        k = keys[mid]
        if k < ledger_index:
            best = k
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def pick_prebook(
    ledger_index: int,
    book_xrp_map: Dict[int, List[Dict[str, Any]]],
    book_rusd_map: Dict[int, List[Dict[str, Any]]],
    xrp_keys: List[int],
    rusd_keys: List[int],
) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    want = ledger_index - 1
    offers_xrp = book_xrp_map.get(want, None)
    offers_rusd = book_rusd_map.get(want, None)

    if offers_xrp is not None or offers_rusd is not None:
        return want, (offers_xrp or []), (offers_rusd or [])

    fx = latest_snapshot_before(xrp_keys, ledger_index)
    fr = latest_snapshot_before(rusd_keys, ledger_index)

    if fx is None and fr is None:
        raise RuntimeError(f"No book snapshots exist before ledger {ledger_index}. Cannot compare.")

    if fx is None:
        used = fr
    elif fr is None:
        used = fx
    else:
        used = max(fx, fr)

    return used, (book_xrp_map.get(used, []) or []), (book_rusd_map.get(used, []) or [])


# -----------------------------
# Build tiers from XRPL offers
# -----------------------------
def quantize_up(v: Decimal, quantum: Decimal) -> Decimal:
    if quantum == 0:
        return v
    n = (v / quantum).to_integral_value(rounding=ROUND_CEILING)
    return n * quantum


def apply_iou_in_transfer_fee(in_cur: str, v: Decimal) -> Decimal:
    if is_xrp(in_cur):
        return v
    if IOU_IN_TRANSFER_RATE is None:
        return v
    if IOU_IN_TRANSFER_RATE <= 0:
        return v
    return v * IOU_IN_TRANSFER_RATE


def build_tiers_from_offers(
    offers: List[Dict[str, Any]],
    in_cur: str,
    out_cur: str,
    *,
    allow_empty: bool = False,
) -> List[Tuple[Quality, Any, Any, Any]]:
    tiers: List[Tuple[Quality, Any, Any, Any]] = []

    for o in offers:
        gets_cur, _, gets_val = parse_amount(o.get("TakerGets"))
        pays_cur, _, pays_val = parse_amount(o.get("TakerPays"))

        if "taker_gets_funded" in o:
            try:
                fg_cur, _, fg_val = parse_amount(o.get("taker_gets_funded"))
                if fg_cur == gets_cur and fg_val is not None and fg_val > 0:
                    gets_val = fg_val
                else:
                    continue
            except Exception:
                continue
        if "taker_pays_funded" in o:
            try:
                fp_cur, _, fp_val = parse_amount(o.get("taker_pays_funded"))
                if fp_cur == pays_cur and fp_val is not None and fp_val > 0:
                    pays_val = fp_val
                else:
                    continue
            except Exception:
                continue

        try:
            if "owner_funds" in o and D(o.get("owner_funds")) <= 0:
                continue
        except Exception:
            pass

        raw_book_directory = o.get("book_directory", o.get("BookDirectory"))

        if gets_cur == out_cur and pays_cur == in_cur:
            out_max_dec = gets_val
            in_need_dec = pays_val
        elif pays_cur == out_cur and gets_cur == in_cur:
            out_max_dec = pays_val
            in_need_dec = gets_val
        else:
            continue

        if out_max_dec <= 0 or in_need_dec <= 0:
            continue

        out_q = XRP_QUANTUM if is_xrp(out_cur) else IOU_QUANTUM
        in_q  = XRP_QUANTUM if is_xrp(in_cur)  else IOU_QUANTUM

        out_max_dec = quantize_down(out_max_dec, out_q)

        in_need_dec = apply_iou_in_transfer_fee(in_cur, in_need_dec)
        in_need_dec = quantize_up(in_need_dec, in_q)

        out_amt = amt_floor(out_cur, out_max_dec)
        in_amt  = amt_floor(in_cur,  in_need_dec)

        if amount_to_decimal(out_amt) <= 0 or amount_to_decimal(in_amt) <= 0:
            continue

        q = _quality_from_book_directory(raw_book_directory)
        tiers.append((q, out_amt, in_amt, raw_book_directory))

    if not tiers:
        if allow_empty:
            return []
        raise RuntimeError("No direction-aligned offers found for this IN/OUT.")

    tiers.sort(key=lambda t: t[0], reverse=True)
    return tiers


def sum_out_capacity(tiers: List[Tuple[Quality, Any, Any, Any]]) -> Decimal:
    return sum((amount_to_decimal(t[1]) for t in tiers), start=Decimal("0"))


# -----------------------------
# RouterQuoteView helpers
# -----------------------------
def tiers_to_segments(
    tiers: List[Tuple[Quality, Any, Any, Any]],
    src: str = "CLOB",
) -> List[Segment]:
    segs: List[Segment] = []
    for i, (q, out_amt, in_amt, xrpl_q_raw) in enumerate(tiers):
        segs.append(
            Segment(
                src=src,
                quality=q,
                out_max=out_amt,
                in_at_out_max=in_amt,
                raw_quality=q,
                source_id=f"tier:{i}",
            )
        )
    return segs


def run_best_quote(
    amm: AMM,
    tiers: List[Tuple[Quality, Any, Any, Any]],
    target_out_amt: Any,
) -> Tuple[Dict[str, Any], List[Any]]:
    segs = tiers_to_segments(tiers, src="CLOB")
    view = RouterQuoteView(lambda: segs, amm=amm)
    q = view.preview_out(target_out_amt)
    return q, segs


# -----------------------------
# Slice parsing
# -----------------------------
def summarise_model_slices(quote: Dict[str, Any]) -> Tuple[int, int, Decimal, Decimal, Decimal, Decimal, List[str], bool]:
    n_clob = 0
    n_amm = 0
    clob_in = Decimal("0")
    clob_out = Decimal("0")
    amm_in = Decimal("0")
    amm_out = Decimal("0")
    path: List[str] = []

    slices = quote.get("slices", []) or []
    for s in slices:
        if not isinstance(s, dict):
            continue
        src = str(s.get("src") or s.get("kind") or "?")
        path.append(src)

        in_take = amount_to_decimal(s.get("in_take")) if s.get("in_take") is not None else Decimal("0")
        out_take = amount_to_decimal(s.get("out_take")) if s.get("out_take") is not None else Decimal("0")

        if "CLOB" in src or "BOOK" in src or "ORDER" in src:
            n_clob += 1
            clob_in += in_take
            clob_out += out_take
        elif "AMM" in src:
            n_amm += 1
            amm_in += in_take
            amm_out += out_take

    used_clob = any(("CLOB" in p) or ("BOOK" in p) or ("ORDER" in p) for p in path)
    return n_clob, n_amm, clob_in, clob_out, amm_in, amm_out, path, used_clob


def print_tiers_head(tiers: List[Tuple[Quality, Any, Any, Any]], n: int = 3) -> None:
    h = min(n, len(tiers))
    for i in range(h):
        q, out_amt, in_amt, raw = tiers[i]
        print(f"    tier[{i}] out_max={fmt(amount_to_decimal(out_amt))} in_need={fmt(amount_to_decimal(in_amt))} quality={q}")


# -----------------------------
# Direction-bucket aggregators
# -----------------------------
class BucketAgg:
    def __init__(self) -> None:
        self.real_in = Decimal("0")
        self.real_out = Decimal("0")
        self.model_in = Decimal("0")
        self.model_out = Decimal("0")
        self.txs = 0
        self.failed = 0
        self.real_used_clob = 0
        self.model_used_clob = 0

    def add_ok(self, real_in, real_out, model_in, model_out, real_clob, model_clob):
        self.txs += 1
        self.real_in += real_in
        self.real_out += real_out
        self.model_in += model_in
        self.model_out += model_out
        if real_clob:
            self.real_used_clob += 1
        if model_clob:
            self.model_used_clob += 1

    def add_fail(self):
        self.failed += 1


def safe_div(a: Decimal, b: Decimal) -> Optional[Decimal]:
    if b == 0:
        return None
    return a / b


# -----------------------------
# Main comparison
# -----------------------------
def main() -> None:
    args = _parse_args()
    _configure_from_args(args)
    for p in [AMM_SWAPS_PARQUET, AMM_FEES_PARQUET, CLOB_LEGS_PARQUET, BOOK_GETS_XRP_NDJSON, BOOK_GETS_RUSD_NDJSON]:
        assert_exists(p)

    print(f"\n=== Window compare: ledgers [{LEDGER_MIN}, {LEDGER_MAX}] rUSD-XRP ===")
    print(f"AMM swaps parquet : {AMM_SWAPS_PARQUET}")
    print(f"AMM fees parquet  : {AMM_FEES_PARQUET}")
    print(f"CLOB legs parquet : {CLOB_LEGS_PARQUET}")
    print(f"Book ndjson (getsXRP) : {BOOK_GETS_XRP_NDJSON}")
    print(f"Book ndjson (getsrUSD): {BOOK_GETS_RUSD_NDJSON}")
    print(f"IOU_IN_TRANSFER_RATE  : {IOU_IN_TRANSFER_RATE}")

    # Load book snapshots into memory
    book_gets_xrp_map  = load_book_ndjson(BOOK_GETS_XRP_NDJSON)
    book_gets_rusd_map = load_book_ndjson(BOOK_GETS_RUSD_NDJSON)

    xrp_keys = build_sorted_keys(book_gets_xrp_map)
    rusd_keys = build_sorted_keys(book_gets_rusd_map)

    if not xrp_keys and not rusd_keys:
        raise RuntimeError("Book snapshot ndjson maps are empty.")

    if xrp_keys:
        print(f"[book] getsXRP snapshots: {len(xrp_keys)} | range [{xrp_keys[0]}, {xrp_keys[-1]}]")
    else:
        print(f"[book] getsXRP snapshots: 0")
    if rusd_keys:
        print(f"[book] getsrUSD snapshots: {len(rusd_keys)} | range [{rusd_keys[0]}, {rusd_keys[-1]}]")
    else:
        print(f"[book] getsrUSD snapshots: 0")

    spark = SparkSession.builder.appName("compare_rusd_xrp_window_vs_model").getOrCreate()

    amm_swaps = (
        spark.read.parquet(AMM_SWAPS_PARQUET)
        .filter((F.col("ledger_index") >= F.lit(LEDGER_MIN)) & (F.col("ledger_index") <= F.lit(LEDGER_MAX)))
        .filter(
            (
                (F.col("amm_asset_currency") == F.lit(RUSD_HEX)) & (F.col("amm_asset2_currency") == F.lit(XRP))
            ) | (
                (F.col("amm_asset_currency") == F.lit(XRP)) & (F.col("amm_asset2_currency") == F.lit(RUSD_HEX))
            )
        )
        .select(
            "transaction_hash",
            "ledger_index",
            "transaction_index",
            "amm_account",
            "asset_in_currency",
            "asset_out_currency",
            "asset_in_value",
            "asset_out_value",
            "amm_asset_currency",
            "amm_asset2_currency",
            "amm_asset_balance_before",
            "amm_asset2_balance_before",
        )
    )

    amm_fees = (
        spark.read.parquet(AMM_FEES_PARQUET)
        .filter((F.col("ledger_index") >= F.lit(LEDGER_MIN)) & (F.col("ledger_index") <= F.lit(LEDGER_MAX)))
        .select("transaction_hash", "ledger_index", "amm_account", "trading_fee", "discounted_fee")
    )

    clob_legs = (
        spark.read.parquet(CLOB_LEGS_PARQUET)
        .filter((F.col("ledger_index") >= F.lit(LEDGER_MIN)) & (F.col("ledger_index") <= F.lit(LEDGER_MAX)))
        .select(
            F.col("tx_hash").alias("transaction_hash"),
            "ledger_index",
            "transaction_index",
            "base_currency",
            "counter_currency",
            "base_amount",
            "counter_amount",
        )
    )

    swaps = (
        amm_swaps.orderBy(F.col("ledger_index").asc(), F.col("transaction_index").asc())
        .collect()
    )
    if not swaps:
        raise RuntimeError(f"No rUSD-XRP swaps found in window [{LEDGER_MIN},{LEDGER_MAX}] under {AMM_SWAPS_PARQUET}")

    print(f"\nTx count (amm_swaps rows): {len(swaps)}")

    fee_rows = amm_fees.collect()
    fee_map: Dict[Tuple[str, str], Decimal] = {}
    for r in fee_rows:
        h = r["transaction_hash"]
        a = r["amm_account"]
        if h is None or a is None:
            continue
        fee_disc = r["discounted_fee"]
        fee_trade = r["trading_fee"]
        # single-window script lacks tx-account/auction-slot context;
        # default to pool trading fee to avoid globally over-discounting AMM.
        if fee_trade is not None:
            f = D(fee_trade)
        elif fee_disc is not None:
            f = D(fee_disc)
        else:
            raise RuntimeError(
                "amm_fees row missing both discounted_fee and trading_fee: "
                f"tx={h} amm_account={a}"
            )
        fee_map[(str(h), str(a))] = f

    legs_rows = clob_legs.collect()
    legs_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in legs_rows:
        h = r["transaction_hash"]
        if h is None:
            continue
        legs_map.setdefault(str(h), []).append({
            "base_currency": r["base_currency"],
            "counter_currency": r["counter_currency"],
            "base_amount": r["base_amount"],
            "counter_amount": r["counter_amount"],
        })

    results: List[Dict[str, Any]] = []

    bucket_xrp_to_rusd = BucketAgg()
    bucket_rusd_to_xrp = BucketAgg()

    failed_txs: List[Tuple[int, int, str, str]] = []

    # NEW: store "perfect same execution path" txs
    perfect_path_txs: List[Tuple[int, int, str, str, str]] = []  # (ledger, tx, hash, direction, path_sig)
    perfect_path_by_dir: Dict[str, int] = {"XRP->rUSD": 0, "rUSD->XRP": 0}

    def normalise_src(src: str) -> str:
        s = (src or "").upper()
        if "AMM" in s:
            return "AMM"
        if "CLOB" in s or "BOOK" in s or "ORDER" in s:
            return "CLOB"
        return "?"

    def real_path_signature(n_legs: int) -> str:
        # REAL always starts with AMM (because amm_swaps row exists)
        if n_legs <= 0:
            return "AMM"
        return "AMM" + ("+CLOB" * n_legs)

    def model_path_signature(path_m: List[str]) -> str:
        # MODEL path based on slice src order
        steps = [normalise_src(p) for p in path_m if normalise_src(p) != "?"]
        return "+".join(steps) if steps else ""

    for tx in swaps:
        tx_hash = str(tx["transaction_hash"])
        ledger_index = int(tx["ledger_index"])
        tx_index = int(tx["transaction_index"]) if tx["transaction_index"] is not None else -1

        in_cur = tx["asset_in_currency"]
        out_cur = tx["asset_out_currency"]
        dir_lbl = direction_label(in_cur, out_cur)

        dir_key = None
        if in_cur == XRP and out_cur == RUSD_HEX:
            bucket = bucket_xrp_to_rusd
            dir_key = "XRP->rUSD"
        elif in_cur == RUSD_HEX and out_cur == XRP:
            bucket = bucket_rusd_to_xrp
            dir_key = "rUSD->XRP"
        else:
            bucket = None

        amm_in_real  = D(tx["asset_in_value"])
        amm_out_real = D(tx["asset_out_value"])

        legs = legs_map.get(tx_hash, [])
        clob_in_sum = Decimal("0")
        clob_out_sum = Decimal("0")

        for r in legs:
            base_cur = r["base_currency"]
            counter_cur = r["counter_currency"]
            base_amt = D(r["base_amount"])
            counter_amt = D(r["counter_amount"])

            if base_cur == out_cur and counter_cur == in_cur:
                clob_out_sum += base_amt
                clob_in_sum  += counter_amt
            elif counter_cur == out_cur and base_cur == in_cur:
                clob_out_sum += counter_amt
                clob_in_sum  += base_amt

        real_total_out = amm_out_real + clob_out_sum
        real_total_in  = amm_in_real  + clob_in_sum

        side1_cur = tx["amm_asset_currency"]
        side2_cur = tx["amm_asset2_currency"]
        side1_pre = D(tx["amm_asset_balance_before"])
        side2_pre = D(tx["amm_asset2_balance_before"])

        if in_cur == side1_cur and out_cur == side2_cur:
            x_reserve, y_reserve = side1_pre, side2_pre
        elif in_cur == side2_cur and out_cur == side1_cur:
            x_reserve, y_reserve = side2_pre, side1_pre
        else:
            err = f"Swap currencies do not match AMM pool sides."
            failed_txs.append((ledger_index, tx_index, tx_hash, err))
            if bucket is not None:
                bucket.add_fail()
            results.append({
                "ledger_index": ledger_index,
                "transaction_index": tx_index,
                "transaction_hash": tx_hash,
                "direction": dir_lbl,
                "used_prebook_ledger": None,
                "real_in": float(real_total_in),
                "real_out": float(real_total_out),
                "model_in": None,
                "model_out": None,
                "diff_in": None,
                "diff_out": None,
                "real_used_clob": len(legs) > 0,
                "model_used_clob": None,
                "error": err,
            })
            continue

        trading_fee = fee_map.get((tx_hash, str(tx["amm_account"])), Decimal("0"))
        amm = AMM(x_reserve, y_reserve, trading_fee, x_is_xrp=is_xrp(in_cur), y_is_xrp=is_xrp(out_cur))

        try:
            used_prebook_ledger, offers_xrp, offers_rusd = pick_prebook(
                ledger_index=ledger_index,
                book_xrp_map=book_gets_xrp_map,
                book_rusd_map=book_gets_rusd_map,
                xrp_keys=xrp_keys,
                rusd_keys=rusd_keys,
            )
        except Exception as e:
            err = f"pick_prebook failed: {type(e).__name__}: {e}"
            failed_txs.append((ledger_index, tx_index, tx_hash, err))
            if bucket is not None:
                bucket.add_fail()
            results.append({
                "ledger_index": ledger_index,
                "transaction_index": tx_index,
                "transaction_hash": tx_hash,
                "direction": dir_lbl,
                "used_prebook_ledger": None,
                "real_in": float(real_total_in),
                "real_out": float(real_total_out),
                "model_in": None,
                "model_out": None,
                "diff_in": None,
                "diff_out": None,
                "real_used_clob": len(legs) > 0,
                "model_used_clob": None,
                "error": err,
            })
            continue

        # Direction-locked: choose the correct book for the swap direction.
        # getsrUSD book: taker_gets=rUSD, taker_pays=XRP  -> XRP->rUSD
        # getsXRP  book: taker_gets=XRP,  taker_pays=rUSD -> rUSD->XRP
        tiers: Optional[List[Tuple[Quality, Any, Any, Any]]] = None
        last_err: Optional[Exception] = None

        if in_cur == XRP and out_cur == RUSD_HEX:
            book = offers_rusd
        elif in_cur == RUSD_HEX and out_cur == XRP:
            book = offers_xrp
        else:
            book = []

        if not book:
            last_err = RuntimeError(f"empty book for direction {pretty_cur(in_cur)}->{pretty_cur(out_cur)}")
        else:
            try:
                tiers = build_tiers_from_offers(book, in_cur, out_cur)
            except Exception as e:
                last_err = e
                tiers = None

        if tiers is None:
            err = f"build_tiers failed: {last_err}"
            failed_txs.append((ledger_index, tx_index, tx_hash, err))
            if bucket is not None:
                bucket.add_fail()
            results.append({
                "ledger_index": ledger_index,
                "transaction_index": tx_index,
                "transaction_hash": tx_hash,
                "direction": dir_lbl,
                "used_prebook_ledger": used_prebook_ledger,
                "real_in": float(real_total_in),
                "real_out": float(real_total_out),
                "model_in": None,
                "model_out": None,
                "diff_in": None,
                "diff_out": None,
                "real_used_clob": len(legs) > 0,
                "model_used_clob": None,
                "error": err,
            })
            continue

        cap_out = sum_out_capacity(tiers)

        target_out = real_total_out
        target_out_amt = amt_floor(out_cur, target_out)

        quote = None
        pred_in = None
        pred_out = None
        used_clob = None
        path_m: List[str] = []
        model_err: Optional[str] = None
        n_model_clob_slices = 0

        try:
            quote, _ = run_best_quote(amm=amm, tiers=tiers, target_out_amt=target_out_amt)
            summary = quote.get("summary", {})

            if summary.get("is_partial"):
                raise RuntimeError("Model returned PARTIAL quote")

            pred_in  = amount_to_decimal(summary["total_in"])
            pred_out = amount_to_decimal(summary["total_out"])

            n_clob, _, _, _, _, _, path_m, used_clob = summarise_model_slices(quote)
            n_model_clob_slices = n_clob

        except Exception as e:
            model_err = f"model_quote failed: {type(e).__name__}: {e}"
            failed_txs.append((ledger_index, tx_index, tx_hash, model_err))
            if bucket is not None:
                bucket.add_fail()

        # ---- Print per-tx
        print("\n---")
        print(f"ledger {ledger_index} | tx {tx_index} | {tx_hash}")
        print(f"dir: {pretty_cur(in_cur)} -> {pretty_cur(out_cur)} | book snapshot ledger: {used_prebook_ledger}")
        real_path = "AMM" if len(legs) == 0 else f"AMM + CLOB({len(legs)} legs)"
        print(f"REAL  path: {real_path}")
        print(f"REAL  TOT : in={fmt(real_total_in)} {pretty_cur(in_cur)}  out={fmt(real_total_out)} {pretty_cur(out_cur)}")

        if pred_in is None or pred_out is None:
            print(f"MODEL: FAILED ({model_err})")
            print(f"  snapshot cap_out={fmt(cap_out)} | target_out={fmt(target_out)}")
            print("  tiers head:")
            print_tiers_head(tiers, n=3)
        else:
            model_path = " + ".join(path_m) if path_m else ("CLOB" if used_clob else "AMM")
            print(f"MODEL path: {model_path}")
            print(f"MODEL TOT : in={fmt(pred_in)} {pretty_cur(in_cur)}  out={fmt(pred_out)} {pretty_cur(out_cur)}")
            print(f"DIFF  in : {fmt(pred_in - real_total_in)} {pretty_cur(in_cur)}")
            print(f"DIFF  out: {fmt(pred_out - real_total_out)} {pretty_cur(out_cur)}")

            # Extra diagnostics for rUSD->XRP
            if DEBUG_DIR_RUSD_TO_XRP and in_cur == RUSD_HEX and out_cur == XRP:
                realised_px = safe_div(real_total_in, real_total_out)
                model_px = safe_div(pred_in, pred_out)

                try:
                    amm_only_quote = RouterQuoteView(lambda: [], amm=amm).preview_out(target_out_amt)
                    amm_only_in = amount_to_decimal(amm_only_quote["summary"]["total_in"])
                    amm_only_px = amm_only_in / real_total_out if real_total_out != 0 else None
                except Exception:
                    amm_only_in = None
                    amm_only_px = None

                print("\n[diag rUSD->XRP]")
                if realised_px is not None:
                    print(f"  realised_px = {fmt(realised_px)} rUSD/XRP")
                if model_px is not None:
                    print(f"  model_px    = {fmt(model_px)} rUSD/XRP")
                if amm_only_px is not None:
                    print(f"  amm_only_px = {fmt(amm_only_px)} rUSD/XRP")
                if amm_only_in is not None:
                    print(f"  amm_only_in = {fmt(amm_only_in)} rUSD (to get same out)")

        # ---- Record parquet row
        if pred_in is None or pred_out is None:
            results.append({
                "ledger_index": ledger_index,
                "transaction_index": tx_index,
                "transaction_hash": tx_hash,
                "direction": dir_lbl,
                "used_prebook_ledger": used_prebook_ledger,
                "real_in": float(real_total_in),
                "real_out": float(real_total_out),
                "model_in": None,
                "model_out": None,
                "diff_in": None,
                "diff_out": None,
                "real_used_clob": len(legs) > 0,
                "model_used_clob": None,
                "error": model_err,
            })
            continue

        diff_in = pred_in - real_total_in
        diff_out = pred_out - real_total_out

        results.append({
            "ledger_index": ledger_index,
            "transaction_index": tx_index,
            "transaction_hash": tx_hash,
            "direction": dir_lbl,
            "used_prebook_ledger": used_prebook_ledger,
            "real_in": float(real_total_in),
            "real_out": float(real_total_out),
            "model_in": float(pred_in),
            "model_out": float(pred_out),
            "diff_in": float(diff_in),
            "diff_out": float(diff_out),
            "real_used_clob": len(legs) > 0,
            "model_used_clob": bool(used_clob),
            "error": None,
        })

        # ---- Bucket add
        if bucket is not None:
            bucket.add_ok(
                real_in=real_total_in,
                real_out=real_total_out,
                model_in=pred_in,
                model_out=pred_out,
                real_clob=(len(legs) > 0),
                model_clob=bool(used_clob),
            )

        # ---- NEW: perfect-path detection (exact path match, ignore diffs)
        if dir_key is not None and pred_in is not None and pred_out is not None:
            real_sig = real_path_signature(len(legs))           # e.g. AMM+CLOB+CLOB+CLOB
            model_sig = model_path_signature(path_m)            # e.g. CLOB+CLOB+AMM+CLOB+AMM
            if real_sig == model_sig:
                perfect_path_txs.append((ledger_index, tx_index, tx_hash, dir_key, real_sig))
                perfect_path_by_dir[dir_key] += 1

    # -----------------------------
    # Write results parquet
    # -----------------------------
    out_path = OUT_PARQUET
    schema = T.StructType([
        T.StructField("ledger_index", T.LongType(), False),
        T.StructField("transaction_index", T.LongType(), False),
        T.StructField("transaction_hash", T.StringType(), False),
        T.StructField("direction", T.StringType(), False),
        T.StructField("used_prebook_ledger", T.LongType(), True),
        T.StructField("real_in", T.DoubleType(), True),
        T.StructField("real_out", T.DoubleType(), True),
        T.StructField("model_in", T.DoubleType(), True),
        T.StructField("model_out", T.DoubleType(), True),
        T.StructField("diff_in", T.DoubleType(), True),
        T.StructField("diff_out", T.DoubleType(), True),
        T.StructField("real_used_clob", T.BooleanType(), True),
        T.StructField("model_used_clob", T.BooleanType(), True),
        T.StructField("error", T.StringType(), True),
    ])

    df_out = spark.createDataFrame(results, schema=schema)
    df_out.coalesce(1).write.mode("overwrite").parquet(out_path)
    print(f"\n[OK] compare results written to: {out_path}")
    _write_manifest(
        os.path.join(os.path.dirname(out_path), "manifest.json"),
        {
            "script": "empirical_compare_single.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pair": args.pair,
            "window": {"ledger_start": LEDGER_MIN, "ledger_end": LEDGER_MAX},
            "inputs": {
                "amm_swaps": AMM_SWAPS_PARQUET,
                "amm_fees": AMM_FEES_PARQUET,
                "clob_legs": CLOB_LEGS_PARQUET,
                "book_gets_xrp": BOOK_GETS_XRP_NDJSON,
                "book_gets_rusd": BOOK_GETS_RUSD_NDJSON,
            },
            "output_parquet": out_path,
            "tx_count": len(swaps),
            "failed_tx_count": len(failed_txs),
        },
    )

    # =========================================================
    # Compact summaries (direction + amount buckets)
    # =========================================================

    def _pct(x: Optional[Decimal]) -> str:
        if x is None:
            return "None"
        return f"{(x * Decimal('100')):.3f}%"

    def _d(x: Any) -> Decimal:
        return Decimal(str(x)) if x is not None else Decimal("0")

    def _quantile(sorted_vals: List[Decimal], q: float) -> Optional[Decimal]:
        if not sorted_vals:
            return None
        if q <= 0:
            return sorted_vals[0]
        if q >= 1:
            return sorted_vals[-1]
        n = len(sorted_vals)
        idx = int((n - 1) * q)
        return sorted_vals[idx]

    def _rel_err(real_in: Decimal, model_in: Decimal) -> Optional[Decimal]:
        if real_in == 0:
            return None
        return (model_in - real_in) / real_in

    dir_rows: Dict[str, List[Dict[str, Any]]] = {"XRP->rUSD": [], "rUSD->XRP": []}
    for r in results:
        if r.get("error") is not None:
            continue
        d = r.get("direction")
        if d == "XRP->rUSD":
            dir_rows["XRP->rUSD"].append(r)
        elif d == "rUSD->XRP":
            dir_rows["rUSD->XRP"].append(r)

    def print_direction_summary(dir_name: str, in_unit: str, out_unit: str) -> None:
        rows = dir_rows[dir_name]
        if not rows:
            print(f"\n=== {dir_name} ===")
            print("No ok txs.")
            return

        real_in_sum  = sum((_d(x["real_in"]) for x in rows), start=Decimal("0"))
        real_out_sum = sum((_d(x["real_out"]) for x in rows), start=Decimal("0"))
        model_in_sum = sum((_d(x["model_in"]) for x in rows), start=Decimal("0"))
        model_out_sum= sum((_d(x["model_out"]) for x in rows), start=Decimal("0"))

        dif_in = model_in_sum - real_in_sum
        dif_out= model_out_sum - real_out_sum

        real_px = real_in_sum / real_out_sum if real_out_sum != 0 else None
        model_px = model_in_sum / model_out_sum if model_out_sum != 0 else None

        rel_errs: List[Decimal] = []
        for x in rows:
            ri = _d(x["real_in"]); mi = _d(x["model_in"])
            e = _rel_err(ri, mi)
            if e is not None:
                rel_errs.append(e)

        rel_errs.sort()
        mean_rel = (sum(rel_errs, start=Decimal("0")) / Decimal(len(rel_errs))) if rel_errs else None
        p50 = _quantile(rel_errs, 0.50)
        p90 = _quantile(rel_errs, 0.90)
        p99 = _quantile(rel_errs, 0.99)

        print(f"\n=== {dir_name} (ok tx={len(rows)}) ===")
        print(f"REAL  TOT : in={fmt(real_in_sum)} {in_unit}  out={fmt(real_out_sum)} {out_unit}")
        print(f"MODEL TOT : in={fmt(model_in_sum)} {in_unit}  out={fmt(model_out_sum)} {out_unit}")
        print(f"DIFF  TOT : in={fmt(dif_in)} {in_unit}  out={fmt(dif_out)} {out_unit}")

        if real_px is not None and model_px is not None:
            print(f"avg_px REAL ={fmt(real_px)} ({in_unit}/{out_unit}) | MODEL ={fmt(model_px)} | DIFF ={fmt(model_px - real_px)}")

        print(f"rel_err_in mean={_pct(mean_rel)} | p50={_pct(p50)} | p90={_pct(p90)} | p99={_pct(p99)}")

        topN = 5
        print(f"top {topN} abs(diff_in):")
        triples: List[Tuple[Decimal, int, int, str]] = []
        for x in rows:
            ri = _d(x["real_in"]); mi = _d(x["model_in"])
            triples.append((abs(mi - ri), int(x["ledger_index"]), int(x["transaction_index"]), x["transaction_hash"]))
        triples.sort(key=lambda t: t[0], reverse=True)
        for i, (v, li, ti, h) in enumerate(triples[:topN], start=1):
            print(f"  {i:02d}. abs_diff_in={fmt(v)} | ledger={li} tx={ti} hash={h}")

    BUCKETS = [
        ("<=10", Decimal("10")),
        ("10-100", Decimal("100")),
        ("100-1k", Decimal("1000")),
        ("1k-10k", Decimal("10000")),
        (">10k", None),
    ]

    def bucket_label(v: Decimal) -> str:
        for name, hi in BUCKETS:
            if hi is None:
                return name
            if v <= hi:
                return name
        return BUCKETS[-1][0]

    def print_amount_buckets(dir_name: str, in_unit: str, out_unit: str) -> None:
        rows = dir_rows[dir_name]
        if not rows:
            return

        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for x in rows:
            outv = _d(x["real_out"])
            bl = bucket_label(outv)
            buckets.setdefault(bl, []).append(x)

        print(f"\n[{dir_name}] amount buckets by real_out ({out_unit}):")
        for name, _ in BUCKETS:
            br = buckets.get(name, [])
            if not br:
                continue
            real_out_sum = sum((_d(x["real_out"]) for x in br), start=Decimal("0"))
            real_in_sum  = sum((_d(x["real_in"]) for x in br), start=Decimal("0"))

            rel_errs = []
            for x in br:
                ri = _d(x["real_in"]); mi = _d(x["model_in"])
                e = _rel_err(ri, mi)
                if e is not None:
                    rel_errs.append(e)
            rel_errs.sort()
            mean_rel = (sum(rel_errs, start=Decimal("0")) / Decimal(len(rel_errs))) if rel_errs else None
            p90 = _quantile(rel_errs, 0.90)

            print(f"  bucket {name:>7} | tx={len(br):>3} | out_sum={fmt(real_out_sum):>12} {out_unit}"
                  f" | in_sum={fmt(real_in_sum):>12} {in_unit} | rel_err mean={_pct(mean_rel)} p90={_pct(p90)}")

    print_direction_summary("XRP->rUSD", "XRP", "rUSD")
    print_amount_buckets("XRP->rUSD", "XRP", "rUSD")

    print_direction_summary("rUSD->XRP", "rUSD", "XRP")
    print_amount_buckets("rUSD->XRP", "rUSD", "XRP")

    # =========================================================
    # NEW: Perfect same execution path (ignore diffs)
    # =========================================================
    print("\n=== PERFECT SAME EXECUTION PATH (ignore diffs) ===")
    total_ok = len(dir_rows["XRP->rUSD"]) + len(dir_rows["rUSD->XRP"])
    total_perfect = len(perfect_path_txs)
    ratio = (Decimal(total_perfect) / Decimal(total_ok)) if total_ok > 0 else Decimal("0")
    print(f"perfect_path txs = {total_perfect} / {total_ok}  ({fmt(ratio * Decimal('100'), 3)}%)")
    print(f"  XRP->rUSD : {perfect_path_by_dir['XRP->rUSD']}")
    print(f"  rUSD->XRP : {perfect_path_by_dir['rUSD->XRP']}")

    if perfect_path_txs:
        print("\nexamples (first 10):")
        for (li, ti, h, d, sig) in perfect_path_txs[:10]:
            print(f"  {d:<8} | ledger {li} tx {ti} | {sig:<12} | hash={h}")
        if len(perfect_path_txs) > 10:
            print(f"  ... ({len(perfect_path_txs)-10} more)")

    if failed_txs:
        print("\n=== FAILURES (first 10) ===")
        for (li, ti, h, e) in failed_txs[:10]:
            print(f"  ledger {li} tx {ti} hash={h}")
            print(f"    err: {e}")
        if len(failed_txs) > 10:
            print(f"  ... ({len(failed_txs)-10} more)")

    spark.stop()
    spark.stop()


if __name__ == "__main__":
    main()
