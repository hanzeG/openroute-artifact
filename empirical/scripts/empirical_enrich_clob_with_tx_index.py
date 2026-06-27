# enrich_clob_with_curl_idx.py
#
# Goal:
#   Enrich local CLOB offers/legs parquet by adding:
#     - ledger_index
#     - transaction_index (meta.TransactionIndex)
#   via curl RPC lookup using transaction hash.
#
# Input:
#   ledger_100893230_to_100895416/clob_rusd_xrp_100893230_100895416
#
# Output:
#   ledger_100893230_to_100895416/clob_rusd_xrp_100893230_100895416_with_idx
#
# Run:
#   spark-submit --properties-file data/spark.properties \
#     ledger_100893230_to_100895416/enrich_clob_with_curl_idx.py

from pyspark.sql import SparkSession, functions as F, types as T
import argparse
import os
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# Config
# ============================================================
IN_PARQUET  = "rlusd_xrp_100891230_100913230_1d/clob_rusd_xrp_DATE_2025-12-15_to_2025-12-16_TIME_20251215_110800Z_to_20251216_110310Z"
OUT_PARQUET = "rlusd_xrp_100891230_100913230_1d/clob_rusd_xrp_DATE_2025-12-15_to_2025-12-16_TIME_20251215_110800Z_to_20251216_110310Z_with_idx"

RIPPLE_TX_RPC = "https://s1.ripple.com:51234/"

# parallelism for curl (tune if you get rate-limited or unstable network)
MAX_WORKERS = 24

# retry policy
RETRY_N = 3
RETRY_SLEEP_SEC = 0.6  # base sleep; grows exponentially


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich CLOB parquet with ledger_index/transaction_index via tx RPC.")
    p.add_argument("--in-parquet", default=IN_PARQUET, help="Input CLOB parquet path")
    p.add_argument("--out-parquet", default=OUT_PARQUET, help="Output enriched parquet path")
    p.add_argument("--rpc", default=RIPPLE_TX_RPC, help="XRPL tx RPC endpoint")
    p.add_argument("--max-workers", type=int, default=MAX_WORKERS, help="Parallel tx fetch workers")
    p.add_argument("--retry-n", type=int, default=RETRY_N, help="Fetch retry count")
    return p.parse_args()


ARGS = parse_args()
IN_PARQUET = ARGS.in_parquet
OUT_PARQUET = ARGS.out_parquet
RIPPLE_TX_RPC = ARGS.rpc
MAX_WORKERS = int(ARGS.max_workers)
RETRY_N = int(ARGS.retry_n)

# ============================================================
# Helpers
# ============================================================
def assert_exists(path: str):
    if not os.path.exists(path):
        raise RuntimeError(f"Missing input parquet folder: {path}")

def curl_tx(hash_: str) -> dict:
    payload = {
        "method": "tx",
        "params": [{
            "transaction": hash_,
            "binary": False
        }]
    }
    cmd = [
        "curl", "-s", "--connect-timeout", "10", "--max-time", "30", RIPPLE_TX_RPC,
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload)
    ]
    out = subprocess.check_output(cmd)
    return json.loads(out.decode("utf-8"))

def extract_tx_meta(tx_resp: dict):
    """
    Return (ledger_index, transaction_index) if available, else (None, None).
    """
    r = tx_resp.get("result", {})
    if r.get("validated") is not True:
        return (None, None)
    ledger_index = r.get("ledger_index", None)
    tx_index = None
    meta = r.get("meta", {})
    if isinstance(meta, dict):
        tx_index = meta.get("TransactionIndex", None)
    return (ledger_index, tx_index)

def fetch_one(hash_: str):
    """
    Curl one tx with retries.
    Return dict: {tx_hash, ledger_index, transaction_index, ok, err}
    """
    last_err = None
    for attempt in range(1, RETRY_N + 1):
        try:
            resp = curl_tx(hash_)
            (lgr, txi) = extract_tx_meta(resp)
            if lgr is None or txi is None:
                last_err = "missing ledger_index or TransactionIndex"
                sleep = RETRY_SLEEP_SEC * (2 ** (attempt - 1))
                time.sleep(sleep)
                continue
            return {
                "tx_hash": hash_,
                "ledger_index": int(lgr) if lgr is not None else None,
                "transaction_index": int(txi) if txi is not None else None,
                "ok": True,
                "err": None
            }
        except Exception as e:
            last_err = repr(e)
            sleep = RETRY_SLEEP_SEC * (2 ** (attempt - 1))
            time.sleep(sleep)

    return {
        "tx_hash": hash_,
        "ledger_index": None,
        "transaction_index": None,
        "ok": False,
        "err": last_err
    }

# ============================================================
# Spark session
# ============================================================
spark = (
    SparkSession.builder
    .appName("enrich_clob_with_curl_ledger_and_tx_index")
    .getOrCreate()
)

assert_exists(IN_PARQUET)

clob_df = spark.read.parquet(IN_PARQUET)

# detect tx hash column name
tx_col = None
for c in ["tx_hash", "transaction_hash", "hash"]:
    if c in clob_df.columns:
        tx_col = c
        break
if tx_col is None:
    raise RuntimeError(
        f"Input CLOB parquet missing tx hash column "
        f"(tx_hash/transaction_hash/hash). columns={clob_df.columns}"
    )

print(f"Detected tx hash column = {tx_col}")
print(f"Input rows = {clob_df.count()}")

# distinct tx hashes
tx_hashes = [r[0] for r in clob_df.select(tx_col).distinct().collect()]
total = len(tx_hashes)
print(f"Distinct tx hashes = {total}")

# ============================================================
# Curl fetch in parallel
# ============================================================
results = []
ok_cnt = 0
fail_cnt = 0
missing_cnt = 0

print(f"Starting curl fetch with MAX_WORKERS={MAX_WORKERS}, RETRY_N={RETRY_N}")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(fetch_one, h): h for h in tx_hashes}
    done = 0
    for fut in as_completed(futures):
        res = fut.result()
        results.append(res)

        done += 1
        if res["ok"]:
            ok_cnt += 1
            if res["ledger_index"] is None or res["transaction_index"] is None:
                missing_cnt += 1
        else:
            fail_cnt += 1

        if done % 50 == 0 or done == total:
            print(f"  progress {done}/{total} | ok={ok_cnt} fail={fail_cnt} ok_but_missing_fields={missing_cnt}")

print("Curl fetch complete.")
print(f"Summary: ok={ok_cnt}, fail={fail_cnt}, ok_but_missing_fields={missing_cnt}")

# ============================================================
# Build mapping DF and join back
# ============================================================
schema = T.StructType([
    T.StructField("tx_hash", T.StringType(), nullable=False),
    T.StructField("ledger_index", T.LongType(), nullable=True),
    T.StructField("transaction_index", T.LongType(), nullable=True),
    T.StructField("ok", T.BooleanType(), nullable=False),
    T.StructField("err", T.StringType(), nullable=True),
])

map_df = spark.createDataFrame(results, schema=schema)

# standardise join key name in original df
base_df = clob_df.withColumnRenamed(tx_col, "tx_hash")

enriched = (
    base_df
    .join(map_df.select("tx_hash", "ledger_index", "transaction_index"), on="tx_hash", how="left")
)

# some sanity checks
enriched_cnt = enriched.count()
null_lgr = enriched.filter(F.col("ledger_index").isNull()).count()
null_txi = enriched.filter(F.col("transaction_index").isNull()).count()

print(f"Enriched rows = {enriched_cnt}")
print(f"Rows with NULL ledger_index = {null_lgr}")
print(f"Rows with NULL transaction_index = {null_txi}")
if null_lgr or null_txi:
    raise RuntimeError(
        f"tx index enrichment incomplete: null ledger_index={null_lgr}, "
        f"null transaction_index={null_txi}"
    )

# write out
(
    enriched
    .write.mode("overwrite")
    .parquet(OUT_PARQUET)
)

print(f"Enriched parquet written to: {OUT_PARQUET}")

spark.stop()
print("Done.")
