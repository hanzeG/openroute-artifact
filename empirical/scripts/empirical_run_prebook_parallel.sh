#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  source .env
else
  echo "missing .env (copy from .env.example first)"
  exit 1
fi

: "${XRPL_RUSTYCHAIN_RPC:?XRPL_RUSTYCHAIN_RPC is required}"
: "${XRPL_CLUSTER_RPC:?XRPL_CLUSTER_RPC is required}"
: "${XRPL_RIPPLE_S1_RPC:?XRPL_RIPPLE_S1_RPC is required}"
: "${XRPL_RIPPLE_S2_RPC:?XRPL_RIPPLE_S2_RPC is required}"
: "${XRPL_RUSD_ISSUER:?XRPL_RUSD_ISSUER is required}"
: "${XRPL_RUSD_HEX:?XRPL_RUSD_HEX is required}"

ASSIGN_DIR="${1:-artifacts/prebook/rlusd_xrp/assignments}"
RUN_BASE="${2:-artifacts/prebook/rlusd_xrp/run_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RUN_BASE/logs"
mkdir -p "$RUN_BASE/pids"

LEDGER_QN="$ASSIGN_DIR/ledgers_qn_pool.txt"
LEDGER_S1="$ASSIGN_DIR/ledgers_s1.txt"
LEDGER_S2="$ASSIGN_DIR/ledgers_s2.txt"
LEDGER_RUSTY="$ASSIGN_DIR/ledgers_rusty.txt"
LEDGER_CLUSTER="$ASSIGN_DIR/ledgers_cluster.txt"

for f in "$LEDGER_QN" "$LEDGER_S1" "$LEDGER_S2" "$LEDGER_RUSTY" "$LEDGER_CLUSTER"; do
  [[ -f "$f" ]] || { echo "missing assignment file: $f"; exit 1; }
done

OUT_QN="$RUN_BASE/prebook_qn_pool"
OUT_S1="$RUN_BASE/prebook_s1"
OUT_S2="$RUN_BASE/prebook_s2"
OUT_RUSTY="$RUN_BASE/prebook_rusty"
OUT_CLUSTER="$RUN_BASE/prebook_cluster"
mkdir -p "$OUT_QN" "$OUT_S1" "$OUT_S2" "$OUT_RUSTY" "$OUT_CLUSTER"

launch() {
  local name="$1"
  local cmd="$2"
  local log="$RUN_BASE/logs/${name}.log"
  echo "[launch] $name"
  echo "  log: $log"
  nohup bash -lc "$cmd" >"$log" 2>&1 &
  echo $! > "$RUN_BASE/pids/${name}.pid"
}

launch "quicknode_pool" \
  "cd '$ROOT_DIR' && bash empirical/scripts/empirical_run_prebook_quicknode_pool.sh '$LEDGER_QN' '$OUT_QN' 12 100"

launch "ripple_s1" \
  "cd '$ROOT_DIR' && python empirical/scripts/empirical_download_clob_offers_from_ledger_list.py --rpc '$XRPL_RIPPLE_S1_RPC' --ledger-list '$LEDGER_S1' --outdir '$OUT_S1' --issuer '$XRPL_RUSD_ISSUER' --currency-hex '$XRPL_RUSD_HEX' --workers 18 --limit 100 --timeout 20 --retries 4 --max-consecutive-failures 120"

launch "ripple_s2" \
  "cd '$ROOT_DIR' && python empirical/scripts/empirical_download_clob_offers_from_ledger_list.py --rpc '$XRPL_RIPPLE_S2_RPC' --ledger-list '$LEDGER_S2' --outdir '$OUT_S2' --issuer '$XRPL_RUSD_ISSUER' --currency-hex '$XRPL_RUSD_HEX' --workers 16 --limit 100 --timeout 20 --retries 4 --max-consecutive-failures 120"

launch "rustychain" \
  "cd '$ROOT_DIR' && python empirical/scripts/empirical_download_clob_offers_from_ledger_list.py --rpc '$XRPL_RUSTYCHAIN_RPC' --ledger-list '$LEDGER_RUSTY' --outdir '$OUT_RUSTY' --issuer '$XRPL_RUSD_ISSUER' --currency-hex '$XRPL_RUSD_HEX' --workers 12 --limit 100 --timeout 20 --retries 4 --max-consecutive-failures 120"

launch "xrplcluster" \
  "cd '$ROOT_DIR' && python empirical/scripts/empirical_download_clob_offers_from_ledger_list.py --rpc '$XRPL_CLUSTER_RPC' --ledger-list '$LEDGER_CLUSTER' --outdir '$OUT_CLUSTER' --issuer '$XRPL_RUSD_ISSUER' --currency-hex '$XRPL_RUSD_HEX' --workers 2 --limit 100 --timeout 20 --retries 4 --max-consecutive-failures 120"

cat > "$RUN_BASE/RUN_INFO.txt" <<EOF
ROOT_DIR=$ROOT_DIR
ASSIGN_DIR=$ASSIGN_DIR
RUN_BASE=$RUN_BASE
OUT_QN=$OUT_QN
OUT_S1=$OUT_S1
OUT_S2=$OUT_S2
OUT_RUSTY=$OUT_RUSTY
OUT_CLUSTER=$OUT_CLUSTER
EOF

echo
echo "started all jobs"
echo "run_base: $RUN_BASE"
echo "monitor:"
echo "  bash empirical/scripts/empirical_monitor_prebook_parallel.sh '$RUN_BASE'"
