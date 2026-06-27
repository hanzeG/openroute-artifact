#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env" ]]; then
  echo "missing .env"
  exit 1
fi

set -a
source .env
set +a

: "${XRPL_QN_RPC:?XRPL_QN_RPC is required}"
: "${XRPL_RIPPLE_S1_RPC:?XRPL_RIPPLE_S1_RPC is required}"
: "${XRPL_RIPPLE_S2_RPC:?XRPL_RIPPLE_S2_RPC is required}"
: "${XRPL_RUSTYCHAIN_RPC:?XRPL_RUSTYCHAIN_RPC is required}"
: "${XRPL_CLUSTER_RPC:?XRPL_CLUSTER_RPC is required}"

ASSIGN_DIR="${1:?assignment dir is required}"
RUN_BASE="${2:-.tmp/full_ledger_meta_parallel_$(date +%Y%m%d_%H%M%S)}"

WORKERS_QN="${WORKERS_QN:-1}"
WORKERS_S1="${WORKERS_S1:-18}"
WORKERS_S2="${WORKERS_S2:-16}"
WORKERS_RUSTY="${WORKERS_RUSTY:-12}"
WORKERS_CLUSTER="${WORKERS_CLUSTER:-2}"

[[ -d "$ASSIGN_DIR" ]] || { echo "assignment dir not found: $ASSIGN_DIR"; exit 1; }

QN_LEDGER="$ASSIGN_DIR/ledgers_qn.txt"
S1_LEDGER="$ASSIGN_DIR/ledgers_s1.txt"
S2_LEDGER="$ASSIGN_DIR/ledgers_s2.txt"
RUSTY_LEDGER="$ASSIGN_DIR/ledgers_rusty.txt"
CLUSTER_LEDGER="$ASSIGN_DIR/ledgers_cluster.txt"

QN_REQ="$ASSIGN_DIR/required_tx_qn.csv"
S1_REQ="$ASSIGN_DIR/required_tx_s1.csv"
S2_REQ="$ASSIGN_DIR/required_tx_s2.csv"
RUSTY_REQ="$ASSIGN_DIR/required_tx_rusty.csv"
CLUSTER_REQ="$ASSIGN_DIR/required_tx_cluster.csv"

for f in \
  "$QN_LEDGER" "$S1_LEDGER" "$S2_LEDGER" "$RUSTY_LEDGER" "$CLUSTER_LEDGER" \
  "$QN_REQ" "$S1_REQ" "$S2_REQ" "$RUSTY_REQ" "$CLUSTER_REQ"; do
  [[ -f "$f" ]] || { echo "missing assignment file: $f"; exit 1; }
done

mkdir -p "$RUN_BASE/logs" "$RUN_BASE/pids"

OUT_QN="$RUN_BASE/full_ledger_qn"
OUT_S1="$RUN_BASE/full_ledger_s1"
OUT_S2="$RUN_BASE/full_ledger_s2"
OUT_RUSTY="$RUN_BASE/full_ledger_rusty"
OUT_CLUSTER="$RUN_BASE/full_ledger_cluster"
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

launch "quicknode" \
  "cd '$ROOT_DIR' && python -u empirical/scripts/empirical_fetch_full_ledger_metadata.py --rpc '$XRPL_QN_RPC' --ledger-list '$QN_LEDGER' --required-tx-csv '$QN_REQ' --outdir '$OUT_QN' --workers '$WORKERS_QN' --timeout 30 --retries 4 --progress-every 25"

launch "ripple_s1" \
  "cd '$ROOT_DIR' && python -u empirical/scripts/empirical_fetch_full_ledger_metadata.py --rpc '$XRPL_RIPPLE_S1_RPC' --ledger-list '$S1_LEDGER' --required-tx-csv '$S1_REQ' --outdir '$OUT_S1' --workers '$WORKERS_S1' --timeout 30 --retries 4 --progress-every 25"

launch "ripple_s2" \
  "cd '$ROOT_DIR' && python -u empirical/scripts/empirical_fetch_full_ledger_metadata.py --rpc '$XRPL_RIPPLE_S2_RPC' --ledger-list '$S2_LEDGER' --required-tx-csv '$S2_REQ' --outdir '$OUT_S2' --workers '$WORKERS_S2' --timeout 30 --retries 4 --progress-every 25"

launch "rustychain" \
  "cd '$ROOT_DIR' && python -u empirical/scripts/empirical_fetch_full_ledger_metadata.py --rpc '$XRPL_RUSTYCHAIN_RPC' --ledger-list '$RUSTY_LEDGER' --required-tx-csv '$RUSTY_REQ' --outdir '$OUT_RUSTY' --workers '$WORKERS_RUSTY' --timeout 30 --retries 4 --progress-every 25"

launch "xrplcluster" \
  "cd '$ROOT_DIR' && python -u empirical/scripts/empirical_fetch_full_ledger_metadata.py --rpc '$XRPL_CLUSTER_RPC' --ledger-list '$CLUSTER_LEDGER' --required-tx-csv '$CLUSTER_REQ' --outdir '$OUT_CLUSTER' --workers '$WORKERS_CLUSTER' --timeout 30 --retries 4 --progress-every 10"

cat > "$RUN_BASE/RUN_INFO.txt" <<EOF
ROOT_DIR=$ROOT_DIR
ASSIGN_DIR=$ASSIGN_DIR
RUN_BASE=$RUN_BASE
OUT_QN=$OUT_QN
OUT_S1=$OUT_S1
OUT_S2=$OUT_S2
OUT_RUSTY=$OUT_RUSTY
OUT_CLUSTER=$OUT_CLUSTER
WORKERS_QN=$WORKERS_QN
WORKERS_S1=$WORKERS_S1
WORKERS_S2=$WORKERS_S2
WORKERS_RUSTY=$WORKERS_RUSTY
WORKERS_CLUSTER=$WORKERS_CLUSTER
EOF

echo
echo "started all full-ledger jobs"
echo "run_base: $RUN_BASE"
echo "logs: $RUN_BASE/logs"
echo
echo "monitor examples:"
echo "  tail -f $RUN_BASE/logs/quicknode.log"
echo "  tail -f $RUN_BASE/logs/ripple_s1.log"
echo "  tail -f $RUN_BASE/logs/ripple_s2.log"
echo "  tail -f $RUN_BASE/logs/rustychain.log"
echo "  tail -f $RUN_BASE/logs/xrplcluster.log"
