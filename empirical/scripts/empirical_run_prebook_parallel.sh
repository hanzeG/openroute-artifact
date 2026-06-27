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

: "${XRPL_RUSD_ISSUER:?XRPL_RUSD_ISSUER is required}"
: "${XRPL_RUSD_HEX:?XRPL_RUSD_HEX is required}"

ASSIGN_DIR="${1:-artifacts/prebook/rlusd_xrp/assignments}"
RUN_BASE="${2:-artifacts/prebook/rlusd_xrp/run_$(date +%Y%m%d_%H%M%S)}"
DEFAULT_WORKERS="${XRPL_PREBOOK_WORKERS:-8}"
LIMIT="${XRPL_PREBOOK_LIMIT:-100}"

[[ -d "$ASSIGN_DIR" ]] || { echo "assignment dir not found: $ASSIGN_DIR"; exit 1; }

ENDPOINT_NAMES=()
ENDPOINT_RPCS=()
ENDPOINT_WORKERS=()

add_endpoint() {
  local name="$1"
  local rpc_var="$2"
  local workers_var="$3"
  local default_workers="$4"
  local rpc="${!rpc_var:-}"
  if [[ -z "$rpc" ]]; then
    return
  fi
  local workers="${!workers_var:-$default_workers}"
  ENDPOINT_NAMES+=("$name")
  ENDPOINT_RPCS+=("$rpc")
  ENDPOINT_WORKERS+=("$workers")
}

add_endpoint "s1" "XRPL_RIPPLE_S1_RPC" "XRPL_WORKERS_S1" "$DEFAULT_WORKERS"
add_endpoint "s2" "XRPL_RIPPLE_S2_RPC" "XRPL_WORKERS_S2" "$DEFAULT_WORKERS"
add_endpoint "cluster" "XRPL_CLUSTER_RPC" "XRPL_WORKERS_CLUSTER" "$DEFAULT_WORKERS"

for i in 1 2 3 4 5 6 7 8; do
  add_endpoint "extra${i}" "XRPL_EXTRA_RPC_${i}" "XRPL_EXTRA_WORKERS_${i}" "${XRPL_EXTRA_WORKERS:-$DEFAULT_WORKERS}"
done

if [[ ${#ENDPOINT_NAMES[@]} -eq 0 ]]; then
  echo "missing XRPL RPC endpoint in .env"
  exit 1
fi

mkdir -p "$RUN_BASE/logs" "$RUN_BASE/pids"

launch() {
  local name="$1"
  local rpc="$2"
  local workers="$3"
  local ledger_file="$ASSIGN_DIR/ledgers_${name}.txt"
  local out_dir="$RUN_BASE/prebook_${name}"
  local log="$RUN_BASE/logs/${name}.log"

  [[ -f "$ledger_file" ]] || { echo "missing assignment file: $ledger_file"; exit 1; }
  mkdir -p "$out_dir"
  echo "[launch] $name workers=$workers"
  echo "  log: $log"
  nohup bash -lc \
    "cd '$ROOT_DIR' && python empirical/scripts/empirical_download_clob_offers_from_ledger_list.py --rpc '$rpc' --ledger-list '$ledger_file' --outdir '$out_dir' --issuer '$XRPL_RUSD_ISSUER' --currency-hex '$XRPL_RUSD_HEX' --workers '$workers' --limit '$LIMIT' --timeout 20 --retries 4 --max-consecutive-failures 120" \
    >"$log" 2>&1 &
  echo $! > "$RUN_BASE/pids/${name}.pid"
}

{
  echo "ROOT_DIR=$ROOT_DIR"
  echo "ASSIGN_DIR=$ASSIGN_DIR"
  echo "RUN_BASE=$RUN_BASE"
  echo "DEFAULT_WORKERS=$DEFAULT_WORKERS"
  echo "LIMIT=$LIMIT"
} > "$RUN_BASE/RUN_INFO.txt"

for idx in "${!ENDPOINT_NAMES[@]}"; do
  name="${ENDPOINT_NAMES[$idx]}"
  launch "$name" "${ENDPOINT_RPCS[$idx]}" "${ENDPOINT_WORKERS[$idx]}"
  {
    echo "ENDPOINT_${name}_OUT=$RUN_BASE/prebook_${name}"
    echo "ENDPOINT_${name}_WORKERS=${ENDPOINT_WORKERS[$idx]}"
  } >> "$RUN_BASE/RUN_INFO.txt"
done

echo
echo "started prebook jobs"
echo "run_base: $RUN_BASE"
echo "monitor:"
echo "  bash empirical/scripts/empirical_monitor_prebook_parallel.sh '$RUN_BASE'"
