#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env" ]]; then
  echo "missing .env"
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

ASSIGN_DIR="${1:?assignment dir is required}"
RUN_BASE="${2:-.tmp/full_ledger_meta_parallel_$(date +%Y%m%d_%H%M%S)}"
DEFAULT_WORKERS="${XRPL_FETCH_WORKERS:-8}"

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
  local required_tx_file="$ASSIGN_DIR/required_tx_${name}.csv"
  local out_dir="$RUN_BASE/full_ledger_${name}"
  local log="$RUN_BASE/logs/${name}.log"
  local required_arg=""

  [[ -f "$ledger_file" ]] || { echo "missing assignment file: $ledger_file"; exit 1; }
  if [[ -f "$required_tx_file" ]]; then
    required_arg="--required-tx-csv '$required_tx_file'"
  fi

  mkdir -p "$out_dir"
  echo "[launch] $name workers=$workers"
  echo "  log: $log"
  nohup bash -lc \
    "cd '$ROOT_DIR' && python -u empirical/scripts/empirical_fetch_full_ledger_metadata.py --rpc '$rpc' --ledger-list '$ledger_file' $required_arg --outdir '$out_dir' --workers '$workers' --timeout 30 --retries 4 --progress-every 25" \
    >"$log" 2>&1 &
  echo $! > "$RUN_BASE/pids/${name}.pid"
}

{
  echo "ROOT_DIR=$ROOT_DIR"
  echo "ASSIGN_DIR=$ASSIGN_DIR"
  echo "RUN_BASE=$RUN_BASE"
  echo "DEFAULT_WORKERS=$DEFAULT_WORKERS"
} > "$RUN_BASE/RUN_INFO.txt"

for idx in "${!ENDPOINT_NAMES[@]}"; do
  name="${ENDPOINT_NAMES[$idx]}"
  launch "$name" "${ENDPOINT_RPCS[$idx]}" "${ENDPOINT_WORKERS[$idx]}"
  {
    echo "ENDPOINT_${name}_OUT=$RUN_BASE/full_ledger_${name}"
    echo "ENDPOINT_${name}_WORKERS=${ENDPOINT_WORKERS[$idx]}"
  } >> "$RUN_BASE/RUN_INFO.txt"
done

echo
echo "started full-ledger jobs"
echo "run_base: $RUN_BASE"
echo "logs: $RUN_BASE/logs"
echo
echo "monitor examples:"
for name in "${ENDPOINT_NAMES[@]}"; do
  echo "  tail -f $RUN_BASE/logs/${name}.log"
done
