#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  source .env
else
  echo "missing .env (copy from .env.example first)"
  exit 1
fi

: "${XRPL_RUSD_ISSUER:?XRPL_RUSD_ISSUER is required in .env}"
: "${XRPL_RUSD_HEX:?XRPL_RUSD_HEX is required in .env}"

if [[ $# -lt 2 ]]; then
  echo "Usage:"
  echo "  bash empirical/scripts/empirical_run_prebook_quicknode_pool.sh <ledger_list.txt> <outdir> [workers] [limit]"
  exit 1
fi

LEDGER_LIST="$1"
OUTDIR="$2"
WORKERS="${3:-12}"
LIMIT="${4:-100}"

mkdir -p "$OUTDIR"

RPCS=()
for var in XRPL_QN_RPC XRPL_QN_RPC_1 XRPL_QN_RPC_2 XRPL_QN_RPC_3 XRPL_QN_RPC_4; do
  if [[ -n "${!var:-}" ]]; then
    RPCS+=("${!var}")
  fi
done

if [[ ${#RPCS[@]} -eq 0 ]]; then
  echo "missing QuickNode RPC in .env; set XRPL_QN_RPC"
  exit 1
fi

for i in "${!RPCS[@]}"; do
  idx=$((i + 1))
  rpc="${RPCS[$i]}"
  echo
  echo "=== QuickNode pool step ${idx}/${#RPCS[@]} ==="
  echo "rpc=${rpc}"

  python empirical/scripts/empirical_download_clob_offers_from_ledger_list.py \
    --rpc "${rpc}" \
    --ledger-list "${LEDGER_LIST}" \
    --outdir "${OUTDIR}" \
    --issuer "${XRPL_RUSD_ISSUER}" \
    --currency-hex "${XRPL_RUSD_HEX}" \
    --workers "${WORKERS}" \
    --limit "${LIMIT}" \
    --timeout 20 \
    --retries 4 \
    --max-consecutive-failures 120
done

echo
echo "QuickNode rotation finished."
echo "If pending still exists, rerun this same command; script will resume from existing outdir."
