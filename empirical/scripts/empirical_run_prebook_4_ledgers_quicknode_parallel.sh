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

: "${XRPL_RUSD_ISSUER:?XRPL_RUSD_ISSUER is required}"
: "${XRPL_RUSD_HEX:?XRPL_RUSD_HEX is required}"

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

OUT_BASE="${1:-artifacts/prebook/rlusd_xrp/run_qn_4_ledgers_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_BASE/logs"

# target tx ledgers -> prebook ledgers (tx-1)
L1=100785277
L2=100805401
L3=100959684
L4=100962472

echo "$L1" > "$OUT_BASE/ledger_01.txt"
echo "$L2" > "$OUT_BASE/ledger_02.txt"
echo "$L3" > "$OUT_BASE/ledger_03.txt"
echo "$L4" > "$OUT_BASE/ledger_04.txt"

run_one() {
  local name="$1"
  local rpc="$2"
  local ll="$3"
  local outd="$4"
  local log="$OUT_BASE/logs/${name}.log"
  mkdir -p "$outd"
  echo "[launch] $name ledger=$(cat "$ll")"
  echo "  rpc: $rpc"
  echo "  log: $log"
  nohup python -u empirical/scripts/empirical_fetch_prebook_from_ledger_data.py \
    --rpc "$rpc" \
    --ledger-list "$ll" \
    --outdir "$outd" \
    --issuer "$XRPL_RUSD_ISSUER" \
    --currency-hex "$XRPL_RUSD_HEX" \
    --limit 400 \
    --page-limit 256 \
    --timeout 12 \
    --retries 1 \
    --progress-pages 10 >"$log" 2>&1 &
  echo $! > "$OUT_BASE/${name}.pid"
}

run_one "qn_1" "${RPCS[0]}" "$OUT_BASE/ledger_01.txt" "$OUT_BASE/qn_1"
run_one "qn_2" "${RPCS[$((1 % ${#RPCS[@]}))]}" "$OUT_BASE/ledger_02.txt" "$OUT_BASE/qn_2"
run_one "qn_3" "${RPCS[$((2 % ${#RPCS[@]}))]}" "$OUT_BASE/ledger_03.txt" "$OUT_BASE/qn_3"
run_one "qn_4" "${RPCS[$((3 % ${#RPCS[@]}))]}" "$OUT_BASE/ledger_04.txt" "$OUT_BASE/qn_4"

cat > "$OUT_BASE/RUN_INFO.txt" <<EOF
OUT_BASE=$OUT_BASE
L1=$L1
L2=$L2
L3=$L3
L4=$L4
EOF

echo
echo "started 4 jobs"
echo "out_base: $OUT_BASE"
echo "logs: $OUT_BASE/logs"
echo
echo "monitor:"
echo "  tail -f $OUT_BASE/logs/qn_1.log"
echo "  tail -f $OUT_BASE/logs/qn_2.log"
echo "  tail -f $OUT_BASE/logs/qn_3.log"
echo "  tail -f $OUT_BASE/logs/qn_4.log"
