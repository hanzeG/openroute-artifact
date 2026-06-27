#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash empirical/scripts/empirical_monitor_prebook_parallel.sh <run_base> [interval_sec]"
  exit 1
fi

RUN_BASE="$1"
INTERVAL="${2:-8}"

if [[ ! -d "$RUN_BASE" ]]; then
  echo "run_base not found: $RUN_BASE"
  exit 1
fi

status_pid() {
  local pid_file="$1"
  if [[ ! -f "$pid_file" ]]; then
    echo "no-pid"
    return
  fi
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    echo "no-pid"
    return
  fi
  if kill -0 "$pid" 2>/dev/null; then
    echo "alive($pid)"
  else
    echo "dead($pid)"
  fi
}

count_lines() {
  local f="$1"
  if [[ -f "$f" ]]; then
    wc -l < "$f" | tr -d ' '
  else
    echo "0"
  fi
}

last_progress() {
  local log="$1"
  if [[ ! -f "$log" ]]; then
    echo "-"
    return
  fi
  local line
  # Progress lines are often written with '\r' (single-line updates).
  # Convert '\r' -> '\n' in the latest chunk, then pick the newest progress snapshot.
  line="$(
    tail -c 200000 "$log" 2>/dev/null \
      | tr '\r' '\n' \
      | grep -E "\\[[0-9]+/[0-9]+" \
      | tail -n 1 || true
  )"
  if [[ -z "$line" ]]; then
    line="$(tail -n 1 "$log" 2>/dev/null || true)"
  fi
  line="${line//$'\r'/ }"
  line="$(echo "$line" | sed -E 's/[[:space:]]+/ /g' | sed -E 's/^ //; s/ $//')"
  # Keep enough width so eff_reqps / eff_lgrps are visible.
  line="${line:0:260}"
  echo "$line"
}

print_row() {
  local name="$1"
  local out="$2"
  local pidf="$RUN_BASE/pids/${name}.pid"
  local logf="$RUN_BASE/logs/${name}.log"
  local xrpf="$out/book_rusd_xrp_getsXRP.ndjson"
  local rusdf="$out/book_rusd_xrp_getsrUSD.ndjson"
  local failf="$out/book_rusd_xrp_fail.ndjson"
  local xcnt rcnt fcnt st prog
  xcnt="$(count_lines "$xrpf")"
  rcnt="$(count_lines "$rusdf")"
  fcnt="$(count_lines "$failf")"
  st="$(status_pid "$pidf")"
  prog="$(last_progress "$logf")"
  printf "%-14s %-12s okX=%-7s okR=%-7s fail=%-6s %s\n" "$name" "$st" "$xcnt" "$rcnt" "$fcnt" "$prog"
}

while true; do
  clear
  echo "prebook monitor | run_base=$RUN_BASE | ts=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "--------------------------------------------------------------------------------------------------------------"
  shopt -s nullglob
  pid_files=("$RUN_BASE"/pids/*.pid)
  shopt -u nullglob
  if [[ ${#pid_files[@]} -eq 0 ]]; then
    echo "no pid files found under $RUN_BASE/pids"
  else
    for pid_file in "${pid_files[@]}"; do
      name="$(basename "$pid_file" .pid)"
      print_row "$name" "$RUN_BASE/prebook_${name}"
    done
  fi
  echo "--------------------------------------------------------------------------------------------------------------"
  echo "stop monitor: Ctrl+C"
  sleep "$INTERVAL"
done
