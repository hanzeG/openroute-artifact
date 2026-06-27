#!/usr/bin/env bash
set -euo pipefail

# Batch export Delta Sharing data by day for a 2-week window.
# Exports all three tables used by empirical_export_window.py:
#   - fact_amm_swaps
#   - offers_fact_tx
#   - fact_amm_fees

usage() {
  cat <<'EOF'
Usage:
  empirical/scripts/empirical_export_two_week_window.sh [options]

Options:
  --start-date YYYY-MM-DD   Inclusive start date (default: 2025-12-08)
  --days N                  Number of days to export (default: 14)
  --pair KEY                Pair key for output path naming (default: rlusd_xrp)
  --share-profile PATH      Delta Sharing profile (default: data/config.share)
  --conda-env NAME          Conda env used to run spark-submit (default: xrpl-amm-clob)
  --out-base PATH           Output base dir (default: artifacts/exports/<pair>)
  --retries N               Retries per day (default: 3)
  --spark-shuffle N         spark.sql.shuffle.partitions (default: 8)
  --spark-parallelism N     spark.default.parallelism (default: 8)
  --help                    Show this help

Example:
  empirical/scripts/empirical_export_two_week_window.sh \
    --start-date 2025-12-08 \
    --days 14 \
    --pair rlusd_xrp \
    --share-profile data/config.share
EOF
}

START_DATE="2025-12-08"
DAYS=14
PAIR="rlusd_xrp"
SHARE_PROFILE="data/config.share"
CONDA_ENV="xrpl-amm-clob"
OUT_BASE=""
RETRIES=3
SPARK_SHUFFLE=8
SPARK_PARALLELISM=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-date)
      START_DATE="$2"
      shift 2
      ;;
    --days)
      DAYS="$2"
      shift 2
      ;;
    --pair)
      PAIR="$2"
      shift 2
      ;;
    --share-profile)
      SHARE_PROFILE="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --out-base)
      OUT_BASE="$2"
      shift 2
      ;;
    --retries)
      RETRIES="$2"
      shift 2
      ;;
    --spark-shuffle)
      SPARK_SHUFFLE="$2"
      shift 2
      ;;
    --spark-parallelism)
      SPARK_PARALLELISM="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$OUT_BASE" ]]; then
  OUT_BASE="artifacts/exports/${PAIR}"
fi

python - <<PY >/dev/null
from datetime import date
date.fromisoformat("${START_DATE}")
assert int("${DAYS}") > 0
assert int("${RETRIES}") > 0
PY

mkdir -p "$OUT_BASE"

date_plus_one() {
  python - <<PY
from datetime import date, timedelta
print((date.fromisoformat("$1") + timedelta(days=1)).isoformat())
PY
}

date_plus_n() {
  python - <<PY
from datetime import date, timedelta
print((date.fromisoformat("$1") + timedelta(days=int("$2"))).isoformat())
PY
}

CURRENT="$START_DATE"
END_DATE="$(date_plus_n "$START_DATE" "$DAYS")"
OK=0
FAIL=0
PROCESSED=0

render_progress() {
  local state="$1"
  local pct=0
  if [[ "$DAYS" -gt 0 ]]; then
    pct=$((PROCESSED * 100 / DAYS))
  fi
  local width=28
  local filled=$((pct * width / 100))
  local empty=$((width - filled))
  local bar
  bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' "$empty" '' | tr ' ' '-')"
  printf "\r[%s] %3d%%  %d/%d  ok=%d fail=%d  %s" "$bar" "$pct" "$PROCESSED" "$DAYS" "$OK" "$FAIL" "$state"
}

echo "Export window: ${START_DATE} (inclusive) -> ${END_DATE} (exclusive)"
echo "Output base : ${OUT_BASE}"
echo "Conda env   : ${CONDA_ENV}"
echo "Progress:"
render_progress "starting"

while [[ "$CURRENT" != "$END_DATE" ]]; do
  NEXT="$(date_plus_one "$CURRENT")"
  OUT_DIR="${OUT_BASE}/date_${CURRENT}"
  LOG_DIR="${OUT_DIR}/_logs"
  mkdir -p "$LOG_DIR"

  if [[ -d "${OUT_DIR}/amm_swaps" && -d "${OUT_DIR}/clob_legs" && -d "${OUT_DIR}/amm_fees" ]]; then
    OK=$((OK + 1))
    PROCESSED=$((PROCESSED + 1))
    render_progress "${CURRENT} skipped"
    CURRENT="$NEXT"
    continue
  fi

  SUCCESS=0

  for ((A=1; A<=RETRIES; A++)); do
    LOG_FILE="${LOG_DIR}/attempt_${A}.log"
    render_progress "${CURRENT} attempt ${A}/${RETRIES}"
    if conda run -n "$CONDA_ENV" spark-submit \
      --conf "spark.master=local[*]" \
      --conf "spark.sql.shuffle.partitions=${SPARK_SHUFFLE}" \
      --conf "spark.default.parallelism=${SPARK_PARALLELISM}" \
      --conf "spark.task.maxFailures=8" \
      --conf "spark.sql.adaptive.enabled=true" \
      --conf "spark.sql.adaptive.coalescePartitions.enabled=true" \
      --conf "spark.jars.packages=io.delta:delta-sharing-spark_2.12:3.1.0" \
      --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension" \
      --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog" \
      --conf "spark.delta.sharing.client.http.timeout=300s" \
      --conf "spark.delta.sharing.client.http.maxRetries=8" \
      --conf "spark.delta.sharing.client.http.retryInterval=3s" \
      empirical/scripts/empirical_export_window.py \
      --share-profile "$SHARE_PROFILE" \
      --pair "$PAIR" \
      --date-start "$CURRENT" \
      --date-end "$NEXT" \
      --output-dir "$OUT_DIR" >"$LOG_FILE" 2>&1; then
      SUCCESS=1
      OK=$((OK + 1))
      PROCESSED=$((PROCESSED + 1))
      render_progress "${CURRENT} done"
      break
    fi

    render_progress "${CURRENT} retrying (${A}/${RETRIES})"
    sleep $((A * 3))
  done

  if [[ "$SUCCESS" -eq 0 ]]; then
    FAIL=$((FAIL + 1))
    PROCESSED=$((PROCESSED + 1))
    render_progress "${CURRENT} failed (see ${LOG_DIR})"
  fi

  CURRENT="$NEXT"
done

echo
echo "Summary: ok=${OK}, fail=${FAIL}, total_days=${DAYS}"
if [[ "$FAIL" -gt 0 ]]; then
  echo "Some days failed. Re-run this script to resume failed/missing days."
  exit 1
fi
