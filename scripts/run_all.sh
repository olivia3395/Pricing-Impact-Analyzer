#!/usr/bin/env bash
# run_all.sh
# ----------
# End-to-end pipeline runner. Logs output to artifacts/pipeline.log.
#
# Usage:
#   ./scripts/run_all.sh                      # full run
#   ./scripts/run_all.sh --skip-download      # skip download (data already present)
#   ./scripts/run_all.sh --steps 5,6,7,10,11  # specific steps only

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/artifacts/pipeline.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
SKIP_DOWNLOAD=""
STEPS_ARG=""

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --skip-download) SKIP_DOWNLOAD="--skip-download" ;;
    --steps=*)       STEPS_ARG="--steps=${arg#*=}" ;;
  esac
done

mkdir -p "${PROJECT_ROOT}/artifacts"

echo "============================================================" | tee -a "$LOG_FILE"
echo "  Pipeline Start: ${TIMESTAMP}" | tee -a "$LOG_FILE"
echo "  Skip download: ${SKIP_DOWNLOAD:-no}" | tee -a "$LOG_FILE"
echo "  Steps: ${STEPS_ARG:-all}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

cd "${PROJECT_ROOT}"

# ── Time each step ────────────────────────────────────────────────────────────
run_step() {
  local step_name="$1"
  shift
  echo "" | tee -a "$LOG_FILE"
  echo ">>> ${step_name}" | tee -a "$LOG_FILE"
  local start_ts=$SECONDS
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local elapsed=$(( SECONDS - start_ts ))
  echo "    Elapsed: ${elapsed}s" | tee -a "$LOG_FILE"
}

run_step "Full Pipeline" \
  python src/run_pipeline.py \
    --config configs/default.yaml \
    ${SKIP_DOWNLOAD} \
    ${STEPS_ARG}

TOTAL=$(( SECONDS ))
echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "  Pipeline COMPLETE. Total time: ${TOTAL}s" | tee -a "$LOG_FILE"
echo "  Log: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "  Figures: ${PROJECT_ROOT}/artifacts/figures/" | tee -a "$LOG_FILE"
echo "  Tables:  ${PROJECT_ROOT}/artifacts/tables/" | tee -a "$LOG_FILE"
echo "  Results: ${PROJECT_ROOT}/artifacts/results/" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  Launch dashboard:" | tee -a "$LOG_FILE"
echo "    streamlit run src/dashboard/app.py" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
