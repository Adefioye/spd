#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash koko_experiments/bash_script/run_alignment_report.sh
#   bash koko_experiments/bash_script/run_alignment_report.sh /path/to/spd_out/spd/s-xxxx
#
# Behavior:
# - Exports RUN_DIR
# - Runs the TMS alignment report for that run
# - Stores report JSON under $SPD_OUT_DIR/tms_alignment/<run_id>/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export SPD_OUT_DIR="${SPD_OUT_DIR:-${REPO_ROOT}/spd_out}"

if [[ "${1:-}" != "" ]]; then
  export RUN_DIR="$1"
else
  export RUN_DIR="$(ls -td "${SPD_OUT_DIR}"/spd/s-* 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${RUN_DIR}" ]]; then
  echo "No SPD run found. Expected something like: ${SPD_OUT_DIR}/spd/s-*"
  exit 1
fi

RUN_ID="$(basename "${RUN_DIR}")"
ALIGN_DIR="${SPD_OUT_DIR}/tms_alignment/${RUN_ID}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_PATH="${ALIGN_DIR}/alignment_report_${TIMESTAMP}.json"
LATEST_REPORT_PATH="${ALIGN_DIR}/alignment_report_latest.json"

mkdir -p "${ALIGN_DIR}"

echo "SPD_OUT_DIR=${SPD_OUT_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "ALIGN_DIR=${ALIGN_DIR}"

cd "${REPO_ROOT}"
uv run python koko_experiments/scripts/tms_alignment_report.py --run-dir "${RUN_DIR}" \
  | tee "${REPORT_PATH}" \
  | tee "${LATEST_REPORT_PATH}"

echo "Saved report: ${REPORT_PATH}"
echo "Updated latest: ${LATEST_REPORT_PATH}"
