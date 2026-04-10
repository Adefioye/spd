#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../.. && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension"
SESSION_NAME="${1:-exp07_tms_depth}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$EXPERIMENT_DIR/logs"
LOG_PATH="$LOG_DIR/exp_07_batch_${STAMP}.log"

mkdir -p "$LOG_DIR"

tmux new-session -d -s "$SESSION_NAME" \
  "cd \"$REPO_ROOT\" && \
   source .venv/bin/activate && \
   export SPD_OUT_DIR=\"\$PWD/spd_out\" && \
   python \"$EXPERIMENT_DIR/run_exp_07_batch.py\" \"$EXPERIMENT_DIR/exp_07_batch.yaml\" \
     --device cuda \
     --spd-replicates 3 \
     2>&1 | tee \"$LOG_PATH\""

echo "session_name=$SESSION_NAME"
echo "log_path=$LOG_PATH"
echo "attach_cmd=tmux attach -t $SESSION_NAME"
