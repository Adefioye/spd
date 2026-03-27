#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../.. && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/koko_experiments/parameter_recovery/experiments/exp_08_tms_40_10_depth_untied_extension"
SESSION_NAME="${1:-exp08_tms_40_10_untied}"
DEVICE="${2:-cuda}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$EXPERIMENT_DIR/logs"
LOG_PATH="$LOG_DIR/exp_08_batch_${STAMP}.log"

mkdir -p "$LOG_DIR"

INNER_CMD="cd \"$REPO_ROOT\" && \
source .venv/bin/activate && \
export SPD_OUT_DIR=\"\$PWD/spd_out\" && "

if [[ "$DEVICE" == "cpu" ]]; then
  INNER_CMD+="export CUDA_VISIBLE_DEVICES='' && "
fi

INNER_CMD+="python \"$EXPERIMENT_DIR/run_exp_08_batch.py\" \"$EXPERIMENT_DIR/exp_08_batch.yaml\" --device \"$DEVICE\" 2>&1 | tee \"$LOG_PATH\""

tmux new-session -d -s "$SESSION_NAME" /bin/zsh -lc "$INNER_CMD"

echo "session_name=$SESSION_NAME"
echo "device=$DEVICE"
echo "log_path=$LOG_PATH"
echo "attach_cmd=tmux attach -t $SESSION_NAME"
