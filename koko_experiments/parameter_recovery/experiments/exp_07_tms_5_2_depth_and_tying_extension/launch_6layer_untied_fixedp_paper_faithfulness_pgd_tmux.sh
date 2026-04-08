#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../.. && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension"
SESSION_NAME="${1:-exp07_tms_6layer_untied_fixedp_paper_faithfulness_pgd}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$EXPERIMENT_DIR/logs"
LOG_PATH="$LOG_DIR/exp_07_6layer_untied_fixedp_paper_faithfulness_pgd_${STAMP}.log"

mkdir -p "$LOG_DIR"

tmux new-session -d -s "$SESSION_NAME" \
  "cd \"$REPO_ROOT\" && \
   source .venv/bin/activate && \
   export SPD_OUT_DIR=\"\$PWD/spd_out\" && \
   export CUDA_VISIBLE_DEVICES='' && \
   python \"$EXPERIMENT_DIR/run_exp_07_4layer_untied_loss_ablations.py\" \"$EXPERIMENT_DIR/exp_07_6layer_untied_fixedp_paper_faithfulness_pgd.yaml\" 2>&1 | tee \"$LOG_PATH\""

echo "session_name=$SESSION_NAME"
echo "log_path=$LOG_PATH"
echo "attach_cmd=tmux attach -t $SESSION_NAME"
