#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../.. && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/koko_experiments/parameter_recovery/experiments/exp_09_tms_5_2_cpu_paper_repro"
SESSION_NAME="${1:-exp09_tms_5_2_cpu_paper}"
SPD_STEPS="${2:-}"
LOG_DIR="$EXPERIMENT_DIR/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_DIR/${SESSION_NAME}_${STAMP}.log"
RUNNER_PATH="/tmp/${SESSION_NAME}_exp09_runner.sh"
CONFIG_PATH="$REPO_ROOT/koko_experiments/configs/tms_5-2_cpu_paper.yaml"
WANDB_PROJECT="spd_tms_5_2_exp09"
WANDB_RUN_NAME="exp_09_tms_5_2_cpu_paper"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

cat > "$RUNNER_PATH" <<EOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$LOG_DIR"
touch "$LOG_PATH"
exec > >(tee -a "$LOG_PATH") 2>&1
cd "$REPO_ROOT"
source .venv/bin/activate
export SPD_OUT_DIR="\$PWD/spd_out"
export CUDA_VISIBLE_DEVICES=""
CMD=(
  python
  "$EXPERIMENT_DIR/run_exp_09_cpu_paper.py"
  --config-path "$CONFIG_PATH"
  --device cpu
  --wandb-project "$WANDB_PROJECT"
  --wandb-run-name "$WANDB_RUN_NAME"
)
if [[ -n "$SPD_STEPS" ]]; then
  CMD+=(--spd-steps "$SPD_STEPS")
fi
"\${CMD[@]}"
EOF
chmod +x "$RUNNER_PATH"

tmux new-session -d -s "$SESSION_NAME" /bin/bash -lc "$RUNNER_PATH"

echo "session_name=$SESSION_NAME"
echo "config_path=$CONFIG_PATH"
if [[ -n "$SPD_STEPS" ]]; then
  echo "spd_steps=$SPD_STEPS"
fi
echo "log_path=$LOG_PATH"
echo "attach_cmd=tmux attach -t $SESSION_NAME"
