#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../.. && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension"
SESSION_NAME="${1:-exp07_tms_delta_shrinkage}"
BATCH_CONFIG_NAME="exp_07_delta_component_shrinkage_batch.yaml"
BATCH_CONFIG_PATH="$EXPERIMENT_DIR/$BATCH_CONFIG_NAME"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$EXPERIMENT_DIR/logs"
LOG_PATH="$LOG_DIR/${BATCH_CONFIG_NAME%.yaml}_${STAMP}.log"
RUNNER_PATH="/tmp/${SESSION_NAME}_exp07_delta_component_shrinkage.sh"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

if [[ ! -f "$BATCH_CONFIG_PATH" ]]; then
  echo "Batch config not found: $BATCH_CONFIG_PATH" >&2
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
python "$EXPERIMENT_DIR/run_exp_07_batch.py" "$BATCH_CONFIG_PATH"
EOF
chmod +x "$RUNNER_PATH"

tmux new-session -d -s "$SESSION_NAME" /bin/bash -lc "$RUNNER_PATH"

echo "session_name=$SESSION_NAME"
echo "batch_config=$BATCH_CONFIG_PATH"
echo "log_path=$LOG_PATH"
echo "attach_cmd=tmux attach -t $SESSION_NAME"
