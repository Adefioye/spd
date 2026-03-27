#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../.. && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/koko_experiments/parameter_recovery/experiments/exp_08_tms_40_10_depth_untied_extension"
SESSION_NAME="${1:-exp08_tms_40_10_untied}"
DEVICE="${2:-cuda}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$EXPERIMENT_DIR/logs"
LOG_PATH="$LOG_DIR/exp_08_batch_${STAMP}.log"
RUNNER_PATH="/tmp/${SESSION_NAME}_exp08_runner.sh"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

if [[ "$DEVICE" != "cpu" && "$DEVICE" != cuda* ]]; then
  echo "Unsupported device: $DEVICE (expected cpu or cuda[:index])" >&2
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
if [[ "$DEVICE" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi
if [[ "$DEVICE" != "cpu" ]]; then
  python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA not available in this environment"
try:
    torch.empty((1,), device="cuda")
except Exception as exc:
    raise RuntimeError(f"CUDA allocation preflight failed: {exc}") from exc
print("cuda_preflight=ok")
PY
fi
python "$EXPERIMENT_DIR/run_exp_08_batch.py" "$EXPERIMENT_DIR/exp_08_batch.yaml" --device "$DEVICE"
EOF
chmod +x "$RUNNER_PATH"

tmux new-session -d -s "$SESSION_NAME" /bin/bash -lc "$RUNNER_PATH"

echo "session_name=$SESSION_NAME"
echo "device=$DEVICE"
echo "log_path=$LOG_PATH"
echo "attach_cmd=tmux attach -t $SESSION_NAME"
