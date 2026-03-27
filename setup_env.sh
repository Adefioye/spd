#!/usr/bin/env bash
set -euo pipefail

# RunPod setup for koko_spd.
#
# Defaults:
# - repo checked out at /workspace/koko_spd
# - virtualenv stored on local disk for faster I/O
# - SPD outputs stored in the repo at /workspace/koko_spd/spd_out
# - frontend install is optional
#
# Usage:
#   bash setup_env.sh
#   source .venv/bin/activate
#   python -m koko_experiments.parameter_recovery.run_tms_hyperparameter_grid_search \
#     koko_experiments/parameter_recovery/experiments/exp_05_tms_5_2_hyperparameter_grid_search

REPO_DIR="${REPO_DIR:-/workspace/koko_spd}"
LOCAL_VENV="${LOCAL_VENV:-/root/koko_spd_venv}"
SYMLINK="${REPO_DIR}/.venv"
SPD_OUT_DIR="${SPD_OUT_DIR:-${REPO_DIR}/spd_out}"
INSTALL_APP="${INSTALL_APP:-0}"
INSTALL_DEV="${INSTALL_DEV:-1}"
RECREATE_VENV="${RECREATE_VENV:-0}"

echo "Repo dir: ${REPO_DIR}"
echo "Local venv: ${LOCAL_VENV}"
echo "SPD_OUT_DIR: ${SPD_OUT_DIR}"

if [ ! -d "${REPO_DIR}" ]; then
    echo "Repo directory not found: ${REPO_DIR}" >&2
    exit 1
fi

apt-get update -qq
apt-get install -y -qq curl git build-essential ca-certificates zsh nvidia-modprobe

mkdir -p /dev/char
if [ -e /dev/nvidiactl ]; then ln -sf /dev/nvidiactl /dev/char/195:255; fi
if [ -e /dev/nvidia-modeset ]; then ln -sf /dev/nvidia-modeset /dev/char/195:254; fi
if [ -e /dev/nvidia5 ]; then ln -sf /dev/nvidia5 /dev/char/195:5; fi
if [ -e /dev/nvidia-uvm ]; then ln -sf /dev/nvidia-uvm /dev/char/506:0; fi
if [ -e /dev/nvidia-uvm-tools ]; then ln -sf /dev/nvidia-uvm-tools /dev/char/506:1; fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

uv python install 3.13

if [ "${RECREATE_VENV}" = "1" ] && [ -d "${LOCAL_VENV}" ]; then
    rm -rf "${LOCAL_VENV}"
fi

if [ ! -x "${LOCAL_VENV}/bin/python" ]; then
    uv venv --python 3.13 "${LOCAL_VENV}"
fi

if [ -L "${SYMLINK}" ] || [ -d "${SYMLINK}" ]; then
    rm -rf "${SYMLINK}"
fi
ln -s "${LOCAL_VENV}" "${SYMLINK}"

cd "${REPO_DIR}"
source "${LOCAL_VENV}/bin/activate"

export SPD_OUT_DIR
export UV_LINK_MODE=copy

make copy-templates

if [ "${INSTALL_DEV}" = "1" ]; then
    uv sync --frozen
else
    uv sync --frozen --no-dev
fi

if [ "${INSTALL_APP}" = "1" ]; then
    if ! command -v node >/dev/null 2>&1; then
        curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
        apt-get install -y -qq nodejs
    fi
    make install-app
fi

python - <<'PY'
import fire
import sae_lens
import torch
from jaxtyping import Float

import spd
from spd.configs import Config
from spd.experiments.tms.models import TMSModel
from spd.run_spd import run_experiment

import koko_experiments.parameter_recovery.analyze_spd_run
import koko_experiments.parameter_recovery.feature_datasets
import koko_experiments.parameter_recovery.run_parameter_recovery
import koko_experiments.parameter_recovery.run_tms_hyperparameter_grid_search
import koko_experiments.parameter_recovery.synthetic

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_version={torch.version.cuda}")
print("koko_experiments imports OK")
PY

echo ""
echo "Setup complete."
echo "Activate with: source ${SYMLINK}/bin/activate"
