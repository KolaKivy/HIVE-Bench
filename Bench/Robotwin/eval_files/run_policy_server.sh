#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/Policy:${REPO_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash Bench/Robotwin/eval_files/run_policy_server.sh <ckpt_path> [gpu_id] [port]" >&2
    exit 1
fi

your_ckpt="$1"
if [[ -d "${your_ckpt}" ]]; then
    if [[ -f "${your_ckpt}/pytorch_model.pt" ]]; then
        your_ckpt="${your_ckpt}/pytorch_model.pt"
    elif [[ -f "${your_ckpt}/pytorch_model_converted.pt" ]]; then
        echo "[WARN] No pytorch_model.pt found; falling back to converted checkpoint." >&2
        your_ckpt="${your_ckpt}/pytorch_model_converted.pt"
    fi
fi
[[ -f "${your_ckpt}" ]] || { echo "Checkpoint does not exist: ${your_ckpt}" >&2; exit 1; }
gpu_id="${2:-${ROBOTWIN_SERVER_GPU:-5}}"
port="${3:-${ROBOTWIN_SERVER_PORT:-5694}}"
hivebench_python="${HIVEBENCH_PYTHON:-python}"

if [[ "${HIVEBENCH_OFFLINE:-0}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
fi

use_bf16_flag=()
if [[ "${ROBOTWIN_USE_BF16:-0}" == "1" ]]; then
    use_bf16_flag+=(--use_bf16)
fi

echo "[INFO] Starting RoboTwin policy server"
echo "[INFO] checkpoint: ${your_ckpt}"
echo "[INFO] gpu: ${gpu_id}"
echo "[INFO] port: ${port}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${hivebench_python}" "${REPO_ROOT}/Policy/deployment/model_server/server_policy.py" \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" "${use_bf16_flag[@]}"
