#!/bin/bash
# HIVE-Bench: 16-step IDM visual-encoder evaluation.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-playground/Checkpoints}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

# User configuration
MODE="${MODE:-idm}"       # idm | fdm | state | object
DATASET="${DATASET:-robocasa}" # robocasa | robotwin
MODEL="${MODEL:-dinov3_base}"      # encoder/VLM name (see SUPPORTED_MODELS below)
# Full policy checkpoint. Leave empty to evaluate the pretrained base encoder.
FINETUNED_ENCODER_CHECKPOINT="${FINETUNED_ENCODER_CHECKPOINT:-}"
# VLM token width is model-dependent; set ENCODER_DIM when running a VLM.
ENCODER_DIM="${ENCODER_DIM:-}"
EPOCHS="${EPOCHS:-4}"

# All names below are routed through the same IDM/FDM/state/object probe.
SUPPORTED_MODELS=(
  dinov2_small dinov2_base dinov2_large dinov2_giant
  dinov3_small dinov3_base dinov3_large dinov3_huge
  cradio_so400m vc1_large spa_large
  vjepa2_1_large vjepa2_1_giant vjepa2.1_large vjepa2.1_giant
  depthvlm depthvlm_layer16 lingbot_large qwen3 qwen3_layer16 xiaomi xiaomi_layer16
)

if [ "$DATASET" = "robotwin" ]; then
    CONFIG_NAME="idm_fdm_robotwin"
    DATA_ROOT="${ROBOTWIN_DATA_ROOT:-playground/RoboTwin_LeRobot_HeadCam/Randomized}"
    OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/IDM_eval_robotwin}"
else
    CONFIG_NAME="idm_fdm_robocasa"
    DATA_ROOT="${ROBOCASA_DATA_ROOT:-playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim}"
    OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/IDM_eval_16step}"
fi

# These two wrappers import their upstream repositories at construction time.
case "${MODEL}" in
  spa_base|spa_large|spa_base_ft|spa_large_ft) export PYTHONPATH="${SPA_ROOT:-third_party/SPA}:${PYTHONPATH:-}" ;;
  vggt_omega|vggt_omega_register|vggt_omega_ft) export PYTHONPATH="${VGGT_OMEGA_ROOT:-third_party/VGGT_Omega}:${PYTHONPATH:-}" ;;
esac
# Friendly ft aliases: restore only the vision_encoder.* tensors from the policy.
if [[ "$MODEL" == "dinov2_base_ft" ]]; then MODEL="dinov2_base"; FINETUNED_ENCODER_CHECKPOINT="playground/Checkpoints/all_task_dinov2_base_ft_robotwin_new/final_model/pytorch_model.pt"; fi
if [[ "$MODEL" == "dinov3_base_ft" ]]; then MODEL="dinov3_base"; FINETUNED_ENCODER_CHECKPOINT="playground/Checkpoints/all_task_dinov3_base_ft_robotwin_new/final_model/pytorch_model.pt"; fi
if [[ "$MODEL" == "spa_base_ft" ]]; then MODEL="spa_base"; FINETUNED_ENCODER_CHECKPOINT="playground/Checkpoints/all_task_spa_base_ft_robotwin_new/final_model/pytorch_model.pt"; fi
if [[ "$MODEL" == "vc1_base_ft" ]]; then MODEL="vc1_base"; FINETUNED_ENCODER_CHECKPOINT="playground/Checkpoints/all_task_vc1_base_robotwin_new/final_model/pytorch_model.pt"; fi
if [[ "$MODEL" == "vggt_omega_ft" ]]; then MODEL="vggt_omega"; FINETUNED_ENCODER_CHECKPOINT="playground/Checkpoints/all_task_vggt_omega_ft_robotwin_new/final_model/pytorch_model.pt"; fi
if [[ "$MODEL" == "cradio_ft" ]]; then MODEL="cradio"; FINETUNED_ENCODER_CHECKPOINT="playground/Checkpoints/all_task_cradio-ft_robotwin_new/final_model/pytorch_model.pt"; fi
if [[ "$MODEL" == "siglip2_ft" ]]; then MODEL="siglip2"; FINETUNED_ENCODER_CHECKPOINT="playground/Checkpoints/all_task_siglip2_ft_robotwin_new/final_model/pytorch_model.pt"; fi
args=(
  "${PYTHON_BIN}" -m Analyze.analyse.train_idm_fdm
  --config-name="${CONFIG_NAME}"
  mode="${MODE}"
  encoder_name="${MODEL}"
  data_root="${DATA_ROOT}"
  output_dir="${OUTPUT_DIR}"
  epochs="${EPOCHS}"
  device=cuda
)
if [[ -n "${FINETUNED_ENCODER_CHECKPOINT}" ]]; then
  args+=(finetuned_encoder_checkpoint="${FINETUNED_ENCODER_CHECKPOINT}")
fi
if [[ -n "${ENCODER_DIM}" ]]; then
  args+=(encoder_dim="${ENCODER_DIM}")
fi
"${args[@]}"
