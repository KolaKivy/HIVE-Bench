#!/bin/bash
# HIVE-Bench: Train VisionGR00T on RoboTwin

Framework_name=VisionGR00T
freeze_module_list='vision_encoder'
DIT_TYPE="DiT-B"
VISION_MODEL=${VISION_MODEL:-}
vision_model_args=()
if [[ -n "${VISION_MODEL}" ]]; then
  vision_model_args+=(--framework.vision_model "${VISION_MODEL}")
fi
num_trajectories=${1:-${NUM_TRAJECTORIES:-500}}
trajectory_subset_seed=${TRAJECTORY_SUBSET_SEED:-42}
run_root_dir=./playground/Checkpoints
run_id=robotiwn_2

export CUDA_VISIBLE_DEVICES=4,5,6,7

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file Policy/hivebench/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --main_process_port 29506 \
  Policy/hivebench/training/train.py \
  --config_yaml ./Bench/Robotwin/train_files/vision_robotwin.yaml \
  --framework.name "${Framework_name}" \
  --framework.action_model.action_model_type "${DIT_TYPE}" \
  "${vision_model_args[@]}" \
  --datasets.vla_data.max_trajectories "${num_trajectories}" \
  --datasets.vla_data.trajectory_subset_seed "${trajectory_subset_seed}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
