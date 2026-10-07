#!/bin/bash
# HIVE-Bench: RobotWin multi-task training with a pluggable vision encoder + CLIP text.
# Run from the HIVE-Bench root. An optional first argument overrides YAML trajectory count.
Framework_name=VisionCLIPGR00T
freeze_module_list='text_encoder,vision_encoder'
DIT_TYPE=DiT-B
VISION_MODEL=${VISION_MODEL:-}
vision_model_args=()
if [[ -n "${VISION_MODEL}" ]]; then
  vision_model_args+=(--framework.vision_model "${VISION_MODEL}")
fi
config_yaml=./Bench/Robotwin/train_files/vision_clip_robotwin.yaml
run_root_dir=./playground/Checkpoints
# With no override, YAML remains the source of truth (500 trajectories by default).
num_trajectories=${1:-${NUM_TRAJECTORIES:-}}
trajectory_subset_seed=${TRAJECTORY_SUBSET_SEED:-}
data_size_label=${num_trajectories:-yaml}
subset_seed_label=${trajectory_subset_seed:-yaml}
data_override_args=()
if [[ -n "${num_trajectories}" ]]; then
  data_override_args+=(--datasets.vla_data.max_trajectories "${num_trajectories}")
fi
if [[ -n "${trajectory_subset_seed}" ]]; then
  data_override_args+=(--datasets.vla_data.trajectory_subset_seed "${trajectory_subset_seed}")
fi
run_id=test_lejepa_robotwin_new
export CUDA_VISIBLE_DEVICES=4,5,6,7
output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
accelerate launch \
  --config_file Policy/hivebench/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --main_process_port 29501 \
  Policy/hivebench/training/train.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.action_model.action_model_type "${DIT_TYPE}" \
  "${vision_model_args[@]}" \
  "${data_override_args[@]}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
