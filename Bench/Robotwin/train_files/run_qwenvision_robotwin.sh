#!/bin/bash
# HIVE-Bench: RobotWin multi-task training with Qwen-Vision tokens + CLIP text.
# Run from the HIVE-Bench root. An optional first argument overrides YAML trajectory count.

Framework_name=QwenVisionGR00T
base_vlm="${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}"
freeze_module_list='vlm,clip_text_model'
DIT_TYPE=DiT-B
config_yaml=./Bench/Robotwin/train_files/qwenvision_robotwin.yaml
run_root_dir="${RUN_ROOT_DIR:-./playground/Checkpoints}"

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

run_id=qwenvision_robotwin_layer16
export CUDA_VISIBLE_DEVICES=0,1,2,3
output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file Policy/hivebench/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --main_process_port 29506 \
  Policy/hivebench/training/train.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.action_model.action_model_type "${DIT_TYPE}" \
  --framework.vision_text_fusion.vlm_layer_idx 16 \
  "${data_override_args[@]}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
