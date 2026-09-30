#!/bin/bash
# HIVE-Bench: Train DinoGR00T on RoboCasa tabletop
# Run from project root: bash Bench/Robocasa_tabletop/train_files/run_dino_robocasa.sh

Framework_name=DinoGR00T

# Freeze vision encoder, only train DiT
freeze_module_list=''

DIT_TYPE="DiT-B"
VISION_MODEL=${VISION_MODEL:-}
vision_model_args=()
if [[ -n "${VISION_MODEL}" ]]; then
  vision_model_args+=(--framework.vision_model "${VISION_MODEL}")
fi
data_root_dir="${DATA_ROOT_DIR:-playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim}"
data_mix=fourier_gr1_unified_1
num_trajectories=${1:-${NUM_TRAJECTORIES:-1000}}
trajectory_subset_seed=${TRAJECTORY_SUBSET_SEED:-42}

run_root_dir="${RUN_ROOT_DIR:-./playground/Checkpoints}"
run_id=gr00t_robocasa_152


export CUDA_VISIBLE_DEVICES=0,1,2,3

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file ./Policy/hivebench/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --main_process_port 29503 \
  Policy/hivebench/training/train.py \
  --config_yaml ./Bench/Robocasa_tabletop/train_files/dino_robocasa.yaml \
  --framework.name ${Framework_name} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  "${vision_model_args[@]}" \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.max_trajectories ${num_trajectories} \
  --datasets.vla_data.trajectory_subset_seed ${trajectory_subset_seed} \
  --trainer.freeze_modules ${freeze_module_list} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}
