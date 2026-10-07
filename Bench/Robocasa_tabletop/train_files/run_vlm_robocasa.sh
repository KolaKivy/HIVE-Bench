#!/bin/bash
# run_vlm_robocasa.sh
# VLMVisionGR00T: VLM vision tokens + CLIP text tokens → DiT
# VLMVisionGR00T training launcher.

Framework_name=VLMVisionGR00T
base_vlm="${BASE_VLM:-playground/Pretrained_models/Xiaomi-Robotics}"
data_root_dir="${DATA_ROOT_DIR:-playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim}"
data_mix=fourier_gr1_unified_1
DIT_TYPE="DiT-B"
run_root_dir="${RUN_ROOT_DIR:-./playground/Checkpoints}"
run_id=all_task_xiaomi_layer16_robocasa


export CUDA_VISIBLE_DEVICES=0,1,2,3

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file ./Policy/hivebench/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --main_process_port 29506 \
  Policy/hivebench/training/train.py \
  --config_yaml ./Bench/Robocasa_tabletop/train_files/vlm_robocasa.yaml \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.action_model.action_model_type "${DIT_TYPE}" \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --trainer.freeze_modules 'vlm,clip_text_model' \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 100000 \
  --trainer.logging_frequency 100 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
