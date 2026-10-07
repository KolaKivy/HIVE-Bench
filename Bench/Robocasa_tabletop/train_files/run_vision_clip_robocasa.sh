Framework_name=VisionCLIPGR00T
clip_model=openai/clip-vit-base-patch32
freeze_module_list='vision_encoder,text_encoder'
DIT_TYPE="DiT-B"
VISION_MODEL=${VISION_MODEL:-}
vision_model_args=()
if [[ -n "${VISION_MODEL}" ]]; then
  vision_model_args+=(--framework.vision_model "${VISION_MODEL}")
fi
data_root_dir="${DATA_ROOT_DIR:-playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim}"
data_mix=fourier_gr1_unified_1
num_trajectories=${1:-${NUM_TRAJECTORIES:-}}
trajectory_subset_seed=${TRAJECTORY_SUBSET_SEED:-}
data_override_args=()
if [[ -n "${num_trajectories}" ]]; then
  data_override_args+=(--datasets.vla_data.max_trajectories "${num_trajectories}")
fi
if [[ -n "${trajectory_subset_seed}" ]]; then
  data_override_args+=(--datasets.vla_data.trajectory_subset_seed "${trajectory_subset_seed}")
fi

run_root_dir="${RUN_ROOT_DIR:-./playground/Checkpoints}"
run_id=all_task_dinov3_base_robocasa

export CUDA_VISIBLE_DEVICES=4,5,6,7

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --main_process_port 10000 \
  --config_file Policy/hivebench/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  Policy/hivebench/training/train.py \
  --config_yaml ./Bench/Robocasa_tabletop/train_files/vision_clip_robocasa.yaml \
  --framework.name "${Framework_name}" \
  --framework.clip_model "${clip_model}" \
  --framework.action_model.action_model_type "${DIT_TYPE}" \
  "${vision_model_args[@]}" \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  "${data_override_args[@]}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.learning_rate.base 3e-5 \
  --trainer.learning_rate.text_projector 3e-5 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
