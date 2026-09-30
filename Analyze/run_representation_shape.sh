#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET=${DATASET:-robotwin}
GPU_LIST=${GPU_LIST:-0}
IFS=, read -ra GPUS <<< "$GPU_LIST"
TARGETS=(gaussian sphere)
# ENCODERS=(dinov2_base dinov3_base spa_base vc1_base clip siglip siglip2 mae vit radio cradio theia internvit voltron vggt_omega vjepa2_base depthvlm lingbot_large qwen3 xiaomi dinov2_base_ft dinov3_base_ft spa_base_ft vc1_base_ft vggt_omega_ft)
read -r -a ENCODERS <<< "${ENCODER_LIST:-dinov2_base dinov3_base}"
run_one(){
  local dataset="$1" gpu="$2" e="$3" t="$4"; local pp="${PYTHONPATH:-}"
  case "$e" in
    spa_base|spa_base_ft) pp="${SPA_ROOT:-third_party/SPA}:${pp}" ;;
    vggt_omega|vggt_omega_ft) pp="${VGGT_OMEGA_ROOT:-third_party/VGGT_Omega}:${pp}" ;;
  esac
  local force_arg=""; [[ "${FORCE:-0}" == 1 ]] && force_arg="--force"
  local default_points=2000
  [[ "$dataset" == "robocasa" ]] && default_points=500
  local points="${POINTS_PER_TASK:-$default_points}"
  echo "[shape] gpu=$gpu dataset=$dataset encoder=$e target=$t points_per_task=$points"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$pp" "$PYTHON_BIN" -m Analyze.analyse.representation_shape \
    --dataset "$dataset" --name "$e" --target "$t" \
    --points-per-task "$points" --stride "${STRIDE:-10}" \
    --epochs "${EPOCHS:-10}" --batch-size "${BATCH_SIZE:-256}" \
    --encode-batch-size "${ENCODE_BATCH_SIZE:-2}" $force_arg
}

run_dataset(){
  local dataset="$1"
  echo "[shape] ===== starting dataset: $dataset ====="
  if [[ "${SMOKE:-0}" == 1 ]]; then
    export POINTS_PER_TASK=1 EPOCHS=1 BATCH_SIZE=1 FORCE=1
    for i in "${!ENCODERS[@]}"; do
      local gpu=${GPUS[$((i % ${#GPUS[@]}))]}
      run_one "$dataset" "$gpu" "${ENCODERS[$i]}" gaussian
    done
    return
  fi
  local pids=() idx=0
  for e in "${ENCODERS[@]}"; do
    local gpu=${GPUS[$((idx % ${#GPUS[@]}))]}; idx=$((idx+1))
    (for t in "${TARGETS[@]}"; do run_one "$dataset" "$gpu" "$e" "$t"; done) & pids+=("$!")
    if (( ${#pids[@]} >= ${#GPUS[@]} )); then
      for p in "${pids[@]}"; do wait "$p"; done
      pids=()
    fi
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "[shape] ===== finished dataset: $dataset ====="
}

case "$DATASET" in
  robotwin|robocasa) run_dataset "$DATASET" ;;
  both) run_dataset robotwin; run_dataset robocasa ;;
  *) echo "DATASET must be robotwin, robocasa, or both" >&2; exit 2 ;;
esac
