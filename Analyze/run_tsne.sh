#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATASET="${DATASET:-both}"
GPU_LIST="${GPU_LIST:-0,1}"                 # comma-separated physical GPU ids, e.g. 0,1,2,3
POINTS_PER_TASK="${POINTS_PER_TASK:-5000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
STRIDE="${STRIDE:-10}"
ENCODER_LIST="${ENCODER_LIST:-cradio_ft dinov2_base_ft dinov3_base_ft siglip2_ft vc1_base_ft depthvlm lingbot_large qwen3 xiaomi}"
IFS=, read -r -a GPUS <<< "$GPU_LIST"
read -r -a ENCODERS <<< "$ENCODER_LIST"
((${#GPUS[@]} > 0)) || { echo "GPU_LIST is empty" >&2; exit 2; }

pids=()
for gi in "${!GPUS[@]}"; do
  gpu="${GPUS[$gi]}"
  log="Analyze/outputs/tsne/logs/worker.gpu${gpu}.log"
  mkdir -p "$(dirname "$log")"
  echo "[launch] worker_gpu=$gpu log=$log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    for i in "${!ENCODERS[@]}"; do
      (( i % ${#GPUS[@]} == gi )) || continue
      enc="${ENCODERS[$i]}"
      echo "[start] encoder=$enc gpu=$gpu"
      for ds in robotwin robocasa; do
        [[ "$DATASET" == both || "$DATASET" == "$ds" ]] || continue
        ${PYTHON_BIN:-python} -m Analyze.analyse.tsne_analysis extract --dataset "$ds" --encoder "$enc" --device cuda --batch-size "$BATCH_SIZE" --stride "$STRIDE"
        ${PYTHON_BIN:-python} -m Analyze.analyse.tsne_analysis plot --dataset "$ds" --encoder "$enc" --points-per-task "$POINTS_PER_TASK"
      done
    done
  ) >"$log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
