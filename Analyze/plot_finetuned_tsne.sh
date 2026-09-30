#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1}"; IFS=, read -r -a GPUS <<< "$GPU_LIST"
ENCODER_LIST="${ENCODER_LIST:-cradio dinov2_base dinov3_base siglip2 vc1_base}"
read -r -a ENCS <<< "$ENCODER_LIST"; mkdir -p Analyze/outputs/tsne/logs
for gi in "${!GPUS[@]}"; do (
  for i in "${!ENCS[@]}"; do
    (( i % ${#GPUS[@]} == gi )) || continue; e="${ENCS[$i]}"
    for d in robotwin robocasa; do
      test -d "Analyze/outputs/tsne/cache/${d}/${e}_ft" || { echo "[skip] no cache: ${d}/${e}_ft"; continue; }
      "$PYTHON_BIN" -m Analyze.analyse.tsne_analysis plot --dataset "$d" --encoder "$e" --name "${e}_ft" --points-per-task 5000
    done
  done
) >"Analyze/outputs/tsne/logs/plot_finetuned.gpu${GPUS[$gi]}.log" 2>&1 & done
wait
