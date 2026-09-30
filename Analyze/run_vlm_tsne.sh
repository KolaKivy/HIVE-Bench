#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"; DATASET="${DATASET:-both}"; GPU_LIST="${GPU_LIST:-0,1}"
for spec in 'depthvlm:depthvlm' 'qwen3:qwen3' 'xiaomi:xiaomi'; do IFS=: read -r v n <<< "$spec"; for d in robotwin robocasa; do [[ "$DATASET" == both || "$DATASET" == "$d" ]] || continue; CUDA_VISIBLE_DEVICES="$GPU_LIST" $PYTHON_BIN -m Analyze.analyse.tsne_analysis vlm --vlm "$v" --dataset "$d" --name "$n" --device cuda; CUDA_VISIBLE_DEVICES="$GPU_LIST" $PYTHON_BIN -m Analyze.analyse.tsne_analysis plot --dataset "$d" --encoder clip --name "$n" --points-per-task 5000; done; done
