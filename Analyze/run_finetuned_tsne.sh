#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-both}"; GPU_LIST="${GPU_LIST:-0,1}"
declare -A CKPT=(
 [cradio]="playground/Checkpoints/all_task_robotwin_cradio-ft/final_model/pytorch_model.pt"
 [dinov2_base]="playground/Checkpoints/all_task_robotwin_dinov2_ft/final_model/pytorch_model.pt"
 [dinov3_base]="playground/Checkpoints/all_task_robotwin_dinov3_ft/final_model/pytorch_model.pt"
 [siglip2]="playground/Checkpoints/all_task_robotwin_siglip2-ft/final_model/pytorch_model.pt"
#  [spa_base]="playground/Checkpoints/all_task_robotwin_spa_ft/final_model/pytorch_model.pt"
 [vc1_base]="playground/Checkpoints/all_task_robotwin_vc1_ft/final_model/pytorch_model.pt"
#  [vggt_omega]="playground/Checkpoints/all_task_robotwin_vggt_og_ft/final_model/pytorch_model.pt"
)
ENCODER_LIST="${ENCODER_LIST:-cradio dinov2_base dinov3_base siglip2  vc1_base }"
IFS=, read -r -a GPUS <<< "$GPU_LIST"; read -r -a ENCS <<< "$ENCODER_LIST"; mkdir -p Analyze/outputs/tsne/logs
for gi in "${!GPUS[@]}"; do ( export CUDA_VISIBLE_DEVICES="${GPUS[$gi]}"; for i in "${!ENCS[@]}"; do (( i % ${#GPUS[@]} == gi )) || continue; e="${ENCS[$i]}"; for d in robotwin robocasa; do [[ "$DATASET" == both || "$DATASET" == "$d" ]] || continue; $PYTHON_BIN -m Analyze.analyse.tsne_analysis extract --dataset "$d" --encoder "$e" --checkpoint "${CKPT[$e]}" --name "${e}_ft" --device cuda; $PYTHON_BIN -m Analyze.analyse.tsne_analysis plot --dataset "$d" --encoder "$e" --name "${e}_ft" --points-per-task 5000; done; done ) >"Analyze/outputs/tsne/logs/finetuned.gpu${GPUS[$gi]}.log" 2>&1 & done
wait
