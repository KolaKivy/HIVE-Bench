#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
DATASET=${DATASET:-both}; POINTS_PER_TASK=${POINTS_PER_TASK:-5000}; MAX_ITER=${MAX_ITER:-1000}; PYTHON_BIN=${PYTHON_BIN:-python}
robotwin=(dinov2_base dinov3_base spa_base vc1_base clip siglip siglip2 mae vit radio cradio theia internvit voltron vggt_omega vjepa2_base cradio_ft dinov2_base_ft dinov3_base_ft siglip2_ft spa_base_ft vc1_base_ft vggt_omega_ft depthvlm lingbot_large qwen3 xiaomi)
robocasa=("${robotwin[@]}")
for d in robotwin robocasa; do [[ "$DATASET" != both && "$DATASET" != "$d" ]] && continue; names=(${d}_dummy); for n in "${robotwin[@]}"; do [[ -d "Analyze/outputs/tsne/cache/$d/$n" ]] || continue; echo "[split] $d $n"; "$PYTHON_BIN" -m Analyze.analyse.split_half_tsne --dataset "$d" --name "$n" --points-per-task "$POINTS_PER_TASK" --max-iter "$MAX_ITER"; done; done
