#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
# pca_vis avg_token_cos dist_sim_decay mean_token_norm neighbor_sim token_cov_rank token_norm_entropy token_norm_var token_to_global frequency_metrics
# temporal_smoothness temporal_cosine_shift lag_distance_curve temporal_variance temporal_effective_rank temporal_spectral_entropy autocorrelation total_trajectory_variation patch_temporal_smoothness temporal_token_norm_entropy within_between_var trajectory_var_ratio
export CUDA_VISIBLE_DEVICES=0
ADDITION_NAME="video_level"
DATA_NAME="PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000"

python Analyze/run.py \
    model=dinov3 \
    analysis=[temporal_smoothness,temporal_cosine_shift,lag_distance_curve,temporal_variance,temporal_effective_rank,temporal_spectral_entropy,autocorrelation,total_trajectory_variation,patch_temporal_smoothness,temporal_token_norm_entropy,within_between_var,trajectory_var_ratio] \
    video_path=playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000/videos/chunk-000/observation.images.ego_view/ \
    batch_size=2 \
    stride=5 \
    addition_name=$ADDITION_NAME \
    data_name=$DATA_NAME
