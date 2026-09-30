#!/bin/bash
# bash run_12task.sh <model_name> video_level 6 "2 3 4 5 6 7"

BASE_DIR="playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"

# ANALYSIS="[avg_token_cos,dist_sim_decay,mean_token_norm,neighbor_sim,token_cov_rank,token_norm_entropy,token_norm_var,token_to_global,frequency_metrics]"
ANALYSIS="[temporal_smoothness,temporal_cosine_shift,lag_distance_curve,temporal_variance,temporal_effective_rank,temporal_spectral_entropy,autocorrelation,total_trajectory_variation,patch_temporal_smoothness,temporal_token_norm_entropy,within_between_var,trajectory_var_ratio]"

NUM_GPUS=${3}
GPU_IDS=${4:-$(seq 0 $((NUM_GPUS - 1)))}
model=${1}
ADDITION_NAME=${2}

LOG_DIR="./outputs/logs/${model}_video_analysis"
mkdir -p "$LOG_DIR"

# Twelve-task benchmark list
TASK_DIRS=(
"gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000"
"gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000"
"gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000"
"gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000"
"gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000"
"gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000"
"gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
"gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
"gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
"gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
"gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000"
"gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000"
)

TOTAL_TASKS=${#TASK_DIRS[@]}

echo "Using model: $model"
echo "Total tasks: $TOTAL_TASKS"

if [ "$TOTAL_TASKS" -ne 12 ]; then
    echo "Warning: expected 12 tasks, but got $TOTAL_TASKS"
fi

# Signal handling
cleanup() {
    echo "Termination signal received; stopping child processes..."
    # Stop all background processes created by this script.
    pkill -P $$   # -P selects processes by parent ID.
    wait          # Wait for child processes to exit.
    echo "Cleanup complete."
    exit 0
}

trap cleanup SIGINT SIGTERM

run_on_gpu() {
    GPU_ID="$1"

    echo "Start worker on GPU $GPU_ID"

    for (( IDX=GPU_ID; IDX<TOTAL_TASKS; IDX+=NUM_GPUS )); do
        TASK_NAME="${TASK_DIRS[$IDX]}"
        TASK_DIR="$BASE_DIR/$TASK_NAME"

        DATA_NAME="${TASK_NAME#gr1_unified.}"

        VIDEO_PATH="$TASK_DIR/videos/chunk-000/observation.images.ego_view/"

        if [ ! -d "$VIDEO_PATH" ]; then
            echo "[GPU $GPU_ID] Skip: $VIDEO_PATH does not exist"
            continue
        fi

        LOG_FILE="$LOG_DIR/${DATA_NAME}_gpu${GPU_ID}.log"

        echo "======================================"
        echo "[GPU $GPU_ID] Running task: $DATA_NAME"
        echo "[GPU $GPU_ID] Video path: $VIDEO_PATH"
        echo "[GPU $GPU_ID] Log file: $LOG_FILE"
        echo "======================================"

        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
            model=$model \
            analysis="$ANALYSIS" \
            video_path="$VIDEO_PATH" \
            batch_size=64 \
            frame_start=0 \
            frame_end=100 \
            stride=5 \
            addition_name="$ADDITION_NAME" \
            data_name="$DATA_NAME" \
            > "$LOG_FILE" 2>&1

        if [ $? -eq 0 ]; then
            echo "[GPU $GPU_ID] Finished: $DATA_NAME"
        else
            echo "[GPU $GPU_ID] Failed: $DATA_NAME, check log: $LOG_FILE"
        fi
    done

    echo "Worker on GPU $GPU_ID finished."
}

# Start background workers.
for GPU_ID in $GPU_IDS; do
    run_on_gpu "$GPU_ID" &
done

echo "All GPU workers started. Press Ctrl+C to stop tasks for model=$model."

wait
echo "All tasks finished for model=$model."