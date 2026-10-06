#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
# bash run_12task_robocasa.sh <model_name> <addition_name> "0 1 2 3 4 5"

BASE_DIR="playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"

ANALYSIS="[avg_token_cos,dist_sim_decay,mean_token_norm,neighbor_sim,token_cov_rank,token_norm_entropy,token_norm_var,token_to_global,frequency_metrics,within_between_var,temporal_smoothness,temporal_cosine_shift,lag_distance_curve,temporal_variance,temporal_effective_rank,temporal_spectral_entropy,autocorrelation,total_trajectory_variation,patch_temporal_smoothness,temporal_token_norm_entropy,trajectory_var_ratio]"

model=${1}
ADDITION_NAME=${2}
GPU_IDS=${3}


read -r -a GPU_IDS_ARR <<< "$GPU_IDS"


NUM_GPUS=${#GPU_IDS_ARR[@]}
if [ "$#" -ne 3 ] || [ "$NUM_GPUS" -eq 0 ]; then
    echo "Usage: bash Analyze/run_12task_robocasa.sh <model_name> <addition_name> \"<gpu_ids>\""
    exit 2
fi

LOG_DIR="./Analyze/outputs/logs/${model}_analysis"
mkdir -p "$LOG_DIR"


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
echo "GPU_IDS: ${GPU_IDS_ARR[*]}"
echo "NUM_GPUS: $NUM_GPUS"

if [ "$TOTAL_TASKS" -ne 12 ]; then
    echo "Warning: expected 12 tasks, but got $TOTAL_TASKS"
fi


cleanup() {
    echo "Stopping analysis workers..."
    pkill -P $$
    wait
    echo "Cleanup complete."
    exit 0
}

trap cleanup SIGINT SIGTERM

run_on_gpu() {
    local GPU_ID="$1"
    local WORKER_ID="$2"

    echo "Start worker on GPU $GPU_ID, WORKER_ID=$WORKER_ID"


    for (( IDX=WORKER_ID; IDX<TOTAL_TASKS; IDX+=NUM_GPUS )); do
        local TASK_NAME="${TASK_DIRS[$IDX]}"
        local TASK_DIR="$BASE_DIR/$TASK_NAME"
        local DATA_NAME="${TASK_NAME#gr1_unified.}"
        local VIDEO_PATH="$TASK_DIR/videos/chunk-000/observation.images.ego_view/"

        if [ ! -d "$VIDEO_PATH" ]; then
            echo "[GPU $GPU_ID][WORKER $WORKER_ID] Skip IDX=$IDX task=$DATA_NAME: $VIDEO_PATH does not exist"
            continue
        fi

        local LOG_FILE="$LOG_DIR/${DATA_NAME}_gpu${GPU_ID}.log"

        echo "Running task analysis"
        echo "[GPU $GPU_ID][WORKER $WORKER_ID] Running IDX=$IDX task: $DATA_NAME"
        echo "[GPU $GPU_ID] Video path: $VIDEO_PATH"
        echo "[GPU $GPU_ID] Log file: $LOG_FILE"
        echo "Running task analysis"

        CUDA_VISIBLE_DEVICES=$GPU_ID python Analyze/run.py \
            model=$model \
            analysis="$ANALYSIS" \
            video_path="$VIDEO_PATH" \
            batch_size=64 \
            frame_start=0 \
            frame_end=100 \
            stride=5 \
            addition_name="$ADDITION_NAME" \
            data_name="$DATA_NAME" \
            hydra.run.dir="Analyze/outputs/${model}_${ADDITION_NAME}/${DATA_NAME}" \
            > "$LOG_FILE" 2>&1

        if [ $? -eq 0 ]; then
            echo "[GPU $GPU_ID] Finished: $DATA_NAME"
        else
            echo "[GPU $GPU_ID] Failed: $DATA_NAME, check log: $LOG_FILE"
        fi
    done

    echo "Worker on GPU $GPU_ID finished."
}


echo "Task assignments:"
for (( i=0; i<TOTAL_TASKS; i++ )); do
    worker=$(( i % NUM_GPUS ))
    gpu="${GPU_IDS_ARR[$worker]}"
    echo "  IDX $i -> ${TASK_DIRS[$i]} -> GPU $gpu (worker $worker)"
done


worker=0
for GPU_ID in "${GPU_IDS_ARR[@]}"; do
    run_on_gpu "$GPU_ID" "$worker" &
    worker=$((worker + 1))
done

echo "All GPU workers started. Press Ctrl+C to stop model=$model."

wait
echo "All tasks finished for model=$model."

METRICS_BASE_DIR="./Analyze/outputs/${model}_${ADDITION_NAME}"
echo "Averaging core metrics from: $METRICS_BASE_DIR"
python Analyze/avg_core_metrics_robocasa.py "$METRICS_BASE_DIR"
