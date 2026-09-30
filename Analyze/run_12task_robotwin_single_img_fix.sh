#!/bin/bash
# bash run_12task_robotwin_single_img_fix.sh depthvlm robotwin_new 2 "0 1"

BASE_DIR="playground/RoboTwin_LeRobot/Randomized"

ANALYSIS="[avg_token_cos,dist_sim_decay,mean_token_norm,neighbor_sim,token_cov_rank,token_norm_entropy,token_norm_var,token_to_global,frequency_metrics]"
# ANALYSIS="[temporal_smoothness,temporal_cosine_shift,lag_distance_curve,temporal_variance,temporal_effective_rank,temporal_spectral_entropy,autocorrelation,total_trajectory_variation,patch_temporal_smoothness,temporal_token_norm_entropy,within_between_var,trajectory_var_ratio]"

model=${1}
ADDITION_NAME=${2}
NUM_GPUS=${3}
GPU_IDS=${4:-$(seq 0 $((NUM_GPUS - 1)))}

# Parse the space-separated GPU list.
read -r -a GPU_IDS_ARR <<< "$GPU_IDS"

# Derive the worker count from the parsed GPU list.
NUM_GPUS=${#GPU_IDS_ARR[@]}

LOG_DIR="./outputs/logs/robotwin_${model}_analysis"
mkdir -p "$LOG_DIR"

# Twelve-task benchmark list
TASK_DIRS=(
    "adjust_bottle"
    "click_alarmclock"
)

TOTAL_TASKS=${#TASK_DIRS[@]}

echo "Using model: $model"
echo "Total tasks: $TOTAL_TASKS"
echo "GPU_IDS: ${GPU_IDS_ARR[*]}"
echo "NUM_GPUS: $NUM_GPUS"

if [ "$TOTAL_TASKS" -ne 12 ]; then
    echo "Warning: expected 12 tasks, but got $TOTAL_TASKS"
fi

# Signal handling
cleanup() {
    echo "Termination signal received; stopping child processes..."
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

    # Assign tasks by worker index rather than physical GPU ID.
    for (( IDX=WORKER_ID; IDX<TOTAL_TASKS; IDX+=NUM_GPUS )); do
        local TASK_NAME="${TASK_DIRS[$IDX]}"
        local TASK_DIR="$BASE_DIR/$TASK_NAME"
        local DATA_NAME="${TASK_NAME}"
        local VIDEO_PATH="$TASK_DIR/videos/chunk-000/observation.images.cam_high/"

        if [ ! -d "$VIDEO_PATH" ]; then
            echo "[GPU $GPU_ID][WORKER $WORKER_ID] Skip IDX=$IDX task=$TASK_NAME: $VIDEO_PATH does not exist"
            continue
        fi

        local LOG_FILE="$LOG_DIR/${DATA_NAME}_gpu${GPU_ID}.log"

        echo "======================================"
        echo "[GPU $GPU_ID][WORKER $WORKER_ID] Running IDX=$IDX task: $DATA_NAME"
        echo "[GPU $GPU_ID] Video path: $VIDEO_PATH"
        echo "[GPU $GPU_ID] Log file: $LOG_FILE"
        echo "======================================"

        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
            model=$model \
            analysis="$ANALYSIS" \
            video_path="$VIDEO_PATH" \
            batch_size=32 \
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

# Preview task assignments and verify full task coverage.
echo "Task assignment preview:"
for (( i=0; i<TOTAL_TASKS; i++ )); do
    worker=$(( i % NUM_GPUS ))
    gpu="${GPU_IDS_ARR[$worker]}"
    echo "  IDX $i -> ${TASK_DIRS[$i]} -> GPU $gpu (worker $worker)"
done

# Start background workers.
worker=0
for GPU_ID in "${GPU_IDS_ARR[@]}"; do
    run_on_gpu "$GPU_ID" "$worker" &
    worker=$((worker + 1))
done

echo "All GPU workers started. Press Ctrl+C to stop tasks for model=$model."

wait
echo "All tasks finished for model=$model."