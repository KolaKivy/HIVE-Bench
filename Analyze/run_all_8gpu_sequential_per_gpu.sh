#!/bin/bash
# Run three tasks sequentially on each of eight GPUs.
BASE_DIR="playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"

ANALYSIS="[avg_token_cos,dist_sim_decay,mean_token_norm,neighbor_sim,token_cov_rank,token_norm_entropy,token_norm_var,token_to_global,frequency_metrics]"

NUM_GPUS=8

model=dinov2

LOG_DIR="./outputs/logs_${model}_analysis"
mkdir -p "$LOG_DIR"

mapfile -t TASK_DIRS < <(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d -name "gr1_unified.*" | sort)

TOTAL_TASKS=${#TASK_DIRS[@]}

echo "Found $TOTAL_TASKS task folders."

if [ "$TOTAL_TASKS" -ne 24 ]; then
    echo "Warning: expected 24 folders, but found $TOTAL_TASKS folders."
fi

run_on_gpu() {
    GPU_ID="$1"

    echo "Start worker on GPU $GPU_ID"

    for (( IDX=GPU_ID; IDX<TOTAL_TASKS; IDX+=NUM_GPUS )); do
        TASK_DIR="${TASK_DIRS[$IDX]}"

        TASK_NAME=$(basename "$TASK_DIR")
        ADDITION_NAME=""
        DATA_NAME="${TASK_NAME#gr1_unified.}"

        VIDEO_PATH="$TASK_DIR/videos/chunk-000/observation.images.ego_view/"

        if [ ! -d "$VIDEO_PATH" ]; then
            echo "[GPU $GPU_ID] Skip: $VIDEO_PATH does not exist"
            continue
        fi

        LOG_FILE="$LOG_DIR/${DATA_NAME}_gpu${GPU_ID}.log"

        echo "======================================"
        echo "[GPU $GPU_ID] Running task: $ADDITION_NAME"
        echo "[GPU $GPU_ID] Data name: $DATA_NAME"
        echo "[GPU $GPU_ID] Video path: $VIDEO_PATH"
        echo "[GPU $GPU_ID] Log file: $LOG_FILE"
        echo "======================================"

        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
            model=$model \
            analysis="$ANALYSIS" \
            video_path="$VIDEO_PATH" \
            batch_size=2 \
            frame_start=0 \
            frame_end=100 \
            stride=5 \
            addition_name="$ADDITION_NAME" \
            data_name="$DATA_NAME" \
            > "$LOG_FILE" 2>&1

        if [ $? -eq 0 ]; then
            echo "[GPU $GPU_ID] Finished: $ADDITION_NAME"
        else
            echo "[GPU $GPU_ID] Failed: $ADDITION_NAME, check log: $LOG_FILE"
        fi
    done

    echo "Worker on GPU $GPU_ID finished."
}

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    run_on_gpu "$GPU_ID" &
done

wait

echo "All tasks finished."