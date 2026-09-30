#!/usr/bin/env bash
# Evaluate one HIVE-Bench checkpoint on the 12-task RoboCasa suite.
# Each task is evaluated three times. One worker is created per configured GPU.
#
# Edit CHECKPOINT_PATH, GPUS_STR, and PORT_BASE in the configuration block,
# then run:
#   bash Bench/Robocasa_tabletop/eval_files/run_encoder_eval.sh

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HIVE_BENCH_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# User configuration
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${HIVE_BENCH_DIR}/playground/Checkpoints/your_run/pytorch_model.pt}"
GPUS_STR="${GPUS_STR:-0,1,2,3}"
PORT_BASE="${PORT_BASE:-8000}"

# Runtime environments
CONDA_BASE="${CONDA_BASE:-}"
SERVER_CONDA_ENV="${SERVER_CONDA_ENV:-hivebench}"
CLIENT_CONDA_ENV="${CLIENT_CONDA_ENV:-robocasa}"

# Evaluation settings
RUN_TAG="${RUN_TAG:-robocasa_12_tasks}"
N_EPISODES="${N_EPISODES:-50}"
N_ENVS="${N_ENVS:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-12}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"
HF_OFFLINE="${HF_OFFLINE:-0}"
USE_BF16="${USE_BF16:-0}"
N_REPEATS=3

if [[ -z "${CONDA_BASE}" ]] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
fi

IFS=',' read -r -a GPUS <<< "${GPUS_STR}"

SERVER_BF16_ARG=""
if [[ "${USE_BF16}" == "1" || "${USE_BF16}" == "true" ]]; then
    SERVER_BF16_ARG="--use_bf16"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${HIVE_BENCH_DIR}/results/robocasa_eval/${RUN_TAG}_${TIMESTAMP}_$$"
VIDEO_ROOT="${RUN_ROOT}/videos"
LOG_ROOT="${RUN_ROOT}/logs"
SUMMARY_CSV="${RUN_ROOT}/results_summary.csv"
RAW_CSV="${RUN_ROOT}/results_raw.csv"
PID_FILE="${RUN_ROOT}/spawned_pids.txt"
TASK_QUEUE_FILE="${RUN_ROOT}/task_queue.txt"
QUEUE_LOCK_FILE="${RUN_ROOT}/task_queue.lock"

RAW_TASKS=(
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
)

validate_configuration() {
    local command_name gpu_id port
    local -A seen_gpus=()

    for command_name in flock setsid python3; do
        command -v "${command_name}" >/dev/null 2>&1 || {
            echo "[ERROR] Required command not found: ${command_name}" >&2
            return 1
        }
    done

    [[ -f "${CHECKPOINT_PATH}" ]] || {
        echo "[ERROR] Checkpoint not found: ${CHECKPOINT_PATH}" >&2
        echo "[ERROR] Edit CHECKPOINT_PATH at the top of this script." >&2
        return 1
    }

    [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]] || {
        echo "[ERROR] Conda initialization script not found." >&2
        echo "[ERROR] Set CONDA_BASE to the output of 'conda info --base'." >&2
        return 1
    }

    [[ "${PORT_BASE}" =~ ^[0-9]+$ ]] || {
        echo "[ERROR] PORT_BASE must be an integer: ${PORT_BASE}" >&2
        return 1
    }

    (( ${#GPUS[@]} > 0 )) || {
        echo "[ERROR] GPUS_STR must contain at least one GPU id." >&2
        return 1
    }

    for gpu_id in "${GPUS[@]}"; do
        [[ "${gpu_id}" =~ ^[0-9]+$ ]] || {
            echo "[ERROR] Invalid GPU id in GPUS_STR: ${gpu_id}" >&2
            return 1
        }
        [[ -z "${seen_gpus[${gpu_id}]:-}" ]] || {
            echo "[ERROR] Duplicate GPU id in GPUS_STR: ${gpu_id}" >&2
            return 1
        }
        seen_gpus["${gpu_id}"]=1
        port=$((PORT_BASE + gpu_id))
        (( port > 0 && port <= 65535 )) || {
            echo "[ERROR] Computed port is outside 1-65535: ${port}" >&2
            return 1
        }
    done
}

register_pid() {
    local pid="$1"
    (
        flock -x 201
        echo "${pid}" >> "${PID_FILE}"
    ) 201>>"${PID_FILE}.lock"
}

cleanup_all_spawned() {
    local exit_code="${1:-130}"
    local pid

    trap - INT TERM
    echo
    echo "[INFO] Stopping all policy-server and simulator process groups."

    if [[ -f "${PID_FILE}" ]]; then
        while read -r pid; do
            [[ -z "${pid}" ]] && continue
            kill -TERM -- "-${pid}" 2>/dev/null || true
        done < "${PID_FILE}"

        sleep 3

        while read -r pid; do
            [[ -z "${pid}" ]] && continue
            kill -KILL -- "-${pid}" 2>/dev/null || true
        done < "${PID_FILE}"
    fi

    exit "${exit_code}"
}

trap 'cleanup_all_spawned 130' INT TERM

convert_env_name() {
    local raw="$1"
    local converted="${raw//./\/}"
    converted="${converted%_1000}"
    converted="${converted%_Env}"
    echo "${converted}_Env"
}

short_name_from_raw() {
    local raw="$1"
    local short="${raw#*.}"
    short="${short%_GR1ArmsAndWaistFourierHands_1000}"
    short="${short%_GR1ArmsAndWaistFourierHands_Env}"
    short="${short%_1000}"
    short="${short%_Env}"
    echo "${short}"
}

port_for_gpu() {
    local gpu_id="$1"
    echo $((PORT_BASE + gpu_id))
}

wait_for_port_or_death() {
    local port="$1"
    local pid="$2"
    local timeout="${3:-180}"
    local waited=0

    while true; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[ERROR] Policy server ${pid} exited before port ${port} became ready." >&2
            return 1
        fi

        if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
            exec 3>&- 2>/dev/null
            return 0
        fi

        sleep 2
        waited=$((waited + 2))
        if (( waited >= timeout )); then
            echo "[ERROR] Timed out after ${timeout}s while waiting for port ${port}." >&2
            return 1
        fi
    done
}

extract_success_rate() {
    local log_file="$1"
    local success_rate

    success_rate="$(
        awk -F': ' '/Success rate:/ { value=$2 } END { if (value != "") print value }' "${log_file}"
    )"

    if [[ -n "${success_rate}" ]]; then
        printf "%s" "${success_rate}"
        return 0
    fi

    awk '
        /\| Episode [0-9]+\/[0-9]+ \|/ {
            total += 1
            if ($0 ~ /Success: True/) {
                success += 1
            }
        }
        END {
            if (total > 0) {
                printf "%.2f", success / total
            }
        }
    ' "${log_file}"
}

write_raw_result() {
    local task="$1"
    local run_idx="$2"
    local success_rate="$3"

    (
        flock -x 200
        echo "${task},${run_idx},${success_rate}" >> "${RAW_CSV}"
    ) 200>>"${RAW_CSV}.lock"
}

stop_process_group() {
    local pid="$1"
    kill -TERM -- "-${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
}

run_one_task() {
    local gpu_id="$1"
    local raw_task="$2"
    local env_name short port task_log_dir server_log server_pid

    env_name="$(convert_env_name "${raw_task}")"
    short="$(short_name_from_raw "${raw_task}")"
    port="$(port_for_gpu "${gpu_id}")"
    task_log_dir="${LOG_ROOT}/${short}"
    server_log="${task_log_dir}/server.log"
    mkdir -p "${task_log_dir}"

    echo "[GPU ${gpu_id}] Starting ${short} on port ${port}."

    if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
        exec 3>&- 2>/dev/null
        echo "[GPU ${gpu_id}] [ERROR] Port ${port} is already in use; skipping ${short}." >&2
        return 1
    fi

    setsid bash -c "
        source '${CONDA_BASE}/etc/profile.d/conda.sh'
        conda activate '${SERVER_CONDA_ENV}'
        cd '${HIVE_BENCH_DIR}/Policy'
        exec env CUDA_VISIBLE_DEVICES=${gpu_id} \
            HF_HUB_OFFLINE=${HF_OFFLINE} TRANSFORMERS_OFFLINE=${HF_OFFLINE} \
            python deployment/model_server/server_policy.py \
            --ckpt_path '${CHECKPOINT_PATH}' \
            --port ${port} \
            ${SERVER_BF16_ARG}
    " > "${server_log}" 2>&1 &
    server_pid=$!
    register_pid "${server_pid}"

    if ! wait_for_port_or_death "${port}" "${server_pid}" "${SERVER_START_TIMEOUT}"; then
        echo "[GPU ${gpu_id}] [ERROR] Server failed for ${short}; see ${server_log}." >&2
        stop_process_group "${server_pid}"
        return 1
    fi

    echo "[GPU ${gpu_id}] Policy server ready for ${short}."

    local run_idx video_out client_log client_pid client_status success_rate
    for run_idx in 1 2 3; do
        video_out="${VIDEO_ROOT}/${short}/run${run_idx}"
        client_log="${task_log_dir}/client_run${run_idx}.log"
        mkdir -p "${video_out}"

        echo "[GPU ${gpu_id}] Evaluating ${short}, run ${run_idx}/${N_REPEATS}."

        setsid bash -c "
            source '${CONDA_BASE}/etc/profile.d/conda.sh'
            conda activate '${CLIENT_CONDA_ENV}'
            cd '${HIVE_BENCH_DIR}'
            exec env CUDA_VISIBLE_DEVICES=${gpu_id} \
                python -u Bench/Robocasa_tabletop/eval_files/simulation_env.py \
                    --args.env_name '${env_name}' \
                    --args.port ${port} \
                    --args.n_episodes ${N_EPISODES} \
                    --args.n_envs ${N_ENVS} \
                    --args.max_episode_steps ${MAX_EPISODE_STEPS} \
                    --args.n_action_steps ${N_ACTION_STEPS} \
                    --args.video_out_path '${video_out}' \
                    --args.pretrained_path '${CHECKPOINT_PATH}'
        " > "${client_log}" 2>&1 &
        client_pid=$!
        register_pid "${client_pid}"

        wait "${client_pid}"
        client_status=$?
        if (( client_status != 0 )); then
            echo "[GPU ${gpu_id}] [ERROR] ${short} run ${run_idx} exited with status ${client_status}; see ${client_log}." >&2
            stop_process_group "${server_pid}"
            return 1
        fi

        success_rate="$(extract_success_rate "${client_log}")"
        if [[ -z "${success_rate}" ]]; then
            echo "[GPU ${gpu_id}] [WARN] No success rate found for ${short} run ${run_idx}." >&2
            success_rate="NA"
        else
            echo "[GPU ${gpu_id}] ${short} run ${run_idx}: success rate ${success_rate}."
        fi
        write_raw_result "${short}" "${run_idx}" "${success_rate}"
    done

    stop_process_group "${server_pid}"
    echo "[GPU ${gpu_id}] Completed ${short}."
}

pop_task() {
    (
        flock -x 203
        if [[ -s "${TASK_QUEUE_FILE}" ]]; then
            head -n 1 "${TASK_QUEUE_FILE}"
            tail -n +2 "${TASK_QUEUE_FILE}" > "${TASK_QUEUE_FILE}.tmp"
            mv "${TASK_QUEUE_FILE}.tmp" "${TASK_QUEUE_FILE}"
        fi
    ) 203>"${QUEUE_LOCK_FILE}"
}

worker() {
    local gpu_id="$1"
    local raw_task

    while true; do
        raw_task="$(pop_task)"
        [[ -z "${raw_task}" ]] && break
        run_one_task "${gpu_id}" "${raw_task}" || true
    done
}

summarize_results() {
    python3 - "${RAW_CSV}" "${SUMMARY_CSV}" <<'PYEOF'
import csv
import sys
from collections import defaultdict

raw_path, output_path = sys.argv[1], sys.argv[2]
results = defaultdict(dict)

with open(raw_path, encoding="utf-8") as source:
    for row in csv.DictReader(source):
        results[row["task"]][row["run"]] = row["success_rate"]

with open(output_path, "w", newline="", encoding="utf-8") as destination:
    writer = csv.writer(destination)
    writer.writerow(["task", "run1", "run2", "run3", "mean"])

    for task in sorted(results):
        values = []
        output_row = [task]
        for run_idx in ("1", "2", "3"):
            value = results[task].get(run_idx, "NA")
            output_row.append(value)
            try:
                values.append(float(value))
            except ValueError:
                pass
        output_row.append(f"{sum(values) / len(values):.4f}" if values else "NA")
        writer.writerow(output_row)

print(f"Summarized {len(results)} tasks.")
PYEOF
}

main() {
    validate_configuration || exit 1

    mkdir -p "${VIDEO_ROOT}" "${LOG_ROOT}"
    : > "${PID_FILE}"
    echo "task,run,success_rate" > "${RAW_CSV}"
    printf '%s\n' "${RAW_TASKS[@]}" > "${TASK_QUEUE_FILE}"

    echo "[INFO] Checkpoint: ${CHECKPOINT_PATH}"
    echo "[INFO] GPUs: ${GPUS_STR}"
    echo "[INFO] Port base: ${PORT_BASE}"
    echo "[INFO] Tasks: ${#RAW_TASKS[@]}, runs per task: ${N_REPEATS}"
    echo "[INFO] Output: ${RUN_ROOT}"

    local gpu_id pid
    local -a worker_pids=()

    for gpu_id in "${GPUS[@]}"; do
        worker "${gpu_id}" &
        worker_pids+=("$!")
    done

    for pid in "${worker_pids[@]}"; do
        wait "${pid}" || true
    done

    summarize_results
    echo "[INFO] Evaluation complete: ${SUMMARY_CSV}"
}

main "$@"
