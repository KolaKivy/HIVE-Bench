#!/usr/bin/env bash
# Evaluate the 12-task RoboTwin benchmark.  Each worker owns one policy-server
# GPU and one *different* RoboTwin simulation GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASKS=(
    adjust_bottle
    beat_block_hammer
    click_alarmclock
    handover_block
    lift_pot
    open_laptop
    place_can_basket
    rotate_qrcode
    stamp_seal
    turn_switch
    place_burger_fries
    move_playingcard_away
)

usage() {
    cat <<'EOF'
Usage:
  bash eval_12_tasks.sh <checkpoint.pt> [options]

This launches one policy server per GPU pair, runs all requested RoboTwin
tasks three times by default, and writes a per-run and per-task summary.  The
policy server is restarted after every task (after that task's repeats finish).

Options:
  --server-gpus IDS       Comma-separated server GPUs (default: 2)
  --eval-gpus IDS         Comma-separated RoboTwin eval GPUs (default: 5)
  -m, --mode MODE         demo_randomized or demo_clean (default: demo_randomized)
  -n, --name NAME         RoboTwin ckpt_setting (default: main_test_v1)
  -r, --repeats N         Runs per task (default: 3)
  -s, --seed N            First evaluation seed; repeats use N, N+1, ... (default: 0)
  -p, --base-port PORT    First policy-server port (default: 5555)
  --server-timeout SEC    Wait limit for every policy server (default: 600)
  --log-dir DIR           Output directory (default: beside checkpoint)
  --tasks IDS             Optional comma-separated task subset
  --dry-run               Validate and print the execution plan only
  -h, --help              Show this help

The two GPU lists must be the same length and no GPU may occur in both lists.
For example, two parallel server/eval pairs:
  bash eval_12_tasks.sh playground/Checkpoints/example/steps_50000_compatible_pytorch_model.pt \
      --server-gpus 2,3 --eval-gpus 5,6 --base-port 5555

Optional environment overrides:
  HIVEBENCH_PYTHON=python
  ROBOTWIN_PYTHON=python
EOF
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}

parse_gpu_list() {
    local raw="$1"
    local -n destination="$2"
    local item
    local -a parts=()
    IFS=',' read -r -a parts <<< "${raw}"
    destination=()
    for item in "${parts[@]}"; do
        item="$(trim "${item}")"
        [[ "${item}" =~ ^[0-9]+$ ]] || die "Invalid GPU id: '${item}'"
        destination+=("${item}")
    done
    (( ${#destination[@]} > 0 )) || die "At least one GPU is required."
}

strip_ansi() {
    # RoboTwin colors its progress line; remove the ANSI escapes before parsing.
    # -u is essential here: without it sed block-buffers the first few long
    # 400-step episodes, making a live evaluation appear to have no output.
    sed -u -E $'s/\033\\[[0-9;]*[[:alpha:]]//g'
}

port_listening() {
    "${HIVEBENCH_PYTHON}" -c \
        'import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' \
        "$1" 2>/dev/null
}

port_available() {
    # Connecting to a port is insufficient: it can succeed against a stale
    # policy server from a previous launcher.  Binding tests the exact thing
    # the server process will do before we select or reuse a port.
    "${HIVEBENCH_PYTHON}" -c \
        'import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("0.0.0.0", int(sys.argv[1]))); s.close()' \
        "$1" 2>/dev/null
}

kill_descendants() {
    local pid="$1"
    local signal="${2:-TERM}"
    local child
    for child in $(ps -o pid= --ppid "${pid}" 2>/dev/null || true); do
        kill_descendants "${child}" "${signal}"
    done
    kill -"${signal}" "${pid}" 2>/dev/null || true
}

wait_for_server() {
    local port="$1"
    local timeout="$2"
    local server_pid="$3"
    local server_log="$4"
    local elapsed=0
    while (( elapsed < timeout )); do
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            return 2
        fi
        # A listener alone may belong to an old launcher.  Require the
        # readiness message emitted by this worker's server process as well.
        if grep -q "server listening on .*:${port}" "${server_log}" 2>/dev/null \
            && port_listening "${port}"; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1
}

extract_final_result() {
    local log_file="$1"
    local line successes episodes
    local success_pattern='Success[[:space:]]rate:[[:space:]]*([0-9]+)/([0-9]+)[[:space:]]*=>'
    FINAL_SUCCESSES=""
    FINAL_EPISODES=""
    while IFS= read -r line; do
        if [[ "${line}" =~ ${success_pattern} ]]; then
            successes="${BASH_REMATCH[1]}"
            episodes="${BASH_REMATCH[2]}"
            FINAL_SUCCESSES="${successes}"
            FINAL_EPISODES="${episodes}"
        fi
    done < <(strip_ansi < "${log_file}")
    [[ -n "${FINAL_SUCCESSES}" && -n "${FINAL_EPISODES}" ]]
}

run_worker() {
    local worker_id="$1"
    local server_gpu="$2"
    local eval_gpu="$3"
    local port="$4"
    local result_file="${LOG_DIR}/worker_${worker_id}_results.tsv"
    local server_pid=""
    local task_index repeat task seed eval_log eval_status status rate server_log

    cleanup_worker() {
        trap - EXIT INT TERM
        if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
            kill_descendants "${server_pid}" TERM
            sleep 1
            kill_descendants "${server_pid}" KILL
            wait "${server_pid}" 2>/dev/null || true
        fi
    }
    trap cleanup_worker EXIT INT TERM

    printf 'task\trepeat\tseed\tserver_gpu\teval_gpu\tport\tsuccesses\tepisodes\trate_percent\tstatus\teval_log\n' > "${result_file}"
    echo "[INFO] Worker ${worker_id}: server GPU ${server_gpu}, eval GPU ${eval_gpu}, port ${port}"

    # A worker owns whole tasks, not individual repeats.  This ensures a
    # server serves exactly one task's N repeats and is then restarted, which
    # prevents RoboTwin's server idle timeout from affecting later tasks.
    for ((task_index = worker_id; task_index < ${#TASKS[@]}; task_index += WORKER_COUNT)); do
        task="${TASKS[$task_index]}"
        server_log="${LOG_DIR}/${task}_worker${worker_id}_server_gpu${server_gpu}_eval_gpu${eval_gpu}_port${port}.log"
        if ! port_available "${port}"; then
            echo "[ERROR] Worker ${worker_id}: port ${port} is already occupied before ${task}. Refusing to connect to an unknown server." >&2
            return 1
        fi
        echo "[INFO] [worker ${worker_id}] starting server for ${task} on GPU ${server_gpu}, port ${port}"
        bash "${SCRIPT_DIR}/run_policy_server.sh" "${CKPT_PATH}" "${server_gpu}" "${port}" > "${server_log}" 2>&1 &
        server_pid=$!
        if ! wait_for_server "${port}" "${SERVER_TIMEOUT}" "${server_pid}" "${server_log}"; then
            echo "[ERROR] Worker ${worker_id}: server exited or did not become ready for ${task}. See ${server_log}" >&2
            return 1
        fi
        echo "[INFO] [worker ${worker_id}] policy server ready for ${task}"

        for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
            seed=$((BASE_SEED + repeat - 1))
            eval_log="${LOG_DIR}/${task}_run${repeat}_seed${seed}_servergpu${server_gpu}_evalgpu${eval_gpu}_port${port}.log"
            echo "[INFO] [worker ${worker_id}] ${task}, run ${repeat}/${REPEATS}, seed=${seed}"

            set +e
            bash "${SCRIPT_DIR}/eval.sh" \
                "${task}" "${MODE}" "${POLICY_NAME}" "${seed}" "${eval_gpu}" "${CKPT_PATH}" "${port}" \
                2>&1 | tee "${eval_log}" | strip_ansi | grep --line-buffered 'Success rate:' | sed -u "s/^/[RESULT] ${task} run${repeat}: /"
            eval_status=${PIPESTATUS[0]}
            set -e

            status="ok"
            rate=""
            if (( eval_status != 0 )); then
                status="eval_exit_${eval_status}"
                FINAL_SUCCESSES=""
                FINAL_EPISODES=""
            elif ! extract_final_result "${eval_log}"; then
                status="result_parse_error"
            fi

            if [[ -n "${FINAL_SUCCESSES:-}" && -n "${FINAL_EPISODES:-}" ]]; then
                rate="$(awk -v s="${FINAL_SUCCESSES}" -v n="${FINAL_EPISODES}" 'BEGIN { printf "%.1f", 100*s/n }')"
                echo "[DONE] ${task} run${repeat}: ${FINAL_SUCCESSES}/${FINAL_EPISODES} (${rate}%)"
            else
                FINAL_SUCCESSES="-"
                FINAL_EPISODES="-"
                rate="-"
                echo "[ERROR] ${task} run${repeat}: ${status}; see ${eval_log}" >&2
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${task}" "${repeat}" "${seed}" "${server_gpu}" "${eval_gpu}" "${port}" \
                "${FINAL_SUCCESSES}" "${FINAL_EPISODES}" "${rate}" "${status}" "${eval_log}" >> "${result_file}"
        done

        echo "[INFO] [worker ${worker_id}] ${task} complete; restarting server before the next task"
        kill_descendants "${server_pid}" TERM
        sleep 2
        if kill -0 "${server_pid}" 2>/dev/null; then
            kill_descendants "${server_pid}" KILL
        fi
        wait "${server_pid}" 2>/dev/null || true
        server_pid=""
    done
}

CKPT_PATH=""
SERVER_GPUS_RAW="2"
EVAL_GPUS_RAW="5"
MODE="demo_randomized"
POLICY_NAME="main_test_v1"
REPEATS=3
BASE_SEED=0
BASE_PORT=5555
SERVER_TIMEOUT=600
LOG_DIR=""
TASKS_RAW=""
DRY_RUN=false

while (( $# > 0 )); do
    case "$1" in
        --server-gpus) SERVER_GPUS_RAW="$2"; shift 2 ;;
        --eval-gpus) EVAL_GPUS_RAW="$2"; shift 2 ;;
        -m|--mode) MODE="$2"; shift 2 ;;
        -n|--name) POLICY_NAME="$2"; shift 2 ;;
        -r|--repeats) REPEATS="$2"; shift 2 ;;
        -s|--seed) BASE_SEED="$2"; shift 2 ;;
        -p|--base-port) BASE_PORT="$2"; shift 2 ;;
        --server-timeout) SERVER_TIMEOUT="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --tasks) TASKS_RAW="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) die "Unknown option: $1" ;;
        *)
            [[ -z "${CKPT_PATH}" ]] || die "Only one checkpoint path is allowed."
            CKPT_PATH="$1"
            shift
            ;;
    esac
done

[[ -n "${CKPT_PATH}" ]] || { usage >&2; exit 1; }
if [[ -d "${CKPT_PATH}" ]]; then
    if [[ -f "${CKPT_PATH}/pytorch_model.pt" ]]; then
        CKPT_PATH="${CKPT_PATH}/pytorch_model.pt"
    elif [[ -f "${CKPT_PATH}/pytorch_model_converted.pt" ]]; then
        echo "[WARN] No pytorch_model.pt found; falling back to converted checkpoint." >&2
        CKPT_PATH="${CKPT_PATH}/pytorch_model_converted.pt"
    fi
fi
[[ -f "${CKPT_PATH}" ]] || die "Checkpoint does not exist: ${CKPT_PATH}"
[[ "${MODE}" == "demo_randomized" || "${MODE}" == "demo_clean" ]] || die "Unsupported mode: ${MODE}"
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || die "--repeats must be a positive integer."
[[ "${BASE_SEED}" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer."
[[ "${BASE_PORT}" =~ ^[0-9]+$ && "${BASE_PORT}" -ge 1 && "${BASE_PORT}" -le 65535 ]] || die "Invalid base port."
[[ "${SERVER_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || die "Invalid --server-timeout."

if [[ -n "${TASKS_RAW}" ]]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_RAW}"
    for task_index in "${!TASKS[@]}"; do
        TASKS[$task_index]="$(trim "${TASKS[$task_index]}")"
        [[ -n "${TASKS[$task_index]}" ]] || die "Empty task in --tasks."
    done
fi

HIVEBENCH_PYTHON="${HIVEBENCH_PYTHON:-$(command -v python)}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-$(command -v python)}"
[[ -x "${HIVEBENCH_PYTHON}" ]] || die "HIVEBENCH_PYTHON is not executable: ${HIVEBENCH_PYTHON}"
[[ -x "${ROBOTWIN_PYTHON}" ]] || die "ROBOTWIN_PYTHON is not executable: ${ROBOTWIN_PYTHON}"
export HIVEBENCH_PYTHON ROBOTWIN_PYTHON

parse_gpu_list "${SERVER_GPUS_RAW}" SERVER_GPUS
parse_gpu_list "${EVAL_GPUS_RAW}" EVAL_GPUS
(( ${#SERVER_GPUS[@]} == ${#EVAL_GPUS[@]} )) || die "--server-gpus and --eval-gpus must have the same number of ids."
declare -A seen_gpus=()
for gpu in "${SERVER_GPUS[@]}" "${EVAL_GPUS[@]}"; do
    [[ -z "${seen_gpus[$gpu]:-}" ]] || die "GPU ${gpu} occurs more than once; server and eval GPUs must be disjoint."
    seen_gpus[$gpu]=1
done

WORKER_COUNT=${#SERVER_GPUS[@]}
TOTAL_JOBS=$(( ${#TASKS[@]} * REPEATS ))
if [[ -z "${LOG_DIR}" ]]; then
    ckpt_name="$(basename "${CKPT_PATH}")"
    ckpt_stem="${ckpt_name%.*}"
    LOG_DIR="$(dirname "${CKPT_PATH}")/robotwin_auto_eval/${ckpt_stem}_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi

PORTS=()
candidate_port=${BASE_PORT}
while (( ${#PORTS[@]} < WORKER_COUNT )); do
    (( candidate_port <= 65535 )) || die "No free port at or above ${BASE_PORT}."
    if port_available "${candidate_port}"; then
        PORTS+=("${candidate_port}")
    fi
    candidate_port=$((candidate_port + 1))
done

echo "[INFO] checkpoint: ${CKPT_PATH}"
echo "[INFO] mode=${MODE}, repeats=${REPEATS}, seeds=${BASE_SEED}..$((BASE_SEED + REPEATS - 1))"
echo "[INFO] tasks (${#TASKS[@]}): ${TASKS[*]}"
echo "[INFO] logs: ${LOG_DIR}"
for ((worker_id = 0; worker_id < WORKER_COUNT; ++worker_id)); do
    echo "[INFO] pair ${worker_id}: server GPU ${SERVER_GPUS[$worker_id]} -> eval GPU ${EVAL_GPUS[$worker_id]}, port ${PORTS[$worker_id]}"
done

if ${DRY_RUN}; then
    echo "[INFO] Dry run complete; no server or evaluation was started."
    exit 0
fi

mkdir -p "${LOG_DIR}"

WORKER_PIDS=()
cleanup_all() {
    trap - EXIT INT TERM
    local pid
    echo "[INFO] Cleaning up running evaluation workers..."
    for pid in "${WORKER_PIDS[@]:-}"; do
        [[ -n "${pid}" ]] && kill_descendants "${pid}" TERM
    done
    sleep 2
    for pid in "${WORKER_PIDS[@]:-}"; do
        [[ -n "${pid}" ]] && kill_descendants "${pid}" KILL
    done
}
trap cleanup_all EXIT INT TERM

for ((worker_id = 0; worker_id < WORKER_COUNT; ++worker_id)); do
    run_worker "${worker_id}" "${SERVER_GPUS[$worker_id]}" "${EVAL_GPUS[$worker_id]}" "${PORTS[$worker_id]}" &
    WORKER_PIDS+=("$!")
done

worker_failed=false
for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "${pid}"; then
        worker_failed=true
    fi
done
# All workers have been reaped.  Do not let the EXIT trap act on a PID which
# could theoretically be recycled while the summaries are being written.
WORKER_PIDS=()
trap - EXIT INT TERM

RESULTS_FILE="${LOG_DIR}/all_runs.tsv"
SUMMARY_FILE="${LOG_DIR}/summary.tsv"
{
    printf 'task\trepeat\tseed\tserver_gpu\teval_gpu\tport\tsuccesses\tepisodes\trate_percent\tstatus\teval_log\n'
    for ((worker_id = 0; worker_id < WORKER_COUNT; ++worker_id)); do
        result_file="${LOG_DIR}/worker_${worker_id}_results.tsv"
        [[ -f "${result_file}" ]] && tail -n +2 "${result_file}"
    done
} > "${RESULTS_FILE}"

printf 'task\tcompleted_runs\tfailed_runs\tsuccesses\tepisodes\tpooled_rate_percent\n' > "${SUMMARY_FILE}"
echo
printf '%-24s %10s %10s %14s %10s\n' 'task' 'completed' 'failed' 'success/total' 'rate'
for task in "${TASKS[@]}"; do
    stats="$(awk -F '\t' -v task="${task}" '
        $1 == task {
            if ($10 == "ok") { ok++; successes += $7; episodes += $8 }
            else { failed++ }
        }
        END {
            if (episodes > 0) rate = 100 * successes / episodes; else rate = 0
            printf "%d\t%d\t%d\t%d\t%.1f", ok, failed, successes, episodes, rate
        }' "${RESULTS_FILE}")"
    IFS=$'\t' read -r completed failed successes episodes rate <<< "${stats}"
    printf '%-24s %10s %10s %7s/%-6s %9s%%\n' "${task}" "${completed}" "${failed}" "${successes}" "${episodes}" "${rate}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${task}" "${completed}" "${failed}" "${successes}" "${episodes}" "${rate}" >> "${SUMMARY_FILE}"
done

overall="$(awk -F '\t' '
    NR > 1 && $10 == "ok" { successes += $7; episodes += $8; completed++ }
    NR > 1 && $10 != "ok" { failed++ }
    END {
        if (episodes > 0) rate = 100 * successes / episodes; else rate = 0
        printf "%d\t%d\t%d\t%d\t%.1f", completed, failed, successes, episodes, rate
    }' "${RESULTS_FILE}")"
IFS=$'\t' read -r completed failed successes episodes rate <<< "${overall}"
printf '%-24s %10s %10s %7s/%-6s %9s%%\n' 'OVERALL (pooled)' "${completed}" "${failed}" "${successes}" "${episodes}" "${rate}"
printf 'OVERALL (pooled)\t%s\t%s\t%s\t%s\t%s\n' "${completed}" "${failed}" "${successes}" "${episodes}" "${rate}" >> "${SUMMARY_FILE}"

echo
echo "[INFO] Per-run results: ${RESULTS_FILE}"
echo "[INFO] Summary: ${SUMMARY_FILE}"
if ${worker_failed}; then
    echo "[ERROR] At least one worker failed before completing its assigned runs. Check ${LOG_DIR}." >&2
    exit 1
fi
if awk -F '\t' 'NR > 1 && $10 != "ok" { failed = 1 } END { exit !failed }' "${RESULTS_FILE}"; then
    echo "[ERROR] Some evaluation runs failed; see ${RESULTS_FILE}." >&2
    exit 1
fi
echo "[INFO] All RoboTwin evaluations completed successfully."
