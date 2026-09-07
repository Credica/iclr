#!/usr/bin/env bash
# Run a command queue with at most two concurrent workers on each GPU.
#
# Job-file format: one shell command per non-empty, non-comment line. Run the
# same immutable job file on both machines and use --shard-index 0/1 so every
# command is claimed by exactly one machine.

set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_two_per_gpu_queue.sh --jobs-file FILE --log-dir DIR [options]

Required:
  --jobs-file FILE       One shell command per line.
  --log-dir DIR          Queue logs/status directory (keep outside the repo).

Options:
  --gpus LIST            Comma-separated local GPU IDs (default: 0,1,2,3,4,5,6,7).
  --slots-per-gpu N      Concurrent commands per GPU (default: 2).
  --num-shards N         Number of machines sharing the job list (default: 1).
  --shard-index N        This machine's zero-based shard index (default: 0).
  --workdir DIR          Working directory for commands (default: repository root).
  --poll-seconds N       Queue polling interval (default: 5).
  --dry-run              Print this shard's commands without starting them.
  -h, --help             Show this help.

Example for two eight-GPU machines:
  # machine 1
  bash scripts/run_two_per_gpu_queue.sh --jobs-file /data/baseline_jobs.txt \
    --log-dir /data/baseline_queue/node0 --num-shards 2 --shard-index 0

  # machine 2
  bash scripts/run_two_per_gpu_queue.sh --jobs-file /data/baseline_jobs.txt \
    --log-dir /data/baseline_queue/node1 --num-shards 2 --shard-index 1
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
JOBS_FILE=''
LOG_DIR=''
GPU_LIST='0,1,2,3,4,5,6,7'
SLOTS_PER_GPU=2
NUM_SHARDS=1
SHARD_INDEX=0
WORKDIR=$REPO_DIR
POLL_SECONDS=5
DRY_RUN=false

while (($#)); do
    case "$1" in
        --jobs-file) JOBS_FILE=${2:?missing value for --jobs-file}; shift 2 ;;
        --log-dir) LOG_DIR=${2:?missing value for --log-dir}; shift 2 ;;
        --gpus) GPU_LIST=${2:?missing value for --gpus}; shift 2 ;;
        --slots-per-gpu) SLOTS_PER_GPU=${2:?missing value for --slots-per-gpu}; shift 2 ;;
        --num-shards) NUM_SHARDS=${2:?missing value for --num-shards}; shift 2 ;;
        --shard-index) SHARD_INDEX=${2:?missing value for --shard-index}; shift 2 ;;
        --workdir) WORKDIR=${2:?missing value for --workdir}; shift 2 ;;
        --poll-seconds) POLL_SECONDS=${2:?missing value for --poll-seconds}; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n $JOBS_FILE ]] || { echo '--jobs-file is required' >&2; exit 2; }
[[ -n $LOG_DIR ]] || { echo '--log-dir is required' >&2; exit 2; }
[[ -f $JOBS_FILE ]] || { echo "Job file not found: $JOBS_FILE" >&2; exit 2; }
[[ -d $WORKDIR ]] || { echo "Working directory not found: $WORKDIR" >&2; exit 2; }
[[ $SLOTS_PER_GPU =~ ^[1-9][0-9]*$ ]] || { echo '--slots-per-gpu must be positive' >&2; exit 2; }
[[ $NUM_SHARDS =~ ^[1-9][0-9]*$ ]] || { echo '--num-shards must be positive' >&2; exit 2; }
[[ $SHARD_INDEX =~ ^[0-9]+$ ]] || { echo '--shard-index must be non-negative' >&2; exit 2; }
[[ $POLL_SECONDS =~ ^[1-9][0-9]*$ ]] || { echo '--poll-seconds must be positive' >&2; exit 2; }
((SHARD_INDEX < NUM_SHARDS)) || { echo '--shard-index must be smaller than --num-shards' >&2; exit 2; }

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
((${#GPUS[@]} > 0)) || { echo '--gpus must not be empty' >&2; exit 2; }
declare -A GPU_SEEN=()
for gpu in "${GPUS[@]}"; do
    [[ $gpu =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: $gpu" >&2; exit 2; }
    [[ -z ${GPU_SEEN[$gpu]+x} ]] || { echo "Duplicate GPU ID: $gpu" >&2; exit 2; }
    GPU_SEEN[$gpu]=1
done

declare -a COMMANDS=()
declare -a JOB_IDS=()
global_job_index=0
while IFS= read -r command || [[ -n $command ]]; do
    [[ $command =~ ^[[:space:]]*$ ]] && continue
    [[ $command =~ ^[[:space:]]*# ]] && continue
    if ((global_job_index % NUM_SHARDS == SHARD_INDEX)); then
        COMMANDS+=("$command")
        JOB_IDS+=("$((global_job_index + 1))")
    fi
    ((global_job_index += 1))
done < "$JOBS_FILE"

if $DRY_RUN; then
    echo "SHARD $SHARD_INDEX/$NUM_SHARDS: ${#COMMANDS[@]} of $global_job_index jobs"
    for index in "${!COMMANDS[@]}"; do
        printf 'JOB %05d\t%s\n' "${JOB_IDS[$index]}" "${COMMANDS[$index]}"
    done
    exit 0
fi

mkdir -p "$LOG_DIR"
STATUS_FILE=$LOG_DIR/queue_status.tsv
printf 'timestamp\tevent\tjob_id\tgpu\tpid\texit_code\tlog\n' > "$STATUS_FILE"

declare -A GPU_LOAD=()
declare -A PID_JOB=()
declare -A PID_GPU=()
declare -A PID_LOG=()
for gpu in "${GPUS[@]}"; do
    GPU_LOAD[$gpu]=0
done

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

append_status() {
    printf '%s\t%s\t%05d\t%s\t%s\t%s\t%s\n' \
        "$(timestamp)" "$1" "$2" "$3" "$4" "$5" "$6" >> "$STATUS_FILE"
}

stop_children() {
    trap - INT TERM
    echo 'Stopping active workers...' >&2
    for pid in "${!PID_JOB[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${!PID_JOB[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit 130
}
trap stop_children INT TERM

next_job=0
completed=0
failures=0
total=${#COMMANDS[@]}

echo "QUEUE_START jobs=$total gpus=${GPUS[*]} slots_per_gpu=$SLOTS_PER_GPU shard=$SHARD_INDEX/$NUM_SHARDS"

while ((completed < total)); do
    # Reap completed processes and release their GPU slots.
    for pid in "${!PID_JOB[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            exit_code=0
            wait "$pid" || exit_code=$?
            job_id=${PID_JOB[$pid]}
            gpu=${PID_GPU[$pid]}
            log=${PID_LOG[$pid]}
            ((GPU_LOAD[$gpu] -= 1))
            ((completed += 1))
            if ((exit_code == 0)); then
                event=completed
            else
                event=failed
                ((failures += 1))
            fi
            append_status "$event" "$job_id" "$gpu" "$pid" "$exit_code" "$log"
            echo "JOB_${event^^} id=$job_id gpu=$gpu pid=$pid exit=$exit_code ($completed/$total)"
            unset 'PID_JOB[$pid]' 'PID_GPU[$pid]' 'PID_LOG[$pid]'
        fi
    done

    # Fill every free slot. A finished job is therefore replaced automatically.
    for gpu in "${GPUS[@]}"; do
        while ((GPU_LOAD[$gpu] < SLOTS_PER_GPU && next_job < total)); do
            job_id=${JOB_IDS[$next_job]}
            command=${COMMANDS[$next_job]}
            log=$LOG_DIR/job_$(printf '%05d' "$job_id")_gpu${gpu}.log
            (
                cd "$WORKDIR" || exit 125
                export CUDA_VISIBLE_DEVICES=$gpu
                export OMP_NUM_THREADS=1
                export MKL_NUM_THREADS=1
                export OPENBLAS_NUM_THREADS=1
                export NUMEXPR_NUM_THREADS=1
                bash -c "$command"
            ) > "$log" 2>&1 &
            pid=$!
            PID_JOB[$pid]=$job_id
            PID_GPU[$pid]=$gpu
            PID_LOG[$pid]=$log
            ((GPU_LOAD[$gpu] += 1))
            ((next_job += 1))
            append_status launched "$job_id" "$gpu" "$pid" '-' "$log"
            echo "JOB_LAUNCHED id=$job_id gpu=$gpu pid=$pid log=$log"
        done
    done

    ((completed < total)) && sleep "$POLL_SECONDS"
done

echo "QUEUE_DONE jobs=$total failures=$failures status=$STATUS_FILE"
((failures == 0))
