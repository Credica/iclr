#!/usr/bin/env bash
# Independent launcher for the second eight-GPU machine.

set -euo pipefail

if (($# != 2)); then
    echo "Usage: $0 ARTIFACT_ROOT (--prepare-only|--run)" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARTIFACT_ROOT=$(realpath -m "$1")
MODE=$2
[[ $MODE == --run || $MODE == --prepare-only ]] || {
    echo "Second argument must be --prepare-only or --run" >&2
    exit 2
}
JOBS_FILE=$ARTIFACT_ROOT/manifests/baseline_jobs_machine_2.txt
PREREQUISITES_FILE=$ARTIFACT_ROOT/manifests/baseline_prerequisites_machine_2.txt
MANIFEST=$ARTIFACT_ROOT/manifests/baseline_jobs_machine_2.json
LOG_DIR=$ARTIFACT_ROOT/queue/machine2

python "$SCRIPT_DIR/generate_baseline_matrix.py" \
    --machine 2 --artifact-root "$ARTIFACT_ROOT" \
    --prerequisite-jobs-file "$PREREQUISITES_FILE" \
    --jobs-file "$JOBS_FILE" --manifest "$MANIFEST"

if [[ $MODE == --prepare-only ]]; then
    exit 0
fi

bash "$SCRIPT_DIR/run_two_per_gpu_queue.sh" \
    --jobs-file "$PREREQUISITES_FILE" \
    --log-dir "$LOG_DIR/teachers" \
    --gpus 0,1,2,3,4,5,6,7 \
    --slots-per-gpu 2 \
    --num-shards 1 \
    --shard-index 0

exec bash "$SCRIPT_DIR/run_two_per_gpu_queue.sh" \
    --jobs-file "$JOBS_FILE" \
    --log-dir "$LOG_DIR/baselines" \
    --gpus 0,1,2,3,4,5,6,7 \
    --slots-per-gpu 2 \
    --num-shards 1 \
    --shard-index 0
