#!/usr/bin/env bash
# Independent launcher for the first eight-GPU machine.

set -euo pipefail

if (($# != 2)); then
    echo "Usage: $0 ARTIFACT_ROOT (--prepare-only|--run)" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARTIFACT_ROOT=$(realpath -m "$1")
MODE=$2
[[ $MODE == --run || $MODE == --prepare-only ]] || {
    echo "Use --prepare-only to inspect queues, or --run to execute all phases" >&2
    exit 2
}
JOBS_FILE=$ARTIFACT_ROOT/manifests/baseline_jobs_machine_1.txt
NON_RND_JOBS_FILE=$ARTIFACT_ROOT/manifests/baseline_non_rnd_machine_1.txt
RND_JOBS_FILE=$ARTIFACT_ROOT/manifests/baseline_rnd_machine_1.txt
PREREQUISITES_FILE=$ARTIFACT_ROOT/manifests/baseline_prerequisites_machine_1.txt
MANIFEST=$ARTIFACT_ROOT/manifests/baseline_jobs_machine_1.json
LOG_DIR=$ARTIFACT_ROOT/queue/machine1

python "$SCRIPT_DIR/generate_baseline_matrix.py" \
    --machine 1 --artifact-root "$ARTIFACT_ROOT" \
    --prerequisite-jobs-file "$PREREQUISITES_FILE" \
    --non-rnd-jobs-file "$NON_RND_JOBS_FILE" --rnd-jobs-file "$RND_JOBS_FILE" \
    --jobs-file "$JOBS_FILE" --manifest "$MANIFEST"

if [[ $MODE == --prepare-only ]]; then
    exit 0
fi

echo 'PHASE 1/2: non-R&D baselines (44 runs)'
bash "$SCRIPT_DIR/run_two_per_gpu_queue.sh" \
    --jobs-file "$NON_RND_JOBS_FILE" \
    --log-dir "$LOG_DIR/non_rnd" \
    --gpus 0,1,2,3,4,5,6,7 \
    --slots-per-gpu 2 \
    --num-shards 1 \
    --shard-index 0

echo 'PHASE 2a/2: R&D teachers and rollouts (48 Meta-World prerequisites)'
bash "$SCRIPT_DIR/run_two_per_gpu_queue.sh" \
    --jobs-file "$PREREQUISITES_FILE" \
    --log-dir "$LOG_DIR/teachers" \
    --gpus 0,1,2,3,4,5,6,7 \
    --slots-per-gpu 2 \
    --num-shards 1 \
    --shard-index 0

echo 'PHASE 2b/2: R&D student distillation (9 Meta-World runs)'
exec bash "$SCRIPT_DIR/run_two_per_gpu_queue.sh" \
    --jobs-file "$RND_JOBS_FILE" \
    --log-dir "$LOG_DIR/rnd" \
    --gpus 0,1,2,3,4,5,6,7 \
    --slots-per-gpu 2 \
    --num-shards 1 \
    --shard-index 0
