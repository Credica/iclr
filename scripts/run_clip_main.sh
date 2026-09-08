#!/usr/bin/env bash
# Local Clip queue, independent of the two friends' baseline queues.
set -euo pipefail
if (($# < 3)); then
    echo "Usage: $0 ARTIFACT_ROOT GPU_IDS (--prepare-only|--run) [--singular_clip_min LOWER] [--singular_clip_max UPPER]" >&2
    exit 2
fi
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARTIFACT_ROOT=$(realpath -m "$1")
GPU_IDS=$2
MODE=$3
shift 3
[[ $MODE == --run || $MODE == --prepare-only ]] || {
    echo "Third argument must be --prepare-only or --run" >&2
    exit 2
}
python "$SCRIPT_DIR/generate_clip_matrix.py" --artifact-root "$ARTIFACT_ROOT" "$@"
if [[ $MODE == --prepare-only ]]; then
    exit 0
fi
exec bash "$SCRIPT_DIR/run_two_per_gpu_queue.sh" \
    --jobs-file "$ARTIFACT_ROOT/manifests/clip_jobs.txt" \
    --log-dir "$ARTIFACT_ROOT/queue/clip" \
    --gpus "$GPU_IDS" --slots-per-gpu 2 --num-shards 1 --shard-index 0
