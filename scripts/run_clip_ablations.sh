#!/usr/bin/env bash
# Prepare or run three groups of full-sequence Clip ablations.
set -euo pipefail
if (($# < 3)); then
    echo "Usage: $0 ARTIFACT_ROOT GPU_IDS (--prepare-only|--run) [generator options]" >&2
    echo "Options: --groups sides schedule interval --sequences F1 D-W6 --seeds 1 2 3" >&2
    echo "         --clip-intervals 100000 200000 400000 --include-controls" >&2
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
python "$SCRIPT_DIR/generate_clip_ablations.py" --artifact-root "$ARTIFACT_ROOT" "$@"
if [[ $MODE == --prepare-only ]]; then
    exit 0
fi
exec bash "$SCRIPT_DIR/run_two_per_gpu_queue.sh" \
    --jobs-file "$ARTIFACT_ROOT/manifests/clip_ablation_jobs.txt" \
    --log-dir "$ARTIFACT_ROOT/queue/clip_ablations" \
    --gpus "$GPU_IDS" --slots-per-gpu 2 --num-shards 1 --shard-index 0
