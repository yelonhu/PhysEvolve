#!/usr/bin/env bash
# Launch official PhysBench evaluation shards, then merge and score once.
#
# Usage:
#   bash scripts/eval/run_physbench_official_eval_multi_gpu.sh \
#       --out-dir outputs/evaluation/physbench \
#       --gpu-list 0,1,2,3 \
#       --model-path Qwen/Qwen3-VL-8B-Instruct \
#       --model-tag Qwen3-VL-8B-Instruct \
#       --split test \
#       --no-resume
#
# Wrapper-only options: --out-dir, --num-gpus, --gpu-list.
# All other options are forwarded directly to run_physbench_official_eval.py.
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
GPU_LIST="${GPU_LIST:-}"   # e.g. "0,1,2,3"; empty -> 0..NUM_GPUS-1
PYTHON_BIN="${PYTHON_BIN:-python3}"

USER_ARGS=()
OUT_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpu-list)
            GPU_LIST="$2"
            shift 2
            ;;
        *)
            USER_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$OUT_DIR" ]]; then
    echo "[fatal] --out-dir is required for sharded official runs." >&2
    exit 2
fi
mkdir -p "$OUT_DIR"

if [[ -z "$GPU_LIST" ]]; then
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
fi
IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
NUM_SHARDS="${#GPU_ARR[@]}"
echo "[info] official sharding across $NUM_SHARDS GPUs: ${GPU_LIST}"
echo "[info] out_dir = $OUT_DIR"
echo "[info] forwarded args = ${USER_ARGS[*]}"

PIDS=()
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    GPU_ID="${GPU_ARR[$SHARD_ID]}"
    LOG_FILE="${OUT_DIR}/shard${SHARD_ID}.stdout.log"
    echo "[info] launching shard ${SHARD_ID}/${NUM_SHARDS} on GPU ${GPU_ID} -> ${LOG_FILE}"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/eval/run_physbench_official_eval.py \
        --num-shards "$NUM_SHARDS" \
        --shard-id "$SHARD_ID" \
        --out-dir "$OUT_DIR" \
        "${USER_ARGS[@]}" > "$LOG_FILE" 2>&1 &
    PIDS+=("$!")
done

FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "[fatal] official shard pid=$PID exited non-zero" >&2
        FAIL=1
    fi
done
if [[ $FAIL -ne 0 ]]; then
    exit 1
fi

echo "[info] all official shards finished; merging predictions and scoring"
"$PYTHON_BIN" scripts/eval/run_physbench_official_eval.py \
    --merge-shards "$OUT_DIR" \
    "${USER_ARGS[@]}"
