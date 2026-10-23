#!/usr/bin/env bash
# Launch CausalPhys zero-shot VLM shards, then merge and score locally.
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
GPU_LIST="${GPU_LIST:-}"
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
    echo "[fatal] --out-dir is required for sharded CausalPhys runs." >&2
    exit 2
fi
mkdir -p "$OUT_DIR"

if [[ -z "$GPU_LIST" ]]; then
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
fi
IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
NUM_SHARDS="${#GPU_ARR[@]}"

echo "[info] CausalPhys sharding across $NUM_SHARDS GPUs: $GPU_LIST"
echo "[info] out_dir = $OUT_DIR"

PIDS=()
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    GPU_ID="${GPU_ARR[$SHARD_ID]}"
    LOG_FILE="${OUT_DIR}/shard${SHARD_ID}.stdout.log"
    echo "[info] shard $SHARD_ID/$NUM_SHARDS on GPU $GPU_ID -> $LOG_FILE"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
        scripts/eval/run_causalphys_vlm_eval.py \
        --num-shards "$NUM_SHARDS" \
        --shard-id "$SHARD_ID" \
        --out-dir "$OUT_DIR" \
        "${USER_ARGS[@]}" > "$LOG_FILE" 2>&1 &
    PIDS+=("$!")
done

FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "[fatal] CausalPhys shard pid=$PID exited non-zero" >&2
        FAIL=1
    fi
done
if [[ $FAIL -ne 0 ]]; then
    exit 1
fi

echo "[info] all CausalPhys shards finished; merging and scoring"
"$PYTHON_BIN" scripts/eval/run_causalphys_vlm_eval.py \
    --merge-shards "$OUT_DIR" \
    "${USER_ARGS[@]}"
