#!/usr/bin/env bash
set -euo pipefail

# Train the paper single-trajectory Naive baseline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"

cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" scripts/smoke/run_selfplay_2step_smoke.py \
  --max-steps 2 \
  --checkpoint-freq 0 \
  --scenes-per-batch 4 \
  --group-size 4 \
  --n-gpus-per-node 8 \
  --ray-num-cpus 64 \
  --model "${MODEL_PATH}" \
  --out-root "${OUT_ROOT:-${REPO_ROOT}/outputs}" \
  --experiment-name naive_baseline_smoke \
  --trainer-factory physevolve.training.verl_factory:create_initialized_physevolve_verl_trainer \
  --wandb-project physevolve-stage1 \
  --wandb-run-name naive_baseline_smoke \
  --gpu all \
  "$@"
