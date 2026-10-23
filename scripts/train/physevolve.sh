#!/usr/bin/env bash
set -euo pipefail

# Paper PhysEvolve recipe: 100 steps, four GPUs, KL 0.03.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-physevolve}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_RUN_NAME_ARGS=()
if [[ -n "${WANDB_RUN_NAME}" ]]; then
  WANDB_RUN_NAME_ARGS=(--wandb-run-name "${WANDB_RUN_NAME}")
fi

cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" scripts/train/_entry.py \
  --max-steps 100 \
  --checkpoint-freq 25 \
  --video-save-freq 20 \
  --video-save-max 4 \
  --scenes-per-batch 4 \
  --group-size 4 \
  --temperature 0.7 \
  --top-p 0.95 \
  --solver-rollout-chunk-size 8 \
  --stage2-intrinsic-enabled \
  --stage2-intrinsic-weight 0.1 \
  --stage2-intrinsic-component-weights 0,0,0,1,0 \
  --stage2-proposer-failure-weight 1.5 \
  --stage2-color-variants-per-scene 1 \
  --stage2-max-solver-questions-per-scene 5 \
  --stage2-max-solver-groups-per-step 32 \
  --stage2-min-scene-kind-picks-per-step throwing_pit_board=1,collision_arena=1,motion_lab=1 \
  --stage2-camera-depth-bypass-enabled \
  --stage2-branch-family-weight-multiplier 1.2 \
  --stage2-cold-start-fewshot-steps 100 \
  --stage2-anti-copy-start-step 0 \
  --stage2-learnability-weight 0.5 \
  --stage2-learnability-weight-during-fewshot 0.0 \
  --stage2-family-difficulty-beta 1.0 \
  --stage2-solver-reward-scale 1.5 \
  --stage2-reference-buffer-enabled \
  --stage2-reference-buffer-during-fewshot \
  --stage2-reference-k 2 \
  --stage2-copy-penalty-enabled \
  --stage2-copy-exact-reward -0.8 \
  --stage2-batch-dedup-enabled \
  --stage2-learnability-saturation-threshold 0.75 \
  --stage2-learnability-saturation-weight 1.0 \
  --stage2-diversity-edit-distance-weight 1.5 \
  --kl-coef 0.03 \
  --n-gpus-per-node 4 \
  --ray-num-cpus 32 \
  --executor-workers 4 \
  --model "${MODEL_PATH}" \
  --out-root "${OUT_ROOT:-${REPO_ROOT}/outputs}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --trainer-factory physevolve.training.verl_factory:create_initialized_physevolve_verl_trainer \
  --wandb-project physevolve-stage2 \
  ${WANDB_RUN_NAME_ARGS[@]+"${WANDB_RUN_NAME_ARGS[@]}"} \
  --gpu 0,1,2,3 \
  "$@"
