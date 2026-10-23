#!/usr/bin/env bash
set -euo pipefail

# Stage-2 VQA training, paper 100-step recipe with a frozen Proposer.
#
# This is a single-variable wrapper around scripts/train/physevolve.sh. It keeps
# every paper reward, sampling, curriculum, and KL setting unchanged. The current
# actor still generates gen_vqa scenes; PPO drops those rows so only pred_vqa
# trains. The only recipe deltas are:
#   --freeze-proposer
#   --checkpoint-freq 100      (step-100 checkpoint only)
#
# Usage:
#   bash scripts/train/ablations/freeze_proposer.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-physevolve-freeze-proposer}"

exec bash "${SCRIPT_DIR}/../physevolve.sh" "$@" \
  --freeze-proposer --checkpoint-freq 100
