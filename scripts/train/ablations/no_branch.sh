#!/usr/bin/env bash
set -euo pipefail

# Stage-2 VQA training, paper 100-step recipe without branch elision.
#
# This is a single-variable wrapper around scripts/train/physevolve.sh. It keeps
# every paper reward, sampling multiplier, curriculum, and KL setting unchanged.
# The only recipe deltas are:
#   --disable-branch-elision   (drop CF / inverse families + skip branch sims)
#   --checkpoint-freq 100      (step-100 checkpoint only)
#
# Usage:
#   bash scripts/train/ablations/no_branch.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-physevolve-no-branch}"

exec bash "${SCRIPT_DIR}/../physevolve.sh" "$@" \
  --disable-branch-elision --checkpoint-freq 100
