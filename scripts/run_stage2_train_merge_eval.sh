#!/usr/bin/env bash
set -euo pipefail

# Run the complete PhysEvolve experiment workflow:
#   training -> FSDP checkpoint merge -> PhysBench val -> PhysBench test.
#
# The default recipe is the paper 100-step launcher on GPUs 0,1,2,3. Pipeline
# options are parsed here; unrecognized arguments are forwarded to the training
# launcher. Use "--" to make that forwarding explicit.
#
# Examples:
#   bash scripts/run_stage2_train_merge_eval.sh
#
#   bash scripts/run_stage2_train_merge_eval.sh \
#     --run-dir outputs/<run_id>
#
#   bash scripts/run_stage2_train_merge_eval.sh \
#     --train-script scripts/train/physevolve.sh \
#     --steps "25 50 75 100"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train/physevolve.sh}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs}"
VERL_ROOT="${VERL_ROOT:-${REPO_ROOT}/../verl}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/outputs/evaluation/physbench}"
ANSWERS_DIR="${ANSWERS_DIR:-${REPO_ROOT}/evaluation/physbench/third_party/physbench_answers}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_PATH="${DATASET_PATH:-}"
RUN_DIR=""
RUN_ID=""
STEP_SPEC="${STEPS:-}"
EVAL_TAG=""
SKIP_TRAIN=0
SKIP_MERGE=0
SKIP_VAL=0
SKIP_TEST=0
FORCE_MERGE=0
TRAIN_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_stage2_train_merge_eval.sh [pipeline options] [training options]

Pipeline options:
  --dataset-path PATH  PhysBench dataset root (or DATASET_PATH).
  --run-dir PATH        Use an existing run and skip training automatically.
  --run-id ID           Set the new training run ID, or identify an existing run.
  --train-script PATH   Training launcher (default: scripts/train/physevolve.sh).
  --steps "N ..."       Checkpoint steps. Default: discover them from the run.
  --eval-tag TAG        Result directory prefix. Default: <timestamp>_n<max-step>.
  --out-root PATH       Training output root.
  --verl-root PATH      veRL checkout containing verl.model_merger.
  --results-root PATH   PhysBench result root.
  --gpu-list LIST       Comma-separated evaluation GPUs (default: 0,1,2,3).
  --skip-train          Start from an existing --run-dir or --run-id.
  --skip-merge          Evaluate checkpoints that are already merged.
  --skip-val            Skip PhysBench val.
  --skip-test           Skip PhysBench full test.
  --force-merge         Replace an existing merged checkpoint directory.
  -h, --help            Show this help.

All unrecognized arguments, or arguments after "--", are forwarded to the
training launcher.
EOF
}

fatal() {
    echo "[fatal] $*" >&2
    exit 2
}

require_value() {
    local option="$1"
    local value="${2:-}"
    [[ -n "${value}" ]] || fatal "${option} requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-path)
            require_value "$1" "${2:-}"
            DATASET_PATH="$2"
            shift 2
            ;;
        --run-dir)
            require_value "$1" "${2:-}"
            RUN_DIR="$2"
            shift 2
            ;;
        --run-id)
            require_value "$1" "${2:-}"
            RUN_ID="$2"
            shift 2
            ;;
        --train-script)
            require_value "$1" "${2:-}"
            TRAIN_SCRIPT="$2"
            shift 2
            ;;
        --steps)
            require_value "$1" "${2:-}"
            STEP_SPEC="$2"
            shift 2
            ;;
        --eval-tag)
            require_value "$1" "${2:-}"
            EVAL_TAG="$2"
            shift 2
            ;;
        --out-root)
            require_value "$1" "${2:-}"
            OUT_ROOT="$2"
            shift 2
            ;;
        --verl-root)
            require_value "$1" "${2:-}"
            VERL_ROOT="$2"
            shift 2
            ;;
        --results-root)
            require_value "$1" "${2:-}"
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --gpu-list)
            require_value "$1" "${2:-}"
            GPU_LIST="$2"
            shift 2
            ;;
        --skip-train)
            SKIP_TRAIN=1
            shift
            ;;
        --skip-merge)
            SKIP_MERGE=1
            shift
            ;;
        --skip-val)
            SKIP_VAL=1
            shift
            ;;
        --skip-test)
            SKIP_TEST=1
            shift
            ;;
        --force-merge)
            FORCE_MERGE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            TRAIN_ARGS+=("$@")
            break
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "${SKIP_VAL}" -eq 0 || "${SKIP_TEST}" -eq 0 ]]; then
    [[ -d "${DATASET_PATH}" ]] || fatal "set --dataset-path to the PhysBench dataset directory"
fi

# Keep all filesystem paths stable when the merger changes its working directory.
absolute_path() {
    "${PYTHON_BIN}" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}
OUT_ROOT="$(absolute_path "${OUT_ROOT}")"
VERL_ROOT="$(absolute_path "${VERL_ROOT}")"
RESULTS_ROOT="$(absolute_path "${RESULTS_ROOT}")"
ANSWERS_DIR="$(absolute_path "${ANSWERS_DIR}")"
if [[ -n "${DATASET_PATH}" ]]; then
    DATASET_PATH="$(absolute_path "${DATASET_PATH}")"
fi

[[ -n "${GPU_LIST}" ]] || fatal "--gpu-list cannot be empty"
[[ -d "${VERL_ROOT}" ]] || fatal "veRL root does not exist: ${VERL_ROOT}"
[[ -d "${ANSWERS_DIR}" ]] || fatal "PhysBench answers directory does not exist: ${ANSWERS_DIR}"

if [[ "${TRAIN_SCRIPT}" = /* ]]; then
    TRAIN_SCRIPT_PATH="${TRAIN_SCRIPT}"
else
    TRAIN_SCRIPT_PATH="${REPO_ROOT}/${TRAIN_SCRIPT}"
fi
[[ -f "${TRAIN_SCRIPT_PATH}" ]] || fatal "training launcher does not exist: ${TRAIN_SCRIPT_PATH}"

if [[ -n "${RUN_DIR}" ]]; then
    SKIP_TRAIN=1
    RUN_DIR="$(realpath -m "${RUN_DIR}")"
    if [[ -n "${RUN_ID}" && "$(basename "${RUN_DIR}")" != "${RUN_ID}" ]]; then
        fatal "--run-dir basename and --run-id disagree"
    fi
    RUN_ID="$(basename "${RUN_DIR}")"
else
    if [[ -z "${RUN_ID}" ]]; then
        if [[ "${SKIP_TRAIN}" -eq 1 ]]; then
            fatal "--skip-train requires --run-dir or --run-id"
        fi
        DEFAULT_MAX_STEPS=100
        TRAIN_SCRIPT_NAME="$(basename "${TRAIN_SCRIPT_PATH}")"
        if [[ "${TRAIN_SCRIPT_NAME}" =~ _([0-9]+)step_ ]]; then
            DEFAULT_MAX_STEPS="${BASH_REMATCH[1]}"
        fi
        RUN_ID="$(date '+%Y%m%d_%H%M')_physevolve_qwen3vl8binstruct_native_verl_trainer_n${DEFAULT_MAX_STEPS}"
    fi
    [[ "${RUN_ID}" != */* ]] || fatal "--run-id must be a directory name, not a path"
    RUN_DIR="${OUT_ROOT%/}/${RUN_ID}"
fi

if [[ "${SKIP_TRAIN}" -eq 1 ]]; then
    [[ -d "${RUN_DIR}" ]] || fatal "existing run directory does not exist: ${RUN_DIR}"
else
    [[ ! -e "${RUN_DIR}" ]] || fatal "run directory already exists; use --run-dir to resume: ${RUN_DIR}"
    mkdir -p "${OUT_ROOT}"
    TRAIN_TEMP_LOG="${OUT_ROOT%/}/.${RUN_ID}.train.log"

    echo "[pipeline] run_dir=${RUN_DIR}"
    echo "[pipeline] phase=train launcher=${TRAIN_SCRIPT_PATH}"
    if ! (
        cd "${REPO_ROOT}"
        bash "${TRAIN_SCRIPT_PATH}" \
            "${TRAIN_ARGS[@]}" \
            --run-id "${RUN_ID}" \
            --out-root "${OUT_ROOT}"
    ) 2>&1 | tee "${TRAIN_TEMP_LOG}"; then
        if [[ -d "${RUN_DIR}" ]]; then
            mkdir -p "${RUN_DIR}/_pipeline"
            mv "${TRAIN_TEMP_LOG}" "${RUN_DIR}/_pipeline/train.log"
        fi
        echo "[fatal] training failed" >&2
        exit 1
    fi

    [[ -d "${RUN_DIR}" ]] || fatal "training succeeded but run directory was not created: ${RUN_DIR}"
    mkdir -p "${RUN_DIR}/_pipeline"
    mv "${TRAIN_TEMP_LOG}" "${RUN_DIR}/_pipeline/train.log"
fi

PIPELINE_LOG_DIR="${RUN_DIR}/_pipeline"
mkdir -p "${PIPELINE_LOG_DIR}"

declare -a STEPS_ARRAY=()
if [[ -n "${STEP_SPEC}" ]]; then
    STEP_SPEC="${STEP_SPEC//,/ }"
    read -r -a RAW_STEPS <<< "${STEP_SPEC}"
    for STEP in "${RAW_STEPS[@]}"; do
        [[ "${STEP}" =~ ^[1-9][0-9]*$ ]] || fatal "invalid checkpoint step: ${STEP}"
    done
    mapfile -t STEPS_ARRAY < <(printf '%s\n' "${RAW_STEPS[@]}" | sort -nu)
else
    declare -a DISCOVERED_STEPS=()
    shopt -s nullglob
    for ACTOR_DIR in "${RUN_DIR}"/global_step_*/actor; do
        STEP_DIR="$(basename "$(dirname "${ACTOR_DIR}")")"
        STEP="${STEP_DIR#global_step_}"
        [[ "${STEP}" =~ ^[1-9][0-9]*$ ]] && DISCOVERED_STEPS+=("${STEP}")
    done
    for MERGED_DIR in "${RUN_DIR}"/merged_hf_global_step_*; do
        STEP="${MERGED_DIR##*_}"
        [[ "${STEP}" =~ ^[1-9][0-9]*$ ]] && DISCOVERED_STEPS+=("${STEP}")
    done
    shopt -u nullglob
    if [[ "${#DISCOVERED_STEPS[@]}" -gt 0 ]]; then
        mapfile -t STEPS_ARRAY < <(printf '%s\n' "${DISCOVERED_STEPS[@]}" | sort -nu)
    fi
fi

[[ "${#STEPS_ARRAY[@]}" -gt 0 ]] || fatal "no checkpoint steps found under ${RUN_DIR}; pass --steps explicitly"
STEPS_TEXT="${STEPS_ARRAY[*]}"
MAX_STEP="${STEPS_ARRAY[$((${#STEPS_ARRAY[@]} - 1))]}"

if [[ -z "${EVAL_TAG}" ]]; then
    RUN_BASENAME="$(basename "${RUN_DIR}")"
    if [[ "${RUN_BASENAME}" =~ ^([0-9]{8}_[0-9]{4})_ ]]; then
        RUN_STAMP="${BASH_REMATCH[1]}"
    else
        RUN_STAMP="$(date '+%Y%m%d_%H%M')"
    fi
    EVAL_TAG="${RUN_STAMP}_n${MAX_STEP}"
fi

VAL_ROOT="${RESULTS_ROOT%/}/${EVAL_TAG}_val"
TEST_ROOT="${RESULTS_ROOT%/}/${EVAL_TAG}_test9802"

{
    echo "[pipeline] run_dir=${RUN_DIR}"
    echo "[pipeline] steps=${STEPS_TEXT}"
    echo "[pipeline] gpu_list=${GPU_LIST}"
    echo "[pipeline] val_root=${VAL_ROOT}"
    echo "[pipeline] test_root=${TEST_ROOT}"
} | tee "${PIPELINE_LOG_DIR}/pipeline.log"

if [[ "${SKIP_MERGE}" -eq 0 ]]; then
    echo "[pipeline] phase=merge" | tee "${PIPELINE_LOG_DIR}/merge.log"
    for STEP in "${STEPS_ARRAY[@]}"; do
        LOCAL_DIR="${RUN_DIR}/global_step_${STEP}/actor"
        TARGET_DIR="${RUN_DIR}/merged_hf_global_step_${STEP}"

        if [[ -d "${TARGET_DIR}" && -n "$(ls -A "${TARGET_DIR}")" && "${FORCE_MERGE}" -eq 0 ]]; then
            echo "[merge] step=${STEP} already merged; skipping ${TARGET_DIR}" \
                | tee -a "${PIPELINE_LOG_DIR}/merge.log"
            continue
        fi
        [[ -d "${LOCAL_DIR}" ]] || fatal "FSDP actor checkpoint does not exist: ${LOCAL_DIR}"

        if [[ -e "${TARGET_DIR}" ]]; then
            if [[ "${FORCE_MERGE}" -eq 1 ]]; then
                [[ "${TARGET_DIR}" == "${RUN_DIR}/merged_hf_global_step_${STEP}" ]] \
                    || fatal "refusing to remove unexpected merge target: ${TARGET_DIR}"
                rm -rf "${TARGET_DIR}"
            else
                fatal "merge target exists but is empty; rerun with --force-merge: ${TARGET_DIR}"
            fi
        fi

        echo "[merge] step=${STEP} target=${TARGET_DIR}" \
            | tee -a "${PIPELINE_LOG_DIR}/merge.log"
        (
            cd "${VERL_ROOT}"
            "${PYTHON_BIN}" -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "${LOCAL_DIR}" \
                --target_dir "${TARGET_DIR}"
        ) 2>&1 | tee -a "${PIPELINE_LOG_DIR}/merge.log"
    done
else
    echo "[pipeline] phase=merge skipped" | tee "${PIPELINE_LOG_DIR}/merge.log"
fi

for STEP in "${STEPS_ARRAY[@]}"; do
    MERGED_DIR="${RUN_DIR}/merged_hf_global_step_${STEP}"
    [[ -d "${MERGED_DIR}" && -n "$(ls -A "${MERGED_DIR}")" ]] \
        || fatal "merged checkpoint is missing or empty: ${MERGED_DIR}"
done

run_eval_phase() {
    local split="$1"
    local phase_root="$2"
    local output_suffix="$3"
    local all_log="$4"

    mkdir -p "${phase_root}/_logs"
    for step in "${STEPS_ARRAY[@]}"; do
        local tag="merged_hf_global_step_${step}"
        local output_dir="${phase_root}/${tag}_${output_suffix}"
        local step_log="${phase_root}/_logs/${tag}_${output_suffix}.log"

        echo "===== $(date '+%F %T') | ${tag} | ${split} =====" | tee -a "${all_log}"
        (
            cd "${REPO_ROOT}"
            PYTHON_BIN="${PYTHON_BIN}" bash scripts/eval/run_physbench_official_eval_multi_gpu.sh \
                --out-dir "${output_dir}" \
                --gpu-list "${GPU_LIST}" \
                --model-path "${RUN_DIR}/${tag}" \
                --model-tag "${tag}" \
                --split "${split}" \
                --dataset-path "${DATASET_PATH}" \
                --answers-dir "${ANSWERS_DIR}" \
                --no-resume
        ) 2>&1 | tee -a "${step_log}" "${all_log}"
    done
}

if [[ "${SKIP_VAL}" -eq 0 ]]; then
    echo "[pipeline] phase=val root=${VAL_ROOT}"
    mkdir -p "${VAL_ROOT}/_logs"
    run_eval_phase "val" "${VAL_ROOT}" "val" "${VAL_ROOT}/_logs/val_all.log"
else
    echo "[pipeline] phase=val skipped"
fi

if [[ "${SKIP_TEST}" -eq 0 ]]; then
    echo "[pipeline] phase=test root=${TEST_ROOT}"
    mkdir -p "${TEST_ROOT}/_logs"
    run_eval_phase "test" "${TEST_ROOT}" "test" "${TEST_ROOT}/_logs/test_all.log"
else
    echo "[pipeline] phase=test skipped"
fi

echo "[pipeline] completed run_dir=${RUN_DIR}"
