#!/usr/bin/env bash
# RL-Compositionality unified entry point.
#
# Can be invoked from ANY cwd (including a freshly spawned shell). It first
# sources bootstrap.sh to set up env, then dispatches to the appropriate
# experiment script with the right env vars forwarded.
#
# Usage:
#     bash /home/ma-user/work/RL-Compositionality/scripts/env/run.sh <action> [options...]
#
#     actions:
#       prepare           Generate all newops parquet datasets.
#       stage1            Run Stage 1 (RFT/SFT).
#       stage2            Run Stage 2 (GRPO RL).
#       infer             Run the newops inference matrix on the current stage1 model.
#       all               prepare -> stage1 -> stage2 -> infer
#       env               Just source bootstrap and print env (sanity check).
#
#     common options (env passthrough):
#       CASE=baseline|newops          [default: newops]
#       AUTO_RESUME=0|1               [default: 1]
#       N_GPUS_PER_NODE=N             [default: auto-detect]
#       SAVE_FREQ=N                   [default: 50]
#       REMOVE_PREVIOUS_CKPT_IN_SAVE=True|False  [default: True]
#       WANDB_MODE=offline|online|disabled [default: offline]
#       RLCOMP_DRY_RUN=1              Print commands but do not execute.
#
# Examples:
#     bash scripts/env/run.sh env
#     bash scripts/env/run.sh prepare
#     bash scripts/env/run.sh stage1
#     CASE=newops bash scripts/env/run.sh stage2
#     RLCOMP_DRY_RUN=1 bash scripts/env/run.sh all

set -euo pipefail

_RUN_SRC="${BASH_SOURCE[0]}"
_RUN_DIR="$(cd "$(dirname "${_RUN_SRC}")" && pwd)"

# shellcheck disable=SC1091
source "${_RUN_DIR}/bootstrap.sh"

# Safety: bootstrap sets RLCOMP_REPO_ROOT and cds there.
cd "${RLCOMP_REPO_ROOT}"

ACTION="${1:-}"
shift || true

if [ -z "${ACTION}" ]; then
    echo "Usage: bash scripts/env/run.sh <env|prepare|stage1|stage2|infer|all>" >&2
    exit 2
fi

DRY_RUN="${RLCOMP_DRY_RUN:-0}"

_run() {
    local tag
    if [ "${DRY_RUN}" = "1" ]; then tag='[dry-run]'; else tag='[run]'; fi
    printf '%s' "${tag}"
    printf ' %q' "$@"
    printf '\n'
    if [ "${DRY_RUN}" != "1" ]; then
        "$@"
    fi
}

# ---------- action: env ----------
do_env() {
    echo "[run] env action: bootstrap already sourced. Above banner is the state."
    "${RLCOMP_PYTHON}" -c 'import verl, vllm, ray, torch; print(f"[run] verl/vllm/ray/torch import OK, cuda_avail={torch.cuda.is_available()}, n_gpu={torch.cuda.device_count()}")'
}

# ---------- action: prepare ----------
do_prepare() {
    _run bash scripts/experiments/prepare_newops_datasets.sh
}

# ---------- action: stage1 ----------
do_stage1() {
    # run_stage1_newops_rft.sh reads these from env (${VAR:-default}).
    #   N_GPUS_PER_NODE, NNODES, PYTHON_BIN, HF_*, USER_CKPT_ROOT, etc.
    # It also calls examples/sft/template.sh which reads NPROC_PER_NODE.
    export PYTHON_BIN="${RLCOMP_PYTHON}"
    export TORCHRUN_BIN="${VIRTUAL_ENV}/bin/torchrun"
    _run bash scripts/experiments/run_stage1_newops_rft.sh
}

# ---------- action: stage2 ----------
do_stage2() {
    # Uses run_stage2_single_track.sh wrapper (handles auto-resume + wandb run id).
    # That wrapper calls run_stage2_rl_newops_compare.sh which reads:
    #   SAVE_FREQ, TEST_FREQ, REMOVE_PREVIOUS_CKPT_IN_SAVE, N_GPUS_PER_NODE, ...
    export PYTHON_BIN="${RLCOMP_PYTHON}"
    export CASE="${CASE:-newops}"
    export AUTO_RESUME="${AUTO_RESUME:-1}"
    export WANDB_ENABLE="${WANDB_ENABLE:-1}"
    # Pre-flight: verify the datasets Stage 2 needs exist before launching Ray.
    local missing=0
    for f in \
        data/string_task/stage2_level1/train.parquet \
        data/string_task/stage2_level2/train.parquet \
        data/string_task/stage2_level1to8/forward_test.parquet \
        data/string_task/exp_new_ops/stage2_level1to8_newops_test.parquet; do
        if [ ! -f "${f}" ]; then
            echo "[run] stage2 needs missing dataset: ${f}" >&2
            missing=1
        fi
    done
    if [ "${missing}" -ne 0 ]; then
        echo "[run] run 'bash scripts/env/run.sh prepare' (plus stage2_level1/2 generators) first." >&2
        return 1
    fi
    _run bash scripts/experiments/run_stage2_single_track.sh
}

# ---------- action: infer ----------
do_infer() {
    export PYTHON_BIN="${RLCOMP_PYTHON}"
    _run bash scripts/experiments/run_newops_inference_matrix.sh
}

# ---------- action: all ----------
do_all() {
    do_prepare
    do_stage1
    do_stage2
    do_infer
}

case "${ACTION}" in
    env)      do_env ;;
    prepare)  do_prepare ;;
    stage1)   do_stage1 ;;
    stage2)   do_stage2 ;;
    infer)    do_infer ;;
    all)      do_all ;;
    -h|--help)
        sed -n '2,40p' "${_RUN_SRC}"
        ;;
    *)
        echo "Unknown action: ${ACTION}" >&2
        echo "Usage: bash scripts/env/run.sh <env|prepare|stage1|stage2|infer|all>" >&2
        exit 2
        ;;
esac
