#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CASE="${CASE:-baseline}"  # baseline | newops
TRACK_NAME="${TRACK_NAME:-stage2-${CASE}-single-track}"
WANDB_TRACK_DIR="${WANDB_TRACK_DIR:-results/exp_rlcompose/wandb_tracks}"
RUN_ID_FILE="${WANDB_TRACK_DIR}/${TRACK_NAME}.run_id"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-rlcomp/bin/python}"
: "${N_GPUS_PER_NODE:?N_GPUS_PER_NODE not set; source scripts/env/bootstrap.sh first (it auto-detects via nvidia-smi)}"
NNODES="${NNODES:-1}"
WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rlcompose-new-abilities}"
WANDB_RESUME_MODE="${WANDB_RESUME_MODE:-allow}"  # allow|must|never
_RAY_CAND="${ROOT_DIR}/.raytmp"
if [ "${#_RAY_CAND}" -lt 100 ]; then
    RAY_TMPDIR="${RAY_TMPDIR:-${_RAY_CAND}}"
else
    RAY_TMPDIR="${RAY_TMPDIR:-.raytmp}"
fi
unset _RAY_CAND
TMPDIR="${TMPDIR:-${ROOT_DIR}/.tmp}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints_fresh/stage2_runs/${TRACK_NAME}}"
AUTO_RESUME="${AUTO_RESUME:-0}"  # 0: always fresh run, 1: auto-resume from latest ckpt

mkdir -p "${WANDB_TRACK_DIR}"
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}" "${SAVE_ROOT}"
export RAY_TMPDIR TMPDIR

if [[ ! "${CASE}" =~ ^(baseline|newops)$ ]]; then
    echo "Invalid CASE=${CASE}. Expected baseline or newops."
    exit 1
fi

get_max_checkpoint_step() {
    local v
    v="$(ls -1 "${SAVE_ROOT}" 2>/dev/null | rg '^global_step_[0-9]+$' | sed -E 's/global_step_([0-9]+)/\1/' | sort -n | tail -n 1 || true)"
    if [[ -n "${v}" ]]; then
        echo "${v}"
    else
        echo "0"
    fi
}

if [[ -s "${RUN_ID_FILE}" ]]; then
    WANDB_RUN_ID="$(cat "${RUN_ID_FILE}" | tr -d '\n\r')"
else
    WANDB_RUN_ID="$("${PYTHON_BIN}" - <<'PY'
import uuid
print(uuid.uuid4().hex[:16])
PY
)"
    echo "${WANDB_RUN_ID}" > "${RUN_ID_FILE}"
fi

latest_step="$(get_max_checkpoint_step)"
if [[ "${AUTO_RESUME}" == "1" && "${latest_step}" -gt 0 && -d "${SAVE_ROOT}/global_step_${latest_step}" ]]; then
    RESUME_MODE="${SAVE_ROOT}/global_step_${latest_step}"
else
    RESUME_MODE="disable"
fi

echo "[run_stage2_single_track] case=${CASE}"
echo "[run_stage2_single_track] track=${TRACK_NAME}"
echo "[run_stage2_single_track] wandb_run_id=${WANDB_RUN_ID}"
echo "[run_stage2_single_track] resume_mode=${RESUME_MODE}"
echo "[run_stage2_single_track] save_root=${SAVE_ROOT}"

if [[ "${CASE}" == "baseline" ]]; then
    export RUN_CASES="baseline"
    export BASELINE_RESUME_MODE="${RESUME_MODE}"
    export NEWOPS_RESUME_MODE="disable"
    export BASELINE_WANDB_RUN_ID="${WANDB_RUN_ID}"
    export BASELINE_WANDB_RESUME="${WANDB_RESUME_MODE}"
    export BASELINE_WANDB_RUN_NAME="${TRACK_NAME}"
else
    export RUN_CASES="newops"
    export BASELINE_RESUME_MODE="disable"
    export NEWOPS_RESUME_MODE="${RESUME_MODE}"
    export NEWOPS_WANDB_RUN_ID="${WANDB_RUN_ID}"
    export NEWOPS_WANDB_RESUME="${WANDB_RESUME_MODE}"
    export NEWOPS_WANDB_RUN_NAME="${TRACK_NAME}"
fi

export PYTHON_BIN
export N_GPUS_PER_NODE
export NNODES
export WANDB_ENABLE
export WANDB_PROJECT
export SAVE_ROOT

env -u https_proxy -u http_proxy -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u all_proxy \
    bash scripts/experiments/run_stage2_rl_newops_compare.sh
