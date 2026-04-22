#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-rlcomp/bin/python}"
# Keep large temp artifacts off root disk (/, /tmp) by default.
TMP_WORK_DIR="${TMP_WORK_DIR:-${ROOT_DIR}/.tmp}"
export TMPDIR="${TMPDIR:-${TMP_WORK_DIR}}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
# Keep Ray socket path short to avoid AF_UNIX (107-byte) path-length failures.
export RAY_TMPDIR="${RAY_TMPDIR:-/home/ma-user/work/.raytmp}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"
# Avoid mixing user-site packages (~/.local) into training runtime.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

MODEL_STORE_ROOT="${MODEL_STORE_ROOT:-/home/ma-user/work/model_store}"
HF_MODEL_ROOT="${HF_MODEL_ROOT:-${MODEL_STORE_ROOT}/hf_models}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${MODEL_STORE_ROOT}/hf_cache}"
USER_CKPT_ROOT="${USER_CKPT_ROOT:-${ROOT_DIR}/checkpoints_fresh}"
mkdir -p "${HF_MODEL_ROOT}" "${HF_CACHE_ROOT}" "${USER_CKPT_ROOT}"
export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_ROOT}/transformers}"

TRAIN_FILES="${TRAIN_FILES:-['data/string_task/stage2_level1/train.parquet','data/string_task/stage2_level2/train.parquet']}"
VAL_FILES="${VAL_FILES:-['data/string_task/stage2_level1to8/forward_test.parquet','data/string_task/exp_new_ops/stage2_level1to8_newops_test.parquet']}"

NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
SAVE_ROOT="${SAVE_ROOT:-${USER_CKPT_ROOT}/stage2_runs}"
PPO_EXTRA_OVERRIDES="${PPO_EXTRA_OVERRIDES:-}"
PPO_MEMORY_SAFE_OVERRIDES="${PPO_MEMORY_SAFE_OVERRIDES:-}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
TEST_FREQ="${TEST_FREQ:-500}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
REMOVE_PREVIOUS_CKPT_IN_SAVE="${REMOVE_PREVIOUS_CKPT_IN_SAVE:-False}"
ROLLOUT_ENGINE="${ROLLOUT_ENGINE:-vllm}"
BASELINE_RESUME_MODE="${BASELINE_RESUME_MODE:-disable}"
NEWOPS_RESUME_MODE="${NEWOPS_RESUME_MODE:-disable}"
RUN_CASES="${RUN_CASES:-baseline}"
BASELINE_WANDB_RUN_ID="${BASELINE_WANDB_RUN_ID:-}"
BASELINE_WANDB_RESUME="${BASELINE_WANDB_RESUME:-}"
BASELINE_WANDB_RUN_NAME="${BASELINE_WANDB_RUN_NAME:-}"
NEWOPS_WANDB_RUN_ID="${NEWOPS_WANDB_RUN_ID:-}"
NEWOPS_WANDB_RESUME="${NEWOPS_WANDB_RESUME:-}"
NEWOPS_WANDB_RUN_NAME="${NEWOPS_WANDB_RUN_NAME:-}"
FLASH_ATTN_AVAILABLE=1

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import flash_attn
PY
then
    FLASH_ATTN_AVAILABLE=0
fi

if [[ "${FLASH_ATTN_AVAILABLE}" -eq 0 ]]; then
    if [[ -z "${PPO_MEMORY_SAFE_OVERRIDES}" ]]; then
        PPO_MEMORY_SAFE_OVERRIDES="data.max_response_length=512 actor_rollout_ref.rollout.response_length=512 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 data.gen_batch_size=4 data.train_batch_size=4 actor_rollout_ref.actor.ppo_mini_batch_size=4 actor_rollout_ref.rollout.n=2 actor_rollout_ref.rollout.gpu_memory_utilization=0.45 actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True algorithm.filter_groups.max_num_gen_batches=16"
    fi
    # Keep fallback values conservative, but allow caller overrides to take precedence.
    PPO_EXTRA_OVERRIDES="actor_rollout_ref.model.use_remove_padding=False critic.model.use_remove_padding=False reward_model.model.use_remove_padding=False ${PPO_MEMORY_SAFE_OVERRIDES} ${PPO_EXTRA_OVERRIDES}"
    # vLLM's CuMemAllocator is incompatible with expandable_segments mode.
    if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
        echo "[run_stage2_rl_newops_compare] removing incompatible PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
        unset PYTORCH_CUDA_ALLOC_CONF
    fi
    echo "[run_stage2_rl_newops_compare] flash-attn unavailable; using fallback overrides:${PPO_EXTRA_OVERRIDES}"
    echo "[run_stage2_rl_newops_compare] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-<unset>}"
fi
if [[ "${ROLLOUT_ENGINE}" != "vllm" ]]; then
    echo "[run_stage2_rl_newops_compare] Unsupported ROLLOUT_ENGINE=${ROLLOUT_ENGINE}. This pipeline enforces vllm."
    exit 1
fi
echo "[run_stage2_rl_newops_compare] checkpoint policy: save_freq=${SAVE_FREQ}, remove_previous_ckpt_in_save=${REMOVE_PREVIOUS_CKPT_IN_SAVE}"
echo "[run_stage2_rl_newops_compare] eval policy: test_freq=${TEST_FREQ}, total_epochs=${TOTAL_EPOCHS}"
echo "[run_stage2_rl_newops_compare] rollout policy: actor_rollout_ref.rollout.name=${ROLLOUT_ENGINE}"

for required in \
    data/string_task/stage2_level1/train.parquet \
    data/string_task/stage2_level2/train.parquet \
    data/string_task/stage2_level1to8/forward_test.parquet \
    data/string_task/exp_new_ops/stage2_level1to8_newops_test.parquet; do
    if [[ ! -f "${required}" ]]; then
        echo "Missing required dataset: ${required}"
        echo "Please prepare Stage2 train/eval data first (e.g. bash bash/section41_42/stage2_create_problems.sh)."
        exit 1
    fi
done

run_rl_case() {
    local model_path="$1"
    local experiment_name="$2"
    local resume_mode="$3"
    local wandb_run_id="$4"
    local wandb_resume="$5"
    local wandb_run_name="$6"

    export MODEL_PATH="${model_path}"
    export PROJECT_NAME="${PROJECT_NAME:-string-task}"
    export EXPERIMENT_NAME="${experiment_name}"
    export TRAIN_FILES="${TRAIN_FILES}"
    export VAL_FILES="${VAL_FILES}"
    export NNODES="${NNODES}"
    export N_GPUS_PER_NODE="${N_GPUS_PER_NODE}"
    export SAVE_DIR="${SAVE_ROOT}"
    export PYTHON_BIN="${PYTHON_BIN}"
    if [[ -n "${wandb_run_id}" ]]; then
        export WANDB_RUN_ID="${wandb_run_id}"
    else
        unset WANDB_RUN_ID || true
    fi
    if [[ -n "${wandb_resume}" ]]; then
        export WANDB_RESUME="${wandb_resume}"
    else
        unset WANDB_RESUME || true
    fi
    if [[ -n "${wandb_run_name}" ]]; then
        export WANDB_RUN_NAME="${wandb_run_name}"
    else
        unset WANDB_RUN_NAME || true
    fi

    if [[ -n "${PPO_EXTRA_OVERRIDES}" ]]; then
        # shellcheck disable=SC2086
        bash examples/grpo_trainer/template.sh \
            trainer.resume_mode="${resume_mode}" \
            trainer.save_freq="${SAVE_FREQ}" \
            trainer.test_freq="${TEST_FREQ}" \
            trainer.total_epochs="${TOTAL_EPOCHS}" \
            trainer.remove_previous_ckpt_in_save="${REMOVE_PREVIOUS_CKPT_IN_SAVE}" \
            ${PPO_EXTRA_OVERRIDES} \
            actor_rollout_ref.rollout.name="${ROLLOUT_ENGINE}"
    else
        bash examples/grpo_trainer/template.sh \
            trainer.resume_mode="${resume_mode}" \
            trainer.save_freq="${SAVE_FREQ}" \
            trainer.test_freq="${TEST_FREQ}" \
            trainer.total_epochs="${TOTAL_EPOCHS}" \
            trainer.remove_previous_ckpt_in_save="${REMOVE_PREVIOUS_CKPT_IN_SAVE}" \
            actor_rollout_ref.rollout.name="${ROLLOUT_ENGINE}"
    fi
}

case_enabled() {
    local wanted="$1"
    local cases=",${RUN_CASES},"
    [[ "${cases}" == *",${wanted},"* ]]
}

BASELINE_INIT_MODEL="${BASELINE_INIT_MODEL:-${HF_MODEL_ROOT}/string-task/stage1-rft-hf}"
NEWOPS_INIT_MODEL="${NEWOPS_INIT_MODEL:-${ROOT_DIR}/checkpoints/string-task/stage1-rft-newops}"

if ! case_enabled "baseline" && ! case_enabled "newops"; then
    echo "[run_stage2_rl_newops_compare] RUN_CASES=${RUN_CASES} has no valid case. Use baseline,newops."
    exit 1
fi

if case_enabled "baseline" && [[ ! -d "${BASELINE_INIT_MODEL}" ]]; then
    echo "[run_stage2_rl_newops_compare] Missing baseline init model directory: ${BASELINE_INIT_MODEL}"
    echo "Please place Stage-1 model at this path before starting Stage-2."
    exit 1
fi

if case_enabled "newops" && [[ ! -d "${NEWOPS_INIT_MODEL}" ]]; then
    echo "[run_stage2_rl_newops_compare] Missing newops init model directory: ${NEWOPS_INIT_MODEL}"
    echo "Please place Stage-1 newops model at this path before starting Stage-2."
    exit 1
fi

if case_enabled "baseline"; then
    run_rl_case \
        "${BASELINE_INIT_MODEL}" \
        "${BASELINE_EXPERIMENT_NAME:-stage2-rl-level1to2-baseline-init}" \
        "${BASELINE_RESUME_MODE}" \
        "${BASELINE_WANDB_RUN_ID}" \
        "${BASELINE_WANDB_RESUME}" \
        "${BASELINE_WANDB_RUN_NAME}"
fi

if case_enabled "newops"; then
    run_rl_case \
        "${NEWOPS_INIT_MODEL}" \
        "${NEWOPS_EXPERIMENT_NAME:-stage2-rl-level1to2-newops-init}" \
        "${NEWOPS_RESUME_MODE}" \
        "${NEWOPS_WANDB_RUN_ID}" \
        "${NEWOPS_WANDB_RESUME}" \
        "${NEWOPS_WANDB_RUN_NAME}"
fi

echo "[run_stage2_rl_newops_compare] Done"
echo "- Baseline init: ${BASELINE_INIT_MODEL}"
echo "- Newops init: ${NEWOPS_INIT_MODEL}"
echo "- Baseline resume_mode: ${BASELINE_RESUME_MODE}"
echo "- Newops resume_mode: ${NEWOPS_RESUME_MODE}"
echo "- Run cases: ${RUN_CASES}"
