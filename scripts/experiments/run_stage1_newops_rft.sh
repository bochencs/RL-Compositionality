#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-rlcomp/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${ROOT_DIR}/.venv-rlcomp/bin/torchrun}"

MODEL_STORE_ROOT="${MODEL_STORE_ROOT:-/home/ma-user/work/model_store}"
HF_MODEL_ROOT="${HF_MODEL_ROOT:-${MODEL_STORE_ROOT}/hf_models}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${MODEL_STORE_ROOT}/hf_cache}"
USER_CKPT_ROOT="${USER_CKPT_ROOT:-${ROOT_DIR}/checkpoints}"
mkdir -p "${HF_MODEL_ROOT}" "${HF_CACHE_ROOT}" "${USER_CKPT_ROOT}"
export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_ROOT}/transformers}"

INIT_MODEL_PATH="${INIT_MODEL_PATH:-${HF_MODEL_ROOT}/string-task/stage1-rft-hf}"
DATA_PATH="${DATA_PATH:-data/string_task/exp_new_ops/stage1_newops_train.parquet}"
ROLLOUT_PATH="${ROLLOUT_PATH:-data/string_task/exp_new_ops/rollout_stage1_newops.parquet}"
RFT_DATA_SAVE_PATH="${RFT_DATA_SAVE_PATH:-data/string_task/exp_new_ops/rft_data}"

N_SAMPLES="${N_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
PROMPT_LENGTH="${PROMPT_LENGTH:-1024}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-512}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
NNODES="${NNODES:-1}"
VAL_SIZE="${VAL_SIZE:-512}"
MAX_CORRECT_RATIO="${MAX_CORRECT_RATIO:-1.0}"

mkdir -p "$(dirname "${ROLLOUT_PATH}")" "${RFT_DATA_SAVE_PATH}"

# 1) Rollout collection on new-ops stage1 data.
DATA_PATH="${DATA_PATH}" \
SAVE_PATH="${ROLLOUT_PATH}" \
MODEL_PATH="${INIT_MODEL_PATH}" \
N_SAMPLES="${N_SAMPLES}" \
TEMPERATURE="${TEMPERATURE}" \
PROMPT_LENGTH="${PROMPT_LENGTH}" \
RESPONSE_LENGTH="${RESPONSE_LENGTH}" \
N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
NNODES="${NNODES}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash examples/generation/run_string.sh

# 2) Filter rollout into SFT/RFT data.
"${PYTHON_BIN}" examples/data_preprocess/string_manipulation_sft.py \
    --gen_path "${ROLLOUT_PATH}" \
    --data_path "${DATA_PATH}" \
    --save_path "${RFT_DATA_SAVE_PATH}" \
    --val_size "${VAL_SIZE}" \
    --max_correct_ratio "${MAX_CORRECT_RATIO}" \
    --no_remove_context

# 3) Incremental Stage1 RFT (SFT trainer).
export MODEL_PATH="${INIT_MODEL_PATH}"
export PROJECT_NAME="${PROJECT_NAME:-string-task}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-stage1-rft-newops}"
export TRAIN_FILES="${TRAIN_FILES:-${RFT_DATA_SAVE_PATH}/train.parquet}"
export VAL_FILES="${VAL_FILES:-${RFT_DATA_SAVE_PATH}/test.parquet}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export MAX_LENGTH="${MAX_LENGTH:-3072}"
export SP_SIZE="${SP_SIZE:-1}"
export EPOCHS="${EPOCHS:-1}"
export SAVE_DIR="${SAVE_DIR:-${USER_CKPT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NNODES="${NNODES}"
export TORCHRUN_BIN="${TORCHRUN_BIN}"
export LOGGER="${LOGGER:-['console','wandb']}"

# Memory-scaling overrides: bootstrap sets these based on GPU count so an 8B SFT
# can fit on 1 A800-80GB (needs cpu_offload + gradient_checkpointing). On 4+ GPUs
# the defaults stay fast (no offload). They are hydra overrides, passed to the
# template via $@. Override individually if you want tighter control.
SFT_CPU_OFFLOAD="${SFT_CPU_OFFLOAD:-${CPU_OFFLOAD:-False}}"
SFT_OFFLOAD_PARAMS="${SFT_OFFLOAD_PARAMS:-${OFFLOAD_PARAMS:-False}}"
SFT_GRAD_CKPT="${SFT_GRAD_CKPT:-${GRAD_CKPT:-False}}"
SFT_MODEL_DTYPE="${SFT_MODEL_DTYPE:-${MODEL_DTYPE:-fp32}}"

bash examples/sft/template.sh \
    "model.fsdp_config.cpu_offload=${SFT_CPU_OFFLOAD}" \
    "model.fsdp_config.offload_params=${SFT_OFFLOAD_PARAMS}" \
    "model.enable_gradient_checkpointing=${SFT_GRAD_CKPT}" \
    "+model.dtype=${SFT_MODEL_DTYPE}"

echo "[run_stage1_newops_rft] Done"
echo "- Rollout: ${ROLLOUT_PATH}"
echo "- RFT data: ${RFT_DATA_SAVE_PATH}"
echo "- Output dir: ${SAVE_DIR}"
