#!/usr/bin/env bash
set -euo pipefail

export PROJECT_NAME="${PROJECT_NAME:-string-task}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-fit-tree-template}"
export MODEL_PATH="${MODEL_PATH:-../config-file/model/Llama-3.1-8B-Instruct}"
export NNODES="${NNODES:-${VC_WORKER_NUM:-1}}"
export SP_SIZE="${SP_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-3072}"
export TRAIN_FILES="${TRAIN_FILES:-data/string_task/new_rft_gen/model1/rft_data/train.parquet}"
export VAL_FILES="${VAL_FILES:-data/string_task/new_rft_gen/model1/rft_data/test.parquet}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export EPOCHS="${EPOCHS:-2}"
export SAVE_DIR="${SAVE_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

bash examples/sft/template.sh \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.default_hdfs_dir=null
