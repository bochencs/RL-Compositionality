#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

export MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/RL-Compositionality/checkpoints/string-task/stage1-rft-with-hint/global_step_1648}"
export PROJECT_NAME="${PROJECT_NAME:-string-task}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-test-llama3-base__dag-b1-d1-only}"
export TRAIN_FILES="${TRAIN_FILES:-['data/string_task/new_rl_gen/dag_b1_d1_only/forward_train.parquet']}"
export VAL_FILES="${VAL_FILES:-$(dag_eval_val_files_literal)}"
export SAVE_DIR="${SAVE_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

bash examples/grpo_trainer/template.sh
