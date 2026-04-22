#!/usr/bin/env bash
set -euo pipefail
set -x

export NNODES="${NNODES:-8}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
export MODEL_PATH="${MODEL_PATH:-../config-file/model/Llama-3.1-8B-Instruct}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export PROMPT_LENGTH="${PROMPT_LENGTH:-1024}"
export RESPONSE_LENGTH="${RESPONSE_LENGTH:-8192}"
export ROLLOUT_PROMPT_HINT="${ROLLOUT_PROMPT_HINT:-First consider the logic of the Python code, then predict the output.}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BASE_DIR="${BASE_DIR:-data/string_task/new_rft_gen/model2}"
PROBLEM_DIR="${BASE_DIR}/problems"
ROLLOUT_DIR="${BASE_DIR}/rollout"
RFT_COMPONENT_DIR="${BASE_DIR}/rft_components"
RFT_DATA_DIR="${BASE_DIR}/rft_data"

mkdir -p "${ROLLOUT_DIR}" "${RFT_COMPONENT_DIR}" "${RFT_DATA_DIR}"

ensure_ray_cluster_if_needed

collect_rft_component_multi \
    "${PROBLEM_DIR}/atomic_train.parquet" \
    "${ROLLOUT_DIR}/atomic_train_prompted.parquet" \
    "${ROLLOUT_DIR}/atomic_rollout.parquet" \
    "${RFT_COMPONENT_DIR}/atomic" \
    "${ATOMIC_N_SAMPLES:-10}"

collect_rft_component_multi \
    "${PROBLEM_DIR}/branch1_depth1_train.parquet" \
    "${ROLLOUT_DIR}/branch1_depth1_train_prompted.parquet" \
    "${ROLLOUT_DIR}/branch1_depth1_rollout.parquet" \
    "${RFT_COMPONENT_DIR}/branch1_depth1" \
    "${NEW_N_SAMPLES:-30}"

collect_rft_component_multi \
    "${PROBLEM_DIR}/branch2_depth1_train.parquet" \
    "${ROLLOUT_DIR}/branch2_depth1_train_prompted.parquet" \
    "${ROLLOUT_DIR}/branch2_depth1_rollout.parquet" \
    "${RFT_COMPONENT_DIR}/branch2_depth1" \
    "${NEW_N_SAMPLES:-30}"

merge_parquet \
    "${RFT_DATA_DIR}/train.parquet" \
    "${RFT_COMPONENT_DIR}/atomic/train.parquet" \
    "${RFT_COMPONENT_DIR}/branch1_depth1/train.parquet" \
    "${RFT_COMPONENT_DIR}/branch2_depth1/train.parquet"

merge_parquet \
    "${RFT_DATA_DIR}/test.parquet" \
    "${RFT_COMPONENT_DIR}/atomic/test.parquet" \
    "${RFT_COMPONENT_DIR}/branch1_depth1/test.parquet" \
    "${RFT_COMPONENT_DIR}/branch2_depth1/test.parquet"

write_rft_mix_stats \
    "${RFT_DATA_DIR}/train.parquet" \
    "${RFT_DATA_DIR}/test.parquet" \
    "${RFT_DATA_DIR}/mix_stats.csv"

echo "[model2_create_train_data_multi] done"
echo "- final rft dir: ${RFT_DATA_DIR}"
