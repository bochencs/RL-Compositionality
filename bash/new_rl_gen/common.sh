#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/rl_comp/bin/python}"
DATA_ROOT="${DATA_ROOT:-data/string_task/new_rl_gen}"
LEGACY_LEVEL2_SOURCE="${LEGACY_LEVEL2_SOURCE:-data/string_task/stage2_level2/forward_train.parquet}"
LINEAR_EVAL_FILE="${LINEAR_EVAL_FILE:-data/string_task/stage2_level1to8/forward_test.parquet}"
DAG_TRAIN_TOTAL="${DAG_TRAIN_TOTAL:-500000}"
MIXED_TOTAL="${MIXED_TOTAL:-500000}"
EVAL_PER_GROUP="${EVAL_PER_GROUP:-256}"
SEED="${SEED:-42}"

generate_explicit_stage2_dataset() {
    local save_path="$1"
    local dataset_split="$2"
    local num_branch="$3"
    local depth="$4"
    local data_num="$5"
    mkdir -p "$(dirname "${save_path}")"
    "${PYTHON_BIN}" scripts/experiments/gen_string_comp_explicit_dag_dataset.py \
        --save_path "${save_path}" \
        --stage 2 \
        --dataset_split "${dataset_split}" \
        --num_branch "${num_branch}" \
        --depth "${depth}" \
        --data_num "${data_num}" \
        --seed "${SEED}"
}

sample_parquet_rows() {
    local input_path="$1"
    local output_path="$2"
    local num_rows="$3"
    mkdir -p "$(dirname "${output_path}")"
    "${PYTHON_BIN}" scripts/sample_parquet_rows.py \
        --input "${input_path}" \
        --output "${output_path}" \
        --num_rows "${num_rows}" \
        --seed "${SEED}"
}

merge_parquet() {
    local output_path="$1"
    shift
    mkdir -p "$(dirname "${output_path}")"
    "${PYTHON_BIN}" scripts/merge_sft_data.py --data "$@" --output-path "${output_path}"
}

resolve_checkpoint_dir() {
    local checkpoint_path="$1"

    if [ -f "${checkpoint_path}/config.json" ]; then
        echo "${checkpoint_path}"
        return 0
    fi

    if [ -d "${checkpoint_path}" ]; then
        local latest
        latest="$(
            find "${checkpoint_path}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' \
                | sed 's#/$##' \
                | awk -F'global_step_' '{print $2 " " $0}' \
                | sort -n \
                | tail -n 1 \
                | cut -d' ' -f2-
        )"
        if [ -n "${latest}" ] && [ -f "${latest}/config.json" ]; then
            echo "${latest}"
            return 0
        fi
    fi

    echo "${checkpoint_path}"
}

dag_eval_val_files_literal() {
    printf "['%s','data/string_task/new_rl_gen/dag_eval/forward_b1_d1.parquet','data/string_task/new_rl_gen/dag_eval/forward_b1_d2.parquet','data/string_task/new_rl_gen/dag_eval/forward_b1_d3.parquet','data/string_task/new_rl_gen/dag_eval/forward_b2_d1.parquet','data/string_task/new_rl_gen/dag_eval/forward_b2_d2.parquet','data/string_task/new_rl_gen/dag_eval/forward_b2_d3.parquet','data/string_task/new_rl_gen/dag_eval/forward_b3_d1.parquet','data/string_task/new_rl_gen/dag_eval/forward_b3_d2.parquet','data/string_task/new_rl_gen/dag_eval/forward_b3_d3.parquet']" "${LINEAR_EVAL_FILE}"
}
