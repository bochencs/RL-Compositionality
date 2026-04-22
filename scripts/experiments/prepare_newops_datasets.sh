#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-rlcomp/bin/python}"
GEN_SCRIPT="scripts/experiments/gen_string_comp_dataset.py"

run_gen() {
    local save_path="$1"
    local stage="$2"
    local min_level="$3"
    local max_level="$4"
    local data_num="$5"
    local seed="$6"
    local allowed_funcs="$7"
    local binary_prob="$8"

    "${PYTHON_BIN}" "${GEN_SCRIPT}" \
        --save_path "${save_path}" \
        --stage "${stage}" \
        --min_level "${min_level}" \
        --max_level "${max_level}" \
        --data_num "${data_num}" \
        --seed "${seed}" \
        --allowed_funcs "${allowed_funcs}" \
        --binary_prob "${binary_prob}" \
        --allow_constants 0 \
        --allow_builtin_methods 0
}

mkdir -p data/string_task/exp_new_ops

# A3 datasets
run_gen \
    data/string_task/exp_new_ops/complete_atomic_new_d1.parquet \
    1 1 1 512 42 \
    "run_length_encode,sort_by_frequency,checksum_rotate" \
    0

run_gen \
    data/string_task/exp_new_ops/complete_linear_new_d3.parquet \
    1 3 3 1024 43 \
    "run_length_encode,sort_by_frequency,checksum_rotate" \
    0

run_gen \
    data/string_task/exp_new_ops/complete_branch_new_d4.parquet \
    1 4 4 1024 44 \
    "run_length_encode,sort_by_frequency,checksum_rotate,interlace_str" \
    0.6

run_gen \
    data/string_task/exp_new_ops/complete_mixed_oldnew_d4.parquet \
    1 4 4 2048 45 \
    "run_length_encode,sort_by_frequency,checksum_rotate,rotate_str,duplicate_every_char,compress_repeats" \
    0.6

# B1 datasets
run_gen \
    data/string_task/exp_new_ops/stage1_newops_train.parquet \
    1 1 1 50000 100 \
    "run_length_encode,sort_by_frequency,checksum_rotate" \
    0

run_gen \
    data/string_task/exp_new_ops/stage1_newops_test.parquet \
    1 1 1 512 101 \
    "run_length_encode,sort_by_frequency,checksum_rotate" \
    0

# B2 evaluation dataset (stage=2 style)
run_gen \
    data/string_task/exp_new_ops/stage2_level1to8_newops_test.parquet \
    2 1 8 2048 102 \
    "deterministic_shuffle,remove_vowels,add_suffix,interlace_str,rotate_str,alternate_case,vowel_to_number,duplicate_every_char,compress_repeats,loop_concat,loop_filter_nonalpha,backchain_palindrome,run_length_encode,sort_by_frequency,checksum_rotate" \
    0.2

"${PYTHON_BIN}" - <<'PY'
import pandas as pd
paths = [
    "data/string_task/exp_new_ops/complete_atomic_new_d1.parquet",
    "data/string_task/exp_new_ops/complete_linear_new_d3.parquet",
    "data/string_task/exp_new_ops/complete_branch_new_d4.parquet",
    "data/string_task/exp_new_ops/complete_mixed_oldnew_d4.parquet",
    "data/string_task/exp_new_ops/stage1_newops_train.parquet",
    "data/string_task/exp_new_ops/stage1_newops_test.parquet",
    "data/string_task/exp_new_ops/stage2_level1to8_newops_test.parquet",
]
for p in paths:
    df = pd.read_parquet(p)
    print(p, len(df))
PY

echo "[prepare_newops_datasets] Done"
