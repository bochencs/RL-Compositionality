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

# ---------- Stage 2 training & validation datasets ----------
# Stage 2 GRPO needs: stage2_level1/train.parquet, stage2_level2/train.parquet
# (both from examples/data_preprocess/string_data.py, stage=2, split=train).
# Plus stage2_level1to8/forward_test.parquet for eval.
# Only regenerate if missing — these are big (50k rows each) and slow.
STRING_DATA="examples/data_preprocess/string_data.py"

regen_stage2() {
    local path="$1"
    local min_level="$2"
    local max_level="$3"
    local split="$4"
    local n="$5"
    if [ -f "${path}" ]; then
        echo "[prepare] skip (exists): ${path}"
        return 0
    fi
    mkdir -p "$(dirname "${path}")"
    "${PYTHON_BIN}" "${STRING_DATA}" \
        --save_path "${path}" \
        --stage 2 \
        --split "${split}" \
        --min_level "${min_level}" \
        --max_level "${max_level}" \
        --data_num "${n}"
}

regen_stage2 data/string_task/stage2_level1/train.parquet 1 1 train 50000
regen_stage2 data/string_task/stage2_level2/train.parquet 2 2 train 50000
regen_stage2 data/string_task/stage2_level1to2/train.parquet 1 2 train 50000
# forward_test.parquet historically was produced by renaming stage2_level1to8/test.parquet.
# Generate test.parquet then copy to forward_test.parquet so both paths are satisfied.
regen_stage2 data/string_task/stage2_level1to8/test.parquet 1 8 test 2048
if [ ! -f data/string_task/stage2_level1to8/forward_test.parquet ]; then
    cp data/string_task/stage2_level1to8/test.parquet \
       data/string_task/stage2_level1to8/forward_test.parquet
    echo "[prepare] created forward_test.parquet (copy of test.parquet)"
fi

"${PYTHON_BIN}" - <<'PY'
import pandas as pd, os
paths = [
    "data/string_task/exp_new_ops/complete_atomic_new_d1.parquet",
    "data/string_task/exp_new_ops/complete_linear_new_d3.parquet",
    "data/string_task/exp_new_ops/complete_branch_new_d4.parquet",
    "data/string_task/exp_new_ops/complete_mixed_oldnew_d4.parquet",
    "data/string_task/exp_new_ops/stage1_newops_train.parquet",
    "data/string_task/exp_new_ops/stage1_newops_test.parquet",
    "data/string_task/exp_new_ops/stage2_level1to8_newops_test.parquet",
    "data/string_task/stage2_level1/train.parquet",
    "data/string_task/stage2_level2/train.parquet",
    "data/string_task/stage2_level1to2/train.parquet",
    "data/string_task/stage2_level1to8/test.parquet",
    "data/string_task/stage2_level1to8/forward_test.parquet",
]
for p in paths:
    if os.path.exists(p):
        df = pd.read_parquet(p)
        print(f"  {p:<70} {len(df):>8} rows")
    else:
        print(f"  {p:<70}    MISSING")
PY

echo "[prepare_newops_datasets] Done"
