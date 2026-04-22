#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TRAIN_TOTAL="${TRAIN_TOTAL:-50000}"
TEST_TOTAL="${TEST_TOTAL:-128}"
BASE_DIR="${BASE_DIR:-data/string_task/new_rft_gen/model2}"
PROBLEM_DIR="${BASE_DIR}/problems"

ATOMIC_TRAIN="$(atomic_count "${TRAIN_TOTAL}")"
NEW_TRAIN="$(new_count "${TRAIN_TOTAL}")"
ATOMIC_TEST="$(atomic_count "${TEST_TOTAL}")"
NEW_TEST="$(new_count "${TEST_TOTAL}")"

BRANCH1_TRAIN=$(( NEW_TRAIN / 2 ))
BRANCH2_TRAIN=$(( NEW_TRAIN - BRANCH1_TRAIN ))
BRANCH1_TEST=$(( NEW_TEST / 2 ))
BRANCH2_TEST=$(( NEW_TEST - BRANCH1_TEST ))

mkdir -p "${PROBLEM_DIR}"

generate_atomic_stage1_dataset "${PROBLEM_DIR}/atomic_train.parquet" train "${ATOMIC_TRAIN}"
generate_atomic_stage1_dataset "${PROBLEM_DIR}/atomic_test.parquet" test "${ATOMIC_TEST}"

generate_explicit_stage1_dataset "${PROBLEM_DIR}/branch1_depth1_train.parquet" train 1 1 "${BRANCH1_TRAIN}"
generate_explicit_stage1_dataset "${PROBLEM_DIR}/branch1_depth1_test.parquet" test 1 1 "${BRANCH1_TEST}"

generate_explicit_stage1_dataset "${PROBLEM_DIR}/branch2_depth1_train.parquet" train 2 1 "${BRANCH2_TRAIN}"
generate_explicit_stage1_dataset "${PROBLEM_DIR}/branch2_depth1_test.parquet" test 2 1 "${BRANCH2_TEST}"

merge_parquet \
    "${PROBLEM_DIR}/mixed_train.parquet" \
    "${PROBLEM_DIR}/atomic_train.parquet" \
    "${PROBLEM_DIR}/branch1_depth1_train.parquet" \
    "${PROBLEM_DIR}/branch2_depth1_train.parquet"

merge_parquet \
    "${PROBLEM_DIR}/mixed_test.parquet" \
    "${PROBLEM_DIR}/atomic_test.parquet" \
    "${PROBLEM_DIR}/branch1_depth1_test.parquet" \
    "${PROBLEM_DIR}/branch2_depth1_test.parquet"

echo "[model2_create_problems] done"
echo "- atomic train/test: ${ATOMIC_TRAIN}/${ATOMIC_TEST}"
echo "- branch1 train/test: ${BRANCH1_TRAIN}/${BRANCH1_TEST}"
echo "- branch2 train/test: ${BRANCH2_TRAIN}/${BRANCH2_TEST}"
echo "- output dir: ${PROBLEM_DIR}"
