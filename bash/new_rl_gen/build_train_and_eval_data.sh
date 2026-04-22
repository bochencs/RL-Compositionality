#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DAG_B1_DIR="${DATA_ROOT}/dag_b1_d1_only"
DAG_B2_DIR="${DATA_ROOT}/dag_b2_d1_only"
MIXED_DIR="${DATA_ROOT}/dag_b1_d1_plus_level2"
DAG_EVAL_DIR="${DATA_ROOT}/dag_eval"

DAG_HALF=$(( MIXED_TOTAL / 2 ))
LEVEL2_HALF=$(( MIXED_TOTAL - DAG_HALF ))

mkdir -p "${DAG_B1_DIR}" "${DAG_B2_DIR}" "${MIXED_DIR}" "${DAG_EVAL_DIR}"

generate_explicit_stage2_dataset "${DAG_B1_DIR}/forward_train.parquet" train 1 1 "${DAG_TRAIN_TOTAL}"
generate_explicit_stage2_dataset "${DAG_B2_DIR}/forward_train.parquet" train 2 1 "${DAG_TRAIN_TOTAL}"

generate_explicit_stage2_dataset "${MIXED_DIR}/dag_branch1_depth1_half.parquet" train 1 1 "${DAG_HALF}"
sample_parquet_rows "${LEGACY_LEVEL2_SOURCE}" "${MIXED_DIR}/legacy_level2_half.parquet" "${LEVEL2_HALF}"
merge_parquet \
    "${MIXED_DIR}/forward_train.parquet" \
    "${MIXED_DIR}/dag_branch1_depth1_half.parquet" \
    "${MIXED_DIR}/legacy_level2_half.parquet"

for branch in 1 2 3; do
    for depth in 1 2 3; do
        generate_explicit_stage2_dataset \
            "${DAG_EVAL_DIR}/forward_b${branch}_d${depth}.parquet" \
            test \
            "${branch}" \
            "${depth}" \
            "${EVAL_PER_GROUP}"
    done
done

echo "[build_train_and_eval_data] done"
echo "- data root: ${DATA_ROOT}"
echo "- dag train total: ${DAG_TRAIN_TOTAL}"
echo "- mixed total: ${MIXED_TOTAL}"
echo "- eval per group: ${EVAL_PER_GROUP}"
