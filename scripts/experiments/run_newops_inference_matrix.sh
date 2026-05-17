#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_WRAPPER="scripts/experiments/run_inference_and_record.sh"
if [[ ! -x "${RUN_WRAPPER}" ]]; then
    echo "Missing executable: ${RUN_WRAPPER}"
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_PREFIX="${RUN_PREFIX:-${TS}_newops_matrix}"
export WANDB_GROUP="${WANDB_GROUP:-${RUN_PREFIX}}"

run_case() {
    local suffix="$1"
    local data_path="$2"
    local n_samples="$3"
    local temperature="$4"
    local k="$5"
    local notes="$6"

    local run_id="${RUN_PREFIX}_${suffix}"
    bash "${RUN_WRAPPER}" \
        "${run_id}" \
        "${data_path}" \
        AUTO \
        "${n_samples}" \
        "${temperature}" \
        "${k}" \
        "${notes}"
}

run_case \
    A_atomic_d1_n1_t0 \
    data/string_task/exp_new_ops/complete_atomic_new_d1.parquet \
    1 0 1 \
    "A: atomic-new deterministic"

run_case \
    A_atomic_d1_n32_t1 \
    data/string_task/exp_new_ops/complete_atomic_new_d1.parquet \
    32 1.0 32 \
    "A: atomic-new pass@32"

run_case \
    B_linear_d3_n1_t0 \
    data/string_task/exp_new_ops/complete_linear_new_d3.parquet \
    1 0 1 \
    "B: linear-compose-new deterministic"

run_case \
    B_linear_d3_n32_t1 \
    data/string_task/exp_new_ops/complete_linear_new_d3.parquet \
    32 1.0 32 \
    "B: linear-compose-new pass@32"

run_case \
    C_branch_d4_n1_t0 \
    data/string_task/exp_new_ops/complete_branch_new_d4.parquet \
    1 0 1 \
    "C: branching-compose-new deterministic"

run_case \
    C_branch_d4_n32_t1 \
    data/string_task/exp_new_ops/complete_branch_new_d4.parquet \
    32 1.0 32 \
    "C: branching-compose-new pass@32"

run_case \
    D_mixed_d4_n1_t0 \
    data/string_task/exp_new_ops/complete_mixed_oldnew_d4.parquet \
    1 0 1 \
    "D: mixed old+new deterministic"

run_case \
    D_mixed_d4_n32_t1 \
    data/string_task/exp_new_ops/complete_mixed_oldnew_d4.parquet \
    32 1.0 32 \
    "D: mixed old+new pass@32"

echo "[run_newops_inference_matrix] Done. run prefix: ${RUN_PREFIX}"
