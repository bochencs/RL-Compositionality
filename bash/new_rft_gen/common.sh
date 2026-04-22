#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/rl_comp/bin/python}"
MODEL_PATH="${MODEL_PATH:-../config-file/model/Llama-3.1-8B-Instruct}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
PROMPT_LENGTH="${PROMPT_LENGTH:-1024}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-8192}"
ROLLOUT_PROMPT_HINT="${ROLLOUT_PROMPT_HINT:-First consider the logic of the Python code, then predict the output.}"
RFT_VAL_SIZE="${RFT_VAL_SIZE:-128}"

atomic_count() {
    local total="$1"
    echo $(( total * 85 / 100 ))
}

new_count() {
    local total="$1"
    local atomic
    atomic="$(atomic_count "${total}")"
    echo $(( total - atomic ))
}

merge_parquet() {
    local output_path="$1"
    shift
    mkdir -p "$(dirname "${output_path}")"
    "${PYTHON_BIN}" scripts/merge_sft_data.py --data "$@" --output-path "${output_path}"
}

write_rft_mix_stats() {
    local train_path="$1"
    local test_path="$2"
    local output_path="$3"
    mkdir -p "$(dirname "${output_path}")"
    "${PYTHON_BIN}" scripts/report_rft_mix_stats.py \
        --train "${train_path}" \
        --test "${test_path}" \
        --output "${output_path}"
}

ensure_ray_cluster_if_needed() {
    if [ "${NNODES}" -gt 1 ]; then
        export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
        echo "[new_rft_gen] NNODES=${NNODES}, expecting an existing Ray cluster. RAY_ADDRESS=${RAY_ADDRESS}"
        if ! ray status >/dev/null 2>&1; then
            echo "[new_rft_gen] ERROR: Ray cluster is not reachable."
            echo "[new_rft_gen] Start Ray first, e.g.:"
            echo "  head node:   ray start --head --port 6379 --num-gpus ${N_GPUS_PER_NODE}"
            echo "  worker node: ray start --address <HEAD_IP>:6379 --num-gpus ${N_GPUS_PER_NODE}"
            exit 1
        fi
    fi
}

add_rollout_hint_dataset() {
    local src="$1"
    local dst="$2"
    mkdir -p "$(dirname "${dst}")"

    SRC="${src}" DST="${dst}" HINT="${ROLLOUT_PROMPT_HINT}" "${PYTHON_BIN}" - <<'PY'
import copy
import os
from datasets import load_dataset

src = os.environ["SRC"]
dst = os.environ["DST"]
hint = os.environ["HINT"].strip()

dataset = load_dataset("parquet", data_files=src)["train"]

def add_rollout_hint(example):
    prompt = example.get("prompt")
    if not hint:
        return example
    if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
        content = prompt[0].get("content", "")
        if hint not in content:
            new_prompt = copy.deepcopy(prompt)
            new_prompt[0]["content"] = f"{content}\n\n{hint}"
            example["prompt"] = new_prompt
    return example

dataset = dataset.map(add_rollout_hint, num_proc=4)
dataset.to_parquet(dst)
print(dst)
PY
}

generate_atomic_stage1_dataset() {
    local save_path="$1"
    local dataset_split="$2"
    local data_num="$3"
    mkdir -p "$(dirname "${save_path}")"
    "${PYTHON_BIN}" examples/data_preprocess/string_data.py \
        --save_path "${save_path}" \
        --stage 1 \
        --split "${dataset_split}" \
        --min_level 1 \
        --max_level 1 \
        --data_num "${data_num}"
}

generate_explicit_stage1_dataset() {
    local save_path="$1"
    local dataset_split="$2"
    local num_branch="$3"
    local depth="$4"
    local data_num="$5"
    mkdir -p "$(dirname "${save_path}")"
    "${PYTHON_BIN}" scripts/experiments/gen_string_comp_explicit_dag_dataset.py \
        --save_path "${save_path}" \
        --stage 1 \
        --dataset_split "${dataset_split}" \
        --num_branch "${num_branch}" \
        --depth "${depth}" \
        --data_num "${data_num}"
}

collect_rft_component() {
    local data_path="$1"
    local rollout_prompted_path="$2"
    local rollout_path="$3"
    local rft_dir="$4"
    local n_samples="$5"

    add_rollout_hint_dataset "${data_path}" "${rollout_prompted_path}"

    DATA_PATH="${rollout_prompted_path}" \
    SAVE_PATH="${rollout_path}" \
    MODEL_PATH="${MODEL_PATH}" \
    N_SAMPLES="${n_samples}" \
    TEMPERATURE="${TEMPERATURE}" \
    PROMPT_LENGTH="${PROMPT_LENGTH}" \
    RESPONSE_LENGTH="${RESPONSE_LENGTH}" \
    NNODES="${NNODES}" \
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash examples/generation/run_string.sh

    mkdir -p "${rft_dir}"
    "${PYTHON_BIN}" examples/data_preprocess/string_manipulation_sft.py \
        --gen_path "${rollout_path}" \
        --data_path "${data_path}" \
        --save_path "${rft_dir}" \
        --val_size "${RFT_VAL_SIZE}" \
        --max_correct_ratio 1.0 \
        --no_remove_context
}

collect_rft_component_multi() {
    local raw_data_path="$1"
    local rollout_prompted_path="$2"
    local rollout_path="$3"
    local rft_dir="$4"
    local n_samples="$5"

    export DATA_PATH="${raw_data_path}"
    export N_SAMPLES="${n_samples}"
    export SAVE_PATH="${rollout_path}"
    export RFT_DATA_SAVE_PATH="${rft_dir}"
    export ROLLOUT_DATA_PATH="${rollout_prompted_path}"
    export ROLLOUT_PROMPT_HINT="${ROLLOUT_PROMPT_HINT}"

    add_rollout_hint_dataset "${DATA_PATH}" "${ROLLOUT_DATA_PATH}"

    DATA_PATH="${ROLLOUT_DATA_PATH}" \
    NNODES="${NNODES}" \
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
    MODEL_PATH="${MODEL_PATH}" \
    TEMPERATURE="${TEMPERATURE}" \
    PROMPT_LENGTH="${PROMPT_LENGTH}" \
    RESPONSE_LENGTH="${RESPONSE_LENGTH}" \
    N_SAMPLES="${N_SAMPLES}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash examples/generation/run_string.sh

    mkdir -p "${RFT_DATA_SAVE_PATH}"
    "${PYTHON_BIN}" examples/data_preprocess/string_manipulation_sft.py \
        --gen_path "${SAVE_PATH}" \
        --data_path "${DATA_PATH}" \
        --save_path "${RFT_DATA_SAVE_PATH}" \
        --val_size "${RFT_VAL_SIZE}" \
        --max_correct_ratio 1.0 \
        --no_remove_context
}
