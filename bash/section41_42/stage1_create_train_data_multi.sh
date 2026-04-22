set -euo pipefail

export NNODES=${NNODES:-8}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export DATA_PATH=data/string_task/stage1_level1/train.parquet
export N_SAMPLES=10
export SAVE_PATH=data/string_task/stage1_level1/rollout.parquet
export MODEL_PATH=../config-file/model/Llama-3.1-8B-Instruct
export TEMPERATURE=1.0
export PROMPT_LENGTH=1024
export RESPONSE_LENGTH=8192
export RFT_DATA_SAVE_PATH=data/string_task/stage1_level1/rft_data
export ROLLOUT_DATA_PATH=${ROLLOUT_DATA_PATH:-data/string_task/stage1_level1/train_rollout_prompted.parquet}
export ROLLOUT_PROMPT_HINT=${ROLLOUT_PROMPT_HINT:-First consider the logic of the Python code, then predict the output.}

if [ "${NNODES}" -gt 1 ]; then
    export RAY_ADDRESS=${RAY_ADDRESS:-auto}
    echo "[stage1_create_train_data_multi] NNODES=${NNODES}, expecting an existing Ray cluster. RAY_ADDRESS=${RAY_ADDRESS}"
    if ! ray status >/dev/null 2>&1; then
        echo "[stage1_create_train_data_multi] ERROR: Ray cluster is not reachable."
        echo "[stage1_create_train_data_multi] Start Ray first, e.g.:"
        echo "  head node:   ray start --head --port 6379 --num-gpus ${N_GPUS_PER_NODE}"
        echo "  worker node: ray start --address <HEAD_IP>:6379 --num-gpus ${N_GPUS_PER_NODE}"
        exit 1
    fi
fi

python3 - <<'PY'
import copy
import os
from datasets import load_dataset

src = os.environ["DATA_PATH"]
dst = os.environ["ROLLOUT_DATA_PATH"]
hint = os.environ["ROLLOUT_PROMPT_HINT"].strip()

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
os.makedirs(os.path.dirname(dst), exist_ok=True)
dataset.to_parquet(dst)
print(f"[stage1_create_train_data] rollout prompt dataset written to: {dst}")
PY

DATA_PATH=${ROLLOUT_DATA_PATH} NNODES=${NNODES} N_GPUS_PER_NODE=${N_GPUS_PER_NODE} bash examples/generation/run_string.sh

python3 examples/data_preprocess/string_manipulation_sft.py \
    --gen_path ${SAVE_PATH} \
    --data_path ${DATA_PATH} \
    --save_path ${RFT_DATA_SAVE_PATH} \
    --val_size 256 \
    --max_correct_ratio 1.0 \
    # --no_remove_context
