#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Keep temp files and Ray session artifacts off root disk by default.
TMP_WORK_DIR="${TMP_WORK_DIR:-${ROOT_DIR}/.tmp}"
export TMPDIR="${TMPDIR:-${TMP_WORK_DIR}}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
# Keep Ray socket path short to avoid AF_UNIX (107-byte) path-length failures.
_RAY_CAND="${ROOT_DIR}/.raytmp"
if [ "${#_RAY_CAND}" -lt 100 ]; then
    export RAY_TMPDIR="${RAY_TMPDIR:-${_RAY_CAND}}"
else
    export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/.raytmp.$(id -u 2>/dev/null || echo 0)}"
fi
unset _RAY_CAND
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

MODEL_STORE_ROOT="${MODEL_STORE_ROOT:-${ROOT_DIR}/model_store}"
HF_MODEL_ROOT="${HF_MODEL_ROOT:-${MODEL_STORE_ROOT}/hf_models}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${MODEL_STORE_ROOT}/hf_cache}"
mkdir -p "${HF_MODEL_ROOT}" "${HF_CACHE_ROOT}"
export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_ROOT}/transformers}"

if [[ "$#" -lt 5 || "$#" -gt 7 ]]; then
    echo "Usage: $0 RUN_ID DATA_PATH SAVE_PATH N_SAMPLES TEMPERATURE [K] [NOTES]"
    exit 1
fi

RUN_ID="$1"
DATA_PATH="$2"
SAVE_PATH="$3"
N_SAMPLES="$4"
TEMPERATURE="$5"
K="${6:-32}"
NOTES="${7:-${NOTES:-}}"

RUN_DIR="results/exp_rlcompose/${RUN_ID}"
ANALYSIS_DIR="${RUN_DIR}/analysis"

if [[ "${SAVE_PATH}" == "AUTO" ]]; then
    SAVE_PATH="${RUN_DIR}/pred.parquet"
fi

mkdir -p "${RUN_DIR}" "${ANALYSIS_DIR}" "$(dirname "${SAVE_PATH}")"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-rlcomp/bin/python}"
MODEL_PATH="${MODEL_PATH:-${HF_MODEL_ROOT}/string-task/stage1-rft-hf}"
PROMPT_LENGTH="${PROMPT_LENGTH:-1024}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-512}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
NNODES="${NNODES:-1}"

export WANDB_ENABLE="${WANDB_ENABLE:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-rlcompose-new-abilities}"
export WANDB_GROUP="${WANDB_GROUP:-exp-rlcompose-${RUN_ID%%_*}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_ID}}"
export WANDB_TAGS="${WANDB_TAGS:-stage1,inference,new-abilities,composition}"

if [[ "${USE_LOCAL_CLASH_PROXY:-0}" == "1" ]]; then
    export http_proxy="http://127.0.0.1:7897"
    export https_proxy="http://127.0.0.1:7897"
    export HTTP_PROXY="${http_proxy}"
    export HTTPS_PROXY="${https_proxy}"
fi

START_TIME_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"

DATA_PATH="${DATA_PATH}" \
SAVE_PATH="${SAVE_PATH}" \
N_SAMPLES="${N_SAMPLES}" \
TEMPERATURE="${TEMPERATURE}" \
PROMPT_LENGTH="${PROMPT_LENGTH}" \
RESPONSE_LENGTH="${RESPONSE_LENGTH}" \
N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
NNODES="${NNODES}" \
PYTHON_BIN="${PYTHON_BIN}" \
MODEL_PATH="${MODEL_PATH}" \
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}" \
bash examples/generation/run_string.sh

# Normalize responses column to Python list for downstream consistency.
export SAVE_PATH PYTHON_BIN
"${PYTHON_BIN}" - <<'PY'
import pandas as pd
import numpy as np
import os

path = os.environ["SAVE_PATH"]
df = pd.read_parquet(path)

if "responses" in df.columns:
    def normalize(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return list(value)
        return value

    df["responses"] = df["responses"].apply(normalize)
    df.to_parquet(path, index=False)
PY

"${PYTHON_BIN}" scripts/experiments/analyze_stage1_generalization.py \
    --pred_parquet "${SAVE_PATH}" \
    --out_csv "${ANALYSIS_DIR}/summary_by_depth.csv" \
    --k "${K}"

END_TIME_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
END_EPOCH="$(date +%s)"
DURATION_SEC="$((END_EPOCH - START_EPOCH))"

# `git rev-parse HEAD` may fail under set -e on shared-FS mounts where git
# emits "fatal: detected dubious ownership". All eval work is already done
# at this point — don't take the whole script down for a metadata field.
# Without this, dag.sh sees rc != 0 and incorrectly logs "[ERROR] failed: ..."
# even though summary_overall.json is on disk and valid.
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
HOSTNAME_VALUE="$(hostname)"
GPU_INFO="$(nvidia-smi -L 2>/dev/null | paste -sd ';' - || true)"
if [[ -z "${GPU_INFO}" ]]; then
    GPU_INFO="unavailable"
fi

META_PATH="${RUN_DIR}/meta.json"
SUMMARY_JSON_PATH="${ANALYSIS_DIR}/summary_overall.json"
RUNS_JSONL_PATH="results/exp_rlcompose/runs.jsonl"
LABBOOK_PATH="results/exp_rlcompose/labbook.md"
mkdir -p "results/exp_rlcompose"
touch "${RUNS_JSONL_PATH}" "${LABBOOK_PATH}"

export RUN_ID DATA_PATH SAVE_PATH N_SAMPLES TEMPERATURE K NOTES
export START_TIME_UTC END_TIME_UTC DURATION_SEC
export GIT_COMMIT HOSTNAME_VALUE GPU_INFO
export MODEL_PATH PROMPT_LENGTH RESPONSE_LENGTH N_GPUS_PER_NODE NNODES
export META_PATH SUMMARY_JSON_PATH RUNS_JSONL_PATH LABBOOK_PATH ANALYSIS_DIR

"${PYTHON_BIN}" - <<'PY'
import json
import os


def to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default

summary_path = os.environ["SUMMARY_JSON_PATH"]
if os.path.exists(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
else:
    summary = {}

metrics = {
    "pass_at_1": to_float(summary.get("pass_at_1", 0.0)),
    "pass_at_k": to_float(summary.get("pass_at_k", 0.0)),
    "num_records": to_int(summary.get("num_records", 0)),
    "k": to_int(summary.get("k", os.environ.get("K", "32")), 32),
}

meta = {
    "run_id": os.environ["RUN_ID"],
    "start_time_utc": os.environ["START_TIME_UTC"],
    "end_time_utc": os.environ["END_TIME_UTC"],
    "duration_sec": to_int(os.environ.get("DURATION_SEC", "0"), 0),
    "git_commit": os.environ["GIT_COMMIT"],
    "host": os.environ["HOSTNAME_VALUE"],
    "gpu_info": os.environ["GPU_INFO"],
    "model_path": os.environ["MODEL_PATH"],
    "data_path": os.environ["DATA_PATH"],
    "n_samples": to_int(os.environ["N_SAMPLES"], 1),
    "temperature": to_float(os.environ["TEMPERATURE"], 0.0),
    "prompt_length": to_int(os.environ["PROMPT_LENGTH"], 1024),
    "response_length": to_int(os.environ["RESPONSE_LENGTH"], 512),
    "n_gpus_per_node": to_int(os.environ["N_GPUS_PER_NODE"], 4),
    "nnodes": to_int(os.environ["NNODES"], 1),
    "pred_parquet_path": os.environ["SAVE_PATH"],
    "analysis_dir": os.environ["ANALYSIS_DIR"],
    "metrics": metrics,
    "notes": os.environ.get("NOTES", ""),
}

meta_path = os.environ["META_PATH"]
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=True, indent=2)

runs_jsonl_path = os.environ["RUNS_JSONL_PATH"]
with open(runs_jsonl_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(meta, ensure_ascii=True) + "\n")

labbook_path = os.environ["LABBOOK_PATH"]
with open(labbook_path, "a", encoding="utf-8") as f:
    f.write(f"## {meta['run_id']}\n")
    f.write(f"- Start (UTC): {meta['start_time_utc']}\n")
    f.write(f"- End (UTC): {meta['end_time_utc']}\n")
    f.write(f"- Duration (sec): {meta['duration_sec']}\n")
    f.write(f"- Model: {meta['model_path']}\n")
    f.write(f"- Data: {meta['data_path']}\n")
    f.write(f"- N samples / Temp: {meta['n_samples']} / {meta['temperature']}\n")
    f.write(f"- pass@1: {metrics['pass_at_1']:.6f}\n")
    f.write(f"- pass@{metrics['k']}: {metrics['pass_at_k']:.6f}\n")
    f.write(f"- Notes: {meta['notes']}\n\n")

print(f"Wrote meta: {meta_path}")
print(f"Appended: {runs_jsonl_path}")
print(f"Appended: {labbook_path}")
PY

echo "[run_inference_and_record] Done: ${RUN_ID}"
