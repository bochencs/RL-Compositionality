# shellcheck shell=bash
# RL-Compositionality environment bootstrap.
#
# Purpose: make a fresh shell fully ready to run Stage1/Stage2/Inference without
# touching ~/.bashrc. Must be *sourced*, not executed.
#
# Usage:
#     source /home/ma-user/work/RL-Compositionality/scripts/env/bootstrap.sh
#
# Works from any cwd — self-locates the repo relative to this file.
# Re-sourcing is idempotent.
#
# Public outputs (exported):
#   RLCOMP_REPO_ROOT, RLCOMP_PYTHON, N_GPUS_PER_NODE, NNODES, NPROC_PER_NODE,
#   MODEL_STORE_ROOT, HF_MODEL_ROOT, HF_CACHE_ROOT, HF_HOME,
#   HUGGINGFACE_HUB_CACHE, TRANSFORMERS_CACHE, RAY_TMPDIR, TMPDIR, TMP, TEMP,
#   TORCH_EXTENSIONS_DIR, TRITON_CACHE_DIR, PYTHONNOUSERSITE, PYTHONPATH,
#   TOKENIZERS_PARALLELISM, NCCL_DEBUG, VLLM_LOGGING_LEVEL,
#   VLLM_ATTENTION_BACKEND, RAY_DEDUP_LOGS, WANDB_MODE, WANDB_DIR,
#   WANDB_PROJECT, SAVE_FREQ, REMOVE_PREVIOUS_CKPT_IN_SAVE, RLCOMP_KEEP_LAST_N

# ---------- guard: must be sourced ----------
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "[bootstrap] ERROR: this script must be sourced, not executed." >&2
    echo "[bootstrap] Run:  source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

# ---------- 1. locate repo root ----------
# Resolve symlinks in case this file is linked.
_RLCOMP_BOOTSTRAP_SRC="${BASH_SOURCE[0]}"
while [ -L "${_RLCOMP_BOOTSTRAP_SRC}" ]; do
    _RLCOMP_LINK="$(readlink "${_RLCOMP_BOOTSTRAP_SRC}")"
    case "${_RLCOMP_LINK}" in
        /*) _RLCOMP_BOOTSTRAP_SRC="${_RLCOMP_LINK}" ;;
        *)  _RLCOMP_BOOTSTRAP_SRC="$(dirname "${_RLCOMP_BOOTSTRAP_SRC}")/${_RLCOMP_LINK}" ;;
    esac
done
export RLCOMP_REPO_ROOT="$(cd "$(dirname "${_RLCOMP_BOOTSTRAP_SRC}")/../.." && pwd)"
unset _RLCOMP_BOOTSTRAP_SRC _RLCOMP_LINK

if [ ! -d "${RLCOMP_REPO_ROOT}/verl" ] || [ ! -d "${RLCOMP_REPO_ROOT}/scripts/experiments" ]; then
    echo "[bootstrap] ERROR: RLCOMP_REPO_ROOT=${RLCOMP_REPO_ROOT} does not look like the RL-Compositionality repo." >&2
    return 1 2>/dev/null || exit 1
fi

cd "${RLCOMP_REPO_ROOT}"

# ---------- 2. drop proxies (per README) ----------
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# ---------- 3. activate venv ----------
_RLCOMP_VENV="${RLCOMP_REPO_ROOT}/.venv-rlcomp"
if [ ! -f "${_RLCOMP_VENV}/bin/activate" ]; then
    echo "[bootstrap] ERROR: venv not found at ${_RLCOMP_VENV}." >&2
    echo "[bootstrap] Create it per README.md then re-source this file." >&2
    unset _RLCOMP_VENV
    return 1 2>/dev/null || exit 1
fi
# Deactivate prior venv if any, so repeated sourcing is clean.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -n "$(type -t deactivate 2>/dev/null)" ]; then
    deactivate 2>/dev/null || true
fi
# shellcheck disable=SC1090,SC1091
. "${_RLCOMP_VENV}/bin/activate"
unset _RLCOMP_VENV
export RLCOMP_PYTHON="${VIRTUAL_ENV}/bin/python"

# ---------- 4. python / import hygiene ----------
export PYTHONNOUSERSITE=1
# Prepend repo root to PYTHONPATH so `import verl` resolves without re-install.
case ":${PYTHONPATH:-}:" in
    *":${RLCOMP_REPO_ROOT}:"*) : ;;
    *) export PYTHONPATH="${RLCOMP_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

# ---------- 5. model / HF cache ----------
export MODEL_STORE_ROOT="${MODEL_STORE_ROOT:-/home/ma-user/work/model_store}"
export HF_MODEL_ROOT="${HF_MODEL_ROOT:-${MODEL_STORE_ROOT}/hf_models}"
export HF_CACHE_ROOT="${HF_CACHE_ROOT:-${MODEL_STORE_ROOT}/hf_cache}"
export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_ROOT}/transformers}"

# ---------- 6. temp / scratch ----------
# Repo-local .tmp for most short-lived files (shared FS, visible across nodes).
export TMPDIR="${TMPDIR:-${RLCOMP_REPO_ROOT}/.tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
# Ray sockets need a SHORT path (<107 bytes for AF_UNIX). /home/ma-user/work/.raytmp = 26 bytes.
export RAY_TMPDIR="${RAY_TMPDIR:-/home/ma-user/work/.raytmp}"
if [ "${#RAY_TMPDIR}" -ge 107 ]; then
    echo "[bootstrap] ERROR: RAY_TMPDIR length ${#RAY_TMPDIR} >= 107 will break AF_UNIX sockets." >&2
    return 1 2>/dev/null || exit 1
fi
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR}/torch_ext}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"

mkdir -p \
    "${HF_MODEL_ROOT}" "${HF_CACHE_ROOT}" \
    "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
    "${TMPDIR}" "${RAY_TMPDIR}" \
    "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}"

# ---------- 6b. ensure libcuda.so is on the linker path ----------
# Triton (used by vLLM's XFORMERS attention backend) JIT-compiles CUDA stubs at
# runtime and needs libcuda.so.1. A normal login shell inherits LD_LIBRARY_PATH
# from /etc/profile.d or Docker ENV, but a clean shell (env -i, some CI runners,
# some ssh-with-nologin contexts) does not. Auto-discover and append — ordered
# so the installed NVIDIA driver wins over older cuda-compat shims (mixing
# driver 535.x with an older cuda-12.2/compat libcuda.so triggers
# "system has unsupported display driver / cuda driver combination", CUDA
# error 803). Only fall back to cuda-*/compat if the real driver libs are
# missing entirely.
for _RLCOMP_CUDA_DIR in \
    /usr/local/nvidia/lib64 \
    /usr/lib64 \
    /usr/lib/x86_64-linux-gnu \
    /usr/local/cuda/lib64; do
    if [ -f "${_RLCOMP_CUDA_DIR}/libcuda.so.1" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":${_RLCOMP_CUDA_DIR}:"*) : ;;
            *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${_RLCOMP_CUDA_DIR}" ;;
        esac
    fi
done
unset _RLCOMP_CUDA_DIR

# If nothing got added above, fall back to cuda-*/compat — better than nothing.
if ! { python -c 'import ctypes; ctypes.CDLL("libcuda.so.1")' 2>/dev/null; }; then
    for _RLCOMP_CUDA_DIR in /usr/local/cuda/compat /usr/local/cuda-12.2/compat; do
        if [ -f "${_RLCOMP_CUDA_DIR}/libcuda.so.1" ]; then
            case ":${LD_LIBRARY_PATH:-}:" in
                *":${_RLCOMP_CUDA_DIR}:"*) : ;;
                *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${_RLCOMP_CUDA_DIR}" ;;
            esac
        fi
    done
    unset _RLCOMP_CUDA_DIR
fi

# ---------- 7. GPU auto-detection ----------
# Respect caller override; otherwise count via nvidia-smi, fall back to 1.
if [ -z "${N_GPUS_PER_NODE:-}" ]; then
    _RLCOMP_GPU_COUNT=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        _RLCOMP_GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    fi
    # Guard against non-numeric / empty output.
    case "${_RLCOMP_GPU_COUNT}" in
        ''|*[!0-9]*) _RLCOMP_GPU_COUNT=0 ;;
    esac
    if [ "${_RLCOMP_GPU_COUNT}" -le 0 ]; then
        _RLCOMP_GPU_COUNT=1
    fi
    export N_GPUS_PER_NODE="${_RLCOMP_GPU_COUNT}"
    unset _RLCOMP_GPU_COUNT
fi
export NNODES="${NNODES:-1}"
# SFT path uses torchrun --nproc_per_node; mirror GPU count.
export NPROC_PER_NODE="${NPROC_PER_NODE:-${N_GPUS_PER_NODE}}"
# Do NOT set CUDA_VISIBLE_DEVICES here — let Ray / torchrun schedule.

# vLLM gpu_memory_utilization: upstream scripts hardcode 0.8 (run_string) or 0.9
# (grpo template on hw-pipline), which OOMs on 1 GPU because the FSDP actor
# (loaded in fp32 ≈ 32GB for Llama-8B) + vLLM's reservation exceeds an 80GB
# A100/A800. Scale down when GPU count is low so the pipeline works on any box.
# Override by setting GPU_MEM_UTIL before sourcing.
if [ -z "${GPU_MEM_UTIL:-}" ]; then
    case "${N_GPUS_PER_NODE}" in
        1) GPU_MEM_UTIL="0.40" ;;
        2) GPU_MEM_UTIL="0.55" ;;
        *) GPU_MEM_UTIL="0.80" ;;
    esac
fi
export GPU_MEM_UTIL

# FSDP offload: hw-pipline flipped these to False for 4+ GPU speed, but with
# 1 GPU the 8B actor + 8B ref + vLLM engine won't fit without offload. Default
# to offload=True for 1 GPU, False otherwise. Override with ACTOR_PARAM_OFFLOAD,
# ACTOR_OPTIMIZER_OFFLOAD, REF_PARAM_OFFLOAD if needed.
if [ -z "${ACTOR_PARAM_OFFLOAD:-}" ]; then
    if [ "${N_GPUS_PER_NODE}" -le 1 ]; then
        ACTOR_PARAM_OFFLOAD="True"
    else
        ACTOR_PARAM_OFFLOAD="False"
    fi
fi
if [ -z "${ACTOR_OPTIMIZER_OFFLOAD:-}" ]; then
    if [ "${N_GPUS_PER_NODE}" -le 1 ]; then
        ACTOR_OPTIMIZER_OFFLOAD="True"
    else
        ACTOR_OPTIMIZER_OFFLOAD="False"
    fi
fi
if [ -z "${REF_PARAM_OFFLOAD:-}" ]; then
    if [ "${N_GPUS_PER_NODE}" -le 1 ]; then
        REF_PARAM_OFFLOAD="True"
    else
        REF_PARAM_OFFLOAD="False"
    fi
fi
export ACTOR_PARAM_OFFLOAD ACTOR_OPTIMIZER_OFFLOAD REF_PARAM_OFFLOAD

# SFT memory: single GPU with 8B model needs cpu_offload + gradient_checkpointing
# or it OOMs (no FSDP sharding when world_size=1). Same rule as RL offload.
if [ -z "${CPU_OFFLOAD:-}" ]; then
    if [ "${N_GPUS_PER_NODE}" -le 1 ]; then
        CPU_OFFLOAD="True"
    else
        CPU_OFFLOAD="False"
    fi
fi
if [ -z "${OFFLOAD_PARAMS:-}" ]; then
    if [ "${N_GPUS_PER_NODE}" -le 1 ]; then
        OFFLOAD_PARAMS="True"
    else
        OFFLOAD_PARAMS="False"
    fi
fi
if [ -z "${GRAD_CKPT:-}" ]; then
    if [ "${N_GPUS_PER_NODE}" -le 1 ]; then
        GRAD_CKPT="True"
    else
        GRAD_CKPT="False"
    fi
fi
export CPU_OFFLOAD OFFLOAD_PARAMS GRAD_CKPT

# Model dtype for SFT: bf16 halves CPU RAM footprint so the container cgroup
# (often ~100-120GB in ModelArts containers) can fit an 8B model. On multi-GPU,
# fp32 stays the default because FSDP FULL_SHARD already shards memory.
if [ -z "${MODEL_DTYPE:-}" ]; then
    if [ "${N_GPUS_PER_NODE}" -le 1 ]; then
        MODEL_DTYPE="bf16"
    else
        MODEL_DTYPE="fp32"
    fi
fi
export MODEL_DTYPE

# ---------- 8. runtime hygiene ----------
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARN}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
# vLLM's CuMemAllocator is incompatible with expandable_segments; unset if user has it.
case "${PYTORCH_CUDA_ALLOC_CONF:-}" in
    *expandable_segments:True*)
        echo "[bootstrap] Unsetting incompatible PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
        unset PYTORCH_CUDA_ALLOC_CONF
        ;;
esac

# ---------- 9. wandb ----------
# Default to offline so a brand-new shell never hangs/fails on missing API key.
# To log online:   export WANDB_MODE=online && export WANDB_API_KEY=...
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${RLCOMP_REPO_ROOT}/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-rlcompose-new-abilities}"
mkdir -p "${WANDB_DIR}"

# ---------- 10. checkpoint throttling ----------
# Disk is tight: save frequently but aggressively reclaim old ckpts.
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-50}"
export REMOVE_PREVIOUS_CKPT_IN_SAVE="${REMOVE_PREVIOUS_CKPT_IN_SAVE:-True}"
# Advisory knob for run.sh cleanup hook; verl itself keeps only the latest ckpt
# when REMOVE_PREVIOUS_CKPT_IN_SAVE=True.
export RLCOMP_KEEP_LAST_N="${RLCOMP_KEEP_LAST_N:-2}"

# ---------- 11. banner ----------
_rlcomp_free_disk() {
    df -hP "${RLCOMP_REPO_ROOT}" 2>/dev/null | awk 'NR==2 {print $4 " free / " $2 " total"}'
}
_rlcomp_gpu_brief() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/^ *//'
    else
        echo "none"
    fi
}
_rlcomp_git_branch() {
    (cd "${RLCOMP_REPO_ROOT}" && git rev-parse --abbrev-ref HEAD 2>/dev/null) || echo "unknown"
}
cat <<BANNER
[rlcomp-bootstrap] ready
  repo root      : ${RLCOMP_REPO_ROOT}
  git branch     : $(_rlcomp_git_branch)
  venv           : ${VIRTUAL_ENV}
  python         : $(${RLCOMP_PYTHON} -V 2>&1 | tr -d '\n')
  GPUs           : ${N_GPUS_PER_NODE} x $(_rlcomp_gpu_brief)  (gpu_memory_utilization=${GPU_MEM_UTIL})
  libcuda        : $(python -c 'import ctypes, sys; sys.stderr.write(""); print(ctypes.CDLL("libcuda.so.1")._name)' 2>/dev/null || echo "NOT FOUND — Triton/vLLM will fail")
  FSDP offload   : actor_param=${ACTOR_PARAM_OFFLOAD}, actor_optim=${ACTOR_OPTIMIZER_OFFLOAD}, ref_param=${REF_PARAM_OFFLOAD}
  SFT offload    : cpu_offload=${CPU_OFFLOAD}, offload_params=${OFFLOAD_PARAMS}, grad_ckpt=${GRAD_CKPT}, dtype=${MODEL_DTYPE}
  HF cache       : ${HF_HOME}
  model store    : ${HF_MODEL_ROOT}
  TMPDIR         : ${TMPDIR}
  RAY_TMPDIR     : ${RAY_TMPDIR} (${#RAY_TMPDIR} bytes)
  WANDB_MODE     : ${WANDB_MODE} (dir: ${WANDB_DIR})
  save policy    : save_freq=${SAVE_FREQ}, remove_previous_ckpt_in_save=${REMOVE_PREVIOUS_CKPT_IN_SAVE}, keep_last_n=${RLCOMP_KEEP_LAST_N}
  disk (work FS) : $(_rlcomp_free_disk)
Next: bash scripts/env/run.sh {prepare|stage1|stage2|infer|all}
BANNER
unset -f _rlcomp_free_disk _rlcomp_gpu_brief _rlcomp_git_branch
