# shellcheck shell=bash
# RL-Compositionality environment bootstrap.
#
# Purpose: make a fresh shell fully ready to run Stage1/Stage2/Inference without
# touching ~/.bashrc. Must be *sourced*, not executed.
#
# Usage:
#     source scripts/env/bootstrap.sh    (from repo root)
#     source /path/to/RL-Compositionality/scripts/env/bootstrap.sh
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

# ---------- 2a. normalize pip config if a repo-local config exists ----------
# Huawei ModelArts images pre-set PIP_INDEX_URL to a mirror that lacks many
# pinned versions. A follow-up offline-environment PR adds a versioned
# pip.conf; when present, prefer it over host-level mirror settings.
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST PIP_NO_CACHE_DIR \
      MA_PIP_URL MA_PIP_HOST
if [ -f "${RLCOMP_REPO_ROOT}/pip.conf" ]; then
    export PIP_CONFIG_FILE="${RLCOMP_REPO_ROOT}/pip.conf"
fi

# ---------- 2b. local secrets / per-machine overrides ----------
# .env.local is gitignored and intended for credentials (WANDB_API_KEY,
# HUGGING_FACE_HUB_TOKEN, etc.) and per-machine knob overrides. If present,
# source it so every downstream process inherits the values. KEY=VALUE lines
# are auto-exported via `set -a`.
if [ -f "${RLCOMP_REPO_ROOT}/.env.local" ]; then
    set -a
    # shellcheck disable=SC1091
    . "${RLCOMP_REPO_ROOT}/.env.local"
    set +a
fi

# ---------- 3. activate venv ----------
# The pipeline expects an already-created project venv under .venv-rlcomp/.
# The offline cache/rebuild scripts that create or repair this venv are added
# in the follow-up environment PR.
_RLCOMP_VENV="${RLCOMP_REPO_ROOT}/.venv-rlcomp"
if [ ! -f "${_RLCOMP_VENV}/bin/activate" ]; then
    echo "[bootstrap] ERROR: venv not found at ${_RLCOMP_VENV}." >&2
    echo "[bootstrap] Create or link the project environment before running rlcomp.sh." >&2
    unset _RLCOMP_VENV
    return 1 2>/dev/null || exit 1
fi
# Deactivate prior venv if any, so repeated sourcing is clean.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -n "$(type -t deactivate 2>/dev/null)" ]; then
    deactivate 2>/dev/null || true
fi
# shellcheck disable=SC1090,SC1091
. "${_RLCOMP_VENV}/bin/activate"

# Python venvs bake their absolute creation path into bin/activate. On a
# shared filesystem accessed via a different mount prefix (e.g. the venv was
# created at <repo-root>/.venv-rlcomp but this
# machine sees the same files at /inspire/.../dllm/RL-Compositionality/.venv-rlcomp),
# activate sets VIRTUAL_ENV to the stale path and every python/torchrun call
# hits "No such file or directory". Force VIRTUAL_ENV and PATH to the path we
# resolved from this file's own location.
if [ "${VIRTUAL_ENV}" != "${_RLCOMP_VENV}" ]; then
    _RLCOMP_STALE_BIN="${VIRTUAL_ENV}/bin"
    export VIRTUAL_ENV="${_RLCOMP_VENV}"
    # Strip the stale venv bin (if present) from PATH, prepend the real one.
    _RLCOMP_NEW_PATH=":${PATH}:"
    _RLCOMP_NEW_PATH="${_RLCOMP_NEW_PATH//:${_RLCOMP_STALE_BIN}:/:}"
    _RLCOMP_NEW_PATH="${_RLCOMP_NEW_PATH#:}"
    _RLCOMP_NEW_PATH="${_RLCOMP_NEW_PATH%:}"
    export PATH="${VIRTUAL_ENV}/bin:${_RLCOMP_NEW_PATH}"
    unset _RLCOMP_STALE_BIN _RLCOMP_NEW_PATH
fi
unset _RLCOMP_VENV
export RLCOMP_PYTHON="${VIRTUAL_ENV}/bin/python"

# ---------- 3b. venv entry-point self-heal ----------
# Some wheels (observed with torch 2.6 on this venv) install the package
# correctly but skip generating console_scripts in ${VIRTUAL_ENV}/bin/.
# On 2026-04-24 the rlcomps H100 node failed stage1 with:
#   examples/sft/template.sh: line 37: .venv-rlcomp/bin/torchrun: No such file
# torch-*.dist-info/entry_points.txt declares `torchrun = torch.distributed.run:main`
# so this shim is what pip *should* have generated. Recreate if absent.
if [ ! -x "${VIRTUAL_ENV}/bin/torchrun" ]; then
    cat > "${VIRTUAL_ENV}/bin/torchrun" <<'_RLCOMP_TORCHRUN_EOF'
#!/bin/sh
'''exec' "$(dirname -- "$(realpath -- "$0")")"/'python' "$0" "$@"
' '''
# -*- coding: utf-8 -*-
# Generated by scripts/env/bootstrap.sh (self-heal): mirrors
# torch-*.dist-info/entry_points.txt "torchrun = torch.distributed.run:main".
# Same sh+python polyglot shebang as pip's other scripts in this venv for
# mount-relative relocatability across shared-FS nodes.
import sys
from torch.distributed.run import main
if __name__ == "__main__":
    if sys.argv[0].endswith("-script.pyw"):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith(".exe"):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())
_RLCOMP_TORCHRUN_EOF
    chmod +x "${VIRTUAL_ENV}/bin/torchrun" 2>/dev/null || true
    echo "[bootstrap] NOTE: generated missing ${VIRTUAL_ENV}/bin/torchrun entry-point" >&2
fi

# ---------- 4. python / import hygiene ----------
export PYTHONNOUSERSITE=1
# Prepend repo root to PYTHONPATH so `import verl` resolves without re-install.
case ":${PYTHONPATH:-}:" in
    *":${RLCOMP_REPO_ROOT}:"*) : ;;
    *) export PYTHONPATH="${RLCOMP_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

# ---------- 5. model / HF cache ----------
# Derive from repo root rather than hardcoding an absolute path, so the
# same script works on any mount point (shared FS, different container path).
# Pseudo-root mode: everything lives inside the repo by default.
# Override RLCOMP_WORK_ROOT before sourcing if you want models/caches outside.
export RLCOMP_WORK_ROOT="${RLCOMP_WORK_ROOT:-${RLCOMP_REPO_ROOT}}"
export MODEL_STORE_ROOT="${MODEL_STORE_ROOT:-${RLCOMP_WORK_ROOT}/model_store}"
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
# Ray sockets need a SHORT path: AF_UNIX's sockaddr_un.sun_path caps the full
# socket name at 107 bytes (108 incl NUL). Ray appends
#     /ray/session_<YYYY-MM-DD_HH-MM-SS>_<microsec>_<pid>/sockets/plasma_store
# after RAY_TMPDIR. Max suffix length (7-digit microsec + 7-digit PID) is
# 13 + 19 + 1 + 7 + 1 + 7 + 21 = 69 bytes, so RAY_TMPDIR itself must be
# <= 107 - 69 = 38 bytes for the socket path to fit.
#
# The prior threshold of 100 was incorrect: it checked RAY_TMPDIR's own length,
# not the total. On 2026-04-24 an H100 node failed stage1 with
#   OSError: AF_UNIX path length cannot exceed 107 bytes:
#   /home/ma-user/work/RL-Compositionality/.raytmp/ray/session_..._57979/sockets/plasma_store
# because RLCOMP_WORK_ROOT/.raytmp = 46 bytes left only 61 bytes for the 66-byte
# suffix Ray generated. Budget check below uses the real limit.
#
# Prefer a path under the shared work dir (persistent, cross-node visible), but
# fall back to /tmp/.raytmp.<uid> (per-node, not shared — fine for runtime
# sockets) when the work-dir path exceeds the budget.
_RLCOMP_RAY_SUFFIX_MAX=69
_RLCOMP_RAY_BUDGET=$((107 - _RLCOMP_RAY_SUFFIX_MAX))  # = 38
if [ -z "${RAY_TMPDIR:-}" ]; then
    _RLCOMP_RAY_CAND="${RLCOMP_WORK_ROOT}/.raytmp"
    if [ "${#_RLCOMP_RAY_CAND}" -le "${_RLCOMP_RAY_BUDGET}" ]; then
        RAY_TMPDIR="${_RLCOMP_RAY_CAND}"
    else
        RAY_TMPDIR="/tmp/.raytmp.$(id -u 2>/dev/null || echo 0)"
    fi
    unset _RLCOMP_RAY_CAND
fi
export RAY_TMPDIR
# Sanity check: RAY_TMPDIR may be too long because (a) the user explicitly set
# it too long, or (b) a PREVIOUS source of bootstrap.sh — pre-fix — exported a
# stale value that a child shell (e.g. `bash rlcomp.sh all` after the parent
# did `source rlcomp.sh` with the old logic) now inherits. The -z guard above
# intentionally honors caller-provided RAY_TMPDIR, so we can't rely on the
# if-set branch to recompute. Warn and auto-fallback here instead of hard
# error: Ray would crash at ray.init() anyway, so rescuing is strictly safer.
if [ "${#RAY_TMPDIR}" -gt "${_RLCOMP_RAY_BUDGET}" ]; then
    _RLCOMP_RAY_OLD="${RAY_TMPDIR}"
    RAY_TMPDIR="/tmp/.raytmp.$(id -u 2>/dev/null || echo 0)"
    export RAY_TMPDIR
    echo "[bootstrap] WARN: inherited RAY_TMPDIR='${_RLCOMP_RAY_OLD}' (${#_RLCOMP_RAY_OLD} bytes)" >&2
    echo "[bootstrap]       exceeds ${_RLCOMP_RAY_BUDGET}-byte budget (AF_UNIX cap 107 - suffix ${_RLCOMP_RAY_SUFFIX_MAX})." >&2
    echo "[bootstrap]       Auto-falling back to RAY_TMPDIR='${RAY_TMPDIR}' (${#RAY_TMPDIR} bytes)." >&2
    unset _RLCOMP_RAY_OLD
fi
# Final defensive check: fallback path itself could be too long on exotic
# systems (e.g. /tmp remapped to a long container path). This really is
# unrecoverable without caller help.
if [ "${#RAY_TMPDIR}" -gt "${_RLCOMP_RAY_BUDGET}" ]; then
    echo "[bootstrap] ERROR: even fallback RAY_TMPDIR='${RAY_TMPDIR}' (${#RAY_TMPDIR} bytes)" >&2
    echo "[bootstrap]        exceeds ${_RLCOMP_RAY_BUDGET}-byte budget. Set a short RAY_TMPDIR manually." >&2
    unset _RLCOMP_RAY_SUFFIX_MAX _RLCOMP_RAY_BUDGET
    return 1 2>/dev/null || exit 1
fi
unset _RLCOMP_RAY_SUFFIX_MAX _RLCOMP_RAY_BUDGET
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR}/torch_ext}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"

# ---------- 6b. SFT DataLoader workers vs. /dev/shm size ----------
# PyTorch DataLoader with num_workers>0 uses /dev/shm (tmpfs) to ferry collated
# batches from worker subprocesses back to the trainer. K8s/Docker pods default
# /dev/shm to 64 MiB (historical Docker default), which CANNOT fit the 4 rank x
# 8 worker x 2 dataloader peak of a 128x3072 int64 batch (~700 MB needed). On
# 2026-04-24 this caused stage1 SFT to crash at iter-0 with
# "RuntimeError: unable to write to file </torch_XXX>: No space left on device (28)".
# Here we probe the filesystem and set SFT_NUM_WORKERS=0 when /dev/shm is
# "small" so that run_stage1_newops_rft.sh threads `data.num_workers=0` to
# the trainer via Hydra. Threshold 1 GiB: empirically any host with >1 GiB
# /dev/shm (physical hosts, properly sized pods) can safely run 8 workers.
if [ -z "${SFT_NUM_WORKERS:-}" ]; then
    _RLCOMP_SHM_BYTES="$(df -B1 --output=size /dev/shm 2>/dev/null | awk 'NR==2{print $1}')"
    if [ -n "${_RLCOMP_SHM_BYTES}" ] && [ "${_RLCOMP_SHM_BYTES}" -lt $((1024 * 1024 * 1024)) ]; then
        export SFT_NUM_WORKERS=0
        echo "[bootstrap] NOTE: /dev/shm is $((_RLCOMP_SHM_BYTES / 1024 / 1024)) MiB (<1 GiB);" >&2
        echo "[bootstrap]       forcing SFT_NUM_WORKERS=0 to avoid DataLoader Bus error." >&2
    else
        export SFT_NUM_WORKERS=8
    fi
    unset _RLCOMP_SHM_BYTES
fi

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

# vLLM gpu_memory_utilization: scaled to leave headroom for the FSDP-vLLM
# hybrid engine handoff at fsdp_vllm.py:__enter__, which materializes the
# full actor state_dict() back to GPU RIGHT BEFORE calling vllm.wake_up().
# Even with param_offload=True the state_dict step temporarily pulls the
# offloaded params back, so peak memory at wake_up = vLLM weights + state_dict
# + KV cache to remap. Verified on 2026-04-25: 0.85 on 8xH100 OOMed at
# wake_up because state_dict (32GB fp32) + vLLM weights (16GB) + KV cache
# region (52GB) = 100GB > 80GB.
# 0.70 on 3+ GPUs leaves ~8 GB margin even with bf16 state_dict.
#
# **Unconditional assignment** — same stale-env reasoning as the offload knobs:
# hw-pipline's old bootstrap exported GPU_MEM_UTIL=0.85 on 8+ GPUs and any
# interactive shell that sourced rlcomp.sh before this fix has stale "0.85"
# in env which `${VAR:-...}` would honor. To force a different value (e.g.
# for a host with extra VRAM), set RLCOMP_FORCE_GPU_MEM_UTIL before sourcing.
if [ -n "${RLCOMP_FORCE_GPU_MEM_UTIL:-}" ]; then
    GPU_MEM_UTIL="${RLCOMP_FORCE_GPU_MEM_UTIL}"
else
    case "${N_GPUS_PER_NODE}" in
        1)  GPU_MEM_UTIL="0.40" ;;
        2)  GPU_MEM_UTIL="0.55" ;;
        *)  GPU_MEM_UTIL="0.70" ;;
    esac
fi
export GPU_MEM_UTIL

# FSDP offload (Stage2 GRPO/PPO hybrid engine): align with upstream
# PRIME-RL/RL-Compositionality default of True for actor params, optimizer state,
# and ref weights. The hybrid-engine sleep/wake pattern in
# verl/workers/sharding_manager/fsdp_vllm.py:__enter__ tries to remap vLLM's
# frozen ~0.8*GPU_MEM cumem region while FSDP weights are still GPU-resident
# (the "TODO: offload FSDP model weights" at fsdp_vllm.py:117 is commented out
# upstream). Without these offloads the wake_up() collides with the resident
# actor and OOMs at cumem_allocator.cpp:62. Verified on 2026-04-25 on 8xH100.
#
# **Unconditional assignment** — NOT `${VAR:-True}` — because hw-pipline's old
# bootstrap exported these as "False" on multi-GPU, and any interactive shell
# that sourced rlcomp.sh once before this fix has stale "False" in its env.
# `bash rlcomp.sh stage2` then `exec bash run.sh stage2` inherits those stale
# values, and `:-True` honors them. Force-override here guarantees a clean
# value regardless of inherited shell state. To intentionally disable offload
# (e.g. on a 192GB B200 where you want speed > memory headroom), set
# RLCOMP_FORCE_FSDP_OFFLOAD=False before sourcing bootstrap.sh.
if [ -n "${RLCOMP_FORCE_FSDP_OFFLOAD:-}" ]; then
    ACTOR_PARAM_OFFLOAD="${RLCOMP_FORCE_FSDP_OFFLOAD}"
    ACTOR_OPTIMIZER_OFFLOAD="${RLCOMP_FORCE_FSDP_OFFLOAD}"
    REF_PARAM_OFFLOAD="${RLCOMP_FORCE_FSDP_OFFLOAD}"
else
    ACTOR_PARAM_OFFLOAD="True"
    ACTOR_OPTIMIZER_OFFLOAD="True"
    REF_PARAM_OFFLOAD="True"
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

# Model dtype: bf16 default for ALL GPU counts.
# - 1 GPU: bf16 halves CPU RAM footprint so the cgroup (often ~100-120GB in
#   ModelArts containers) can fit an 8B model.
# - Multi-GPU: bf16 halves the size of state_dict() materialized at the
#   stage2 fsdp_vllm.py:__enter__ handoff (32 GB fp32 -> 16 GB bf16),
#   which is what causes vllm.wake_up() to OOM with fp32 even when
#   param_offload=True. Llama-3-8B was trained in bf16 so this matches
#   the model's native precision and the upstream PRIME-RL convention.
#
# **Unconditional assignment** — same reasoning as the offload knobs above:
# hw-pipline's old bootstrap exported MODEL_DTYPE=fp32 on multi-GPU, and any
# interactive shell that sourced rlcomp.sh once before this fix has stale
# "fp32" in env which `${VAR:-bf16}` would honor. To force fp32 (e.g. to
# investigate a precision-sensitive bug), set RLCOMP_FORCE_MODEL_DTYPE=fp32
# before sourcing bootstrap.sh.
if [ -n "${RLCOMP_FORCE_MODEL_DTYPE:-}" ]; then
    MODEL_DTYPE="${RLCOMP_FORCE_MODEL_DTYPE}"
else
    MODEL_DTYPE="bf16"
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
# Default mode:
#   - online  if WANDB_API_KEY is set AND api.wandb.ai reachable in <=3s
#   - offline if WANDB_API_KEY is set but api.wandb.ai is NOT reachable
#             (still keeps a local run dir under WANDB_DIR; user can sync
#              later with `wandb sync wandb/offline-run-*` from a connected box)
#   - offline if WANDB_API_KEY is unset (no creds -> nothing to upload)
# Caller can force any mode explicitly via WANDB_MODE=... (online/offline/disabled).
#
# Safety contract: this block MUST NOT hang or fail bootstrap. The reachability
# probe is hard-bounded to 3s wallclock + 2s connect; any curl error (DNS fail,
# TCP refuse, timeout, no curl binary) short-circuits to offline. We tested 0
# training steps after a 90s wandb.init() timeout on a no-egress K8s pod on
# 2026-04-24 — auto-fallback to offline prevents that from ever blocking again.
if [ -z "${WANDB_MODE:-}" ]; then
    if [ -n "${WANDB_API_KEY:-}" ]; then
        if command -v curl >/dev/null 2>&1 \
           && curl --max-time 3 --connect-timeout 2 -sS -o /dev/null \
                   https://api.wandb.ai/ </dev/null 2>/dev/null; then
            WANDB_MODE="online"
        else
            WANDB_MODE="offline"
            echo "[bootstrap] NOTE: WANDB_API_KEY set but api.wandb.ai unreachable in 3s;" >&2
            echo "[bootstrap]       WANDB_MODE=offline. Local run records still go to" >&2
            echo "[bootstrap]       ${WANDB_DIR:-${RLCOMP_REPO_ROOT}/wandb}/. Sync later with:" >&2
            echo "[bootstrap]         wandb sync ${WANDB_DIR:-${RLCOMP_REPO_ROOT}/wandb}/offline-run-*" >&2
        fi
    else
        WANDB_MODE="offline"
    fi
fi
export WANDB_MODE
export WANDB_DIR="${WANDB_DIR:-${RLCOMP_REPO_ROOT}/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-rlcompose-new-abilities}"
mkdir -p "${WANDB_DIR}"

# ---------- 10. checkpoint + validation throttling ----------
# These two knobs have DIFFERENT failure modes — pick them independently:
#
# SAVE_FREQ=200:
#   Each stage2 PPO save_checkpoint writes ~90 GB to NFS (8 ranks × 11 GB
#   shards). At save_freq=50 we observed 5 saves in 19h burning ~450 GB into
#   the OS page cache, climbing pod RAM monotonically 33% → 75% (K8s cgroup
#   OOM-kills at 100%). Verified 2026-04-26 in
#   results/pipeline_runs/20260425_115819_*/stage2.stdout.log:
#   step:50,100,150,200,250 each shows perf/cpu_memory_used_gb climbing.
#   200 spreads saves to once per ~10h, drops page-cache pressure 4×.
#
# TEST_FREQ=25:
#   Matches upstream's hardcoded test_freq in examples/grpo_trainer/template.sh.
#   Each _validate run takes 15-40 min (8 difficulty levels × val set, growing
#   with response length over training). Test does NOT contribute to RAM/page-
#   cache pressure (no big NFS writes), so the lever here is purely wall-clock
#   vs trajectory visibility. 25 yields 60 trajectory points per ~1500-step
#   epoch — dense enough to see complete RL learning dynamics, including any
#   transient OOD regression (we observed depth2 acc dropping 0.40→0.24
#   between steps 200 and 250 — at freq=25 this would be a 2-step transition).
#   Cost: ~30h testing wallclock per epoch (vs ~104h training proper). User
#   explicitly chose dense trajectory > wallclock economy.
#
# **Unconditional assignment** — same stale-env reasoning as offload/dtype:
# any interactive shell that sourced rlcomp.sh before this fix has stale "50"
# in env, which `${VAR:-...}` would honor. Force-overwrite to guarantee the
# new value lands on the H100 pod's next `bash rlcomp.sh stage2`. Power users
# can override via RLCOMP_FORCE_SAVE_FREQ / RLCOMP_FORCE_TEST_FREQ.
if [ -n "${RLCOMP_FORCE_SAVE_FREQ:-}" ]; then
    SAVE_FREQ="${RLCOMP_FORCE_SAVE_FREQ}"
else
    SAVE_FREQ="150"
fi
if [ -n "${RLCOMP_FORCE_TEST_FREQ:-}" ]; then
    TEST_FREQ="${RLCOMP_FORCE_TEST_FREQ}"
else
    TEST_FREQ="10"
fi
export SAVE_FREQ TEST_FREQ
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
  save policy    : save_freq=${SAVE_FREQ}, test_freq=${TEST_FREQ}, remove_previous_ckpt_in_save=${REMOVE_PREVIOUS_CKPT_IN_SAVE}, keep_last_n=${RLCOMP_KEEP_LAST_N}
  disk (work FS) : $(_rlcomp_free_disk)
Next: bash scripts/env/run.sh {prepare|stage1|stage2|infer|all}
BANNER
unset -f _rlcomp_free_disk _rlcomp_gpu_brief _rlcomp_git_branch
