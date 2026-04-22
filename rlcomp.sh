#!/usr/bin/env bash
# RL-Compositionality — single-entrypoint script for install, env setup,
# training, and inference. Designed to work on any machine that shares this
# repo's filesystem (Huawei ModelArts clusters under /home/ma-user/work/...).
#
# USAGE
# -----
# Source mode (set env in the CURRENT shell; no action run):
#     source /home/ma-user/work/RL-Compositionality/rlcomp.sh
#
# Execute mode (sets env in a subshell, runs ACTION, then exits):
#     bash /home/ma-user/work/RL-Compositionality/rlcomp.sh ACTION [options]
#
# Actions:
#     install   Create/refresh .venv-rlcomp (runs scripts/install_env_py312.sh).
#               Only needed when the venv is missing or on a completely fresh
#               box. On shared-FS ModelArts nodes the venv already exists, so
#               start from "env".
#     env       Source bootstrap.sh and run a sanity check (imports verl/ray/
#               vllm/torch, prints GPU/cache/offload/save-policy summary).
#     prepare   Generate the 7 newops datasets via scripts/experiments/
#               prepare_newops_datasets.sh.
#     stage1    Stage 1 RFT (vLLM rollout -> filter -> FSDP SFT).
#     stage2    Stage 2 GRPO (needs >=4 GPU for full 8B model).
#     infer     Inference matrix.
#     all       prepare -> stage1 -> stage2 -> infer (serial).
#
# PORTABILITY CONTRACT
# --------------------
# "Environment setup is portable" means: on any new machine where
#   (a) this repo path /home/ma-user/work/RL-Compositionality/ resolves to the
#       same shared filesystem (so .venv-rlcomp/ is already present), AND
#   (b) an NVIDIA driver is installed (libcuda.so.1 reachable), AND
#   (c) Python 3.10 is available (the venv is bound to that interpreter),
# this script guarantees:
#   * PATH, PYTHONPATH, PYTHONNOUSERSITE, LD_LIBRARY_PATH, HF_* caches, RAY_TMPDIR,
#     TMPDIR, TRITON_CACHE_DIR, TORCH_EXTENSIONS_DIR, WANDB_* are all set
#     correctly regardless of what the calling shell inherited.
#   * GPU count auto-detected; memory knobs (GPU_MEM_UTIL, FSDP offload, SFT
#     offload, MODEL_DTYPE) auto-scale so the same script fits on 1 GPU or
#     saturates 4/8 GPUs.
#   * No .bashrc or /etc/environment modification required.
#
# Anything beyond env setup (dataset generation, model weights, distributed
# Ray cluster) comes from the shared filesystem or on-demand actions.

_RLCOMP_SELF="${BASH_SOURCE[0]:-$0}"
_RLCOMP_HERE="$(cd "$(dirname "${_RLCOMP_SELF}")" && pwd)"
_RLCOMP_BOOT="${_RLCOMP_HERE}/scripts/env/bootstrap.sh"
_RLCOMP_RUN="${_RLCOMP_HERE}/scripts/env/run.sh"
_RLCOMP_INSTALL="${_RLCOMP_HERE}/scripts/install_env_py312.sh"

if [ ! -f "${_RLCOMP_BOOT}" ]; then
    echo "[rlcomp] ERROR: cannot find ${_RLCOMP_BOOT}" >&2
    echo "[rlcomp]        Is this the RL-Compositionality repo root?" >&2
    return 1 2>/dev/null || exit 1
fi

# ---------- source mode: set env and return, leaving caller's shell intact ----------
# (Do NOT apply `set -e` / `set -u` here — it would leak into the caller.)
if [ "${BASH_SOURCE[0]:-}" != "${0}" ]; then
    # shellcheck disable=SC1090
    source "${_RLCOMP_BOOT}"
    return 0 2>/dev/null || true
fi

# ---------- execute mode: strict shell, single ACTION ----------
set -euo pipefail
ACTION="${1:-}"
if [ -z "${ACTION}" ]; then
    sed -n '2,40p' "${_RLCOMP_SELF}"
    exit 2
fi
shift || true

case "${ACTION}" in
    install)
        if [ ! -x "${_RLCOMP_INSTALL}" ]; then
            echo "[rlcomp] ERROR: install script missing: ${_RLCOMP_INSTALL}" >&2
            exit 1
        fi
        exec bash "${_RLCOMP_INSTALL}" "$@"
        ;;
    env|prepare|stage1|stage2|infer|all|-h|--help)
        exec bash "${_RLCOMP_RUN}" "${ACTION}" "$@"
        ;;
    *)
        echo "[rlcomp] Unknown action: ${ACTION}" >&2
        echo "[rlcomp] Try: install | env | prepare | stage1 | stage2 | infer | all" >&2
        exit 2
        ;;
esac
