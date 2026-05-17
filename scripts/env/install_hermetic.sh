#!/usr/bin/env bash
# ============================================================================
# RL-Compositionality — fully offline environment install
#
# Rebuilds `.venv-rlcomp/` using ONLY files already present in the repo:
#   - `.python/cpython-3.10.*-linux-x86_64-gnu/` — standalone Python 3.10
#   - `wheels/`                                  — pre-downloaded wheel cache
#   - `requirements.lock.txt`                    — pinned package versions
#
# This script MUST work with zero network access. Anything that needs the
# internet belongs in `scripts/env/bootstrap_offline_cache.sh` (a DIFFERENT
# script that populates `.python/` and `wheels/` once, ideally from a machine
# with PyPI reach).
#
# Usage:
#   bash scripts/env/install_hermetic.sh            # creates venv if missing
#   bash scripts/env/install_hermetic.sh --force    # wipe existing venv first
#   bash scripts/env/install_hermetic.sh --check    # verify only, no install
# ============================================================================
set -euo pipefail

# ---------- locate repo ----------
_INSTALL_SRC="${BASH_SOURCE[0]}"
REPO_ROOT="$(cd "$(dirname "${_INSTALL_SRC}")/../.." && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${RLCOMP_VENV_DIR:-${REPO_ROOT}/.venv-rlcomp}"
WHEELS_DIR="${RLCOMP_WHEELS_DIR:-${REPO_ROOT}/wheels}"
REQ_FILE="${RLCOMP_REQ_FILE:-${REPO_ROOT}/requirements.lock.txt}"
PYTHON_ROOT="${RLCOMP_PYTHON_ROOT:-${REPO_ROOT}/.python}"

_project_path() {
    local label="$1"
    local raw="$2"
    local abs
    case "${raw}" in
        /*) abs="$(realpath -m "${raw}")" ;;
        *)  abs="$(realpath -m "${REPO_ROOT}/${raw}")" ;;
    esac
    case "${abs}" in
        "${REPO_ROOT}"|"${REPO_ROOT}"/*) printf '%s\n' "${abs}" ;;
        *)
            echo "[install] ERROR: ${label} must be inside the current project directory: ${REPO_ROOT}" >&2
            echo "[install]        got: ${abs}" >&2
            exit 1
            ;;
    esac
}

VENV_DIR="$(_project_path VENV_DIR "${VENV_DIR}")"
WHEELS_DIR="$(_project_path WHEELS_DIR "${WHEELS_DIR}")"
REQ_FILE="$(_project_path REQ_FILE "${REQ_FILE}")"
PYTHON_ROOT="$(_project_path PYTHON_ROOT "${PYTHON_ROOT}")"
TMPDIR="$(_project_path TMPDIR "${TMPDIR:-${REPO_ROOT}/.tmp}")"
TMP="$(_project_path TMP "${TMP:-${TMPDIR}}")"
TEMP="$(_project_path TEMP "${TEMP:-${TMPDIR}}")"
PIP_CACHE_DIR="$(_project_path PIP_CACHE_DIR "${PIP_CACHE_DIR:-${REPO_ROOT}/.pip-cache}")"
export TMPDIR TMP TEMP PIP_CACHE_DIR
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST PIP_NO_CACHE_DIR \
      MA_PIP_URL MA_PIP_HOST

FORCE=0
CHECK_ONLY=0
for arg in "$@"; do
    case "${arg}" in
        --force) FORCE=1 ;;
        --check) CHECK_ONLY=1 ;;
        -h|--help)
            sed -n '2,18p' "${_INSTALL_SRC}"
            exit 0
            ;;
        *)
            echo "[install] unknown arg: ${arg}" >&2
            exit 2
            ;;
    esac
done

# ---------- locate standalone Python ----------
# Expected layout: .python/cpython-3.10.NN-linux-x86_64-gnu/bin/python3.10
PYTHON_BIN=""
if [ -d "${PYTHON_ROOT}" ]; then
    # Prefer the versioned dir; fall back to the `cpython-3.10-...` symlink.
    _CAND="$(ls -d "${PYTHON_ROOT}"/cpython-3.10.*-linux-x86_64-gnu 2>/dev/null | sort | tail -1)"
    if [ -z "${_CAND}" ]; then
        _CAND="${PYTHON_ROOT}/cpython-3.10-linux-x86_64-gnu"
    fi
    if [ -x "${_CAND}/bin/python3.10" ]; then
        PYTHON_BIN="${_CAND}/bin/python3.10"
    fi
fi

# ---------- preflight checks ----------
errors=0
echo "[install] preflight checks:"
echo "  repo root     : ${REPO_ROOT}"
echo "  venv dir      : ${VENV_DIR} $( [ -d "${VENV_DIR}" ] && echo "(exists)" || echo "(to create)" )"
echo "  python        : ${PYTHON_BIN:-MISSING}"
echo "  wheels dir    : ${WHEELS_DIR} $( [ -d "${WHEELS_DIR}" ] && echo "($(ls "${WHEELS_DIR}" 2>/dev/null | wc -l) files)" || echo "(MISSING)" )"
echo "  requirements  : ${REQ_FILE} $( [ -f "${REQ_FILE}" ] && echo "($(wc -l <"${REQ_FILE}") lines)" || echo "(MISSING)" )"

if [ -z "${PYTHON_BIN}" ]; then
    echo "[install] ERROR: no standalone Python at ${PYTHON_ROOT}/cpython-3.10.*-linux-x86_64-gnu" >&2
    echo "[install]        Run scripts/env/bootstrap_offline_cache.sh on a networked machine first." >&2
    errors=$((errors + 1))
fi
if [ ! -d "${WHEELS_DIR}" ] || [ -z "$(ls -A "${WHEELS_DIR}" 2>/dev/null)" ]; then
    echo "[install] ERROR: no wheels at ${WHEELS_DIR}" >&2
    echo "[install]        Run scripts/env/bootstrap_offline_cache.sh on a networked machine first." >&2
    errors=$((errors + 1))
fi
if [ ! -f "${REQ_FILE}" ]; then
    echo "[install] ERROR: no ${REQ_FILE}" >&2
    errors=$((errors + 1))
fi

if [ "${errors}" -gt 0 ]; then
    echo "[install] ${errors} preflight error(s); aborting." >&2
    exit 1
fi

if [ "${CHECK_ONLY}" -eq 1 ]; then
    echo "[install] --check passed; no changes made."
    exit 0
fi

# ---------- create / repair venv ----------
if [ -d "${VENV_DIR}" ] && [ "${FORCE}" -ne 1 ]; then
    echo "[install] ${VENV_DIR} already exists."
    if [ -x "${VENV_DIR}/bin/python" ] && "${VENV_DIR}/bin/python" -c 'import sys; sys.exit(0)' >/dev/null 2>&1; then
        echo "[install] existing venv python starts successfully."
        echo "[install]   - to rebuild/repair from wheels, re-run with --force"
        echo "[install]   - for full diagnostics, run: bash rlcomp.sh doctor"
        exit 0
    fi
    echo "[install] ERROR: existing venv python is broken." >&2
    echo "[install]        Re-run with --force to rebuild from offline files." >&2
    exit 1
fi

if [ -d "${VENV_DIR}" ] && [ "${FORCE}" -eq 1 ]; then
    echo "[install] --force: removing existing venv"
    rm -rf "${VENV_DIR}"
fi

echo "[install] creating venv with standalone Python..."
# IMPORTANT: use --symlinks, not --copies.
# python-build-standalone binaries hardcode their build-time prefix ('/install')
# and resolve stdlib relative to sys.executable's realpath. A naked `cp` of the
# binary loses that resolution and crashes with:
#   Fatal Python error: init_fs_encoding: failed to get the Python codec of the
#   filesystem encoding  /  ModuleNotFoundError: No module named 'encodings'
# --symlinks makes bin/python a symlink to .python/.../bin/python3.10 so its
# realpath resolves back into the standalone install tree where stdlib lives.
"${PYTHON_BIN}" -m venv --symlinks "${VENV_DIR}"

# Activate and bootstrap pip/setuptools (standalone Python ships ensurepip).
# shellcheck disable=SC1091
. "${VENV_DIR}/bin/activate"
export PYTHONNOUSERSITE=1

echo "[install] bootstrapping pip / setuptools / wheel from wheels dir..."
"${VENV_DIR}/bin/python" -m pip install \
    --no-index --find-links="${WHEELS_DIR}" \
    --disable-pip-version-check \
    --upgrade pip setuptools wheel packaging 2>&1 | tail -5 || {
    # If pip/setuptools wheels aren't in the cache, fall back to ensurepip.
    echo "[install] warning: pip wheel not in cache, falling back to ensurepip"
    "${VENV_DIR}/bin/python" -m ensurepip --upgrade
}

echo "[install] installing ${REQ_FILE} from ${WHEELS_DIR}..."
"${VENV_DIR}/bin/python" -m pip install \
    --no-index --find-links="${WHEELS_DIR}" \
    --disable-pip-version-check \
    -r "${REQ_FILE}"

echo "[install] installing verl (editable, no deps)..."
"${VENV_DIR}/bin/python" -m pip install \
    --no-index --find-links="${WHEELS_DIR}" \
    --disable-pip-version-check \
    -e . --no-deps --no-build-isolation

echo "[install] verifying critical imports..."
"${VENV_DIR}/bin/python" - <<'PY'
import importlib, sys
mods = ["torch", "vllm", "flash_attn", "xformers", "triton", "ray",
        "transformers", "datasets", "hydra", "wandb", "verl"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<18} {getattr(mod, '__version__', 'n/a')}  OK")
    except Exception as e:
        print(f"  {m:<18} FAIL: {type(e).__name__}: {e}")
        bad.append(m)
import torch
print(f"  CUDA available   : {torch.cuda.is_available()}")
print(f"  CUDA runtime     : {torch.version.cuda}")
print(f"  GPU count        : {torch.cuda.device_count()}")
if bad:
    print(f"[install] {len(bad)} module(s) failed to import: {bad}", file=sys.stderr)
    sys.exit(1)
PY

echo "[install] DONE. Activate with:  source ${REPO_ROOT}/rlcomp.sh"
