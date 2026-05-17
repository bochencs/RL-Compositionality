#!/usr/bin/env bash
# ============================================================================
# RL-Compositionality — one-time online bootstrap of the offline cache.
#
# Downloads a standalone Python interpreter and every pip wheel the project
# needs into the repo (under `.python/` and `wheels/`). After this has run
# ONCE from a machine that reaches PyPI + github + pytorch.org, the repo is
# self-sufficient and `install_hermetic.sh` can (re)build `.venv-rlcomp/`
# from those files on any machine with no network.
#
# Re-running is safe — missing files are re-fetched, existing wheels are
# kept.
#
# Usage:
#   bash scripts/env/bootstrap_offline_cache.sh
#   bash scripts/env/bootstrap_offline_cache.sh --check
# ============================================================================
set -euo pipefail

_SELF="${BASH_SOURCE[0]}"
REPO_ROOT="$(cd "$(dirname "${_SELF}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_ROOT="${REPO_ROOT}/.python"
WHEELS_DIR="${REPO_ROOT}/wheels"
REQ_FILE="${REPO_ROOT}/requirements.lock.txt"

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
            echo "[cache] ERROR: ${label} must be inside the current project directory: ${REPO_ROOT}" >&2
            echo "[cache]        got: ${abs}" >&2
            exit 1
            ;;
    esac
}

PYTHON_ROOT="$(_project_path PYTHON_ROOT "${PYTHON_ROOT}")"
WHEELS_DIR="$(_project_path WHEELS_DIR "${WHEELS_DIR}")"
REQ_FILE="$(_project_path REQ_FILE "${REQ_FILE}")"
UV_CACHE_DIR="$(_project_path UV_CACHE_DIR "${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}")"
PIP_CACHE_DIR="$(_project_path PIP_CACHE_DIR "${PIP_CACHE_DIR:-${REPO_ROOT}/.pip-cache}")"
TMPDIR="$(_project_path TMPDIR "${TMPDIR:-${REPO_ROOT}/.tmp}")"
TMP="$(_project_path TMP "${TMP:-${TMPDIR}}")"
TEMP="$(_project_path TEMP "${TEMP:-${TMPDIR}}")"
export UV_CACHE_DIR PIP_CACHE_DIR TMPDIR TMP TEMP
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${TMPDIR}"

CHECK_ONLY=0
usage() {
    sed -n '2,17p' "${_SELF}"
    cat <<'EOF'

Options:
  --check    Verify .python/, wheels/, and requirements.lock.txt; do not download.
EOF
}

for arg in "$@"; do
    case "${arg}" in
        --check) CHECK_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "[cache] unknown arg: ${arg}" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ "${CHECK_ONLY}" -eq 1 ]; then
    errors=0
    echo "[cache] checking offline cache inputs"
    if ls -d "${PYTHON_ROOT}"/cpython-3.10.*-linux-x86_64-gnu >/dev/null 2>&1; then
        echo "  OK  .python/cpython-3.10.*-linux-x86_64-gnu"
    else
        echo "  MISS .python/cpython-3.10.*-linux-x86_64-gnu"
        errors=$((errors + 1))
    fi
    if [ -d "${WHEELS_DIR}" ] && [ -n "$(find "${WHEELS_DIR}" -maxdepth 1 -name '*.whl' -print -quit 2>/dev/null)" ]; then
        echo "  OK  wheels ($(find "${WHEELS_DIR}" -maxdepth 1 -name '*.whl' | wc -l) wheel files)"
    else
        echo "  MISS wheels/*.whl"
        errors=$((errors + 1))
    fi
    if [ -f "${REQ_FILE}" ]; then
        echo "  OK  requirements.lock.txt"
    else
        echo "  MISS requirements.lock.txt"
        errors=$((errors + 1))
    fi
    exit "${errors}"
fi

UV_BIN="${UV_BIN:-$(command -v uv 2>/dev/null || echo "${HOME}/.local/bin/uv")}"
if [ ! -x "${UV_BIN}" ]; then
    echo "[cache] ERROR: uv not found at ${UV_BIN}" >&2
    echo "[cache]        Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if [ ! -f "${REQ_FILE}" ]; then
    echo "[cache] ERROR: ${REQ_FILE} missing. This is a versioned artifact." >&2
    exit 1
fi

# ---------- 1. ensure standalone Python 3.10 ----------
mkdir -p "${PYTHON_ROOT}"
export UV_PYTHON_INSTALL_DIR="${PYTHON_ROOT}"
if ! ls -d "${PYTHON_ROOT}"/cpython-3.10.*-linux-x86_64-gnu >/dev/null 2>&1; then
    echo "[cache] installing standalone Python 3.10 -> ${PYTHON_ROOT}"
    "${UV_BIN}" python install 3.10 --install-dir "${PYTHON_ROOT}"
else
    echo "[cache] standalone Python 3.10 already present under ${PYTHON_ROOT}"
fi
PYTHON_BIN="$(ls -d "${PYTHON_ROOT}"/cpython-3.10.*-linux-x86_64-gnu 2>/dev/null | sort | tail -1)/bin/python3.10"
echo "[cache] python: ${PYTHON_BIN} ($("${PYTHON_BIN}" -V))"

# ---------- 2. prime a temp venv that we'll use to download wheels ----------
TMP_VENV="${REPO_ROOT}/.venv-wheelbuilder"
if [ ! -f "${TMP_VENV}/bin/python" ]; then
    echo "[cache] creating temporary wheel-builder venv at ${TMP_VENV}"
    "${PYTHON_BIN}" -m venv --symlinks "${TMP_VENV}"
fi
# shellcheck disable=SC1091
. "${TMP_VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel packaging 2>&1 | tail -3

# ---------- 3. download every wheel the project needs ----------
# Huawei ModelArts pre-sets PIP_INDEX_URL to their mirror which often lacks
# the exact pinned versions in requirements.lock.txt — use public PyPI.
# PIP_NO_CACHE_DIR=1 makes the resolver slow (re-downloads metadata); point
# the cache at the repo so subsequent runs are fast.
unset PIP_INDEX_URL PIP_TRUSTED_HOST PIP_NO_CACHE_DIR MA_PIP_URL MA_PIP_HOST
mkdir -p "${WHEELS_DIR}"
echo "[cache] downloading wheels to ${WHEELS_DIR} (via public PyPI + pytorch.org)..."
python -m pip download \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --cache-dir "${PIP_CACHE_DIR}" \
    -r "${REQ_FILE}" \
    -d "${WHEELS_DIR}" \
    --only-binary=:all: \
    --no-build-isolation
echo "[cache] wheel count: $(ls "${WHEELS_DIR}" | wc -l)"
echo "[cache] wheel total: $(du -sh "${WHEELS_DIR}" | awk '{print $1}')"

# ---------- 4. also ensure pip/setuptools/wheel themselves are in the cache ----------
# install_hermetic.sh wants to install these before anything else.
python -m pip download \
    pip setuptools wheel packaging \
    -d "${WHEELS_DIR}" \
    --only-binary=:all: 2>&1 | tail -5

deactivate

echo ""
echo "[cache] DONE. Repo is now self-sufficient:"
echo "  ${PYTHON_ROOT}/cpython-3.10.*/   (standalone Python)"
echo "  ${WHEELS_DIR}/                    (offline wheel cache)"
echo "  ${REQ_FILE}                       (pinned versions)"
echo ""
echo "You can delete .venv-wheelbuilder/ if disk is tight:"
echo "  rm -rf ${TMP_VENV}"
echo ""
echo "On any shared-FS machine:"
echo "  bash rlcomp.sh install    # rebuilds .venv-rlcomp from cache (no network)"
