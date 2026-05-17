#!/usr/bin/env bash
# ============================================================================
# RL-Compositionality — freeze current venv into an offline-reinstallable state.
#
# Walks .venv-rlcomp/, writes:
#   - requirements.lock.txt       (pip freeze --exclude-editable with URL
#                                  substitutions for flash-attn etc.)
#   - wheels/                     (all pip wheels needed to reinstall)
#
# After freeze, `install_hermetic.sh` can rebuild from these files without
# network access.
#
# Usage:
#   bash scripts/env/freeze.sh              # update both
#   bash scripts/env/freeze.sh --lock-only  # just rewrite requirements.lock.txt
# ============================================================================
set -euo pipefail

_SELF="${BASH_SOURCE[0]}"
REPO_ROOT="$(cd "$(dirname "${_SELF}")/../.." && pwd)"
cd "${REPO_ROOT}"

VENV="${REPO_ROOT}/.venv-rlcomp"
WHEELS="${REPO_ROOT}/wheels"
REQ="${REPO_ROOT}/requirements.lock.txt"
PIP_CACHE="${REPO_ROOT}/.pip-cache"
TMPDIR="${REPO_ROOT}/.tmp"
TMP="${TMPDIR}"
TEMP="${TMPDIR}"
export TMPDIR TMP TEMP PIP_CACHE_DIR="${PIP_CACHE}"
mkdir -p "${TMPDIR}" "${PIP_CACHE}"

LOCK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --lock-only) LOCK_ONLY=1 ;;
        -h|--help) sed -n '2,15p' "$_SELF"; exit 0 ;;
        *) echo "[freeze] unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [ ! -x "${VENV}/bin/pip" ]; then
    echo "[freeze] ERROR: no pip in ${VENV}. Did you run 'rlcomp.sh install'?" >&2
    exit 1
fi

echo "[freeze] writing ${REQ}..."
"${VENV}/bin/pip" freeze --exclude-editable > "${REQ}.tmp"

# Substitute: flash-attn git ref -> prebuilt wheel URL, prepend torch index.
"${VENV}/bin/python" <<'PY'
import re, pathlib
p = pathlib.Path("requirements.lock.txt.tmp")
lines = p.read_text().splitlines()
out = ["--extra-index-url https://download.pytorch.org/whl/cu124", ""]
for line in lines:
    if line.startswith("flash_attn @") or line.startswith("flash-attn @"):
        # The upstream prebuilt flash-attn wheels on github releases have
        # historically had ABI-naming mismatches (cxx11abiFALSE label vs
        # cxx11 symbols in the .so). The reliable approach is to use the
        # locally-packed wheel produced by `wheel pack` on the installed
        # flash_attn, stored in `wheels/`. A simple version pin resolves
        # against that local wheel during offline install.
        out.append("flash_attn==2.8.0.post2")
    else:
        out.append(line)
p.with_name("requirements.lock.txt").write_text("\n".join(out) + "\n")
p.unlink()
print(f"[freeze] wrote requirements.lock.txt ({len(out)} lines)")
PY

if [ "${LOCK_ONLY}" -eq 1 ]; then
    exit 0
fi

echo "[freeze] downloading wheels to ${WHEELS}..."
mkdir -p "${WHEELS}"
# Unset huawei ModelArts mirror env so we hit public PyPI for missing wheels.
unset PIP_INDEX_URL PIP_TRUSTED_HOST
"${VENV}/bin/pip" download \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --cache-dir "${PIP_CACHE}" \
    -r "${REQ}" \
    -d "${WHEELS}" \
    --only-binary=:all: \
    --no-build-isolation

# Also pip/setuptools/wheel themselves so install_hermetic.sh can bootstrap.
"${VENV}/bin/pip" download \
    --index-url https://pypi.org/simple \
    --cache-dir "${PIP_CACHE}" \
    pip setuptools wheel packaging \
    --only-binary=:all: \
    -d "${WHEELS}" >/dev/null

echo "[freeze] DONE."
echo "  wheels count : $(ls "${WHEELS}" | wc -l)"
echo "  wheels size  : $(du -sh "${WHEELS}" | awk '{print $1}')"
echo "  lock file    : ${REQ}"
