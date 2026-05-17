#!/usr/bin/env bash
# ============================================================================
# RL-Compositionality — verify/package the repo-local offline bundle.
#
# The bundle is intentionally file-based: all heavy runtime assets stay in
# gitignored directories under the repo (or shared filesystem). This script can
# either validate that the required directories exist, write a manifest, or
# create a tar archive for transfer to another offline machine. The archive
# contains the tracked repo files plus the gitignored offline assets.
#
# Usage:
#   bash scripts/env/package_offline_bundle.sh --check
#   bash scripts/env/package_offline_bundle.sh --manifest results/offline_manifest.txt
#   bash scripts/env/package_offline_bundle.sh --tar results/rlcomp-offline.tar
# ============================================================================
set -euo pipefail

_SELF="${BASH_SOURCE[0]}"
REPO_ROOT="$(cd "$(dirname "${_SELF}")/../.." && pwd)"
cd "${REPO_ROOT}"

CHECK_ONLY=0
MANIFEST_PATH=""
TAR_PATH=""
INCLUDE_VENV=0

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
            echo "[bundle] ERROR: ${label} must be inside the current project directory: ${REPO_ROOT}" >&2
            echo "[bundle]        got: ${abs}" >&2
            exit 1
            ;;
    esac
}

usage() {
    sed -n '2,15p' "${_SELF}"
    cat <<'EOF'

Options:
  --check             Verify required offline files; do not write outputs.
  --manifest PATH     Write a file manifest with sizes.
  --tar PATH          Create a tar archive containing tracked source plus assets.
  --with-venv         Include .venv-rlcomp in checks and outputs.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --check) CHECK_ONLY=1; shift ;;
        --manifest)
            MANIFEST_PATH="${2:-}"
            if [ -z "${MANIFEST_PATH}" ]; then
                echo "[bundle] --manifest requires a path" >&2
                exit 2
            fi
            shift 2
            ;;
        --tar)
            TAR_PATH="${2:-}"
            if [ -z "${TAR_PATH}" ]; then
                echo "[bundle] --tar requires a path" >&2
                exit 2
            fi
            shift 2
            ;;
        --with-venv) INCLUDE_VENV=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "[bundle] unknown arg: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -n "${MANIFEST_PATH}" ]; then
    MANIFEST_PATH="$(_project_path MANIFEST_PATH "${MANIFEST_PATH}")"
fi
if [ -n "${TAR_PATH}" ]; then
    TAR_PATH="$(_project_path TAR_PATH "${TAR_PATH}")"
fi

required_paths=(
    "requirements.lock.txt"
    "pip.conf"
    ".python"
    "wheels"
    "model_store/hf_models/string-task/stage1-rft-hf"
    "data/string_task/stage1_level1/rft_data/train.parquet"
    "data/string_task/stage1_level1/rft_data/test.parquet"
    "data/string_task/stage2_level1/train.parquet"
    "data/string_task/stage2_level1/forward_train.parquet"
    "data/string_task/stage2_level2/train.parquet"
    "data/string_task/stage2_level2/forward_train.parquet"
    "data/string_task/stage2_level1to8/test.parquet"
    "data/string_task/stage2_level1to8/forward_test.parquet"
)

if [ "${INCLUDE_VENV}" -eq 1 ]; then
    required_paths+=(".venv-rlcomp")
fi

bundle_paths=(
    "requirements.lock.txt"
    "pip.conf"
    ".python"
    "wheels"
    "model_store"
    "data/string_task"
)

if [ "${INCLUDE_VENV}" -eq 1 ]; then
    bundle_paths+=(".venv-rlcomp")
fi

failures=0
echo "[bundle] checking offline bundle inputs"
for p in "${required_paths[@]}"; do
    if [ -e "${REPO_ROOT}/${p}" ]; then
        size="$(du -sh "${REPO_ROOT}/${p}" 2>/dev/null | awk '{print $1}')"
        echo "  OK  ${p} ${size:+(${size})}"
    else
        echo "  MISS ${p}"
        failures=$((failures + 1))
    fi
done

if [ "${failures}" -ne 0 ]; then
    echo "[bundle] ${failures} required path(s) missing." >&2
    echo "[bundle] Run 'bash rlcomp.sh bootstrap-cache' and 'bash rlcomp.sh fetch-assets' on a networked machine first." >&2
    exit 1
fi

if [ -n "${MANIFEST_PATH}" ]; then
    mkdir -p "$(dirname "${MANIFEST_PATH}")"
    {
        echo "# RL-Compositionality offline bundle manifest"
        echo "repo_root=${REPO_ROOT}"
        echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
        echo ""
        for p in "${bundle_paths[@]}"; do
            if [ -e "${REPO_ROOT}/${p}" ]; then
                du -sh "${REPO_ROOT}/${p}"
            fi
        done
        echo ""
        echo "# Selected files"
        for p in "${bundle_paths[@]}"; do
            if [ -d "${REPO_ROOT}/${p}" ]; then
                find "${REPO_ROOT}/${p}" -type f -printf "${p}/%P\t%s\n" | sort
            elif [ -f "${REPO_ROOT}/${p}" ]; then
                printf '%s\t%s\n' "${p}" "$(wc -c <"${REPO_ROOT}/${p}")"
            fi
        done
    } > "${MANIFEST_PATH}"
    echo "[bundle] wrote manifest: ${MANIFEST_PATH}"
fi

if [ -n "${TAR_PATH}" ]; then
    mkdir -p "$(dirname "${TAR_PATH}")"
    echo "[bundle] creating tar archive: ${TAR_PATH}"
    mkdir -p "${REPO_ROOT}/.tmp"
    tmp_list="$(mktemp "${REPO_ROOT}/.tmp/rlcomp-bundle-list.XXXXXX")"
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git ls-files > "${tmp_list}"
    else
        : > "${tmp_list}"
    fi
    printf '%s\n' "${bundle_paths[@]}" >> "${tmp_list}"
    tar \
        --exclude='results/pipeline_runs' \
        --exclude='wandb' \
        --exclude='.git' \
        -cf "${TAR_PATH}" \
        --files-from "${tmp_list}"
    rm -f "${tmp_list}"
    echo "[bundle] tar size: $(du -sh "${TAR_PATH}" | awk '{print $1}')"
fi

if [ "${CHECK_ONLY}" -eq 1 ] || { [ -z "${MANIFEST_PATH}" ] && [ -z "${TAR_PATH}" ]; }; then
    echo "[bundle] --check passed; no bundle file written."
fi
