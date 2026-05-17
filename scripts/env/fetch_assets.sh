#!/usr/bin/env bash
# ============================================================================
# RL-Compositionality — fetch model/data assets for offline runs.
#
# Run this on a networked machine. It downloads the Hugging Face assets used by
# the main rlcomp.sh pipeline into repo-local, gitignored directories. Once the
# files are present on the shared filesystem, offline machines can run without
# reaching the Hub.
#
# Usage:
#   bash scripts/env/fetch_assets.sh
#   bash scripts/env/fetch_assets.sh --check
#   bash scripts/env/fetch_assets.sh --model-only
#   bash scripts/env/fetch_assets.sh --data-only
# ============================================================================
set -euo pipefail

_SELF="${BASH_SOURCE[0]}"
REPO_ROOT="$(cd "$(dirname "${_SELF}")/../.." && pwd)"
cd "${REPO_ROOT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

MODEL_STORE_ROOT="${MODEL_STORE_ROOT:-${REPO_ROOT}/model_store}"
HF_MODEL_ROOT="${HF_MODEL_ROOT:-${MODEL_STORE_ROOT}/hf_models}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${MODEL_STORE_ROOT}/hf_cache}"
HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"
HF_XET_CACHE="${HF_XET_CACHE:-${HF_CACHE_ROOT}/xet}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/string_task}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}"

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
            echo "[assets] ERROR: ${label} must be inside the current project directory: ${REPO_ROOT}" >&2
            echo "[assets]        got: ${abs}" >&2
            exit 1
            ;;
    esac
}

MODEL_STORE_ROOT="$(_project_path MODEL_STORE_ROOT "${MODEL_STORE_ROOT}")"
HF_MODEL_ROOT="$(_project_path HF_MODEL_ROOT "${HF_MODEL_ROOT}")"
HF_CACHE_ROOT="$(_project_path HF_CACHE_ROOT "${HF_CACHE_ROOT}")"
HF_HOME="$(_project_path HF_HOME "${HF_HOME}")"
HUGGINGFACE_HUB_CACHE="$(_project_path HUGGINGFACE_HUB_CACHE "${HUGGINGFACE_HUB_CACHE}")"
HF_XET_CACHE="$(_project_path HF_XET_CACHE "${HF_XET_CACHE}")"
DATA_ROOT="$(_project_path DATA_ROOT "${DATA_ROOT}")"
TMPDIR="$(_project_path TMPDIR "${TMPDIR:-${REPO_ROOT}/.tmp}")"
TMP="$(_project_path TMP "${TMP:-${TMPDIR}}")"
TEMP="$(_project_path TEMP "${TEMP:-${TMPDIR}}")"

export HF_HOME HUGGINGFACE_HUB_CACHE HF_XET_CACHE TMPDIR TMP TEMP
mkdir -p "${TMPDIR}"

STAGE1_MODEL_REPO="${RLCOMP_STAGE1_MODEL_REPO:-weizechen/RL-Compositionality-Stage-1-Model}"
STAGE1_RFT_DATA_REPO="${RLCOMP_STAGE1_RFT_DATA_REPO:-weizechen/RL-Compositionality-Stage1-RFT-Data}"
STAGE2_LEVEL1_DATA_REPO="${RLCOMP_STAGE2_LEVEL1_DATA_REPO:-weizechen/RL-Compositionality-Stage2-RL-Level1-TrainData}"
STAGE2_LEVEL2_DATA_REPO="${RLCOMP_STAGE2_LEVEL2_DATA_REPO:-weizechen/RL-Compositionality-Stage2-RL-Level2-TrainData}"
STAGE2_EVAL_DATA_REPO="${RLCOMP_STAGE2_EVAL_DATA_REPO:-weizechen/RL-Compositionality-Stage2-RL-Level8-TestData}"

STAGE1_MODEL_DIR="${RLCOMP_STAGE1_MODEL_DIR:-${HF_MODEL_ROOT}/string-task/stage1-rft-hf}"
STAGE1_RFT_DATA_DIR="${RLCOMP_STAGE1_RFT_DATA_DIR:-${DATA_ROOT}/stage1_level1/rft_data}"
STAGE2_LEVEL1_DATA_DIR="${RLCOMP_STAGE2_LEVEL1_DATA_DIR:-${DATA_ROOT}/stage2_level1}"
STAGE2_LEVEL2_DATA_DIR="${RLCOMP_STAGE2_LEVEL2_DATA_DIR:-${DATA_ROOT}/stage2_level2}"
STAGE2_EVAL_DATA_DIR="${RLCOMP_STAGE2_EVAL_DATA_DIR:-${DATA_ROOT}/stage2_level1to8}"

STAGE1_MODEL_DIR="$(_project_path STAGE1_MODEL_DIR "${STAGE1_MODEL_DIR}")"
STAGE1_RFT_DATA_DIR="$(_project_path STAGE1_RFT_DATA_DIR "${STAGE1_RFT_DATA_DIR}")"
STAGE2_LEVEL1_DATA_DIR="$(_project_path STAGE2_LEVEL1_DATA_DIR "${STAGE2_LEVEL1_DATA_DIR}")"
STAGE2_LEVEL2_DATA_DIR="$(_project_path STAGE2_LEVEL2_DATA_DIR "${STAGE2_LEVEL2_DATA_DIR}")"
STAGE2_EVAL_DATA_DIR="$(_project_path STAGE2_EVAL_DATA_DIR "${STAGE2_EVAL_DATA_DIR}")"

FETCH_MODEL=1
FETCH_DATA=1
CHECK_ONLY=0
DRY_RUN=0
FORCE=0

usage() {
    sed -n '2,14p' "${_SELF}"
    cat <<'EOF'

Options:
  --check        Verify expected files exist; do not download.
  --dry-run      Print downloads without changing files.
  --force        Pass --force-download to hf download.
  --model-only   Download/check only the Stage 1 model.
  --data-only    Download/check only parquet datasets.
  --skip-model   Download/check datasets only.
  --skip-data    Download/check model only.
EOF
}

for arg in "$@"; do
    case "${arg}" in
        --check) CHECK_ONLY=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        --model-only) FETCH_MODEL=1; FETCH_DATA=0 ;;
        --data-only) FETCH_MODEL=0; FETCH_DATA=1 ;;
        --skip-model) FETCH_MODEL=0 ;;
        --skip-data) FETCH_DATA=0 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "[assets] unknown arg: ${arg}" >&2
            usage >&2
            exit 2
            ;;
    esac
done

HF_BIN="${HF_BIN:-$(command -v hf 2>/dev/null || true)}"
if [ "${CHECK_ONLY}" -eq 0 ] && [ "${DRY_RUN}" -eq 0 ] && [ -z "${HF_BIN}" ]; then
    echo "[assets] ERROR: hf CLI not found." >&2
    echo "[assets]        Install it on the networked cache builder:" >&2
    echo "[assets]        curl -LsSf https://hf.co/cli/install.sh | bash -s" >&2
    exit 1
fi

_download() {
    local kind="$1"
    local repo="$2"
    local dest="$3"

    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '[assets] dry-run: hf download %q --type %q --local-dir %q --cache-dir %q --max-workers %q' \
            "${repo}" "${kind}" "${dest}" "${HUGGINGFACE_HUB_CACHE}" "${HF_MAX_WORKERS}"
        if [ "${FORCE}" -eq 1 ]; then
            printf ' --force-download'
        fi
        printf '\n'
        return 0
    fi

    mkdir -p "${dest}" "${HUGGINGFACE_HUB_CACHE}" "${HF_XET_CACHE}"

    echo "[assets] downloading ${kind}: ${repo}"
    echo "[assets]   -> ${dest}"
    if [ "${FORCE}" -eq 1 ]; then
        "${HF_BIN}" download "${repo}" --type "${kind}" \
            --local-dir "${dest}" \
            --cache-dir "${HUGGINGFACE_HUB_CACHE}" \
            --max-workers "${HF_MAX_WORKERS}" \
            --force-download
    else
        "${HF_BIN}" download "${repo}" --type "${kind}" \
            --local-dir "${dest}" \
            --cache-dir "${HUGGINGFACE_HUB_CACHE}" \
            --max-workers "${HF_MAX_WORKERS}"
    fi
}

_copy_if_needed() {
    local src="$1"
    local dst="$2"
    if [ ! -f "${dst}" ] && [ -f "${src}" ]; then
        cp -p "${src}" "${dst}"
        echo "[assets] created compatibility copy: ${dst}"
    fi
}

_normalize_dataset_names() {
    _copy_if_needed "${STAGE2_LEVEL1_DATA_DIR}/forward_train.parquet" \
                    "${STAGE2_LEVEL1_DATA_DIR}/train.parquet"
    _copy_if_needed "${STAGE2_LEVEL1_DATA_DIR}/train.parquet" \
                    "${STAGE2_LEVEL1_DATA_DIR}/forward_train.parquet"
    _copy_if_needed "${STAGE2_LEVEL2_DATA_DIR}/forward_train.parquet" \
                    "${STAGE2_LEVEL2_DATA_DIR}/train.parquet"
    _copy_if_needed "${STAGE2_LEVEL2_DATA_DIR}/train.parquet" \
                    "${STAGE2_LEVEL2_DATA_DIR}/forward_train.parquet"
    _copy_if_needed "${STAGE2_EVAL_DATA_DIR}/test.parquet" \
                    "${STAGE2_EVAL_DATA_DIR}/forward_test.parquet"
    _copy_if_needed "${STAGE2_EVAL_DATA_DIR}/forward_test.parquet" \
                    "${STAGE2_EVAL_DATA_DIR}/test.parquet"
}

_check_file() {
    local path="$1"
    if [ -f "${path}" ]; then
        echo "  OK  ${path#${REPO_ROOT}/}"
    else
        echo "  MISS ${path#${REPO_ROOT}/}"
        return 1
    fi
}

_check_assets() {
    local missing=0
    echo "[assets] checking local assets"
    if [ "${FETCH_MODEL}" -eq 1 ]; then
        _check_file "${STAGE1_MODEL_DIR}/config.json" || missing=$((missing + 1))
        _check_file "${STAGE1_MODEL_DIR}/tokenizer.json" || missing=$((missing + 1))
        if ! ls "${STAGE1_MODEL_DIR}"/model-*.safetensors >/dev/null 2>&1; then
            echo "  MISS ${STAGE1_MODEL_DIR#${REPO_ROOT}/}/model-*.safetensors"
            missing=$((missing + 1))
        else
            echo "  OK  ${STAGE1_MODEL_DIR#${REPO_ROOT}/}/model-*.safetensors"
        fi
    fi
    if [ "${FETCH_DATA}" -eq 1 ]; then
        _check_file "${STAGE1_RFT_DATA_DIR}/train.parquet" || missing=$((missing + 1))
        _check_file "${STAGE1_RFT_DATA_DIR}/test.parquet" || missing=$((missing + 1))
        _check_file "${STAGE2_LEVEL1_DATA_DIR}/train.parquet" || missing=$((missing + 1))
        _check_file "${STAGE2_LEVEL1_DATA_DIR}/forward_train.parquet" || missing=$((missing + 1))
        _check_file "${STAGE2_LEVEL2_DATA_DIR}/train.parquet" || missing=$((missing + 1))
        _check_file "${STAGE2_LEVEL2_DATA_DIR}/forward_train.parquet" || missing=$((missing + 1))
        _check_file "${STAGE2_EVAL_DATA_DIR}/forward_test.parquet" || missing=$((missing + 1))
        _check_file "${STAGE2_EVAL_DATA_DIR}/test.parquet" || missing=$((missing + 1))
    fi

    if [ "${missing}" -ne 0 ]; then
        echo "[assets] ${missing} required asset check(s) failed." >&2
        return 1
    fi
    echo "[assets] all selected assets are present."
}

echo "[assets] repo root : ${REPO_ROOT}"
echo "[assets] model dir : ${STAGE1_MODEL_DIR}"
echo "[assets] data root : ${DATA_ROOT}"
echo "[assets] HF cache  : ${HUGGINGFACE_HUB_CACHE}"

if [ "${CHECK_ONLY}" -eq 1 ]; then
    _check_assets
    exit $?
fi

if [ "${FETCH_MODEL}" -eq 1 ]; then
    _download model "${STAGE1_MODEL_REPO}" "${STAGE1_MODEL_DIR}"
fi

if [ "${FETCH_DATA}" -eq 1 ]; then
    _download dataset "${STAGE1_RFT_DATA_REPO}" "${STAGE1_RFT_DATA_DIR}"
    _download dataset "${STAGE2_LEVEL1_DATA_REPO}" "${STAGE2_LEVEL1_DATA_DIR}"
    _download dataset "${STAGE2_LEVEL2_DATA_REPO}" "${STAGE2_LEVEL2_DATA_DIR}"
    _download dataset "${STAGE2_EVAL_DATA_REPO}" "${STAGE2_EVAL_DATA_DIR}"
    if [ "${DRY_RUN}" -eq 0 ]; then
        _normalize_dataset_names
    fi
fi

if [ "${DRY_RUN}" -eq 0 ]; then
    _check_assets
fi
