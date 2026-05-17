#!/usr/bin/env bash
# ============================================================================
# RL-Compositionality — env health diagnostic.
#
# Non-destructive. Reports the state of:
#   - standalone Python (.python/cpython-3.10.*)
#   - venv (.venv-rlcomp/)
#   - wheel cache (wheels/)
#   - critical imports (torch, vllm, flash-attn, xformers, verl, ...)
#   - CUDA / libcuda / GPU
#   - model/data assets used by the unified pipeline
#   - disk, shared caches, write permissions
#
# Exits 0 if everything looks good. Non-zero if any hard failure found.
# ============================================================================
set -uo pipefail

_SELF="${BASH_SOURCE[0]}"
REPO_ROOT="$(cd "$(dirname "${_SELF}")/../.." && pwd)"
cd "${REPO_ROOT}"

PY_ROOT="${REPO_ROOT}/.python"
VENV="${REPO_ROOT}/.venv-rlcomp"
WHEELS="${REPO_ROOT}/wheels"
REQ="${REPO_ROOT}/requirements.lock.txt"
MODEL_STORE_ROOT="${MODEL_STORE_ROOT:-${REPO_ROOT}/model_store}"
HF_MODEL_ROOT="${HF_MODEL_ROOT:-${MODEL_STORE_ROOT}/hf_models}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/string_task}"
CHECK_PREPARED="${RLCOMP_DOCTOR_CHECK_PREPARED:-0}"

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
            echo "[doctor] ERROR: ${label} must be inside the current project directory: ${REPO_ROOT}" >&2
            echo "[doctor]        got: ${abs}" >&2
            exit 1
            ;;
    esac
}

PY_ROOT="$(_project_path PY_ROOT "${PY_ROOT}")"
VENV="$(_project_path VENV "${VENV}")"
WHEELS="$(_project_path WHEELS "${WHEELS}")"
REQ="$(_project_path REQ "${REQ}")"
MODEL_STORE_ROOT="$(_project_path MODEL_STORE_ROOT "${MODEL_STORE_ROOT}")"
HF_MODEL_ROOT="$(_project_path HF_MODEL_ROOT "${HF_MODEL_ROOT}")"
DATA_ROOT="$(_project_path DATA_ROOT "${DATA_ROOT}")"

usage() {
    sed -n '2,15p' "${_SELF}"
    cat <<'EOF'

Options:
  --check-prepared    Treat generated new-ops parquet files as required.
EOF
}

for arg in "$@"; do
    case "${arg}" in
        --check-prepared) CHECK_PREPARED=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "[doctor] unknown arg: ${arg}" >&2
            usage >&2
            exit 2
            ;;
    esac
done

fails=0
warns=0

pass() { printf "  \033[32m✔\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✘\033[0m %s\n" "$*"; fails=$((fails + 1)); }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; warns=$((warns + 1)); }

echo "[doctor] RL-Compositionality env health check"
echo "[doctor] repo root: ${REPO_ROOT}"
echo ""

# ----- 1. standalone Python -----
echo "[1/7] standalone Python"
PY_CANDIDATE="$(ls -d "${PY_ROOT}"/cpython-3.10.*-linux-x86_64-gnu 2>/dev/null | sort | tail -1)"
if [ -n "${PY_CANDIDATE}" ] && [ -x "${PY_CANDIDATE}/bin/python3.10" ]; then
    pass "installed at ${PY_CANDIDATE}"
    pass "$("${PY_CANDIDATE}/bin/python3.10" -V)"
else
    fail "not found at ${PY_ROOT}/cpython-3.10.*-linux-x86_64-gnu"
    fail "run 'bash rlcomp.sh bootstrap-cache' on a networked machine"
fi
echo ""

# ----- 2. venv -----
echo "[2/7] virtual environment"
if [ -f "${VENV}/bin/python" ]; then
    pass "venv exists at ${VENV}"
    # Check that it's hermetic (pyvenv.cfg's home should point into .python/)
    if grep -q "^home = ${REPO_ROOT}/.python/" "${VENV}/pyvenv.cfg" 2>/dev/null; then
        pass "pyvenv.cfg home points inside repo (hermetic)"
    else
        warn "pyvenv.cfg home does NOT point inside repo — may be fragile"
        warn "  $(grep '^home' "${VENV}/pyvenv.cfg" 2>/dev/null || echo 'missing pyvenv.cfg')"
    fi
    # Quick sanity: is python executable working?
    if "${VENV}/bin/python" -c 'import sys; sys.exit(0)' 2>/dev/null; then
        pass "$("${VENV}/bin/python" -V)"
    else
        fail "venv python is broken"
    fi
else
    fail "venv missing at ${VENV}"
    fail "run 'bash rlcomp.sh install' to rebuild"
fi
echo ""

# ----- 3. wheel cache -----
echo "[3/7] offline wheel cache"
if [ -d "${WHEELS}" ] && [ -n "$(ls -A "${WHEELS}" 2>/dev/null)" ]; then
    pass "wheels/ has $(ls "${WHEELS}" | wc -l) files ($(du -sh "${WHEELS}" | awk '{print $1}'))"
else
    warn "wheels/ is empty — cannot rebuild offline"
    warn "run 'bash rlcomp.sh bootstrap-cache' on a networked machine"
fi
if [ -f "${REQ}" ]; then
    pass "requirements.lock.txt: $(wc -l <"${REQ}") lines"
else
    warn "requirements.lock.txt missing"
fi
echo ""

# ----- 4. critical imports -----
echo "[4/7] critical imports"
if [ -f "${VENV}/bin/python" ]; then
    "${VENV}/bin/python" - <<'PY' 2>&1 | sed 's/^/  /'
import importlib, sys
mods = ["torch", "vllm", "flash_attn", "xformers", "triton", "ray",
        "transformers", "datasets", "hydra", "wandb", "verl", "flashinfer"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "n/a")
        print(f"\033[32m✔\033[0m {m:<16} {ver}")
    except Exception as e:
        print(f"\033[31m✘\033[0m {m:<16} FAIL: {type(e).__name__}: {e}")
        bad.append(m)
sys.exit(1 if bad else 0)
PY
    if [ $? -ne 0 ]; then fails=$((fails + 1)); fi
    mkdir -p "${REPO_ROOT}/.tmp"
    pip_check_log="$(mktemp "${REPO_ROOT}/.tmp/rlcomp-pip-check.XXXXXX")"
    if "${VENV}/bin/python" -m pip check >"${pip_check_log}" 2>&1; then
        pass "pip dependency check passed"
    else
        warn "pip dependency check reported issues:"
        sed 's/^/    /' "${pip_check_log}"
    fi
    rm -f "${pip_check_log}"
else
    fail "skipped (venv missing)"
fi
echo ""

# ----- 5. CUDA / GPU -----
echo "[5/7] CUDA / GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    pass "nvidia-smi: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
    pass "GPU count: ${gpu_count}"
else
    warn "nvidia-smi not found; no GPU check possible"
fi
# libcuda.so.1 discoverable?
libcuda_found=0
for d in /usr/local/nvidia/lib64 /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64; do
    if [ -f "$d/libcuda.so.1" ]; then
        pass "libcuda.so.1: $d/libcuda.so.1"
        libcuda_found=1
        break
    fi
done
if [ "${libcuda_found}" -eq 0 ]; then
    warn "libcuda.so.1 not in standard dirs; Triton/vLLM may fail to JIT"
fi
echo ""

# ----- 6. paths / disk -----
echo "[6/7] paths & disk"
df -h "${REPO_ROOT}" 2>/dev/null | awk 'NR==2 {printf "  disk: %s used of %s (%s free)\n", $3, $2, $4}'
for p in model_store .tmp wheels wandb outputs results; do
    if [ -d "${REPO_ROOT}/${p}" ]; then
        pass "${p}/  $(du -sh "${REPO_ROOT}/${p}" 2>/dev/null | awk '{print $1}')"
    else
        warn "${p}/ not present (will be created on demand)"
    fi
done
# Writability
if [ -w "${REPO_ROOT}" ]; then
    pass "repo is writable"
else
    fail "repo is NOT writable — training will fail"
fi
echo ""

# ----- 7. model/data assets -----
echo "[7/7] model/data assets"
check_asset_file() {
    local path="$1"
    local required="${2:-1}"
    if [ -f "${path}" ]; then
        pass "${path#${REPO_ROOT}/}"
    elif [ "${required}" -eq 1 ]; then
        fail "missing ${path#${REPO_ROOT}/}"
    else
        warn "missing ${path#${REPO_ROOT}/}"
    fi
}
STAGE1_MODEL="${HF_MODEL_ROOT}/string-task/stage1-rft-hf"
check_asset_file "${STAGE1_MODEL}/config.json" 1
check_asset_file "${STAGE1_MODEL}/tokenizer.json" 1
if ls "${STAGE1_MODEL}"/model-*.safetensors >/dev/null 2>&1; then
    pass "${STAGE1_MODEL#${REPO_ROOT}/}/model-*.safetensors"
else
    fail "missing ${STAGE1_MODEL#${REPO_ROOT}/}/model-*.safetensors"
fi
check_asset_file "${DATA_ROOT}/stage1_level1/rft_data/train.parquet" 1
check_asset_file "${DATA_ROOT}/stage1_level1/rft_data/test.parquet" 1
check_asset_file "${DATA_ROOT}/stage2_level1/train.parquet" 1
check_asset_file "${DATA_ROOT}/stage2_level1/forward_train.parquet" 1
check_asset_file "${DATA_ROOT}/stage2_level2/train.parquet" 1
check_asset_file "${DATA_ROOT}/stage2_level2/forward_train.parquet" 1
check_asset_file "${DATA_ROOT}/stage2_level1to8/test.parquet" 1
check_asset_file "${DATA_ROOT}/stage2_level1to8/forward_test.parquet" 1
if [ "${CHECK_PREPARED}" -eq 1 ]; then
    echo "  prepared pipeline data: required (--check-prepared)"
    prepared_required=1
else
    echo "  prepared pipeline data: advisory (run 'bash rlcomp.sh prepare' before stage1/stage2/infer)"
    prepared_required=0
fi
check_asset_file "${DATA_ROOT}/exp_new_ops/stage1_newops_train.parquet" "${prepared_required}"
check_asset_file "${DATA_ROOT}/exp_new_ops/stage1_newops_test.parquet" "${prepared_required}"
check_asset_file "${DATA_ROOT}/exp_new_ops/stage2_level1to8_newops_test.parquet" "${prepared_required}"
check_asset_file "${DATA_ROOT}/exp_new_ops/complete_atomic_new_d1.parquet" "${prepared_required}"
check_asset_file "${DATA_ROOT}/exp_new_ops/complete_linear_new_d3.parquet" "${prepared_required}"
check_asset_file "${DATA_ROOT}/exp_new_ops/complete_branch_new_d4.parquet" "${prepared_required}"
check_asset_file "${DATA_ROOT}/exp_new_ops/complete_mixed_oldnew_d4.parquet" "${prepared_required}"
unset -f check_asset_file
echo ""

# ----- summary -----
echo "[doctor] summary: ${fails} fail(s), ${warns} warn(s)"
if [ "${fails}" -gt 0 ]; then
    exit 1
fi
exit 0
