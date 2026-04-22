#!/usr/bin/env bash
# ============================================================================
# RL-Compositionality 环境安装脚本
#
# 假设基础镜像: python:3.12-slim-bookworm (或其它干净的 Python 3.12 镜像)
# 前提条件   : 宿主机 NVIDIA 驱动 >= 550 (支持 CUDA 12.4 runtime);
#              torch pip wheel 会自带 CUDA 运行时库,无需系统 CUDA toolkit。
# 用法       :
#     chmod +x scripts/install_env_py312.sh
#     ./scripts/install_env_py312.sh
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 可配置项
# ---------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv-rlcomp}"
TORCH_VER="2.6.0"
TORCHVISION_VER="0.21.0"
TORCHAUDIO_VER="2.6.0"
CUDA_TAG="cu124"
VLLM_VER="0.8.2"
FLASH_ATTN_VER="2.8.0.post2"
FLASHINFER_VER="0.5.3"
TENSORDICT_VER="0.8.3"

# ---------------------------------------------------------------------------
# 0. 系统级依赖 (python:3.12-slim 里没有 git/gcc/ninja)
# ---------------------------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        build-essential ninja-build pkg-config \
        libnuma-dev libibverbs-dev
    rm -rf /var/lib/apt/lists/*
fi

# ---------------------------------------------------------------------------
# 1. 创建 venv & 升级基础工具
# ---------------------------------------------------------------------------
python3.12 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel packaging

# ---------------------------------------------------------------------------
# 2. PyTorch 2.6.0 (CUDA 12.4 wheel, cxx11_abi=False)
#    triton 3.2.0 会作为 torch 依赖自动装上
# ---------------------------------------------------------------------------
pip install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    "torch==${TORCH_VER}" \
    "torchvision==${TORCHVISION_VER}" \
    "torchaudio==${TORCHAUDIO_VER}"

# ---------------------------------------------------------------------------
# 3. vLLM 0.8.2 (rollout 引擎;会带 xformers / outlines / tokenizers 等)
#    --no-deps 先禁用会覆盖 torch 的版本拉扯,然后显式补齐需要的子依赖
# ---------------------------------------------------------------------------
pip install "vllm==${VLLM_VER}"

# FlashInfer (vllm 0.8.x 对应 0.5.x)
pip install "flashinfer-python==${FLASHINFER_VER}"

# ---------------------------------------------------------------------------
# 4. FlashAttention-2 预编译 wheel
#    命名约定: flash_attn-<ver>+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
# ---------------------------------------------------------------------------
FA_WHL="flash_attn-${FLASH_ATTN_VER}+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VER}/${FA_WHL}"
pip install --no-build-isolation "${FA_URL}"

# ---------------------------------------------------------------------------
# 5. RL / 分布式 / HF 生态 / 数据处理
# ---------------------------------------------------------------------------
pip install \
    "ray[default]>=2.10" \
    accelerate \
    peft \
    transformers \
    datasets \
    "tensordict==${TENSORDICT_VER}" \
    torchdata \
    "numpy<2.0" \
    pandas \
    "pyarrow>=15.0.0" \
    dill \
    hydra-core \
    codetiming \
    pybind11 \
    pylatexenc \
    wandb \
    tabulate \
    reasoning-gym

# ---------------------------------------------------------------------------
# 6. 可选 GPU 加速组件
# ---------------------------------------------------------------------------
pip install liger-kernel bitsandbytes

# ---------------------------------------------------------------------------
# 7. 开发 / 测试 / 评测 extras
# ---------------------------------------------------------------------------
pip install pytest yapf py-spy math-verify

# ---------------------------------------------------------------------------
# 8. 安装本仓库 verl 包 (editable, 不重装依赖)
# ---------------------------------------------------------------------------
cd "${REPO_DIR}"
pip install -e . --no-deps

# ---------------------------------------------------------------------------
# 9. 校验
# ---------------------------------------------------------------------------
python - <<'PY'
import importlib
mods = ["torch", "vllm", "flash_attn", "flashinfer", "xformers",
        "tensordict", "transformers", "ray", "accelerate", "peft",
        "hydra", "wandb", "reasoning_gym", "verl"]
print(f"{'package':<20} {'version':<20} status")
print("-" * 60)
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "n/a")
        print(f"{m:<20} {ver:<20} OK")
    except Exception as e:
        print(f"{m:<20} {'-':<20} FAIL: {e}")

import torch
print()
print("CUDA available :", torch.cuda.is_available())
print("CUDA runtime   :", torch.version.cuda)
print("GPU count      :", torch.cuda.device_count())
print("cxx11_abi      :", torch.compiled_with_cxx11_abi())
PY

echo
echo "✅ 安装完成。"
echo "   激活: source ${VENV_DIR}/bin/activate"
