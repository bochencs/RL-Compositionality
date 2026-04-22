# RL-Compositionality

📘 论文：[From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones](https://arxiv.org/abs/2509.25123)

## 使用

根目录单脚本 `rlcomp.sh` 覆盖整条 pipeline。

```bash
# 设环境（可在新 shell 下、任何 cwd，不改 .bashrc）
source /home/ma-user/work/RL-Compositionality/rlcomp.sh

# 执行 action
bash /home/ma-user/work/RL-Compositionality/rlcomp.sh <action>
```

| action | 说明 |
|---|---|
| `install` | 新机器重装 venv（`.venv-rlcomp/`） |
| `env`     | 仅设环境 + 健康检查 |
| `prepare` | 生成 newops 数据集 |
| `stage1`  | Stage 1 RFT（rollout → 过滤 → SFT） |
| `stage2`  | Stage 2 GRPO RL（建议 ≥ 4 GPU） |
| `infer`   | 推理评估矩阵 |
| `all`     | 依次 prepare → stage1 → stage2 → infer |

**可迁移性前提**：文件系统共享到 `/home/ma-user/work/RL-Compositionality/`、NVIDIA driver 已装、系统有 Python 3.10。脚本按 `nvidia-smi -L | wc -l` 自动配 `GPU_MEM_UTIL`、FSDP offload、`MODEL_DTYPE`、`SAVE_FREQ`，无需手动调参。

**覆盖默认值**：直接 `export VAR=value` 再调 `rlcomp.sh`，例如：

```bash
export WANDB_MODE=online WANDB_API_KEY=... GPU_MEM_UTIL=0.5
bash rlcomp.sh stage1
```

## Citing

```bibtex
@article{yuan2025rlcompose,
  author    = {Lifan Yuan and Weize Chen and Yuchen Zhang and Ganqu Cui and Hanbin Wang and Ziming You and Ning Ding and Zhiyuan Liu and Maosong Sun and Hao Peng},
  title     = {From $f(x)$ and $g(x)$ to $f(g(x))$: {LLMs} Learn New Skills in {RL} by Composing Old Ones},
  journal   = {arXiv preprint arXiv:2509.25123},
  year      = {2025},
  url       = {https://arxiv.org/abs/2509.25123}
}
```
