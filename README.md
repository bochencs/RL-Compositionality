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
export GPU_MEM_UTIL=0.5
bash rlcomp.sh stage1
```

## 配置 wandb 和其他 secret

在仓库根建 `.env.local`（已加 gitignore，不会被提交），bootstrap 自动载入：

```bash
# .env.local
WANDB_API_KEY=wandb_v1_...
# 可选：HUGGING_FACE_HUB_TOKEN=...
```

有 `WANDB_API_KEY` 时 bootstrap 默认 `WANDB_MODE=online`；无则默认 offline。

## 日志与调试

每次 `bash rlcomp.sh <action>` 都会在 `results/pipeline_runs/` 下生成一份完整记录（gitignored，每机本地）：

```
results/pipeline_runs/
├── history.jsonl            # 一行一条，适合 jq/grep
├── labbook.md               # 人读表格
├── latest -> YYYYMMDD_..../ # 最近一次运行
└── YYYYMMDD_HHMMSS_<action>/
    ├── invocation.meta.json # 总体 meta (host, git, GPU, env, 耗时, exit code)
    ├── invocation.context.txt
    ├── <step>.stdout.log    # 每步的完整 stdout
    ├── <step>.stderr.log
    └── <step>.meta.json     # 每步的 exit code + 耗时
```

常用命令：

```bash
# 最近一次出错在哪一步
cat results/pipeline_runs/latest/invocation.meta.json | jq '.sub_actions[] | select(.exit_code != 0)'

# 看最近一次 stage1 的完整 stderr
tail -100 results/pipeline_runs/latest/stage1.stderr.log

# 所有失败过的运行
jq 'select(.exit_code != 0)' results/pipeline_runs/history.jsonl
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
