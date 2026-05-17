set -x

model_path="${MODEL_PATH:-string-rft}"
project_name="${PROJECT_NAME:-string-rft}"
experiment_name="${EXPERIMENT_NAME:-llama-3.1-8b}"
train_files="${TRAIN_FILES:-}"
val_files="${VAL_FILES:-}"
bsz="${BATCH_SIZE:-128}"
max_length="${MAX_LENGTH:-3072}"
nnodes="${NNODES:-1}"
# Auto-detect via nvidia-smi if env not set; no hardcoded number. Bootstrap.sh
# normally exports NPROC_PER_NODE so this fallback only fires when running
# template.sh standalone outside the rlcomp pipeline.
nproc_per_node="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
[ -z "${nproc_per_node}" ] || ! [ "${nproc_per_node}" -ge 1 ] 2>/dev/null && nproc_per_node=1
sp_size="${SP_SIZE:-1}"
epochs="${EPOCHS:-1}"
save_dir="${SAVE_DIR:-checkpoints}"
use_remove_padding="${USE_REMOVE_PADDING:-true}"
torchrun_bin="${TORCHRUN_BIN:-torchrun}"
logger="${LOGGER:-['console','wandb']}"

# Per-GPU micro batch. Default 1 to match upstream PRIME-RL/RL-Compositionality.
# A previous heuristic scaled this to 4 on a 4-GPU box assuming 8B bf16 + grad_ckpt,
# but with the upstream-aligned defaults (fp32, no grad_ckpt, no offload) on
# 4×H100 80GB the 4-GPU/micro=4 path OOMs in LlamaMLP.forward (each rank ~79 GB
# allocated). Keeping micro=1 holds per-rank activations to ~6 GB and matches
# upstream training math byte-for-byte (same global batch=128, just more
# gradient-accumulation steps). Override via SFT_MICRO_BSZ for hosts with
# spare VRAM.
micro_bsz="${SFT_MICRO_BSZ:-1}"

if [ $nnodes -eq 1 ]; then
    STANDALONE="--standalone"
else
    STANDALONE="--master_addr=${MLP_WORKER_0_HOST} --master_port=${MLP_WORKER_0_PORT} --node_rank=${MLP_ROLE_INDEX}"
fi

${torchrun_bin} ${STANDALONE} --nnodes=${nnodes} --nproc_per_node=${nproc_per_node} \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${train_files} \
    data.val_files=${val_files} \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=${max_length} \
    data.truncation=right \
    optim.lr=2e-5 \
    data.train_batch_size=${bsz} \
    data.micro_batch_size_per_gpu=${micro_bsz} \
    model.partial_pretrain=${model_path} \
    trainer.default_hdfs_dir=${save_dir} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.logger=${logger} \
    trainer.total_epochs=${epochs} \
    ulysses_sequence_parallel_size=${sp_size} \
    use_remove_padding=${use_remove_padding} \
    "$@"
