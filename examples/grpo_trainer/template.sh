set -x

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

model_path="${MODEL_PATH:-}"
project_name="${PROJECT_NAME:-string-task}"
experiment_name="${EXPERIMENT_NAME:-}"
train_files="${TRAIN_FILES:-['data/string_task/stage2_level1/forward_train.parquet']}"
val_files="${VAL_FILES:-['data/string_task/stage2_level18/forward_test.parquet']}"
nnodes="${NNODES:-1}"
save_dir="${SAVE_DIR:-checkpoints}"
# Auto-detect via nvidia-smi if env not set; no hardcoded number. Bootstrap.sh
# normally exports N_GPUS_PER_NODE so this fallback only fires when running
# template.sh standalone outside the rlcomp pipeline.
n_gpus_per_node="${N_GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
[ -z "${n_gpus_per_node}" ] || ! [ "${n_gpus_per_node}" -ge 1 ] 2>/dev/null && n_gpus_per_node=1
logger="${LOGGER:-['console','wandb']}"
python_bin="${PYTHON_BIN:-python3}"
gpu_mem_util="${GPU_MEM_UTIL:-0.9}"
# FSDP offload: False is fast on 4+ GPUs; True is required for 1 GPU (actor+ref+vllm won't fit otherwise).
actor_param_offload="${ACTOR_PARAM_OFFLOAD:-False}"
actor_optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
ref_param_offload="${REF_PARAM_OFFLOAD:-False}"

# GRPO hyperparams — scale with GPU count. Env override wins.
# With `use_dynamic_bsz=True`, effective batch = n_gpus × mbsz × n rollouts.
# Keep per-GPU work ≈ constant so throughput scales linearly with GPU count.
bsz="${PPO_BSZ:-$(( n_gpus_per_node * 4 ))}"            # global train batch
mbsz="${PPO_MINI_BSZ:-$(( n_gpus_per_node * 4 ))}"      # mini batch for ppo
n="${PPO_ROLLOUT_N:-$(( n_gpus_per_node * 4 ))}"        # rollouts per prompt
prompt_length="${PROMPT_LENGTH:-1024}"
response_length="${RESPONSE_LENGTH:-8192}"

enable_filter_groups=True
filter_groups_metric=seq_reward
max_num_gen_batches=10
gen_prompt_bsz=$((bsz * 2))

${python_bin} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_files} \
    data.val_files=${val_files} \
    data.train_batch_size=$bsz \
    data.max_prompt_length=$prompt_length \
    data.max_response_length=$response_length \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.use_token_level_loss=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=17408 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mbsz \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_ACTOR_MICRO_BSZ_PER_GPU:-1}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_mem_util} \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=34816 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=34816 \
    algorithm.kl_ctrl.kl_coef=0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    reward_model.reward_manager="mp" \
    reward_model.penalize_overlong=True \
    trainer.critic_warmup=0 \
    trainer.logger=${logger} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${nnodes} \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.default_local_dir=${save_dir} \
    trainer.total_epochs=1 $@
