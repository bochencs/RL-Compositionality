set -x

export MODEL_PATH=checkpoints/string-task/stage1-rft-with-hint/global_step_1648
export PROJECT_NAME=string-task-with-hint
export EXPERIMENT_NAME=stage2-rl-level1to2
export TRAIN_FILES="['data/string_task/stage2_level1/forward_train.parquet','data/string_task/stage2_level2/forward_train.parquet']"
export VAL_FILES="['data/string_task/stage2_level1to8/forward_test.parquet']"
# export NNODES=4
export SAVE_DIR=checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

bash examples/grpo_trainer/template.sh
