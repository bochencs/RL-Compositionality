export PROJECT_NAME=string-task
export EXPERIMENT_NAME=stage1-rft-with-hint
export MODEL_PATH=~/work/config-file/model/Llama-3.1-8B-Instruct
export NNODES="${VC_WORKER_NUM:-1}"
export SP_SIZE=1
export MAX_LENGTH=3072
export TRAIN_FILES=data/string_task/stage1_level1/rft_data/train.parquet 
export VAL_FILES=data/string_task/stage1_level1/rft_data/test.parquet 
export BATCH_SIZE=128
export EPOCHS=2
export SAVE_DIR=checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

bash examples/sft/template.sh \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.default_hdfs_dir=null
