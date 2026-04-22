#!/bin/bash
  set -euo pipefail

  # ModelArts 预置变量
  NNODES="${VC_WORKER_NUM:-1}"
  NODE_RANK="${VC_TASK_INDEX:-0}"
  GPUS_PER_NODE="${MA_NUM_GPUS:-1}"
  HEAD_HOST="$(echo "${VC_WORKER_HOSTS:-127.0.0.1}" | cut -d',' -f1)"
  HEAD_IP="$(getent hosts "${HEAD_HOST}" | awk 'NR==1{print $1}')"
  RAY_PORT="${RAY_PORT:-6379}"
  HEAD_ADDR="${HEAD_IP:-$HEAD_HOST}:${RAY_PORT}"

  echo "[INFO] NNODES=${NNODES}, NODE_RANK=${NODE_RANK}, GPUS_PER_NODE=${GPUS_PER_NODE}"
  echo "[INFO] VC_WORKER_HOSTS=${VC_WORKER_HOSTS:-}"
  echo "[INFO] HEAD_HOST=${HEAD_HOST}, HEAD_IP=${HEAD_IP:-N/A}, HEAD_ADDR=${HEAD_ADDR}"

  ray stop --force || true

  if [ "${NODE_RANK}" = "0" ]; then
    # Head 节点
    export RAY_ADDRESS=auto
    ray start --head \
      --port="${RAY_PORT}" \
      --node-ip-address="${MA_CURRENT_IP:-${HEAD_IP:-127.0.0.1}}" \
      --num-gpus="${GPUS_PER_NODE}" \
      --disable-usage-stats

    # 可选：等几秒让 worker 加入
    sleep 15
    ray status || true

    # 只在 head 跑你的主流程
    NNODES="${NNODES}" N_GPUS_PER_NODE="${GPUS_PER_NODE}" \
    bash /home/ma-user/work/RL-Compositionality/bash/section41_42/stage1_create_train_data_multi.sh

    ray stop --force || true
  else
    # Worker 节点：反复尝试加入 head
    until ray start --address="${HEAD_ADDR}" --num-gpus="${GPUS_PER_NODE}" --disable-usage-stats; do
      echo "[INFO] worker ${NODE_RANK} waiting head ${HEAD_ADDR}..."
      sleep 5
    done

    # 保持worker存活，直到head结束（集群不可达）
    while ray status >/dev/null 2>&1; do
      sleep 10
    done
  fi