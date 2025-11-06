#!/bin/bash
set -e

# Test splitwise deployment
# v0 requires prefill and decode in one node and it uses local scheduler
# v1 supports prefill and decode in multi node and it uses splitwise scheduler
# v2 supports prefill and decode in multi node and it uses router and local scheduler

wait_for_health() {
       local server_port=$1
       while true; do
       status_code=$(curl -s -o /dev/null -w "%{http_code}" "http://0.0.0.0:${server_port}/health" || echo "000")
       if [ "$status_code" -eq 200 ]; then
              break
       else
              echo "Service not ready. Retrying in 2s..."
              sleep 2
       fi
       done
}

# prepare environment
MODEL_NAME="PaddlePaddle/ERNIE-4.5-0.3B-Paddle"
# MODEL_NAME="baidu/ERNIE-4.5-21B-A3B-Paddle"

export FD_DEBUG=1
export ENABLE_V1_KVCACHE_SCHEDULER=0
export KVCACHE_GDRCOPY_FLUSH_ENABLE=1

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
export $(bash ${SCRIPT_DIR}/../../scripts/get_rdma_nics.sh gpu)
echo "KVCACHE_RDMA_NICS:${KVCACHE_RDMA_NICS}"
if [ -z "${KVCACHE_RDMA_NICS}" ]; then
  echo "KVCACHE_RDMA_NICS is empty, please check the output of get_rdma_nics.sh"
  exit 1
fi

unset http_proxy && unset https_proxy
rm -rf log_*

# start redis
if ! redis-cli ping &>/dev/null; then
    echo "Redis is not running. Starting redis-server..."
    redis-server --daemonize yes
    sleep 1
else
    echo "Redis is already running."
fi
sleep 1

# start prefill
export CUDA_VISIBLE_DEVICES=0,1
export FD_LOG_DIR="log_prefill"
mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port 8100 \
       --metrics-port 8101 \
       --engine-worker-queue-port 8102 \
       --cache-queue-port 8103 \
       --max-model-len 32768 \
       --tensor-parallel-size 2 \
       --splitwise-role "prefill" \
       --cache-transfer-protocol "rdma,ipc" \
       --pd-comm-port 8104 \
       --rdma-comm-ports 8105,8106 \
       --scheduler-name "splitwise" \
       --scheduler-host "127.0.0.1" \
       --scheduler-port 6379 \
       --scheduler-ttl 9000 \
       2>&1 >${FD_LOG_DIR}/nohup &
wait_for_health 8100

# start decode
export CUDA_VISIBLE_DEVICES=2,3
export FD_LOG_DIR="log_decode"
mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port 9000 \
       --metrics-port 9001 \
       --engine-worker-queue-port 9002 \
       --cache-queue-port 9003 \
       --max-model-len 32768 \
       --tensor-parallel-size 2 \
       --splitwise-role "decode" \
       --cache-transfer-protocol "rdma,ipc" \
       --pd-comm-port 9004 \
       --rdma-comm-ports 9005,9006 \
       --scheduler-name "splitwise" \
       --scheduler-host "127.0.0.1" \
       --scheduler-port 6379 \
       --scheduler-ttl 9000 \
       2>&1 >${FD_LOG_DIR}/nohup &
wait_for_health 9000
