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

# start router
export FD_LOG_DIR="log_router"
mkdir -p ${FD_LOG_DIR}

echo "start router"
router_port=9000
nohup python -m fastdeploy.router.launch \
    --port ${router_port} \
    --splitwise \
    2>&1 >${FD_LOG_DIR}/nohup &
sleep 1

# start prefill
export CUDA_VISIBLE_DEVICES=0,1
export FD_LOG_DIR="log_prefill"
mkdir -p ${FD_LOG_DIR}

echo "start prefill"
nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port 8100 \
       --metrics-port 8101 \
       --engine-worker-queue-port 8102 \
       --cache-queue-port 8103 \
       --tensor-parallel-size 2 \
       --max-model-len 32768 \
       --splitwise-role "prefill" \
       --pd-comm-port 8104 \
       --rdma-comm-ports 8105,8106 \
       --router "0.0.0.0:${router_port}" \
       2>&1 >${FD_LOG_DIR}/nohup &

wait_for_health 8100

# start decode
export CUDA_VISIBLE_DEVICES=2,3
export FD_LOG_DIR="log_decode"
mkdir -p ${FD_LOG_DIR}

echo "start decode"
nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port 8200 \
       --metrics-port 8201 \
       --engine-worker-queue-port 8202 \
       --cache-queue-port 8203 \
       --max-model-len 32768 \
       --tensor-parallel-size 2 \
       --splitwise-role "decode" \
       --pd-comm-port 8204 \
       --rdma-comm-ports 8205,8206 \
       --router "0.0.0.0:${router_port}" \
       2>&1 >${FD_LOG_DIR}/nohup &

wait_for_health 8200
