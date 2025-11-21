#!/bin/bash
set -e

# Test splitwise deployment
# There are two methods for splitwise deployment:
# v0: using splitwise_scheduler or dp_scheduler
# v1: using local_scheduler + router

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

export FD_DEBUG=1
export ENABLE_V1_KVCACHE_SCHEDULER=1
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

P_PORT=52400
D_PORT=52500
ROUTER_PORT=52600

# start router
export FD_LOG_DIR="log_router"
mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.router.launch \
    --port ${ROUTER_PORT} \
    --splitwise \
    2>&1 >${FD_LOG_DIR}/nohup &
sleep 1

# start prefill
export CUDA_VISIBLE_DEVICES=0
export FD_LOG_DIR="log_prefill"
mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port "${P_PORT}" \
       --metrics-port "$((P_PORT + 1))" \
       --engine-worker-queue-port "$((P_PORT + 2))" \
       --cache-queue-port "$((P_PORT + 3))" \
       --max-model-len 32768 \
       --splitwise-role "prefill" \
       --cache-transfer-protocol "rdma" \
       --rdma-comm-ports "$((P_PORT + 4))" \
       --pd-comm-port "$((P_PORT + 5))" \
       --router "0.0.0.0:${ROUTER_PORT}" \
       2>&1 >${FD_LOG_DIR}/nohup &

wait_for_health ${P_PORT}

# start decode
export CUDA_VISIBLE_DEVICES=1
export FD_LOG_DIR="log_decode"
mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port "${D_PORT}" \
       --metrics-port "$((D_PORT + 2))" \
       --engine-worker-queue-port "$((D_PORT + 3))" \
       --cache-queue-port "$((D_PORT + 1))" \
       --max-model-len 32768 \
       --splitwise-role "decode" \
       --cache-transfer-protocol "rdma" \
       --rdma-comm-ports "$((D_PORT + 4))" \
       --pd-comm-port "$((D_PORT + 5))" \
       --router "0.0.0.0:${ROUTER_PORT}" \
       2>&1 >${FD_LOG_DIR}/nohup &

wait_for_health ${D_PORT}

# send request
sleep 10  # make sure server is registered to router
curl -X POST "http://0.0.0.0:${ROUTER_PORT}/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "hello"}
  ],
  "max_tokens": 20,
  "stream": true
}'
