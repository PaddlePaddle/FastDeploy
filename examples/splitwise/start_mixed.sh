#!/bin/bash
set -e

# Test mixed server + router

# prepare environment
export MODEL_NAME="PaddlePaddle/ERNIE-4.5-0.3B-Paddle"
export FD_DEBUG=1

unset http_proxy && unset https_proxy
rm -rf log_*
source ./utils.sh

S1_PORT=52400
S2_PORT=52500
ROUTER_PORT=52600

ports=(
    $S1_PORT $((S1_PORT + 1)) $((S1_PORT + 2)) $((S1_PORT + 3))
    $S2_PORT $((S2_PORT + 1)) $((S2_PORT + 2)) $((S2_PORT + 3))
    $ROUTER_PORT
)
check_ports "${ports[@]}" || {
    echo "❌ Some ports are in use. Please release them."
    exit 1
}

# start router
export FD_LOG_DIR="log_router"
mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.router.launch \
    --port ${ROUTER_PORT} \
    2>&1 >${FD_LOG_DIR}/nohup &

# start modelserver 0
export CUDA_VISIBLE_DEVICES=0
export FD_LOG_DIR="log_server_0"
mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port ${S1_PORT} \
       --metrics-port $((S1_PORT + 1)) \
       --engine-worker-queue-port $((S1_PORT + 2)) \
       --cache-queue-port $((S1_PORT + 3)) \
       --max-model-len 32768 \
       --router "0.0.0.0:${ROUTER_PORT}" \
       2>&1 >${FD_LOG_DIR}/nohup &

wait_for_health ${S1_PORT}

# start modelserver 1
export CUDA_VISIBLE_DEVICES=1
export FD_LOG_DIR="log_server_1"
mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port ${S2_PORT} \
       --metrics-port $((S2_PORT + 1)) \
       --engine-worker-queue-port $((S2_PORT + 2)) \
       --cache-queue-port $((S2_PORT + 3)) \
       --max-model-len 32768 \
       --router "0.0.0.0:${ROUTER_PORT}" \
       2>&1 >${FD_LOG_DIR}/nohup &

wait_for_health ${S2_PORT}

# send request
sleep 10  # make sure server is registered to router
echo "send request..."
curl -X POST "http://0.0.0.0:${ROUTER_PORT}/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "hello"}
  ],
  "max_tokens": 20,
  "stream": false
}'
