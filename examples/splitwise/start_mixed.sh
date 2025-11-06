#!/bin/bash
set -e

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

unset http_proxy && unset https_proxy
rm -rf log_*

# start router
export FD_LOG_DIR="log_router"
mkdir -p ${FD_LOG_DIR}

router_port=9000
nohup python -m fastdeploy.router.launch \
    --port ${router_port} \
    2>&1 >${FD_LOG_DIR}/nohup &
sleep 1

# start modelserver 0
export CUDA_VISIBLE_DEVICES=0
export FD_LOG_DIR="log_server_0"
mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port 8100 \
       --metrics-port 8101 \
       --engine-worker-queue-port 8102 \
       --cache-queue-port 8103 \
       --max-model-len 32768 \
       --router "0.0.0.0:${router_port}" \
       2>&1 >${FD_LOG_DIR}/nohup &
sleep 1

wait_for_health 8100

# start modelserver 1
export CUDA_VISIBLE_DEVICES=1
export FD_LOG_DIR="log_server_1"
mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port 8200 \
       --metrics-port 8201 \
       --engine-worker-queue-port 8202 \
       --cache-queue-port 8203 \
       --max-model-len 32768 \
       --router "0.0.0.0:${router_port}" \
       2>&1 >${FD_LOG_DIR}/nohup &

wait_for_health 8200
