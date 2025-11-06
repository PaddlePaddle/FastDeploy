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

MODEL_NAME="PaddlePaddle/ERNIE-4.5-0.3B-Paddle"
# MODEL_NAME="baidu/ERNIE-4.5-21B-A3B-Paddle"
aistudio download --model ${MODEL_NAME}

unset http_proxy && unset https_proxy
rm -rf log_*

# start prefill
export FD_LOG_DIR="log_prefill"
mkdir -p ${FD_LOG_DIR}

export CUDA_VISIBLE_DEVICES=0
export FD_DEBUG=1
export ENABLE_V1_KVCACHE_SCHEDULER=0

nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port 8100 \
       --metrics-port 8101 \
       --engine-worker-queue-port 8102 \
       --cache-queue-port 8103 \
       --max-model-len 32768 \
       --splitwise-role "prefill" \
       2>&1 >${FD_LOG_DIR}/nohup &
wait_for_health 8100

# start decode
export FD_LOG_DIR="log_decode"
mkdir -p ${FD_LOG_DIR}

export CUDA_VISIBLE_DEVICES=1
export FD_DEBUG=1
export ENABLE_V1_KVCACHE_SCHEDULER=0

nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port 9000 \
       --metrics-port 9001 \
       --engine-worker-queue-port 9002 \
       --cache-queue-port 9003 \
       --max-model-len 32768 \
       --splitwise-role "decode" \
       --innode-prefill-ports 8102 \
       2>&1 >${FD_LOG_DIR}/nohup &
wait_for_health 9000
