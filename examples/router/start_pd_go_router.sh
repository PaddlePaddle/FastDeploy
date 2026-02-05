#!/bin/bash
set -e

# Test splitwise deployment
# There are two methods for splitwise deployment:
# v0: using splitwise_scheduler or dp_scheduler
# v1: using local_scheduler + router
# v2: using local_scheduler + golang_router

# prepare environment
export MODEL_NAME="PaddlePaddle/ERNIE-4.5-0.3B-Paddle"
export FD_DEBUG=1

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
source ${SCRIPT_DIR}/utils.sh

unset http_proxy && unset https_proxy

P_PORT=52400
D_PORT=52500
ROUTER_PORT=52700
LOG_DATE=$(date +%Y%m%d_%H%M%S)

FD_BIN_DIR="/usr/local/bin"
FD_ROUTER_BIN="${FD_BIN_DIR}/fd-router"
FD_ROUTER_URL="https://paddle-qa.bj.bcebos.com/paddle-pipeline/FastDeploy_ActionCE/develop/latest/fd-router"

ports=($P_PORT $D_PORT $ROUTER_PORT)
check_ports "${ports[@]}" || {
    echo "❌ Some ports are in use. Please release them."
    exit 1
}

# check fd-router binary
if [ ! -x "${FD_ROUTER_BIN}" ]; then
    echo "⚠️ fd-router not found, downloading..."

    mkdir -p "${FD_BIN_DIR}"
    TMP_BIN="${FD_ROUTER_BIN}.tmp"

    wget -q --no-proxy "${FD_ROUTER_URL}" -O "${TMP_BIN}" || {
        echo "❌ Download fd-router failed"
        rm -f "${TMP_BIN}"
        exit 1
    }

    # ------- sanity checks (no fixed hash) -------

    # 1. must be ELF binary
    file "${TMP_BIN}" || grep -q "ELF" || {
        echo "❌ fd-router is not an ELF binary"
        rm -f "${TMP_BIN}"
        exit 1
    }

    # 2. must be x86_64 architecture
    file "${TMP_BIN}" | grep -q "x86-64" || {
        echo "❌ fd-router architecture mismatch"
        rm -f "${TMP_BIN}"
        exit 1
    }

    # 3. size check (avoid HTML / empty / error pages)
    SIZE=$(stat -c%s "${TMP_BIN}")
    if [ "$SIZE" -lt 1000000 ]; then
        echo "❌ fd-router size is too small ($SIZE bytes), suspicious"
        rm -f "${TMP_BIN}"
        exit 1
    fi

    # -------------------------------------

    mv "${TMP_BIN}" "${FD_ROUTER_BIN}"
    chmod +x "${FD_ROUTER_BIN}"

    echo "✅ fd-router installed with sanity checks"
else
    echo "✅ fd-router already exists"
fi

# start router
export FD_LOG_DIR="log/$LOG_DATE/router"
rm -rf ${FD_LOG_DIR} && mkdir -p ${FD_LOG_DIR}

nohup /usr/local/bin/fd-router \
    --port ${ROUTER_PORT} \
    --splitwise \
    2>&1 >${FD_LOG_DIR}/nohup &

# start prefill
export CUDA_VISIBLE_DEVICES=0
export FD_LOG_DIR="log/$LOG_DATE/prefill"
rm -rf ${FD_LOG_DIR} && mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --port "${P_PORT}" \
    --splitwise-role "prefill" \
    --router "0.0.0.0:${ROUTER_PORT}" \
2>&1 >${FD_LOG_DIR}/nohup &

wait_for_health ${P_PORT}

# start decode
export CUDA_VISIBLE_DEVICES=1
export FD_LOG_DIR="log/$LOG_DATE/decode"
rm -rf ${FD_LOG_DIR} && mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --port "${D_PORT}" \
    --splitwise-role "decode" \
    --router "0.0.0.0:${ROUTER_PORT}" \
2>&1 >${FD_LOG_DIR}/nohup &

wait_for_health ${D_PORT}

# send request
sleep 10  # make sure server is registered to router
echo "send request..."
curl -X POST "http://0.0.0.0:${ROUTER_PORT}/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "hello"}
  ],
  "max_tokens": 100,
  "stream": false
}'
