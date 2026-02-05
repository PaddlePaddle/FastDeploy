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

FD_BIN_DIR="/usr/local/bin"
FD_ROUTER_BIN="${FD_BIN_DIR}/fd-router"
FD_ROUTER_URL="https://paddle-qa.bj.bcebos.com/paddle-pipeline/FastDeploy_ActionCE/develop/latest/fd-router"

ports=(
    $S1_PORT $((S1_PORT + 1)) $((S1_PORT + 2)) $((S1_PORT + 3))
    $S2_PORT $((S2_PORT + 1)) $((S2_PORT + 2)) $((S2_PORT + 3))
    $ROUTER_PORT
)
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
export FD_LOG_DIR="log_router"
mkdir -p ${FD_LOG_DIR}

nohup /usr/local/bin/fd-router \
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
