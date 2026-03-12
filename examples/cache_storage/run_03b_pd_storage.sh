#!/bin/bash
set -e

# =============================================================================
# PD Disaggregation + Global Cache Pooling Test Script
# Reference: start_v1_tp1.sh (PD Disaggregation) + run.sh (Mooncake Cache Pooling)
# Note: Modify CUDA_VISIBLE_DEVICES environment variables for PD instances
# =============================================================================

# ======================== Environment Variables Configuration ========================
export MODEL_NAME="/work/models/PaddlePaddle/ERNIE-4.5-0.3B-Paddle"
export FD_DEBUG=1

# Mooncake Configuration (using environment variables)
master_ip="127.0.0.1"
master_port=15001
metadata_port=15002

export MOONCAKE_MASTER_SERVER_ADDR="${master_ip}:${master_port}"
export MOONCAKE_METADATA_SERVER="http://${master_ip}:${metadata_port}/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="50000000000"
# export MOONCAKE_PROTOCOL="tcp"
export MOONCAKE_PROTOCOL="rdma"
# export MOONCAKE_RDMA_DEVICES="mlx5_0"

# ======================== Port Configuration ========================
P_PORT=52400
D_PORT=52500
ROUTER_PORT=52700
LOG_DATE=$(date +%Y%m%d_%H%M%S)

# ======================== Cleanup and Preparation ========================
unset http_proxy && unset https_proxy
rm -rf log_*

source ./utils.sh

# Check ports
ports=($P_PORT $D_PORT $ROUTER_PORT $master_port $metadata_port)
check_ports "${ports[@]}" || {
    echo "❌ Some ports are in use. Please release them."
    exit 1
}

# ======================== Start Mooncake Master ========================
echo "=== Starting Mooncake Master ==="
export FD_LOG_DIR="log_master"
mkdir -p ${FD_LOG_DIR}

nohup mooncake_master \
    --port=${master_port} \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port=${metadata_port} \
    2>&1 > ${FD_LOG_DIR}/nohup &

sleep 2  # Wait for Mooncake Master to start

# ======================== Start Router ========================
echo "=== Starting Router ==="
export FD_LOG_DIR="log_router"
mkdir -p ${FD_LOG_DIR}
echo "Router log: ${FD_LOG_DIR}, port: ${ROUTER_PORT}"

nohup python -m fastdeploy.router.launch \
    --port ${ROUTER_PORT} \
    --splitwise \
    2>&1 > ${FD_LOG_DIR}/nohup &

sleep 2  # Wait for Router to start

# ======================== Start P Instance (Prefill) ========================
echo "=== Starting Prefill Instance ==="
export CUDA_VISIBLE_DEVICES=3
export FD_LOG_DIR="log_prefill"
mkdir -p ${FD_LOG_DIR}
echo "Prefill log: ${FD_LOG_DIR}, port: ${P_PORT}, GPU: ${CUDA_VISIBLE_DEVICES}"

nohup python -m fastdeploy.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --port ${P_PORT} \
    --max-model-len 32768 \
    --max-num-seqs 32 \
    --splitwise-role prefill \
    --cache-transfer-protocol rdma \
    --router "0.0.0.0:${ROUTER_PORT}" \
    --kvcache-storage-backend mooncake \
    2>&1 > ${FD_LOG_DIR}/nohup &


# ======================== Start D Instance (Decode) ========================
echo "=== Starting Decode Instance ==="
export CUDA_VISIBLE_DEVICES=7
export FD_LOG_DIR="log_decode"
mkdir -p ${FD_LOG_DIR}
echo "Decode log: ${FD_LOG_DIR}, port: ${D_PORT}, GPU: ${CUDA_VISIBLE_DEVICES}"

nohup python -m fastdeploy.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --port ${D_PORT} \
    --max-model-len 32768 \
    --max-num-seqs 32 \
    --splitwise-role decode \
    --cache-transfer-protocol rdma \
    --router "0.0.0.0:${ROUTER_PORT}" \
    --enable-output-caching \
    --kvcache-storage-backend mooncake \
    2>&1 > ${FD_LOG_DIR}/nohup &


# ======================== Wait for Services to be Ready ========================
echo "=== Waiting for services to be ready ==="
wait_for_health ${P_PORT}
wait_for_health ${D_PORT}

# Wait for services to register to Router
sleep 10
echo "✅ All services are ready!"

# ======================== Send Test Requests ========================
# Test scenario: Multi-turn conversation, verify that output cache written by D instance can be read by P instance
#
# Flow:
# 1. Request 1: Send first round question, D instance generates answer and writes to global cache (prompt + output)
# 2. Request 2: Send second round conversation (first round Q&A + follow-up), P instance should hit global cache for first round's complete KV cache
#
echo ""
echo "=== Multi-turn Conversation Test for Global Cache Pooling ==="

# First round question
msg1="深圳是中国经济实力最强的城市之一。近年来，深圳 GDP 持续稳步增长，2023 年突破 3.4 万亿元人民币，2024 年接近 3.7 万亿元，长期位居全国城市前列。深圳经济以第二产业和第三产业为主，高端制造业、电子信息产业和现代服务业发达，形成了以科技创新为核心的产业结构。依托华为、腾讯、大疆等龙头企业，深圳在数字经济、人工智能、新能源等领域具有显著优势。同时，深圳进出口总额常年位居全国城市第一，是中国对外开放和高质量发展的重要引擎。深圳2024年 GDP 是多少？"

echo ""
echo ">>> Request 1: First round question"
echo "    Purpose: D instance generates output and writes to global cache (prompt + output)"
echo ""

# Send first round request and get response
response1=$(curl -s -X POST "http://0.0.0.0:${ROUTER_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"user\", \"content\": \"${msg1}\"}
    ],
    \"max_tokens\": 200,
    \"min_tokens\": 130,
    \"stream\": false,
    \"top_p\": 0
  }")

echo "Response 1:"
echo "${response1}" | python3 -m json.tool 2>/dev/null || echo "${response1}"

# Extract first round response content
assistant_reply=$(echo "${response1}" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['choices'][0]['message']['content'])" 2>/dev/null || echo "")

if [ -z "${assistant_reply}" ]; then
    echo "❌ Failed to get response from Request 1"
    exit 1
fi

# JSON escape assistant_reply to prevent newlines, quotes, and other special characters from breaking JSON format
assistant_reply_escaped=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read().strip()))" <<< "${assistant_reply}")

echo ""
echo "Assistant reply extracted: ${assistant_reply}..."

# Wait for D instance to write output cache to global storage
echo ""
echo ">>> Waiting for D instance to write output cache to global storage..."
sleep 5

# Second round follow-up question
msg2="那深圳2023年的GDP是多少？和2024年相比增长了多少？"

echo ""
echo ">>> Request 2: Second round (multi-turn conversation)"
echo "    Purpose: P instance should hit global cache including D's output from Request 1"
echo "    Check log_prefill/nohup for 'storage_match' to verify cache hit"
echo ""

# Send second round request (including complete multi-turn conversation history)
response2=$(curl -s -X POST "http://0.0.0.0:${ROUTER_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"user\", \"content\": \"${msg1}\"},
      {\"role\": \"assistant\", \"content\": ${assistant_reply_escaped}},
      {\"role\": \"user\", \"content\": \"${msg2}\"}
    ],
    \"max_tokens\": 100,
    \"stream\": false,
    \"top_p\": 0
  }")

echo "Response 2:"
echo "${response2}" | python3 -m json.tool 2>/dev/null || echo "${response2}"

# Extract second round response content and display
assistant_reply2=$(echo "${response2}" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['choices'][0]['message']['content'])" 2>/dev/null || echo "")
echo ""
echo "Assistant reply 2: ${assistant_reply2}"

echo ""
echo ""
echo "=== Test completed ==="
echo ""
echo "Verification Steps:"
echo "1. Check log_prefill/nohup for Request 2's cache hit info:"
echo "   grep -E 'storage_match|cache_hit|matched.*block' log_prefill/nohup"
echo ""
echo "2. If 'storage_match_token_num > 0' in Request 2, it means P instance"
echo "   successfully read the output cache written by D instance from Request 1"
echo ""
echo "Log files:"
echo "  - Prefill: log_prefill/nohup"
echo "  - Decode:  log_decode/nohup"
echo "  - Router:  log_router/nohup"
echo "  - Master:  log_master/nohup"
