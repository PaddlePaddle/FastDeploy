#!/bin/bash
set -e

# =============================================================================
# PD 分离 + 全局 Cache 池化测试脚本
# 参考: start_v1_tp1.sh (PD 分离) + run.sh (Mooncake Cache 池化)
# 注意修改：PD实例的CUDA_VISIBLE_DEVICES环境变量
# =============================================================================

# ======================== 环境变量配置 ========================
export MODEL_NAME="/work/models/PaddlePaddle/ERNIE-4.5-0.3B-Paddle"
export FD_DEBUG=1

# Mooncake 配置（使用环境变量方式）
master_ip="127.0.0.1"
master_port=15001
metadata_port=15002

export MOONCAKE_MASTER_SERVER_ADDR="${master_ip}:${master_port}"
export MOONCAKE_METADATA_SERVER="http://${master_ip}:${metadata_port}/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="50000000000"
# export MOONCAKE_PROTOCOL="tcp"
export MOONCAKE_PROTOCOL="rdma"
# export MOONCAKE_RDMA_DEVICES="mlx5_0"

# ======================== 端口配置 ========================
P_PORT=52400
D_PORT=52500
ROUTER_PORT=52700
LOG_DATE=$(date +%Y%m%d_%H%M%S)

# ======================== 清理和准备 ========================
unset http_proxy && unset https_proxy
rm -rf log_*

source ./utils.sh

# 检查端口
ports=($P_PORT $D_PORT $ROUTER_PORT $master_port $metadata_port)
check_ports "${ports[@]}" || {
    echo "❌ Some ports are in use. Please release them."
    exit 1
}

# ======================== 启动 Mooncake Master ========================
echo "=== Starting Mooncake Master ==="
export FD_LOG_DIR="log_master"
mkdir -p ${FD_LOG_DIR}

nohup mooncake_master \
    --port=${master_port} \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port=${metadata_port} \
    2>&1 > ${FD_LOG_DIR}/nohup &

sleep 2  # 等待 Mooncake Master 启动

# ======================== 启动 Router ========================
echo "=== Starting Router ==="
export FD_LOG_DIR="log_router"
mkdir -p ${FD_LOG_DIR}
echo "Router log: ${FD_LOG_DIR}, port: ${ROUTER_PORT}"

nohup python -m fastdeploy.router.launch \
    --port ${ROUTER_PORT} \
    --splitwise \
    2>&1 > ${FD_LOG_DIR}/nohup &

sleep 2  # 等待 Router 启动

# ======================== 启动 P 实例（Prefill） ========================
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


# ======================== 启动 D 实例（Decode） ========================
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


# ======================== 等待服务就绪 ========================
echo "=== Waiting for services to be ready ==="
wait_for_health ${P_PORT}
wait_for_health ${D_PORT}

# 等待服务注册到 Router
sleep 10
echo "✅ All services are ready!"

# ======================== 发送测试请求 ========================
# 验证场景：多轮对话，验证 D 实例写入的 output cache 能被 P 实例读取
#
# 流程：
# 1. Request 1: 发送第一轮问题，D 实例生成回答并写入全局 Cache（prompt + output）
# 2. Request 2: 发送第二轮对话（第一轮问答 + 追问），P 实例应从全局 Cache 命中第一轮的完整 KV Cache
#
echo ""
echo "=== Multi-turn Conversation Test for Global Cache Pooling ==="

# 第一轮问题
msg1="深圳是中国经济实力最强的城市之一。近年来，深圳 GDP 持续稳步增长，2023 年突破 3.4 万亿元人民币，2024 年接近 3.7 万亿元，长期位居全国城市前列。深圳经济以第二产业和第三产业为主，高端制造业、电子信息产业和现代服务业发达，形成了以科技创新为核心的产业结构。依托华为、腾讯、大疆等龙头企业，深圳在数字经济、人工智能、新能源等领域具有显著优势。同时，深圳进出口总额常年位居全国城市第一，是中国对外开放和高质量发展的重要引擎。深圳2024年 GDP 是多少？"

echo ""
echo ">>> Request 1: First round question"
echo "    Purpose: D instance generates output and writes to global cache (prompt + output)"
echo ""

# 发送第一轮请求，获取回答
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

# 提取第一轮回答内容
assistant_reply=$(echo "${response1}" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['choices'][0]['message']['content'])" 2>/dev/null || echo "")

if [ -z "${assistant_reply}" ]; then
    echo "❌ Failed to get response from Request 1"
    exit 1
fi

# JSON 转义 assistant_reply，避免换行符、引号等特殊字符破坏 JSON 格式
assistant_reply_escaped=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read().strip()))" <<< "${assistant_reply}")

echo ""
echo "Assistant reply extracted: ${assistant_reply}..."

# 等待 D 实例将 output cache 写入全局存储
echo ""
echo ">>> Waiting for D instance to write output cache to global storage..."
sleep 5

# 第二轮追问
msg2="那深圳2023年的GDP是多少？和2024年相比增长了多少？"

echo ""
echo ">>> Request 2: Second round (multi-turn conversation)"
echo "    Purpose: P instance should hit global cache including D's output from Request 1"
echo "    Check log_prefill/nohup for 'storage_match' to verify cache hit"
echo ""

# 发送第二轮请求（包含完整的多轮对话历史）
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

# 提取第二轮回答内容并显示
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
