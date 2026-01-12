#!/bin/bash
set -e

export MODEL_NAME="PaddlePaddle/ERNIE-4.5-0.3B-Paddle"
export MOONCAKE_CONFIG_PATH=./mooncake_config.json
export FD_DEBUG=1

unset http_proxy && unset https_proxy
rm -rf log_*
bash stop.sh

source ./utils.sh

S0_PORT=52700
S1_PORT=52800

ports=(
    $S0_PORT $((S0_PORT + 1)) $((S0_PORT + 2)) $((S0_PORT + 3))
    $S1_PORT $((S1_PORT + 1)) $((S1_PORT + 2)) $((S1_PORT + 3))
)
check_ports "${ports[@]}" || {
    echo "❌ Some ports are in use. Please release them."
    exit 1
}

# Launch MoonCake master
nohup mooncake_master \
    --port=15001 \
    --enable_http_metadata_server=true  \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port=15002 \
    --metrics_port=15003 \
     2>&1 > log_master &

# Launch FD server 0
export CUDA_VISIBLE_DEVICES=0
export FD_LOG_DIR="log_0"
mkdir -p ${FD_LOG_DIR}
echo "server 0 port: ${S0_PORT}"

nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port ${S0_PORT} \
       --metrics-port $((S0_PORT + 1)) \
       --engine-worker-queue-port $((S0_PORT + 2)) \
       --cache-queue-port $((S0_PORT + 3)) \
       --max-model-len 32768 \
       --max-num-seqs 32 \
       --kvcache-storage-backend mooncake \
       2>&1 >${FD_LOG_DIR}/nohup &

# Launch FD server 1
export CUDA_VISIBLE_DEVICES=1
export FD_LOG_DIR="log_1"
mkdir -p ${FD_LOG_DIR}
echo "server 1 port: ${S1_PORT}"

nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port ${S1_PORT} \
       --metrics-port $((S1_PORT + 1)) \
       --engine-worker-queue-port $((S1_PORT + 2)) \
       --cache-queue-port $((S1_PORT + 3)) \
       --max-model-len 32768 \
       --max-num-seqs 32 \
       --kvcache-storage-backend mooncake \
       2>&1 >${FD_LOG_DIR}/nohup &

wait_for_health ${S0_PORT}
wait_for_health ${S1_PORT}

# send request
msg="深圳是中国经济实力最强的城市之一。近年来，深圳 GDP 持续稳步增长，**2023 年突破 3.4 万亿元人民币，2024 年接近 3.7 万亿元**，长期位居全国城市前列。深圳经济以第二产业和第三产业为主，高端制造业、电子信息产业和现代服务业发达，形成了以科技创新为核心的产业结构。依托华为、腾讯、大疆等龙头企业，深圳在数字经济、人工智能、新能源等领域具有显著优势。同时，深圳进出口总额常年位居全国城市第一，是中国对外开放和高质量发展的重要引擎。深圳2024年 GDP 是多少？"

echo "send request to server_0"
curl -X POST "http://0.0.0.0:${S0_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"user\", \"content\": \"${msg}\"}
    ],
    \"max_tokens\": 50,
    \"stream\": false,
    \"top_p\": 0
  }"

sleep 5

echo "send request to server_1"
curl -X POST "http://0.0.0.0:${S1_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"user\", \"content\": \"${msg}\"}
    ],
    \"max_tokens\": 50,
    \"stream\": false,
    \"top_p\": 0
  }"
