#!/bin/bash
set -e

MODEL_NAME="PaddlePaddle/ERNIE-4.5-0.3B-Paddle"
DATA_PARALLEL_SIZE=2
TENSOR_PARALLEL_SIZE=1
LOG_DATE=$(date +%Y%m%d_%H%M%S)

export FD_DEBUG=1
export FD_ENABLE_MULTI_API_SERVER=1

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
source ${SCRIPT_DIR}/utils.sh

unset http_proxy && unset https_proxy

# start router
ROUTER_PORT=$(get_free_ports 1)
echo ROUTER_PORT:  $ROUTER_PORT

export FD_LOG_DIR="log/$LOG_DATE/router"
rm -rf $FD_LOG_DIR
mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.router.launch \
    --port ${ROUTER_PORT} \
    --splitwise \
    2>&1 >${FD_LOG_DIR}/nohup &
sleep 1


# start prefill
P_SERVER_PORTS=$(get_free_ports $DATA_PARALLEL_SIZE)
echo P_SERVER_PORTS:  $P_SERVER_PORTS

export CUDA_VISIBLE_DEVICES="0,1"
export FD_LOG_DIR="log/$LOG_DATE/prefill"
rm -rf $FD_LOG_DIR
mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.entrypoints.openai.multi_api_server \
    --num-servers ${DATA_PARALLEL_SIZE}\
    --ports ${P_SERVER_PORTS} \
    --args --model ${MODEL_NAME} \
    --max-model-len 32768 \
    --data-parallel-size ${DATA_PARALLEL_SIZE} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --splitwise-role "prefill" \
    --router "0.0.0.0:${ROUTER_PORT}" \
2>&1 >${FD_LOG_DIR}/nohup &

wait_for_health ${P_SERVER_PORTS}


# start decode
D_SERVER_PORTS=$(get_free_ports $DATA_PARALLEL_SIZE)
echo D_SERVER_PORTS:  $D_SERVER_PORTS

export CUDA_VISIBLE_DEVICES="4,5"
export FD_LOG_DIR="log/$LOG_DATE/decode"
rm -rf $FD_LOG_DIR
mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.entrypoints.openai.multi_api_server \
    --num-servers ${DATA_PARALLEL_SIZE}\
    --ports ${D_SERVER_PORTS} \
    --args --model ${MODEL_NAME} \
    --max-model-len 32768 \
    --data-parallel-size ${DATA_PARALLEL_SIZE} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --splitwise-role "decode" \
    --router "0.0.0.0:${ROUTER_PORT}" \
2>&1 >${FD_LOG_DIR}/nohup &

wait_for_health ${D_SERVER_PORTS}


# send request
sleep 10  # make sure server is registered to router
curl -X POST "http://0.0.0.0:${ROUTER_PORT}/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "hello"}
  ],
  "max_tokens": 100,
  "stream": false
}'
