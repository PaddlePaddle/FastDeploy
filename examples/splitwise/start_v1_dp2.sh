#!/bin/bash
set -e

# Test splitwise deployment
# There are two methods for splitwise deployment:
# v0: using splitwise_scheduler or dp_scheduler
# v1: using local_scheduler + router

MODEL_NAME="PaddlePaddle/ERNIE-4.5-0.3B-Paddle"
DATA_PARALLEL_SIZE=2
TENSOR_PARALLEL_SIZE=1
NUM_GPUS=$(($DATA_PARALLEL_SIZE * $TENSOR_PARALLEL_SIZE))
LOG_DATE=$(date +%Y%m%d_%H%M%S)

export FD_DEBUG=1
export ENABLE_V1_KVCACHE_SCHEDULER=1
export KVCACHE_GDRCOPY_FLUSH_ENABLE=1
export FD_ENABLE_MULTI_API_SERVER=1

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
export $(bash ${SCRIPT_DIR}/../../scripts/get_rdma_nics.sh gpu)
echo "KVCACHE_RDMA_NICS:${KVCACHE_RDMA_NICS}"
if [ -z "${KVCACHE_RDMA_NICS}" ]; then
  echo "KVCACHE_RDMA_NICS is empty, please check the output of get_rdma_nics.sh"
  exit 1
fi

unset http_proxy && unset https_proxy
source ${SCRIPT_DIR}/utils.sh

# start router
ROUTER_PORT=$(get_free_ports 1)
echo "---------------------------"
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
P_METRICS_PORTS=$(get_free_ports $DATA_PARALLEL_SIZE)
P_ENGINE_WORKER_QUEUE_PORTS=$(get_free_ports $DATA_PARALLEL_SIZE)
P_CACHE_QUEUE_PORTS=$(get_free_ports $DATA_PARALLEL_SIZE)
P_RDMA_COMM_PORTS=$(get_free_ports $NUM_GPUS)
P_PD_COMM_PORTS=$(get_free_ports $DATA_PARALLEL_SIZE)
echo "---------------------------"
echo P_SERVER_PORTS:  $P_SERVER_PORTS
echo P_METRICS_PORTS:  $P_METRICS_PORTS
echo P_ENGINE_WORKER_QUEUE_PORTS:  $P_ENGINE_WORKER_QUEUE_PORTS
echo P_CACHE_QUEUE_PORTS:  $P_CACHE_QUEUE_PORTS
echo P_RDMA_COMM_PORTS:  $P_RDMA_COMM_PORTS
echo P_PD_COMM_PORTS:  $P_PD_COMM_PORTS

export CUDA_VISIBLE_DEVICES="0,1"
export FD_LOG_DIR="log/$LOG_DATE/prefill"
rm -rf $FD_LOG_DIR
mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.entrypoints.openai.multi_api_server \
    --num-servers ${DATA_PARALLEL_SIZE}\
    --ports ${P_SERVER_PORTS} \
    --metrics-port ${P_METRICS_PORTS} \
    --args --model ${MODEL_NAME} \
    --engine-worker-queue-port ${P_ENGINE_WORKER_QUEUE_PORTS} \
    --cache-queue-port ${P_CACHE_QUEUE_PORTS} \
    --max-model-len 32768 \
    --data-parallel-size ${DATA_PARALLEL_SIZE} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --splitwise-role "prefill" \
    --cache-transfer-protocol "rdma" \
    --rdma-comm-ports ${P_RDMA_COMM_PORTS} \
    --pd-comm-port ${P_PD_COMM_PORTS} \
    --router "0.0.0.0:${ROUTER_PORT}" \
2>&1 >${FD_LOG_DIR}/nohup &

echo "--- Health Check Status ---"
wait_for_health ${P_SERVER_PORTS}


# start decode
D_SERVER_PORTS=$(get_free_ports $DATA_PARALLEL_SIZE)
D_ENGINE_WORKER_QUEUE_PORTS=$(get_free_ports $DATA_PARALLEL_SIZE)
D_CACHE_QUEUE_PORTS=$(get_free_ports $DATA_PARALLEL_SIZE)
D_METRICS_PORTS=$(get_free_ports $DATA_PARALLEL_SIZE)
D_RDMA_COMM_PORTS=$(get_free_ports $NUM_GPUS)
D_PD_COMM_PORTS=$(get_free_ports $DATA_PARALLEL_SIZE)
echo "---------------------------"
echo D_SERVER_PORTS:  $D_SERVER_PORTS
echo D_ENGINE_WORKER_QUEUE_PORTS:  $D_ENGINE_WORKER_QUEUE_PORTS
echo D_CACHE_QUEUE_PORTS:  $D_CACHE_QUEUE_PORTS
echo D_METRICS_PORTS:  $D_METRICS_PORTS
echo D_RDMA_COMM_PORTS:  $D_RDMA_COMM_PORTS
echo D_PD_COMM_PORTS:  $D_PD_COMM_PORTS

export CUDA_VISIBLE_DEVICES="2,3"
export FD_LOG_DIR="log/$LOG_DATE/decode"
rm -rf $FD_LOG_DIR
mkdir -p ${FD_LOG_DIR}

nohup python -m fastdeploy.entrypoints.openai.multi_api_server \
    --num-servers ${DATA_PARALLEL_SIZE}\
    --ports ${D_SERVER_PORTS} \
    --metrics-port ${D_METRICS_PORTS} \
    --args --model ${MODEL_NAME} \
    --engine-worker-queue-port ${D_ENGINE_WORKER_QUEUE_PORTS} \
    --cache-queue-port ${D_CACHE_QUEUE_PORTS} \
    --max-model-len 32768 \
    --data-parallel-size ${DATA_PARALLEL_SIZE} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --splitwise-role "decode" \
    --cache-transfer-protocol "rdma" \
    --rdma-comm-ports ${D_RDMA_COMM_PORTS} \
    --pd-comm-port ${D_PD_COMM_PORTS} \
    --router "0.0.0.0:${ROUTER_PORT}" \
2>&1 >${FD_LOG_DIR}/nohup &

echo "--- Health Check Status ---"
wait_for_health ${D_SERVER_PORTS}


# send request
echo "------ Request Check ------"
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
