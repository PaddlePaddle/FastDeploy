#!/bin/bash
set -e

# Test splitwise deployment
# There are two methods for splitwise deployment:
# v0: using splitwise_scheduler or dp_scheduler
# v1: using local_scheduler + router

wait_for_health() {
    IFS=',' read -r -a server_ports <<< "$1"
    local num_ports=${#server_ports[@]}
    local total_lines=$((num_ports + 1))
    local first_run=true
    local GREEN='\033[0;32m'
    local RED='\033[0;31m'
    local NC='\033[0m' # No Color
    local start_time=$(date +%s)

    while true; do
        local all_ready=true
        for port in "${server_ports[@]}"; do
            status_code=$(curl -s --max-time 1 -o /dev/null -w "%{http_code}" "http://0.0.0.0:${port}/health" || echo "000")
            if [ "$status_code" -eq 200 ]; then
                printf "Port %s: ${GREEN}[OK]   200${NC}\033[K\n" "$port"
            else
                all_ready=false
                printf "Port %s: ${RED}[WAIT] %s${NC}\033[K\n" "$port" "$status_code"
            fi
        done
        cur_time=$(date +%s)
        if [ "$all_ready" = "true" ]; then
            echo "All services are ready!    [$((cur_time-start_time))s]"
            break
        else
            echo "Waiting for services...    [$((cur_time-start_time))s]"
            printf "\033[%dA" "$total_lines"  # roll back cursor
            sleep 1
        fi
    done
}


# serving config
MODEL_NAME="PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle"
DATA_PARALLEL_SIZE=2
TENSOR_PARALLEL_SIZE=1
NUM_GPUS=$(($DATA_PARALLEL_SIZE * $TENSOR_PARALLEL_SIZE))
LOG_DATE=$(date +%Y%m%d_%H%M%S)

# fastdeploy environment
export FD_DEBUG=1
export ENABLE_V1_KVCACHE_SCHEDULER=1
export KVCACHE_GDRCOPY_FLUSH_ENABLE=1
export FD_ENABLE_MULTI_API_SERVER=1

# set rdma nics
SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
export $(bash ${SCRIPT_DIR}/../../scripts/get_rdma_nics.sh gpu)
echo "KVCACHE_RDMA_NICS:${KVCACHE_RDMA_NICS}"
if [ -z "${KVCACHE_RDMA_NICS}" ]; then
  echo "KVCACHE_RDMA_NICS is empty, please check the output of get_rdma_nics.sh"
  exit 1
fi

# clean up proxy and files
unset http_proxy && unset https_proxy

# start router
ROUTER_PORT=$(bash $SCRIPT_DIR/get_free_ports.sh 1)
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
P_SERVER_PORTS=$(bash $SCRIPT_DIR/get_free_ports.sh $DATA_PARALLEL_SIZE)
P_METRICS_PORTS=$(bash $SCRIPT_DIR/get_free_ports.sh $DATA_PARALLEL_SIZE)
P_ENGINE_WORKER_QUEUE_PORTS=$(bash $SCRIPT_DIR/get_free_ports.sh $DATA_PARALLEL_SIZE)
P_CACHE_QUEUE_PORTS=$(bash $SCRIPT_DIR/get_free_ports.sh $DATA_PARALLEL_SIZE)
P_RDMA_COMM_PORTS=$(bash $SCRIPT_DIR/get_free_ports.sh $NUM_GPUS)
P_PD_COMM_PORTS=$(bash $SCRIPT_DIR/get_free_ports.sh $DATA_PARALLEL_SIZE)
echo "---------------------------"
echo P_SERVER_PORTS:  $P_SERVER_PORTS
echo P_METRICS_PORTS:  $P_METRICS_PORTS
echo P_ENGINE_WORKER_QUEUE_PORTS:  $P_ENGINE_WORKER_QUEUE_PORTS
echo P_CACHE_QUEUE_PORTS:  $P_CACHE_QUEUE_PORTS
echo P_RDMA_COMM_PORTS:  $P_RDMA_COMM_PORTS
echo P_PD_COMM_PORTS:  $P_PD_COMM_PORTS

export CUDA_VISIBLE_DEVICES="4,5"
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
D_SERVER_PORTS=$(bash $SCRIPT_DIR/get_free_ports.sh $DATA_PARALLEL_SIZE)
D_ENGINE_WORKER_QUEUE_PORTS=$(bash $SCRIPT_DIR/get_free_ports.sh $DATA_PARALLEL_SIZE)
D_CACHE_QUEUE_PORTS=$(bash $SCRIPT_DIR/get_free_ports.sh $DATA_PARALLEL_SIZE)
D_METRICS_PORTS=$(bash $SCRIPT_DIR/get_free_ports.sh $DATA_PARALLEL_SIZE)
D_RDMA_COMM_PORTS=$(bash $SCRIPT_DIR/get_free_ports.sh $NUM_GPUS)
D_PD_COMM_PORTS=$(bash $SCRIPT_DIR/get_free_ports.sh $DATA_PARALLEL_SIZE)
echo "---------------------------"
echo D_SERVER_PORTS:  $D_SERVER_PORTS
echo D_ENGINE_WORKER_QUEUE_PORTS:  $D_ENGINE_WORKER_QUEUE_PORTS
echo D_CACHE_QUEUE_PORTS:  $D_CACHE_QUEUE_PORTS
echo D_METRICS_PORTS:  $D_METRICS_PORTS
echo D_RDMA_COMM_PORTS:  $D_RDMA_COMM_PORTS
echo D_PD_COMM_PORTS:  $D_PD_COMM_PORTS

export CUDA_VISIBLE_DEVICES="6,7"
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
    {"role": "user", "content": "鲁迅是谁"}
  ],
  "max_tokens": 200,
  "stream": false
}'
