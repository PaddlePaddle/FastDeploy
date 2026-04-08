#!/bin/bash
MODEL_PATH="${1}/TP2"
FD_API_PORT=${FD_API_PORT:-8180}
FD_ENGINE_QUEUE_PORT=${FD_ENGINE_QUEUE_PORT:-8181}
FD_METRICS_PORT=${FD_METRICS_PORT:-8182}
FD_CACHE_QUEUE_PORT=${FD_CACHE_QUEUE_PORT:-8183}

if [ -z "$MODEL_PATH" ]; then
  echo "❌ Usage: $0 <model_path>"
  exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ Error: Model directory does not exist: $MODEL_PATH"
  exit 1
fi

echo "Using model: $MODEL_PATH"

# Clean logs
rm -rf log/*
mkdir -p log

# Environment variables
export CUDA_VISIBLE_DEVICES=0,1
export INFERENCE_MSG_QUEUE_ID=${FD_INFERENCE_MSG_QUEUE_ID:-7679}
export ENABLE_V1_KVCACHE_SCHEDULER=1

echo "Starting API server"
python -m fastdeploy.entrypoints.openai.api_server \
       --tensor-parallel-size 2 \
       --port ${FD_API_PORT} \
       --engine-worker-queue-port ${FD_ENGINE_QUEUE_PORT} \
       --metrics-port ${FD_METRICS_PORT} \
       --cache-queue-port ${FD_CACHE_QUEUE_PORT} \
       --max-model-len 32768 \
       --max-num-seqs 1 \
       --gpu-memory-utilization 0.9 \
       --model "$MODEL_PATH" \
       --no-shutdown-comm-group-if-worker-idle \
       --load-strategy ipc_snapshot \
       --dynamic-load-weight &

success=0

for i in $(seq 1 300); do
    if (echo > /dev/tcp/127.0.0.1/$FD_API_PORT) >/dev/null 2>&1; then
        echo "API server is up on port $FD_API_PORT at iteration $i"
        success=1
        break
    fi
    sleep 1
done

if [ $success -eq 0 ]; then
    echo "Timeout: API server did not start within 300 seconds (port $FD_API_PORT)"
fi
