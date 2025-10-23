#!/bin/bash
MODEL_PATH="${1}/TP2"
FD_API_PORT=${FD_API_PORT:-8000}
FD_ENGINE_QUEUE_PORT=${FD_ENGINE_QUEUE_PORT:-8001}
FD_METRICS_PORT=${FD_METRICS_PORT:-8002}
FD_CACHE_QUEUE_PORT=${FD_CACHE_QUEUE_PORT:-8003}



if [ -z "$MODEL_PATH" ]; then
  echo "❌ 用法: $0 <模型路径>"
  exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ 错误：模型目录不存在: $MODEL_PATH"
  exit 1
fi

echo "使用模型: $MODEL_PATH"


# 清理日志
rm -rf log/*
mkdir -p log

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1
export INFERENCE_MSG_QUEUE_ID=${FD_INFERENCE_MSG_QUEUE_ID:-7679}
export ENABLE_V1_KVCACHE_SCHEDULER=1


python -m fastdeploy.entrypoints.openai.api_server \
       --tensor-parallel-size 2 \
       --port ${FD_API_PORT} \
       --engine-worker-queue-port ${FD_ENGINE_QUEUE_PORT} \
       --metrics-port ${FD_METRICS_PORT} \
       --cache-queue-port ${FD_CACHE_QUEUE_PORT} \
       --quantization wint8 \
       --max-model-len 32768 \
       --max-num-seqs 1 \
       --gpu-memory-utilization 0.9 \
       --model "$MODEL_PATH" \
       --load-strategy ipc_snapshot \
       --dynamic-load-weight &

success=0

for i in $(seq 1 300); do
    if (echo > /dev/tcp/127.0.0.1/$FD_API_PORT) >/dev/null 2>&1; then
        echo "API server is up on port $FD_API_PORT on iteration $i"
        success=1
        break
    fi
    sleep 1
done
if [ $success -eq 0 ]; then
    echo "超时: API 服务在 300 秒内未启动 (端口 $FD_API_PORT)"
fi
