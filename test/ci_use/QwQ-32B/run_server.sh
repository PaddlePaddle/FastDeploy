#!/bin/bash
MODEL_PATH=${1}

if [ -z "$MODEL_PATH" ]; then
  echo "❌ 用法: $0 <模型路径>"
  exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ 错误：模型目录不存在: $MODEL_PATH"
  exit 1
fi

echo "📁 使用模型: $MODEL_PATH"


# 清理日志
rm -rf log/*
mkdir -p log

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1
export INFERENCE_MSG_QUEUE_ID=7679
export ENABLE_V1_KVCACHE_SCHEDULER=1


python -m fastdeploy.entrypoints.openai.api_server \
       --tensor-parallel-size 2 \
       --port 8787 \
       --engine-worker-queue-port 7679 \
       --metrics-port 7877 \
       --quantization wint8 \
       --max-model-len 32768 \
       --max-num-seqs 256 \
       --gpu-memory-utilization 0.9 \
       --model "$MODEL_PATH" \
       --load-strategy ipc_snapshot \
       --dynamic-load-weight
