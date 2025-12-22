#!/bin/bash
export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
export GC_KERNEL_PATH=/usr/local/lib/python3.10/dist-packages/paddle_custom_device/intel_hpu/libcustom_tpc_perf_lib.so:$GC_KERNEL_PATH
export INTEL_HPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PADDLE_DISTRI_BACKEND=xccl
export PADDLE_XCCL_BACKEND=intel_hpu
# export FLAGS_intel_hpu_recipe_cache_config=/tmp/recipe,false,10240
export FLAGS_intel_hpu_recipe_cache_num=20480
export SERVER_PORT=8188
export ENGINE_WORKER_QUEUE_PORT=8002
export METRICS_PORT=8001
export CACHE_QUEUE_PORT=8003
export HABANA_PROFILE=0
export HPU_VISIBLE_DEVICES=0
rm -rf log 2>/dev/null
ENABLE_V1_KVCACHE_SCHEDULER=1 FD_ENC_DEC_BLOCK_NUM=8 HPU_PERF_BREAKDOWN_SYNC_MODE=1 HPU_WARMUP_BUCKET=1 HPU_WARMUP_MODEL_LEN=4096 FD_ATTENTION_BACKEND=HPU_ATTN \
    python -m fastdeploy.entrypoints.openai.api_server \
    --model ERNIE-4.5-21B-A3B-Paddle \
    --port ${SERVER_PORT} \
    --engine-worker-queue-port ${ENGINE_WORKER_QUEUE_PORT} \
    --metrics-port ${METRICS_PORT} \
    --cache-queue-port ${CACHE_QUEUE_PORT} \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --block-size 128 \
    --num-gpu-blocks-override 3100 \
    --kv-cache-ratio 0.991 \
    --no-enable-prefix-caching \
    --graph-optimization-config '{"use_cudagraph":false}'

# (2k + 1k) / 128(block_size) * 128(batch) = 3072
# export HPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# rm -rf log 2>/dev/null
# ENABLE_V1_KVCACHE_SCHEDULER=1 FD_ENC_DEC_BLOCK_NUM=8 HPU_PERF_BREAKDOWN_SYNC_MODE=1 HPU_WARMUP_BUCKET=1 HPU_WARMUP_MODEL_LEN=3072 FD_ATTENTION_BACKEND=HPU_ATTN \
#     python -m fastdeploy.entrypoints.openai.api_server \
#     --model ERNIE-4.5-300B-A47B-Paddle \
#     --port ${SERVER_PORT} \
#     --engine-worker-queue-port ${ENGINE_WORKER_QUEUE_PORT} \
#     --metrics-port ${METRICS_PORT} \
#     --cache-queue-port ${CACHE_QUEUE_PORT} \
#     --tensor-parallel-size 8 \
#     --max-model-len 32768 \
#     --max-num-seqs 128 \
#     --block-size 128 \
#     --num-gpu-blocks-override 3100 \
#     --kv-cache-ratio 0.991 \
#     --no-enable-prefix-caching \
#     --graph-optimization-config '{"use_cudagraph":false}'
