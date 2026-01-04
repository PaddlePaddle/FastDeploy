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

CARD_NUM=$1

if [[ "$CARD_NUM" == "1" ]]; then
    export HPU_VISIBLE_DEVICES=0
    export MODEL="ERNIE-4.5-21B-A3B-Paddle"
    export GPU_BLOCKS=5000
elif [[ "$CARD_NUM" == "8" ]]; then
    export HPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export MODEL="ERNIE-4.5-300B-A47B-Paddle"
    export GPU_BLOCKS=3000
else
    exit 0
fi

rm -rf log 2>/dev/null
ENABLE_V1_KVCACHE_SCHEDULER=1 FD_ENC_DEC_BLOCK_NUM=8 HPU_PERF_BREAKDOWN_SYNC_MODE=1 HPU_WARMUP_BUCKET=0 FD_ATTENTION_BACKEND=HPU_ATTN ENABLE_V1_KVCACHE_SCHEDULER=0 \
    python -m fastdeploy.entrypoints.openai.api_server --model ${MODEL} --port ${SERVER_PORT} \
    --engine-worker-queue-port ${ENGINE_WORKER_QUEUE_PORT} --metrics-port ${METRICS_PORT} \
    --cache-queue-port ${CACHE_QUEUE_PORT} --tensor-parallel-size ${CARD_NUM} --max-model-len 16384 \
    --max-num-seqs 128 --block-size 128  --kv-cache-ratio 0.5 --num-gpu-blocks-override ${GPU_BLOCKS} \
    --graph-optimization-config '{"use_cudagraph":false}'
