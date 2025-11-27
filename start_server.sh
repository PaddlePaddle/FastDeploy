

export PYTHONPATH=$PWD:$PYTHONPATH
rm -rf log/*

export CUDA_VISIBLE_DEVICES=2
# export FD_SAMPLING_CLASS=rejection
# export PADDLE_COMPATIBLE_API=true
export FD_USE_DEEP_GEMM=1
python -m fastdeploy.entrypoints.openai.api_server \
    --model /workspace3/chenjianye/models/ERNIE-4.5-21B-A3B-Paddle/ \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --load-choices "default_v1" \
    --graph-optimization-config '{"use_cudagraph":false}' \
    --port 8188 \
    --quantization "block_wise_fp8" \
    # --num-gpu-blocks-override 5000

# export CUDA_VISIBLE_DEVICES=2,3
# python -m fastdeploy.entrypoints.openai.api_server \
#     --model /root/PaddlePaddle/ERNIE-4.5-0.3B-PT  \
#     --tensor-parallel-size 2 \
#     --max-model-len 32768 \
#     --max-num-seqs 128 \
#     --load-choices "default_v1" \
#     --graph-optimization-config '{"use_cudagraph":true}' \
#     --port 8188 \
#     --quantization "block_wise_fp8"
#     # --disable-custom-all-reduce



# export FD_USE_DEEP_GEMM=1
# python -m fastdeploy.entrypoints.openai.api_server \
#     --model /workspace3/chenjianye/models/ERNIE-4.5-300B-A47B-Paddle/ \
#     --tensor-parallel-size 1\
#     --max-model-len 32768 \
#     --max-num-seqs 128 \
#     --load-choices "default_v1" \
#     --graph-optimization-config '{"use_cudagraph":false}' \
#     --port 8188 \
#     --quantization "block_wise_fp8" \
#     --enable-expert-parallel \
#     --data-parallel-size 4  \
#     --engine-worker-queue-port "6077,6078,6079,6080"