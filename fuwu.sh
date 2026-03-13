# online_inference.sh
for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done

rm -rf log_eb
export FD_LOG_DIR=log_eb

# model_path="/root/paddlejob/tmpspace/models/paddle/benchmark/checkpoint-320-safetensors-PT"
# /root/paddlejob/tmpspace/models/torch/Qwen3-30B-A3B
# model_path="/root/paddlejob/tmpspace/models/paddle/ERNIE-4.5-21B-A3B-Paddle"
model_path="/raid0/ERNIE-4.5-21B-A3B-FP4"

export PYTHONPATH=/root/paddlejob/workspace/output/lizexu/FastDeploy:$PYTHONPATH


export FD_SAMPLING_CLASS=rejection
export INFERENCE_MSG_QUEUE_ID=8908

export FD_MOE_BACKEND="flashinfer-cutlass"

python -m fastdeploy.entrypoints.openai.api_server \
    --model $model_path \
    --port 8183 \
    --tensor-parallel-size 1 \
    --max-model-len  32768 \
    --enable-overlap-schedule \
    --num-gpu-blocks-override 1024 \
    --max-num-seqs 128 \
    --graph-optimization-config '{"use_cudagraph":false}'