unset http_proxy
unset https_proxy
export ENABLE_V1_KVCACHE_SCHEDULER=1
export FD_DEBUG=1
export PYTHONPATH=/root/paddlejob/workspace/env_run/output/gongshaotian/baidu/paddle_internal/FastDeploy:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export SPECULATE_VERIFY_USE_TARGET_SAMPLING=1

rm -rf log
rm -rf core.*

config_yaml=./benchmarks/yaml/eb45-32k-wint2-tp4.yaml
model_path=/root/paddlejob/workspace/env_run/output/models/paddle/ERNIE-4.5-21B-A3B-Paddle
python -m fastdeploy.entrypoints.openai.api_server --config ${config_yaml} --model ${model_path} \
    --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 1 \
    --enable-chunked-prefill --enable-prefix-caching --port 8888 --max-num-batched-tokens 64 --metrics-port 8889 --engine-worker-queue-port 9999 \
    --graph-optimization-config '{"use_cudagraph": true}' \
    --routing-replay-config '{"enable_routing_replay":true, "routing_store_type":"local", "local_store_dir":"./routing_replay_output", "use_fused_put":false}' \
    # --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "num_model_steps": 1,"model": "'$model_path'/mtp"}' \


curl -X POST "http://0.0.0.0:8888/v1/chat/completions" -H "Content-Type: application/json" -d '{
    "messages": [
        {"role": "system", "content": "你是谁"}
    ] ,
    "temperature":0
  }'
