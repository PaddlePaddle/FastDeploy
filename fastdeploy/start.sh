export PYTHONPATH=/root/paddlejob/workspace/env_run/output/changwenbin/swa/FastDeploy:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3,4
# export LD_LIBRARY_PATH=/usr/local/nccl:$LD_LIBRARY_PATH
export NCCL_DEBUG=ERROR

rm -rf core*

# export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
# export CUDA_COREDUMP_SHOW_PROGRESS=1

# export FD_DEBUG=1
# --cuda-graph-trace=node
# /opt/nvidia/nsight-compute/2025.1.0/ncu --set full -o mla_ncu -k MLAWithKVCacheKernel \
# --python-backtrace=cuda --python-sampling=true



##################################### DSK_TP ####################################################


# MODEL_PATH=/root/paddlejob/workspace/models/DeepSeek-V3.1-Terminus-BF16-5layers
MODEL_PATH=/root/paddlejob/workspace/models/DeepSeek-V3.2-Exp-BF16-5layers
# MODEL_PATH=/models/DeepSeek-V3.2-Exp-BF16
# MODEL_PATH=/root/paddlejob/workspace/models/Kimi-K2.5-Part

export FD_DISABLE_CHUNKED_PREFILL=1
export FD_ATTENTION_BACKEND="DSA_ATTN"
export FLAGS_flash_attn_version=3
# export FD_SAMPLING_CLASS=rejection

# # /ssd1/nvidia/nsight-systems/2023.1.1/bin/nsys launch  --cuda-event-trace=true --cuda-memory-usage=true --cudabacktrace=all --dask=functions-trace --osrt-file-access=true  --trace=cuda,cublas,oshmem,ucx,osrt,cudnn --cuda-graph-trace=node  --session=binbin \
# nohup
python -m fastdeploy.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --port 8180 \
  --metrics-port 8181 \
  --engine-worker-queue-port 8182 \
  --cache-queue-port 8183 \
  --tensor-parallel-size 2 \
  --max-model-len  8192 \
  --max-num-seq 1 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --num-gpu-blocks-override 2000 \
  --graph-optimization-config '{"use_cudagraph":false}' \
  --quantization wint4


  # --load-choices default_v1 \
  #  --no-enable-prefix-caching \
  # --graph-optimization-config '{"use_cudagraph":false}' \
  # --graph-optimization-config '{"use_unique_memory_pool": true}' \
  # --max-num-batched-tokens 4096 \

##################################### DSK_TP ####################################################



##################################### DSK_EP ####################################################


# MODEL_PATH=/models/DeepSeek-V3.2-Exp-BF16

# export FD_DISABLE_CHUNKED_PREFILL=1
# export FD_ATTENTION_BACKEND="MLA_ATTN"
# export FLAGS_flash_attn_version=3

# # 暂时只支持 tp_size为8，ep_size 为 16的 配置


# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export FD_ENABLE_MULTI_API_SERVER=1
# python -m fastdeploy.entrypoints.openai.multi_api_server \
#        --ports "9811" \
#        --num-servers 1 \
#        --args --model "$model_path" \
#        --ips "10.95.247.24,10.95.244.147" \
#        --no-enable-prefix-caching \
#        --quantization block_wise_fp8 \
#        --disable-sequence-parallel-moe \
#        --tensor-parallel-size 8 \
#        --num-gpu-blocks-override 1024 \
#        --data-parallel-size 2 \
#        --max-model-len 16384 \
#        --enable-expert-parallel \
#        --max-num-seqs 20 \
#        --graph-optimization-config '{"use_cudagraph":true}' \


##################################### DSK_EP ####################################################




##################################### wint2 ####################################################

# /root/paddlejob/workspace/env_run/output/changwenbin/nvidia/nsight-systems/2025.3.1/bin/nsys launch  --cuda-event-trace=true --cuda-memory-usage=true --cudabacktrace=all --trace=cuda,cublas,oshmem,ucx,osrt,cudnn --cuda-graph-trace=node  --session=binbin \
# nohup python -m fastdeploy.entrypoints.openai.api_server \
#   --model /ssd2/dsv3.1-terminous-L7 \
#   --port 8180 \
#   --metrics-port 8181 \
#   --engine-worker-queue-port 8182 \
#   --cache-queue-port 8183 \
#   --tensor-parallel-size 4 \
#   --no-enable-prefix-caching \
#   --graph-optimization-config '{"use_cudagraph":false}' \
#   --max-model-len  32768 \
#   --max-num-seq 256 &






  # --max-num-batched-tokens 2048 \
  # --gpu-memory-utilization 0.85 &
  # --load_choices default_v1 \
  # --graph-optimization-config '{"cudagraph_capture_sizes": [1,35]}' \
  # --use-cudagraph \
  # --graph-optimization-config '{"cudagraph_capture_sizes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}' \
  # --graph-optimization-config '{"use_cudagraph":true, "graph_opt_level":2}' \


  #/root/.cache/modelscope/hub/models/PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle
  # --enable-prefix-caching \
  # --enable-chunked-prefill \
  # MODEL_PATH=/model python -m coverage run --parallel-mode -m pytest /root/paddlejob/workspace/env_run/output/changwenbin/FastDeploy/tests/model_loader/test_common_model.py -s -vv




# ################################################## 21B ###################################################
# unset http_proxy
# unset https_proxy
# # export FD_DEBUG=1
# # rm -rf log
# # rm -rf core.*
# # import paddle
# # paddle.utils.run_check()

# export CUDA_VISIBLE_DEVICES=1
# config_yaml=../benchmarks/yaml/eb45-21b-a3b-32k-wint4.yaml
# model_path=/ssd2/ERNIE-4.5-21B-A3B-Paddle
# nohup python -m fastdeploy.entrypoints.openai.api_server --config ${config_yaml} --model ${model_path} \
#     --port 8888 --metrics-port 8889 --engine-worker-queue-port 8092 \
#     --graph-optimization-config '{"use_cudagraph":false}' &
