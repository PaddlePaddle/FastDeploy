# export PYTHONPATH=/root/paddlejob/workspace/env_run/output/changwenbin/Sparse_GQA/baidu/paddle_internal/FastDeploy:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export LD_LIBRARY_PATH=/usr/local/nccl:$LD_LIBRARY_PATH
export NCCL_DEBUG=ERROR

rm -rf core*

# # export FLAGS_mla_use_tensorcore=1
# export FD_ATTENTION_BACKEND="MLA_ATTN"
# export FLAGS_flash_attn_version=3
# export FD_SAMPLING_CLASS=rejection
# export FD_USE_MACHETE=1
# export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
# export CUDA_COREDUMP_SHOW_PROGRESS=1

# export FD_MOE_BACKEND="marlin"
# export FD_DEBUG=1
# --cuda-graph-trace=node
# /opt/nvidia/nsight-compute/2025.1.0/ncu --set full -o mla_ncu -k MLAWithKVCacheKernel \
# --python-backtrace=cuda --python-sampling=true

# /ssd1/nvidia/nsight-systems/2025.5.1/bin/nsys start --output=binbin_launch --session=binbin

export FD_ATTENTION_BACKEND="MLA_ATTN"
export FLAGS_flash_attn_version=3
export FD_SAMPLING_CLASS=rejection
export FD_USE_MACHETE=1
# # /ssd1/nvidia/nsight-systems/2023.1.1/bin/nsys launch  --cuda-event-trace=true --cuda-memory-usage=true --cudabacktrace=all --dask=functions-trace --osrt-file-access=true  --trace=cuda,cublas,oshmem,ucx,osrt,cudnn --cuda-graph-trace=node  --session=binbin \
nohup python -m fastdeploy.entrypoints.openai.api_server \
  --model /ssd2/DeepSeek-V3.1-Terminus-BF16 \
  --port 8180 \
  --metrics-port 8181 \
  --engine-worker-queue-port 8182 \
  --cache-queue-port 8183 \
  --tensor-parallel-size 8 \
  --max-model-len  32768 \
  --max-num-seq 256 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 32768 \
  --quantization wint4 &


  # --load-choices default_v1 \
  #  --no-enable-prefix-caching \
  # --graph-optimization-config '{"use_cudagraph":false}' \
  # --graph-optimization-config '{"use_unique_memory_pool": true}' \
  # --max-num-batched-tokens 4096 \
  # --num-gpu-blocks-override 80000 \





# /ssd2/DeepSeek-V3-0324-bf16-5layers
# /ssd2/DeepSeek-V3.1-Terminus-BF16-5layers
# /ssd2/DeepSeek-V3.1-Terminus-BF16
# /ssd3/DeepSeek-V3.2-Exp-BF16-5layers
# /ssd3/DeepSeek-V3.2-Exp-BF16
# /ssd2/dsv3.1-terminous-L7

# /ssd2/safetensor_ckpt_step500




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
