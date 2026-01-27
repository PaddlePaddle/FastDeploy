# !/bin/bash

# unset PADDLE_TRAINER_ENDPOINTS
# unset DISTRIBUTED_TRAINER_ENDPOINTS
unset http_proxy
unset https_proxy
# data_path=/root/paddlejob/workspace/env_run/output/changwenbin/FastDeploy/benchmarks/0724_ShareGPT_range_1800_3000_num_5000_FD.jsonl
# data_path=/root/paddlejob/workspace/env_run/output/changwenbin/Sparse_GQA/baidu/paddle_internal/FastDeploy/benchmarks/0419_demo
# data_path=/root/paddlejob/workspace/env_run/output/changwenbin/baidu/paddle_internal/FastDeploy/benchmarks/44K_data
data_path=/root/paddlejob/workspace/env_run/output/changwenbin/baidu/paddle_internal/FastDeploy/benchmarks/ruler_mk2_32k_with_answer.jsonl
yaml_path=/root/paddlejob/workspace/env_run/output/changwenbin/baidu/paddle_internal/FastDeploy/benchmarks/yaml/request_yaml/eb-128k.yaml
# request_yaml=yaml/request_yaml/eb45-32k.yaml
# data_path=/root/paddlejob/workspace/env_run/output/changwenbin/FastDeploy/benchmarks/0419_api9_yiyan_spv5_forqianfan_4872_fd
# data_path=/root/paddlejob/workspace/env_run/output/changwenbin/FastDeploy/benchmarks/0419_demo



# server_ip=0.0.0.0
# python /root/paddlejob/workspace/env_run/output/changwenbin/FastDeploy/benchmarks/benchmark_serving.py \
#     --backend openai-chat \
#     --model /model/DeepSeek-V3.2-Exp \
#     --endpoint /v1/chat/completions \
#     --host ${server_ip} \
#     --port 8000 \
#     --hyperparameter-path ${yaml_path} \
#     --dataset-name EBChat \
#     --dataset-path ${data_path} \
#     --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
#     --metric-percentiles 80,95,99,99.9,99.95,99.99 \
#     --num-prompts 1000 \
#     --max-concurrency 256 \
#     --drop-ratio 0.2 \
#     --save-result --debug  > vllm_dsk32_TP8 2>&1

server_ip=0.0.0.0
python /root/paddlejob/workspace/env_run/output/changwenbin/baidu/paddle_internal/FastDeploy/benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --model default \
    --endpoint /v1/chat/completions \
    --host ${server_ip} \
    --port 8179 \
    --hyperparameter-path ${yaml_path} \
    --dataset-name EBChat \
    --dataset-path ${data_path} \
    --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
    --metric-percentiles 80,95,99,99.9,99.95,99.99 \
    --num-prompts 20 \
    --max-concurrency 2 \
    --drop-ratio 0.2 \
    --save-result --debug > EB5indexdecode 2>&1



# unset http_proxy
# unset https_proxy
# data_path=/root/paddlejob/workspace/env_run/output/changwenbin/FastDeploy/benchmarks/test_data.json
# # dataset=/root/paddlejob/workspace/env_run/output/changwenbin/FastDeploy/benchmarks/1000_repeat.json
# request_yaml=yaml/request_yaml/eb45-32k.yaml
# python benchmark_serving.py \
#   --backend openai-chat \
#   --model EB45T \
#   --endpoint /v1/chat/completions \
#   --host 0.0.0.0 \
#   --port 8888 \
#   --dataset-name EBChat \
#   --dataset-path ${data_path} \
#   --hyperparameter-path ${request_yaml} \
#   --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
#   --metric-percentiles 80,95,99,99.9,99.95,99.99 \
#   --num-prompts 500 \
#   --max-concurrency 35 \
#   --drop-ratio 0.2 \
#   --save-result --debug > "21b_prefix_cache_test" 2>&1


# server_ip=127.0.0.1
# python /root/paddlejob/workspace/env_run/output/changwenbin/FastDeploy/benchmarks/benchmark_serving.py \
#     --backend openai-chat \
#     --model EB \
#     --endpoint /v1/chat/completions \
#     --host ${server_ip} \
#     --port 8888 \
#     --dataset-name EBChat \
#     --dataset-path ${data_path} \
#     --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
#     --metric-percentiles 80,95,99,99.9,99.95,99.99 \
#     --num-prompts 1 \
#     --max-concurrency 1 \
#     --save-result --debug > eb_test1 2>&1

