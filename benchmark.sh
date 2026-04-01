export PYTHONPATH="/root/paddlejob/workspace/env_run/output/lizexu/FastDeploy":$PYTHONPATH

python benchmarks/benchmark_serving.py \
  --backend openai-chat \
  --model EB5 \
  --endpoint /v1/chat/completions \
  --ip-list 127.0.0.1:1211,127.0.0.1:1222,127.0.0.1:1223,127.0.0.1:1224,127.0.0.1:1225,127.0.0.1:1226,127.0.0.1:1227,127.0.0.1:1228 \
  --dataset-name EBChat \
  --dataset-path /raid0/dataset/dis_query_eb_0_32k_5k_converted.json \
  --hyperparameter-path benchmarks/yaml/request_yaml/eb45-32k.yaml \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 1000 \
  --max-concurrency 32 \
  --save-result
