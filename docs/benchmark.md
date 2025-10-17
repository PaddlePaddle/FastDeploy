[简体中文](zh/benchmark.md)

# Benchmark

FastDeploy extends the [vLLM benchmark](https://github.com/vllm-project/vllm/blob/main/benchmarks/) script with additional metrics, enabling more detailed performance benchmarking for FastDeploy.

## Benchmark Dataset

The following dataset is sourced from open-source data (original data from [HuggingFace Datasets](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json)):

| Dataset | Description |
| :------ | :---------- |
| https://fastdeploy.bj.bcebos.com/eb_query/filtered_sharedgpt_2000_input_1136_output_200_fd.json | Open-source dataset |

## How to Run

```
cd FastDeploy/benchmarks
python -m pip install -r requirements.txt

# Start service
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-0.3B-Base-Paddle \
       --port 8188 \
       --tensor-parallel-size 1 \
       --max-model-len 8192

# Run benchmark
python benchmark_serving.py \
  --backend openai-chat \
  --model baidu/ERNIE-4.5-0.3B-Base-Paddle \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port 8188 \
  --dataset-name EBChat \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --save-result
```
