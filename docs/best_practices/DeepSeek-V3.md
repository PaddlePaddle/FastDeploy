[简体中文](../zh/best_practices/DeepSeek-V3.md)

# DeepSeek-V3/V3.1 Model

## I. Environment Preparation

### 1.1 Support Requirements
The minimum number of GPUs required for deployment on the following hardware for each quantization precision of DeepSeek-V3/V3.1 is as follows:

| | WINT4 |
|-----|-----|-----|
|H800 80GB| 8 |

### 1.2 Installing FastDeploy

Installation process reference document [FastDeploy GPU Installation](../get_started/installation/nvidia_gpu.md)

## II. How to Use

### 2.1 Basics: Starting the Service

**Example 1:** Deploying a Wint4 model 16K context service on an H800 with eight GPUs

```shell

MODEL_PATH=/models/DeepSeek-V3.2-Exp-BF16
export FD_DISABLE_CHUNKED_PREFILL=1
export FD_ATTENTION_BACKEND="MLA_ATTN"
export FLAGS_flash_attn_version=3

python -m fastdeploy.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port 8180 \
    --metrics-port 8181 \
    --engine-worker-queue-port 8182 \
    --cache-queue-port 8183 \
    --tensor-parallel-size 8 \
    --max-model-len 16384 \
    --max-num-seq 100 \
    --no-enable-prefix-caching \
    --quantization wint4

```
