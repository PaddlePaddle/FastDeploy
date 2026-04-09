[简体中文](../zh/best_practices/DeepSeek-V3.md)

# DeepSeek-V3/V3.1 Model

## I. Environment Preparation

### 1.1 Support Requirements
The minimum number of GPUs required for deployment on the following hardware for each quantization precision of DeepSeek-V3/V3.1 is as follows:

| | WINT4 |

|-----|-----|

|H800 80GB| 8 |

### 1.2 Installing FastDeploy

Refer to the installation process document [FastDeploy GPU Installation](../get_started/installation/nvidia_gpu.md)

## II. How to Use

### 2.1 Basics: Starting the Service

**Example 1:** Deploying a Wint4 model with 16K context on an H800 with eight GPUs

```shell
MODEL_PATH=/models/DeepSeek/DeepSeek-V3.1-Terminus-BF16
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

**Example 2:** Deploying a 16K context service for the block_wise_fp8 model on 16 cards on an H800

```shell
# Currently only supports configurations with tp_size of 8 and ep_size of 16

MODEL_PATH=models/DeepSeek/DeepSeek-V3.1-Terminus-BF16

export FD_DISABLE_CHUNKED_PREFILL=1
export FD_ATTENTION_BACKEND="MLA_ATTN"
export FLAGS_flash_attn_version=3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FD_ENABLE_MULTI_API_SERVER=1

python -m fastdeploy.entrypoints.openai.multi_api_server \
    --ports "9811" \
    --num-servers 1 \
    --args --model "$MODEL_PATH" \
    --ips "10.95.247.24,10.95.244.147" \
    --no-enable-prefix-caching \
    --quantization block_wise_fp8 \
    --disable-sequence-parallel-moe \
    --tensor-parallel-size 8 \
    --num-gpu-blocks-override 1024 \
    --data-parallel-size 2 \
    --max-model-len 16384 \
    --enable-expert-parallel \
    --max-num-seqs 20 \
    --graph-optimization-config '{"use_cudagraph":true}'

```

**Example 3:** Deploying a 16-card block_wise_fp8 model service with 16K contexts on an H800

This example supports MLA computation using the FlashMLA operator

```shell
MODEL_PATH=models/DeepSeek/DeepSeek-V3.1-Terminus-BF16
export FD_DISABLE_CHUNKED_PREFILL=1
export FD_ATTENTION_BACKEND="MLA_ATTN"
export FLAGS_flash_attn_version=3
export USE_FLASH_MLA=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FD_ENABLE_MULTI_API_SERVER=1
python -m fastdeploy.entrypoints.openai.multi_api_server \
    --ports "9811,9812,9813,9814,9815,9816,9817,9818" \
    --num-servers 8 \
    --args --model "$MODEL_PATH" \
    --ips "10.95.246.220,10.95.230.91" \
    --no-enable-prefix-caching \
    --quantization block_wise_fp8 \
    --disable-sequence-parallel-moe \
    --tensor-parallel-size 1 \
    --num-gpu-blocks-override 1024 \
    --data-parallel-size 16 \
    --max-model-len 16384 \
    --enable-expert-parallel \
    --max-num-seqs 20 \
    --graph-optimization-config '{"use_cudagraph":true}'
```

# DeepSeek-V3.2 Model

## I. Environment Preparation

### 1.1 Support Requirements

The minimum number of GPUs required to deploy the DeepSeek-V3.2 model on the block_wise_fp8 platform under current quantization is as follows:

| | block_wise_fp8 |

|-----|-----|

|H800 80GB| 16 |

### 1.2 Installing FastDeploy

Refer to the installation process document [FastDeploy GPU Installation](../get_started/installation/nvidia_gpu.md)

## II. How to Use

### 2.1 Basics: Starting the Service

**Example 1:** Deploying an 8K context service for the block_wise_fp8 model on a 16-GPU H800

```shell
MODEL_PATH=/models/DeepSeek-V3.2-Exp-BF16
export FD_DISABLE_CHUNKED_PREFILL=1
export FD_ATTENTION_BACKEND="DSA_ATTN"
export FD_ENABLE_MULTI_API_SERVER=1


python -m fastdeploy.entrypoints.openai.multi_api_server \
       --ports "8091,8092,8093,8094,8095,8096,8097,8098" \
       --num-servers 8 \
       --args --model "$MODEL_PATH" \
       --ips "10.95.246.79,10.95.239.17" \
       --no-enable-prefix-caching \
       --quantization block_wise_fp8 \
       --disable-sequence-parallel-moe \
       --tensor-parallel-size 1 \
       --gpu-memory-utilization 0.85 \
       --max-num-batched-tokens 8192 \
       --data-parallel-size 16 \
       --max-model-len 8192 \
       --enable-expert-parallel \
       --max-num-seqs 20 \
       --num-gpu-blocks-override 2048 \
       --graph-optimization-config '{"use_cudagraph":false}' \
       --no-enable-overlap-schedule
```
