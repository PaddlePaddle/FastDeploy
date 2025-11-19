[简体中文](../zh/best_practices/PaddleOCR-VL-0.9B.md)

# PaddleOCR-VL-0.9B

## 1. Environment Preparation
### 1.1 Support Status
Recommended Hardware Configuration:
- GPU Memory: 8GB or more
- Shared Memory: 4GB or more

### 1.2 Install Fastdeploy

Installation process reference documentation [FastDeploy GPU Install](../get_started/installation/nvidia_gpu.md)

## 2.How to Use
### 2.1 Basic: Launching the Service
**Example 1:** Deploying a 16K Context Service on a Single RTX 3060 GPU
```shell
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/PaddleOCR-VL \
    --port 8185 \
    --metrics-port 8186 \
    --engine-worker-queue-port 8187 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 256
```

**Example 2:** Deploying a 16K Context Service on a Single RTX 4090 GPU
```shell
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/PaddleOCR-VL \
    --port 8185 \
    --metrics-port 8186 \
    --engine-worker-queue-port 8187 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.7 \
    --max-num-seqs 256
```

**Example 3:** Deploying a 16K Context Service on a Single A100 GPU
```shell
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/PaddleOCR-VL \
    --port 8185 \
    --metrics-port 8186 \
    --engine-worker-queue-port 8187 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.7 \
    --max-num-seqs 256
```

An example is a set of configurations that can run stably while also delivering relatively good performance. If you have further requirements for precision or performance, please continue reading the content below.
### 2.2 Advanced: How to Achieve Better Performance

#### 2.2.1 Evaluating Application Scenarios and Setting Parameters Correctly
> **Context Length**
- **Parameters：** `--max-model-len`
- **Description：** Controls the maximum context length that the model can process.
- **Recommendation：** Longer context lengths may reduce throughput. Adjust based on actual needs, with a maximum supported context length of **16k** (16,384) for `PaddleOCR-VL-0.9B`.

   ⚠️ Note: Longer context lengths will significantly increase GPU memory requirements. Ensure your hardware resources are sufficient before setting a longer context.
> **Maximum sequence count**
- **Parameters：** `--max-num-seqs`
- **Description：** Controls the maximum number of sequences the service can handle, supporting a range of 1 to 256.
- **Recommendation：** If you are unsure of the average number of sequences per request in your actual application scenario, and with sufficient GPU memory, we recommend setting it to **256**. If the average number of sequences per request in your application is significantly fewer than 256, or with insufficient GPU memory， we suggest setting it to a slightly higher value than the average to further reduce GPU memory usage and optimize service performance.

> **Available GPU memory ratio during initialization**
- **Parameters：** `--gpu-memory-utilization`
- **Description：** Controls the available GPU memory for FastDeploy service initialization. The default value is 0.9, meaning 10% of the memory is reserved for backup.
- **Recommendation：** It is recommended to use 0.7. If an "out of memory" error occurs during stress testing, you may attempt to reduce this value.

#### 2.2.2 Chunked Prefill
- **Parameters：** `--max-num-batched-tokens`
- **Description：**  Limits the maximum number of tokens per chunk in `chunked prefill`.
- **Recommendation：** We recommend setting it to 16384, which means disables chunked prefill.

#### 2.2.3  **Adjustable environment variables**
> **Flash Attention 3：**`FLAGS_flash_attn_version=3`
- **Description：** Enable the Flash Attention 3 algorithm. This feature is only supported on the latest generation of NVIDIA GPUs, such as H-series cards (Hopper architecture, e.g., H800) and B-series cards (Blackwell architecture, e.g., B200).
- **Recommendation：**On hardware that supports this feature, Flash Attention 3 can significantly enhance performance without typically affecting model accuracy. It is highly recommended to enable this feature.

> **Rejection sampling：**`FD_SAMPLING_CLASS=rejection`
- **Description：** Rejection sampling involves generating samples from a proposal distribution that is easy to sample from, thereby avoiding explicit sorting and achieving an effect of improving sampling speed, which can enhance inference performance.
- **Recommendation：** This is a relatively aggressive optimization strategy that affects the results, and we are still conducting comprehensive validation of its impact. If you have high performance requirements and can accept potential compromises in results, you may consider enabling this strategy.

## 3. FAQ

### 3.1 Out of Memory
If the service prompts "Out of Memory" during startup, please try the following solutions:
1. Ensure no other processes are occupying GPU memory;
2. Reduce context length and maximum sequence count as needed.

If the service starts normally but later reports insufficient memory, try:
1. Adjust the initial GPU memory utilization ratio by modifying `--gpu-memory-utilization`.
