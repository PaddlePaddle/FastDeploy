[简体中文](../zh/best_practices/ERNIE-4.5-VL-424B-A47B-Paddle.md)

# ERNIE-4.5-VL-424B-A47B-Paddle

## 1. Environment Preparation
### 1.1 Support Status
The minimum number of cards required for deployment on the following hardware is as follows:

| Device [GPU Mem] | WINT4 | WINT8 | BFLOAT16 |
|:----------:|:----------:|:------:| :------:|
| H20 [144G] | 8 | 8 |  8 |
| A100 [80G] | 8 | 8 |  - |
| H800 [80G] | 8 | 8 |  - |

### 1.2 Install Fastdeploy

Installation process reference documentation [FastDeploy GPU Install](../get_started/installation/nvidia_gpu.md)

## 2.How to Use
### 2.1 Basic: Launching the Service
**Example 1:** Deploying a 128K context service on 8x H800 GPUs.
```shell
python -m fastdeploy.entrypoints.openai.api_server \
  --model baidu/ERNIE-4.5-VL-424B-A47B-Paddle \
  --port 8180 \
  --metrics-port 8181 \
  --engine-worker-queue-port 8182 \
  --tensor-parallel-size 8 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image": 100, "video": 100}' \
  --reasoning-parser ernie-45-vl \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 384 \
  --quantization wint4
```

An example is a set of configurations that can run stably while also delivering relatively good performance. If you have further requirements for precision or performance, please continue reading the content below.
### 2.2 Advanced: How to Achieve Better Performance

#### 2.2.1 Evaluating Application Scenarios and Setting Parameters Correctly
> **Context Length**
- **Parameters：** `--max-model-len`
- **Description：** Controls the maximum context length that the model can process.
- **Recommendation：** Longer context lengths may reduce throughput. Adjust based on actual needs, with a maximum supported context length of **128k** (131,072).

   ⚠️ Note: Longer context lengths will significantly increase GPU memory requirements. Ensure your hardware resources are sufficient before setting a longer context.
> **Maximum sequence count**
- **Parameters：** `--max-num-seqs`
- **Description：** Controls the maximum number of sequences the service can handle, supporting a range of 1 to 256.
- **Recommendation：** If you are unsure of the average number of sequences per request in your actual application scenario, we recommend setting it to **256**. If the average number of sequences per request in your application is significantly fewer than 256, we suggest setting it to a slightly higher value than the average to further reduce GPU memory usage and optimize service performance.

> **Multi-image and multi-video input**
- **Parameters**：`--limit-mm-per-prompt`
- **Description**：Our model supports multi-image and multi-video input in a single prompt. Please use this **Parameters** setting to limit the number of images/videos per request, ensuring efficient resource utilization.
- **Recommendation**：We recommend setting the number of images and videos in a single prompt to **100 each** to balance performance and memory usage.

> **Available GPU memory ratio during initialization**
- **Parameters：** `--gpu-memory-utilization`
- **Description：** Controls the available GPU memory for FastDeploy service initialization. The default value is 0.9, meaning 10% of the memory is reserved for backup.
- **Recommendation：** It is recommended to use the default value of 0.9. If an "out of memory" error occurs during stress testing, you may attempt to reduce this value.

#### 2.2.2 Chunked Prefill
- **Parameters：** `--enable-chunked-prefill`
- **Description：** Enabling `chunked prefill` can reduce peak GPU memory usage and improve service throughput. Version 2.2 has **enabled by default**; for versions prior to 2.2, you need to enable it manually—refer to the best practices documentation for 2.1.
- **Relevant configurations**:

    `--max-num-batched-tokens`：Limit the maximum number of tokens per chunk, with a recommended setting of 384.

#### 2.2.3  **Quantization precision**
- **Parameters：** `--quantization`

- **Supported precision types：**
  - wint4 (Suitable for most users)
  - wint8
  - bfloat16 (When the `--quantization` parameter is not set, bfloat16 is used by default.)

- **Recommendation：**
  - Unless you have extremely stringent precision requirements, we strongly recommend using wint4 quantization. This will significantly reduce memory consumption and increase throughput.
  - If slightly higher precision is required, you may try wint8.
  - Only consider using bfloat16 if your application scenario demands extreme precision, as it requires significantly more GPU memory.

#### 2.2.4  **Adjustable environment variables**
> **Rejection sampling：**`FD_SAMPLING_CLASS=rejection`
- **Description：** Rejection sampling involves generating samples from a proposal distribution that is easy to sample from, thereby avoiding explicit sorting and achieving an effect of improving sampling speed, which can enhance inference performance.
- **Recommendation：** This is a relatively aggressive optimization strategy that affects the results, and we are still conducting comprehensive validation of its impact. If you have high performance requirements and can accept potential compromises in results, you may consider enabling this strategy.

## 3. FAQ

### 3.1 Out of Memory
If the service prompts "Out of Memory" during startup, please try the following solutions:
1. Ensure no other processes are occupying GPU memory;
2. Use wint4/wint8 quantization and enable chunked prefill;
3. Reduce context length and maximum sequence count as needed.

If the service starts normally but later reports insufficient memory, try:
1. Adjust the initial GPU memory utilization ratio by modifying `--gpu-memory-utilization`.
