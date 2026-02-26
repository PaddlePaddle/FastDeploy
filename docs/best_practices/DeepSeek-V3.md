[简体中文](../zh/best_practices/DeepSeek-V3.md)

# DeepSeek-V3/V3.1 Model

## I. Environment Preparation

### 1.1 Support Requirements
The minimum number of GPUs required for deployment on the following hardware for each quantization precision of DeepSeek-V3/V3.1 is as follows:

| | WINT8 | WINT4 | FP8 |
|-----|-----|-----|-----|
|H800 80GB| 16 | 8 | 16 |

**Note:**

1. Specify `--tensor-parallel-size 8` after the startup command to modify the number of GPUs required.
2. For hardware not listed in the table, estimate whether deployment is possible based on the available video memory.
3. Wint4 is recommended for quantization precision (can be deployed with 8 GPUs on a single machine).

### 1.2 Installing FastDeploy

Installation process reference document [FastDeploy GPU Installation](../get_started/installation/nvidia_gpu.md)

## II. How to Use

### 2.1 Basics: Starting the Service

**Example 1:** Deploying a Wint4 model 16K context service on an H100 with eight GPUs

```shell

MODEL_PATH=/models/DeepSeek-V3.2-Exp-BF16
export FD_DISABLE_CHUNKED_PREFILL=1
export FD_ATTENTION_BACKEND="MLA_ATTN"
export FLAGS_flash_attn_version=3

python -m fastdeploy.entrypoints.openai.api_server \
--model "$MODEL_PATH" \
--port 8180 \
--metrics-port 8181 \
--engine-worker-queue-port 8182
--cache-queue-port 8183
--tensor-parallel-size 8
--max-model-len 16384
--max-num-seq 100
--no-enable-prefix-caching
--quantization wint4

```

Where:

- `--quantization`: Indicates the quantization strategy used by the model. Different quantization strategies will result in different model performance and accuracy. Optional values ​​include: `wint8` / `wint4` / `wfp8afp8` (requires Hopper architecture).

- `--max-model-len`: Indicates the maximum number of tokens supported by the currently deployed service. A larger setting allows for a larger context length that the model can support, but also consumes more GPU memory, potentially affecting concurrency.

For more parameter meanings and default settings, please refer to [FastDeploy Parameter Description](../parameters.md).

### 2.2 Advanced: How to Achieve Better Performance

#### 2.2.1 Evaluate Application Scenarios and Set Parameters Correctly

Evaluate the average input length, average output length, and maximum context length based on the application scenario. For example, if the average input length is 1000 and the output length is 30000, then it is recommended to set it to 32768.

- Set `max-model-len` according to the maximum context length.

#### 2.2.2 Prefix Caching

**Principle:** The core idea of ​​Prefix Caching is to avoid redundant calculations by caching intermediate computation results of the input sequence (KV Cache), thereby accelerating the response speed of multiple requests with the same prefix. See [prefix-cache](../features/prefix_caching.md) for details.

**Enabling:** From version 2.2 onwards (including the develop branch), Prefix Caching is enabled by default. Use `--no-enable-prefix-caching` to disable prefix caching.

For versions 2.1 and earlier, it needs to be enabled manually. The `--enable-prefix-caching` option enables prefix caching, and `--swap-space` enables additional CPU caching on top of GPU caching, with a size of GB. This should be adjusted based on the actual machine requirements. A recommended value is `(total machine memory - model size) * 20%`. If service startup fails due to other programs consuming memory, try reducing the value of `--swap-space`.

```
--enable-prefix-caching

--swap-space 50

```

#### 2.2.3 Chunked Prefill

**Principle:** This uses a chunking strategy, breaking down prefill requests into small subtasks and batch-processing them with decoding requests. This better balances computationally intensive (prefill) and memory-intensive (decode) operations, optimizes GPU resource utilization, reduces the computational load and memory usage of a single prefill, thereby reducing peak memory usage and avoiding memory shortages. For details, please refer to [Chunked Prefill](../features/chunked_prefill.md)

**Enabling Method:** Starting with version 2.2 (including the develop branch), Chunked Prefill is enabled by default. Use `export FD_DISABLE_CHUNKED_PREFILL=1` to disable Chunked Prefill.

For versions 2.1 and earlier, it needs to be enabled manually.

``` --enable-chunked-prefill

```

#### 2.2.4 CUDAGraph

**Principle:**
CUDAGraph is a GPU computing acceleration technology provided by NVIDIA. It achieves efficient execution and optimization of GPU tasks by capturing CUDA operation sequences into a graph structure. The core idea of ​​CUDAGraph is to encapsulate a series of GPU computations and memory operations into a repeatable graph, thereby reducing CPU-GPU communication overhead, lowering kernel startup latency, and improving overall computing performance.

**Enabling Method:** Prior to version 2.3, CUDAGraph needed to be enabled via `--use-cudagraph`.

Starting from version 2.3, CUDAGraph is enabled by default in some scenarios. For features not yet compatible with CUDAGraph (speculation decoding, reinforcement learning training, multi-modal model inference), CUDAGraph will be automatically disabled.

Note:
- Generally, no additional parameters need to be set, but CUDAGraph incurs some additional GPU memory overhead. Adjustments may be necessary in scenarios with limited GPU memory. For detailed parameter adjustments, please refer to the relevant configuration parameter descriptions in [GraphOptimizationBackend](../features/graph_optimization.md).

#### 2.2.5 Rejection Sampling

**Principle:** Rejection sampling generates samples from an easily sampled proposal distribution, avoiding explicit sorting and thus improving sampling speed. This has a significant improvement for small-sized models.

**Enablement Method:** Add the following environment variables before startup:

``` export FD_SAMPLING_CLASS=rejection

```
## III. Frequently Asked Questions (FAQ) If you encounter any problems during use, you can refer to [FAQ](./FAQ.md).
