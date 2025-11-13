[简体中文](../zh/best_practices/ERNIE-4.5-0.3B-Paddle.md)

# ERNIE-4.5-0.3B
## Environmental Preparation
### 1.1 Hardware requirements
The minimum number of GPUs required to deploy `ERNIE-4.5-0.3B` on the following hardware for each quantization is as follows:

| | WINT8 | WINT4 | FP8 |
|-----|-----|-----|-----|
|H800 80GB| 1 | 1 | 1 |
|A800 80GB| 1 | 1 | / |
|H20 96GB| 1 | 1 | 1 |
|L20 48GB| 1 | 1 | 1 |
|A30 40GB| 1 | 1 | / |
|A10 24GB| 1 | 1 | / |

**Tips:**
1. To modify the number of deployment GPUs, specify `--tensor-parallel-size 2` in starting command.
2. For hardware not listed in the table, you can estimate whether it can be deployed based on the GPU memory.

### 1.2 Install fastdeploy
- Installation: For detail, please refer to [Fastdeploy Installation](../get_started/installation/README.md).

- Model Download，For detail, please refer to [Supported Models](../supported_models.md).

## 2.How to Use
### 2.1 Basic: Launching the Service
Start the service by following command:
```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-0.3B-Paddle \
       --tensor-parallel-size 1 \
       --quantization wint4 \
       --max-model-len 32768 \
       --max-num-seqs 128
```
- `--quantization`: indicates the quantization strategy used by the model. Different quantization strategies will result in different performance and accuracy of the model. It could be one of `wint8` / `wint4` / `block_wise_fp8`(Hopper is needed).
- `--max-model-len`: Indicates the maximum number of tokens supported by the currently deployed service. The larger the value, the longer the context length the model can support, but the more GPU memory is occupied, which may affect the concurrency.

For more parameter meanings and default settings, see [FastDeploy Parameter Documentation](../parameters.md)。

### 2.2 Advanced: How to get better performance
#### 2.2.1 Correctly set parameters that match the application scenario
Evaluate average input length, average output length, and maximum context length
- Set max-model-len according to the maximum context length. For example, if the average input length is 1000 and the output length is 30000, then it is recommended to set it to 32768

#### 2.2.2 Prefix Caching
**Idea:** The core idea of Prefix Caching is to avoid repeated calculations by caching the intermediate calculation results of the input sequence (KV Cache), thereby speeding up the response speed of multiple requests with the same prefix. For details, refer to [prefix-cache](../features/prefix_caching.md)

**How to enable:**
Since version 2.2 (including the develop branch), Prefix Caching has been enabled by default.

For versions 2.1 and earlier, you need to enable it manually by adding following lines to the startup parameters, where `--enable-prefix-caching` enables prefix caching, and `--swap-space` enables CPU cache in addition to GPU cache. The size is GB and should be adjusted according to the actual situation of the machine. The recommended value is `(total machine memory - model size) * 20%`. If the service fails to start because other programs are occupying memory, try reducing the `--swap-space` value.
```
--enable-prefix-caching
--swap-space 50
```

#### 2.2.3 Chunked Prefill
**Idea:** This strategy is adopted to split the prefill stage request into small-scale sub-chunks, and execute them in batches mixed with the decode request. This can better balance the computation-intensive (Prefill) and memory-intensive (Decode) operations, optimize GPU resource utilization, reduce the computational workload and memory usage of a single Prefill, thereby reducing the peak memory usage and avoiding the problem of insufficient memory. For details, please refer to [Chunked Prefill](../features/chunked_prefill.md)

**How to enable:**
Since version 2.2 (including the develop branch), Chunked Prefill has been enabled by default.

For versions 2.1 and earlier, you need to enable it manually by adding
```
--enable-chunked-prefill
```

#### 2.2.4 CudaGraph
**Idea:**
CUDAGraph is a GPU computing acceleration technology provided by NVIDIA. It achieves efficient execution and optimization of GPU tasks by capturing CUDA operation sequences into a graph structure. The core idea of CUDAGraph is to encapsulate a series of GPU computing and memory operations into a re-executable graph, thereby reducing CPU-GPU communication overhead, reducing kernel startup latency, and improving overall computing performance.

**How to enable:**
Before version 2.3, it needs to be enabled through `--use-cudagraph`.
CUDAGraph has been enabled by default in some scenarios at the beginning of version 2.3. CUDAGraph will be automatically closed for functions that are not compatible with CUDAGraph (speculative decoding, RL training, multi-mode model).
Notes:
- Usually, no additional parameters need to be set, but CUDAGraph will generate some additional memory overhead, which may need to be adjusted in some scenarios with limited memory. For detailed parameter adjustments, please refer to [GraphOptimizationBackend](../features/graph_optimization.md) for related configuration parameter descriptions

#### 2.2.5 Rejection Sampling
**Idea:**
Rejection sampling is to generate samples from a proposal distribution that is easy to sample, avoiding explicit sorting to increase the sampling speed, which has a significant improvement on small-sized models.

**How to enable:**
Add the following environment variables before starting
```
export FD_SAMPLING_CLASS=rejection
```

## FAQ
If you encounter any problems during use, you can refer to [FAQ](./FAQ.md).
