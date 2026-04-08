[简体中文](../zh/best_practices/ERNIE-4.5-300B-A47B-Paddle.md)

# ERNIE-4.5-300B-A47B
## Environmental Preparation
### 1.1 Hardware requirements
The minimum number of GPUs required to deploy `ERNIE-4.5-300B-A47B` on the following hardware for each quantization is as follows:

| | WINT8 | WINT4 | FP8 | WINT2 | W4A8 |
|-----|-----|-----|-----|-----|-----|
|H800 80GB| 8 | 4 | 8 | 2 | 4 |
|A800 80GB| 8 | 4 | / | 2 | 4 |

**Tips:**
1. To modify the number of deployment GPUs, specify `--tensor-parallel-size 4` in starting command.
2. Since only 4-GPSs quantization scale is provided, the W4A8 model needs to be deployed on 4 GPUs.
3. For hardware not listed in the table, you can estimate whether it can be deployed based on the GPU memory.

### 1.2 Install fastdeploy
- Installation: For detail, please refer to [Fastdeploy Installation](../get_started/installation/README.md).

- Model Download，For detail, please refer to [Supported Models](../supported_models.md).

## 2.How to Use
### 2.1 Basic: Launching the Service
Start the service by following command:
```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-300B-A47B-Paddle \
       --tensor-parallel-size 8 \
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

#### 2.2.4 MTP (Multi-Token Prediction)
**Idea:**
By predicting multiple tokens at once, the number of decoding steps is reduced to significantly speed up the generation speed, while maintaining the generation quality through certain strategies. For details, please refer to [Speculative Decoding](../features/speculative_decoding.md)。

**How to enable:**
Add the following lines to the startup parameters
```
--speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "${path_to_mtp_model}"}'
```
Notes:
1. MTP currently does not support simultaneous use with Prefix Caching, Chunked Prefill, and CUDAGraph.
   - Use `export FD_DISABLE_CHUNKED_PREFILL=1` to disable Chunked Prefill.
   - When setting `speculative-config`, Prefix Caching will be automatically disabled.
2. MTP currently does not support service management global blocks, When setting `speculative-config`, service management global blocks will be automatically disabled.
3. MTP currently does not support rejection sampling, i.e. do not run with `export FD_SAMPLING_CLASS=rejection`

#### 2.2.5 W4A8C8 Quantization
**Idea:**
Quantization can achieve model compression, reduce GPU memory usage and speed up inference. To achieve better inference results, per-channel symmetric 4-bit quantization is used for MoE weights. static per-tensor symmetric 8-bit quantization is used for activation. And static per-channel symmetric 8-bit quantization is used for KVCache.

**How to enable:**
Just specify the corresponding model name in the startup command, `baidu/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle`
```
--model baidu/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle
```

Note:
- W4A8C8 quantized models are not supported when loaded via `--load-choices "default_v1"`.

#### 2.2.6 Rejection Sampling
**Idea:**
Rejection sampling is to generate samples from a proposal distribution that is easy to sample, avoiding explicit sorting to increase the sampling speed, which has a significant improvement on small-sized models.

**How to enable:**
Add the following environment variables before starting
```
export FD_SAMPLING_CLASS=rejection
```

#### 2.2.7 Disaggregated Deployment
**Idea:** Deploying Prefill and Decode separately in certain scenarios can improve hardware utilization, effectively increase throughput, and reduce overall sentence latency.

**How to enable:** Take the deployment of a single machine with 8 GPUs and 1P1D (4 GPUs each) as an example. Compared with the default hybrid deployment method, `--splitwise-role` is required to specify the role of the node. And the GPUs and logs of the two nodes are isolated through the environment variables `FD_LOG_DIR` and `CUDA_VISIBLE_DEVICES`.
```
export FD_LOG_DIR="log_prefill"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-300B-A47B-Paddle \
       --port 8180 --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --cache-queue-port 8183 \
       --tensor-parallel-size 4 \
       --quantization wint4 \
       --splitwise-role "prefill"
```
```
export FD_LOG_DIR="log_decode"
export CUDA_VISIBLE_DEVICES=4,5,6,7
# Note that innode-prefill-ports is specified as the Prefill serviceengine-worker-queue-port
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-300B-A47B-Paddle\
       --port 8184 --metrics-port 8185 \
       --engine-worker-queue-port 8186 \
       --cache-queue-port 8187 \
       --tensor-parallel-size 4 \
       --quantization wint4 \
       --innode-prefill-ports 8182 \
       --splitwise-role "decode"
```

#### 2.2.8 CUDAGraph
**Idea:**
CUDAGraph is a GPU computing acceleration technology provided by NVIDIA. It achieves efficient execution and optimization of GPU tasks by capturing CUDA operation sequences into a graph structure. The core idea of CUDAGraph is to encapsulate a series of GPU computing and memory operations into a re-executable graph, thereby reducing CPU-GPU communication overhead, reducing kernel startup latency, and improving overall computing performance.

**How to enable:**
Before version 2.3, it needs to be enabled through `--use-cudagraph`.
CUDAGraph has been enabled by default in some scenarios at the beginning of version 2.3. CUDAGraph will be automatically closed for functions that are not compatible with CUDAGraph (speculative decoding, RL training, multi-mode model).
Notes:
- Usually, no additional parameters need to be set, but CUDAGraph will generate some additional memory overhead, which may need to be adjusted in some scenarios with limited memory. For detailed parameter adjustments, please refer to [GraphOptimizationBackend](../features/graph_optimization.md) for related configuration parameter descriptions

## FAQ
If you encounter any problems during use, you can refer to [FAQ](./FAQ.md).
