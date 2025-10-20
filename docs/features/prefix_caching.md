[简体中文](../zh/features/prefix_caching.md)

# Prefix Caching

Prefix Caching is a technique to optimize the inference efficiency of generative models. Its core idea is to cache intermediate computation results (KV Cache) of input sequences, avoiding redundant computations and thereby accelerating response times for multiple requests sharing the same prefix.

**How It Works**

Prefix Identification: When multiple requests share identical input prefixes (e.g., prompts or initial context), the system caches the intermediate states (KV Cache) corresponding to that prefix.

Incremental Computation: For subsequent requests, only the newly added portions (e.g., user-appended input) need computation while reusing cached intermediate results, significantly reducing computational overhead.

## Enabling Prefix Caching for Service Deployment

To enable prefix caching when launching the service, add the parameter `enable-prefix-caching`. By default, only first-level caching (GPU cache) is enabled.

To enable CPU caching, specify the `swap-space` parameter to allocate CPU cache space (in GB). The size should be set based on available machine memory after model loading.

> Note: The ERNIE-4.5-VL multimodal model currently does not support prefix caching.

For detailed parameter descriptions, refer to the [Parameters Documentation](../parameters.md).

Example launch command:

```shell
python -m fastdeploy.entrypoints.openai.api_server \
       --model "baidu/ERNIE-4.5-21B-A3B-Paddle" \
       --port 8180 --engine-worker-queue-port 8181 \
       --metrics-port 8182 \
       --cache-queue-port 8183 \
       --enable-prefix-caching \
       --swap-space 50 \
       --max-model-len 8192 \
       --max-num-seqs 32
```

## Enabling Prefix Caching for Offline Inference

Set `enable_prefix_caching=True` when launching FastDeploy. Enable CPU caching via `swap_space` based on available machine memory.

A test example is provided: `demo/offline_prefix_caching_demo.py`
