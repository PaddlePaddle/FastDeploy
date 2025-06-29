# FastDeploy Parameter Documentation

## Parameter Description

When using FastDeploy to deploy models (including offline inference and service deployment), the following parameter configurations are involved. Please note that for offline inference, the parameter configurations are the parameter names as shown below; while when starting the service via command line, the separators in the corresponding parameters need to be changed from ```_``` to ```-```, for example ```max_model_len``` becomes ```--max-model-len``` in command line.

| Parameter Name | Type | Description |
|:--------------|:----|:-----------|
| ```port``` | `int` | Only required for service deployment, HTTP service port number, default: 8000 |
| ```metrics_port``` | `int` | Only required for service deployment, metrics monitoring port number, default: 8001 |
| ```engine_worker_queue_port``` | `int` | FastDeploy internal engine communication port, default: 8002 |
| ```cache_queue_port``` | `int` | FastDeploy internal KVCache process communication port, default: 8003 |
| ```max_model_len``` | `int` | Default maximum supported context length for inference, default: 2048 |
| ```tensor_parallel_size``` | `int` | Default tensor parallelism degree for model, default: 1 |
| ```data_parallel_size``` | `int` | Default data parallelism degree for model, default: 1 |
| ```block_size``` | `int` | KVCache management granularity (Token count), recommended default: 64 |
| ```max_num_seqs``` | `int` | Maximum concurrent number in Decode phase, default: 8 |
| ```mm_processor_kwargs``` | `dict[str]` | Multimodal processor parameter configuration, e.g.: {"image_min_pixels": 3136, "video_fps": 2} |
| ```tokenizer``` | `str` | Tokenizer name or path, defaults to model path |
| ```use_warmup``` | `int` | Whether to perform warmup at startup, will automatically generate maximum length data for warmup, enabled by default when automatically calculating KV Cache |
| ```limit_mm_per_prompt``` | `dict[str]` | Limit the amount of multimodal data per prompt, e.g.: {"image": 10, "video": 3}, default: 1 for all |
| ```enable_mm``` | `bool` | Whether to support multimodal data (for multimodal models only), default: False |
| ```quantization``` | `str` | Model quantization strategy, when loading BF16 CKPT, specifying wint4 or wint8 supports lossless online 4bit/8bit quantization |
| ```gpu_memory_utilization``` | `float` | GPU memory utilization, default: 0.9 |
| ```num_gpu_blocks_override``` | `int` | Preallocated KVCache blocks, this parameter can be automatically calculated by FastDeploy based on memory situation, no need for user configuration, default: None |
| ```max_num_batched_tokens``` | `int` | Maximum batch token count in Prefill phase, default: None (same as max_model_len) |
| ```kv_cache_ratio``` | `float` | KVCache blocks are divided between Prefill phase and Decode phase according to kv_cache_ratio ratio, default: 0.75 |
| ```enable_prefix_caching``` | `bool` | Whether to enable Prefix Caching, default: False |
| ```swap_space``` | `float` | When Prefix Caching is enabled, CPU memory size for KVCache swapping, unit: GB, default: None |
| ```enable_chunk_prefill``` | `bool` | Enable Chunked Prefill, default: False |
| ```max_num_partial_prefills``` | `int` | When Chunked Prefill is enabled, maximum concurrent number of partial prefill batches, default: 1 |
| ```max_long_partial_prefills``` | `int` | When Chunked Prefill is enabled, maximum number of long requests in concurrent partial prefill batches, default: 1 |
| ```long_prefill_token_threshold``` | `int` | When Chunked Prefill is enabled, requests with token count exceeding this value are considered long requests, default: max_model_len*0.04 |
| ```static_decode_blocks``` | `int` | During inference, each request is forced to allocate corresponding number of blocks from Prefill's KVCache for Decode use, default: 2 |
| ```reasoning_parser``` | `str` | Specify the reasoning parser to extract reasoning content from model output |
| ```enable_static_graph_inference``` | `bool` | Whether to use static graph inference mode, default: False |
| ```use_cudagraph``` | `bool` | Whether to use cuda graph, default: False |
| ```max_capture_batch_size``` | `int` | When cuda graph is enabled, maximum batch size of captured cuda graph, default: 64 |
| ```splitwise_role``` | `str` | Whether to enable splitwise inference, default value: mixed, supported parameters: ["mixed", "decode", "prefill"] |
| ```innode_prefill_ports``` | `str` | Internal engine startup ports for prefill instances (only required for single-machine PD separation), default: None |
| ```guided_decoding_backend``` | `str` | Specify the guided decoding backend to use, supports `auto`, `xgrammar`, `off`, default: `off` |
| ```guided_decoding_disable_any_whitespace``` | `bool` | Whether to disable whitespace generation during guided decoding, default: False |
| ```speculative_config``` | `dict[str]` | Speculative decoding configuration, only supports standard format JSON string, default: None |
| ```dynamic_load_weight``` | `int` | Whether to enable dynamic weight loading, default: 0 |
| ```enable_expert_parallel``` | `bool` | Whether to enable expert parallel |


## 1. Relationship between KVCache allocation, ```num_gpu_blocks_override``` and ```block_size```?

During FastDeploy inference, GPU memory is occupied by ```model weights```, ```preallocated KVCache blocks``` and ```model computation intermediate activation values```. The preallocated KVCache blocks are determined by ```num_gpu_blocks_override```, with ```block_size``` (default: 64) as its unit, meaning one block can store KVCache for 64 Tokens.

In actual inference, it's difficult for users to know how to properly configure ```num_gpu_blocks_override```, so FastDeploy uses the following method to automatically derive and configure this value:

- Load the model, after completing model loading, record current memory usage ```total_memory_after_load``` and FastDeploy framework memory usage ```fd_memory_after_load```; note the former is actual GPU memory usage (may include other processes), the latter is memory used by FD framework itself;

- According to user-configured ```max_num_batched_tokens``` (default: ```max_model_len```), perform fake prefill computation with corresponding length input data, record current maximum FastDeploy framework memory allocation ```fd_memory_after_prefill```, thus ```model computation intermediate activation values``` can be considered as ```fd_memory_after_prefill - fd_memory_after_load```;
    - At this point, available GPU memory for KVCache allocation (taking A800 80G as example) is ```80GB * gpu_memory_utilization - total_memory_after_load - (fd_memory_after_prefill - fd_memory_after_load)```
    - Based on model KVCache precision (e.g. 8bit/16bit), calculate memory size per block, then calculate total allocatable blocks, assign to ```num_gpu_blocks_override```

> In service startup logs, we can find ```Reset block num, the total_block_num:17220, prefill_kvcache_block_num:12915``` in log/fastdeploy.log, where ```total_block_num``` is the automatically calculated KVCache block count, multiply by ```block_size``` to get total cacheable Tokens.

## 2. Relationship between ```kv_cache_ratio```, ```block_size``` and ```max_num_seqs```?
   - FastDeploy divides KVCache between Prefill and Decode phases according to ```kv_cache_ratio```. When configuring this parameter, you can use ```kv_cache_ratio = average input Tokens / (average input + average output Tokens)```. Typically input is 3x output, so can be configured as 0.75.
   - ```max_num_seqs``` is the maximum concurrency in Decode phase, generally can be set to maximum 128, but users can also configure based on KVCache situation, e.g. output KVCache Token amount is ```decode_token_cache = total_block_num * (1 - kv_cache_ratio) * block_size```, to prevent extreme OOM situations, can configure ```max_num_seqs = decode_token_cache / average output Tokens```, not exceeding 128.

## 3. ```enable_chunked_prefill``` parameter description

When `enable_chunked_prefill` is enabled, the service processes long input sequences through dynamic chunking, significantly improving GPU resource utilization. In this mode, the original `max_num_batched_tokens` parameter no longer constrains the batch token count in prefill phase (limiting single prefill token count), thus introducing `max_num_partial_prefills` parameter specifically to limit concurrently processed partial batches.

To optimize scheduling priority for short requests, new `max_long_partial_prefills` and `long_prefill_token_threshold` parameter combination is added. The former limits the number of long requests in single prefill batch, the latter defines the token threshold for long requests. The system will prioritize batch space for short requests, thereby reducing short request latency in mixed workload scenarios while maintaining stable throughput.

## 4. GraphOptimizationBackend related configuration parameters

### Static graph inference related parameters

- When ```enable_static_graph_inference``` is enabled, dynamic-to-static graph conversion will be performed, using static graph for inference.

### CudaGraph related parameters

For adapted models, FastDeploy's CudaGraph can support both dynamic and static graphs. Using CudaGraph incurs some additional memory overhead, divided into two categories in FastDeploy:
* Additional input Buffer overhead
* CudaGraph uses dedicated memory pool, thus holding some intermediate activation memory isolated from main framework

FastDeploy initialization sequence first uses `gpu_memory_utilization` parameter to calculate available memory for `KVCache`, after initializing `KVCache` then uses remaining memory to initialize CudaGraph. Since CudaGraph is not enabled by default currently, using default startup parameters may encounter `Out of memory` errors, can try following solutions:
* Lower `gpu_memory_utilization` value, reserve more memory for CudaGraph.
* Lower `max_capture_batch_size` value, reduce CudaGraph memory usage, but also reduce CudaGraph usage during inference.

- Before use, must ensure loaded model is properly decorated with ```@support_graph_optimization```.

  ```python
  # 1. import decorator
  from fastdeploy.model_executor.graph_optimization.decorator import support_graph_optimization
  ...

  # 2. add decorator
  @support_graph_optimization
  class Ernie4_5_Model(nn.Layer): # Note decorator is added to nn.Layer subclass
      ...

  # 3. modify parameter passing in ModelForCasualLM subclass's self.model()
   class Ernie4_5_MoeForCausalLM(ModelForCasualLM):
      ...
      def forward(
          self,
          ids_remove_padding: paddle.Tensor,
          forward_meta: ForwardMeta,
      ):
          hidden_states = self.model(ids_remove_padding=ids_remove_padding, # specify parameter name when passing
                                     forward_meta=forward_meta)
          return hidden_statesfrom fastdeploy.model_executor.graph_optimization.decorator import support_graph_optimization
  ...

  @support_graph_optimization
  class Ernie45TModel(nn.Layer): # Note decorator is added to nn.Layer subclass
      ...
  ```
- When ```use_cudagraph``` is enabled, currently only supports single-GPU inference, i.e. ```tensor_parallel_size``` set to 1.
- When ```use_cudagraph``` is enabled, cannot enable ```enable_prefix_caching``` or ```enable_chunk_prefill```.
- When ```use_cudagraph``` is enabled, batches with size ≤ ```max_capture_batch_size``` will be executed by CudaGraph, batches > ```max_capture_batch_size``` will be executed by original dynamic/static graph. To have all batch sizes executed by CudaGraph, ```max_capture_batch_size``` value should match ```max_num_seqs```. ```max_capture_batch_size``` > ```max_num_seqs``` will cause waste by capturing batches that won't be encountered during inference, occupying more time and memory.