# FastDeploy参数说明

在使用FastDeploy部署模型（包括离线推理、服务化部署），涉及如下参数配置，其实需要注意，在使用离线推理时，各参数配置即为如下参数名；而在使用命令行启动服务时，相应参数中的分隔符需要从```_```修改为```-```，如```max_model_len```在命令行中则为```--max-model-len```。

| 参数名                                | 类型        | 说明 |
|:-----------------------------------|:----------| :----- |
| ```port```                         | `int`       | 仅服务化部署需配置，服务HTTP请求端口号，默认8000 |
| ```metrics_port```                 | `int`       | 仅服务化部署需配置，服务监控Metrics端口号，默认8001 |
| ```engine_worker_queue_port```     | `int`       | FastDeploy内部引擎进程通信端口, 默认8002 |
| ```cache_queue_port```             | `int`       | FastDeploy内部KVCache进程通信端口, 默认8003 |
| ```max_model_len```                | `int`       | 推理默认最大支持上下文长度，默认2048 |
| ```tensor_parallel_size```         | `int`       | 模型默认张量并行数，默认1 |
| ```data_parallel_size```           | `int`       | 模型默认数据并行数，默认1 |
| ```block_size```                   | `int`       | KVCache管理粒度(Token数)，推荐默认值64 |
| ```max_num_seqs```                 | `int`       | Decode阶段最大的并发数，默认为8 |
| ```mm_processor_kwargs```          | `dict[str]` | 多模态处理器参数配置，如：{"image_min_pixels": 3136, "video_fps": 2} |
| ```tokenizer```                    | `str`      | tokenizer 名或路径，默认为模型路径 |
| ```use_warmup```                   | `int`      | 是否在启动时进行warmup，会自动生成极限长度数据进行warmup，默认自动计算KV Cache时会使用 |
| ```limit_mm_per_prompt```          | `dict[str]` | 限制每个prompt中多模态数据的数量，如：{"image": 10, "video": 3}，默认都为1 |
| ```enable_mm```                    | `bool`      | __[已废弃]__ 是否支持多模态数据（仅针对多模模型），默认False |
| ```quantization```                 | `str`       | 模型量化策略，当在加载BF16 CKPT时，指定wint4或wint8时，支持无损在线4bit/8bit量化 |
| ```gpu_memory_utilization```       | `float`     | GPU显存利用率，默认0.9 |
| ```num_gpu_blocks_override```      | `int`       | 预分配KVCache块数，此参数可由FastDeploy自动根据显存情况计算，无需用户配置，默认为None |
| ```max_num_batched_tokens```       | `int`       | Prefill阶段最大Batch的Token数量，默认为None(与max_model_len一致) |
| ```kv_cache_ratio```               | `float`     | KVCache块按kv_cache_ratio比例分给Prefill阶段和Decode阶段, 默认0.75 |
| ```enable_prefix_caching```        | `bool`      | 是否开启Prefix Caching，默认False |
| ```swap_space```                   | `float`     | 开启Prefix Caching时，用于swap KVCache的CPU内存大小，单位GB，默认None |
| ```enable_chunked_prefill```         | `bool`      | 开启Chunked Prefill，默认False |
| ```max_num_partial_prefills```     | `int`       | 开启Chunked Prefill时，Prefill阶段的最大并发数，默认1 |
| ```max_long_partial_prefills```    | `int`       | 开启Chunked Prefill时，Prefill阶段并发中包启的最多长请求数，默认1 |
| ```long_prefill_token_threshold``` | `int`       | 开启Chunked Prefill时，请求Token数超过此值的请求被视为长请求，默认为max_model_len*0.04 |
| ```static_decode_blocks```         | `int`       | 推理过程中，每条请求强制从Prefill的KVCache分配对应块数给Decode使用，默认2|
| ```reasoning_parser```             | `str`       | 指定要使用的推理解析器，以便从模型输出中提取推理内容 |
| ```use_cudagraph```                | `bool`      | 是否使用cuda graph，默认False |
｜```graph_optimization_config```    | `str`       | 可以配置计算图优化相关的参数，默认值为'{"use_cudagraph":false, "graph_opt_level":0, "cudagraph_capture_sizes": null }' |
| ```enable_custom_all_reduce```     | `bool`      | 开启Custom all-reduce，默认False |
| ```splitwise_role```               | `str`       | 是否开启splitwise推理，默认值mixed， 支持参数为["mixed", "decode", "prefill"] |
| ```innode_prefill_ports```         | `str`       | prefill 实例内部引擎启动端口 （仅单机PD分离需要），默认值None |
| ```guided_decoding_backend```      | `str`       | 指定要使用的guided decoding后端，支持 `auto`、`xgrammar`、`off`, 默认为 `off` |
| ```guided_decoding_disable_any_whitespace``` | `bool`   | guided decoding期间是否禁止生成空格，默认False |
| ```speculative_config```           | `dict[str]` | 投机解码配置，仅支持标准格式json字符串，默认为None |
| ```dynamic_load_weight```          | `int`       | 是否动态加载权重，默认0 |
| ```enable_expert_parallel```       | `bool`      | 是否启用专家并行 |
| ```enable_logprob```       | `bool`      | 是否启用输出token返回logprob。如果未使用 logrpob，则在启动时可以省略此参数。 |

## 1. KVCache分配与```num_gpu_blocks_override```、```block_size```的关系？

FastDeploy在推理过程中，显存被```模型权重```、```预分配KVCache块```和```模型计算中间激活值```占用。其中预分配KVCache块由```num_gpu_blocks_override```决定，其单位为```block_size```(默认64），即一个块可以存储64个Token的KVCache。

在实际推理中，用户很难知道```num_gpu_blocks_override```到底该配置到多少合适，因此FastDeploy采用如下方式来自动推导并配置这个值，流程如下:
- 加载模型，在完成模型加载后，记录当前显存占用情况```total_memory_after_load```和FastDeploy框架占用的显存值```fd_memory_after_load```; 注意前者为GPU实际被占用显存（可能有其它进程也占用），后者是FD框架本身占用显存；

- 根据用户配置的```max_num_batched_tokens```(默认为```max_model_len```)，Fake相应长度的输入数据进行Prefill计算，记录当前FastDeploy框架显存最大分配值```fd_memory_after_prefill```，因此可以认为```模型计算中间激活值```为```fd_memory_after_prefill - fd_memory_after_load```;
  - 截止当前，认为GPU卡可以剩分配KVCache的显存(以A800 80G为例)为```80GB * gpu_memory_utilization - total_memory_after_load - (fd_memory_after_prefill - fd_memory_after_load)```
  - 根据模型KVCache的精度（如8bit/16bit)，计算一个block占用的KVCache大小，从而计算出总共可分配的block数量，赋值给```num_gpu_blocks_override```

> 在服务启动日志中，我们可以在log/fastdeploy.log中找到```Reset block num, the total_block_num:17220, prefill_kvcache_block_num:12915```，其中```total_block_num```即为自动计算出来的KVCache block数量，将其乘以```block_size```即可知道整个服务可以缓存多少Token的KV值。

## 2. ```kv_cache_ratio```、```block_size```、```max_num_seqs```的关系？
- FastDeploy里面将KVCache按照```kv_cache_ratio```分为Prefill阶段使用和Decode阶段使用，在配置这个参数时，可以按照```kv_cache_ratio = 平均输入Token数/(平均输入+平均输出Token数)```进行配置，常规情况输入是输出的3倍，因此可以配置成0.75
- ```max_num_seqs```是Decode阶段的最大并发数，一般而言可以配置成最大值128，但用户也可以根据KVCache情况作调用，例如输出的KVCache Token量为```decode_token_cache = total_block_num * (1 - kv_cache_ratio) * block_size```，为了防止极端情况下的显存不足问题，可以配置```max_num_seqs = decode_token_cache / 平均输出Token数```，不高于128即可。

## 3. ```enable_chunked_prefill```参数配置说明

当启用 `enable_chunked_prefill` 时，服务通过动态分块处理长输入序列，显著提升GPU资源利用率。在此模式下，原有 `max_num_batched_tokens` 参数不再约束预填充阶段的批处理token数量（限制单次prefill的token数量），因此引入 `max_num_partial_prefills` 参数，专门用于限制同时处理的分块批次数。

为优化短请求的调度优先级，新增 `max_long_partial_prefills` 与 `long_prefill_token_threshold` 参数组合。前者限制单个预填充批次中的长请求数量，后者定义长请求的token阈值。系统会优先保障短请求的批处理空间，从而在混合负载场景下降低短请求延迟，同时保持整体吞吐稳定。

## 4. GraphOptimizationBackend 相关配置参数说明
当前仅支持用户配置以下参数：
- `use_cudagraph` : bool = False
- `graph_optimization_config` :  Dict[str, Any]
  - `graph_opt_level`: int = 0
  - `use_cudagraph`: bool = False
  - `cudagraph_capture_sizes` : List[int] = None

可以通过设置 `--use-cudagraph` 或 `--graph-optimization-config '{"use_cudagraph":true}'` 开启 CudaGrpah。

`--graph-optimization-config` 中的 `graph_opt_level` 参数用于配置图优化等级，可选项如下：
- `0`: 动态图，默认为 0
- `1`: 静态图，初始化阶段会使用 Paddle API 将动态图转换为静态图
- `2`: 在静态图的基础上，使用 Paddle 框架编译器（CINN, Compiler Infrastructure for Neural Networks）进行编译优化

一般情况下静态图比动态图的 Kernel Launch 开销更小，推荐使用静态图。
对于已适配的模型，FastDeploy 的 CudaGraph **可同时支持动态图与静态图**。

在默认配置下开启 CudaGraph 时，会根据 `max_num_seqs` 参数自动设置 CudaGraph 需要捕获的 Batch Size 列表，需要捕获的 Batch Size 的列表自动生成逻辑如下：
1. 生成一个范围为 [1,1024] Batch Size 的候选列表

```
        # Batch Size [1, 2, 4, 8, 16, ... 120, 128]
        candidate_capture_sizes = [1, 2, 4] + [8 * i for i in range(1, 17)]
        # Batch Size (128, 144, ... 240, 256]
        candidate_capture_sizes += [16 * i for i in range(9, 17)]
        # Batch Size (256, 288, ... 992, 1024]
        candidate_capture_sizes += [32 * i for i in range(17, 33)]
```

2. 根据用户设置的 `max_num_seqs` 裁剪候选列表，得到范围为 [1, `max_num_seqs`] 的 CudaGraph 捕获列表。

用户也可以通过 `--graph-optimization-config` 中的 `cudagraph_capture_sizes` 参数自定义需要被 CudaGraph 捕获的 Batch Size 列表:

```
--graph-optimization-config '{"cudagraph_capture_sizes": [1, 3, 5, 7, 9]}'
```

### CudaGraph相关参数说明
使用 CudaGraph 会产生一些额外的显存开销，在FastDeploy中分为下面两类：
- 额外的输入 Buffer 开销
- CudaGraph 使用了专用的显存池，因此会持有一部分与主框架隔离的中间激活显存

FastDeploy 的初始化顺序为先使用 `gpu_memory_utilization` 参数计算 `KVCache` 可用的显存，初始化完 `KVCache` 之后才会使用剩余显存初始化 CudaGraph。由于 CudaGraph 目前还不是默认开启的，因此使用默认启动参数可能会遇到 `Out Of Memory` 错误，可以尝试使用下面三种方式解决：
- 调低 `gpu_memory_utilization` 的值，多预留一些显存给CudaGraph使用。
- 调低 `max_num_seqs` 的值，降低最大并发数。
- 通过 `graph_optimization_config` 自定义需要 CudaGraph 捕获的 Batch Size 列表 `cudagraph_capture_sizes`，减少捕获的图的数量

使用CudaGraph之前，需要确保加载的模型被装饰器 ```@support_graph_optimization```正确修饰。

  ```python
  # 1. import 装饰器
  from fastdeploy.model_executor.graph_optimization.decorator import support_graph_optimization
  ...

  # 2. 添加装饰器
  @support_graph_optimization
  class Ernie4_5_Model(nn.Layer): # 注意 decorator 加在 nn.Layer 的子类上
      ...

  # 3. 修改 ModelForCasualLM 子类中 self.model() 的传参方式
   class Ernie4_5_MoeForCausalLM(ModelForCasualLM):
      ...
      def forward(
          self,
          ids_remove_padding: paddle.Tensor,
          forward_meta: ForwardMeta,
      ):
          hidden_states = self.model(ids_remove_padding=ids_remove_padding, # 传参时指定参数名
                                     forward_meta=forward_meta)
          return hidden_statesfrom fastdeploy.model_executor.graph_optimization.decorator import support_graph_optimization
  ...

  @support_graph_optimization
  class Ernie45TModel(nn.Layer): # 注意 decorator 加在 nn.Layer 的子类上
      ...
  ```

- 当开启 ```use_cudagraph``` 时，暂时只支持单卡推理，即 ```tensor_parallel_size``` 设为1。
- 当开启 ```use_cudagraph``` 时，暂不支持开启 ```enable_prefix_caching``` 或 ```enable_chunked_prefill``` 。
