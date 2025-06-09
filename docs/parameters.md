# 参数说明

## 服务启动参数

|         字段名         | 字段类型 |                       说明                       | 是否必填 |   默认值   |
| :---------------------: | :------: | :-----------------------------------------------: | :------: | :---------: |
|          model          |   str   |                     模型路径                     |    是    |  llama-7b  |
|        tokenizer        |   str   |                  tokenizer的地址                  |    否    |  模型地址  |
|      max_model_len      |   int   |            模型支持的最长的上下文长度            |    否    |    2048    |
|  tensor_parallel_size  |   int   |                   tensor 并行度                   |    否    |      1      |
|       block_size       |   int   |               每个block的token数量               |    否    |     64     |
|          task          |   str   |     任务类型，目前仅支持generate：token 返回     |    否    |  generate  |
|      max_num_seqs      |   int   |                同时推理的最大条数                |    否    |      8      |
|   mm_processor_kwargs   |   dict   |                  多模态输入参数                  |    否    |    None    |
| gpu_memory_utilization |  float  |        最大显存利用率，用于计算block 数目        |    否    |     0.9     |
| num_gpu_blocks_override |   int   |        设置分配的gpu的KV Cache的block 数目        |    否    |    None    |
| max_num_batched_tokens |   int   |         单次支持的最大prefill的token 数目         |    否    |    None    |
|     kv_cache_ratio     |  float  | 模型输入的长度 / 模型支持的最长的上下文长度的比例 |    否    |    0.75    |
|          nnode          |   int   |                     节点数量                     |    否    |      1      |
|         pod_ips         |   str   |                   各个节点的ip                   |    否    |    None    |
|       use_warmup       |   bool   |                   是否进行预热                   |    否    |    False    |
|  enable_prefix_caching  |   bool   |                 是否开启前缀缓存                 |    否    |    False    |
|       enabe_mm          |   bool   |                 是否开启多模态                 |    否    |    False    |

## 请求参数

|         字段名         | 字段类型 |                       说明                       | 是否必填 |   默认值   |
| :---------------------: | :------: | :-----------------------------------------------: | :------: | :---------: |
|        request_id        |   str   |                     请求id                     |    是    |  None  |
|        prompt        |   str   |                     输入prompt                     |    是    |  None  |
|  prompt_token_ids  |   list[int]   |                     输入prompt的token id                     |    否    |  None  |
|  prompt_token_ids_len  |   int   |                     输入prompt的token id的长度                     |    否    |  None  |
|        messages        |   list[list[dict[str, Any]]]   |             上下文对话信息                     |    否    |  None  |
|        history        |   list[list[str]]   |             历史对话信息                     |    否    |  None  |
|        system        |   str   |                     系统prompt                     |    否    |  None  |
|        sampling_params        |   SamplingParams   |                     推理超参设置（具体参数说明见下表）               |    是    |  None  |
|        eos_token_ids        |   list[int]   |                     结束token id                     |    否    |  None  |
|        arrival_time        |   float   |                     请求到达时间                     |    是    |  None  |
|        preprocess_start_time        |   float   |                     预处理开始时间                     |    否    |  None  |
|        preprocess_end_time        |   float   |                     预处理结束时间                     |    否    |  None  |
|        multi_modal_inputs        |   dict   |                     多模态输入  （目前不支持）                   |    否    |  None  |

### 推理参数Sampling Parameters

|       字段名        |      字段类型      |                             说明                             | 是否必填 |    默认值    |
|---------------------|-------------------|------------------------------------------------------------|----------|-------------|
| `n`                | int          | 需要返回的生成序列数量（当前仅支持1）                                      | 是       | -           |
| `presence_penalty` | float             |    话题新鲜度           | 否       | -           |
| `frequency_penalty`| float             |     频率惩罚度        | 否       | -           |
| `repetition_penalty`| float             |     重复词或短语的惩罚系数   | 否       | -           |
| `temperature`      | float             | 表示输出的确定性                    | 否       | -           |
| `top_p`            | float       | 仅考虑累积概率超过此值的候选词                              | 否       | 1           |
| `seed`             | int         | 控制生成随机性的种子                                        | 否       | -           |
| `stop`             | list[str]         | 生成遇到这些字符串时停止（结果不包含它们）                  | 否       | -           |
| `stop_token_ids`    | list[int]     | 生成遇到这些token时停止（结果包含token，除非是特殊token）   | 否       | -           |
| `bad_words`        | list[int]         | 禁止生成的token id                    | 否       | None          |
| `max_tokens`       | int           | 每个序列生成的最大token数                                   | 是       | -           |
| `min_tokens`       | int           | 生成的最少token数（遇到停止条件前必须生成）                 | 否       | 1          |
| `logprobs`         | int    | 返回每个token的前N个概率（None表示不返回）     （目前暂不支持）             | 否       | `None`      |

### OpenAI Compatible API 请求参数

|       字段名        |      字段类型      |                             说明                             | 是否必填 |    默认值    |
|---------------------|-------------------|------------------------------------------------------------|----------|-------------|
| `model`            | str         | 模型名称                                                   | 否       | default     |
| `prompt`           | Union[List[int], List[List[int]], str, List[str]] | 输入prompt                                                 | 是       | -           |
| `best_of`          | int         | 生成多个序列，返回最好的一个   （当前仅支持1）                                | 否       | 1           |
| `echo`             | bool        | 是否返回输入prompt                                             | 否       | False       |
| `frequency_penalty`| float             |    话题新鲜度           | 否       | -           |
| `logprobs`         | int    | 返回每个token的前N个概率（None表示不返回）     （目前暂不支持）             | 否       | `None`      |
| `max_tokens`       | int           | 每个序列生成的最大token数                                   | 是       | -           |
| `n`                | int          | 需要返回的生成序列数量（当前仅支持1）                                      | 是       | -           |
| `presence_penalty` | float             |    话题新鲜度           | 否       | -           |
| `repetition_penalty`| float             |     频率惩罚度        | 否       | -           |
| `seed`             | int         | 控制生成随机性的种子                                        | 否       | -           |
| `stop`             | Union[str, List[str]]         | 生成遇到这些字符串时停止（结果不包含它们）                  | 否       | -           |
| `stream`           | bool        | 是否流式返回结果                                              | 否       | False       |
| `stream_options`   | StreamOptions | 流式返回的选项，包含输入输出token 数目的统计               | 否       | None        |
| `suffix`           | str         | 生成序列后添加的后缀 （当前不支持）                                           | 否       | None        |
| `temperature`      | float             | 表示输出的确定性                    | 否       | -           |
| `top_p`            | float       | 仅考虑累积概率超过此值的候选词                              | 否       | 1           |
| `user`             | str         | 用户信息（当前不支持）                                                    | 否       | None        |
| `stop_token_ids`   | list[int]   | 生成遇到这些token时停止（结果包含token，除非是特殊token）   | 否       | -           |

### OpenAI Chat API 请求参数

|       字段名        |      字段类型      |                             说明                             | 是否必填 |    默认值    |
|---------------------|-------------------|------------------------------------------------------------|----------|-------------|
| `model`            | str         | 模型名称                                                   | 否       | default     |
| `messages`         | List[Dict[str, Union[str, List[int], List[List[int]]]]] | 输入prompt                                                 | 是       | -           |
| `best_of`          | int         | 生成多个序列，返回最好的一个   （当前仅支持1）                                | 否       | 1           |
| `echo`             | bool        | 是否返回输入prompt                                             | 否       | False       |
| `frequency_penalty`| float             |    话题新鲜度           | 否       | -           |
| `logprobs`         | int    | 返回每个token的前N个概率（None表示不返回）     （目前暂不支持）             | 否       | `None`      |
| `max_tokens`       | int           | 每个序列生成的最大token数                                   | 是       | -           |
| `n`                | int          | 需要返回的生成序列数量（当前仅支持1）                                      | 是       | -           |
| `presence_penalty` | float             |    话题新鲜度           | 否       | -           |
| `repetition_penalty`| float             |     频率惩罚度        | 否       | -           |
| `seed`             | int         | 控制生成随机性的种子                                        | 否       | -           |
| `stop`             | Union[str, List[str]]         | 生成遇到这些字符串时停止（结果不包含它们）                  | 否       | -           |
| `stream`           | bool        | 是否流式返回结果                                              | 否       | False       |
| `stream_options`   | StreamOptions | 流式返回的选项，包含输入输出token 数目的统计               | 否       | None        |
| `suffix`           | str         | 生成序列后添加的后缀 （当前不支持）                                           | 否       | None        |
| `temperature`      | float             | 表示输出的确定性                    | 否       | -           |
| `top_p`            | float       | 仅考虑累积概率超过此值的候选词                              | 否       | 1           |
| `user`             | str         | 用户信息（当前不支持）                                                    | 否       | None        |

## 输出参数说明

### 离线推理输出 RequestOutput

|       字段名        |      字段类型      |                             说明                             | 是否必填 |    默认值    |
|---------------------|-------------------|------------------------------------------------------------|----------|-------------|
| `request_id`       |       str         | 请求id                                                   | 否       | default     |
| `prompt`           | Optional[str]     | 输入prompt                                                | 否       | None        |
| `prompt_token_ids` | Optional[list[int]] | 输入prompt的token id                                      | 否       | None        |
| `outputs`          | CompletionOutput  | 推理输出                                                  | 是       | -           |
| `finished`         | bool              | 是否完成                                                  | 是       | False       |
| `num_cached_tokens`| Optional[int]     | 缓存的token数量                                           | 否       | 0           |
| `metrics`          | Optional[RequestMetrics] | 请求指标                                                | 否       | None        |
| `error_code`       | Optional[int]     | 错误代码                                                  | 否       | None        |
| `error_msg`        | Optional[str]     | 错误信息                                                  | 否       | None        |

#### 离线推理输出 CompletionOutput

|       字段名        |      字段类型      |                             说明                             | 是否必填 |    默认值    |
|---------------------|-------------------|------------------------------------------------------------|----------|-------------|
| `index`            | int         | 输出序列的索引                                               | 是       | -           |
| `token_ids`        | list[int]     | 输出的token id                                          | 是       | -           |
| `text`             | Optional[str]     | 输出文本                                              | 否       | None        |
| `reasoning_content`| Optional[str]     | 输出的思考链 （仅思考模型）                               | 否       | None        |

#### 离线推理输出 RequestMetrics

|       字段名        |      字段类型      |                             说明                             | 是否必填 |    默认值    |
|---------------------|-------------------|------------------------------------------------------------|----------|-------------|
| `arrival_time`      | float         | 请求到达时间                                               | 是       | -           |
| `inference_start_time`| Optional[float]     | 推理开始时间                                              | 否       | None        |
| `first_token_time`  | Optional[float]     | 第一个token生成耗时                                       | 否       | None        |
| `time_in_queue`     | Optional[float]     | 请求在队列中排队时间                                       | 否       | None        |
| `preprocess_cost_time`| Optional[float]     | 预处理耗时                                              | 否       | None        |
| `model_forward_time`| Optional[float]     | 模型前向推理耗时                                       | 否       | None        |
| `model_execute_time`| Optional[float]     | 模型执行耗时（包含预处理及排队时间）                             | 否       | None        |
| `request_start_time`| Optional[float]     | 请求开始时间                                           | 否       | None        |
