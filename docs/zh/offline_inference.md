# 离线推理

## 1. 使用方式
通过FastDeploy离线推理，可支持本地加载模型，并处理用户数据，使用方式如下，

### 续写接口(LLM.generate)

```python
from fastdeploy import LLM, SamplingParams

prompts = [
    "把李白的静夜思改写为现代诗",
    "Write me a poem about large language model.",
]

# 采样参数
sampling_params = SamplingParams(top_p=0.95, max_tokens=6400)

# 加载模型
llm = LLM(model="ERNIE-4.5-0.3B", tensor_parallel_size=1, max_model_len=8192)

# 批量进行推理（llm内部基于资源情况进行请求排队、动态插入处理）
outputs = llm.generate(prompts, sampling_params)

# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
```

### 对话接口(LLM.chat)

```python
from fastdeploy import LLM, SamplingParams

msg1=[
    {"role": "system", "content": "I'm a helpful AI assistant."},
    {"role": "user", "content": "把李白的静夜思改写为现代诗"},
]
msg2 = [
    {"role": "system", "content": "I'm a helpful AI assistant."},
    {"role": "user", "content": "Write me a poem about large language model."},
]
messages = [msg1, msg2]

# 采样参数
sampling_params = SamplingParams(top_p=0.95, max_tokens=6400)

# 加载模型
llm = LLM(model="ERNIE-4.5-0.3B", tensor_parallel_size=1, max_model_len=8192)
# 批量进行推理（llm内部基于资源情况进行请求排队、动态插入处理）
outputs = llm.chat(messages, sampling_params)

# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
```

上述示例中```LLM```配置方式， `SamplingParams` ，`LLM.generate` ，`LLM.chat`以及输出output对应的结构体 `RequestOutput` 接口说明见如下文档说明。

> 注： 若为X1 模型输出

```python
# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    reasoning_text = output.outputs.resoning_content
```

## 2. 接口说明

### 2.1 fastdeploy.LLM

支持配置参数参考 [FastDeploy参数说明](./parameters.md)

> 参数配置说明：
> 1. 离线推理不需要配置 `port` 和`metrics_port` 参数。
> 2. 模型服务启动后，会在日志文件log/fastdeploy.log中打印如 `Doing profile, the total_block_num:640` 的日志，其中640即表示自动计算得到的KV Cache block数量，将它乘以block_size(默认值64)，即可得到部署后总共可以在KV Cache中缓存的Token数。
> 3. `max_num_seqs` 用于配置decode阶段最大并发处理请求数，该参数可以基于第1点中缓存的Token数来计算一个较优值，例如线上统计输入平均token数800, 输出平均token数500，本次计>算得到KV Cache block为640， block_size为64。那么我们可以配置 `kv_cache_ratio = 800 / (800 + 500) = 0.6` , 配置 `max_seq_len = 640 * 64 / (800 + 500) = 31`。

### 2.2 fastdeploy.LLM.generate

* prompts(str,list[str],list[int]): 输入的prompt, 支持batch prompt 输入，解码后的token ids 进行输入
* sampling_params: 模型超参设置具体说明见2.4
* use_tqdm: 是否打开推理进度可视化

### 2.3 fastdeploy.LLM.chat

* messages(list[dict],list[list[dict]]): 输入的message, 支持batch message 输入
* sampling_params: 模型超参设置具体说明见2.4
* use_tqdm: 是否打开推理进度可视化
* chat_template_kwargs(dict): 传递给对话模板的额外参数，当前支持enable_thinking(bool)

### 2.4 fastdeploy.SamplingParams

* presence_penalty(float): 控制模型生成重复内容的惩罚系数，正值降低重复话题出现的概率
* frequency_penalty(float): 控制重复token的惩罚力度，比presence_penalty更严格，会惩罚高频重复
* repetition_penalty(float): 直接对重复生成的token进行惩罚的系数（>1时惩罚重复，<1时鼓励重复）
* temperature(float): 控制生成随机性的参数，值越高结果越随机，值越低结果越确定
* top_p(float): 概率累积分布截断阈值，仅考虑累计概率达到此阈值的最可能token集合
* max_tokens(int): 限制模型生成的最大token数量（包括输入和输出）
* min_tokens(int): 强制模型生成的最少token数量，避免过早结束

### 2.5 fastdeploy.engine.request.RequestOutput

* request_id(str): 标识request 的id
* prompt(str)：输入请求的request内容
* prompt_token_ids(list[int]): 拼接后经过词典解码的输入的token 列表
* outputs(fastdeploy.engine.request.CompletionOutput): 输出结果
* finished(bool)：标识当前query 是否推理结束
* metrics(fastdeploy.engine.request.RequestMetrics)：记录推理耗时指标
* num_cached_tokens(int): 缓存的token数量, 仅在开启```enable_prefix_caching```时有效
* error_code(int): 错误码
* error_msg(str): 错误信息

### 2.6 fastdeploy.engine.request.CompletionOutput

* index(int)：推理服务时的 batch index
* send_idx(int): 当前请求返回的 token 序号
* token_ids(list[int])：输出的 token 列表
* text(str):  token ids 对应的文本
* reasoning_content(str):（仅思考模型有效）返回思考链的结果

### 2.7 fastdeploy.engine.request.RequestMetrics

* arrival_time(float):：收到数据的时间，若流式返回则该时间为拿到推理结果的时间，若非流式返回则为收到推理数据
* inference_start_time(float):：开始推理的时间点
* first_token_time(float):：推理侧首token 耗时
* time_in_queue(float)：等待推理的排队耗时
* model_forward_time(float):：推理侧模型前向的耗时
* model_execute_time(float):: 模型执行耗时，包括前向推理，排队，预处理（文本拼接，解码操作）的耗时
