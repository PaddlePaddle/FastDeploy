[English](../../online_serving/README.md)

# 兼容 OpenAI 协议的服务化部署

FastDeploy 提供与 OpenAI 协议兼容的服务化部署方案。用户可以通过如下命令快速进行部署：

```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-0.3B-Paddle \
       --port 8188 --tensor-parallel-size 8 \
       --max-model-len 32768
```

如果要启用输出token的logprob，用户可以通过如下命令快速进行部署：

```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-0.3B-Paddle \
       --port 8188 --tensor-parallel-size 8 \
       --max-model-len 32768 \
       --enable-logprob
```

服务部署时的命令行更多使用方式参考[参数说明](../parameters.md)。

## Chat Completion API
FastDeploy 接口兼容 OpenAI 的 Chat Completion API，用户可以通过 OpenAI 协议发送用户请求。

### 发送用户请求

使用 curl 命令发送用户请求示例如下：

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'
```

使用 curl 命令示例，演示如何在用户请求中包含logprobs参数：

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "logprobs": true, "top_logprobs": 0,
}'
```

使用 Python 脚本发送用户请求示例如下：

```python
import openai
host = "0.0.0.0"
port = "8170"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "把李白的静夜思改写为现代诗"},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

关于 OpenAI 协议的说明可参考文档 [OpenAI Chat Completion API](https://platform.openai.com/docs/api-reference/chat/create)。

### 兼容OpenAI 参数
```python
messages: Union[List[Any], List[int]]
# 输入消息列表，可以是文本消息（`List[Any]`，通常为 `List[dict]`）或 token ID 列表（`List[int]`）。

tools: Optional[List[ChatCompletionToolsParam]] = None
# 工具调用配置列表，用于启用函数调用（Function Calling）或工具使用（如 ReAct 框架）。

model: Optional[str] = "default"
# 指定使用的模型名称或版本，默认值为 `"default"`（可能指向基础模型）。

frequency_penalty: Optional[float] = None
# 频率惩罚系数，降低重复生成相同 token 的概率（`>1.0` 抑制重复，`<1.0` 鼓励重复，默认 `None` 禁用）。

logprobs: Optional[bool] = False
# 是否返回每个生成 token 的对数概率（log probabilities），用于调试或分析。

top_logprobs: Optional[int] = 0
# 返回每个生成位置概率最高的 `top_logprobs` 个 token 及其对数概率（默认 `0` 表示不返回）。

max_tokens: Optional[int] = Field(
    default=None,
    deprecated="max_tokens is deprecated in favor of the max_completion_tokens field",
)
# 已弃用：生成的最大 token 数（建议改用 `max_completion_tokens`）。

max_completion_tokens: Optional[int] = None
# 生成的最大 token 数（推荐替代 `max_tokens`），默认无限制（受模型上下文窗口限制）。

presence_penalty: Optional[float] = None
# 存在惩罚系数，降低新主题（未出现过的话题）的生成概率（`>1.0` 抑制新话题，`<1.0` 鼓励新话题，默认 `None` 禁用）。

stream: Optional[bool] = False
# 是否启用流式输出（逐 token 返回结果），默认 `False`（一次性返回完整结果）。

stream_options: Optional[StreamOptions] = None
# 流式输出的额外配置（如分块大小、超时等），需参考 `StreamOptions` 的具体定义。

temperature: Optional[float] = None
# 温度系数，控制生成随机性（`0.0` 确定性生成，`>1.0` 更随机，默认 `None` 使用模型默认值）。

top_p: Optional[float] = None
# 核采样（nucleus sampling）阈值，只保留概率累计超过 `top_p` 的 token（默认 `None` 禁用）。

response_format: Optional[AnyResponseFormat] = None
# 指定输出格式（如 JSON、XML 等），需传入预定义的格式配置对象。

user: Optional[str] = None
# 用户标识符，用于跟踪或区分不同用户的请求（默认 `None` 不传递）。

metadata: Optional[dict] = None
# 附加元数据，用于传递自定义信息（如请求 ID、调试标记等）。

n: Optional[int] = 1
# 生成结果的候选数量（即返回多少个独立生成的文本），默认 1（仅返回一个结果）。

seed: Optional[int] = Field(default=None, ge=0, le=922337203685477580)
# 随机种子，用于控制生成过程的确定性（相同种子和输入会得到相同结果）。范围需在 `[0, 922337203685477580]` 之间，默认 None 表示不固定种子。

stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
# 停止生成的条件，可以是单个字符串或字符串列表。当模型生成任一停止字符串时，生成过程会提前终止（默认空列表表示不启用）。

```

### FastDeploy 增加额外参数

> 注：
使用 curl 命令发送请求时， 可以直接使用以下参数；
使用openai.Client 发送请求时，需要使用将以下参数放入 `extra_body` 参数中， 如：`extra_body={"chat_template_kwargs": {"enable_thinking":True}, "include_stop_str_in_output": True}`。

额外采样参数的支持如下：
```python
top_k: Optional[int] = None
# 限制每一步生成时只考虑概率最高的 K 个 token，用于控制随机性（默认 None 表示不限制）。

min_p: Optional[float] = None
# 核采样（nucleus sampling）阈值，只保留概率累计超过 min_p 的 token（默认 None 表示禁用）。

min_tokens: Optional[int] = None
# 强制生成的最小 token 数，避免过早截断（默认 None 表示不限制）。

include_stop_str_in_output: Optional[bool] = False
# 是否在输出中包含停止符（stop string）的内容（默认 False，即遇到停止符时截断输出）。

bad_words: Optional[List[str]] = None
# 禁止生成的词汇列表（例如敏感词），模型会避免输出这些词（默认 None 表示不限制）。

bad_words_token_ids: Optional[List[int]] = None
# 禁止生成的token id列表，模型会避免输出这些词（默认 None 表示不限制）。

repetition_penalty: Optional[float] = None
# 重复惩罚系数，降低已生成 token 的重复概率（>1.0 抑制重复，<1.0 鼓励重复，默认 None 表示禁用）。

stop_token_ids: Optional[List[int]] = Field(default_factory=list)
# 停止生成的 token ID 列表，当模型生成任一指定 token 时，生成过程会提前终止（默认空列表表示不启用）。通常与 `stop` 参数互补使用。

```
其他参数的支持如下：
```python
chat_template_kwargs: Optional[dict] = None
# 传递给聊天模板（chat template）的额外参数，用于自定义对话格式（默认 None）。

chat_template: Optional[str] = None
# 自定义聊天模板，会覆盖模型默认的聊天模板，（默认 None）。

reasoning_max_tokens: Optional[int] = None
# 推理（如 CoT, 思维链）过程中生成的最大 token 数（默认 None 表示使用全局 max_tokens）。

structural_tag: Optional[str] = None
# 结构化标签，用于标记生成内容的特定结构（如 JSON、XML 等，默认 None）。

guided_json: Optional[Union[str, dict, BaseModel]] = None
# 引导生成符合 JSON 结构的内容，可以是 JSON 字符串、字典或 Pydantic 模型（默认 None）。

guided_regex: Optional[str] = None
# 引导生成符合正则表达式规则的内容（默认 None 表示不限制）。

guided_choice: Optional[List[str]] = None
# 引导生成内容从指定的候选列表中选择（默认 None 表示不限制）。

guided_grammar: Optional[str] = None
# 引导生成符合语法规则（如 BNF）的内容（默认 None 表示不限制）。

return_token_ids: Optional[bool] = None
# 是否返回生成结果的 token ID 而非文本（默认 None 表示返回文本）。

prompt_token_ids: Optional[List[int]] = None
# 直接传入 prompt 的 token ID 列表，跳过文本编码步骤（默认 None 表示使用文本输入）。

disable_chat_template: Optional[bool] = False
# 是否禁用聊天模板渲染，直接使用原始输入（默认 False 表示启用模板）。

temp_scaled_logprobs: Optional[bool] = False
# 计算logprob时是否对logits除以温度系数（默认 False 表示不除以温度系数）。

top_p_normalized_logprobs: Optional[bool] = False
# 计算logprob时是否进行 top_p 归一化（默认 False 表示不进行top_p归一化）。

include_draft_logprobs: Optional[bool] = False
# 是否在预生成或中间步骤返回对数概率（log probabilities），用于调试或分析生成过程（默认 False 表示不返回）。

logits_processors_args: Optional[Dict] = None
# 传递给 logits 处理器（logits processors）的额外参数，用于自定义生成过程中的逻辑（如动态调整概率分布）。

mm_hashes: Optional[list] = None
# 多模态（multimodal）输入的哈希值列表，用于验证或跟踪输入内容（如图像、音频等）。默认 None 表示无多模态输入或无需哈希验证。

```

### 返回字段差异

FastDeploy 增加的返回字段如下：

- `arrival_time`：返回所有 token 的累计耗时
- `reasoning_content`: 思考链的返回结果
- `prompt_token_ids`: 输入序列的 token id 列表
- `completion_token_ids`: 输出序列的 token id 列表

返回参数总览：

```python

ChatCompletionResponse:
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
ChatCompletionResponseChoice:
    index: int
    message: ChatMessage
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "recover_stop"]]
ChatMessage:
    role: str
    content: str
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[DeltaToolCall | ToolCall]] = None
    prompt_token_ids: Optional[List[int]] = None
    completion_token_ids: Optional[List[int]] = None
    prompt_tokens: Optional[str] = None
    completion_tokens: Optional[str] = None
UsageInfo:
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[PromptTokenUsageInfo] = None
    completion_tokens_details: Optional[CompletionTokenUsageInfo] = None
PromptTokenUsageInfo:
    cached_tokens: Optional[int] = None
    image_tokens: Optional[int] = None
    video_tokens: Optional[int] = None
CompletionTokenUsageInfo:
    reasoning_tokens: Optional[int] = None
    image_tokens: Optional[int] = None
ToolCall:
    id: str = None
    type: Literal["function"] = "function"
    function: FunctionCall
FunctionCall:
    name: str
    arguments: str

# 返回流式响应的字段
ChatCompletionStreamResponse:
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None
ChatCompletionResponseStreamChoice:
    index: int
    delta: DeltaMessage
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None
    arrival_time: Optional[float] = None
DeltaMessage:
    role: Optional[str] = None
    content: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None
    completion_token_ids: Optional[List[int]] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[DeltaToolCall | ToolCall]] = None
    prompt_tokens: Optional[str] = None
    completion_tokens: Optional[str] = None
DeltaToolCall:
    id: Optional[str] = None
    type: Optional[Literal["function"]] = None
    index: int
    function: Optional[DeltaFunctionCall] = None
DeltaFunctionCall:
    name: Optional[str] = None
    arguments: Optional[str] = None
```

## Completion API
Completion API 接口主要用于续聊场景, 适应于用户自定义好上下文输入, 并希望模型仅输出续写内容的场景; 推理过程不会增加其他 `prompt`拼接。：

### 发送用户请求

使用 curl 命令发送用户请求示例如下：

```bash
curl -X POST "http://0.0.0.0:8188/v1/completions" \
-H "Content-Type: application/json" \
-d '{
  "prompt": "以下是一篇关于深圳文心公园的500字游记和赏析："
}'
```

使用 Python 脚本发送用户请求示例如下：

```python
import openai
host = "0.0.0.0"
port = "8170"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.completions.create(
    model="default",
    prompt="以下是一篇关于深圳文心公园的500字游记和赏析：",
    stream=False,
)
print(response.choices[0].text)
```

关于 OpenAI 协议的说明可参考文档 [OpenAI Completion API](https://platform.openai.com/docs/api-reference/completions/create)。

### 兼容OpenAI 参数
```python
model: Optional[str] = "default"
# 指定使用的模型名称或版本，默认值为 `"default"`（可能指向基础模型）。

prompt: Union[List[int], List[List[int]], str, List[str]]
# 输入提示，支持多种格式：
#   - `str`: 纯文本提示（如 `"Hello, how are you?"`）。
#   - `List[str]`: 多段文本（如 `["User:", "Hello!", "Assistant:", "Hi!"]`）。
#   - `List[int]`: 直接传入 token ID 列表（如 `[123, 456]`）。
#   - `List[List[int]]`: 多段 token ID 列表（如 `[[123], [456, 789]]`）。

best_of: Optional[int] = None
# 生成 `best_of` 个候选结果，然后返回其中评分最高的一个（需配合 `n=1` 使用）。

frequency_penalty: Optional[float] = None
# 频率惩罚系数，降低重复生成相同 token 的概率（`>1.0` 抑制重复，`<1.0` 鼓励重复）。

logprobs: Optional[int] = None
# 返回每个生成 token 的对数概率（log probabilities），可指定返回的候选数量。

max_tokens: Optional[int] = None
# 生成的最大 token 数（包括输入和输出），默认无限制（受模型上下文窗口限制）。

presence_penalty: Optional[float] = None
# 存在惩罚系数，降低新主题（未出现过的话题）的生成概率（`>1.0` 抑制新话题，`<1.0` 鼓励新话题）。

echo: Optional[bool] = False
# 是否将输入的 prompt 包含在输出中（默认 False，即不输出 prompt）。

n: Optional[int] = 1
# 生成结果的候选数量（即返回多少个独立生成的文本），默认 1（仅返回一个结果）。

seed: Optional[int] = Field(default=None, ge=0, le=922337203685477580)
# 随机种子，用于控制生成过程的确定性（相同种子和输入会得到相同结果）。范围需在 `[0, 922337203685477580]` 之间，默认 None 表示不固定种子。

stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
# 停止生成的条件，可以是单个字符串或字符串列表。当模型生成任一停止字符串时，生成过程会提前终止（默认空列表表示不启用）。

stream: Optional[bool] = False
# 是否启用流式输出（逐 token 返回结果），默认 `False`（一次性返回完整结果）。

stream_options: Optional[StreamOptions] = None
# 流式输出的额外配置（如分块大小、超时等），需参考 `StreamOptions` 的具体定义。

temperature: Optional[float] = None
# 温度系数，控制生成随机性（`0.0` 确定性生成，`>1.0` 更随机，默认 `None` 使用模型默认值）。

top_p: Optional[float] = None
# 核采样（nucleus sampling）阈值，只保留概率累计超过 `top_p` 的 token（默认 `None` 禁用）。

response_format: Optional[AnyResponseFormat] = None
# 指定输出格式（如 JSON、XML 等），需传入预定义的格式配置对象。

user: Optional[str] = None
# 用户标识符，用于跟踪或区分不同用户的请求（默认 `None` 不传递）。

```

### FastDeploy 增加额外参数

> 注：
使用 curl 命令发送请求时， 可以直接使用以下参数；
使用openai.Client 发送请求时，需要使用将以下参数放入 `extra_body` 参数中， 如：`extra_body={"chat_template_kwargs": {"enable_thinking":True}, "include_stop_str_in_output": True}`。

额外采样参数的支持如下：
```python
top_k: Optional[int] = None
# 限制每一步生成时只考虑概率最高的 K 个 token，用于控制随机性（默认 None 表示不限制）。

min_p: Optional[float] = None
# 核采样（nucleus sampling）阈值，只保留概率累计超过 min_p 的 token（默认 None 表示禁用）。

min_tokens: Optional[int] = None
# 强制生成的最小 token 数，避免过早截断（默认 None 表示不限制）。

include_stop_str_in_output: Optional[bool] = False
# 是否在输出中包含停止符（stop string）的内容（默认 False，即遇到停止符时截断输出）。

bad_words: Optional[List[str]] = None
# 禁止生成的词汇列表（例如敏感词），模型会避免输出这些词（默认 None 表示不限制）。

bad_words_token_ids: Optional[List[int]] = None
# 禁止生成的token id列表，模型会避免输出这些词（默认 None 表示不限制）。

stop_token_ids: Optional[List[int]] = Field(default_factory=list)
# 停止生成的 token ID 列表，当模型生成任一指定 token 时，生成过程会提前终止（默认空列表表示不启用）。通常与 `stop` 参数互补使用。

repetition_penalty: Optional[float] = None
# 重复惩罚系数，降低已生成 token 的重复概率（>1.0 抑制重复，<1.0 鼓励重复，默认 None 表示禁用）。
```
其他参数的支持如下：
```python
guided_json: Optional[Union[str, dict, BaseModel]] = None
# 引导生成符合 JSON 结构的内容，可以是 JSON 字符串、字典或 Pydantic 模型（默认 None）。

guided_regex: Optional[str] = None
# 引导生成符合正则表达式规则的内容（默认 None 表示不限制）。

guided_choice: Optional[List[str]] = None
# 引导生成内容从指定的候选列表中选择（默认 None 表示不限制）。

guided_grammar: Optional[str] = None
# 引导生成符合语法规则（如 BNF）的内容（默认 None 表示不限制）。

return_token_ids: Optional[bool] = None
# 是否返回生成结果的 token ID 而非文本（默认 None 表示返回文本）。

prompt_token_ids: Optional[List[int]] = None
# 直接传入 prompt 的 token ID 列表，跳过文本编码步骤（默认 None 表示使用文本输入）。

temp_scaled_logprobs: Optional[bool] = False
# 计算logprob时是否对logits除以温度系数（默认 False 表示不除以温度系数）。

top_p_normalized_logprobs: Optional[bool] = False
# 计算logprob时是否进行 top_p 归一化（默认 False 表示不进行top_p归一化）。

include_draft_logprobs: Optional[bool] = False
# 是否在预生成或中间步骤返回对数概率（log probabilities），用于调试或分析生成过程（默认 False 表示不返回）。

logits_processors_args: Optional[Dict] = None
# 传递给 logits 处理器（logits processors）的额外参数，用于自定义生成过程中的逻辑（如动态调整概率分布）。

mm_hashes: Optional[list] = None
# 多模态（multimodal）输入的哈希值列表，用于验证或跟踪输入内容（如图像、音频等）。默认 None 表示无多模态输入或无需哈希验证。
```

### 返回参数总览

```python

CompletionResponse:
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo
CompletionResponseChoice:
    index: int
    text: str
    prompt_token_ids: Optional[List[int]] = None
    completion_token_ids: Optional[List[int]] = None
    prompt_tokens: Optional[str] = None
    completion_tokens: Optional[str] = None
    arrival_time: Optional[float] = None
    logprobs: Optional[int] = None
    reasoning_content: Optional[str] = None
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]]
    tool_calls: Optional[List[DeltaToolCall | ToolCall]] = None
UsageInfo:
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[PromptTokenUsageInfo] = None
    completion_tokens_details: Optional[CompletionTokenUsageInfo] = None
PromptTokenUsageInfo:
    cached_tokens: Optional[int] = None
    image_tokens: Optional[int] = None
    video_tokens: Optional[int] = None
CompletionTokenUsageInfo:
    reasoning_tokens: Optional[int] = None
    image_tokens: Optional[int] = None
ToolCall:
    id: str = None
    type: Literal["function"] = "function"
    function: FunctionCall
FunctionCall:
    name: str
    arguments: str

# 返回流式响应的字段
CompletionStreamResponse：
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None
CompletionResponseStreamChoice:
    index: int
    text: str
    arrival_time: float = None
    prompt_token_ids: Optional[List[int]] = None
    completion_token_ids: Optional[List[int]] = None
    prompt_tokens: Optional[str] = None
    completion_tokens: Optional[str] = None
    logprobs: Optional[float] = None
    reasoning_content: Optional[str] = None
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None
    tool_calls: Optional[List[DeltaToolCall | ToolCall]] = None
DeltaToolCall:
    id: Optional[str] = None
    type: Optional[Literal["function"]] = None
    index: int
    function: Optional[DeltaFunctionCall] = None
DeltaFunctionCall:
    name: Optional[str] = None
    arguments: Optional[str] = None
```
