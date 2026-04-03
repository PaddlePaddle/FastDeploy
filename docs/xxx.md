# FastDeploy OpenAI Chat Serving 数据处理全链路分析

> 本文档以 `fastdeploy/entrypoints/openai/serving_chat.py` 为起点，逐层追踪从 HTTP 请求接收到模型推理输入、再到最终响应输出的完整数据处理流程。

---

## 目录

1. [入口：HTTP 请求接收](#1-入口http-请求接收)
2. [请求数据模型：ChatCompletionRequest](#2-请求数据模型chatcompletionrequest)
3. [create_chat_completion 核心流程](#3-create_chat_completion-核心流程)
4. [预处理管道：EngineClient.format_and_add_data](#4-预处理管道engineclientformat_and_add_data)
5. [处理器工厂：InputPreprocessor.create_processor](#5-处理器工厂inputpreprocessorcreate_processor)
6. [分词器集成：BaseTextProcessor / TextProcessor](#6-分词器集成basetextprocessor--textprocessor)
7. [流式与非流式响应的差异处理](#7-流式与非流式响应的差异处理)
8. [输出解码：ChatResponseProcessor + BaseTextProcessor](#8-输出解码chatresponseprocessor--basetextprocessor)
9. [OpenAI 格式响应构建](#9-openai-格式响应构建)
10. [中间件与预处理步骤](#10-中间件与预处理步骤)
11. [完整调用链总览](#11-完整调用链总览)
12. [核心数据模型关系表](#12-核心数据模型关系表)

---

## 1. 入口：HTTP 请求接收

**文件：** `fastdeploy/entrypoints/openai/api_server.py`（第 541–575 行）

FastAPI 应用在 `/v1/chat/completions` 注册 POST 路由：

```python
@app.post("/v1/chat/completions")
@with_cancellation
async def create_chat_completion(request: ChatCompletionRequest, req: Request):
```

FastAPI 自动将 JSON 请求体反序列化并验证为 `ChatCompletionRequest` Pydantic 模型。`@with_cancellation` 装饰器（来自 `utils.py`）处理客户端断连时的 `asyncio.CancelledError`。

**请求到达后，服务器依次执行：**

1. 从 HTTP 请求头中提取 OpenTelemetry 链路追踪上下文（第 548–552 行）
2. 若启用 `dynamic_load_weight` 则检查 worker 健康状态
3. 获取**全局连接信号量**（`connection_manager()`，第 318–330 行）——限制总并发数为 `max_concurrency // workers`
4. 调用 `app.state.chat_handler.create_chat_completion(request)`

根据运行模式，`chat_handler` 为：
- `OpenAIServingChat`（ZMQ 模式，`serving_chat.py`）
- `OpenAIServingChatV1`（AsyncLLM 模式，`v1/serving_chat.py`）

---

## 2. 请求数据模型：ChatCompletionRequest

**文件：** `fastdeploy/entrypoints/openai/protocol.py`（第 663–866 行）

```python
class ChatCompletionRequest(BaseModel):
    messages: Union[List[Any], List[int]]        # 对话消息列表，或直接传入 token ID 列表
    tools: Optional[List[ChatCompletionToolsParam]]  # 工具定义（函数调用）
    model: Optional[str] = "default"
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    max_tokens: Optional[int]
    max_completion_tokens: Optional[int]
    n: Optional[int] = 1                          # 同时生成的候选数量
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions]
    stop: Optional[Union[str, List[str]]]
    stop_token_ids: Optional[List[int]]
    seed: Optional[int]
    response_format: Optional[AnyResponseFormat]  # text/json_object/json_schema/structural_tag
    guided_json / guided_regex / guided_choice / guided_grammar: ...
    chat_template: Optional[str]                  # 自定义 chat template（覆盖模型默认）
    chat_template_kwargs: Optional[dict]
    disable_chat_template: Optional[bool]         # 禁用 template，直接用 messages[0].content
    reasoning_max_tokens: Optional[int]
    reasoning_effort: Literal["minimal","low","medium","high"] | None
    logprobs: Optional[bool]
    top_logprobs: Optional[int]
    prompt_logprobs: Optional[int]
    return_token_ids: Optional[bool]              # 是否在响应中返回 token ID
    mm_hashes: Optional[list]                     # 多模态内容哈希（图片等）
    ...
```

### 验证器（`@model_validator`）

- `validate_stream_options`（第 821–849 行）：确保 `stream_options` 仅在 `stream=True` 时设置；同时确保 guided decoding 参数只启用其中一种
- `check_logprobs`（第 851–865 行）：确保 `top_logprobs ≥ -1`，并在使用 `top_logprobs` 时必须设置 `logprobs=true`

### `to_dict_for_infer(request_id)` 方法（第 743–819 行）

将 Pydantic 模型转换为纯字典，供下游处理使用：
- `max_tokens = max_completion_tokens or max_tokens`
- 合并 `metadata` 字典（已废弃参数）
- 处理 `response_format` → 设置 `guided_json_object` / `guided_json` / `structural_tag`
- 若 `disable_chat_template=True`：提取 `messages[0]["content"]` 作为 `prompt`，删除 `messages` 字段

---

## 3. create_chat_completion 核心流程

**文件：** `fastdeploy/entrypoints/openai/serving_chat.py`（第 106–198 行）

### 类初始化（第 74–101 行）

```python
class OpenAIServingChat:
    def __init__(self, engine_client, models, pid, ips, max_waiting_time,
                 chat_template, enable_mm_output, tokenizer_base_url):
        self.engine_client      # EngineClient —— ZMQ 引擎连接
        self.models             # OpenAIServingModels —— 模型注册表
        self.chat_template      # 启动时从文件/路径加载
        self.enable_mm_output   # 是否处理多模态输出
        self.tokenizer_base_url # 可选的远程分词器服务地址
```

### `create_chat_completion()` 执行步骤

1. **主节点检查**（第 111 行）：仅主节点接受请求，否则返回 `ErrorResponse`
2. **模型检查**（第 118–125 行）：对照支持的模型列表验证模型名称
3. **信号量获取**（第 128–131 行）：获取 `engine_client.semaphore`（每 worker 并发限制），超过 `max_waiting_time` 则超时报错
4. **请求 ID 生成**（第 134–141 行）：生成 `chatcmpl-{user}-{uuid}` 或 `chatcmpl-{uuid}`
5. **链路追踪埋点**（第 142 行）：挂载分布式追踪上下文
6. **分词 / 预处理**（第 148–157 行）：
   ```python
   current_req_dict = request.to_dict_for_infer(f"{request_id}_0")
   current_req_dict["chat_template"] = self.chat_template  # 若请求中未设置
   current_req_dict["metrics"]["arrival_time"] = time.time()
   prompt_token_ids = await self.engine_client.format_and_add_data(current_req_dict)
   # 返回 np.ndarray 类型的 token IDs，转换为 list
   ```
7. **流式路由**（第 170–182 行）：
   - `stream=True` → 返回异步生成器 `chat_completion_stream_generator()`
   - `stream=False` → 等待 `chat_completion_full_generator()` 返回完整响应

---

## 4. 预处理管道：EngineClient.format_and_add_data

**文件：** `fastdeploy/entrypoints/engine_client.py`（第 283–445 行）

### `format_and_add_data()`（第 283–295 行）

```python
async def format_and_add_data(self, request):
    # 确保 request_id 存在
    # 设置默认 max_tokens = max_model_len
    await self.add_requests(request)
    return request["prompt_token_ids"]  # 返回 token IDs
```

### `add_requests(task)`（第 297–435 行）—— 核心预处理方法

**步骤 A：chat_template 参数合并**（第 343–348 行）
```python
chat_template_kwargs = task.get("chat_template_kwargs") or {}
chat_template_kwargs.update({"chat_template": task.get("chat_template")})
# reasoning_effort 也在此注入
task["chat_template_kwargs"] = chat_template_kwargs
```

**步骤 B：消息内容规范化**（第 349 行）
```python
self.process_messages(task.get("messages", []))
# 将多模态消息内容（图片等）转换为标准化格式
```

**步骤 C：数据处理器分词**（第 350–353 行）
```python
self.data_processor.process_request_dict(task, self.max_model_len)
# 实际的 chat template 应用和 tokenization 在此发生（见第6节）
```

**步骤 D：token 长度校验**（第 355–404 行）
- 设置 `prompt_token_ids_len`
- 调整 `max_tokens = min(max_model_len - input_len, max_tokens)`
- 校验不超过 `max_model_len`
- 校验 stop 序列数量和长度

**步骤 E：参数合法性验证**（第 413 行）：`valid_parameters(task)` 验证采样超参数

**步骤 F：ZMQ 发送**（第 416–435 行）
```python
self._send_task(child_task)   # 通过 ZmqIpcClient 发送
```
对于 n>1 的请求，创建 n 个 child_task（ID 为 `{request_id}_{i}`）分别发送。

### `_send_task()`（第 437–445 行）
- 纯文本：`self.zmq_client.send_json(task)`
- 多模态：`self.zmq_client.send_pyobj(task)`（可选地先转换为 tensor）

---

## 5. 处理器工厂：InputPreprocessor.create_processor

**文件：** `fastdeploy/input/preprocess.py`（第 60–144 行）

```python
class InputPreprocessor:
    def create_processor(self):
        # 解析推理解析器和工具解析器
        reasoning_parser_obj = ReasoningParserManager.get_reasoning_parser(...)
        tool_parser_obj = ToolParserManager.get_tool_parser(...)

        architecture = self.model_config.architectures[0]

        if not self.model_config.enable_mm:
            # 纯文本模型：使用 TextProcessor
            tokenizer_type = "ernie4_5" if ErnieArchitectures.contains_ernie_arch(arch) else "auto"
            self.processor = TextProcessor(
                model_name_or_path=...,
                tokenizer_type=tokenizer_type,
                reasoning_parser_obj=...,
                tool_parser_obj=...,
            )
        else:
            # 多模态模型：根据架构选择对应处理器
            # Ernie4_5_VLProcessor / QwenVLProcessor / Qwen3VLProcessor / PaddleOCRVLProcessor
```

`EngineClient` 在初始化时（第 88–98 行）创建此处理器：
```python
input_processor = InputPreprocessor(fd_config.model_config, reasoning_parser, ...)
self.data_processor = input_processor.create_processor()
```

---

## 6. 分词器集成：BaseTextProcessor / TextProcessor

**文件：** `fastdeploy/input/base_processor.py`（第 63–455 行）  
**文件：** `fastdeploy/input/text_processor.py`（第 274–333 行）

### 分词器加载（`TextProcessor._load_tokenizer()`，第 300–325 行）

```python
if tokenizer_type == "ernie4_5":
    return Ernie4_5Tokenizer.from_pretrained(model_name_or_path)
else:  # "auto"
    if FD_USE_HF_TOKENIZER:
        # HuggingFace 分词器
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    else:
        # PaddleFormers 分词器（默认）
        return AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", use_fast=True)
```

EOS token 通过 `get_eos_token_id(tokenizer, generation_config)` 获取，合并分词器与生成配置中的 EOS。

### `process_request_dict()` —— 核心分词步骤（第 360–454 行）

```python
def process_request_dict(self, request, max_model_len=None):
    # 1. 应用生成配置默认值（top_p=0.7, temperature=1.0 等）
    request = self._apply_default_parameters(request)

    # 2. 设置 EOS token IDs
    if not request.get("eos_token_ids"):
        request["eos_token_ids"] = self.eos_token_ids

    # 3. 处理 stop 序列 → token IDs
    process_stop_token_ids(request, self.update_stop_seq)

    # 4. 处理 bad_words → token IDs
    if bad_words:
        request["bad_words_token_ids"] = self.update_bad_words(...)

    # 5. 准备 thinking stop sentence（<think> token 强制截止的 token IDs）
    logits_processors_args = self._prepare_think_stop_sentence(...)

    # 6. 分词提示词：
    if not request.get("prompt_token_ids"):
        if request.get("prompt"):
            # 纯文本 → text2ids()
            token_ids = self.text2ids(prompt, max_model_len)
            request["prompt_token_ids"] = token_ids
        elif request.get("messages"):
            # 对话消息 → 应用 chat template → 分词
            request["prompt_token_ids"] = self.messages2ids(request, **chat_template_kwargs)

    # 7. 截断超过 max_model_len 的输入
    if len(request["prompt_token_ids"]) > max_model_len:
        request["prompt_token_ids"] = request["prompt_token_ids"][:max_model_len - 1]

    # 8. 更新 thinking prompt 状态（检测 <think> token）
    logits_processors_args = self._update_thinking_prompt_state(prompt_token_ids, ...)

    # 9. 计算 max_tokens = min(max_model_len - len(prompt), 请求的 max_tokens)

    # 10. temperature/top_p 裁剪（零温度 → 贪心：top_k=1）
    if temperature < ε: temperature=1, top_k=1
    if top_p < ε: top_p=ε, top_k=1

    # 11. 推理解析器模型状态检测
    if self.reasoning_parser:
        model_status = self.reasoning_parser.get_model_status(prompt_token_ids)
        self.model_status_dict[req_id] = model_status
        request["enable_thinking"] = (model_status == "think_start")
```

### `messages2ids()` —— Chat Template 应用（第 144–168 行）

```python
def messages2ids(self, request, **kwargs):
    # 应用分词器内置的 Jinja2 chat template
    spliced_message = self.tokenizer.apply_chat_template(
        request,                    # 包含 "messages" 键
        tokenize=False,             # 返回字符串而非 token IDs
        add_generation_prompt=True, # 追加 assistant 轮次起始标记
        **kwargs                    # 可包含自定义 chat_template、reasoning_effort 等
    )
    request["prompt_tokens"] = spliced_message   # 存储渲染后的字符串
    tokens = self.tokenizer.tokenize(spliced_message)
    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    return token_ids
```

> **关键点**：分词器的 `chat_template`（存储在 `tokenizer_config.json` 中的 Jinja2 模板）将对话消息格式化为单个字符串，再进行分词。

---

## 7. 流式与非流式响应的差异处理

两种模式均使用通过 `engine_client.connection_manager.get_connection(request_id, num_choices)` 建立的 **ZMQ 响应队列**。

### 7.1 流式：`chat_completion_stream_generator()`（第 205–540 行）

**主要特点：**
- 立即返回**异步生成器**，API 服务器将其包装为 `StreamingResponse`
- 初始化每个候选的状态数组：`previous_num_tokens`、`reasoning_num_tokens`、`tool_called` 等
- `max_streaming_response_tokens` 控制刷新前的 chunk 缓冲数量（默认 1）
- 使用 `ChatResponseProcessor.process_response_chat(response, stream=True, ...)`

**每个响应包的处理：**
- **首包**：发送包含角色信息和 prompt logprobs（如有请求）的初始 chunk
- **后续包**：构建 `DeltaMessage(content=delta_text, reasoning_content=..., tool_calls=...)`
- 封装为 `ChatCompletionResponseStreamChoice` → `ChatCompletionStreamResponse`
- 以 SSE 格式输出：`f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"`
- 在 `finished=True` 时：设置 `finish_reason = "stop"/"length"/"tool_calls"/"recover_stop"`

**收尾处理：**
- 若 `include_usage=True`：发送最终 usage 统计 chunk
- 始终以 `"data: [DONE]\n\n"` 结束

### 7.2 非流式：`chat_completion_full_generator()`（第 542–735 行）

**主要特点：**
- 使用 `ChatResponseProcessor.process_response_chat(response, stream=False, ...)`
- **累积所有 token**：`completion_token_ids[idx].extend(...)`
- 累积 logprobs：`logprob_contents[idx].extend(...)`
- 仅在 `finished=True` 时（第 683 行）调用 `_create_chat_completion_choice()`
- 返回完整的 `ChatCompletionResponse`（而非生成器）

> **对比**：非流式模式下，`process_response_dict_normal()` 在收到全部 token 后一次性解码整个序列。

---

## 8. 输出解码：ChatResponseProcessor + BaseTextProcessor

**文件：** `fastdeploy/entrypoints/openai/response_processors.py`  
**文件：** `fastdeploy/input/base_processor.py`（第 234–358 行）

### `ChatResponseProcessor.process_response_chat()`（第 75–231 行）

- **纯文本**（非多模态）：调用 `data_processor.process_response_dict(response_dict, stream=...)`
- **多模态流式**：处理 `decode_type`（0=文本，1=图片，2=音频）
  - 累积图片 token IDs 直到 `<eoi>` token，再通过 `AsyncTokenizerClient` 解码
  - 处理音频 token 的缓冲，直到 EOS

### 流式解码路径（`process_response_dict_streaming()`，第 290–358 行）

```python
delta_text, previous_token_ids, previous_texts = self.ids2tokens(token_ids, req_id)
response_dict["outputs"]["text"] = delta_text

if self.reasoning_parser:
    # 从响应文本中分离 <think>...</think> 内容
    reasoning_delta = self.reasoning_parser.extract_reasoning_content_streaming(
        previous_texts, previous_texts + delta_text, delta_text, ...)

if self.tool_parser_obj:
    # 增量解析 streaming 工具调用 JSON
    tool_call_delta = tool_parser.extract_tool_calls_streaming(...)
```

### 非流式解码路径（`process_response_dict_normal()`，第 245–288 行）

```python
delta_text, _, previous_texts = self.ids2tokens(token_ids, req_id)
if is_end:
    full_text = previous_texts + delta_text
    # 提取推理内容：分离思考过程与响应内容
    # 提取工具调用：解析完整的 JSON 工具调用
    response_dict["outputs"]["text"] = 推理后的文本
    response_dict["outputs"]["tool_calls"] = tool_call_info.tool_calls
```

### 增量解码：`ids2tokens()`（第 188–228 行）

| 路径 | 方法 |
|---|---|
| HuggingFace 分词器 | 累积全部 token，调用 `batch_decode()`，通过差值计算增量字符串 |
| PaddleFormers 分词器（默认） | 使用 `tokenizer.decode_token(all_ids, prefix_offset, read_offset)` 高效增量解码 |

---

## 9. OpenAI 格式响应构建

### 流式响应（`serving_chat.py`，第 415–498 行）

```python
delta_message = DeltaMessage(
    reasoning_content=output["reasoning_content"],
    tool_calls=output["tool_calls"],
)
delta_message.content = output["text"]   # 多模态时为 output["multipart"]

choice = ChatCompletionResponseStreamChoice(
    index=idx,
    delta=delta_message,
    logprobs=logprobs_res,
    finish_reason=...,           # 仅在最终 chunk 中设置
    speculate_metrics=...,       # 投机解码统计
)
chunk = ChatCompletionStreamResponse(
    id=request_id, model=model_name, choices=[choice], usage=usage
)
yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
```

### 非流式响应（`_create_chat_completion_choice()`，第 737–815 行）

```python
message = ChatMessage(
    role="assistant",
    content=output["text"],
    reasoning_content=output.get("reasoning_content"),
    tool_calls=output.get("tool_calls"),
    prompt_token_ids=...,        # 若 return_token_ids=True
    completion_token_ids=...,
)
return ChatCompletionResponseChoice(
    index=idx,
    message=message,
    logprobs=...,
    finish_reason=...,
    speculate_metrics=...
)
```

最终的 `ChatCompletionResponse`（第 727–734 行）：
```python
res = ChatCompletionResponse(
    id=request_id,
    model=model_name,
    choices=sorted(choices, key=lambda x: x.index),
    usage=UsageInfo(
        prompt_tokens=len(prompt_token_ids),
        completion_tokens=sum(previous_num_tokens),
        total_tokens=...,
        prompt_tokens_details=PromptTokenUsageInfo(
            cached_tokens=..., image_tokens=..., video_tokens=...
        ),
        completion_tokens_details=CompletionTokenUsageInfo(
            reasoning_tokens=..., image_tokens=...
        ),
    ),
)
```

### finish_reason 取值

| 值 | 触发条件 |
|---|---|
| `"stop"` | 正常生成结束（EOS 或 stop 序列） |
| `"length"` | 达到 `max_tokens` 限制 |
| `"tool_calls"` | 模型调用了工具/函数 |
| `"recover_stop"` | 错误恢复后停止 |

---

## 10. 中间件与预处理步骤

### 身份验证（`middleware.py`）

`AuthenticationMiddleware` 检查 `Authorization: Bearer <token>` 请求头，与 `--api-key` 或 `FD_API_KEY` 环境变量中的 token 列表比对。

### Chat Template 处理（`serving_engine.py`，第 258–269 行）

`ZmqOpenAIServing._process_chat_template_kwargs()` 将服务端配置的 `chat_template` 注入到请求字典中（若请求未设置），并构建 `chat_template_kwargs` 字典（包含 `add_stop_sequences`）。

### 工具调用处理

- `ToolParserManager`（`preprocess.py:20`）按名称解析工具解析器类
- 非流式：`tool_parser_obj.extract_tool_calls(text, request)` 解析完整 JSON
- 流式：`tool_parser_obj.extract_tool_calls_streaming(prev_text, curr_text, delta, ...)` 增量解析 JSON
- 返回 `ExtractedToolCallInformation(tools_called, tool_calls=[ToolCall(id, type, function)], content)`

**注意**：流式处理中，仅当 `tool_call_delta_message.tool_calls` 为 truthy 时，才会用其 content/tool_calls 覆盖 outputs；tool_calls 为空列表会导致 content 被忽略（`base_processor.py:329-346`）。

### 推理解析器（Reasoning Parser）

- `ReasoningParserManager.get_reasoning_parser(name)` 解析对应的解析器（如 DeepSeek R1、Qwen3）
- 流式：`extract_reasoning_content_streaming()` 从响应文本中分离 `<think>...</think>` 内容
- 非流式：`extract_reasoning_content()` 返回 `(reasoning_content, response_text)` 元组
- `enable_thinking` 标志根据检测到的 prompt 状态在请求中设置

### Logprobs 处理（`_create_chat_logprobs()`，第 817–846 行）

将原始 `top_logprobs` 张量 `[token_ids, logprobs, sampled_token_ranks]` 转换为：
`LogProbs(content=[LogProbEntry(token, logprob, bytes, top_logprobs)])`

---

## 11. 完整调用链总览

```
HTTP POST /v1/chat/completions
  └── FastAPI 验证 ChatCompletionRequest（Pydantic）
  └── api_server.create_chat_completion()
        ├── connection_manager()         ← 全局信号量（max_concurrency // workers）
        └── OpenAIServingChat.create_chat_completion(request)
              ├── [主节点检查、模型检查]
              ├── engine_client.semaphore.acquire()  ← 每 worker 信号量
              ├── request.to_dict_for_infer(request_id)
              │     └── 处理 response_format、guided decoding、disable_chat_template
              ├── EngineClient.format_and_add_data(req_dict)
              │     └── add_requests(task)
              │           ├── process_messages()            ← 规范化多模态内容
              │           ├── data_processor.process_request_dict(task, max_len)
              │           │     ├── _apply_default_parameters()  ← 应用生成配置默认值
              │           │     ├── update_stop_seq()            ← stop 字符串 → token IDs
              │           │     ├── update_bad_words()           ← bad_words → token IDs
              │           │     ├── messages2ids()（若有 messages）:
              │           │     │     └── tokenizer.apply_chat_template()  ← Jinja2 模板
              │           │     │     └── tokenizer.tokenize() + convert_tokens_to_ids()
              │           │     ├── text2ids()（若为纯文本 prompt）
              │           │     ├── 截断 prompt_token_ids
              │           │     ├── 计算 max_tokens
              │           │     ├── temperature/top_p 裁剪
              │           │     └── reasoning_parser.get_model_status()
              │           ├── valid_parameters()
              │           └── zmq_client.send_json(task)    ← 通过 ZMQ 发送至引擎 worker
              │
              ├── [stream=True] → chat_completion_stream_generator()
              │     ├── connection_manager.get_connection()   ← ZMQ DEALER socket
              │     ├── 循环从 ZMQ 队列接收响应：
              │     │     ├── ChatResponseProcessor.process_response_chat(stream=True)
              │     │     │     └── data_processor.process_response_dict(stream=True)
              │     │     │           ├── ids2tokens()              ← 增量解码
              │     │     │           ├── reasoning_parser.extract_reasoning_content_streaming()
              │     │     │           └── tool_parser.extract_tool_calls_streaming()
              │     │     └── 构建 DeltaMessage + ChatCompletionResponseStreamChoice
              │     │     └── yield "data: {chunk.model_dump_json()}\n\n"
              │     └── yield "data: [DONE]\n\n"
              │
              └── [stream=False] → chat_completion_full_generator()
                    ├── connection_manager.get_connection()
                    ├── 循环接收响应（累积所有 token）：
                    │     ├── ChatResponseProcessor.process_response_chat(stream=False)
                    │     │     └── data_processor.process_response_dict(stream=False)
                    │     │           ├── ids2tokens()              ← 完整序列一次性解码
                    │     │           ├── reasoning_parser.extract_reasoning_content()
                    │     │           └── tool_parser.extract_tool_calls()
                    │     └── finished=True 时：_create_chat_completion_choice()
                    │           └── 构建 ChatMessage + ChatCompletionResponseChoice
                    └── return ChatCompletionResponse(choices, usage)
```

---

## 12. 核心数据模型关系表

### Pydantic 模型（`protocol.py`）

| 模型类 | 行号 | 用途 |
|---|---|---|
| `ChatCompletionRequest` | 663 | 入站 HTTP 请求 |
| `ChatCompletionResponse` | 275 | 非流式响应 |
| `ChatCompletionStreamResponse` | 341 | 流式 chunk |
| `ChatCompletionResponseChoice` | 260 | 非流式候选项 |
| `ChatCompletionResponseStreamChoice` | 325 | 流式候选项 |
| `DeltaMessage` | 308 | 流式消息增量 |
| `ChatMessage` | 243 | 非流式消息 |
| `UsageInfo` | 116 | Token 用量统计 |
| `PromptTokenUsageInfo` | 90 | Prompt token 详情（缓存/图片/视频） |
| `CompletionTokenUsageInfo` | 71 | 生成 token 详情（推理/图片） |
| `LogProbs` / `LogProbEntry` | 288–306 | Logprob 数据 |
| `ToolCall` / `DeltaToolCall` | 181–209 | 工具调用结构 |
| `ErrorResponse` / `ErrorInfo` | 56–68 | 错误响应 |
| `StreamOptions` | 437 | 流式选项（include_usage 等） |
| `ResponseFormat` / `AnyResponseFormat` | 477–486 | 输出格式控制 |

### 数据类（其他文件）

| 类 | 文件 | 用途 |
|---|---|---|
| `SamplingParams` | `engine/sampling_params.py:30` | 引擎侧采样配置 |
| `Request` | `engine/request.py:77` | 引擎内部请求 |
| `SpeculateMetrics` | `worker/output.py` | 投机解码统计（实现了 `to_dict()`） |
| `LogprobsLists` / `LogprobsTensors` | `worker/output.py` | 原始 logprob 张量 |
| `ServeContext` | `serving_engine.py:41` | 请求上下文（通用泛型容器） |

### 关键环境变量

| 变量 | 默认值 | 作用 |
|---|---|---|
| `FD_USE_HF_TOKENIZER` | `False` | 使用 HuggingFace 分词器（而非 PaddleFormers） |
| `FD_SUPPORT_MAX_CONNECTIONS` | — | 最大并发连接数 |
| `ZMQ_SEND_BATCH_DATA` | — | ZMQ 批量发送模式开关 |
| `FD_ENABLE_ASYNC_LLM` | `False` | 启用 AsyncLLM 引擎路径 |
| `FD_WORKER_ALIVE_TIMEOUT` | — | Worker 心跳超时阈值 |
| `FD_MAX_STOP_SEQS_NUM` | — | 最大 stop 序列数量 |
| `FD_STOP_SEQS_MAX_LEN` | — | 单个 stop 序列最大长度 |

---

*文档生成时间：2026-04-03*  
*基于 FastDeploy 代码库分析，追踪路径：`fastdeploy/entrypoints/openai/serving_chat.py`*
