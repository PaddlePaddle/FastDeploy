# FastDeploy 输入长度验证逻辑梳理

本文档梳理了FastDeploy代码库中所有判断输入长度超长的逻辑，包括与组合参数（如min_token、max_tokens等）导致超长的情况。

## 一、主要验证位置

### 1. 引擎层验证 (Engine Layer)

#### 1.1 `/fastdeploy/engine/engine.py`

**验证点1：输入token数 + min_tokens >= max_model_len**
- 位置：第286-293行
- 触发条件：`input_ids_len + min_tokens >= self.cfg.model_config.max_model_len`
- 错误消息（已统一）：
  ```
  This model's maximum context length is {max_model_len} tokens. 
  However, your messages resulted in {input_ids_len} tokens. 
  `inputs` tokens + `min_tokens` must be <= {max_model_len}.
  ```

**验证点2：输入token数 > max_model_len**
- 位置：第295-302行
- 触发条件：`input_ids_len > self.cfg.model_config.max_model_len`
- 错误消息（已统一）：
  ```
  This model's maximum context length is {max_model_len} tokens. 
  However, your messages resulted in {input_ids_len} tokens. 
  Input tokens exceed the configured limit.
  ```

#### 1.2 `/fastdeploy/engine/async_llm.py`

**验证点3：异步LLM引擎的输入验证**
- 位置：第427-434行
- 触发条件：`input_ids_len + min_tokens >= self.cfg.model_config.max_model_len`
- 错误消息（已统一）：
  ```
  This model's maximum context length is {max_model_len} tokens. 
  However, your messages resulted in {input_ids_len} tokens. 
  `inputs` tokens + `min_tokens` must be <= {max_model_len}.
  ```

### 2. API入口层验证 (API Entrypoint Layer)

#### 2.1 `/fastdeploy/entrypoints/engine_client.py`

**验证点4：引擎客户端的输入验证（input + min_tokens）**
- 位置：第307-314行
- 触发条件：`input_ids_len + min_tokens >= self.max_model_len`
- 错误消息（已统一）：
  ```
  This model's maximum context length is {max_model_len} tokens. 
  However, your messages resulted in {input_ids_len} tokens. 
  `inputs` tokens + `min_tokens` must be <= {max_model_len}.
  ```

**验证点5：引擎客户端的输入验证（纯输入长度）**
- 位置：第316-323行
- 触发条件：`input_ids_len > self.max_model_len`
- 错误消息（已统一）：
  ```
  This model's maximum context length is {max_model_len} tokens. 
  However, your messages resulted in {input_ids_len} tokens. 
  Input tokens exceed the configured limit.
  ```

### 3. 参数验证 (Parameter Validation)

#### 3.1 `/fastdeploy/engine/sampling_params.py`

**验证点6：max_tokens参数验证**
- 位置：第199-200行
- 触发条件：`self.max_tokens is not None and self.max_tokens < 1`
- 错误消息：`max_tokens must be at least 1, got {self.max_tokens}.`

**验证点7：reasoning_max_tokens vs max_tokens**
- 位置：第202-203行
- 触发条件：`self.reasoning_max_tokens is not None and self.reasoning_max_tokens > self.max_tokens`
- 错误消息：`reasoning_max_tokens must be less than max_tokens...`

**验证点8：min_tokens参数验证**
- 位置：第205-206行
- 触发条件：`self.min_tokens < 0`
- 错误消息：`min_tokens must be greater than or equal to 0...`

**验证点9：min_tokens vs max_tokens**
- 位置：第207-210行
- 触发条件：`self.max_tokens is not None and self.min_tokens > self.max_tokens`
- 错误消息：`min_tokens must be less than or equal to max_tokens...`

### 4. 数据处理器层 (Data Processor Layer)

#### 4.1 文本处理器

**位置：`/fastdeploy/input/text_processor.py`**

- 第262-263行：截断超长的prompt（静默处理，不抛出错误）
  ```python
  if max_model_len is not None and len(request.prompt_token_ids) > max_model_len:
      request.prompt_token_ids = request.prompt_token_ids[: max_model_len - 1]
  ```

- 第347-348行：字典格式请求的截断（静默处理，不抛出错误）
  ```python
  if max_model_len is not None and len(request["prompt_token_ids"]) > max_model_len:
      request["prompt_token_ids"] = request["prompt_token_ids"][: max_model_len - 1]
  ```

#### 4.2 多模态处理器

**Qwen VL处理器：`/fastdeploy/input/qwen_vl_processor/qwen_vl_processor.py`**
- 第262-265行：截断处理

**Qwen3 VL处理器：`/fastdeploy/input/qwen3_vl_processor/qwen3_vl_processor.py`**
- 第264-267行：截断处理

**ERNIE 4.5处理器：`/fastdeploy/input/ernie4_5_processor.py`**
- 第143-144行：截断处理
- 第227-228行：字典格式的截断处理

**ERNIE 4.5 VL处理器：`/fastdeploy/input/ernie4_5_vl_processor/ernie4_5_vl_processor.py`**
- 第263-264行：截断处理

**PaddleOCR VL处理器：`/fastdeploy/input/paddleocr_vl_processor/paddleocr_vl_processor.py`**
- 第245-248行：截断处理

## 二、错误提示统一方案

### 2.1 LiteLLM兼容的错误消息模式

为了与LiteLLM SDK兼容，我们使用以下错误消息模式：

1. **"this model's maximum context length is"** - 用于所有长度超限错误
2. **"input tokens exceed the configured limit"** - 用于纯输入长度超限
3. **"`inputs` tokens + `min_tokens` must be"** - 用于输入+min_tokens组合超限

### 2.2 统一后的错误消息

#### 场景1：输入 + min_tokens >= max_model_len
```
This model's maximum context length is {max_model_len} tokens. 
However, your messages resulted in {input_ids_len} tokens. 
`inputs` tokens + `min_tokens` must be <= {max_model_len}.
```

#### 场景2：输入 > max_model_len
```
This model's maximum context length is {max_model_len} tokens. 
However, your messages resulted in {input_ids_len} tokens. 
Input tokens exceed the configured limit.
```

## 三、其他相关验证

### 3.1 stop_seqs验证

**位置：**
- `/fastdeploy/engine/engine.py` 第299-317行
- `/fastdeploy/entrypoints/engine_client.py` 第322-340行

**验证类型：**
1. stop序列数量：`len(stop_seqs_len) > max_stop_seqs_num`
2. 单个stop序列长度：`single_stop_seq_len > stop_seqs_max_len`

**错误消息格式：**（未修改，保持原样）
- "Length of stop ({stop_seqs_len}) exceeds the limit max_stop_seqs_num({max_stop_seqs_num})."
- "Length of stop_seqs({single_stop_seq_len}) exceeds the limit stop_seqs_max_len({stop_seqs_max_len})."

## 四、总结

### 4.1 验证策略

FastDeploy采用**分层验证**策略：

1. **数据处理器层（静默截断）**：在输入处理阶段，如果输入超长，会自动截断到max_model_len-1，不抛出错误
2. **引擎层（严格验证）**：在引擎执行前，严格验证输入长度+min_tokens是否超限，超限则抛出错误

### 4.2 验证覆盖的场景

1. ✅ 输入token数量超过max_model_len
2. ✅ 输入token数量 + min_tokens >= max_model_len
3. ✅ max_tokens参数验证（>= 1）
4. ✅ min_tokens参数验证（>= 0且<= max_tokens）
5. ✅ reasoning_max_tokens参数验证（<= max_tokens）
6. ✅ stop_seqs数量和长度验证

### 4.3 LiteLLM兼容性

所有主要的长度超限错误消息已统一，使用LiteLLM可识别的模式：
- ✅ "this model's maximum context length is"
- ✅ "input tokens exceed the configured limit"
- ✅ "`inputs` tokens + `min_tokens` must be"

这确保了FastDeploy可以与LiteLLM SDK无缝集成，LiteLLM能够正确识别和处理长度超限错误。
