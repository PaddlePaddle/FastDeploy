# Logits Processors

## 概述

Logits Processor（LP）位于“模型输出 logits → 采样器（top-k/top-p/temperature…）” 之间，用于在采样前对 logits 做可插拔的变换（加权、屏蔽、惩罚、偏置等）。

## 关键特性

- **服务级注册**：启动时用 `--logits-processors` 声明可用处理器，其中的声明顺序即 logits 处理器的执行顺序
- **请求级控制**：请求体通过 `logits_processors_args` 字段按需启用并传参
- **内置处理器**：提供常用处理器，如：`LogitBiasLogitsProcessor`，可直接按类名加载
- **可扩展接口**：提供 `LogitsProcessor` 类的标准接口，支持用户基于此接口编写自定义处理器，并按 FQCN 加载：`module.path:ClassName`

## 使用方法

### 在线服务

#### 1. 启动服务（注册 logits 处理器）

在启动服务时，通过 `--logits-processors` 参数注册处理器。如果应用内置的 logits 处理器，以 `LogitBiasLogitsProcessor` 为例，直接传入类名即可：

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model /path/to/model \
  --port 8180 --metrics-port 8181 --engine-worker-queue-port 8182 --cache-queue-port 8183 \
  --logits-processors LogitBiasLogitsProcessor
```

#### 2. 发送请求（按需启用并传参）

通过 RESTful API 发送请求时，通过 `logits_processors_args` 字段启用并传参，**不同的 logits 处理器需要不同的参数**。以 `LogitBiasLogitsProcessor` 为例，该处理器用于对指定 token 添加偏置。它接收 `logit_bias` 参数，为一个 dict 字典，表示 token id 到偏置值的映射。
```bash
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" -H "Content-Type: application/json" -d \
'{
  "messages": [{"role":"user", "content":"今天天气真好"}],
  "logits_processors_args": {
    "logit_bias": {"128": 5.0, "50256": -10.0},
  }
}'
```

通过 OpenAI Python SDK 发送请求时，通过 `extra_body` 参数传入 `logits_processor_args` 字段启用并传参。
```python
import openai

client = openai.Client(base_url=f"http://0.0.0.0:8180/v1", api_key="EMPTY_API_KEY")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "今天天气真好"}],
    extra_body={
        "logits_processors_args": {
           "logit_bias": {"128": 5.0, "50256": -10.0},
        }
    }
)
```

### 离线推理

离线调用场景，在初始化 `LLM` 实例时传入 `logits_processors` 参数，类型为 `list[str]`。在调用离线 `chat()` 或 `generate()` 接口生成文本时，通过 `sampling_params`.`logits_processors_args` 传入 logits 处理器参数，启用并传参给对应处理器。

```python
from fastdeploy import LLM, SamplingParams

llm = LLM(
    model="path/to/model",
    engine_worker_queue_port=8282,
    cache_queue_port=8383,
    logits_processors=['LogitBiasLogitsProcessor'],
)

messages = [{"role": "user", "content": "鲁迅是谁"}]
sampling_params = SamplingParams(
    top_p=0.95,
    max_tokens=128,
    logits_processors_args={"logit_bias": {128: 5.0, 50256: -10.0}},
)
outputs = llm.chat(messages, sampling_params)
print(outputs[0].outputs.text)
```

## 自定义 Logits Processor

### 1. 定义自己的 LogitsProcessor 类

继承 `fastdeploy.openai.logits_processor.LogitsProcessor` 类，实现 `update_state()` 和 `apply()` 方法。
- **`update_state()` 用于更新 logits 处理器状态。** 输入为推理后端的推理状态 `share_inputs`，无需返回值。你需要从推理状态中提取对 logits 处理器状态更新的有用信息。
  - 例如，在下面的示例中，我们从 `share_inputs` 中取出当前 batch 的 `logits_processors_args`，然后批量修改当前 batch 的 logits 处理器启用状态；
  - 你需要在编写类时事先约定好你的 logits 处理器参数名，例如添加请求参数 `enable_your_logits_processor`，用于控制请求是否启用你的 logits 处理器；
- **`apply()` 用于实际修改 logits 张量。** 在 apply() 执行前，模型会调用 update_state() 方法，更新 logits 处理器状态。因此，请确保你的 update_state() 实现正确更新了 logits 处理器状态变量。
  - 在下面的示例中，我们通过 `self.enabled` 判断当前 batch 各请求是否启用你的 logits 处理器，并动态调整 logits 张量。

```python

from paddle import Tensor
from fastdeploy.config import FDConfig
from fastdeploy.openai.logits_processor import LogitsProcessor

class YourLogitsProcessor(LogitsProcessor):

    def __init__(self, fd_config: FDConfig) -> None:
        # 在这里初始化你的状态变量，例如从 fd_config 取出 dtype, device 等信息
        # 你可以自由设定需要存储的状态变量，并在每一步推理中通过 update_state() 方法更新
        self.enabled = None
        return

    def update_state(self, share_inputs: dict) -> None:
        """Called when there are new output tokens, prior to each forward pass.

        Each field in the `share_inputs` dict typically stores information for all request
        slots. It has a `stop_flags` array that indicates whether a slot currently has a
        running request (`False` means the slot is active). Therefore, it is recommended to
        filter entries by `stop_flags` to keep only data for the current batch.
        """
        stop_flags = share_inputs["stop_flags"]
        logits_processors_args = share_inputs["logits_processors_args"]
        logits_processors_args = [a for a, f in zip(logits_processors_args, stop_flags) if not f]
        # 在这里更新你的状态变量，便于在每一步推理中动态调整你的 logits 处理器行为
        # 最新的状态应该在 apply() 方法中读取并使用
        self.enabled = [a.enable_your_logits_processor for a in logits_processors_args]
        return

    def apply(self, logits: Tensor) -> Tensor:
        """Apply LogitsProcessor to batch logits tensor.

        The updated tensor must be returned but may be modified in-place.
        """
        for i, e in enumerate(self.enabled):
            # 在这里实现你的核心 logits 处理逻辑，并返回修改后的 logits 张量
            logits[i] = ...
        return logits

```

### 2. 通过在线服务使用自己的 logits 处理器

#### 2.1. 启动服务（注册自己的 logits 处理器）

在启动服务时，通过 `--logits-processors` 参数注册你的处理器。在传入自定义处理器时，需要传入 FQCN（Fully Qualified Class Name），即 `module.path:ClassName`。

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model /path/to/model \
  --port 8180 --metrics-port 8181 --engine-worker-queue-port 8182 --cache-queue-port 8183 \
  --logits-processors your.dotted.path.to.module:YourLogitsProcessor
```

#### 2.2. 发送请求（按需启用并传参）

通过 RESTful API 发送请求时，通过 `logits_processors_args` 字段启用并传参：
```bash
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" -H "Content-Type: application/json" -d \
'{
  "messages": [{"role":"user", "content":"今天天气真好"}],
  "logits_processors_args": {
    "enable_your_logits_processor": true
  }
}'
```

通过 OpenAI Python SDK 发送请求时，通过 `extra_body` 参数传入 `logits_processor_args` 字段启用并传参：

```python
import openai

client = openai.Client(base_url=f"http://0.0.0.0:8180/v1", api_key="EMPTY_API_KEY")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "今天天气真好"}],
    extra_body={
        "logits_processors_args": {
            "enable_your_logits_processor": True
        }
    }
)
```

### 3. 通过离线调用使用自己的 logits 处理器

在初始化 `LLM` 实例时传入 `logits_processors` 参数，类型为 `list[str]`。在传入自定义处理器时，需要传入 FQCN（Fully Qualified Class Name），即 `module.path:ClassName`。在调用离线 `chat()` 或 `generate()` 接口生成文本时，通过 `sampling_params`.`logits_processors_args` 传入 logits 处理器参数，启用并传参给对应处理器。

```python
from fastdeploy import LLM, SamplingParams

llm = LLM(
    model="path/to/model",
    engine_worker_queue_port=8282,
    cache_queue_port=8383,
    logits_processors=['your.dotted.path.to.module:YourLogitsProcessor'],
)

messages = [{"role": "user", "content": "鲁迅是谁"}]
sampling_params = SamplingParams(
    top_p=0.95,
    max_tokens=128,
    logits_processors_args={"enable_your_logits_processor": True},
)
outputs = llm.chat(messages, sampling_params)
print(outputs[0].outputs.text)
```
