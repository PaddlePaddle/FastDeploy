# Logits Processors

## Overview

A **Logits Processor (LP)** sits between *model output logits* and the *sampler* (top-k/top-p/temperature…). It applies pluggable transformations to logits **before** sampling (e.g., weighting, masking, penalties, biases).

## Key Features

- **Server-level registration**: declare available processors at startup via `--logits-processors`. The declaration order is the execution order.
- **Per-request control**: enable and configure processors via the `logits_processors_args` field in the request body.
- **Built-in processor**: commonly used processors are provided, e.g., `LogitBiasLogitsProcessor`, which can be loaded directly by class name.
- **Extensible interface**: a standard `LogitsProcessor` interface is provided for user-defined processors, which can be loaded by FQCN `module.path:ClassName`.

## Usage

### Online Service

#### 1. Start the service (register logits processors)

Register processors with `--logits-processors` when starting the service. For a built-in processor like `LogitBiasLogitsProcessor`, pass the class name directly:

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model /path/to/model \
  --port 8180 --metrics-port 8181 --engine-worker-queue-port 8182 --cache-queue-port 8183 \
  --logits-processors LogitBiasLogitsProcessor
```

#### 2. Send a request (enable and configure as needed)

Use the `logits_processors_args` field in the REST request body to enable and configure processors. Example with `LogitBiasLogitsProcessor`, which adds a bias to specified tokens. It accepts a `logit_bias` dictionary mapping *token_id* → *bias value*:

```bash
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" -H "Content-Type: application/json" -d '{
  "messages": [{"role":"user", "content":"Who is Lu Xun?"}],
  "logits_processors_args": {
    "logit_bias": {"128": 5.0, "50256": -10.0}
  }
}'
```

When using the OpenAI Python SDK, pass `logits_processors_args` through `extra_body`:

```python
import openai

client = openai.Client(base_url="http://0.0.0.0:8180/v1", api_key="EMPTY_API_KEY")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Who is Lu Xun?"}],
    extra_body={
        "logits_processors_args": {
           "logit_bias": {"128": 5.0, "50256": -10.0}
        }
    }
)
```

### Offline Inference

For offline inference, pass the `logits_processors` argument (type `list[str]`) when initializing the `LLM` instance. When generating text via the offline `chat()` or `generate()` APIs, provide the logits-processor parameters through `sampling_params`.`logits_processors_args` to enable and pass arguments to the corresponding processors.

```python
from fastdeploy import LLM, SamplingParams

llm = LLM(
    model="path/to/model",
    engine_worker_queue_port=8282,
    cache_queue_port=8383,
    logits_processors=['LogitBiasLogitsProcessor'],
)

messages = [{"role": "user", "content": "Who is Lu Xun?"}]
sampling_params = SamplingParams(
    top_p=0.95,
    max_tokens=128,
    logits_processors_args={"logit_bias": {128: 5.0, 50256: -10.0}},
)
outputs = llm.chat(messages, sampling_params)
print(outputs[0].outputs.text)
```

## Custom Logits Processor

### 1. Define your own LogitsProcessor class
Inherit the `fastdeploy.openai.logits_processor.LogitsProcessor` class and implement the `update_state()` and `apply()` methods.

- **`update_state()` is used to update the logits processor state.** The input is the inference backend’s runtime state `share_inputs`, and it returns nothing. You need to extract useful information from the runtime state to update the logits processor’s internal state.
  - For example, in the following example, we retrieve the current batch’s `logits_processors_args` from `share_inputs`, and then bulk-modify the enablement status of the logits processor for the current batch;
  - When writing your class, you should predefine the parameter names for your logits processor, such as adding a request parameter `enable_your_logits_processor` to control whether your logits processor is enabled for a request;
- **`apply()` is used to actually modify the logits tensor.** Before `apply()` runs, the model will call `update_state()` to refresh the logits processor state. Therefore, ensure your `update_state()` correctly updates the state variables used by the logits processor.
  - In the following example, we use `self.enabled`to determine whether each request in the current batch enables your logits processor, and adjust the logits tensor dynamically.
```python
from paddle import Tensor
from fastdeploy.config import FDConfig
from fastdeploy.openai.logits_processor import LogitsProcessor

class YourLogitsProcessor(LogitsProcessor):

    def __init__(self, fd_config: FDConfig) -> None:
        # Initialize your state variables here, for example obtain dtype, device, etc.
        # from fd_config. You can freely set the state variables you need, and update them
        # during each step of inference update_state()
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
        # Update your state variables here to facilitate dynamically
        # adjusting your logits processor behavior at each step of inference.
        # The latest state should be read and used in the apply() method
        self.enabled = [a.enable_your_logits_processor for a in logits_processors_args]
        return

    def apply(self, logits: Tensor) -> Tensor:
        """Apply LogitsProcessor to batch logits tensor.

        The updated tensor must be returned but may be modified in-place.
        """
        for i, e in enumerate(self.enabled):
            # Implement your core logits transformation here, and return the modified logits tensor
            logits[i] = ...
        return logits
```

### 2. Use your logits processor via online service

#### 2.2. Start the service (register your logits processor)

When registering a custom processor, pass its **FQCN** (`module.path:ClassName`) to `--logits-processors`:

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model /path/to/model \
  --port 8180 --metrics-port 8181 --engine-worker-queue-port 8182 --cache-queue-port 8183 \
  --logits-processors your.dotted.path.to.module:YourLogitsProcessor
```

#### 2.2. Send a request (enable and configure as needed)

Enable your processor per request via `logits_processors_args`:

```bash
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" -H "Content-Type: application/json" -d '{
  "messages": [{"role":"user", "content":"Who is Lu Xun?"}],
  "logits_processors_args": {
    "enable_your_logits_processor": true
  }
}'
```

Using the OpenAI Python SDK:

```python
import openai

client = openai.Client(base_url="http://0.0.0.0:8180/v1", api_key="EMPTY_API_KEY")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Who is Lu Xun?"}],
    extra_body={
        "logits_processors_args": {
            "enable_your_logits_processor": True
        }
    }
)
```

### 3. Use your logits processor via offline inference

For offline inference, pass the `logits_processors` argument (type `list[str]`) when initializing the `LLM` instance. To specify your custom logits processor, pass its FQCN (`module.path:ClassName`). When generating text via the offline `chat()` or `generate()` APIs, provide the logits-processor parameters through `sampling_params`.`logits_processors_args` to enable and pass arguments to the corresponding processors.

```python
from fastdeploy import LLM, SamplingParams

llm = LLM(
    model="path/to/model",
    engine_worker_queue_port=8282,
    cache_queue_port=8383,
    logits_processors=['your.dotted.path.to.module:YourLogitsProcessor'],
)

messages = [{"role": "user", "content": "Who is Lu Xun?"}]
sampling_params = SamplingParams(
    top_p=0.95,
    max_tokens=128,
    logits_processors_args={"enable_your_logits_processor": True},
)
outputs = llm.chat(messages, sampling_params)
print(outputs[0].outputs.text)
```
