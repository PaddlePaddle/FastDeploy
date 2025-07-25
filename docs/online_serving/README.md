# OpenAI Protocol-Compatible API Server

FastDeploy provides a service-oriented deployment solution that is compatible with the OpenAI protocol. Users can quickly deploy it using the following command:

```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-0.3B-Paddle \
       --port 8188 --tensor-parallel-size 8 \
       --max-model-len 32768
```

To enable log probability output, simply deploy with the following command:

```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-0.3B-Paddle \
       --port 8188 --tensor-parallel-size 8 \
       --max-model-len 32768 \
       --enable-logprob
```

For more usage methods of the command line during service deployment, refer to [Parameter Descriptions](../parameters.md).

## Sending User Requests

The FastDeploy interface is compatible with the OpenAI protocol, allowing user requests to be sent directly using OpenAI's request method.

Here is an example of sending a user request using the curl command:

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'
```

Here's an example curl command demonstrating how to include the logprobs parameter in a user request:

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Hello!"}, "logprobs": true, "top_logprobs": 5
  ]
}'
```

Here is an example of sending a user request using a Python script:

```python
import openai
host = "0.0.0.0"
port = "8170"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "Rewrite Li Bai's 'Quiet Night Thought' as a modern poem"},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

For a description of the OpenAI protocol, refer to the document [OpenAI Chat Completion API](https://platform.openai.com/docs/api-reference/chat/create).

## Parameter Differences
### Request Parameter Differences
The differences in request parameters between FastDeploy and the OpenAI protocol are as follows. Other request parameters will be ignored:

- `prompt` (supported only in the `v1/completions` interface)
- `messages` (supported only in the `v1/chat/completions` interface)
- `logprobs`: Optional[bool] = False (supported only in the `v1/chat/completions` interface)
- `top_logprobs`: Optional[int] = None (supported only in the `v1/chat/completions` interface. An integer between 0 and 20,logprobs must be set to true if this parameter is used)
- `frequency_penalty`: Optional[float] = 0.0
- `max_tokens`: Optional[int] = 16
- `presence_penalty`: Optional[float] = 0.0
- `stream`: Optional[bool] = False
- `stream_options`: Optional[StreamOptions] = None
- `temperature`: Optional[float] = None
- `top_p`: Optional[float] = None
- `metadata`: Optional[dict] = None (supported only in `v1/chat/completions` for configuring additional parameters, e.g., `metadata={"enable_thinking": True}`)
  - `min_tokens`: Optional[int] = 1 (minimum number of tokens generated)
  - `reasoning_max_tokens`: Optional[int] = None (maximum number of tokens for reasoning content, defaults to the same as `max_tokens`)
  - `enable_thinking`: Optional[bool] = True (whether to enable reasoning for models that support deep thinking)
  - `repetition_penalty`: Optional[float] = None (coefficient for directly penalizing repeated token generation (>1 penalizes repetition, <1 encourages repetition))

> Note: For multimodal models, since the reasoning chain is enabled by default, resulting in overly long outputs, `max_tokens` can be set to the model's maximum output length or the default value can be used.

### Return Field Differences

The additional return fields added by FastDeploy are as follows:

- `arrival_time`: Returns the cumulative time taken for all tokens
- `reasoning_content`: The returned result of the reasoning chain

Overview of return parameters:

```python
ChatCompletionStreamResponse:
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
 ChatCompletionResponseStreamChoice:
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None
    arrival_time: Optional[float] = None
DeltaMessage:
    role: Optional[str] = None
    content: Optional[str] = None
    token_ids: Optional[List[int]] = None
    reasoning_content: Optional[str] = None
```
