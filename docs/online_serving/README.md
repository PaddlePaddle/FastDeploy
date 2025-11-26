[简体中文](../zh/online_serving/README.md)

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

## Chat Completion API
FastDeploy provides a Chat Completion API that is compatible with the OpenAI protocol, allowing user requests to be sent directly using OpenAI's request method.

### Sending User Requests

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
    {"role": "user", "content": "Hello!"}
  ],
  "logprobs": true, "top_logprobs": 0,
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

### Compatible OpenAI Parameters
```python
messages: Union[List[Any], List[int]]
# List of input messages, which can be text messages (`List[Any]`, typically `List[dict]`) or token ID lists (`List[int]`).

tools: Optional[List[ChatCompletionToolsParam]] = None
# List of tool call configurations, used for enabling function calling (Function Calling) or tool usage (e.g., ReAct framework).

model: Optional[str] = "default"
# Specifies the model name or version to use, defaulting to `"default"` (which may point to the base model).

frequency_penalty: Optional[float] = None
# Frequency penalty coefficient, reducing the probability of generating the same token repeatedly (`>1.0` suppresses repetition, `<1.0` encourages repetition, default `None` disables).

logprobs: Optional[bool] = False
# Whether to return the log probabilities of each generated token, used for debugging or analysis.

top_logprobs: Optional[int] = 0
# Returns the top `top_logprobs` tokens and their log probabilities for each generated position (default `0` means no return).

max_tokens: Optional[int] = Field(
    default=None,
    deprecated="max_tokens is deprecated in favor of the max_completion_tokens field",
)
# Deprecated: Maximum number of tokens to generate (recommended to use `max_completion_tokens` instead).

max_completion_tokens: Optional[int] = None
# Maximum number of tokens to generate (recommended alternative to `max_tokens`), no default limit (restricted by the model's context window).

presence_penalty: Optional[float] = None
# Presence penalty coefficient, reducing the probability of generating new topics (unseen topics) (`>1.0` suppresses new topics, `<1.0` encourages new topics, default `None` disables).

stream: Optional[bool] = False
# Whether to enable streaming output (return results token by token), default `False` (returns complete results at once).

stream_options: Optional[StreamOptions] = None
# Additional configurations for streaming output (such as chunk size, timeout, etc.), refer to the specific definition of `StreamOptions`.

temperature: Optional[float] = None
# Temperature coefficient, controlling generation randomness (`0.0` for deterministic generation, `>1.0` for more randomness, default `None` uses model default).

top_p: Optional[float] = None
# Nucleus sampling threshold, only retaining tokens whose cumulative probability exceeds `top_p` (default `None` disables).

response_format: Optional[AnyResponseFormat] = None
# Specifies the output format (such as JSON, XML, etc.), requires passing a predefined format configuration object.

user: Optional[str] = None
# User identifier, used for tracking or distinguishing requests from different users (default `None` does not pass).

metadata: Optional[dict] = None
# Additional metadata, used for passing custom information (such as request ID, debug markers, etc.).

n: Optional[int] = 1
# Number of candidate outputs to generate (i.e., return multiple independent text completions). Default 1 (return only one result).

seed: Optional[int] = Field(default=None, ge=0, le=922337203685477580)
# Random seed for controlling deterministic generation (same seed + input yields identical results).
# Must be in range `[0, 922337203685477580]`. Default None means no fixed seed.

stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
# Stop generation conditions - can be a single string or list of strings.
# Generation terminates when any stop string is produced (default empty list means disabled).

```

### Additional Parameters Added by FastDeploy

> Note:
When sending requests using curl, the following parameters can be used directly;
When sending requests using openai.Client, these parameters need to be placed in the `extra_body` parameter, e.g. `extra_body={"chat_template_kwargs": {"enable_thinking":True}, "include_stop_str_in_output": True}`.

The following sampling parameters are supported.
```python
top_k: Optional[int] = None
# Limits the consideration to the top K tokens with the highest probability at each generation step, used to control randomness (default None means no limit).

min_p: Optional[float] = None
# Nucleus sampling threshold, only retaining tokens whose cumulative probability exceeds min_p (default None means disabled).

min_tokens: Optional[int] = None
# Forces a minimum number of tokens to be generated, avoiding premature truncation (default None means no limit).

include_stop_str_in_output: Optional[bool] = False
# Whether to include the stop string content in the output (default False, meaning output is truncated when a stop string is encountered).

bad_words: Optional[List[str]] = None
# List of forbidden words (e.g., sensitive words) that the model should avoid generating (default None means no restriction).

bad_words_token_ids: Optional[List[int]] = None
# List of forbidden token ids that the model should avoid generating (default None means no restriction).

repetition_penalty: Optional[float] = None
# Repetition penalty coefficient, reducing the probability of repeating already generated tokens (`>1.0` suppresses repetition, `<1.0` encourages repetition, default None means disabled).

stop_token_ids: Optional[List[int]] = Field(default_factory=list)
# Stop generation token IDs - list of token IDs that trigger early termination when generated.
# Typically used alongside `stop` for complementary stopping conditions (default empty list means disabled).

```

The following extra parameters are supported:
```python
chat_template_kwargs: Optional[dict] = None
# Additional parameters passed to the chat template, used for customizing dialogue formats (default None).

chat_template: Optional[str] = None
# Custom chat template will override the model's default chat template (default None).

reasoning_max_tokens: Optional[int] = None
# Maximum number of tokens to generate during reasoning (e.g., CoT, chain of thought) (default None means using global max_tokens).

structural_tag: Optional[str] = None
# Structural tag, used to mark specific structures of generated content (such as JSON, XML, etc., default None).

guided_json: Optional[Union[str, dict, BaseModel]] = None
# Guides the generation of content conforming to JSON structure, can be a JSON string, dictionary, or Pydantic model (default None).

guided_regex: Optional[str] = None
# Guides the generation of content conforming to regular expression rules (default None means no restriction).

guided_choice: Optional[List[str]] = None
# Guides the generation of content selected from a specified candidate list (default None means no restriction).

guided_grammar: Optional[str] = None
# Guides the generation of content conforming to grammar rules (such as BNF) (default None means no restriction).

return_token_ids: Optional[bool] = None
# Whether to return the token IDs of the generation results instead of text (default None means return text).

prompt_token_ids: Optional[List[int]] = None
# Directly passes the token ID list of the prompt, skipping the text encoding step (default None means using text input).

disable_chat_template: Optional[bool] = False
# Whether to disable chat template rendering, using raw input directly (default False means template is enabled).

temp_scaled_logprobs: Optional[bool] = False
# Whether to divide the logits by the temperature coefficient when calculating logprobs (default is False, meaning the logits are not divided by the temperature coefficient).

top_p_normalized_logprobs: Optional[bool] = False
# Whether to perform top-p normalization when calculating logprobs (default is False, indicating that top-p normalization is not performed).

include_draft_logprobs: Optional[bool] = False
# Whether to return log probabilities during draft stages (e.g., pre-generation or intermediate steps)
# for debugging or analysis of the generation process (default False means not returned).

logits_processors_args: Optional[Dict] = None
# Additional arguments for logits processors, enabling customization of generation logic
# (e.g., dynamically adjusting probability distributions).

mm_hashes: Optional[list] = None
# Hash values for multimodal (e.g., image/audio) inputs, used for verification or tracking.
# Default None indicates no multimodal input or hash validation required.

```

### Differences in Return Fields

Additional return fields added by FastDeploy:

- `arrival_time`: Cumulative time consumed for all tokens
- `reasoning_content`: Return results of the chain of thought
- `prompt_token_ids`: List of token IDs for the input sequence
- `completion_token_ids`: List of token IDs for the output sequence

Overview of return parameters:

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

# Fields returned for streaming responses
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
The Completion API interface is mainly used for continuation scenarios, suitable for users who have customized context input and expect the model to only output continuation content; the inference process does not add other `prompt` concatenations.

### Sending User Requests

Here is an example of sending a user request using the curl command:

```bash
curl -X POST "http://0.0.0.0:8188/v1/completions" \
-H "Content-Type: application/json" \
-d '{
  "prompt": "以下是一篇关于深圳文心公园的500字游记和赏析："
}'
```

Here is an example of sending a user request using a Python script:

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

For an explanation of the OpenAI protocol, refer to the [OpenAI Completion API](https://platform.openai.com/docs/api-reference/completions/create)。

### Compatible OpenAI Parameters
```python
model: Optional[str] = "default"
# Specifies the model name or version to use, defaulting to `"default"` (which may point to the base model).

prompt: Union[List[int], List[List[int]], str, List[str]]
# Input prompt, supporting multiple formats:
#   - `str`: Plain text prompt (e.g., `"Hello, how are you?"`).
#   - `List[str]`: Multiple text segments (e.g., `["User:", "Hello!", "Assistant:", "Hi!"]`).
#   - `List[int]`: Directly passes a list of token IDs (e.g., `[123, 456]`).
#   - `List[List[int]]`: List of multiple token ID lists (e.g., `[[123], [456, 789]]`).

best_of: Optional[int] = None
# Generates `best_of` candidate results and returns the highest-scoring one (requires `n=1`).

frequency_penalty: Optional[float] = None
# Frequency penalty coefficient, reducing the probability of generating the same token repeatedly (`>1.0` suppresses repetition, `<1.0` encourages repetition).

logprobs: Optional[int] = None
# Returns the log probabilities of each generated token, can specify the number of candidates to return.

max_tokens: Optional[int] = None
# Maximum number of tokens to generate (including input and output), no default limit (restricted by the model's context window).

presence_penalty: Optional[float] = None
# Presence penalty coefficient, reducing the probability of generating new topics (unseen topics) (`>1.0` suppresses new topics, `<1.0` encourages new topics).

echo: Optional[bool] = False
# Whether to include the input prompt in the generated output (default: `False`, i.e., exclude the prompt).

n: Optional[int] = 1
# Number of candidate outputs to generate (i.e., return multiple independent text completions). Default 1 (return only one result).

seed: Optional[int] = Field(default=None, ge=0, le=922337203685477580)
# Random seed for controlling deterministic generation (same seed + input yields identical results).
# Must be in range `[0, 922337203685477580]`. Default None means no fixed seed.

stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
# Stop generation conditions - can be a single string or list of strings.
# Generation terminates when any stop string is produced (default empty list means disabled).

stream: Optional[bool] = False
# Whether to enable streaming output (return results token by token), default `False` (returns complete results at once).

stream_options: Optional[StreamOptions] = None
# Additional configurations for streaming output (such as chunk size, timeout, etc.), refer to the specific definition of `StreamOptions`.

temperature: Optional[float] = None
# Temperature coefficient, controlling generation randomness (`0.0` for deterministic generation, `>1.0` for more randomness, default `None` uses model default).

top_p: Optional[float] = None
# Nucleus sampling threshold, only retaining tokens whose cumulative probability exceeds `top_p` (default `None` disables).

response_format: Optional[AnyResponseFormat] = None
# Specifies the output format (such as JSON, XML, etc.), requires passing a predefined format configuration object.

user: Optional[str] = None
# User identifier, used for tracking or distinguishing requests from different users (default `None` does not pass).

```

### Additional Parameters Added by FastDeploy

> Note:
When sending requests using curl, the following parameters can be used directly;
When sending requests using openai.Client, these parameters need to be placed in the `extra_body` parameter, e.g. `extra_body={"chat_template_kwargs": {"enable_thinking":True}, "include_stop_str_in_output": True}`.

The following sampling parameters are supported.
```python
top_k: Optional[int] = None
# Limits the consideration to the top K tokens with the highest probability at each generation step, used to control randomness (default None means no limit).

min_p: Optional[float] = None
# Nucleus sampling threshold, only retaining tokens whose cumulative probability exceeds min_p (default None means disabled).

min_tokens: Optional[int] = None
# Forces a minimum number of tokens to be generated, avoiding premature truncation (default None means no limit).

include_stop_str_in_output: Optional[bool] = False
# Whether to include the stop string content in the output (default False, meaning output is truncated when a stop string is encountered).

bad_words: Optional[List[str]] = None
# List of forbidden words (e.g., sensitive words) that the model should avoid generating (default None means no restriction).

bad_words_token_ids: Optional[List[int]] = None
# List of forbidden token ids that the model should avoid generating (default None means no restriction).

stop_token_ids: Optional[List[int]] = Field(default_factory=list)
# Stop generation token IDs - list of token IDs that trigger early termination when generated.
# Typically used alongside `stop` for complementary stopping conditions (default empty list means disabled).

repetition_penalty: Optional[float] = None
# Repetition penalty coefficient, reducing the probability of repeating already generated tokens (`>1.0` suppresses repetition, `<1.0` encourages repetition, default None means disabled).
```

The following extra parameters are supported:
```python
guided_json: Optional[Union[str, dict, BaseModel]] = None
# Guides the generation of content conforming to JSON structure, can be a JSON string, dictionary, or Pydantic model (default None).

guided_regex: Optional[str] = None
# Guides the generation of content conforming to regular expression rules (default None means no restriction).

guided_choice: Optional[List[str]] = None
# Guides the generation of content selected from a specified candidate list (default None means no restriction).

guided_grammar: Optional[str] = None
# Guides the generation of content conforming to grammar rules (such as BNF) (default None means no restriction).

return_token_ids: Optional[bool] = None
# Whether to return the token IDs of the generation results instead of text (default None means return text).

prompt_token_ids: Optional[List[int]] = None
# Directly passes the token ID list of the prompt, skipping the text encoding step (default None means using text input).

temp_scaled_logprobs: Optional[bool] = False
# Whether to divide the logits by the temperature coefficient when calculating logprobs (default is False, meaning the logits are not divided by the temperature coefficient).

top_p_normalized_logprobs: Optional[bool] = False
# Whether to perform top-p normalization when calculating logprobs (default is False, indicating that top-p normalization is not performed).

include_draft_logprobs: Optional[bool] = False
# Whether to return log probabilities during draft stages (e.g., pre-generation or intermediate steps)
# for debugging or analysis of the generation process (default False means not returned).

logits_processors_args: Optional[Dict] = None
# Additional arguments for logits processors, enabling customization of generation logic
# (e.g., dynamically adjusting probability distributions).

mm_hashes: Optional[list] = None
# Hash values for multimodal (e.g., image/audio) inputs, used for verification or tracking.
# Default None indicates no multimodal input or hash validation required.

```

### Overview of Return Parameters

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

# Fields returned for streaming responses
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
