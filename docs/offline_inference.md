# Offline Inference

## 1. Usage
FastDeploy supports offline inference by loading models locally and processing user data. Usage examples:

### Text Completion Interface (LLM.generate)

```python
from fastdeploy import LLM, SamplingParams

prompts = [
    "把李白的静夜思改写为现代诗",
    "Write me a poem about large language model.",
]

# Sampling parameters
sampling_params = SamplingParams(top_p=0.95, max_tokens=6400)

# Load model
llm = LLM(model="ERNIE-4.5-0.3B", tensor_parallel_size=1, max_model_len=8192)

# Batch inference (internal request queuing and dynamic batching)
outputs = llm.generate(prompts, sampling_params)

# Output results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
```

### Chat Interface (LLM.chat)
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

# Sampling parameters
sampling_params = SamplingParams(top_p=0.95, max_tokens=6400)

# Load model
llm = LLM(model="ERNIE-4.5-0.3B", tensor_parallel_size=1, max_model_len=8192)
# Batch inference (internal request queuing and dynamic batching)
outputs = llm.chat(messages, sampling_params)

# Output results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
```

Documentation for `SamplingParams`, `LLM.generate`, `LLM.chat`, and output structure `RequestOutput` is provided below.

> Note: For X1 model output

```python
# Output results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    reasoning_text = output.outputs.resoning_content
```

## 2. API Documentation

### 2.1 fastdeploy.LLM

For ```LLM``` configuration, refer to [Parameter Documentation](parameters.md).

> Configuration Notes:
> 1. `port` and `metrics_port` is only used for online inference.
> 2. After startup, the service logs KV Cache block count (e.g. `total_block_num:640`). Multiply this by block_size (default 64) to get total cacheable tokens.
> 3. Calculate `max_num_seqs` based on cacheable tokens. Example: avg input=800 tokens, output=500 tokens, blocks=640 → `kv_cache_ratio = 800/(800+500)=0.6`, `max_seq_len = 640*64/(800+500)=31`.

### 2.2 fastdeploy.LLM.generate

* prompts(str,list[str],list[int]): Input prompts (batch supported), accepts decoded token ids
* sampling_params: See 2.4 for parameter details
* use_tqdm: Enable progress visualization

### 2.3 fastdeploy.LLM.chat

* messages(list[dict],list[list[dict]]): Input messages (batch supported)
* sampling_params: See 2.4 for parameter details
* use_tqdm: Enable progress visualization
* chat_template_kwargs(dict): Extra template parameters (currently supports enable_thinking(bool))

### 2.4 fastdeploy.SamplingParams

* presence_penalty(float): Penalizes repeated topics (positive values reduce repetition)
* frequency_penalty(float): Strict penalty for repeated tokens
* repetition_penalty(float): Direct penalty for repeated tokens (>1 penalizes, <1 encourages)
* temperature(float): Controls randomness (higher = more random)
* top_p(float): Probability threshold for token selection
* max_tokens(int): Maximum generated tokens (input + output)
* min_tokens(int): Minimum forced generation length

### 2.5 fastdeploy.engine.request.RequestOutput

* request_id(str): Request identifier
* prompt(str): Input content
* prompt_token_ids(list[int]): Tokenized input
* outputs(fastdeploy.engine.request.CompletionOutput): Results
* finished(bool): Completion status
* metrics(fastdeploy.engine.request.RequestMetrics): Performance metrics
* num_cached_tokens(int): Cached token count (only valid when enable_prefix_caching``` is enabled)
* error_code(int): Error code
* error_msg(str): Error message

### 2.6 fastdeploy.engine.request.CompletionOutput

* index(int): Batch index
* send_idx(int): Request token index
* token_ids(list[int]): Output tokens
* text(str): Decoded text
* reasoning_content(str): (X1 model only) Chain-of-thought output

### 2.7 fastdeploy.engine.request.RequestMetrics

* arrival_time(float): Request receipt time
* inference_start_time(float): Inference start time
* first_token_time(float): First token latency
* time_in_queue(float): Queuing time
* model_forward_time(float): Forward pass duration
* model_execute_time(float): Total execution time (including preprocessing)