[简体中文](zh/offline_inference.md)

# Offline Inference

## 1. Usage

FastDeploy supports offline inference by loading models locally and processing user data. Usage examples:

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

> Note: For reasoning models, when loading the model, you need to specify the reasoning_parser parameter. Additionally, during the request, you can toggle the reasoning feature on or off by configuring the `enable_thinking` parameter within `chat_template_kwargs`.

```python
from fastdeploy.entrypoints.llm import LLM
# 加载模型
llm = LLM(model="baidu/ERNIE-4.5-VL-28B-A3B-Paddle", tensor_parallel_size=1, max_model_len=32768, limit_mm_per_prompt={"image": 100}, reasoning_parser="ernie-45-vl")

outputs = llm.chat(
    messages=[
        {"role": "user", "content": [ {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
                                     {"type": "text", "text": "图中的文物属于哪个年代"}]}
    ],
    chat_template_kwargs={"enable_thinking": False})

# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    reasoning_text = output.outputs.reasoning_content
```

### Text Completion Interface (LLM.generate)

```python
from fastdeploy import LLM, SamplingParams

prompts = [
    "User: 帮我写一篇关于深圳文心公园的500字游记和赏析。\nAssistant: 好的。"
]

# 采样参数
sampling_params = SamplingParams(top_p=0.95, max_tokens=6400)

# 加载模型
llm = LLM(model="baidu/ERNIE-4.5-21B-A3B-Base-Paddle", tensor_parallel_size=1, max_model_len=8192)

# 批量进行推理（llm内部基于资源情况进行请求排队、动态插入处理）
outputs = llm.generate(prompts, sampling_params)

# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
```

> Note: Text completion interface, suitable for scenarios where users have predefined the context input and expect the model to output only the continuation content. No additional `prompt` concatenation will be added during the inference process.
> For the `chat` model, it is recommended to use the Chat Interface (`LLM.chat`).

For multimodal models, such as `baidu/ERNIE-4.5-VL-28B-A3B-Paddle`, when calling the `generate interface`, you need to provide a prompt that includes images. The usage is as follows:

```python
import io
import requests
from PIL import Image

from fastdeploy.entrypoints.llm import LLM
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer

PATH = "baidu/ERNIE-4.5-VL-28B-A3B-Paddle"
tokenizer = Ernie4_5Tokenizer.from_pretrained(PATH)

messages = [
    {
        "role": "user",
        "content": [
            {"type":"image_url", "image_url": {"url":"https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
            {"type":"text", "text":"图中的文物属于哪个年代"}
        ]
     }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = [], []
for message in messages:
    content = message["content"]
    if not isinstance(content, list):
        continue
    for part in content:
        if part["type"] == "image_url":
            url = part["image_url"]["url"]
            image_bytes = requests.get(url).content
            img = Image.open(io.BytesIO(image_bytes))
            images.append(img)
        elif part["type"] == "video_url":
            url = part["video_url"]["url"]
            video_bytes = requests.get(url).content
            videos.append({
                "video": video_bytes,
                "max_frames": 30
            })

sampling_params = SamplingParams(temperature=0.1, max_tokens=6400)
llm = LLM(model=PATH, tensor_parallel_size=1, max_model_len=32768, limit_mm_per_prompt={"image": 100}, reasoning_parser="ernie-45-vl")
outputs = llm.generate(prompts={
    "prompt": prompt,
    "multimodal_data": {
        "image": images,
        "video": videos
    }
}, sampling_params=sampling_params)

# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    reasoning_text = output.outputs.reasoning_content

```

> Note: The `generate interface` does not currently support passing parameters to control the thinking function (on/off). It always uses the model's default parameters.

## 2. API Documentation

### 2.1 fastdeploy.LLM

For ``LLM`` configuration, refer to [Parameter Documentation](parameters.md).

> Configuration Notes:
>
> 1. `port` and `metrics_port` is only used for online inference.
> 2. After startup, the service logs KV Cache block count (e.g. `total_block_num:640`). Multiply this by block_size (default 64) to get total cacheable tokens.
> 3. Calculate `max_num_seqs` based on cacheable tokens. Example: avg input=800 tokens, output=500 tokens, blocks=640 → `kv_cache_ratio = 800/(800+500)=0.6`, `max_seq_len = 640*64/(800+500)=31`.

### 2.2 fastdeploy.LLM.chat

* messages(list[dict],list[list[dict]]): Input messages (batch supported)
* sampling_params: See 2.4 for parameter details
* use_tqdm: Enable progress visualization
* chat_template_kwargs(dict): Extra template parameters (currently supports enable_thinking(bool))
  *usage example: `chat_template_kwargs={"enable_thinking": False}`*

### 2.3 fastdeploy.LLM.generate

* prompts(str, list[str], list[int], list[list[int]], dict[str, Any], list[dict[str, Any]]): : Input prompts (batch supported), accepts decoded token ids
  *example of using a dict-type parameter: `prompts={"prompt": prompt, "multimodal_data": {"image": images}}`*
* sampling_params: See 2.4 for parameter details
* use_tqdm: Enable progress visualization

### 2.4 fastdeploy.SamplingParams

* presence_penalty(float): Penalizes repeated topics (positive values reduce repetition)
* frequency_penalty(float): Strict penalty for repeated tokens
* repetition_penalty(float): Direct penalty for repeated tokens (>1 penalizes, <1 encourages)
* temperature(float): Controls randomness (higher = more random)
* top_p(float): Probability threshold for token selection
* top_k(int): Number of tokens considered for sampling
* min_p(float): Minimum probability relative to the maximum probability for a token to be considered (>0 filters low-probability tokens to improve quality)
* max_tokens(int): Maximum generated tokens (input + output)
* min_tokens(int): Minimum forced generation length
* bad_words(list[str]): Prohibited words

### 2.5 fastdeploy.engine.request.RequestOutput

* request_id(str): Request identifier
* prompt(str): Input content
* prompt_token_ids(list[int]): Tokenized input
* outputs(fastdeploy.engine.request.CompletionOutput): Results
* finished(bool): Completion status
* metrics(fastdeploy.engine.request.RequestMetrics): Performance metrics
* num_cached_tokens(int): Cached token count (only valid when enable_prefix_caching``` is enabled)
* num_input_image_tokens(int): Number of input image tokens.
* num_input_video_tokens(int): Number of input video tokens.
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
