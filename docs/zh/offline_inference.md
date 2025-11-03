[English](../offline_inference.md)

# 离线推理

## 1. 使用方式

通过FastDeploy离线推理，可支持本地加载模型，并处理用户数据，使用方式如下，

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
llm = LLM(model="baidu/ERNIE-4.5-0.3B-Paddle", tensor_parallel_size=1, max_model_len=8192)
# 批量进行推理（llm内部基于资源情况进行请求排队、动态插入处理）
outputs = llm.chat(messages, sampling_params)

# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
```

上述示例中 ``LLM``配置方式， `SamplingParams` ，`LLM.generate` ，`LLM.chat`以及输出output对应的结构体 `RequestOutput` 接口说明见如下文档说明。

> 注： 若为思考模型, 加载模型时需要指定 `reasoning_parser` 参数，并在请求时, 可以通过配置 `chat_template_kwargs` 中 `enable_thinking`参数, 进行开关思考。

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

### 续写接口(LLM.generate)

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

> 注： 续写接口, 适应于用户自定义好上下文输入, 并希望模型仅输出续写内容的场景; 推理过程不会增加其他 `prompt`拼接。
> 对于 `chat`模型, 建议使用对话接口(LLM.chat)。

对于多模模型, 例如 `baidu/ERNIE-4.5-VL-28B-A3B-Paddle`, 在调用 `generate接口`时, 需要提供包含图片的prompt, 使用方式如下:

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

> 注： `generate` 接口, 暂时不支持思考开关参数控制, 均使用模型默认思考能力。

## 2. 接口说明

### 2.1 fastdeploy.LLM

支持配置参数参考 [FastDeploy参数说明](./parameters.md)

> 参数配置说明：
>
> 1. 离线推理不需要配置 `port` 和 `metrics_port` 参数。
> 2. 模型服务启动后，会在日志文件log/fastdeploy.log中打印如 `Doing profile, the total_block_num:640` 的日志，其中640即表示自动计算得到的KV Cache block数量，将它乘以block_size(默认值64)，即可得到部署后总共可以在KV Cache中缓存的Token数。
> 3. `max_num_seqs` 用于配置decode阶段最大并发处理请求数，该参数可以基于第1点中缓存的Token数来计算一个较优值，例如线上统计输入平均token数800, 输出平均token数500，本次计>算得到KV Cache block为640， block_size为64。那么我们可以配置 `kv_cache_ratio = 800 / (800 + 500) = 0.6` , 配置 `max_seq_len = 640 * 64 / (800 + 500) = 31`。

### 2.2 fastdeploy.LLM.chat

* messages(list[dict],list[list[dict]]): 输入的message, 支持batch message 输入
* sampling_params: 模型超参设置具体说明见2.4
* use_tqdm: 是否打开推理进度可视化
* chat_template_kwargs(dict): 传递给对话模板的额外参数，当前支持enable_thinking(bool)
  *使用示例 `chat_template_kwargs={"enable_thinking": False}`*

### 2.3 fastdeploy.LLM.generate

* prompts(str, list[str], list[int], list[list[int]], dict[str, Any], list[dict[str, Any]]): 输入的prompt, 支持batch prompt 输入，解码后的token ids 进行输入
  *dict 类型使用示例 `prompts={"prompt": prompt, "multimodal_data": {"image": images}}`*
* sampling_params: 模型超参设置具体说明见2.4
* use_tqdm: 是否打开推理进度可视化

### 2.4 fastdeploy.SamplingParams

* presence_penalty(float): 控制模型生成重复内容的惩罚系数，正值降低重复话题出现的概率
* frequency_penalty(float): 控制重复token的惩罚力度，比presence_penalty更严格，会惩罚高频重复
* repetition_penalty(float): 直接对重复生成的token进行惩罚的系数（>1时惩罚重复，<1时鼓励重复）
* temperature(float): 控制生成随机性的参数，值越高结果越随机，值越低结果越确定
* top_p(float): 概率累积分布截断阈值，仅考虑累计概率达到此阈值的最可能token集合
* top_k(int): 采样概率最高的token数量，考虑概率最高的k个token进行采样
* min_p(float): token入选的最小概率阈值(相对于最高概率token的比值，设为>0可通过过滤低概率token来提升文本生成质量)
* max_tokens(int): 限制模型生成的最大token数量（包括输入和输出）
* min_tokens(int): 强制模型生成的最少token数量，避免过早结束
* bad_words(list[str]): 禁止生成的词列表, 防止模型生成不希望出现的词

### 2.5 fastdeploy.engine.request.RequestOutput

* request_id(str): 标识request 的id
* prompt(str)：输入请求的request内容
* prompt_token_ids(list[int]): 拼接后经过词典解码的输入的token 列表
* outputs(fastdeploy.engine.request.CompletionOutput): 输出结果
* finished(bool)：标识当前query 是否推理结束
* metrics(fastdeploy.engine.request.RequestMetrics)：记录推理耗时指标
* num_cached_tokens(int): 缓存的token数量, 仅在开启 ``enable_prefix_caching``时有效
* num_input_image_tokens(int): 输入图片token的数量
* num_input_video_tokens(int): 输入视频token的数量
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
