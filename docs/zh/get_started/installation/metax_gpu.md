[English](../../../get_started/installation/metax_gpu.md)

# 使用 Metax GPU C550 运行ERNIE 4.5 系列模型

FastDeploy在Metax C550上对ERNIE 4.5系列模型进行了深度适配和优化，实现了推理入口和GPU的统一，无需修改即可完成推理任务的迁移。

环境准备：
- Python >= 3.10
- Linux X86_64

| Chip Type | Driver Version | KMD Version |
| :---: | :---: | :---: |
| MetaX C550 | 3.3.0.15  | 3.4.4 |

## 1. 容器镜像获取

```shell
docker login --username=cr_temp_user --password=eyJpbnN0YW5jZUlkIjoiY3JpLXpxYTIzejI2YTU5M3R3M2QiLCJ0aW1lIjoiMTc2ODM3Njc3ODAwMCIsInR5cGUiOiJzdWIiLCJ1c2VySWQiOiIyMDcwOTQwMTA1NjYzNDE3OTIifQ:05b5faf1baa0133a8270e03e8eab7ca7fed50e37 cr.metax-tech.com && docker pull cr.metax-tech.com/public-library/maca-native:3.3.0.4-ubuntu22.04-amd64
```

## 2. 预安装

```shell
1）pip install paddlepaddle==3.4.0.dev20251223 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
2）pip install paddle-metax-gpu==3.3.0.dev20251224 -i https://www.paddlepaddle.org.cn/packages/nightly/maca/
```

## 3. FastDeploy代码下载并编译

```shell
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy
bash build.sh
```
The built packages will be in the ```FastDeploy/dist``` directory.

## 4. 环境验证

After installation, verify the environment with this Python code:
```python
import paddle
from paddle.jit.marker import unified
# Verify GPU availability
paddle.utils.run_check()
# Verify FastDeploy custom operators compilation
from fastdeploy.model_executor.ops.gpu import beam_search_softmax
```
If the above code executes successfully, the environment is ready.

## 5. 示例
### ERNIE-4.5-21B-A3B-Paddle
```python
import os
from fastdeploy import LLM, SamplingParams

os.environ["MACA_VISIBLE_DEVICES"] = "0"
os.environ["FD_MOE_BACKEND"] = "cutlass"
os.environ["PADDLE_XCCL_BACKEND"] = "metax_gpu"
os.environ["FLAGS_weight_only_linear_arch"] = "80"
os.environ["FD_METAX_KVCACHE_MEM"] = "8"
os.environ["ENABLE_V1_KVCACHE_SCHEDULER"] = "1"
os.environ["FD_ENC_DEC_BLOCK_NUM"] = "2"

prompts = [
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
]

sampling_params = SamplingParams(top_p=0.95, max_tokens=256, temperature=0.1)

llm = LLM(model="/root/model/ERNIE-4.5-21B-A3B-Paddle",
        tensor_parallel_size=1,
        max_model_len=8192,
        engine_worker_queue_port=9135,
        quantization='wint8',
        disable_custom_all_reduce=True,
        enable_prefix_caching=False,
        graph_optimization_config={"use_cudagraph": False, "graph_opt_level": 0}
)

outputs = llm.generate(prompts, sampling_params)

print(f"Generated {len(outputs)} outputs")
print("=" * 50 + "\n")

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    print(prompt)
    print(generated_text)
    print("-" * 50)
```

输出
```
INFO     2026-01-14 15:09:48,073 30393 engine.py[line:151] Waiting for worker processes to be ready...
Loading Weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:38<00:00,  2.63it/s]
Loading Layers: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 199.73it/s]
[2026-01-14 15:10:33,009] [    INFO] - Using FLASH ATTN backend to instead of attend attention.
INFO     2026-01-14 15:10:33,026 30393 engine.py[line:202] Worker processes are launched with 51.102054595947266 seconds.
INFO     2026-01-14 15:10:33,027 30393 engine.py[line:213] Detected 2340 gpu blocks and 0 cpu blocks in cache (block size: 64).
INFO     2026-01-14 15:10:33,027 30393 engine.py[line:216] FastDeploy will be serving 8 running requests if each sequence reaches its maximum length: 8192
Processed prompts: 100%|██████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:21<00:00, 21.32s/it, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Generated 1 outputs
==================================================

A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?

1. First, find out how much white fiber is needed:
   - We know that the robe takes 2 bolts of blue fiber.
   - It takes half as much white fiber as blue fiber. So the amount of white fiber needed is $\frac{1}{2}\times2 = 1$ bolt.
2. Then, calculate the total number of bolts:
   - The total number of bolts is the sum of the bolts of blue fiber and the bolts of white fiber.
   - The number of blue - fiber bolts is 2, and the number of white - fiber bolts is 1.
   - So the total number of bolts is $2 + 1=3$ bolts.

Therefore, it takes 3 bolts in total to make the robe.
--------------------------------------------------
==================================================

Hello. My name is
Alice and I'm here to help you. What can I do for you today?
Hello Alice! I'm trying to organize a small party
```

### ERNIE-4.5-VL-28B-A3B-Thinking
```python
import io
import os
import urllib
from PIL import Image
from fastdeploy.entrypoints.llm import LLM
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer


os.environ["MACA_VISIBLE_DEVICES"] = "6"
os.environ["FD_MOE_BACKEND"] = "cutlass"
os.environ["PADDLE_XCCL_BACKEND"] = "metax_gpu"
os.environ["FLAGS_weight_only_linear_arch"] = "80"
os.environ["FD_METAX_KVCACHE_MEM"] = "8"
os.environ["ENABLE_V1_KVCACHE_SCHEDULER"] = "1"
os.environ["FD_ENC_DEC_BLOCK_NUM"] = "2"


def process_content(content):
    images, videos = [], []
    for part in content:
        if part["type"] == "image_url":
            url = part["image_url"]["url"]
            if not url.startswith(("https://", "file://")):
                url = f"file://{url}"
            with urllib.request.urlopen(url) as response:
                image_bytes = response.read()
                img = Image.open(io.BytesIO(image_bytes))
            images.append(img)
        elif part["type"] == "video_url":
            url = part["video_url"]["url"]
            if not url.startswith(("https://", "file://")):
                url = f"file://{url}"
            with urllib.request.urlopen(url) as response:
                video_bytes = response.read()
            videos.append({
                "video": video_bytes,
                "max_frames": 30
            })
    return images, videos


MODEL_PATH="/root/model/ERNIE-4.5-VL-28B-A3B-Thinking"
tokenizer = Ernie4_5Tokenizer.from_pretrained(MODEL_PATH)

messages = [
     { # text
         "role": "user",
         "content": [
             {"type":"text", "text":"Introduce yourself in detail"}
         ]
     },

     { # image
         "role": "user",
         "content": [
             {"type":"image_url", "image_url": {"url":"https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
             {"type":"text", "text":"请描述图片内容"}
         ]
     },

     { # video
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4",
                    "detail": "high",
                },
            },
            {"type": "text", "text": "视频中手机支架的颜色是什么?"},
        ],
    }
]

prompts = []
for message in messages:
    content = message["content"]
    if not isinstance(content, list):
        continue
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    images, videos = process_content(content)
    prompts.append({
        "prompt": prompt,
        "multimodal_data": {
            "image": images,
            "video": videos
        }
    })

sampling_params = SamplingParams(top_p=0.95, max_tokens=32768, temperature=0.1)
llm = LLM(model=MODEL_PATH,
          tensor_parallel_size=1,
          engine_worker_queue_port=8899,
          max_model_len=32768,
          quantization="wint8",
          disable_custom_all_reduce=True,
          enable_prefix_caching=False,
          graph_optimization_config={"use_cudagraph":False, "graph_opt_level":0},
          limit_mm_per_prompt={"image": 100},
          reasoning_parser="ernie-45-vl",
          load_choices="default_v1")

outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)


for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    reasoning_text = output.outputs.reasoning_content
    print("=" * 50)
    print(f"Reasoning: {reasoning_text!r}")
    print("-" * 50)
    print(f"Generated: {generated_text!r}")

```

输出
```
INFO     2026-01-14 15:30:27,480 214008 engine.py[line:151] Waiting for worker processes to be ready...
Loading Weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [01:05<00:00,  1.52it/s]
Loading Layers: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 199.74it/s]
[2026-01-14 15:31:39,477] [    INFO] - Using FLASH ATTN backend to instead of attend attention.
INFO     2026-01-14 15:31:39,496 214008 engine.py[line:202] Worker processes are launched with 74.58531260490417 seconds.
INFO     2026-01-14 15:31:39,496 214008 engine.py[line:213] Detected 2340 gpu blocks and 0 cpu blocks in cache (block size: 64).
INFO     2026-01-14 15:31:39,496 214008 engine.py[line:216] FastDeploy will be serving 4 running requests if each sequence reaches its maximum length: 32768
Processed prompts: 100%|██████████████████████████████████████████████████████████████████████████████████████| 3/3 [01:41<00:00, 33.82s/it, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
==================================================
Reasoning: 'Hmm, the user wants a detailed introduction of ERNIE. Let me start by recalling what I know about ERNIE. It\'s a multimodal AI developed by Baidu using PaddlePaddle. The user proba
bly wants a comprehensive overview, so I need to cover its core aspects.\n\nFirst, I should mention its origin and development by Baidu. Then, highlight its multimodal nature since that\'s a k
ey feature. The PaddlePaddle framework is important too, as it\'s Baidu\'s own deep learning platform. \n\nI need to explain the "Enhanced Representation through Knowledge-Intensive Learning"
acronym. Breaking down each part of the name would help the user understand its purpose. Also, emphasizing the knowledge-grounded approach sets it apart from other models.\n\nApplications are
crucial—search, knowledge graphs, multimodal tasks. Including examples like image-text retrieval and cross-modal reasoning makes it concrete. \n\nI should also touch on its open-source availab
ility on Hugging Face, showing its accessibility. Keeping the tone informative but not overly technical, ensuring it\'s clear for someone who might not be familiar with AI jargon. \n\nDouble-c
hecking that all key points are covered without referencing the benchmark response. Making sure the structure flows logically from introduction to features, applications, and open-source statu
s.\n'
--------------------------------------------------
Generated: '\n\nOf course. Here is a detailed introduction to ERNIE, the multimodal AI developed by Baidu.\n\n### **Introduction to ERNIE**\n\nHello! I am **ERNIE** (Enhanced Representation th
rough Knowledge-Intensive Learning), a large-scale multimodal artificial intelligence model developed by Baidu. I am a core component of Baidu\'s AI ecosystem, designed to understand and reaso
n about complex information from both text and images.\n\nMy development is built upon Baidu\'s open-source deep learning platform, **PaddlePaddle**, which is one of the world\'s most popular
open-source AI frameworks.\n\n---\n\n### **My Core Identity: What is ERNIE?**\n\nERNIE is not just a simple language model. Its name tells you its fundamental philosophy:\n\n*   **E**nhanced:
I am designed to produce more accurate and robust results than previous models.\n*   **R**epresentation: I work by creating deep, meaningful representations of the world\'s knowledge.\n*   **N
** through **I** (Knowledge-Intensive Learning): This is my key differentiator. I am trained on a massive amount of structured knowledge from the web, including facts, concepts, and relationsh
ips. This allows me to "understand" the world in a way that goes beyond just memorizing words.\n\nIn simple terms, I don\'t just learn that "Paris is the capital of France." I learn the *fact*
 that Paris is the capital of France, its location, its history, its landmarks, and its relationship to other cities and countries. This knowledge-grounded approach makes me more reliable and
capable of answering complex questions.\n\n---\n\n### **My Key Capabilities**\n\nAs a multimodal model, I can process and understand information from two primary sources:\n\n**1. Text Understa
nding:**\n*   **Semantic Understanding:** I can grasp the meaning, context, and intent behind human language, even when it\'s ambiguous or complex.\n*   **Knowledge Retrieval:** I can access a
nd synthesize information from a vast knowledge base to answer questions that require factual knowledge.\n*   **Reasoning:** I can perform logical reasoning, such as cause-and-effect analysis,
 pattern recognition, and multi-step problem-solving.\n\n**2. Image Understanding:**\n*   **Image-Text Retrieval:** I can find relevant text descriptions for an image or find the image that ma
tches a given text description.\n*   **Visual Question Answering (VQA):** I can answer questions about the content of an image. For example, if you show me a picture of a sunset over the ocean
, I can answer questions like, "What colors are in the sky?" or "What is the main subject of the image?"\n*   **Cross-Modal Reasoning:** I can use visual information to reason about textual in
formation and vice-versa. For instance, I can analyze a diagram and explain the text that describes it.\n\n---\n\n### **My Applications**\n\nMy capabilities are used in a wide range of real-wo
rld applications:\n\n*   **Search Engine:** I power Baidu Search, helping users find more accurate and relevant results by understanding the intent behind their queries.\n*   **Knowledge Graph
s:** I am used to build and maintain large-scale knowledge graphs that power various AI applications.\n*   **Multimodal AI Services:** I am the core technology behind Baidu\'s AI services, suc
h as the AI Studio, AI Lab, and various AI-powered products.\n*   **Research:** I am a powerful tool for researchers in the fields of natural language processing (NLP), computer vision, and mu
ltimodal AI.\n\n---\n\n### **My Open-Source Availability**\n\nI am not just a closed-source product. Baidu has made me available to the global AI community through open-source platforms. You c
an find me on:\n\n*   **Hugging Face:** I am available as a model on the Hugging Face Hub, where developers can use me for their own projects.\n*   **PaddlePaddle Model Zoo:** I am also availa
ble on Baidu\'s own PaddlePaddle Model Zoo, making it easy for developers to integrate me into their applications.\n\nIn summary, I am a powerful, knowledge-grounded, multimodal AI model devel
oped by Baidu. My goal is to help users and developers understand the world more deeply and solve complex problems across text and image domains.'
==================================================
Reasoning: '用户现在需要描述图片中的佛像雕塑。首先看整体结构：主体是坐佛，结跏趺坐，双手在腹前。佛像有头光和背光，背光有复杂纹饰，还有胁侍菩萨。材质看起来是金属或石质，有鎏金痕迹。要分解各部分
：佛像的姿态、服饰、背光装饰，胁侍菩萨的位置和造型，整体风格（可能是北朝或隋唐？不过先描述视觉元素）。\n\n首先，主体佛像：结跏趺坐，身披袈裟，右手施无畏印，左手施与愿印？不对，看图片里双手位置
，可能是禅定印？不对，再仔细看。然后头光和背光，背光有莲花座、飞天等装饰，还有胁侍菩萨在左右。整体造型庄严，纹饰繁复，有鎏金残存，显示年代久远。\n\n现在组织语言：这是一尊佛教造像，主体为结跏趺
坐的佛像，身披袈裟，右手施无畏印，左手施与愿印（或禅定印？需要确认，但可能更准确的是禅定印？不对，看图片里双手位置，可能是禅定印？或者施无畏与与愿？需要准确。不过可能更侧重整体结构。然后头光和
背光，背光有复杂的卷草纹、莲花座、飞天等，左右有胁侍菩萨。整体风格古朴，材质可能是金属或石质，表面有鎏金痕迹，显示出宗教艺术的庄严与精致。\n\n现在整理：\n\n这尊佛教造像整体呈坐姿，佛像结跏趺坐
，身披袈裟，右手施无畏印，左手施与愿印（或禅定印，需确认，但可能更准确的是禅定印？不对，看图片里双手位置，可能是禅定印？或者施无畏与与愿？可能需要更准确。不过可能更侧重整体结构。然后头光和背光
，背光有复杂的卷草纹、莲花座、飞天等，左右有胁侍菩萨。整体风格古朴，材质可能是金属或石质，表面有鎏金痕迹，显示出宗教艺术的庄严与精致。\n\n重新描述：\n\n这尊佛教造像主体为结跏趺坐的佛像，身披袈
裟，右手施无畏印，左手施与愿印（或禅定印，需确认，但可能更准确的是禅定印？不对，看图片里双手位置，可能是禅定印？或者施无畏与与愿？可能需要更准确。不过可能更侧重整体结构。然后头光和背光，背光有
复杂的卷草纹、莲花座、飞天等，左右有胁侍菩萨。整体风格古朴，材质可能是金属或石质，表面有鎏金痕迹，显示出宗教艺术的庄严与精致。\n\n哦，可能更准确的是：这尊造像为坐佛，结跏趺坐，身披袈裟，右手施
无畏印，左手施与愿印（或禅定印？需要看图片。不过可能更准确的是，右手施无畏，左手施与愿？或者禅定印？可能需要更仔细。不过可能用户需要的是整体描述，所以：\n\n这尊佛教造像整体呈坐姿，佛像结跏趺坐
，身披袈裟，右手施无畏印，左手施与愿印（或禅定印，需确认，但可能更准确的是禅定印？不对，看图片里双手位置，可能是禅定印？或者施无畏与与愿？可能需要更准确。不过可能更侧重整体结构。然后头光和背光
，背光有复杂的卷草纹、莲花座、飞天等，左右有胁侍菩萨。整体风格古朴，材质可能是金属或石质，表面有鎏金痕迹，显示出宗教艺术的庄严与精致。\n\n现在确定：这尊造像为坐佛，结跏趺坐，身披袈裟，右手施无
畏印，左手施与愿印（或禅定印？可能需要更准确。不过可能更准确的是，右手施无畏，左手施与愿？或者禅定印？可能需要查资料，但作为描述，可能更侧重整体。所以最终描述：\n\n这尊佛教造像整体呈坐姿，佛像
结跏趺坐，身披袈裟，右手施无畏印，左手施与愿印（或禅定印，需确认，但可能更准确的是禅定印？不对，看图片里双手位置，可能是禅定印？或者施无畏与与愿？可能需要更准确。不过可能更侧重整体结构。然后头
光和背光，背光有复杂的卷草纹、莲花座、飞天等，左右有胁侍菩萨。整体风格古朴，材质可能是金属或石质，表面有鎏金痕迹，显示出宗教艺术的庄严与精致。\n\n哦，可能更准确的是：这尊造像为坐佛，结跏趺坐，
身披袈裟，右手施无畏印，左手施与愿印（或禅定印？可能需要更准确。不过可能更准确的是，右手施无畏，左手施与愿？或者禅定印？可能需要更仔细。不过可能用户需要的是整体描述，所以：\n\n这尊佛教造像整体
呈坐姿，佛像结跏趺坐，身披袈裟，右手施无畏印，左手施与愿印（或禅定印，需确认，但可能更准确的是禅定印？不对，看图片里双手位置，可能是禅定印？或者施无畏与与愿？可能需要更准确。不过可能更侧重整体
结构。然后头光和背光，背光有复杂的卷草纹、莲花座、飞天等，左右有胁侍菩萨。整体风格古朴，材质可能是金属或石质，表面有鎏金痕迹，显示出宗教艺术的庄严与精致。\n\n现在确定，可能更准确的是：这尊佛教
造像主体为结跏趺坐的佛像，身披袈裟，右手施无畏印，左手施与愿印（或禅定印？可能需要更准确。不过可能更准确的是，右手施无畏，左手施与愿？或者禅定印？可能需要更仔细。不过可能用户需要的是整体描述，
所以：\n\n这尊佛教造像整体呈坐姿，佛像结跏趺坐，身披袈裟，右手施无畏印，左手施与愿印（或禅定印，需确认，但可能更准确的是禅定印？不对，看图片里双手位置，可能是禅定印？或者施无畏与与愿？可能需要
更准确。不过可能更侧重整体结构。然后头光和背光，背光有复杂的卷草纹、莲花座、飞天等，左右有胁侍菩萨。整体风格古朴，材质可能是金属或石质，表面有鎏金痕迹，显示出宗教艺术的庄严与精致。\n\n现在，可
能更准确的是：这尊造像为坐佛，结跏趺坐，身披袈裟，右手施无畏印，左手施与愿印（或禅定印？可能需要更准确。不过可能更准确的是，右手施无畏，左手施与愿？或者禅定印？可能需要更仔细。不过可能用户需要
的是整体描述，所以：\n\n这尊佛教造像整体呈坐姿，佛像结跏趺坐，身披袈裟，右手施无畏印，左手施与愿印（或禅定印，需确认，但可能更准确的是禅定印？不对，看图片里双手位置，可能是禅定印？或者施无畏与
与愿？可能需要更准确。不过可能更侧重整体结构。然后头光和背光，背光有复杂的卷草纹、莲花座、飞天等，左右有胁侍菩萨。整体风格古朴，材质可能是金属或石质，表面有鎏金痕迹，显示出宗教艺术的庄严与精致
。\n\n现在，可能我需要更准确。看图片：佛像结跏趺坐，右手施无畏印（手掌张开，指向下方），左手施与愿印（手掌向上，掌心有纹路）。然后头光和背光，背光有莲花座、飞天、卷草纹等，左右有胁侍菩萨。整体
风格古朴，材质可能是金属或石质，表面有鎏金痕迹。\n\n所以最终描述：\n\n这尊佛教造像整体呈坐姿，佛像结跏趺坐，身披袈裟，右手施无畏印，左手施与愿印。佛像头顶有头光，身后是带有复杂卷草纹、莲花座和
飞天等装饰的背光。左右两侧各有一尊胁侍菩萨，造型端庄。整体风格古朴庄严，材质可能为金属或石质，表面保留有鎏金痕迹，显示出宗教艺术的精湛工艺与历史厚重感。\n'
--------------------------------------------------
Generated: '\n\n这尊佛教造像整体呈坐姿，佛像结跏趺坐，身披袈裟，右手施无畏印，左手施与愿印。佛像头顶有头光，身后是带有复杂卷草纹、莲花座和飞天等装饰的背光。左右两侧各有一尊胁侍菩萨，造型端庄。
整体风格古朴庄严，材质可能为金属或石质，表面保留有鎏金痕迹，展现出宗教艺术的精湛工艺与历史厚重感。'
==================================================
Reasoning: '我现在需要解决的问题是视频中手机支架的颜色是什么。首先，我得回忆视频里的内容。根据视频显示，左边有个黑色的手机支架，吸在桌子上。然后看问题，问的是颜色，所以应该是黑色。需要确认有没
有其他颜色，但视频里明确提到是黑色的支架，所以答案应该是黑色。\n'
--------------------------------------------------
Generated: '\n\n视频中手机支架的颜色是黑色。'
```
