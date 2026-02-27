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
docker login --username=cr_temp_user --password=eyJpbnN0YW5jZUlkIjoiY3JpLXpxYTIzejI2YTU5M3R3M2QiLCJ0aW1lIjoiMTc3MjA4Mjg4MzAwMCIsInR5cGUiOiJzdWIiLCJ1c2VySWQiOiIyMDcwOTQwMTA1NjYzNDE3OTIifQ:af1dd00652cd43b2bca08a03b3df03c9cffd4c5e cr.metax-tech.com && docker pull cr.metax-tech.com/public-ai-release/maca/paddle-metax:3.3.0-maca.ai3.3.0.10-py310-ubuntu22.04-amd64
```

## 2. 预安装

```shell
1）pip install  --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
2）pip install --pre paddle-metax-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/maca/


// PaddleOCR need
3）pip install -U "paddleocr[doc-parser]"
4）pip install opencv-contrib-python-headless==4.10.0.84
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

### PaddleOCR
#### 服务端
```bash
export PADDLE_XCCL_BACKEND=metax_gpu
export FLAGS_weight_only_linear_arch=80
export FD_MOE_BACKEND=cutlass
export FD_METAX_KVCACHE_MEM=8
export ENABLE_V1_KVCACHE_SCHEDULER=1
export FD_ENC_DEC_BLOCK_NUM=2
export FD_SAMPLING_CLASS="rejection"

paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --backend fastdeploy --port 8118 --model_dir ${YOUR_MODEL_PATH}/PaddleOCR-VL-1.5/
```

#### 客户端
```bash
paddleocr doc_parser \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png \
        --vl_rec_backend fastdeploy-server \
        --vl_rec_server_url http://localhost:8118/v1 \
        --layout_detection_model_dir ${YOUR_MODEL_PATH}/PP-DocLayoutV3/ \
        --vl_rec_model_dir ${YOUR_MODEL_PATH}/PaddleOCR-VL-1.5/
```

#### 终端输出
```bash
[33mChecking connectivity to the model hosters, this may take a while. To bypass this check, set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` to `True`.[0m
[33mNo model hoster is available! Please check your network connection to one of the following model hoster: HuggingFace (https://huggingface.co), ModelScope (https://modelscope.cn), AIStudio (https://aistudio.baidu.com), or BOS (https://paddle-model-ecology.bj.bcebos.com). Otherwise, only local models can be used.[0m
I0212 09:14:05.937799 163217 init.cc:254] ENV [CUSTOM_DEVICE_ROOT]=/opt/conda/lib/python3.10/site-packages/paddle_custom_device
I0212 09:14:05.937847 163217 init.cc:162] Try loading custom device libs from: [/opt/conda/lib/python3.10/site-packages/paddle_custom_device]
I0212 09:14:07.750711 163217 custom_device_load.cc:51] Succeed in loading custom runtime in lib: /opt/conda/lib/python3.10/site-packages/paddle_custom_device/libpaddle-metax-gpu.so
I0212 09:14:07.750746 163217 custom_device_load.cc:58] Skipped lib [/opt/conda/lib/python3.10/site-packages/paddle_custom_device/libpaddle-metax-gpu.so]: no custom engine Plugin symbol in this lib.
I0212 09:14:07.753039 163217 custom_kernel.cc:68] Succeed in loading 973 custom kernel(s) from loaded lib(s), will be used like native ones.
I0212 09:14:07.753350 163217 init.cc:174] Finished in LoadCustomDevice with libs_path: [/opt/conda/lib/python3.10/site-packages/paddle_custom_device]
I0212 09:14:07.753376 163217 init.cc:260] CustomDevice: metax_gpu, visible devices count: 1
[32mCreating model: ('PP-DocLayoutV3', '/mingkunzhang/workspace/models/modelscope.hub.metax-tech.com/models/PaddlePaddle/PP-DocLayoutV3/')[0m
[32mCreating model: ('PaddleOCR-VL-1.5-0.9B', '/mingkunzhang/workspace/models/modelscope.hub.metax-tech.com/models/PaddlePaddle/PaddleOCR-VL-1.5/')[0m
/opt/conda/lib/python3.10/site-packages/paddlex/inference/models/base/predictor/base_predictor.py:131: UserWarning: `model_dir` will be ignored, as it is not needed.
  warnings.warn("`model_dir` will be ignored, as it is not needed.")
/opt/conda/lib/python3.10/site-packages/paddlex/inference/models/doc_vlm/predictor.py:493: UserWarning: 'fastdeploy-server' does not support `min_pixels`.
  warnings.warn(
/opt/conda/lib/python3.10/site-packages/paddlex/inference/models/doc_vlm/predictor.py:506: UserWarning: 'fastdeploy-server' does not support `max_pixels`.
  warnings.warn(
[2026-02-12 09:14:46,592] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,605] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,616] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,627] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,629] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,776] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,782] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,799] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,820] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,829] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,909] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,910] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,914] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,939] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,951] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,980] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,986] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:46,993] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:47,034] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:47,042] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:47,048] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:47,088] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:47,095] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:47,161] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:47,278] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:47,381] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026-02-12 09:14:48,256] [    INFO] _client.py:1740 - HTTP Request: POST http://localhost:8118/v1/chat/completions "HTTP/1.1 200 OK"
[2026/02/12 09:14:48] paddleocr INFO: Processed item 0 in 39545.345067977905 ms
[32m{'res': {'input_path': 'paddleocr_vl_demo.png', 'page_index': None, 'page_count': None, 'width': 1524, 'height': 1368, 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True, 'use_chart_recognition': False, 'use_seal_recognition': False, 'use_ocr_for_image_block': False, 'format_block_content': False, 'merge_layout_blocks': True, 'markdown_ignore_labels': ['number', 'footnote', 'header', 'header_image', 'footer', 'footer_image', 'aside_text'], 'return_layout_polygon_points': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 6, 'label': 'doc_title', 'score': 0.9300567507743835, 'coordinate': [130, 35, 1384, 127], 'order': 1, 'polygon_points': array([[130.,  35.],
       ...,
       [130., 127.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.8483847379684448, 'coordinate': [582, 157, 930, 183], 'order': 2, 'polygon_points': array([[582., 157.],
       ...,
       [582., 183.]], dtype=float32)}, {'cls_id': 14, 'label': 'image', 'score': 0.9810925126075745, 'coordinate': [777, 201, 1502, 685], 'order': None, 'polygon_points': array([[777., 201.],
       ...,
       [777., 685.]], dtype=float32)}, {'cls_id': 24, 'label': 'vision_footnote', 'score': 0.431095153093338, 'coordinate': [810, 702, 1452, 724], 'order': None, 'polygon_points': array([[810., 702.],
       ...,
       [810., 724.]], dtype=float32)}, {'cls_id': 24, 'label': 'vision_footnote', 'score': 0.6346690654754639, 'coordinate': [809, 702, 1486, 750], 'order': None, 'polygon_points': array([[ 809,  702],
       ...,
       [1455,  702]], dtype=int32)}, {'cls_id': 24, 'label': 'vision_footnote', 'score': 0.35058093070983887, 'coordinate': [1246, 729, 1487, 750], 'order': None, 'polygon_points': array([[1246,  729],
       ...,
       [1486,  729]], dtype=int32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9565537571907043, 'coordinate': [9, 199, 361, 342], 'order': 3, 'polygon_points': array([[  9., 199.],
       ...,
       [  9., 342.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9485597610473633, 'coordinate': [8, 344, 360, 440], 'order': 4, 'polygon_points': array([[  8., 344.],
       ...,
       [  8., 440.]], dtype=float32)}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9114848375320435, 'coordinate': [27, 455, 341, 520], 'order': 5, 'polygon_points': array([[ 27., 455.],
       ...,
       [ 27., 520.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9484402537345886, 'coordinate': [8, 535, 359, 655], 'order': 6, 'polygon_points': array([[  8., 535.],
       ...,
       [  8., 655.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9521226286888123, 'coordinate': [8, 656, 361, 773], 'order': 7, 'polygon_points': array([[  8., 656.],
       ...,
       [  8., 773.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9330596923828125, 'coordinate': [8, 776, 360, 846], 'order': 8, 'polygon_points': array([[  8., 776.],
       ...,
       [  8., 846.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9604430794715881, 'coordinate': [8, 847, 361, 1061], 'order': 9, 'polygon_points': array([[   8.,  847.],
       ...,
       [   8., 1061.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9469948410987854, 'coordinate': [8, 1063, 360, 1181], 'order': 10, 'polygon_points': array([[   8., 1063.],
       ...,
       [   8., 1181.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9434540867805481, 'coordinate': [8, 1183, 361, 1301], 'order': 11, 'polygon_points': array([[   8., 1183.],
       ...,
       [   8., 1301.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9266975522041321, 'coordinate': [9, 1303, 361, 1351], 'order': 12, 'polygon_points': array([[   9., 1303.],
       ...,
       [   9., 1351.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9517078399658203, 'coordinate': [389, 199, 742, 294], 'order': 13, 'polygon_points': array([[389., 199.],
       ...,
       [389., 294.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9540072679519653, 'coordinate': [389, 296, 743, 440], 'order': 14, 'polygon_points': array([[389., 296.],
       ...,
       [389., 440.]], dtype=float32)}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.8874742388725281, 'coordinate': [407, 454, 721, 520], 'order': 15, 'polygon_points': array([[407., 454.],
       ...,
       [407., 520.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.929058313369751, 'coordinate': [390, 535, 742, 607], 'order': 16, 'polygon_points': array([[390., 535.],
       ...,
       [390., 607.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9516089558601379, 'coordinate': [389, 609, 742, 749], 'order': 17, 'polygon_points': array([[389., 609.],
       ...,
       [389., 749.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9524988532066345, 'coordinate': [389, 751, 741, 893], 'order': 18, 'polygon_points': array([[389., 751.],
       ...,
       [389., 893.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9566522240638733, 'coordinate': [389, 895, 742, 1037], 'order': 19, 'polygon_points': array([[ 389.,  895.],
       ...,
       [ 389., 1037.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9500834345817566, 'coordinate': [389, 1039, 742, 1133], 'order': 20, 'polygon_points': array([[ 389., 1039.],
       ...,
       [ 389., 1133.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9630914330482483, 'coordinate': [388, 1135, 742, 1351], 'order': 21, 'polygon_points': array([[ 388., 1135.],
       ...,
       [ 388., 1351.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9659959077835083, 'coordinate': [770, 773, 1124, 1062], 'order': 22, 'polygon_points': array([[ 770.,  773.],
       ...,
       [ 770., 1062.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9498739838600159, 'coordinate': [770, 1062, 1124, 1183], 'order': 23, 'polygon_points': array([[ 770., 1062.],
       ...,
       [ 770., 1183.]], dtype=float32)}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.8923305869102478, 'coordinate': [790, 1198, 1103, 1263], 'order': 24, 'polygon_points': array([[ 790., 1198.],
       ...,
       [ 790., 1263.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9308299422264099, 'coordinate': [770, 1278, 1124, 1352], 'order': 25, 'polygon_points': array([[ 770., 1278.],
       ...,
       [ 770., 1352.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.8169823288917542, 'coordinate': [1154, 774, 1333, 797], 'order': 26, 'polygon_points': array([[1154.,  774.],
       ...,
       [1154.,  797.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9588348269462585, 'coordinate': [1151, 798, 1506, 989], 'order': 27, 'polygon_points': array([[1151.,  798.],
       ...,
       [1151.,  989.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9523345828056335, 'coordinate': [1152, 991, 1506, 1109], 'order': 28, 'polygon_points': array([[1152.,  991.],
       ...,
       [1152., 1109.]], dtype=float32)}, {'cls_id': 22, 'label': 'text', 'score': 0.9643592238426208, 'coordinate': [1152, 1111, 1507, 1352], 'order': 29, 'polygon_points': array([[1152., 1111.],
       ...,
       [1152., 1352.]], dtype=float32)}]}, 'parsing_res_list': [{'block_label': 'doc_title', 'block_content': '助力双方交往 搭建友谊桥梁', 'block_bbox': [130, 35, 1384, 127]}, {'block_label': 'text', 'block_content': '本报记者 沈小晓 任彦 黄培昭', 'block_bbox': [582, 157, 930, 183]}, {'block_label': 'image', 'block_content': '', 'block_bbox': [777, 201, 1502, 685]}, {'block_label': 'vision_footnote', 'block_content': '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。中国驻厄立特里亚大使馆供图', 'block_bbox': [809, 702, 1486, 750]}, {'block_label': 'text', 'block_content': '身着中国传统民族服装的厄立特里亚青年依次登台表演中国民族舞、现代舞、扇子舞等，曼妙的舞姿赢得现场观众阵阵掌声。这是日前厄立特里亚高等教育与研究院孔子学院(以下简称“厄特孔院”)举办“喜迎新年”中国歌舞比赛的场景。', 'block_bbox': [9, 199, 361, 342]}, {'block_label': 'text', 'block_content': '中国和厄立特里亚传统友谊深厚。近年来，在高质量共建“一带一路”框架下，中厄两国人文交流不断深化，互利合作的民意基础日益深厚。', 'block_bbox': [8, 344, 360, 440]}, {'block_label': 'paragraph_title', 'block_content': '“学好中文，我们的未来不是梦”', 'block_bbox': [27, 455, 341, 520]}, {'block_label': 'text', 'block_content': '“鲜花曾告诉我你怎样走过，大地知道你心中的每一个角落……”厄立特里亚阿斯马拉大学综合楼二层，一阵优美的歌声在走廊里回响。循着熟悉的旋律轻轻推开一间教室的门，学生们正跟着老师学唱中文歌曲《同一首歌》。', 'block_bbox': [8, 535, 359, 655]}, {'block_label': 'text', 'block_content': '这是厄特孔院阿斯马拉大学教学点的一节中文歌曲课。为了让学生们更好地理解歌词大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐字翻译和解释歌词。随着伴奏声响起，学生们边唱边随着节拍摇动身体，现场气氛热烈。', 'block_bbox': [8, 656, 361, 773]}, {'block_label': 'text', 'block_content': '“这是中文歌曲初级班，共有32人。学生大部分来自首都阿斯马拉的中小学，年龄最小的仅有6岁。”尤斯拉告诉记者。', 'block_bbox': [8, 776, 360, 846]}, {'block_label': 'text', 'block_content': '尤斯拉今年23岁，是厄立特里亚一所公立学校的艺术老师。她12岁开始在厄特孔院学习中文，在2017年第十届“汉语桥”世界中学生中文比赛中获得厄立特里亚赛区第一名，并和同伴代表厄立特里亚前往中国参加决赛，获得团体优胜奖。2022年起，尤斯拉开始在厄特孔院兼职教授中文歌曲，每周末两个课时。“中国文化博大精深，我希望我的学生们能够通过中文歌曲更好地理解中国文化。”她说。', 'block_bbox': [8, 847, 361, 1061]}, {'block_label': 'text', 'block_content': '“姐姐，你想去中国吗？”“非常想！我想去看故宫、爬长城。”尤斯拉的学生中有一对能歌善舞的姐妹，姐姐露娅今年15岁，妹妹莉娅14岁，两人都已在厄特孔院学习多年，中文说得格外流利。', 'block_bbox': [8, 1063, 360, 1181]}, {'block_label': 'text', 'block_content': '露娅对记者说：“这些年来，怀着对中文和中国文化的热爱，我们姐妹俩始终相互鼓励，一起学习。我们的中文一天比一天好，还学会了中文歌和中国舞。我们一定要到中国去。学好中文，我们的未来不是梦！”', 'block_bbox': [8, 1183, 361, 1301]}, {'block_label': 'text', 'block_content': '据厄特孔院中方院长黄鸣飞介绍，这所孔院成立于2013年3月，由贵州财经大学和厄立特里亚高等教育与研究院合作建立，开设了中国语言课程和中国文化课程，注册学生2万余人次。10余年来，厄特孔院已成为当地民众了解中国的一扇窗口。', 'block_bbox': [9, 1303, 361, 1351]}, {'block_label': 'text', 'block_content': '', 'block_bbox': [389, 199, 742, 294]}, {'block_label': 'text', 'block_content': '黄鸣飞表示，随着来学习中文的人日益增多，阿斯马拉大学教学点已难以满足教学需要。2024年4月，由中企蜀道集团所属四川路桥承建的孔院教学楼项目在阿斯马拉开工建设，预计今年上半年竣工，建成后将为厄特孔院提供全新的办学场地。', 'block_bbox': [389, 296, 743, 440]}, {'block_label': 'paragraph_title', 'block_content': '“在中国学习的经历让我看到更广阔的世界”', 'block_bbox': [407, 454, 721, 520]}, {'block_label': 'text', 'block_content': '多年来，厄立特里亚广大赴华留学生和培训人员积极投身国家建设，成为助力该国发展的人才和厄中友好的见证者和推动者。', 'block_bbox': [390, 535, 742, 607]}, {'block_label': 'text', 'block_content': '在厄立特里亚全国妇女联盟工作的约翰娜·特韦尔德·凯莱塔就是其中一位。她曾在中华女子学院攻读硕士学位，研究方向是女性领导力与社会发展。其间，她实地走访中国多个地区，获得了观察中国社会发展的第一手资料。', 'block_bbox': [389, 609, 742, 749]}, {'block_label': 'text', 'block_content': '谈起在中国求学的经历，约翰娜记忆犹新：“中国的发展在当今世界是独一无二的。沿着中国特色社会主义道路坚定前行，中国创造了发展奇迹，这一切都离不开中国共产党的领导。中国的发展经验值得许多国家学习借鉴。”', 'block_bbox': [389, 751, 741, 893]}, {'block_label': 'text', 'block_content': '正在西南大学学习的厄立特里亚博士生穆卢盖塔·泽穆伊对中国怀有深厚感情。8年前，在北京师范大学获得硕士学位后，穆卢盖塔在社交媒体上写下这样一段话：“这是我人生的重要一步，自此我拥有了一双坚固的鞋子，赋予我穿越荆棘的力量。”', 'block_bbox': [389, 895, 742, 1037]}, {'block_label': 'text', 'block_content': '穆卢盖塔密切关注中国在经济、科技、教育等领域的发展，“中国在科研等方面的实力与日俱增。在中国学习的经历让我看到更广阔的世界，从中受益匪浅。”', 'block_bbox': [389, 1039, 742, 1133]}, {'block_label': 'text', 'block_content': '23岁的莉迪亚·埃斯蒂法诺斯已在厄特孔院学习3年，在中国书法、中国画等方面表现十分优秀，在2024年厄立特里亚赛区的“汉语桥”比赛中获得一等奖。莉迪亚说：“学习中国书法让我的内心变得安宁和纯粹。我也喜欢中国的服饰，希望未来能去中国学习，把中国不同民族元素融入服装设计中，创作出更多精美作品，也把厄特文化分享给更多的中国朋友。”\n“不管远近都是客人，请不用客气；相约好了在一起，我们欢迎你……”在一场中厄青年联谊活动上，四川路桥中方员工同当地大学生合唱《北京欢迎你》。厄立特里亚技术学院计算机科学与工程专业学生鲁夫塔·谢拉是其中一名演唱者，她很早便在孔院学习中文，一直在为去中国留学作准备。“这句歌词是我们两国人民友谊的生动写照。无论是投身于厄立特里亚基础设施建设的中企员工，还是在中国留学的厄立特里亚学子，两国人民携手努力，必将推动两国关系不断向前发展。”鲁夫塔说。', 'block_bbox': [388, 1135, 742, 1351]}, {'block_label': 'text', 'block_content': '', 'block_bbox': [770, 773, 1124, 1062]}, {'block_label': 'text', 'block_content': '厄立特里亚高等教育委员会主任助理萨马瑞表示：“每年我们都会组织学生到中国访问学习，目前有超过5000名厄立特里亚学生在中国留学。学习中国的教育经验，有助于提升厄立特里亚的教育水平。”', 'block_bbox': [770, 1062, 1124, 1183]}, {'block_label': 'paragraph_title', 'block_content': '“共同向世界展示非洲和亚洲的灿烂文明”', 'block_bbox': [790, 1198, 1103, 1263]}, {'block_label': 'text', 'block_content': '从阿斯马拉出发，沿着蜿蜒曲折的盘山公路一路向东寻找丝路印迹。驱车两个小时，记者来到位于厄立特里亚港口城市马萨瓦的北红海省博物馆。', 'block_bbox': [770, 1278, 1124, 1352]}, {'block_label': 'text', 'block_content': '', 'block_bbox': [1154, 774, 1333, 797]}, {'block_label': 'text', 'block_content': '博物馆二层陈列着一个发掘自阿杜利斯古城的中国古代陶制酒器，罐身上写着“万”“和”“禅”“山”等汉字。“这件文物证明，很早以前我们就通过海上丝绸之路进行贸易往来与文化交流。这也是厄立特里亚与中国友好交往历史的有力证明。”北红海省博物馆研究与文献部负责人伊萨亚斯·特斯法兹吉说。', 'block_bbox': [1151, 798, 1506, 989]}, {'block_label': 'text', 'block_content': '厄立特里亚国家博物馆考古学和人类学研究员菲尔蒙·特韦尔德十分喜爱中国文化。他表示：“学习彼此的语言和文化，将帮助厄中两国人民更好地理解彼此，助力双方交往，搭建友谊桥梁。”', 'block_bbox': [1152, 991, 1506, 1109]}, {'block_label': 'text', 'block_content': '厄立特里亚国家博物馆馆长塔吉丁·努里达姆·优素福曾多次访问中国，对中华文明的传承与创新、现代化博物馆的建设与发展印象深刻。“中国博物馆不仅有许多保存完好的文物，还充分运用先进科技手段进行展示，帮助人们更好理解中华文明。”塔吉丁说，“厄立特里亚与中国都拥有悠久的文明，始终相互理解、相互尊重。我希望未来与中国同行加强合作，共同向世界展示非洲和亚洲的灿烂文明。”', 'block_bbox': [1152, 1111, 1507, 1352]}]}}[0m

```
