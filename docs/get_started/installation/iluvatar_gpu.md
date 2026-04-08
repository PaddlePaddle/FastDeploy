[简体中文](../../zh/get_started/installation/iluvatar_gpu.md)

## 1. Machine Preparation

| CPU | Memory | Card | Hard Disk|
| :---: | :---: | :---: | :---: |
| x86 | 1TB| 16xBI150| 1TB|

## 2. Image Preparation
Pull the Docker image

```bash
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:3.3.0
```

## 3. Container Preparation
### 3.1 Start Container

```bash
docker run -itd --name paddle_infer --network host -v /usr/src:/usr/src -v /lib/modules:/lib/modules -v /dev:/dev -v /home/paddle:/home/paddle --privileged --cap-add=ALL --pid=host ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:3.3.0
docker exec -it paddle_infer bash
```

/home/paddle contains the model files, *.whl packages, and scripts.

### 3.2 Install paddle

```bash
pip3 install paddlepaddle==3.3.0.dev20251219 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
pip3 install paddle-iluvatar-gpu==3.0.0.dev20251223 -i https://www.paddlepaddle.org.cn/packages/nightly/ixuca/
```

### 3.3 Install or build FastDeploy
```bash
pip3 install fastdeploy_iluvatar_gpu==2.4.0.dev0 -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/ --extra-index-url https://mirrors.aliyun.com/pypi/simple/
```
You can build FastDeploy from source if you need the ```latest version```.
```bash
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy
ln -sf /usr/local/bin/python3 /usr/local/bin/python
pip3 install -r requirements_iluvatar.txt
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
bash build.sh
```

## 4. Test models on iluvatar machine
### 4.1 ERNIE-4.5 series
#### 4.1.1 ERNIE-4.5-21B-A3B-Paddle

**offline demo**

script list bellow:
```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export INFERENCE_MSG_QUEUE_ID=232132
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export FD_SAMPLING_CLASS=rejection
python3 run_demo.py
```

`run_demo.py`:

```python
from fastdeploy import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The largest ocean is",
]

# sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

# load the model
llm = LLM(model="/home/paddle/ERNIE-4.5-21B-A3B-Paddle", tensor_parallel_size=1, max_model_len=8192, block_size=16, quantization='wint8')

# Perform batch inference
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    print(prompt, generated_text)
```

The following logs will be printed:

```
/usr/local/lib/python3.10/site-packages/paddle/utils/cpp_extension/extension_utils.py:715: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
  warnings.warn(warning_message)
/usr/local/lib/python3.10/site-packages/_distutils_hack/__init__.py:31: UserWarning: Setuptools is replacing distutils. Support for replacing an already imported distutils is deprecated. In the future, this condition will fail. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
[2025-07-02 11:07:42,393] [    INFO] - Loading configuration file /home/paddle/ernie-4_5-21b-a3b-bf16-paddle/generation_config.json
/usr/local/lib/python3.10/site-packages/paddleformers/generation/configuration_utils.py:250: UserWarning: using greedy search strategy. However, `temperature` is set to `0.8` -- this flag is only used in sample-based generation modes. You should set `decode_strategy="greedy_search" ` or unset `temperature`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/usr/local/lib/python3.10/site-packages/paddleformers/generation/configuration_utils.py:255: UserWarning: using greedy search strategy. However, `top_p` is set to `0.8` -- this flag is only used in sample-based generation modes. You should set `decode_strategy="greedy_search" ` or unset `top_p`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
INFO     2025-07-02 11:07:43,589 577964 engine.py[line:207] Waitting worker processes ready...
Loading Weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:57<00:00,  1.75it/s]
Loading Layers: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:08<00:00, 11.73it/s]
INFO     2025-07-02 11:08:55,261 577964 engine.py[line:277] Worker processes are launched with 73.76574492454529 seconds.
Processed prompts: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [03:59<00:00, 119.96s/it, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Hello, my name is  Christopher. Today, I'm going to teach you how to draw a cute cartoon ghost. Let's get started!
 (1) First, draw a big circle for the ghost's head.
 (2) Then, add two small circles for the eyes, making sure they're not too big.
 (3) Next, draw a wide, open mouth that looks like a big "U".
 (4) After that, create the body by drawing a slightly smaller circle below the head.
 (5) Now, let's add some arms. Draw two short, curly lines on each side of the body.
 (6) Finally, give the ghost a wavy line at the bottom to represent its floating appearance.

Now, let's break down each step:

**Step 1: Drawing the Head**
- Start with a big circle to form the head of the ghost. This will be the foundation of your drawing.

**Step 2: Adding Eyes**
- On the head, place two small circles for the eyes. They should be centered and not too big, to give the ghost a cute and innocent look.

**Step 3: Drawing the
The largest ocean is  the Pacific Ocean, covering an area of approximately â¦ [3], The first scientific expeditions to determine the ocean's depth were the Challenger expedition (1872â1876) and the U.S. Navy Hydrographic Office survey (1877â1879). The oceanic crust is thin and irregular, consisting of upward moving magma from the mantle below, and cooling and solidifying on the surface. The shallowest parts of the ocean are called the continental shelves. Large tides are caused mainly by the alignment of the Sun, Moon, and Earth during new or full moons. The origin of the word "ocean" is not clear. The first global oceanic topography survey was completed by the Challenger expedition (1872â1876). [57] The sound speed in the ocean is primarily a function of water temperature and salinity, and varies with depth. The deep-ocean floor is mostly flat and devoid of life, with the exception of seamounts and various underwater volcanic features, including seamounts and hydrothermal vents. [73] Today, the five ocean
```

**online demo**

Refer to [gpu doc](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/best_practices/ERNIE-4.5-VL-28B-A3B-Paddle.md), the command as bellow:

server:
```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export INFERENCE_MSG_QUEUE_ID=232132
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export FD_SAMPLING_CLASS=rejection
python3 -m fastdeploy.entrypoints.openai.api_server \
       --model /home/paddle/ERNIE-4.5-21B-A3B-Paddle \
       --port 8180 \
       --tensor-parallel-size 1 \
       --quantization wint8 \
       --max-model-len 32768 \
       --max-num-seqs 8 \
       --block-size 16
```
If you want to use v0 loader, please set `--load-choices "default"`.

client:

- Simple request:
```bash
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Write me a poem about large language model."}
  ]
}'
```

- Test GSM8K dataset benchmark
1) Download GSM8K dataset
```bash
wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
```
2) Copy `bench_gsm8k.py` to your workspace
```bash
cp FastDeploy/tests/ci_use/iluvatar_UT/bench_gsm8k.py .
```
3) Execute
```bash
python3 -u bench_gsm8k.py --port 8180 --num-questions 1319 --num-shots 5 --parallel 8
```
It takes about 52 minutes to run the GSM8K dataset.

```
Accuracy: 0.914
Invaild: 0.000
Latency: 3143.301 s
```

#### 4.1.2 ERNIE-4.5-21B-A3B-Thinking

Refer to [gpu doc](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/best_practices/ERNIE-4.5-21B-A3B-Thinking.md), the command as bellow:

server:
```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export INFERENCE_MSG_QUEUE_ID=232132
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export FD_SAMPLING_CLASS=rejection
python3 -m fastdeploy.entrypoints.openai.api_server \
       --model /home/paddle/ERNIE-4.5-21B-A3B-Thinking \
       --port 8180 \
       --tensor-parallel-size 1 \
       --max-model-len 32768 \
       --quantization wint8 \
       --reasoning-parser ernie_x1 \
       --tool-call-parser ernie_x1 \
       --max-num-seqs 8 \
       --block-size 16
```

client:

```bash
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Write me a poem about large language model."}
  ]
}'
```

#### 4.1.3 ERNIE-4.5-300B-A47B
Firstly, the TP=16 when running the `ERNIE-4.5-300B-A47B` and it needs to be loaded into the host memory, which requires more than 600GB of host memory. This issue will be optimized in subsequent versions.

Refer to [gpu doc](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/best_practices/ERNIE-4.5-300B-A47B-Paddle.md), the command as bellow:

server:
```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export INFERENCE_MSG_QUEUE_ID=232132
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export FD_SAMPLING_CLASS=rejection
python3 -m fastdeploy.entrypoints.openai.api_server \
       --model /home/paddle/ERNIE-4.5-300B-A47B \
       --port 8180 \
       --tensor-parallel-size 16 \
       --quantization wint8 \
       --max-model-len 32768 \
       --max-num-seqs 8 \
       --block-size 16
```
If you want to use v0 loader, please set `--load-choices "default"`.

client:

- Simple request:
```bash
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Write me a poem about large language model."}
  ]
}'
```

- Test GSM8K dataset benchmark
1) Download GSM8K dataset
```bash
wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
```
2) Copy `bench_gsm8k.py` to your workspace
```bash
cp FastDeploy/tests/ci_use/iluvatar_UT/bench_gsm8k.py .
```
3) Execute
```bash
python3 -u bench_gsm8k.py --port 8180 --num-questions 1319 --num-shots 5 --parallel 8
```
It takes about 52 minutes to run the GSM8K dataset.

```
Accuracy: 0.962
Invaild: 0.000
Latency: 17332.728 s
```

### 4.2 ERNIE-4.5-VL series
#### 4.2.1 ERNIE-4.5-VL-28B-A3B-Paddle

**offline demo**

The script as bellow:

`run_demo_vl.sh`:

```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export INFERENCE_MSG_QUEUE_ID=232132
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export FD_SAMPLING_CLASS=rejection
python3 run_demo_vl.py
```

`run_demo_vl.py`:

```python
import io
import requests
from PIL import Image

from fastdeploy.entrypoints.llm import LLM
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer


PATH = "/home/paddle/ERNIE-4.5-VL-28B-A3B-Paddle"
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
prompt = tokenizer.apply_chat_template(messages, tokenize=False)
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
llm = LLM(model=PATH, tensor_parallel_size=2, max_model_len=32768, block_size=16, quantization="wint8", limit_mm_per_prompt={"image": 100}, reasoning_parser="ernie-45-vl")
outputs = llm.generate(prompts={
    "prompt": prompt,
    "multimodal_data": {
        "image": images,
        "video": videos
    }
}, sampling_params=sampling_params)
# Output results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    reasoning_text = output.outputs.reasoning_content
    print(f"generated_text={generated_text}")
```

The following logs will be printed:

```
[2025-09-23 10:13:10,844] [    INFO] - Using download source: huggingface
[2025-09-23 10:13:10,844] [    INFO] - loading configuration file /home/paddle/ERNIE-4.5-VL-28B-A3B-Paddle/preprocessor_config.json
[2025-09-23 10:13:10,845] [    INFO] - Using download source: huggingface
[2025-09-23 10:13:10,845] [    INFO] - Loading configuration file /home/paddle/ERNIE-4.5-VL-28B-A3B-Paddle/generation_config.json
/usr/local/lib/python3.10/site-packages/paddleformers/generation/configuration_utils.py:250: UserWarning: using greedy search strategy. However, `temperature` is set to `0.2` -- this flag is only used in sample-based generation modes. You should set `decode_strategy="greedy_search" ` or
unset `temperature`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/usr/local/lib/python3.10/site-packages/paddleformers/generation/configuration_utils.py:255: UserWarning: using greedy search strategy. However, `top_p` is set to `0.8` -- this flag is only used in sample-based generation modes. You should set `decode_strategy="greedy_search" ` or unset
`top_p`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.                                                                                                            warnings.warn(
INFO     2025-09-23 10:13:11,969 3880245 engine.py[line:136] Waiting worker processes ready...
Loading Weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [02:21<00:00,  1.41s/it]
Loading Layers: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:15<00:00,  6.65it/s]
INFO     2025-09-23 10:15:53,672 3880245 engine.py[line:173] Worker processes are launched with 181.2426426410675 seconds.
prompts: 100%|███████████████████████████████████| 1/1 [01:52<00:00, 112.74s/it, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
generated_text=
图中的文物是**北齐释迦牟尼佛像**，属于**北齐（公元550年－577年）**的文物。

这件佛像具有典型的北齐风格，佛像结跏趺坐于莲花座上，身披通肩袈裟，面部圆润，神态安详，体现了北齐佛教艺术的独特魅力。
```

**online demo**

Refer to [gpu doc](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/get_started/ernie-4.5-vl.md), the command as bellow:

server:
```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export INFERENCE_MSG_QUEUE_ID=232132
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export FD_SAMPLING_CLASS=rejection
python3 -m fastdeploy.entrypoints.openai.api_server \
       --model /home/paddle/ERNIE-4.5-VL-28B-A3B-Paddle \
       --port 8180 \
       --tensor-parallel-size 2 \
       --max-model-len 32768 \
       --quantization wint8 \
       --limit-mm-per-prompt '{"image": 100, "video": 100}' \
       --reasoning-parser ernie-45-vl \
       --max-num-seqs 8 \
       --block-size 16
```

client:

```bash
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type": "text", "text": "From which era does the artifact in the image originate?"}
    ]}
  ],
  "chat_template_kwargs":{"enable_thinking": false}
}'
```

#### 4.2.2 ERNIE-4.5-VL-28B-A3B-Thinking

Refer to [gpu doc](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/get_started/ernie-4.5-vl-thinking.md), the command as bellow:

server:
```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export INFERENCE_MSG_QUEUE_ID=232132
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export FD_SAMPLING_CLASS=rejection
python3 -m fastdeploy.entrypoints.openai.api_server \
       --model /home/paddle/ERNIE-4.5-VL-28B-A3B-Thinking \
       --port 8180 \
       --tensor-parallel-size 2 \
       --max-model-len 32768 \
       --quantization wint8 \
       --limit-mm-per-prompt '{"image": 100, "video": 100}' \
       --reasoning-parser ernie-45-vl-thinking \
       --tool-call-parser ernie-45-vl-thinking \
       --mm-processor-kwargs '{"image_max_pixels": 12845056 }' \
       --max-num-seqs 8 \
       --block-size 16
```

client:
```bash
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type":"image_url", "image_url": {"url":"https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type":"text", "text":"From which era does the artifact in the image originate?"}
    ]}
  ]
}'
```

### 4.3 PaddleOCR-VL series
#### 4.3.1 PaddleOCR-VL-0.9B

- (Optional) Install paddleocr

To install the latest `paddleocr`, you can compile it from source. The image contains a compilation and installation based on source code `39128c2c7fd40be44d8f33498cabd4ec10f1bfcd`.

```bash
git clone -b main https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
pip3 install -e ".[doc-parser]"
```

Refer to [gpu doc](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/best_practices/PaddleOCR-VL-0.9B.md), the command as bellow:

server:
```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export INFERENCE_MSG_QUEUE_ID=232132
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export FD_SAMPLING_CLASS=rejection
python3 -m fastdeploy.entrypoints.openai.api_server \
       --model /data1/fastdeploy/PaddleOCR-VL \
       --port 8180 \
       --metrics-port 8471 \
       --engine-worker-queue-port 8472 \
       --cache-queue-port 55660 \
       --max-model-len 16384 \
       --max-num-batched-tokens 16384 \
       --max-num-seqs 64 \
       --workers 2 \
       --block-size 16
```

client:

**simple demo**

```bash
paddleocr doc_parser -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png --vl_rec_backend fastdeploy-server --vl_rec_server_url http://127.0.0.1:8180/v1
```

The output is:

{'res': {'input_path': '/root/.paddlex/predict_input/paddleocr_vl_demo.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True, 'use_chart_recognition': False, 'format_block_content': False}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 6, 'label': 'doc_title', 'score': 0.9636866450309753, 'coordinate': [131.31543, 36.45137, 1384.522, 127.98457]}, {'cls_id': 22, 'label': 'text', 'score': 0.928146243095398, 'coordinate': [585.39355, 158.43787, 930.2197, 182.57446]}, {'cls_id': 22, 'label': 'text', 'score': 0.9840242266654968, 'coordinate': [9.02211, 200.86037, 361.41748, 343.8839]}, {'cls_id': 14, 'label': 'image', 'score': 0.9871442914009094, 'coordinate': [775.5067, 200.66461, 1503.379, 684.9366]}, {'cls_id': 22, 'label': 'text', 'score': 0.9801799058914185, 'coordinate': [9.532669, 344.90558, 361.44202, 440.8252]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9708914756774902, 'coordinate': [28.03984, 455.88013, 341.72076, 520.7113]}, {'cls_id': 22, 'label': 'text', 'score': 0.9825296401977539, 'coordinate': [8.897079, 536.5491, 361.0522, 655.80566]}, {'cls_id': 22, 'label': 'text', 'score': 0.982223391532898, 'coordinate': [8.970978, 657.4961, 362.01614, 774.6245]}, {'cls_id': 24, 'label': 'vision_footnote', 'score': 0.9001952409744263, 'coordinate': [809.06995, 703.70044, 1488.3029, 750.5239]}, {'cls_id': 22, 'label': 'text', 'score': 0.9767361879348755, 'coordinate': [9.407532, 776.5222, 361.31128, 846.8281]}, {'cls_id': 22, 'label': 'text', 'score': 0.9868096113204956, 'coordinate': [8.669312, 848.2549, 361.64832, 1062.8562]}, {'cls_id': 22, 'label': 'text', 'score': 0.9826636910438538, 'coordinate': [8.8025055, 1063.8627, 361.46454, 1182.8519]}, {'cls_id': 22, 'label': 'text', 'score': 0.9825499653816223, 'coordinate': [8.82019, 1184.4667, 361.66507, 1302.4513]}, {'cls_id': 22, 'label': 'text', 'score': 0.9584522843360901, 'coordinate': [9.170425, 1304.2166, 361.48846, 1351.7488]}, {'cls_id': 22, 'label': 'text', 'score': 0.978195309638977, 'coordinate': [389.1593, 200.38223, 742.76196, 295.65167]}, {'cls_id': 22, 'label': 'text', 'score': 0.9844739437103271, 'coordinate': [388.73267, 297.18472, 744.0012, 441.30356]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9680613875389099, 'coordinate': [409.39398, 455.8943, 721.71893, 520.9389]}, {'cls_id': 22, 'label': 'text', 'score': 0.9741637706756592, 'coordinate': [389.7167, 536.8141, 742.71155, 608.0021]}, {'cls_id': 22, 'label': 'text', 'score': 0.9840295910835266, 'coordinate': [389.30914, 609.3971, 743.0931, 750.32263]}, {'cls_id': 22, 'label': 'text', 'score': 0.9845904111862183, 'coordinate': [389.1331, 751.77673, 743.05884, 894.88196]}, {'cls_id': 22, 'label': 'text', 'score': 0.9848388433456421, 'coordinate': [388.83295, 896.0353, 743.5821, 1038.7367]}, {'cls_id': 22, 'label': 'text', 'score': 0.9804726243019104, 'coordinate': [389.0833, 1039.9131, 742.7598, 1134.4902]}, {'cls_id': 22, 'label': 'text', 'score': 0.9864556789398193, 'coordinate': [388.5259, 1135.8118, 743.45215, 1352.0105]}, {'cls_id': 22, 'label': 'text', 'score': 0.9869311451911926, 'coordinate': [769.8312, 775.6598, 1124.9835, 1063.2106]}, {'cls_id': 22, 'label': 'text', 'score': 0.9822818040847778, 'coordinate': [770.3026, 1063.9371, 1124.8307, 1184.2206]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.968923032283783, 'coordinate': [791.3031, 1199.3169, 1104.454, 1264.6992]}, {'cls_id': 22, 'label': 'text', 'score': 0.9712913036346436, 'coordinate': [770.42285, 1279.6072, 1124.6924, 1351.8679]}, {'cls_id': 22, 'label': 'text', 'score': 0.9236321449279785, 'coordinate': [1153.9055, 775.5812, 1334.0662, 798.1588]}, {'cls_id': 22, 'label': 'text', 'score': 0.985789954662323, 'coordinate': [1151.5193, 799.27954, 1506.362, 991.1172]}, {'cls_id': 22, 'label': 'text', 'score': 0.9820653796195984, 'coordinate': [1151.5708, 991.9118, 1506.6016, 1110.8875]}, {'cls_id': 22, 'label': 'text', 'score': 0.9865990877151489, 'coordinate': [1151.6917, 1112.1348, 1507.1611, 1351.9453]}]}, 'parsing_res_list': [{'block_label': 'doc_title', 'block_content': '助力双方交往 搭建友谊桥梁', 'block_bbox': [131, 36, 1384, 127]}, {'block_label': 'text', 'block_content': '本报记者 沈小晓 任彦 黄培昭', 'block_bbox': [585, 158, 930, 182]}, {'block_label': 'text', 'block_content': '身着中国传统民族服装的厄立特里亚青年依次登台表演中国民族舞、现代舞、扇子舞等，曼妙的舞姿赢得现场观众阵阵掌声。这是日前厄立特里亚高等教育与研究院孔子学院(以下简称“厄特孔院”)举办“喜迎新年”中国歌舞比赛的场景。', 'block_bbox': [9, 200, 361, 343]}, {'block_label': 'image', 'block_content': '', 'block_bbox': [775, 200, 1503, 684]}, {'block_label': 'text', 'block_content': '中国和厄立特里亚传统友谊深厚。近年来，在高质量共建“一带一路”框架下，中厄两国人文交流不断深化，互利合作的民意基础日益深厚。', 'block_bbox': [9, 344, 361, 440]}, {'block_label': 'paragraph_title', 'block_content': '“学好中文,我们的未来不是梦”', 'block_bbox': [28, 455, 341, 520]}, {'block_label': 'text', 'block_content': '鲜花曾告诉我你怎样走过，大地知道你心中的每一个角落……”厄立特里亚阿斯马拉大学综合楼二层，一阵优美的歌声在走廊里回响。循着熟悉的旋律轻轻推开一间教室的门，学生们正跟着老师学唱中文歌曲《同一首歌》。', 'block_bbox': [8, 536, 361, 655]}, {'block_label': 'text', 'block_content': '这是厄特孔院阿斯马拉大学教学点的一节中文歌曲课。为了让学生们更好地理解歌词大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐字翻译和解释歌词。随着伴奏声响起，学生们边唱边随着节拍摇动身体，现场气氛热烈。', 'block_bbox': [8, 657, 362, 774]}, {'block_label': 'vision_footnote', 'block_content': '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。\n中国驻厄立特里亚大使馆供图', 'block_bbox': [809, 703, 1488, 750]}, {'block_label': 'text', 'block_content': '“这是中文歌曲初级班，共有32人。学生大部分来自首都阿斯马拉的中小学，年龄最小的仅有6岁。”尤斯拉告诉记者。', 'block_bbox': [9, 776, 361, 846]}, {'block_label': 'text', 'block_content': '尤斯拉今年23岁，是厄立特里亚一所公立学校的艺术老师。她12岁开始在厄特孔院学习中文，在2017年第十届“汉语桥”世界中学生中文比赛中获得厄立特里亚赛区第一名，并和同伴代表厄立特里亚前往中国参加决赛，
获得团体优胜奖。2022年起，尤斯拉开始在厄特孔院兼职教授中文歌曲，每周末两个课时。“中国文化博大精深，我希望我的学生们能够通过中文歌曲更好地理解中国文化。”她说。', 'block_bbox': [8, 848, 361, 1062]}, {'block_label': 'text', 'block_content': '“姐姐，你想去中国吗？”“非常想！我想去看故宫、爬长城。”尤斯拉的学生中有一对能歌善舞的姐妹，姐姐露娅今年15岁，妹妹莉娅14岁，两人都已在厄特孔院学习多年，中文说得格外流利。', 'block_bbox': [8, 1063, 361, 1182]}, {'block_label': 'text', 'block_content': '露娅对记者说：“这些年来，怀着对中文和中国文化的热爱，我们姐妹俩始终相互鼓励，一起学习。我们的中文一天比一天好，还学会了中文歌和中国舞。我们一定要到中国去。学好中文，我们的未来不是梦！”', 'block_bbox': [8, 1184, 361, 1302]}, {'block_label': 'text', 'block_content': '据厄特孔院中方院长黄鸣飞介绍，这所孔院成立于2013年3月，由贵州财经大学和厄立特里亚高等教育与研究院合作建立，开设了中国语言课程和中国文化课程，注册学生2万余人次。10余年来，厄特孔院已成为当地民众了解中国的一扇窗口。', 'block_bbox': [9, 1304, 361, 1351]}, {'block_label': 'text', 'block_content': '', 'block_bbox': [389, 200, 742, 295]}, {'block_label': 'text', 'block_content': '黄鸣飞表示，随着来学习中文的人日益增多，阿斯马拉大学教学点已难以满足教学需要。2024年4月，由中企蜀道集团所属四川路桥承建的孔院教学楼项目在阿斯马拉开工建设，预计今年上半年竣工，建成后将为厄特孔院提供全新的办学场地。', 'block_bbox': [388, 297, 744, 441]}, {'block_label': 'paragraph_title', 'block_content': '“在中国学习的经历让我看到更广阔的世界”', 'block_bbox': [409, 455, 721, 520]}, {'block_label': 'text', 'block_content': '多年来，厄立特里亚广大赴华留学生和培训人员积极投身国家建设，成为助力该国发展的人才和厄中友好的见证者和推动者。', 'block_bbox': [389, 536, 742, 608]}, {'block_label': 'text', 'block_content': '在厄立特里亚全国妇女联盟工作的约翰娜·特韦尔德·凯莱塔就是其中一位。她曾在中华女子学院攻读硕士学位，研究方向是女性领导力与社会发展。其间，她实地走访中国多个地区，获得了观察中国社会发展的第一手资料。', 'block_bbox': [389, 609, 743, 750]}, {'block_label': 'text', 'block_content': '谈起在中国求学的经历，约翰娜记忆犹新：“中国的发展在当今世界是独一无二的。沿着中国特色社会主义道路坚定前行，中国创造了发展奇迹，这一切都离不开中国共产党的领导。中国的发展经验值得许多国家学习借鉴。”', 'block_bbox': [389, 751, 743, 894]}, {'block_label': 'text', 'block_content': '正在西南大学学习的厄立特里亚博士生穆卢盖塔·泽穆伊对中国怀有深厚感情。8年前，在北京师范大学获得硕士学位后，穆卢盖塔在社交媒体上写下这样一段话：“这是我人生的重要一步，自此我拥有了一双坚固的鞋子，赋予我穿越荆棘的力量。”', 'block_bbox': [388, 896, 743, 1038]}, {'block_label': 'text', 'block_content': '穆卢盖塔密切关注中国在经济、科技、教育等领域的发展，“中国在科研等方面的实力与日俱增。在中国学习的经历让我看到更广阔的世界，从中受益匪浅。”', 'block_bbox': [389, 1039, 742, 1134]}, {'block_label': 'text', 'block_content': '23岁的莉迪亚·埃斯蒂法诺斯已在厄特孔院学习3年，在中国书法、中国画等方面表现十分优秀，在2024年厄立特里亚赛区的“汉语桥”比赛中获得一等奖。莉迪亚说：“学习中国书法让我的内心变得安宁和纯粹。我也喜欢中国的服饰，希望未来能去中国学习，把中国不同民族元素融入服装设计中，创作出更多精美作品，也把厄特文化分享给更多的中国朋友。”\n“不管远近都是客人，请不用客气；相约好了在一起，我们欢迎你……”在一场中厄青年联谊活动上，四川路桥中方员工同当地大学生合唱《北京
欢迎你》。厄立特里亚技术学院计算机科学与工程专业学生鲁夫塔·谢拉是其中一名演唱者，她很早便在孔院学习中文，一直在为去中国留学作准备。“这句歌词是我们两国人民友谊的生动写照。无论是投身于厄特里亚基础设施建设的中企员工，还是在中国留学的厄立特里亚学子，两国人民携手努力，必将推动两国关系不断向前发展。”鲁夫塔说。', 'block_bbox': [388, 1135, 743, 1352]}, {'block_label': 'text', 'block_content': '', 'block_bbox': [769, 775, 1124, 1063]}, {'block_label': 'text', 'block_content': '厄立特里亚高等教育委员会主任助理萨马瑞表示：“每年我们都会组织学生到中国访问学习，目前有超过5000名厄立特里亚学生在中国留学。学习中国的教育经验，有助于提升厄立特里亚的教育水平。”', 'block_bbox': [770, 1063, 1124, 1184]}, {'block_label': 'paragraph_title', 'block_content': '“共同向世界展示非洲和亚洲的灿烂文明”', 'block_bbox': [791, 1199, 1104, 1264]}, {'block_label': 'text', 'block_content': '从阿斯马拉出发，沿着蜿蜒曲折的盘山公路一路向东寻找丝路印迹。驱车两个小时，记者来到位于厄立特里亚港口城市马萨瓦的北红海省博物馆。', 'block_bbox': [770, 1279, 1124, 1351]}, {'block_label': 'text', 'block_content': '', 'block_bbox': [1153, 775, 1334, 798]}, {'block_label': 'text', 'block_content': '博物馆二层陈列着一个发掘自阿杜利斯古城的中国古代陶制酒器，罐身上写着“万”“和”“禅”“山”等汉字。“这件文物证明，很早以前我们就通过海上丝绸之路进行贸易往来与文化交流。这也是厄立特里亚与中国友好交往历史的有力证明。”北红海省博物馆研究与文献部负责人伊萨亚斯·特斯法兹吉说。', 'block_bbox': [1151, 799, 1506, 991]}, {'block_label': 'text', 'block_content': '厄立特里亚国家博物馆考古学和人类学研究员菲尔蒙·特韦尔德十分喜爱中国文化。他表示：“学习彼此的语言和文化，将帮助厄中两国人民更好地理解彼此，助力双方交往，搭建友谊桥梁。”', 'block_bbox': [1151, 991, 1506, 1110]}, {'block_label': 'text', 'block_content': '厄立特里亚国家博物馆馆长塔吉丁·努里达姆·优素福曾多次
访问中国，对中华文明的传承与创新、现代化博物馆的建设与发展印象深刻。“中国博物馆不仅有许多保存完好的文物，还充分运用先进科技手段进行展示，帮助人们更好理解中华文明。”塔吉丁说，“厄立特里亚与中国都拥有悠久的文明，始终相互理解、相互尊重。我希望未来与中国同行加强合作，共同向世界展示非洲和亚洲的灿烂文明。”', 'block_bbox': [1151, 1112, 1507, 1351]}]}}

**benchmark**

1. Download and extract image datasets

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/internal/tmp/images.tar
tar xvf images.tar
```

2. Prepare `infer_ocr_vl_benchmark.py`

```python
import os
from paddleocr import PaddleOCRVL

input_path = "./images"
pipeline = PaddleOCRVL(vl_rec_backend="fastdeploy-server", vl_rec_server_url="http://127.0.0.1:8180/v1")
file_list = os.listdir(input_path)
for file_name in file_list:
    file_path = os.path.join(input_path, file_name)
    output = pipeline.predict(file_path)
    for res in output:
        res.print()
        res.save_to_markdown(save_path="output", pretty=False)
```

3. execute `infer_ocr_vl_benchmark.py` on client

```bash
python3 infer_ocr_vl_benchmark.py
```

After each image is inferred, a corresponding `md` file will be generated in the `output` path. Running the entire benchmark (1355 images) takes approximately 5 hours.
