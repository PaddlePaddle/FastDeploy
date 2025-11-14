[简体中文](../../zh/get_started/installation/iluvatar_gpu.md)

# Run ERNIE-4.5-300B-A47B & ERNIE-4.5-21B-A3B model on iluvatar machine

## Machine Preparation
First, the `TP=16` when running the ERNIE4.5 300B model and so you need to prepare a machine with the following configurations:

| CPU | Memory | Card | Hard Disk|
| :---: | :---: | :---: | :---: |
| x86 | 1TB| 16xBI150| 1TB|

Currently, the entire model needs to be loaded into the host memory, which requires more than 600GB of host memory. This issue will be optimized in subsequent versions.

## Image Preparation
Pull the Docker image

```bash
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
```

## Container Preparation
### Start Container

```bash
docker run -itd --name paddle_infer --network host -v /usr/src:/usr/src -v /lib/modules:/lib/modules -v /dev:/dev -v /home/paddle:/home/paddle --privileged --cap-add=ALL --pid=host ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
docker exec -it paddle_infer bash
```

/home/paddle contains the model files, *.whl packages, and scripts.

### Install paddle

```bash
pip3 install paddlepaddle==3.1.0a0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip3 install paddle-iluvatar-gpu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/
```
For latest paddle version on iluvatar. Refer to [PaddlePaddle Installation](https://www.paddlepaddle.org.cn/)

### Install or build FastDeploy
```bash
pip3 install fastdeploy_iluvatar_gpu==2.1.0.dev0 -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simplels
```
You can build FastDeploy from source if you need the ```latest version```.
```bash
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy
pip install -r requirements_iluvatar.txt
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
bash build.sh
```

## Prepare the inference demo script

script list below:

`run_demo.sh`:

```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export INFERENCE_MSG_QUEUE_ID=232132
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export FD_SAMPLING_CLASS=rejection
export FD_DEBUG=1
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
llm = LLM(model="/home/paddle/ernie-4_5-21b-a3b-bf16-paddle", tensor_parallel_size=4, max_model_len=8192, block_size=16, quantization='wint8')

# Perform batch inference
outputs = llm.generate(prompts, sampling_params)
# Note：Replace `/home/paddle/ernie-4_5-21b-a3b-bf16-paddle` in it with the path to the ERNIE model you have downloaded.

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    print(prompt, generated_text)
```

## run demo

```bash
./run_demo.sh
```

The following logs will be printed: Loading the model took approximately 74 seconds, and running the demo took approximately 240 seconds.

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

## Run ernie4.5 300B model with the GSM8K dataset

1. Download GSM8K dataset

```bash
wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
```

2. Prepare `bench_gsm8k.py`

```python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Fastdeploy + ERNIE-4.5-Turbo 的指标评估 """
# adapted from https://github.com/sgl-project/sglang/blob/main/benchmark/gsm8k/bench_other.py
import argparse
import ast
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from tqdm import tqdm

INVALID = -9999999


def call_generate(prompt, **kwargs):
    """
    Generates response based on the input prompt.

    Args:
        prompt (str): The input prompt text.
        **kwargs: Keyword arguments, including server IP address and port number.

    Returns:
        str: The response generated based on the prompt.

    """
    url = f"http://{kwargs['ip']}:{kwargs['port']}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.6,
        "max_tokens": 2047,
        "top_p": 0.95,
        "do_sample": True,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    out = response.json()
    return out["choices"][0]["message"]["content"]


def get_one_example(lines, i, include_answer):
    """
    Retrieves a question-answer example from the given list of text lines.

    Args:
        lines (list of dict): A list of question-answer pairs.
        i (int): The index of the question-answer pair to retrieve from lines.
        include_answer (bool): Whether to include the answer in the returned string.

    Returns:
        str: A formatted question-answer string in the format "Question: <question>\nAnswer: <answer>".

    """
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    """
    Selects k examples from the given list of text lines and concatenates them into a single string.

    Args:
        lines (list): A list containing text lines.
        k (int): The number of examples to select.

    Returns:
        str: A string composed of k examples, separated by two newline characters.
    """
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    """
    Extracts numerical values from an answer string and returns them.

    Args:
        answer_str (str): The string containing the answer.

    Returns:
        The extracted numerical value; returns "INVALID" if extraction fails.
    """
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def read_jsonl(filename: str):
    """
    Reads a JSONL file.

    Args:
        filename (str): Path to the JSONL file.

    Yields:
        dict: A dictionary object corresponding to each line in the JSONL file.
    """
    with open(filename) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            yield json.loads(line)


def main(args):
    """
    Process inputs and generate answers by calling the model in parallel using a thread pool.

    Args:
        args (argparse.Namespace):
            - num_questions (int): Number of questions to process.
            - num_shots (int): Number of few-shot learning examples.
            - ip (str): IP address of the model service.
            - port (int): Port number of the model service.
            - parallel (int): Number of questions to process in parallel.
            - result_file (str): File path to store the results.

    Returns:
        None

    """
    # Read data
    filename = "test.jsonl"

    lines = list(read_jsonl(filename))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)

    states = [None] * len(labels)

    # Use thread pool
    def get_one_answer(i):
        answer = call_generate(
            prompt=few_shot_examples + questions[i],
            # stop=["Question", "Assistant:", "<|separator|>"],
            ip=args.ip,
            port=args.port,
        )
        states[i] = answer

    tic = time.time()
    if args.parallel == 1:
        for i in tqdm(range(len(questions))):
            get_one_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            list(
                tqdm(
                    executor.map(get_one_answer, list(range(len(questions)))),
                    total=len(questions),
                )
            )

    latency = time.time() - tic
    preds = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]))

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")

    with open(args.result_file, "a") as fout:
        value = {
            "task": "gsm8k",
            "backend": "paddlepaddle",
            "num_gpus": 1,
            "latency": round(latency, 3),
            "accuracy": round(acc, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="8188")
    parser.add_argument("--num-shots", type=int, default=10)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=1319)
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()
    main(args)
```

3. Prepare `run_bench.sh`

```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export INFERENCE_MSG_QUEUE_ID=232132
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export FD_SAMPLING_CLASS=rejection

python3 -m fastdeploy.entrypoints.openai.api_server --model "/home/paddle/ernie-45t" --port 8188 --tensor-parallel-size 16 --block-size 16 --quantization wint8
```

4. Running the Script

Firstly, open a terminal and run:
```bash
./run_bench.sh
```
After the service is ready, open another terminal and run:
```bash
python3 -u bench_gsm8k.py --port 8188 --num-questions 1319 --num-shots 5 --parallel 8
```
It takes about 4.8 hours to run the GSM8K dataset.

```
Accuracy: 0.962
Invaild: 0.000
Latency: 17332.728 s
```

# Run ERNIE-4.5-VL-28B-A3B-Paddle model on iluvatar machine

## Machine Preparation
First, the `TP=2` when running the ERNIE-4.5-VL-28B-A3B-Paddle model and so you need to prepare a machine with the following configurations:

| CPU | Memory | Card | Hard Disk|
| :---: | :---: | :---: | :---: |
| x86 | 1TB| 2xBI150| 1TB|

## Image Preparation
Pull the Docker image

```bash
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
```

## Container Preparation
### Start Container

```bash
docker run -itd --name paddle_infer --network host -v /usr/src:/usr/src -v /lib/modules:/lib/modules -v /dev:/dev -v /home/paddle:/home/paddle --privileged --cap-add=ALL --pid=host ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
docker exec -it paddle_infer bash
```

/home/paddle contains the model files, *.whl packages, and scripts.

### Install paddle

```bash
pip3 install paddlepaddle==3.3.0.dev20251028 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
pip3 install paddle-iluvatar-gpu==3.0.0.dev20251029 -i https://www.paddlepaddle.org.cn/packages/nightly/ixuca/
```
For latest paddle version on iluvatar. Refer to [PaddlePaddle Installation](https://www.paddlepaddle.org.cn/)

### Install FastDeploy
```bash
pip3 install fastdeploy_iluvatar_gpu==2.3.0.dev0 -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/ --extra-index-url https://mirrors.aliyun.com/pypi/simple/
```

## Prepare the inference demo script

script list below:

`run_demo_vl.sh`:

```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export INFERENCE_MSG_QUEUE_ID=232132
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export FD_SAMPLING_CLASS=rejection
export FD_DEBUG=1
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

## run demo

```bash
./run_demo_vl.sh
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

## Testing thinking model

### ERNIE-4.5-21B-A3B-Thinking
Refer to [gpu doc](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/best_practices/ERNIE-4.5-21B-A3B-Thinking.md), the command is bellow:

server:
```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export INFERENCE_MSG_QUEUE_ID=232132
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export FD_SAMPLING_CLASS=rejection
export FD_DEBUG=1
python3 -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-21B-A3B-Thinking \
       --port 8180 \
       --tensor-parallel-size 2 \
       --max-model-len 32768 \
       --quantization wint8 \
       --block-size 16 \
       --reasoning-parser ernie_x1 \
       --tool-call-parser ernie_x1 \
       --max-num-seqs 8
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

### ERNIE-4.5-VL-28B-A3B
Refer to [gpu doc](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/get_started/ernie-4.5-vl.md), set `"chat_template_kwargs":{"enable_thinking": true}` and the command is bellow:

server:
```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export INFERENCE_MSG_QUEUE_ID=232132
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export FD_SAMPLING_CLASS=rejection
export FD_DEBUG=1
python3 -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-VL-28B-A3B-Paddle \
       --port 8180 \
       --tensor-parallel-size 2 \
       --max-model-len 32768 \
       --quantization wint8 \
       --block-size 16 \
       --limit-mm-per-prompt '{"image": 100, "video": 100}' \
       --reasoning-parser ernie-45-vl \
       --max-num-seqs 8
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
  "chat_template_kwargs":{"enable_thinking": true}
}'
```

### ERNIE-4.5-VL-28B-A3B-Thinking
Refer to [gpu doc](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/get_started/ernie-4.5-vl-thinking.md), the command is bellow:

server:
```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export INFERENCE_MSG_QUEUE_ID=232132
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export FD_SAMPLING_CLASS=rejection
export FD_DEBUG=1
python3 -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-VL-28B-A3B-Thinking \
       --port 8180 \
       --tensor-parallel-size 2 \
       --max-model-len 32768 \
       --quantization wint8 \
       --block-size 16 \
       --limit-mm-per-prompt '{"image": 100, "video": 100}' \
       --reasoning-parser ernie-45-vl-thinking \
       --tool-call-parser ernie-45-vl-thinking \
       --mm-processor-kwargs '{"image_max_pixels": 12845056 }' \
       --max-num-seqs 8
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
