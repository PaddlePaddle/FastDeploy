# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import io
import os
import sys

import requests
from PIL import Image

from fastdeploy import LLM, SamplingParams
from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer
from fastdeploy.utils import set_random_seed

tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, tests_dir)

from ci_use.iluvatar_UT.utils import TIMEOUT_MSG, timeout


@timeout(240)
def offline_infer_check():
    set_random_seed(123)

    PATH = "/model_data/ERNIE-4.5-VL-28B-A3B-Paddle"
    tokenizer = Ernie4_5Tokenizer.from_pretrained(PATH)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
                    },
                },
                {"type": "text", "text": "图中的文物属于哪个年代"},
            ],
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
                videos.append({"video": video_bytes, "max_frames": 30})

    sampling_params = SamplingParams(temperature=0.1, max_tokens=128)
    graph_optimization_config = {"use_cudagraph": False}
    llm = LLM(
        model=PATH,
        tensor_parallel_size=2,
        max_model_len=32768,
        block_size=16,
        quantization="wint8",
        limit_mm_per_prompt={"image": 100},
        reasoning_parser="ernie-45-vl",
        graph_optimization_config=graph_optimization_config,
    )
    outputs = llm.generate(
        prompts={"prompt": prompt, "multimodal_data": {"image": images, "video": videos}},
        sampling_params=sampling_params,
    )

    for output in outputs:
        generated_text = output.outputs.text
        print(f"generated_text={generated_text}")
        assert any(keyword in generated_text for keyword in ["北魏", "北齐", "释迦牟尼", "北朝"])

    print("PASSED")


if __name__ == "__main__":
    try:
        result = offline_infer_check()
        sys.exit(0)
    except TimeoutError:
        print(TIMEOUT_MSG)
        sys.exit(124)
    except Exception:
        sys.exit(1)
