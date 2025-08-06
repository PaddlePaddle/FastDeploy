"""
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
"""

import json

from core import TEMPLATE, build_request_payload, send_request

URL = "http://0.0.0.0:8566/v1/chat/completions"

def test_stream_single_prompt_echo_response():
    """测试单prompt流式响应是否能够成功回显"""
    data = {
        "model": "TEXT-21B",
        "prompt": "以下是你的自我介绍：",
        "max_tokens": 30,
        "temperature": 0.7,
        "echo": True
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload, stream=True)

    output = ""
    for line in resp.iter_lines(decode_unicode=True):
        if line.strip() == "" or not line.startswith("data: "):
            continue
        line = line[len("data: ") :]
        if line.strip() == "[DONE]":
            break
        chunk = json.loads(line)
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        output += delta.get("content", "")

    print("Stream输出:", output)
    assert "以下是你的自我介绍：" in output

def test_stream_multi_prompt_echo():
    """测试多prompt流式响应是否能够成功回显"""
    data = {
        "model": "TEXT-21B",
        "prompt": ["写一首李白的诗词:","写一首杜甫的诗："],
        "max_tokens": 30,
        "temperature": 0.7,
        "stream": True,
        "echo": True
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload, stream=True)

    outputs = [""] * len(data["prompt"])
    for line in resp.iter_lines(decode_unicode=True):
        if line.strip() == "" or not line.startswith("data: "):
            continue
        line = line[len("data: ") :]
        if line.strip() == "[DONE]":
            break
        chunk = json.loads(line)
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        index = choice.get("index", 0)
        if index < len(outputs):
            outputs[index] += delta.get("content", "")

    for i, prompt in enumerate(data["prompt"]):
        print(f"Prompt {i}: {prompt}")
        print(f"Response {i}: {outputs[i]}")
        assert prompt in outputs[i]

def test_non_stream_single_prompt_echo():
    """测试非流式单prompt的echo功能"""
    data = {
        "model": "TEXT-21B",
        "prompt": "以下是你的自我介绍：",
        "max_tokens": 30,
        "temperature": 0.7,
        "echo": True
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload, stream=True)

    output = resp.json()
    text = output["choices"][0]["message"]["content"]
    
    print("输出:", text)
    assert "以下是你的自我介绍：" in text

def test_non_stream_multi_prompt_echo():
    """测试非流式多prompt的echo功能"""
    data = {
        "model": "TEXT-21B",
        "prompt": ["写一首李白的诗词:","写一首杜甫的诗："],
        "max_tokens": 30,
        "temperature": 0.7,
        "echo": True
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload, stream=True)

    output = resp.json()
    choices = output["choices"]
    
    for i, choice in enumerate(choices):
        text = choice["message"]["content"]
        print(f"Prompt {i}: {data['prompt'][i]}")
        print(f"Response {i}: {text}")
        assert data["prompt"][i] in text