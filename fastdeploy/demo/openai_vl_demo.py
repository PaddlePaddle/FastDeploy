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

import openai

print("hello")
ip = "0.0.0.0"
service_http_port = "9809"
client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY")
print("world")

# 非流式对话
response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI assistant.",
        },  # system不是必需，可选
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://ku.baidu-int.com/vk-assets-ltd/space/2024/09/13/933d1e0a0760498e94ec0f2ccee865e0",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": "请描述图片内容"},
            ],
        },
    ],
    temperature=1,
    max_tokens=53,
    stream=False,
)
print(response)

# 流式对话，历史多轮
response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI assistant.",
        },  # system不是必需，可选
        {"role": "user", "content": "List 3 countries and their capitals."},
        {
            "role": "assistant",
            "content": "China(Beijing), France(Paris), Australia(Canberra).",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://ku.baidu-int.com/vk-assets-ltd/space/2024/09/13/933d1e0a0760498e94ec0f2ccee865e0",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": "请描述图片内容"},
            ],
        },
    ],
    temperature=1,
    max_tokens=512,
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta is not None:
        # print(chunk.choices[0].delta, end='')
        # print("\n")
        print(chunk.choices[0].delta.content, end="")
print(response)
