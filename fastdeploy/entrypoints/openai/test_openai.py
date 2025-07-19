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

ip = "0.0.0.0"
service_http_port = "9908"  # 服务配置的

client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY")

# 非流式返回, completion接口不会使用chat template对输入进行处理
response = client.completions.create(
    model="default",
    prompt="There are 50 kinds of fruits, include apple, banana, pineapple",
    max_tokens=100,
    seed=13,
    stream=False,
)

print(response)
print("\n")

# 流式返回, completion接口不会使用chat template对输入进行处理
response = client.completions.create(
    model="default",
    prompt="Hello, how are you?",
    max_tokens=100,
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].text, end="")
print("\n")

# Chat completion
# 非流式返回, 会基于chat template对输入进行拼接处理
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "Hello, who are you"},
    ],
    temperature=1,
    max_tokens=64,
    stream=False,
)

print(response)
print("\n")


# # 流式返回, 会基于chat template对输入进行拼接处理
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "Hello, who are you"},
    ],
    temperature=1,
    max_tokens=64,
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta is not None:
        print(chunk.choices[0].delta, end="")
        print("\n")
