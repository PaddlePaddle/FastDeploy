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
import os

import openai


def test_w4a8():
    ip = "0.0.0.0"
    xpu_id = int(os.getenv("XPU_ID", "0"))
    service_http_port = 8188 + xpu_id * 100  # 服务配置的
    client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY")
    # base_response_110 = "你好！我是一个基于人工智能技术的助手，可以帮你解答问题、提供建议、聊天或者协助完成各种任务。无论是学习、工作还是生活中的疑问，我都可以尽力提供帮助。😊 你有什么想聊的吗？"
    # base_response_104 = "你好！我是一个人工智能助手，可以帮你解答问题、提供建议、聊天或者完成一些任务。无论是学习、工作还是生活中的疑问，我都可以尽力帮忙哦～有什么需要我做的吗？😊"
    # 非流式对话
    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "user", "content": "你好，你是谁？"},
        ],
        temperature=1,
        top_p=0,
        max_tokens=64,
        stream=False,
    )
    print(response.choices[0].message.content)
    # print(base_response)
    assert any(keyword in response.choices[0].message.content for keyword in ["人工智能", "文心一言"])


if __name__ == "__main__":
    test_w4a8()
