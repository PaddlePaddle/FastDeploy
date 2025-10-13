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

import openai


def test_45t():
    ip = "0.0.0.0"
    service_http_port = "8188"  # 服务配置的
    client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY")
    base_response_110 = "你好！我是一个基于人工智能技术开发的助手，可以帮你解答问题、提供建议、聊天交流或者完成一些任务。无论是学习、工作还是生活中的疑问，都可以随时告诉我哦～😊 你有什么想聊的吗？"
    base_response_104 = "你好！我是一个基于人工智能技术打造的助手，可以帮你解答问题、提供建议、分享知识，或者陪你聊聊天～😊 无论是学习、工作、生活还是娱乐相关的问题，都可以随时告诉我哦！你今天有什么想聊的吗？"
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
    assert (
        response.choices[0].message.content == base_response_110
        or response.choices[0].message.content == base_response_104
    )


if __name__ == "__main__":
    test_45t()
