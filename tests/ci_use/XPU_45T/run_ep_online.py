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


def test_ep():
    ip = "0.0.0.0"
    service_http_port = "8188"  # 服务配置的
    client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY")
    base_response_ep8tp1 = "你好呀！我是一个人工智能助手，可以帮你解答问题、提供建议、陪你聊天，或者做点有趣的事情～比如现在，你想聊点什么？😊"
    base_response_ep8tp8 = "你好呀！我是一个人工智能助手，可以帮你解答问题、提供建议、陪你聊天，或者做任何你需要的帮助～😊 你有什么想聊的或者需要帮忙的吗？"
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
        response.choices[0].message.content == base_response_ep8tp1
        or response.choices[0].message.content == base_response_ep8tp8
    )


if __name__ == "__main__":
    test_ep()
