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

"""
VL模型测试 - ERNIE-4.5-VL-28B 视觉语言模型

测试配置:
- 模型: ERNIE-4.5-VL-28B-A3B-Thinking
- 量化: wint8
- Tensor Parallel: 4
- 特性: reasoning-parser, tool-call-parser, enable-chunked-prefill
"""


import openai
import pytest
from conftest import get_model_path, get_port_num, print_logs_on_failure, start_server


def test_vl_model(xpu_env):
    """VL视觉语言模型测试"""

    print("\n============================开始vl模型测试!============================")

    # 获取配置
    port_num = get_port_num()
    model_path = get_model_path()

    # 构建服务器启动参数
    server_args = [
        "--model",
        f"{model_path}/ERNIE-4.5-VL-28B-A3B-Thinking",
        "--port",
        str(port_num),
        "--engine-worker-queue-port",
        str(port_num + 1),
        "--metrics-port",
        str(port_num + 2),
        "--cache-queue-port",
        str(port_num + 47873),
        "--tensor-parallel-size",
        "4",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "32",
        "--quantization",
        "wint8",
        "--reasoning-parser",
        "ernie-45-vl-thinking",
        "--tool-call-parser",
        "ernie-45-vl-thinking",
        "--mm-processor-kwargs",
        '{"image_max_pixels": 12845056 }',
        "--enable-chunked-prefill",
    ]

    # 启动服务器
    if not start_server(server_args):
        pytest.fail("VL模型服务启动失败")

    # 执行测试
    try:
        ip = "0.0.0.0"
        client = openai.Client(base_url=f"http://{ip}:{port_num}/v1", api_key="EMPTY_API_KEY")

        # 非流式对话(带图像)
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
                            },
                        },
                        {"type": "text", "text": "图片中的文物来自哪个时代?"},
                    ],
                },
            ],
            temperature=1,
            top_p=0,
            stream=False,
        )

        print(f"\n模型回复: {response.choices[0].message.content}")

        # 验证响应
        assert any(
            keyword in response.choices[0].message.content for keyword in ["北魏", "北齐", "释迦牟尼", "北朝"]
        ), f"响应内容不符合预期: {response.choices[0].message.content}"

        print("\nVL模型测试通过!")

    except Exception as e:
        print(f"\nVL模型测试失败: {str(e)}")
        print_logs_on_failure()
        pytest.fail(f"VL模型测试失败: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
