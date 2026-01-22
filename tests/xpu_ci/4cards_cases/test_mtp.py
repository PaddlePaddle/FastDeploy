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
MTP模式测试 - ERNIE-4.5-21B-A3B-Paddle 模型

测试配置:
- 模型: ERNIE-4.5-21B-A3B-Paddle
- 量化: wint4
- Tensor Parallel: 4
"""

import json

import openai
import pytest
from conftest import get_model_path, get_port_num, print_logs_on_failure, start_server


def test_mtp_mode(xpu_env):
    """mtp模式测试"""

    print("\n============================开始mtp模式测试!============================")

    # 获取配置
    port_num = get_port_num()
    model_path = get_model_path()
    spec_config = {"method": "mtp", "num_speculative_tokens": 1, "model": f"{model_path}/ERNIE-4.5-21B-A3B-Paddle/mtp"}
    # 构建服务器启动参数
    server_args = [
        "--model",
        f"{model_path}/ERNIE-4.5-21B-A3B-Paddle",
        "--port",
        str(port_num),
        "--engine-worker-queue-port",
        str(port_num + 1),
        "--metrics-port",
        str(port_num + 2),
        "--tensor-parallel-size",
        "4",
        "--num-gpu-blocks-override",
        "16384",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "128",
        "--quantization",
        "wint4",
        "--speculative-config",
        f"{json.dumps(spec_config)}",
    ]

    # 启动服务器
    if not start_server(server_args):
        pytest.fail("mtp模式服务启动失败")

    # 执行测试
    try:
        ip = "0.0.0.0"
        client = openai.Client(base_url=f"http://{ip}:{port_num}/v1", api_key="EMPTY_API_KEY")

        # 非流式对话
        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "user", "content": "你好,你是谁?"},
            ],
            temperature=1,
            top_p=0,
            max_tokens=64,
            stream=False,
        )

        print(f"\n模型回复: {response.choices[0].message.content}")

        # 验证响应
        assert any(
            keyword in response.choices[0].message.content for keyword in ["人工智能", "文心一言", "百度", "智能助手"]
        ), f"响应内容不符合预期: {response.choices[0].message.content}"

        print("\nmtp模式测试通过!")

    except Exception as e:
        print(f"\nmtp模式测试失败: {str(e)}")
        print_logs_on_failure()
        pytest.fail(f"mtp模式测试失败: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
