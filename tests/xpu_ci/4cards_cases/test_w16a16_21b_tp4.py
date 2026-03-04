# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
w16a16 TP4 测试

测试配置:
- 模型: ERNIE-4.5-21B-A3B-Paddle
- 量化: w16a16
- Tensor Parallel: 4
- Expert Parallel: 不启用
- Data Parallel: 1
"""


import openai
import pytest
from conftest import (
    get_model_path,
    get_port_num,
    print_logs_on_failure,
    restore_env,
    setup_ep_env,
    start_server,
)


def test_w16a16_21b_tp4(xpu_env):
    """w16a16 TP4通信测试"""

    print("\n============================开始 w16a16 tp4 测试!============================")

    # 设置EP环境变量
    original_env = setup_ep_env()

    try:
        # 获取配置
        port_num = get_port_num()
        model_path = get_model_path()

        # 构建服务器启动参数
        # 注意: 与EP4TP4 online相比,这里不使用 --disable-sequence-parallel-moe
        server_args = [
            "--model",
            f"{model_path}/ERNIE-4.5-21B-A3B-Paddle",
            "--port",
            str(port_num),
            "--tensor-parallel-size",
            "4",
            "--enable-prefix-caching",
            "--data-parallel-size",
            "1",
            "--max-model-len",
            "32768",
            "--max-num-seqs",
            "64",
            "--engine-worker-queue-port",
            str(port_num + 10),
            "--metrics-port",
            str(port_num + 2),
            "--gpu-memory-utilization",
            "0.9",
        ]

        # 启动服务器
        if not start_server(server_args):
            pytest.fail("w16a16 TP4服务启动失败")

        # 执行测试
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

        print("\nw16a16 TP4测试通过!")

    except Exception as e:
        print(f"\nw16a16 TP4测试失败: {str(e)}")
        print_logs_on_failure()
        pytest.fail(f"w16a16 TP4测试失败: {str(e)}")

    finally:
        # 恢复环境变量
        restore_env(original_env)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
