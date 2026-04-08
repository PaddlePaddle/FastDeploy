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
V1模式测试 - ERNIE-4.5-21B-A3B 模型

测试配置:
- 模型: ERNIE-4.5-21B-A3B-Paddle
- 量化: wint8
- Tensor Parallel: 4
- 特性: enable-logprob
- 调用方式: 原生 HTTP（不使用 OpenAI SDK）
"""

import os

import pytest
import requests
from conftest import get_model_path, get_port_num, print_logs_on_failure, start_server


def test_logprobs_mode(xpu_env):
    """logprobs 测试（HTTP 直连，不使用 SDK）"""

    print("\n============================开始 logprobs 测试!============================")

    port_num = get_port_num()
    model_path = get_model_path()
    os.system("sysctl -w kernel.msgmax=131072")
    os.system("sysctl -w kernel.msgmnb=33554432")
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
        "32768",
        "--max-num-seqs",
        "128",
        "--quantization",
        "wint8",
        "--gpu-memory-utilization",
        "0.9",
        "--enable-logprob",
    ]

    if not start_server(server_args):
        pytest.fail("logprobs 服务启动失败")

    try:
        url = f"http://127.0.0.1:{port_num}/v1/chat/completions"

        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": "你好,你是谁?"}],
            "temperature": 1,
            "top_p": 0,
            "max_tokens": 64,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 3,
        }

        resp = requests.post(url, json=payload, timeout=300)
        assert resp.status_code == 200, f"HTTP 请求失败: {resp.text}"

        response = resp.json()
        print("\n完整返回:\n", response)
        # ========================
        # 基本返回结构
        # ========================
        assert isinstance(response, dict)
        assert response.get("object") == "chat.completion"
        assert "choices" in response
        assert isinstance(response["choices"], list)
        assert len(response["choices"]) > 0

        choice = response["choices"][0]

        # ========================
        # message 结构
        # ========================
        assert "message" in choice
        message = choice["message"]

        assert isinstance(message, dict)
        assert message.get("role") == "assistant"
        assert "content" in message
        assert isinstance(message["content"], str)
        assert len(message["content"]) > 0

        print(f"\n模型回复: {choice['message']['content']}")
        # ========================
        # logprobs 顶层结构
        # ========================
        assert "logprobs" in choice
        assert choice["logprobs"] is not None

        logprobs = choice["logprobs"]
        assert isinstance(logprobs, dict)
        assert "content" in logprobs
        assert isinstance(logprobs["content"], list)
        assert len(logprobs["content"]) > 0

        # ========================
        # 每个 token 的 logprob 结构
        # ========================
        for token_info in logprobs["content"]:
            assert isinstance(token_info, dict)

            # 必备字段
            assert "token" in token_info
            assert isinstance(token_info["token"], str)

            assert "logprob" in token_info
            assert isinstance(token_info["logprob"], (int, float))

            assert "bytes" in token_info
            assert isinstance(token_info["bytes"], list)
            assert all(isinstance(b, int) for b in token_info["bytes"])

            assert "top_logprobs" in token_info

            # ========================
            # top_logprobs 结构（允许为空）
            # ========================
            if token_info["top_logprobs"] is not None:
                assert isinstance(token_info["top_logprobs"], list)
                assert len(token_info["top_logprobs"]) > 0

                for top_item in token_info["top_logprobs"]:
                    assert isinstance(top_item, dict)
                    assert "token" in top_item
                    assert isinstance(top_item["token"], str)

                    assert "logprob" in top_item
                    assert isinstance(top_item["logprob"], (int, float))

                    assert "bytes" in top_item
                    assert isinstance(top_item["bytes"], list)

        # ========================
        # finish_reason & usage
        # ========================
        assert "finish_reason" in choice
        assert isinstance(choice["finish_reason"], str)

        assert "usage" in response
        usage = response["usage"]
        assert isinstance(usage, dict)
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

        print("\nlogprobs 测试通过!")

    except Exception as e:
        print(f"\nlogprobs 测试失败: {str(e)}")
        print_logs_on_failure()
        pytest.fail(f"logprobs 测试失败: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
