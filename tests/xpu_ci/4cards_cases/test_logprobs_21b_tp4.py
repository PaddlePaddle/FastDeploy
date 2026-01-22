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

import pytest
import requests
from conftest import (
    get_model_path,
    get_port_num,
    print_logs_on_failure,
    restore_env,
    setup_logprobs_env,
    start_server,
)


def test_logprobs_mode(xpu_env):
    """logprobs 测试（HTTP 直连，不使用 SDK）"""

    print("\n============================开始 logprobs 测试!============================")

    port_num = get_port_num()
    model_path = get_model_path()

    original_env = setup_logprobs_env()

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
        "--no-enable-prefix-caching",
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
            "top_logprobs": 1,
            "prompt_logprobs": 1,
        }

        resp = requests.post(url, json=payload, timeout=300)
        assert resp.status_code == 200, f"HTTP 请求失败: {resp.text}"

        response = resp.json()
        print("\n完整返回:\n", response)

        # ========================
        # 基本返回结构
        # ========================
        assert "choices" in response
        assert isinstance(response["choices"], list)
        assert len(response["choices"]) > 0

        choice = response["choices"][0]

        # ========================
        # message 结构
        # ========================
        assert "message" in choice
        assert "content" in choice["message"]
        assert isinstance(choice["message"]["content"], str)
        assert len(choice["message"]["content"]) > 0

        print(f"\n模型回复: {choice['message']['content']}")

        # ========================
        # completion logprobs
        # ========================
        assert "logprobs" in choice
        assert choice["logprobs"] is not None

        assert "content" in choice["logprobs"]
        assert isinstance(choice["logprobs"]["content"], list)
        assert len(choice["logprobs"]["content"]) > 0

        for token_info in choice["logprobs"]["content"]:
            assert "token" in token_info
            assert "logprob" in token_info
            assert "bytes" in token_info
            assert "top_logprobs" in token_info

            assert isinstance(token_info["token"], str)
            assert isinstance(token_info["logprob"], (int, float))
            assert isinstance(token_info["bytes"], list)
            assert token_info["top_logprobs"] is None or isinstance(token_info["top_logprobs"], list)

        # ========================
        # prompt_logprobs（扩展字段）
        # ========================
        assert "prompt_logprobs" in choice
        assert isinstance(choice["prompt_logprobs"], list)
        assert len(choice["prompt_logprobs"]) > 0

        print("\nlogprobs 测试通过!")

    except Exception as e:
        print(f"\nlogprobs 测试失败: {str(e)}")
        print_logs_on_failure()
        pytest.fail(f"logprobs 测试失败: {str(e)}")

    finally:
        restore_env(original_env)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
