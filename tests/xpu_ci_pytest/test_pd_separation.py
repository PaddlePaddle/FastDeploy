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
PD分离测试 - Prefill/Decode分离部署模式

测试配置:
- 模型: ERNIE-4.5-0.3B-Paddle
- Tensor Parallel: 1
- 特性: splitwise PD分离, RDMA cache传输
- 节点: Router + Prefill节点 + Decode节点
"""

import os
import pytest
import openai
from conftest import (
    get_port_num,
    get_model_path,
    stop_processes,
    setup_pd_env,
    restore_pd_env,
    start_pd_server,
    print_pd_logs_on_failure,
)


def test_pd_separation():
    """PD分离部署模式测试"""

    print("\n============================开始PD分离测试!============================")

    # 设置PD分离环境变量
    original_env = setup_pd_env()

    # 检查RDMA网卡是否配置成功
    rdma_nics = os.environ.get("KVCACHE_RDMA_NICS", "")
    if not rdma_nics:
        pytest.fail("KVCACHE_RDMA_NICS is empty, please check the output of get_rdma_nics.sh")
    print(f"KVCACHE_RDMA_NICS: {rdma_nics}")

    try:
        # 获取配置
        port_num = get_port_num()
        model_path = get_model_path()

        # 启动PD分离服务
        if not start_pd_server(model_path, port_num):
            pytest.fail("PD分离服务启动失败")

        # 执行测试 - 通过Router端口访问
        ip = "0.0.0.0"
        client = openai.Client(
            base_url=f"http://{ip}:{port_num}/v1",
            api_key="EMPTY_API_KEY"
        )

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

        print(f"\n模型回复: {response.choices[0].message.content}")

        # 验证响应
        assert any(
            keyword in response.choices[0].message.content
            for keyword in ["AI", "伙伴"]
        ), f"响应内容不符合预期: {response.choices[0].message.content}"

        print("\nPD分离测试通过!")

    except Exception as e:
        print(f"\nPD分离测试失败: {str(e)}")
        print_pd_logs_on_failure()
        pytest.fail(f"PD分离测试失败: {str(e)}")

    finally:
        # 停止服务
        print("\n停止PD分离服务...")
        stop_processes()

        # 恢复环境变量
        restore_pd_env(original_env)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
