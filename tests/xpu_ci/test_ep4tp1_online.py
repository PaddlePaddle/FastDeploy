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
EP4TP1在线服务测试 - Expert Parallel + Tensor Parallel

测试配置:
- 模型: ERNIE-4.5-300B-A47B-Paddle
- 量化: wint4
- Tensor Parallel: 1
- Expert Parallel: 启用
- Data Parallel: 4
"""


import subprocess
import time

import openai
import pytest
from conftest import (
    cleanup_resources,
    download_and_build_xdeepep,
    get_model_path,
    get_port_num,
    print_logs_on_failure,
    restore_env,
    setup_ep_env,
    stop_processes,
)


def wait_for_health_check(port, timeout=900, interval=10):
    """
    等待服务健康检查通过

    Args:
        timeout: 超时时间(秒), 默认15分钟
        interval: 检查间隔(秒), 默认10秒

    Returns:
        bool: 服务是否启动成功
    """
    health_endpoint = f"http://0.0.0.0:{port}/health"
    start_time = time.time()

    print(f"开始服务端口{port}健康检查,最长等待时间:{timeout}秒")

    # 第一阶段: 等待 /health 返回 200
    while True:
        elapsed = int(time.time() - start_time)

        # 超时判断
        if elapsed >= timeout:
            print(f"\n服务启动超时:经过 {timeout//60} 分钟服务仍未启动!")
            return False

        # 发送健康检查请求
        try:
            result = subprocess.run(
                f'curl -s -o /dev/null -w "%{{http_code}}" -m 2 {health_endpoint}',
                shell=True,
                capture_output=True,
                text=True,
            )
            http_code = result.stdout.strip()
        except Exception:
            http_code = "000"

        print(f"\r服务端口{port}健康检查中... 已等待 {elapsed} 秒,当前状态码:{http_code}", end="", flush=True)

        if http_code == "200":
            print(f"\n端口{port}健康检查通过!耗时 {elapsed} 秒")
            break

        time.sleep(interval)

    return True


def test_ep4tp1_online(xpu_env):
    """EP4TP1在线服务测试"""

    print("\n============================开始 EP4TP1 在线服务测试!============================")

    # 下载并编译xDeepEP
    if not download_and_build_xdeepep():
        pytest.fail("xDeepEP下载或编译失败")

    # 设置EP环境变量
    original_env = setup_ep_env()

    stop_processes()

    cleanup_resources()

    try:
        # 获取配置
        model_path = get_model_path()
        port_num = get_port_num()
        router_port = port_num
        server_ports = [port_num + 1, port_num + 2, port_num + 3, port_num + 4]
        metrics_ports = [port_num + 11, port_num + 12, port_num + 13, port_num + 14]
        engine_worker_queue_ports = [port_num + 21, port_num + 22, port_num + 23, port_num + 24]

        # start router
        router_cmd = [
            "python",
            "-m",
            "fastdeploy.router.launch",
            "--port",
            str(router_port),
        ]
        print(f"Router start command: {' '.join(router_cmd)}")
        with open("router.log", "w") as log_file:
            subprocess.Popen(router_cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(1)

        # start server
        server_cmd = [
            "python",
            "-m",
            "fastdeploy.entrypoints.openai.multi_api_server",
            "--ports",
            f"{','.join([str(i) for i in server_ports])}",
            "--num-servers",
            "4",
            "--metrics-ports",
            f"{','.join([str(i) for i in metrics_ports])}",
            "--args",
            "--model",
            f"{model_path}/ERNIE-4.5-21B-A3B-Paddle",
            "--engine-worker-queue-port",
            f"{','.join([str(i) for i in engine_worker_queue_ports])}",
            "--max-model-len",
            "32768",
            "--max-num-seqs",
            "64",
            "--data-parallel-size",
            "4",
            "--tensor-parallel-size",
            "1",
            "--enable-expert-parallel",
            "--quantization",
            "wint4",
            "--enable-prefix-caching",
            "--router",
            f"0.0.0.0:{router_port}",
        ]
        print(f"服务启动命令: {' '.join(server_cmd)}")
        with open("server.log", "w") as log_file:
            subprocess.Popen(server_cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)

        ports_to_check = [router_port] + server_ports
        for port_to_check in ports_to_check:
            if not wait_for_health_check(port_to_check):
                print_logs_on_failure()
                stop_processes()
                pytest.fail("EP4TP1服务启动失败")
        # wait for pd register
        time.sleep(5)
        # 执行测试
        ip = "0.0.0.0"
        client = openai.Client(base_url=f"http://{ip}:{router_port}/v1", api_key="EMPTY_API_KEY")

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

        print("\nEP4TP1在线服务测试通过!")

    except Exception as e:
        print(f"\nEP4TP1在线服务测试失败: {str(e)}")
        print_logs_on_failure()
        pytest.fail(f"EP4TP1在线服务测试失败: {str(e)}")

    finally:
        # 恢复环境变量
        restore_env(original_env)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
