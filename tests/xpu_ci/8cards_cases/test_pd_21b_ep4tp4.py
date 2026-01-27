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
- 模型: ERNIE-4.5-21B-A3B-Paddle
- 量化: wint4
- Tensor Parallel: 4
- 特性: splitwise PD分离, RDMA cache传输
- 节点: Router + Prefill节点 + Decode节点
"""

import os
import shutil
import subprocess
import time

import openai
import pytest
from conftest import (
    cleanup_resources,
    get_model_path,
    get_port_num,
    restore_pd_ep_env,
    setup_pd_ep_env,
    stop_processes,
)


def wait_for_pd_health_check(port_p, port_d, timeout=600, interval=10):
    """
    等待PD分离服务健康检查通过(检查P节点和D节点)

    Args:
        port_p: Prefill节点端口
        port_d: Decode节点端口
        timeout: 超时时间(秒), 默认10分钟
        interval: 检查间隔(秒), 默认10秒

    Returns:
        bool: 服务是否启动成功
    """
    endpoint_p = f"http://0.0.0.0:{port_p}/health"
    endpoint_d = f"http://0.0.0.0:{port_d}/health"
    start_time = time.time()

    print(f"开始PD分离+EP4TP4服务健康检查,最长等待时间:{timeout}秒")

    while True:
        elapsed = int(time.time() - start_time)

        # 超时判断
        if elapsed >= timeout:
            print(f"\nPD分离服务启动超时:经过 {timeout//60} 分钟服务仍未启动!")
            return False

        # 检查P节点
        try:
            result_p = subprocess.run(
                f'curl -s -o /dev/null -w "%{{http_code}}" -m 2 {endpoint_p}',
                shell=True,
                capture_output=True,
                text=True,
            )
            http_code_p = result_p.stdout.strip()
        except Exception:
            http_code_p = "000"

        # 检查D节点
        try:
            result_d = subprocess.run(
                f'curl -s -o /dev/null -w "%{{http_code}}" -m 2 {endpoint_d}',
                shell=True,
                capture_output=True,
                text=True,
            )
            http_code_d = result_d.stdout.strip()
        except Exception:
            http_code_d = "000"

        print(
            f"\r服务健康检查中... 已等待 {elapsed} 秒,P节点状态码:{http_code_p},D节点状态码:{http_code_d}",
            end="",
            flush=True,
        )

        if http_code_p == "200" and http_code_d == "200":
            print(f"\nPD分离服务启动成功!耗时 {elapsed} 秒")
            return True

        time.sleep(interval)


def print_pd_logs_on_failure():
    """失败时打印PD分离相关日志"""
    log_dirs = ["log_router", "log_prefill", "log_decode"]

    for log_dir in log_dirs:
        nohup_path = os.path.join(log_dir, "log_0/worklog.0")
        if os.path.exists(nohup_path):
            print(f"\n========== {nohup_path} ==========")
            with open(nohup_path, "r") as f:
                print(f.read())


def start_pd_server(model_path, port_num, wait_before_check=60):
    """
    启动PD分离服务(Router + Prefill节点 + Decode节点)

    Args:
        model_path: 模型路径
        port_num: 基础端口号
        wait_before_check: 启动后等待多少秒再进行健康检查,默认60秒

    Returns:
        bool: 服务是否启动成功
    """

    # 停止旧进程
    stop_processes()

    # 清理资源
    cleanup_resources()

    # 清理并创建日志目录
    for log_dir in ["log_router", "log_prefill", "log_decode"]:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

    # 1. 启动Router
    print("启动Router...")
    router_env = os.environ.copy()
    router_env["FD_LOG_DIR"] = "log_router"
    router_cmd = [
        "python",
        "-m",
        "fastdeploy.router.launch",
        "--port",
        str(port_num),
        "--splitwise",
    ]

    with open("log_router/nohup", "w") as log_file:
        subprocess.Popen(router_cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True, env=router_env)
    print(f"Router启动命令: {' '.join(router_cmd)}")
    time.sleep(1)

    # 2. 启动Prefill节点
    print("启动Prefill节点...")
    prefill_env = os.environ.copy()
    prefill_env["FD_LOG_DIR"] = "log_prefill"
    prefill_env["XPU_VISIBLE_DEVICES"] = "0,1,2,3"

    prefill_cmd = [
        "python",
        "-m",
        "fastdeploy.entrypoints.openai.multi_api_server",
        "--port",
        str(port_num + 11),
        "--num-servers",
        "1",
        "--args",
        "--model",
        f"{model_path}/ERNIE-4.5-21B-A3B-Paddle",
        "--tensor-parallel-size",
        "4",
        "--data-parallel-size",
        "1",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "64",
        "--quantization",
        "wint4",
        "--splitwise-role",
        "prefill",
        "--cache-transfer-protocol",
        "rdma",
        "--enable-expert-parallel",
        "--disable-sequence-parallel-moe",
        "--router",
        f"0.0.0.0:{port_num}",
    ]

    with open("log_prefill/nohup", "w") as log_file:
        subprocess.Popen(
            prefill_cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True, env=prefill_env
        )
    print(f"Prefill节点启动命令: {' '.join(prefill_cmd)}")

    # 3. 启动Decode节点
    print("启动Decode节点...")
    decode_env = os.environ.copy()
    decode_env["FD_LOG_DIR"] = "log_decode"
    decode_env["XPU_VISIBLE_DEVICES"] = "4,5,6,7"

    decode_cmd = [
        "python",
        "-m",
        "fastdeploy.entrypoints.openai.multi_api_server",
        "--port",
        str(port_num + 21),
        "--num-servers",
        "1",
        "--args",
        "--model",
        f"{model_path}/ERNIE-4.5-21B-A3B-Paddle",
        "--tensor-parallel-size",
        "4",
        "--data-parallel-size",
        "1",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "64",
        "--quantization",
        "wint4",
        "--splitwise-role",
        "decode",
        "--cache-transfer-protocol",
        "rdma",
        "--enable-expert-parallel",
        "--disable-sequence-parallel-moe",
        "--router",
        f"0.0.0.0:{port_num}",
    ]

    with open("log_decode/nohup", "w") as log_file:
        subprocess.Popen(decode_cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True, env=decode_env)
    print(f"Decode节点启动命令: {' '.join(decode_cmd)}")

    # 等待服务启动
    print(f"等待 {wait_before_check} 秒让服务初始化...")
    time.sleep(wait_before_check)

    # 健康检查(检查P节点和D节点)
    port_p = port_num + 11
    port_d = port_num + 21

    if not wait_for_pd_health_check(port_p, port_d):
        print_pd_logs_on_failure()
        stop_processes()
        return False
    # ensure pd service is ready
    time.sleep(5)

    return True


def test_pd_separation():
    """PD分离部署模式测试"""

    print("\n============================开始PD分离+EP4TP4测试!============================")

    # 设置PD分离环境变量
    original_env = setup_pd_ep_env()

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
        client = openai.Client(base_url=f"http://{ip}:{port_num}/v1", api_key="EMPTY_API_KEY")

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
            keyword in response.choices[0].message.content for keyword in ["人工智能", "文心一言", "百度", "智能助手"]
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
        restore_pd_ep_env(original_env)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
