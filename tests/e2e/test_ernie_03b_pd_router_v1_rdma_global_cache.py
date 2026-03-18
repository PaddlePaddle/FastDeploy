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

# Test splitwise deployment with global cache pooling (Mooncake):
# - Use local_scheduler + router
# - Set ENABLE_V1_KVCACHE_SCHEDULER is 1, use rdma to transfer cache
# - Enable Mooncake storage backend for global cache pooling
# - Enable output caching on decode instance

import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest
import requests
from utils.serving_utils import (
    FD_API_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    clean,
    get_registered_number,
    is_port_open,
)

# Read ports from environment variables; use default values if not set
FD_CONNECTOR_PORT = int(os.getenv("FD_CONNECTOR_PORT", 8433))
FD_ROUTER_PORT = int(os.getenv("FD_ROUTER_PORT", 8533))
FD_RDMA_PORT = int(os.getenv("FD_RDMA_PORT", 8623))
FD_MOONCAKE_MASTER_PORT = FD_RDMA_PORT + 2
FD_MOONCAKE_METADATA_PORT = FD_RDMA_PORT + 3

# List of ports to clean before and after tests
PORTS_TO_CLEAN = [
    FD_API_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_CONNECTOR_PORT,
    FD_RDMA_PORT,
    FD_API_PORT + 1,
    FD_ENGINE_QUEUE_PORT + 1,
    FD_METRICS_PORT + 1,
    FD_CACHE_QUEUE_PORT + 1,
    FD_CONNECTOR_PORT + 1,
    FD_RDMA_PORT + 1,
    FD_ROUTER_PORT,
    FD_MOONCAKE_MASTER_PORT,
    FD_MOONCAKE_METADATA_PORT,
]


def wait_for_mooncake_master(host: str = "127.0.0.1", port: int = FD_MOONCAKE_MASTER_PORT, timeout: int = 30):
    """
    Wait for Mooncake master to be ready.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_open(host, port, timeout=1.0):
            print(f"Mooncake master is ready on port {port}")
            return True
        time.sleep(1)
    return False


@pytest.fixture(scope="session", autouse=True)
def setup_and_run_server():
    """
    Pytest fixture that runs once per test session:
    - Cleans ports before tests
    - Starts Mooncake Master for global cache pooling
    - Starts the API server as a subprocess
    - Waits for server port to open (up to 30 seconds)
    - Tears down server after all tests finish
    """
    print("Pre-test port cleanup...")
    clean(PORTS_TO_CLEAN)

    print("log dir clean ")
    if os.path.exists("log_router") and os.path.isdir("log_router"):
        shutil.rmtree("log_router")
    if os.path.exists("log_prefill") and os.path.isdir("log_prefill"):
        shutil.rmtree("log_prefill")
    if os.path.exists("log_decode") and os.path.isdir("log_decode"):
        shutil.rmtree("log_decode")
    if os.path.exists("log_master") and os.path.isdir("log_master"):
        shutil.rmtree("log_master")

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "ERNIE-4.5-0.3B-Paddle")
    else:
        model_path = "baidu/ERNIE-4.5-0.3B-Paddle"
    print(f"model_path: {model_path}")

    # get rdma nics
    current_dir = os.path.dirname(os.path.abspath(__file__))
    shell_path = os.path.join(current_dir, "utils/get_rdma_nics.sh")
    output = subprocess.check_output(["bash", shell_path, "gpu"], text=True)
    _, rdma_nics = output.split("=")
    print(f"shell_path: {shell_path}, rdma_nics: {rdma_nics}")

    # Mooncake environment variables
    master_ip = "127.0.0.1"
    mooncake_env = {
        "MOONCAKE_MASTER_SERVER_ADDR": f"{master_ip}:{FD_MOONCAKE_MASTER_PORT}",
        "MOONCAKE_METADATA_SERVER": f"http://{master_ip}:{FD_MOONCAKE_METADATA_PORT}/metadata",
        "MOONCAKE_GLOBAL_SEGMENT_SIZE": "1000000000",
        "MOONCAKE_PROTOCOL": "rdma",
    }

    # ======================== Start Mooncake Master ========================
    print("=== Starting Mooncake Master ===")

    # Ensure mooncake_master binary is available before starting the test
    if shutil.which("mooncake_master") is None:
        raise RuntimeError(
            "mooncake_master is not installed or not in PATH. "
            "Please install Mooncake and ensure `mooncake_master` is available in PATH "
            "before running this e2e test."
        )

    env_master = os.environ.copy()
    env_master["FD_LOG_DIR"] = "log_master"
    os.makedirs("log_master", exist_ok=True)

    master_cmd = [
        "mooncake_master",
        f"--port={FD_MOONCAKE_MASTER_PORT}",
        "--enable_http_metadata_server=true",
        "--http_metadata_server_host=0.0.0.0",
        f"--http_metadata_server_port={FD_MOONCAKE_METADATA_PORT}",
    ]

    with open("log_master/nohup", "w") as logfile:
        process_master = subprocess.Popen(
            master_cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env_master,
        )

    # Wait for Mooncake Master to be ready
    if not wait_for_mooncake_master(port=FD_MOONCAKE_MASTER_PORT, timeout=30):
        print("[ERROR] Mooncake Master failed to start")
        # Print mooncake master log for debugging
        master_log_path = "log_master/nohup"
        if os.path.exists(master_log_path):
            print(f"\n===== Mooncake Master Log ({master_log_path}) =====")
            with open(master_log_path, "r") as f:
                print(f.read())
            print("===== End of Mooncake Master Log =====\n")
        try:
            os.killpg(process_master.pid, signal.SIGTERM)
        except Exception:
            pass
        raise RuntimeError("Mooncake Master did not start")

    # ======================== Start Router ========================
    print("start router...")
    env_router = os.environ.copy()
    env_router["FD_LOG_DIR"] = "log_router"
    os.makedirs("log_router", exist_ok=True)
    router_log_path = "log_router/nohup.log"

    router_cmd = [
        sys.executable,
        "-m",
        "fastdeploy.router.launch",
        "--port",
        str(FD_ROUTER_PORT),
        "--splitwise",
    ]

    with open(router_log_path, "w") as logfile:
        process_router = subprocess.Popen(
            router_cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Enables killing full group via os.killpg
            env=env_router,
        )

    # ======================== Start Prefill Instance ========================
    print("start prefill...")
    env_prefill = os.environ.copy()
    env_prefill["CUDA_VISIBLE_DEVICES"] = "0"
    env_prefill["FD_LOG_DIR"] = "log_prefill"
    os.makedirs("log_prefill", exist_ok=True)
    # Mooncake environment variables for prefill
    for k, v in mooncake_env.items():
        env_prefill[k] = v

    prefill_log_path = "log_prefill/nohup.log"
    prefill_cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(FD_API_PORT),
        "--engine-worker-queue-port",
        str(FD_ENGINE_QUEUE_PORT),
        "--metrics-port",
        str(FD_METRICS_PORT),
        "--cache-queue-port",
        str(FD_CACHE_QUEUE_PORT),
        "--max-model-len",
        "8192",
        "--splitwise-role",
        "prefill",
        "--cache-transfer-protocol",
        "rdma",
        "--rdma-comm-ports",
        str(FD_RDMA_PORT),
        "--pd-comm-port",
        str(FD_CONNECTOR_PORT),
        "--router",
        f"0.0.0.0:{FD_ROUTER_PORT}",
        "--kvcache-storage-backend",
        "mooncake",
        "--enable-output-caching",
    ]

    # Start subprocess in new process group
    with open(prefill_log_path, "w") as logfile:
        process_prefill = subprocess.Popen(
            prefill_cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Enables killing full group via os.killpg
            env=env_prefill,
        )
    time.sleep(1)

    # ======================== Start Decode Instance ========================
    print("start decode...")
    env_decode = os.environ.copy()
    env_decode["CUDA_VISIBLE_DEVICES"] = "1"
    env_decode["FD_LOG_DIR"] = "log_decode"
    os.makedirs("log_decode", exist_ok=True)
    # Mooncake environment variables for decode
    for k, v in mooncake_env.items():
        env_decode[k] = v

    decode_log_path = "log_decode/nohup.log"
    decode_cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(FD_API_PORT + 1),
        "--engine-worker-queue-port",
        str(FD_ENGINE_QUEUE_PORT + 1),
        "--metrics-port",
        str(FD_METRICS_PORT + 1),
        "--cache-queue-port",
        str(FD_CACHE_QUEUE_PORT + 1),
        "--max-model-len",
        "8192",
        "--splitwise-role",
        "decode",
        "--cache-transfer-protocol",
        "rdma",
        "--rdma-comm-ports",
        str(FD_RDMA_PORT + 1),
        "--pd-comm-port",
        str(FD_CONNECTOR_PORT + 1),
        "--router",
        f"0.0.0.0:{FD_ROUTER_PORT}",
        "--kvcache-storage-backend",
        "mooncake",
        "--enable-output-caching",
    ]

    # Start subprocess in new process group
    with open(decode_log_path, "w") as logfile:
        process_decode = subprocess.Popen(
            decode_cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Enables killing full group via os.killpg
            env=env_decode,
        )

    # Wait up to 300 seconds for API server to be ready
    for _ in range(60):
        registered_numbers = get_registered_number(f"0.0.0.0:{FD_ROUTER_PORT}")
        if registered_numbers["prefill"] >= 1 and registered_numbers["decode"] >= 1:
            print("Prefill and decode servers are both online")
            break
        time.sleep(5)
    else:
        print("[TIMEOUT] API server failed to start in 5 minutes. Cleaning up...")
        try:
            os.killpg(process_master.pid, signal.SIGTERM)
            os.killpg(process_router.pid, signal.SIGTERM)
            os.killpg(process_prefill.pid, signal.SIGTERM)
            os.killpg(process_decode.pid, signal.SIGTERM)
            clean(PORTS_TO_CLEAN)
        except Exception as e:
            print(f"Failed to kill process group: {e}")
        raise RuntimeError(f"API server did not start on port {FD_API_PORT}")

    yield  # Run tests

    print("\n===== Post-test server cleanup... =====")
    try:
        os.killpg(process_master.pid, signal.SIGTERM)
        os.killpg(process_router.pid, signal.SIGTERM)
        os.killpg(process_prefill.pid, signal.SIGTERM)
        os.killpg(process_decode.pid, signal.SIGTERM)
        clean(PORTS_TO_CLEAN)
        print(f"Master server (pid={process_master.pid}) terminated")
        print(f"Router server (pid={process_router.pid}) terminated")
        print(f"Prefill server (pid={process_prefill.pid}) terminated")
        print(f"Decode server (pid={process_decode.pid}) terminated")
    except Exception as e:
        print(f"Failed to terminate API server: {e}")


@pytest.fixture(scope="session")
def api_url(request):
    """
    Returns the API endpoint URL for chat completions.
    """
    return f"http://0.0.0.0:{FD_ROUTER_PORT}/v1/chat/completions"


@pytest.fixture(scope="session")
def metrics_url(request):
    """
    Returns the metrics endpoint URL.
    """
    return f"http://0.0.0.0:{FD_METRICS_PORT}/metrics"


@pytest.fixture
def headers():
    """
    Returns common HTTP request headers.
    """
    return {"Content-Type": "application/json"}


def test_metrics_config(metrics_url):
    timeout = 600
    url = metrics_url.replace("metrics", "config-info")
    res = requests.get(url, timeout=timeout)
    assert res.status_code == 200


def send_request(url, payload, timeout=60):
    """
    发送请求到指定的URL，并返回响应结果。
    """
    headers = {
        "Content-Type": "application/json",
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=timeout)
        print("🟢 接收响应中...\n")
        return res
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时（超过 {timeout} 秒）")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败：{e}")
        return None


def get_stream_chunks(response):
    """解析流式返回，生成chunk List[dict]"""
    chunks = []

    if response.status_code == 200:
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data: "):
                    line = line[len("data: ") :]

                if line.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except Exception as e:
                    print(f"解析失败: {e}, 行内容: {line}")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print("返回内容：", response.text)

    return chunks


def test_chat_usage_stream(api_url):
    """测试流式chat usage"""
    payload = {
        "model": "default",
        "temperature": 0,
        "top_p": 0,
        "seed": 33,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 50,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "metadata": {"min_tokens": 10},
    }

    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)
    result = "".join([x["choices"][0]["delta"]["content"] for x in chunks[:-1]])
    print("Decode Response:", repr(result))
    assert result != "", "结果为空"
    usage = chunks[-1]["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"], "completion_tokens大于max_tokens"
    assert payload["metadata"]["min_tokens"] <= usage["completion_tokens"], "completion_tokens小于min_tokens"
    assert usage["total_tokens"] == total_tokens, "total_tokens不等于prompt_tokens + completion_tokens"


def test_multi_turn_global_cache_pooling(api_url):
    """
    测试多轮对话全局cache池化功能。

    测试流程：
    1. 第一轮请求：发送问题，D实例生成答案并写入全局cache（prompt + output）
    2. 第二轮请求：发送第二轮对话（第一轮Q&A + 追问），P实例应该命中全局cache（包括D实例第一轮的输出）
    """
    # 第一轮问题
    msg1 = (
        "深圳是中国经济实力最强的城市之一。近年来，深圳 GDP 持续稳步增长，"
        "2023 年突破 3.4 万亿元人民币，2024 年接近 3.7 万亿元，长期位居全国城市前列。"
        "深圳经济以第二产业和第三产业为主，高端制造业、电子信息产业和现代服务业发达，"
        "形成了以科技创新为核心的产业结构。依托华为、腾讯、大疆等龙头企业，"
        "深圳在数字经济、人工智能、新能源等领域具有显著优势。同时，深圳进出口总额常年位居全国城市第一，"
        "是中国对外开放和高质量发展的重要引擎。深圳2024年 GDP 是多少？"
    )

    # 第一轮请求
    print("\n>>> Request 1: First round question")
    print("    Purpose: D instance generates output and writes to global cache (prompt + output)")

    payload1 = {
        "model": "default",
        "temperature": 0,
        "top_p": 0,
        "seed": 33,
        "messages": [
            {"role": "user", "content": msg1},
        ],
        "max_tokens": 200,
        "min_tokens": 130,
        "stream": False,
        "collect_metrics": True,
    }

    response1 = send_request(url=api_url, payload=payload1, timeout=120)
    assert response1 is not None, "第一轮请求失败"
    response1_json = response1.json()
    assert "choices" in response1_json, f"第一轮响应格式错误: {response1_json}"
    prompt_tokens_num = response1_json["usage"]["prompt_tokens"]
    # print(f"response1_json: {response1_json}")

    assistant_reply = response1_json["choices"][0]["message"]["content"]
    print(f"First round response: {repr(assistant_reply)}...")
    assert len(assistant_reply) > 0, "第一轮响应为空"

    # 等待D实例将output cache写入全局存储
    print("\n>>> Waiting for D instance to write output cache to global storage...")
    time.sleep(1)

    # 第二轮追问
    msg2 = "那深圳2023年的GDP是多少？和2024年相比增长了多少？"

    print("\n>>> Request 2: Second round (multi-turn conversation)")
    print("    Purpose: P instance should hit global cache including D's output from Request 1")
    print("    Check log_prefill/prefill.log for 'storage_match' to verify cache hit")

    payload2 = {
        "model": "default",
        "temperature": 0,
        "top_p": 0,
        "seed": 33,
        "messages": [
            {"role": "user", "content": msg1},
            {"role": "assistant", "content": assistant_reply},
            {"role": "user", "content": msg2},
        ],
        "max_tokens": 100,
        "stream": False,
        "collect_metrics": True,
    }

    response2 = send_request(url=api_url, payload=payload2, timeout=120)
    assert response2 is not None, "第二轮请求失败"
    response2_json = response2.json()
    assert "choices" in response2_json, f"第二轮响应格式错误: {response2_json}"
    # print(f"response2_json: {response2_json}")
    cached_tokens = response2_json["usage"]["prompt_tokens_details"]["cached_tokens"]

    assistant_reply2 = response2_json["choices"][0]["message"]["content"]
    print(f"\nSecond round response: {repr(assistant_reply2)}")
    assert len(assistant_reply2) > 0, "第二轮响应为空"

    # 校验token命中情况
    print(f"cached_tokens of second round: {cached_tokens}, " f"prompt_tokens_num of first round: {prompt_tokens_num}")
    assert cached_tokens > prompt_tokens_num, "没有从global cache中命中"
