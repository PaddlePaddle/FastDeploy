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

# Test splitwise deployment: use local_scheduler + router,
# set ENABLE_V1_KVCACHE_SCHEDULER is 1, use rdma to transfer cache,
# the tp_size of prefill is 2 and the tp_size of decode is 1.

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
)

# Read ports from environment variables; use default values if not set
FD_CONNECTOR_PORT = int(os.getenv("FD_CONNECTOR_PORT", 8433))
FD_ROUTER_PORT = int(os.getenv("FD_ROUTER_PORT", 8533))
FD_RDMA_PORT = int(os.getenv("FD_RDMA_PORT", 8623))

# List of ports to clean before and after tests
PORTS_TO_CLEAN = [
    FD_API_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_CONNECTOR_PORT,
    FD_RDMA_PORT,
    FD_RDMA_PORT + 1,
    FD_API_PORT + 1,
    FD_ENGINE_QUEUE_PORT + 1,
    FD_METRICS_PORT + 1,
    FD_CACHE_QUEUE_PORT + 1,
    FD_CONNECTOR_PORT + 1,
    FD_RDMA_PORT + 2,
    FD_ROUTER_PORT,
]


@pytest.fixture(scope="session", autouse=True)
def setup_and_run_server():
    """
    Pytest fixture that runs once per test session:
    - Cleans ports before tests
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

    # router
    print("start router...")
    env_router = os.environ.copy()
    env_router["FD_LOG_DIR"] = "log_router"
    router_log_path = "router.log"

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

    # prefill实例
    print("start prefill...")
    env_prefill = os.environ.copy()
    env_prefill["CUDA_VISIBLE_DEVICES"] = "0,1"
    env_prefill["FD_LOG_DIR"] = "log_prefill"
    env_prefill["KVCACHE_RDMA_NICS"] = rdma_nics

    prefill_log_path = "prefill.log"
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
        "--tensor-parallel-size",
        "2",
        "--splitwise-role",
        "prefill",
        "--cache-transfer-protocol",
        "rdma",
        "--rdma-comm-ports",
        f"{FD_RDMA_PORT},{FD_RDMA_PORT+1}",
        "--pd-comm-port",
        str(FD_CONNECTOR_PORT),
        "--router",
        f"0.0.0.0:{FD_ROUTER_PORT}",
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

    # decode实例
    print("start decode...")
    env_decode = os.environ.copy()
    env_decode["CUDA_VISIBLE_DEVICES"] = "1"
    env_decode["FD_LOG_DIR"] = "log_decode"
    env_decode["KVCACHE_RDMA_NICS"] = rdma_nics

    decode_log_path = "decode.log"
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
        str(FD_RDMA_PORT + 2),
        "--pd-comm-port",
        str(FD_CONNECTOR_PORT + 1),
        "--router",
        f"0.0.0.0:{FD_ROUTER_PORT}",
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
        os.killpg(process_router.pid, signal.SIGTERM)
        os.killpg(process_prefill.pid, signal.SIGTERM)
        os.killpg(process_decode.pid, signal.SIGTERM)
        clean(PORTS_TO_CLEAN)
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
    print("Decode Response:", result)
    assert result != "", "结果为空"
    usage = chunks[-1]["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"], "completion_tokens大于max_tokens"
    assert payload["metadata"]["min_tokens"] <= usage["completion_tokens"], "completion_tokens小于min_tokens"
    assert usage["total_tokens"] == total_tokens, "total_tokens不等于prompt_tokens + completion_tokens"


def test_chat_usage_non_stream(api_url):
    """测试非流式chat usage"""
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
        "stream": False,
        "metadata": {"min_tokens": 10},
    }

    response = send_request(url=api_url, payload=payload).json()
    usage = response["usage"]
    result = response["choices"][0]["message"]["content"]
    assert result != "", "结果为空"
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"], "completion_tokens大于max_tokens"
    assert payload["metadata"]["min_tokens"] <= usage["completion_tokens"], "completion_tokens小于min_tokens"
    assert usage["total_tokens"] == total_tokens, "total_tokens不等于prompt_tokens + completion_tokens"


def test_non_chat_usage_stream(api_url):
    """测试流式非chat usage"""
    payload = {
        "model": "default",
        "temperature": 0,
        "top_p": 0,
        "seed": 33,
        "prompt": "牛顿的三大运动定律是什么？",
        "max_tokens": 50,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "metadata": {"min_tokens": 10},
    }
    api_url = api_url.replace("chat/completions", "completions")

    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)
    result = "".join([x["choices"][0]["text"] for x in chunks[:-1]])
    print("Decode Response:", result)
    assert result != "", "结果为空"
    usage = chunks[-1]["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"], "completion_tokens大于max_tokens"
    assert payload["metadata"]["min_tokens"] <= usage["completion_tokens"], "completion_tokens小于min_tokens"
    assert usage["total_tokens"] == total_tokens, "total_tokens不等于prompt_tokens + completion_tokens"


def test_non_chat_usage_non_stream(api_url):
    """测试非流式非chat usage"""
    payload = {
        "model": "default",
        "temperature": 0,
        "top_p": 0,
        "seed": 33,
        "prompt": "牛顿的三大运动定律是什么？",
        "max_tokens": 50,
        "stream": False,
        "metadata": {"min_tokens": 10},
    }
    api_url = api_url.replace("chat/completions", "completions")

    response = send_request(url=api_url, payload=payload).json()
    usage = response["usage"]
    result = response["choices"][0]["text"]
    print("Decode Response:", result)
    assert result != "", "结果为空"
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"], "completion_tokens大于max_tokens"
    assert payload["metadata"]["min_tokens"] <= usage["completion_tokens"], "completion_tokens小于min_tokens"
    assert usage["total_tokens"] == total_tokens, "total_tokens不等于prompt_tokens + completion_tokens"
