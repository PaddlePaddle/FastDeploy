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

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

import pytest
import requests

# Read ports from environment variables; use default values if not set
FD_API_PORT = int(os.getenv("FD_API_PORT", 8188))
FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8133))
FD_METRICS_PORT = int(os.getenv("FD_METRICS_PORT", 8233))
FD_CACHE_QUEUE_PORT = int(os.getenv("FD_CACHE_QUEUE_PORT", 8333))
FD_CONNECTOR_PORT = int(os.getenv("FD_CONNECTOR_PORT", 8433))
FD_ROUTER_PORT = int(os.getenv("FD_ROUTER_PORT", 8533))

# List of ports to clean before and after tests
PORTS_TO_CLEAN = [
    FD_API_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_CONNECTOR_PORT,
    FD_API_PORT + 1,
    FD_ENGINE_QUEUE_PORT + 1,
    FD_METRICS_PORT + 1,
    FD_CACHE_QUEUE_PORT + 1,
    FD_CONNECTOR_PORT + 1,
    FD_ROUTER_PORT,
]


def is_port_open(host: str, port: int, timeout=1.0):
    """
    Check if a TCP port is open on the given host.
    Returns True if connection succeeds, False otherwise.
    """
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except Exception:
        return False


def check_service_health(base_url: str, timeout: int = 3) -> bool:
    """
    Check the health status of a service.

    Args:
        base_url (str): The base URL of the service, e.g. "http://127.0.0.1:8080"
        timeout (int): Request timeout in seconds.

    Returns:
        bool: True if the service is healthy, False otherwise.
    """
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"
    url = f"{base_url.rstrip('/')}/health"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return True
        else:
            return False
    except Exception:
        return False


def get_registered_number(router_url) -> list:
    """
    Get the number of registered models in the router.

    Args:
        router_url (str): The base URL of the router, e.g. "http://localhost:8080".

    Returns:
        int: The number of registered models.
    """
    if not router_url.startswith("http"):
        router_url = f"http://{router_url}"

    try:
        response = requests.get(f"{router_url}/registered_number", timeout=60)
        registered_numbers = response.json()
        return registered_numbers
    except Exception:
        return {"mixed": 0, "prefill": 0, "decode": 0}


def kill_process_on_port(port: int):
    """
    Kill processes that are listening on the given port.
    Uses multiple methods to ensure thorough cleanup.
    """
    current_pid = os.getpid()
    parent_pid = os.getppid()

    # Method 1: Use lsof to find processes
    try:
        output = subprocess.check_output(f"lsof -i:{port} -t", shell=True).decode().strip()
        for pid in output.splitlines():
            pid = int(pid)
            if pid in (current_pid, parent_pid):
                print(f"Skip killing current process (pid={pid}) on port {port}")
                continue
            try:
                # First try SIGTERM for graceful shutdown
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                # Then SIGKILL if still running
                os.kill(pid, signal.SIGKILL)
                print(f"Killed process on port {port}, pid={pid}")
            except ProcessLookupError:
                pass  # Process already terminated
    except subprocess.CalledProcessError:
        pass

    # Method 2: Use netstat and fuser as backup
    try:
        # Find processes using netstat and awk
        cmd = f"netstat -tulpn 2>/dev/null | grep :{port} | awk '{{print $7}}' | cut -d'/' -f1"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        for pid in output.splitlines():
            if pid and pid.isdigit():
                pid = int(pid)
                if pid in (current_pid, parent_pid):
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"Killed process (netstat) on port {port}, pid={pid}")
                except ProcessLookupError:
                    pass
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Method 3: Use fuser if available
    try:
        subprocess.run(f"fuser -k {port}/tcp", shell=True, timeout=5)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass


def clean_ports():
    """
    Kill all processes occupying the ports listed in PORTS_TO_CLEAN.
    """
    print(f"Cleaning ports: {PORTS_TO_CLEAN}")
    for port in PORTS_TO_CLEAN:
        kill_process_on_port(port)

    # Double check and retry if ports are still in use
    time.sleep(2)
    for port in PORTS_TO_CLEAN:
        if is_port_open("127.0.0.1", port, timeout=0.1):
            print(f"Port {port} still in use, retrying cleanup...")
            kill_process_on_port(port)
            time.sleep(1)


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
    clean_ports()

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
    env_prefill["CUDA_VISIBLE_DEVICES"] = "0"
    env_prefill["ENABLE_V1_KVCACHE_SCHEDULER"] = "0"
    env_prefill["FD_LOG_DIR"] = "log_prefill"
    env_prefill["INFERENCE_MSG_QUEUE_ID"] = str(FD_API_PORT)
    prefill_log_path = "server.log"
    prefill_cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(FD_API_PORT),
        "--tensor-parallel-size",
        "1",
        "--engine-worker-queue-port",
        str(FD_ENGINE_QUEUE_PORT),
        "--metrics-port",
        str(FD_METRICS_PORT),
        "--cache-queue-port",
        str(FD_CACHE_QUEUE_PORT),
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "20",
        "--quantization",
        "wint8",
        "--splitwise-role",
        "prefill",
        "--cache-transfer-protocol",
        "ipc",
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
    env_decode["ENABLE_V1_KVCACHE_SCHEDULER"] = "0"
    env_decode["INFERENCE_MSG_QUEUE_ID"] = str(FD_API_PORT + 1)
    env_decode["FD_LOG_DIR"] = "log_decode"
    decode_log_path = "decode_server.log"
    decode_cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(FD_API_PORT + 1),
        "--tensor-parallel-size",
        "1",
        "--engine-worker-queue-port",
        str(FD_ENGINE_QUEUE_PORT + 1),
        "--metrics-port",
        str(FD_METRICS_PORT + 1),
        "--cache-queue-port",
        str(FD_CACHE_QUEUE_PORT + 1),
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "20",
        "--quantization",
        "wint8",
        "--splitwise-role",
        "decode",
        "--cache-transfer-protocol",
        "ipc",
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
            os.killpg(process_prefill.pid, signal.SIGTERM)
            os.killpg(process_decode.pid, signal.SIGTERM)
            clean_ports()
        except Exception as e:
            print(f"Failed to kill process group: {e}")
        raise RuntimeError(f"API server did not start on port {FD_API_PORT}")

    yield  # Run tests

    print("\n===== Post-test server cleanup... =====")
    try:
        os.killpg(process_router.pid, signal.SIGTERM)
        os.killpg(process_prefill.pid, signal.SIGTERM)
        os.killpg(process_decode.pid, signal.SIGTERM)
        clean_ports()
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


def send_request(url, payload, timeout=600):
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
