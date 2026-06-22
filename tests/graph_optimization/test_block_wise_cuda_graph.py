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

import os
import shutil
import signal
import subprocess
import sys
import time

import pytest
import requests

tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, tests_dir)

from e2e.utils.serving_utils import (
    FD_API_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    PORTS_TO_CLEAN,
    clean_ports,
    is_port_open,
)


@pytest.fixture(scope="session", autouse=True)
def setup_and_run_server(api_url):
    """
    Pytest fixture that runs once per test session:
    - Cleans ports before tests
    - Starts the API server with block-wise CUDA graph env vars enabled
    - Waits for server port to open (up to 300 seconds)
    - Tears down server after all tests finish
    """
    print("Pre-test port cleanup...")

    ports_to_add = [
        FD_API_PORT + 1,
        FD_METRICS_PORT + 1,
        FD_CACHE_QUEUE_PORT + 1,
        FD_ENGINE_QUEUE_PORT + 1,
    ]

    for port in ports_to_add:
        if port not in PORTS_TO_CLEAN:
            PORTS_TO_CLEAN.append(port)

    clean_ports(PORTS_TO_CLEAN)

    print("log dir clean")
    if os.path.exists("log") and os.path.isdir("log"):
        shutil.rmtree("log")

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "ernie-4_5-21b-a3b-bf16-paddle")
    else:
        model_path = "./ernie-4_5-21b-a3b-bf16-paddle"

    log_path = "server.log"
    cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.multi_api_server",
        "--num-servers",
        "2",
        "--ports",
        f"{FD_API_PORT},{FD_API_PORT + 1}",
        "--metrics-ports",
        f"{FD_METRICS_PORT},{FD_METRICS_PORT + 1}",
        "--args",
        "--model",
        model_path,
        "--engine-worker-queue-port",
        f"{FD_ENGINE_QUEUE_PORT},{FD_ENGINE_QUEUE_PORT + 1}",
        "--cache-queue-port",
        f"{FD_CACHE_QUEUE_PORT},{FD_CACHE_QUEUE_PORT + 1}",
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        "2",
        "--max-model-len",
        "65536",
        "--max-num-seqs",
        "32",
        "--quantization",
        "block_wise_fp8",
        "--max-num-batched-tokens",
        "128",
    ]

    # Build env with block-wise CUDA graph enabled
    env = os.environ.copy()
    env["FD_USE_BLOCK_WISE_CUDA_GRAPH"] = "1"
    env["FD_BLOCK_WISE_CUDA_GRAPH_SIZES"] = "128"
    env["FD_USE_PHI_FP8_QUANT"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["FD_BLOCK_WISE_DEBUG"] = "1"

    if os.path.exists("log"):
        shutil.rmtree("log")

    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    # Wait up to 300 seconds for API server to be ready
    for _ in range(300):
        if is_port_open("127.0.0.1", FD_API_PORT):
            print(f"Server is up on port {FD_API_PORT}")
            break
        time.sleep(1)
    else:
        print("[TIMEOUT] API server failed to start in 5 minutes. Cleaning up...")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception as e:
            print(f"Failed to kill process group: {e}")
        raise RuntimeError(f"API server did not start on port {FD_API_PORT}")

    yield  # Run tests

    print("\n===== Post-test server cleanup... =====")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        time.sleep(10)
        print(f"server (pid={process.pid}) terminated")
    except Exception as e:
        print(f"Failed to terminate API server: {e}")

    clean_ports(PORTS_TO_CLEAN)


@pytest.fixture(scope="session")
def api_url(request):
    """
    Returns the API endpoint URL for chat completions.
    """
    return f"http://0.0.0.0:{FD_API_PORT}/v1/chat/completions"


@pytest.fixture
def headers():
    """
    Returns common HTTP request headers.
    """
    return {"Content-Type": "application/json"}


def send_request(url, payload, timeout=60):
    """
    Send a POST request to the specified URL with the given payload.
    """
    headers = {"Content-Type": "application/json"}
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=timeout)
        return res
    except requests.exceptions.Timeout:
        print(f"Request timed out (>{timeout} seconds)")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def test_block_wise_cuda_graph_beijing(api_url):
    """
    Verify that block-wise CUDA graph produces correct output.

    With FD_USE_BLOCK_WISE_CUDA_GRAPH=1 and FD_BLOCK_WISE_CUDA_GRAPH_SIZES set,
    ask about Tiananmen Square in Beijing and verify the response mentions "北京".
    """
    payload = {
        "stream": False,
        "messages": [{"role": "user", "content": "北京天安门在哪里"}],
        "max_tokens": 128,
    }

    response = send_request(url=api_url, payload=payload)
    print("response: ", response)
    assert response is not None, "Request returned None (timeout or connection error)"
    assert response.status_code == 200, f"Request failed with status {response.status_code}: {response.text}"
