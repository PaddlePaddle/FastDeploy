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
    clean_ports,
    is_port_open,
)

# FD_API_PORT = "8765"
# FD_CACHE_QUEUE_PORT = "8766"
# FD_ENGINE_QUEUE_PORT = "8767"
# FD_METRICS_PORT = "8768"

os.environ["FD_USE_MACHETE"] = "0"
os.environ["FD_MOE_MXFP4_BACKEND"] = "flashinfer"


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
    clean_ports([FD_API_PORT, FD_CACHE_QUEUE_PORT, FD_ENGINE_QUEUE_PORT, FD_METRICS_PORT])
    print("log dir clean ")
    if os.path.exists("log") and os.path.isdir("log"):
        shutil.rmtree("log")

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "gpt-oss-20b")
    else:
        model_path = "./gpt-oss-20b"

    log_path = "server.log"

    cmd = [
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
        "16384",
        "--max-num-batched-tokens",
        "16384",
        "--max-num-seqs",
        "128",
        "--gpu-memory-utilization",
        "0.9",
        "--load-choices",
        "default_v1",
        "--tensor-parallel-size",
        "2",
    ]

    # Start subprocess in new process group
    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Enables killing full group via os.killpg
        )

    # Wait up to 10 minutes for API server to be ready
    for _ in range(10 * 60):
        if is_port_open("127.0.0.1", FD_API_PORT):
            print(f"API server is up on port {FD_API_PORT}")
            break
        time.sleep(1)
    else:
        print("[TIMEOUT] API server failed to start in 10 minutes. Cleaning up...")
        print("==test output start==")
        # 输出测试日志
        with open(log_path, "r") as f:
            for line in f:
                print(line, end="")
        print("==test output end==")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception as e:
            print(f"Failed to kill process group: {e}")
        raise RuntimeError(f"API server did not start on port {FD_API_PORT}")

    yield  # Run tests

    print("\n===== Post-test server cleanup... =====")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        print(f"API server (pid={process.pid}) terminated")
        clean_ports([FD_API_PORT, FD_CACHE_QUEUE_PORT, FD_ENGINE_QUEUE_PORT, FD_METRICS_PORT])
    except Exception as e:
        print(f"Failed to terminate API server: {e}")


@pytest.fixture(scope="session")
def api_url(request):
    """
    Returns the API endpoint URL for chat completions.
    """
    return f"http://0.0.0.0:{FD_API_PORT}/v1/chat/completions"


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


@pytest.fixture
def consistent_payload():
    """
    Returns a fixed payload for consistency testing,
    including a fixed random seed and temperature.
    """
    return {
        "messages": [{"role": "user", "content": "介绍一下你自己"}],
        "temperature": 0.8,
        "top_p": 0,  # fix top_p to reduce randomness
        "seed": 13,  # fixed random seed
        "max_tokens": 10,
    }


# def test_payload():
#     return {"messages": [{"role": "user", "content": "hello"}], "temperature": 0.9, "max_tokens": 100}


# ==========================
# Helper function to calculate difference rate between two texts
# ==========================
def calculate_diff_rate(text1, text2):
    """
    Calculate the difference rate between two strings
    based on the normalized Levenshtein edit distance.
    Returns a float in [0,1], where 0 means identical.
    """
    if text1 == text2:
        return 0.0

    len1, len2 = len(text1), len(text2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        for j in range(len2 + 1):
            if i == 0 or j == 0:
                dp[i][j] = i + j
            elif text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    edit_distance = dp[len1][len2]
    max_len = max(len1, len2)
    return edit_distance / max_len if max_len > 0 else 0.0


# ==========================
# Consistency test for repeated runs with fixed payload
# ==========================
def test_request(api_url, headers, consistent_payload):
    """
    Test that two runs with the same fixed input produce similar outputs.
    """
    # First request
    resp1 = requests.post(api_url, headers=headers, json=consistent_payload)
    assert resp1.status_code == 200
    result1 = resp1.json()
    content1 = result1["choices"][0]["message"]["content"]

    print(content1)
    # assert "chat" in content1.lower()
