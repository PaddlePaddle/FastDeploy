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

import pytest
import requests
import time
import subprocess
import socket
import os
import signal
import sys


# Read ports from environment variables; use default values if not set
FD_API_PORT = int(os.getenv("FD_API_PORT", 8188))
FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8133))
FD_METRICS_PORT = int(os.getenv("FD_METRICS_PORT", 8233))

# List of ports to clean before and after tests
PORTS_TO_CLEAN = [FD_API_PORT, FD_ENGINE_QUEUE_PORT, FD_METRICS_PORT]

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

def kill_process_on_port(port: int):
    """
    Kill processes that are listening on the given port.
    Uses `lsof` to find process ids and sends SIGKILL.
    """
    try:
        output = subprocess.check_output("lsof -i:{} -t".format(port), shell=True).decode().strip()
        for pid in output.splitlines():
            os.kill(int(pid), signal.SIGKILL)
            print("Killed process on port {}, pid={}".format(port, pid))
    except subprocess.CalledProcessError:
        pass

def clean_ports():
    """
    Kill all processes occupying the ports listed in PORTS_TO_CLEAN.
    """
    for port in PORTS_TO_CLEAN:
        kill_process_on_port(port)

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

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path=os.path.join(base_path, "Qwen3-30B-A3B")
    else:
        model_path="./Qwen3-30B-A3B"

    log_path = "server.log"
    cmd = [
        sys.executable, "-m", "fastdeploy.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(FD_API_PORT),
        "--tensor-parallel-size", "1",
        "--engine-worker-queue-port", str(FD_ENGINE_QUEUE_PORT),
        "--metrics-port", str(FD_METRICS_PORT),
        "--max-model-len", "32768",
        "--max-num-seqs", "50",
        "--quantization", "wint4"
    ]

    # Set environment variables
    env = os.environ.copy()
    env["ENABLE_FASTDEPLOY_LOAD_MODEL_CONCURRENCY"] = "0"
    env["NCCL_ALGO"] = "Ring"
    env["FLAG_SAMPLING_CLASS"] = "rejection"

    # Start subprocess in new process group
    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True  # Enables killing full group via os.killpg
        )

    # Wait up to 300 seconds for API server to be ready
    for _ in range(300):
        if is_port_open("127.0.0.1", FD_API_PORT):
            print("API server is up on port {}".format(FD_API_PORT))
            break
        time.sleep(1)
    else:
        print("API server failed to start in time. Cleaning up...")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception as e:
            print("Failed to kill process group: {}".format(e))
        raise RuntimeError("API server did not start on port {}".format(FD_API_PORT))

    yield  # Run tests

    print("\n===== Post-test server cleanup... =====")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        print("API server (pid={}) terminated".format(process.pid))
    except Exception as e:
        print("Failed to terminate API server: {}".format(e))


@pytest.fixture(scope="session")
def api_url(request):
    """
    Returns the API endpoint URL for chat completions.
    """
    return "http://0.0.0.0:{}/v1/chat/completions".format(FD_API_PORT)


@pytest.fixture(scope="session")
def metrics_url(request):
    """
    Returns the metrics endpoint URL.
    """
    return "http://0.0.0.0:{}/metrics".format(FD_METRICS_PORT)


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
        "messages": [{"role": "user", "content": "用一句话介绍 PaddlePaddle, 30字以内 /no_think"}],
        "temperature": 0.8,
        "top_p": 0,  # fix top_p to reduce randomness
        "seed": 13  # fixed random seed
    }

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
def test_consistency_between_runs(api_url, headers, consistent_payload):
    """
    Test that two runs with the same fixed input produce similar outputs.
    """
    # First request
    resp1 = requests.post(api_url, headers=headers, json=consistent_payload)
    assert resp1.status_code == 200
    result1 = resp1.json()
    content1 = result1["choices"][0]["message"]["content"]

    # Second request
    resp2 = requests.post(api_url, headers=headers, json=consistent_payload)
    assert resp2.status_code == 200
    result2 = resp2.json()
    content2 = result2["choices"][0]["message"]["content"]

    # Calculate difference rate
    diff_rate = calculate_diff_rate(content1, content2)

    # Verify that the difference rate is below the threshold
    assert diff_rate < 0.05, "Output difference too large ({:.4%})".format(diff_rate)

# ==========================
# think Prompt Test
# ==========================

def test_thinking_prompt(api_url, headers):
    """
    Test case to verify normal 'thinking' behavior (no '/no_think' appended).
    """
    messages = [
        {"role": "user", "content": "北京天安门在哪里"}
    ]

    payload = {
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.8,
        "top_p": 0.01
    }

    resp = requests.post(api_url, headers=headers, json=payload)
    assert resp.status_code == 200, "Unexpected status code: {}".format(resp.status_code)

    try:
        response_json = resp.json()
    except Exception as e:
        assert False, "Response is not valid JSON: {}".format(e)
    
    content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
    assert "天安门" in content or "北京" in content, "Expected a location-related response with reasoning"

# ==========================
# no_think Prompt Test
# ==========================

def test_non_thinking_prompt(api_url, headers):
    """
    Test case to verify non-thinking behavior (with '/no_think').
    """
    messages = [
        {"role": "user", "content": "北京天安门在哪里 /no_think"}
    ]

    payload = {
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.8,
        "top_p": 0.01
    }

    resp = requests.post(api_url, headers=headers, json=payload)
    assert resp.status_code == 200, "Unexpected status code: {}".format(resp.status_code)

    try:
        response_json = resp.json()
    except Exception as e:
        assert False, "Response is not valid JSON: {}".format(e)

    content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
    assert not any(x in content for x in ["根据", "我认为", "推测", "可能"]), \
        "Expected no reasoning in non-thinking response"