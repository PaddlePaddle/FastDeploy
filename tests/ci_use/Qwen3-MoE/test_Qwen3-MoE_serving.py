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
import re
import signal
import subprocess
import sys
import time

import pytest
import requests

tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, tests_dir)

from e2e.utils.serving_utils import (
    FD_API_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    clean_ports,
    is_port_open,
)


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
        model_path = os.path.join(base_path, "Qwen3-30B-A3B")
    else:
        model_path = "./Qwen3-30B-A3B"

    log_path = "server.log"
    cmd = [
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
        "32768",
        "--max-num-seqs",
        "50",
        "--quantization",
        "wint4",
    ]

    # Start subprocess in new process group
    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Enables killing full group via os.killpg
        )

    # Wait up to 300 seconds for API server to be ready
    for _ in range(480):
        if is_port_open("127.0.0.1", FD_API_PORT):
            print(f"API server is up on port {FD_API_PORT}")
            break
        time.sleep(1)
    else:
        print("API server failed to start in time. Cleaning up...")
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
        "messages": [
            {
                "role": "user",
                "content": "用一句话介绍 PaddlePaddle, 30字以内 /no_think",
            }
        ],
        "temperature": 0.8,
        "top_p": 0,  # fix top_p to reduce randomness
        "seed": 13,  # fixed random seed
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
    assert diff_rate < 0.05, f"Output difference too large ({diff_rate:.4%})"


# ==========================
# think Prompt Test
# ==========================


def test_thinking_prompt(api_url, headers):
    """
    Test case to verify normal 'thinking' behavior (no '/no_think' appended).
    """
    messages = [{"role": "user", "content": "北京天安门在哪里"}]

    payload = {
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.8,
        "top_p": 0.01,
    }

    resp = requests.post(api_url, headers=headers, json=payload)
    assert resp.status_code == 200, f"Unexpected status code: {resp.status_code}"

    try:
        response_json = resp.json()
    except Exception as e:
        assert False, f"Response is not valid JSON: {e}"

    content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
    assert "天安门" in content or "北京" in content, "Expected a location-related response with reasoning"


# ==========================
# no_think Prompt Test
# ==========================


def test_non_thinking_prompt(api_url, headers):
    """
    Test case to verify non-thinking behavior (with '/no_think').
    """
    messages = [{"role": "user", "content": "北京天安门在哪里 /no_think"}]

    payload = {
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.8,
        "top_p": 0.01,
    }

    resp = requests.post(api_url, headers=headers, json=payload)
    assert resp.status_code == 200, f"Unexpected status code: {resp.status_code}"

    try:
        response_json = resp.json()
    except Exception as e:
        assert False, f"Response is not valid JSON: {e}"

    content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
    assert not any(
        x in content for x in ["根据", "我认为", "推测", "可能"]
    ), "Expected no reasoning in non-thinking response"


def test_profile_reset_block_num():
    """测试profile reset_block_num功能，与baseline diff不能超过5%"""
    log_file = "./log/config.log"
    baseline = 17864

    if not os.path.exists(log_file):
        pytest.fail(f"Log file not found: {log_file}")

    with open(log_file, "r") as f:
        log_lines = f.readlines()

    target_line = None
    for line in log_lines:
        if "Reset block num" in line:
            target_line = line.strip()
            break

    if target_line is None:
        pytest.fail("日志中没有Reset block num信息")

    match = re.search(r"total_block_num:(\d+)", target_line)
    if not match:
        pytest.fail(f"Failed to extract total_block_num from line: {target_line}")

    try:
        actual_value = int(match.group(1))
    except ValueError:
        pytest.fail(f"Invalid number format: {match.group(1)}")

    lower_bound = baseline * (1 - 0.05)
    upper_bound = baseline * (1 + 0.05)
    print(f"Reset total_block_num: {actual_value}. baseline: {baseline}")

    assert lower_bound <= actual_value <= upper_bound, (
        f"Reset total_block_num {actual_value} 与 baseline {baseline} diff需要在5%以内"
        f"Allowed range: [{lower_bound:.1f}, {upper_bound:.1f}]"
    )
