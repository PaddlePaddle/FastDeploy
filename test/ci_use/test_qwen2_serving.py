# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import json
from jsonschema import validate
import concurrent.futures
import numpy as np
import subprocess
import socket
import os
import signal
import sys


# Read ports from environment variables; use default values if not set
FD_API_PORT = int(os.getenv("FD_API_PORT", 8189))
FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8013))
FD_METRICS_PORT = int(os.getenv("FD_METRICS_PORT", 8333))

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
        output = subprocess.check_output(f"lsof -i:{port} -t", shell=True).decode().strip()
        for pid in output.splitlines():
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed process on port {port}, pid={pid}")
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
        model_path=os.path.join(base_path, "Qwen2-7B-Instruct")
    else:
        model_path="./Qwen2-7B-Instruct"

    log_path = "api_server.log"
    cmd = [
        sys.executable, "-m", "fastdeploy.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(FD_API_PORT),
        "--tensor-parallel-size", "1",
        "--engine-worker-queue-port", str(FD_ENGINE_QUEUE_PORT),
        "--metrics-port", str(FD_METRICS_PORT)
    ]

    with open(log_path, "w") as logfile:
        process = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT)

    # Wait up to 120 seconds for API server port to become available
    for _ in range(120):
        if is_port_open("127.0.0.1", FD_API_PORT):
            print(f"API server is up on port {FD_API_PORT}")
            break
        time.sleep(1)
    else:
        process.terminate()
        raise RuntimeError(f"API server did not start on port {FD_API_PORT}")

    yield

    print("Post-test server cleanup...")
    try:
        os.kill(process.pid, signal.SIGTERM)
        print("API server terminated")
    except Exception as e:
        print(f"Failed to kill server: {e}")

    clean_ports()

@pytest.fixture(scope="session")
def api_url(request):
    """
    Returns the API endpoint URL for chat completions.
    """
    return f"http://0.0.0.0:{FD_API_PORT}" + "/v1/chat/completions"


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
        "messages": [{"role": "user", "content": "用一句话介绍 PaddlePaddle"}],
        "temperature": 0.9,
        "top_p": 0,  # fix top_p to reduce randomness
        "seed": 13  # fixed random seed
    }

# ==========================
# JSON Schema for validating chat API responses
# ==========================
chat_response_schema = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "object": {"type": "string"},
        "created": {"type": "number"},
        "model": {"type": "string"},
        "choices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["role", "content"]
                    },
                    "index": {"type": "number"},
                    "finish_reason": {"type": "string"}
                },
                "required": ["message", "index", "finish_reason"]
            }
        }
    },
    "required": ["id", "object", "created", "model", "choices"]
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
# Valid prompt test cases for parameterized testing
# ==========================
valid_prompts = [
    [{"role": "user", "content": "你好"}],
    [{"role": "user", "content": "用一句话介绍 FastDeploy"}],
    [{"role": "user", "content": "今天天气怎么样？"}],
]

@pytest.mark.parametrize("messages", valid_prompts)
def test_valid_chat(messages, api_url, headers):
    """
    Test valid chat requests.
    """
    start = time.time()
    resp = requests.post(api_url, headers=headers, json={"messages": messages})
    duration = time.time() - start

    assert resp.status_code == 200
    validate(instance=resp.json(), schema=chat_response_schema)
    assert duration < 5, "Response too slow：{:.2f}s".format(duration)

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
# Invalid prompt tests
# ==========================

invalid_prompts = [
    [],  # Empty array
    [{}],  # Empty object
    [{"role": "user"}],  # Missing content
    [{"content": "hello"}],  # Missing role
]

@pytest.mark.parametrize("messages", invalid_prompts)
def test_invalid_chat(messages, api_url, headers):
    """
    Test invalid chat inputs
    """
    resp = requests.post(api_url, headers=headers, json={"messages": messages})
    assert resp.status_code >= 400, "Invalid request should return an error status code"


# ==========================
# Test for input exceeding context length
# ==========================

def test_exceed_context_length(api_url, headers):
    """
    Test case for inputs that exceed the model's maximum context length.
    """
    # Construct an overly long message
    long_content = "你好，" * 20000

    messages = [
        {"role": "user", "content": long_content}
    ]

    resp = requests.post(api_url, headers=headers, json={"messages": messages})

    # Check if the response indicates a token limit error or server error (500)
    try:
        response_json = resp.json()
        print("Response JSON content:", json.dumps(response_json, ensure_ascii=False)[:1000])
    except Exception:
        response_json = {}

    # Check status code and response content
    assert resp.status_code != 200 or "token" in json.dumps(response_json).lower(), \
        "Expected token limit error or similar, but got a normal response: {}".format(response_json)

# ==========================
# ChatTemplate Valid Structure Test
# ==========================

chat_template_cases = [
    {"template": "chatml", "messages": [{"role": "user", "content": "你是谁？"}]},
    {"template": "llama", "messages": [{"role": "user", "content": "请自我介绍"}]},
    {"template": "alpaca", "messages": [{"role": "user", "content": "介绍一下 FastDeploy"}]},
]

@pytest.mark.parametrize("payload", chat_template_cases)
def test_chattemplate_valid(payload, api_url, headers):
    """
    Test valid ChatTemplate structures.
    """
    resp = requests.post(api_url, headers=headers, json=payload)
    assert resp.status_code == 200, "Request failed for template={}".format(payload['template'])
    validate(instance=resp.json(), schema=chat_response_schema)

# ==========================
# ChatTemplate Invalid Structure Test
# ==========================

invalid_template_cases = [
    {"template": "nonexist", "messages": [{"role": "user", "content": "你好"}]},
    {"template": 123, "messages": [{"role": "user", "content": "你好"}]},
    {"template": "", "messages": [{"role": "user", "content": "你好"}]},
]


@pytest.mark.parametrize("payload", invalid_template_cases)
@pytest.mark.skip(reason="Validation not yet supported; assertion temporarily disabled")
def test_chattemplate_invalid(payload, api_url, headers):
    """
    Test invalid ChatTemplate structures.
    """
    resp = requests.post(api_url, headers=headers, json=payload)
    assert resp.status_code >= 400, "Invalid template should return an error status code"

# ==========================
# System Role Test
# ==========================

def test_system_role(api_url, headers):
    """
    Test whether the system role can correctly guide model behavior.
    """
    messages = [
        {"role": "system", "content": "You are an English translation assistant."},
        {"role": "user", "content": "Please translate: 你好"},
    ]
    resp = requests.post(api_url, headers=headers, json={"messages": messages})
    assert resp.status_code == 200
    validate(instance=resp.json(), schema=chat_response_schema)
    result = resp.json()["choices"][0]["message"]["content"]
    assert "hello" in result.lower()

# ==========================
# Multi-turn Conversation Test
# ==========================
def test_multi_turn_conversation(api_url, headers):
    """
    Test whether multi-turn conversation context is effective.
    """
    messages = [
        {"role": "user", "content": "你是谁？"},
        {"role": "assistant", "content": "我是AI助手"},
        {"role": "user", "content": "你能做什么？"}
    ]
    resp = requests.post(api_url, headers=headers, json={"messages": messages})
    assert resp.status_code == 200
    validate(instance=resp.json(), schema=chat_response_schema)

# ==========================
# Simple Performance Test
# ==========================

def test_simple_perf(api_url, headers):
    """
    Send 10 requests to check response stability.
    """
    prompts = [{"role": "user", "content": "Introduce FastDeploy."}]
    for _ in range(10):
        resp = requests.post(api_url, headers=headers, json={"messages": prompts})
        assert resp.status_code == 200

# ==========================
# Concurrent Performance Test
# ==========================
@pytest.mark.skip(reason="concurrent is unavailable")
def test_concurrent_perf(api_url, headers):
    """
    Send concurrent requests to test stability and response time.
    """
    prompts = [{"role": "user", "content": "Introduce FastDeploy."}]

    def send_request():
        """
        Send a single request
        """
        resp = requests.post(api_url, headers=headers, json={"messages": prompts})
        assert resp.status_code == 200
        return resp.elapsed.total_seconds()

    with concurrent.futures.ThreadPoolExecutor(max_workers=33) as executor:
        futures = [executor.submit(send_request) for _ in range(33)]
        durations = [f.result() for f in futures]

    print("Response time for each request:", durations)

# ==========================
# Metrics Endpoint Test
# ==========================

def test_metrics_endpoint(metrics_url):
    """
    Test the metrics monitoring endpoint.
    """
    resp = requests.get(metrics_url, timeout=5)

    assert resp.status_code == 200, "Unexpected status code: {}".format(resp.status_code)
    assert "text/plain" in resp.headers["Content-Type"], "Content-Type is not text/plain"

    # Parse Prometheus metrics data
    metrics_data = resp.text
    # print(metrics_data)
    lines = metrics_data.split("\n")

    metric_lines = [line for line in lines if not line.startswith("#") and line.strip() != ""]

    assert len(metric_lines) > 0, "No valid Prometheus metrics found"

    # Assert specific metric values
    num_requests_running_found = False
    num_requests_waiting_found = False
    time_to_first_token_seconds_sum_found = False
    time_per_output_token_seconds_sum_found = False
    e2e_request_latency_seconds_sum_found = False
    request_inference_time_seconds_sum_found = False
    request_queue_time_seconds_sum_found = False

    for line in metric_lines:
        if line.startswith("fastdeploy:num_requests_running"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "Invalid value for num_requests_running"
            num_requests_running_found = True
        elif line.startswith("fastdeploy:num_requests_waiting"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "Invalid value for num_requests_waiting"
            num_requests_waiting_found = True
        elif line.startswith("fastdeploy:time_to_first_token_seconds_sum"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "Invalid value for time_to_first_token_seconds_sum"
            time_to_first_token_seconds_sum_found = True
        elif line.startswith("fastdeploy:time_per_output_token_seconds_sum"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "Invalid value for time_per_output_token_seconds_sum"
            time_per_output_token_seconds_sum_found = True
        elif line.startswith("fastdeploy:e2e_request_latency_seconds_sum"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "Invalid value for e2e_request_latency_seconds_sum"
            e2e_request_latency_seconds_sum_found = True
        elif line.startswith("fastdeploy:request_inference_time_seconds_sum"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "Invalid value for request_inference_time_seconds_sum"
            request_inference_time_seconds_sum_found = True
        elif line.startswith("fastdeploy:request_queue_time_seconds_sum"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "Invalid value for request_queue_time_seconds_sum"
            request_queue_time_seconds_sum_found = True

    assert num_requests_running_found, "Missing metric: fastdeploy:num_requests_running"
    assert num_requests_waiting_found, "Missing metric: fastdeploy:num_requests_waiting"
    assert time_to_first_token_seconds_sum_found, "Missing metric: fastdeploy:time_to_first_token_seconds_sum"
    assert time_per_output_token_seconds_sum_found, "Missing metric: fastdeploy:time_per_output_token_seconds_sum"
    assert e2e_request_latency_seconds_sum_found, "Missing metric: fastdeploy:e2e_request_latency_seconds_sum"
    assert request_inference_time_seconds_sum_found, "Missing metric: fastdeploy:request_inference_time_seconds_sum"
    assert request_queue_time_seconds_sum_found, "Missing metric: fastdeploy:request_queue_time_seconds_sum"