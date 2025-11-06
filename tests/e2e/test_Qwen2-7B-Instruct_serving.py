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

import concurrent.futures
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

import openai
import pytest
import requests
from jsonschema import validate
from utils.serving_utils import (
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

    print("log dir clean ")
    if os.path.exists("log") and os.path.isdir("log"):
        shutil.rmtree("log")

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "Qwen2-7B-Instruct")
    else:
        model_path = "./Qwen2-7B-Instruct"

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
        "128",
        "--quantization",
        "wint8",
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
    for _ in range(300):
        if is_port_open("127.0.0.1", FD_API_PORT):
            print(f"API server is up on port {FD_API_PORT}")
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
        clean_ports()
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
        "messages": [{"role": "user", "content": "用一句话介绍 PaddlePaddle"}],
        "temperature": 0.9,
        "top_p": 0,  # fix top_p to reduce randomness
        "seed": 13,  # fixed random seed
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
                        "required": ["role", "content"],
                    },
                    "index": {"type": "number"},
                    "finish_reason": {"type": "string"},
                },
                "required": ["message", "index", "finish_reason"],
            },
        },
    },
    "required": ["id", "object", "created", "model", "choices"],
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
]


@pytest.mark.parametrize("messages", valid_prompts)
def test_valid_chat(messages, api_url, headers):
    """
    Test valid chat requests.
    """
    resp = requests.post(api_url, headers=headers, json={"messages": messages})

    assert resp.status_code == 200
    validate(instance=resp.json(), schema=chat_response_schema)


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

    messages = [{"role": "user", "content": long_content}]

    resp = requests.post(api_url, headers=headers, json={"messages": messages})

    # Check if the response indicates a token limit error or server error (500)
    try:
        response_json = resp.json()
    except Exception:
        response_json = {}

    # Check status code and response content
    assert (
        resp.status_code != 200 or "token" in json.dumps(response_json).lower()
    ), f"Expected token limit error or similar, but got a normal response: {response_json}"


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
        {"role": "user", "content": "你能做什么？"},
    ]
    resp = requests.post(api_url, headers=headers, json={"messages": messages})
    assert resp.status_code == 200
    validate(instance=resp.json(), schema=chat_response_schema)


# ==========================
# Concurrent Performance Test
# ==========================
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(send_request) for _ in range(8)]
        durations = [f.result() for f in futures]

    print("\nResponse time for each request:", durations)


# ==========================
# Metrics Endpoint Test
# ==========================


def test_metrics_endpoint(metrics_url):
    """
    Test the metrics monitoring endpoint.
    """
    resp = requests.get(metrics_url, timeout=5)

    assert resp.status_code == 200, f"Unexpected status code: {resp.status_code}"
    assert "text/plain" in resp.headers["Content-Type"], "Content-Type is not text/plain"

    # Parse Prometheus metrics data
    metrics_data = resp.text
    lines = metrics_data.split("\n")

    metric_lines = [line for line in lines if not line.startswith("#") and line.strip() != ""]

    # 断言 具体值
    num_requests_running_found = False
    num_requests_waiting_found = False
    time_to_first_token_seconds_sum_found = False
    time_per_output_token_seconds_sum_found = False
    e2e_request_latency_seconds_sum_found = False
    request_inference_time_seconds_sum_found = False
    request_queue_time_seconds_sum_found = False
    request_prefill_time_seconds_sum_found = False
    request_decode_time_seconds_sum_found = False
    prompt_tokens_total_found = False
    generation_tokens_total_found = False
    request_prompt_tokens_sum_found = False
    request_generation_tokens_sum_found = False
    gpu_cache_usage_perc_found = False
    request_params_max_tokens_sum_found = False
    request_success_total_found = False
    cache_config_info_found = False
    available_batch_size_found = False
    hit_req_rate_found = False
    hit_token_rate_found = False
    cpu_hit_token_rate_found = False
    gpu_hit_token_rate_found = False

    for line in metric_lines:
        if line.startswith("fastdeploy:num_requests_running"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "num_requests_running 值错误"
            num_requests_running_found = True
        elif line.startswith("fastdeploy:num_requests_waiting"):
            _, value = line.rsplit(" ", 1)
            num_requests_waiting_found = True
            assert float(value) >= 0, "num_requests_waiting 值错误"
        elif line.startswith("fastdeploy:time_to_first_token_seconds_sum"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "time_to_first_token_seconds_sum 值错误"
            time_to_first_token_seconds_sum_found = True
        elif line.startswith("fastdeploy:time_per_output_token_seconds_sum"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "time_per_output_token_seconds_sum 值错误"
            time_per_output_token_seconds_sum_found = True
        elif line.startswith("fastdeploy:e2e_request_latency_seconds_sum"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "e2e_request_latency_seconds_sum_found 值错误"
            e2e_request_latency_seconds_sum_found = True
        elif line.startswith("fastdeploy:request_inference_time_seconds_sum"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "request_inference_time_seconds_sum 值错误"
            request_inference_time_seconds_sum_found = True
        elif line.startswith("fastdeploy:request_queue_time_seconds_sum"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "request_queue_time_seconds_sum 值错误"
            request_queue_time_seconds_sum_found = True
        elif line.startswith("fastdeploy:request_prefill_time_seconds_sum"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "request_prefill_time_seconds_sum 值错误"
            request_prefill_time_seconds_sum_found = True
        elif line.startswith("fastdeploy:request_decode_time_seconds_sum"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "request_decode_time_seconds_sum 值错误"
            request_decode_time_seconds_sum_found = True
        elif line.startswith("fastdeploy:prompt_tokens_total"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "prompt_tokens_total 值错误"
            prompt_tokens_total_found = True
        elif line.startswith("fastdeploy:generation_tokens_total"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "generation_tokens_total 值错误"
            generation_tokens_total_found = True
        elif line.startswith("fastdeploy:request_prompt_tokens_sum"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "request_prompt_tokens_sum 值错误"
            request_prompt_tokens_sum_found = True
        elif line.startswith("fastdeploy:request_generation_tokens_sum"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "request_generation_tokens_sum 值错误"
            request_generation_tokens_sum_found = True
        elif line.startswith("fastdeploy:gpu_cache_usage_perc"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "gpu_cache_usage_perc 值错误"
            gpu_cache_usage_perc_found = True
        elif line.startswith("fastdeploy:request_params_max_tokens_sum"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "request_params_max_tokens_sum 值错误"
            request_params_max_tokens_sum_found = True
        elif line.startswith("fastdeploy:request_success_total"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "request_success_total 值错误"
            request_success_total_found = True
        elif line.startswith("fastdeploy:cache_config_info"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "cache_config_info 值错误"
            cache_config_info_found = True
        elif line.startswith("fastdeploy:available_batch_size"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "available_batch_size 值错误"
            available_batch_size_found = True
        elif line.startswith("fastdeploy:hit_req_rate"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "hit_req_rate 值错误"
            hit_req_rate_found = True
        elif line.startswith("fastdeploy:hit_token_rate"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "hit_token_rate 值错误"
            hit_token_rate_found = True
        elif line.startswith("fastdeploy:cpu_hit_token_rate"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "cpu_hit_token_rate 值错误"
            cpu_hit_token_rate_found = True
        elif line.startswith("fastdeploy:gpu_hit_token_rate"):
            _, value = line.rsplit(" ", 1)
            assert float(value) >= 0, "gpu_hit_token_rate 值错误"
            gpu_hit_token_rate_found = True
    assert num_requests_running_found, "缺少 fastdeploy:num_requests_running 指标"
    assert num_requests_waiting_found, "缺少 fastdeploy:num_requests_waiting 指标"
    assert time_to_first_token_seconds_sum_found, "缺少 fastdeploy:time_to_first_token_seconds_sum 指标"
    assert time_per_output_token_seconds_sum_found, "缺少 fastdeploy:time_per_output_token_seconds_sum 指标"
    assert e2e_request_latency_seconds_sum_found, "缺少 fastdeploy:e2e_request_latency_seconds_sum_found 指标"
    assert request_inference_time_seconds_sum_found, "缺少 fastdeploy:request_inference_time_seconds_sum 指标"
    assert request_queue_time_seconds_sum_found, "缺少 fastdeploy:request_queue_time_seconds_sum 指标"
    assert request_prefill_time_seconds_sum_found, "缺少 fastdeploy:request_prefill_time_seconds_sum 指标"
    assert request_decode_time_seconds_sum_found, "缺少 fastdeploy:request_decode_time_seconds_sum 指标"
    assert prompt_tokens_total_found, "缺少 fastdeploy:prompt_tokens_total 指标"
    assert generation_tokens_total_found, "缺少 fastdeploy:generation_tokens_total 指标"
    assert request_prompt_tokens_sum_found, "缺少 fastdeploy:request_prompt_tokens_sum 指标"
    assert request_generation_tokens_sum_found, "缺少 fastdeploy:request_generation_tokens_sum 指标"
    assert gpu_cache_usage_perc_found, "缺少 fastdeploy:gpu_cache_usage_perc 指标"
    assert request_params_max_tokens_sum_found, "缺少 fastdeploy:request_params_max_tokens_sum 指标"
    assert request_success_total_found, "缺少 fastdeploy:request_success_total 指标"
    assert cache_config_info_found, "缺少 fastdeploy:cache_config_info 指标"
    assert available_batch_size_found, "缺少 fastdeploy:available_batch_size 指标"
    assert hit_req_rate_found, "缺少 fastdeploy:hit_req_rate 指标"
    assert hit_token_rate_found, "缺少 fastdeploy:hit_token_rate 指标"
    assert cpu_hit_token_rate_found, "缺少 fastdeploy:hit_token_rate 指标"
    assert gpu_hit_token_rate_found, "缺少 fastdeploy:gpu_hit_token_rate 指标"


# ==========================
# OpenAI Client chat.completions Test
# ==========================


@pytest.fixture
def openai_client():
    ip = "0.0.0.0"
    service_http_port = str(FD_API_PORT)
    client = openai.Client(
        base_url=f"http://{ip}:{service_http_port}/v1",
        api_key="EMPTY_API_KEY",
    )
    return client


# Non-streaming test
def test_non_streaming_chat(openai_client):
    """Test non-streaming chat functionality with the local service"""
    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "List 3 countries and their capitals."},
        ],
        temperature=1,
        max_tokens=1024,
        stream=False,
    )

    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], "message")
    assert hasattr(response.choices[0].message, "content")


# Streaming test
def test_streaming_chat(openai_client, capsys):
    """Test streaming chat functionality with the local service"""
    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "List 3 countries and their capitals."},
            {
                "role": "assistant",
                "content": "China(Beijing), France(Paris), Australia(Canberra).",
            },
            {"role": "user", "content": "OK, tell more."},
        ],
        temperature=1,
        max_tokens=1024,
        stream=True,
    )

    output = []
    for chunk in response:
        if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
            output.append(chunk.choices[0].delta.content)
    assert len(output) > 2


# ==========================
# OpenAI Client completions Test
# ==========================


def test_non_streaming(openai_client):
    """Test non-streaming chat functionality with the local service"""
    response = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        max_tokens=1024,
        stream=False,
    )

    # Assertions to check the response structure
    assert hasattr(response, "choices")
    assert len(response.choices) > 0


def test_streaming(openai_client, capsys):
    """Test streaming functionality with the local service"""
    response = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        max_tokens=1024,
        stream=True,
    )

    # Collect streaming output
    output = []
    for chunk in response:
        output.append(chunk.choices[0].text)
    assert len(output) > 0


def test_profile_reset_block_num():
    """测试profile reset_block_num功能，与baseline diff不能超过5%"""
    log_file = "./log/config.log"
    baseline = 32562

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


def test_prompt_token_ids_in_non_streaming_completion(openai_client):
    """
    Test cases for passing token ids through `prompt`/`prompt_token_ids` in non-streaming completion api
    """
    # Test case for passing a token id list in `prompt_token_ids`
    response = openai_client.completions.create(
        model="default",
        prompt="",
        temperature=1,
        max_tokens=5,
        extra_body={"prompt_token_ids": [5209, 626, 274, 45954, 1071, 3265, 3934, 1869, 93937]},
        stream=False,
    )
    assert len(response.choices) == 1
    assert response.usage.prompt_tokens == 9

    # Test case for passing a batch of token id lists in `prompt_token_ids`
    response = openai_client.completions.create(
        model="default",
        prompt="",
        temperature=1,
        max_tokens=5,
        extra_body={"prompt_token_ids": [[5209, 626, 274, 45954, 1071, 3265, 3934, 1869, 93937], [1, 2, 3]]},
        stream=False,
    )
    assert len(response.choices) == 2
    assert response.usage.prompt_tokens == 9 + 3

    # Test case for passing a token id list in `prompt`
    response = openai_client.completions.create(
        model="default",
        prompt=[5209, 626, 274, 45954, 1071, 3265, 3934, 1869, 93937],
        temperature=1,
        max_tokens=5,
        stream=False,
    )
    assert len(response.choices) == 1
    assert response.usage.prompt_tokens == 9

    # Test case for passing a batch of token id lists in `prompt`
    response = openai_client.completions.create(
        model="default",
        prompt=[[5209, 626, 274, 45954, 1071, 3265, 3934, 1869, 93937], [1, 2, 3]],
        temperature=1,
        max_tokens=5,
        stream=False,
    )
    assert len(response.choices) == 2
    assert response.usage.prompt_tokens == 9 + 3


def test_prompt_token_ids_in_streaming_completion(openai_client):
    """
    Test cases for passing token ids through `prompt`/`prompt_token_ids` in streaming completion api
    """
    # Test case for passing a token id list in `prompt_token_ids`
    response = openai_client.completions.create(
        model="default",
        prompt="",
        temperature=1,
        max_tokens=5,
        extra_body={"prompt_token_ids": [5209, 626, 274, 45954, 1071, 3265, 3934, 1869, 93937]},
        stream=True,
        stream_options={"include_usage": True},
    )
    sum_prompt_tokens = 0
    for chunk in response:
        if len(chunk.choices) > 0:
            assert chunk.usage is None
        else:
            sum_prompt_tokens += chunk.usage.prompt_tokens
    assert sum_prompt_tokens == 9

    # Test case for passing a batch of token id lists in `prompt_token_ids`
    response = openai_client.completions.create(
        model="default",
        prompt="",
        temperature=1,
        max_tokens=5,
        extra_body={"prompt_token_ids": [[5209, 626, 274, 45954, 1071, 3265, 3934, 1869, 93937], [1, 2, 3]]},
        stream=True,
        stream_options={"include_usage": True},
    )
    sum_prompt_tokens = 0
    for chunk in response:
        if len(chunk.choices) > 0:
            assert chunk.usage is None
        else:
            sum_prompt_tokens += chunk.usage.prompt_tokens
    assert sum_prompt_tokens == 9 + 3

    # Test case for passing a token id list in `prompt`
    response = openai_client.completions.create(
        model="default",
        prompt=[5209, 626, 274, 45954, 1071, 3265, 3934, 1869, 93937],
        temperature=1,
        max_tokens=5,
        stream=True,
        stream_options={"include_usage": True},
    )
    sum_prompt_tokens = 0
    for chunk in response:
        if len(chunk.choices) > 0:
            assert chunk.usage is None
        else:
            sum_prompt_tokens += chunk.usage.prompt_tokens
    assert sum_prompt_tokens == 9

    # Test case for passing a batch of token id lists in `prompt`
    response = openai_client.completions.create(
        model="default",
        prompt=[[5209, 626, 274, 45954, 1071, 3265, 3934, 1869, 93937], [1, 2, 3]],
        temperature=1,
        max_tokens=5,
        stream=True,
        stream_options={"include_usage": True},
    )
    sum_prompt_tokens = 0
    for chunk in response:
        if len(chunk.choices) > 0:
            assert chunk.usage is None
        else:
            sum_prompt_tokens += chunk.usage.prompt_tokens
    assert sum_prompt_tokens == 9 + 3
