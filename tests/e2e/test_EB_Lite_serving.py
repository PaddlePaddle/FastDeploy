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
import shutil
import signal
import subprocess
import sys
import time

import openai
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
        model_path = os.path.join(base_path, "ernie-4_5-21b-a3b-bf16-paddle")
    else:
        model_path = "./ernie-4_5-21b-a3b-bf16-paddle"

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
        "wint4",
        "--graph-optimization-config",
        '{"cudagraph_capture_sizes": [1], "use_cudagraph":true}',
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
    """
    Test non-streaming chat functionality with the local service
    """
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


def test_non_streaming_chat_finish_reason(openai_client):
    """
    Test non-streaming chat functionality with the local service
    """
    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "List 3 countries and their capitals."},
        ],
        temperature=1,
        max_tokens=5,
        stream=False,
    )

    assert hasattr(response, "choices")
    assert response.choices[0].finish_reason == "length"

    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "List 3 countries and their capitals."},
        ],
        temperature=1,
        max_completion_tokens=5,
        stream=False,
    )

    assert hasattr(response, "choices")
    assert response.choices[0].finish_reason == "length"

    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "List 3 countries and their capitals."},
        ],
        temperature=1,
        max_tokens=5,
        stream=False,
        n=2,
    )
    assert hasattr(response, "choices")
    for choice in response.choices:
        assert choice.finish_reason == "length"

    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "List 3 countries and their capitals."},
        ],
        temperature=1,
        max_completion_tokens=5,
        stream=False,
        n=2,
    )
    assert hasattr(response, "choices")
    for choice in response.choices:
        assert choice.finish_reason == "length"


# Streaming test
def test_streaming_chat(openai_client, capsys):
    """
    Test streaming chat functionality with the local service
    """
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
    """
    Test non-streaming chat functionality with the local service
    """
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
    """
    Test streaming functionality with the local service
    """
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


# ==========================
# OpenAI Client additional chat/completions test
# ==========================
def test_non_streaming_chat_with_n(openai_client):
    """
    Test n param option in non-streaming chat functionality with the local service
    """
    response = openai_client.chat.completions.create(
        model="default",
        messages=[],
        temperature=1,
        max_tokens=5,
        extra_body={"prompt_token_ids": [5209, 626, 274, 45954, 1071, 3265, 3934, 1869, 93937]},
        stream=False,
        n=2,
    )
    assert hasattr(response, "choices")
    assert len(response.choices) == 2
    assert hasattr(response, "usage")
    assert hasattr(response.usage, "prompt_tokens")
    assert response.usage.prompt_tokens == 9


def test_streaming_chat_with_n(openai_client):
    """
    Test n param option in streaming chat functionality with the local service
    """
    response = openai_client.chat.completions.create(
        model="default",
        messages=[],
        temperature=1,
        max_tokens=5,
        extra_body={"prompt_token_ids": [5209, 626, 274, 45954, 1071, 3265, 3934, 1869, 93937]},
        stream=True,
        stream_options={"include_usage": True},
        n=2,
    )
    count: list = [0, 0]
    for chunk in response:
        assert hasattr(chunk, "choices")
        assert hasattr(chunk, "usage")
        if len(chunk.choices) > 0:
            assert chunk.usage is None
            if chunk.choices[0].index == 0:
                count[0] = 1
            elif chunk.choices[0].index == 1:
                count[1] = 1
        else:
            assert hasattr(chunk.usage, "prompt_tokens")
            assert chunk.usage.prompt_tokens == 9
    assert sum(count) == 2


def test_completions_non_streaming_with_n(openai_client):
    """
    Test n param option in non-streaming completions functionality with the local service
    """
    response = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        max_tokens=1024,
        stream=False,
        n=2,
    )

    assert hasattr(response, "choices")
    assert len(response.choices) == 2
    assert hasattr(response.choices[0], "text")
    assert isinstance(response.choices[0].text, str)
    assert hasattr(response.choices[1], "text")
    assert isinstance(response.choices[1].text, str)


def test_completions_streaming_with_n(openai_client):
    """
    Test n param option in streaming completions functionality with the local service
    """
    response = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        max_tokens=1024,
        stream=True,
        n=2,
    )

    output_chunks = []
    count: list = [0, 0]
    for chunk in response:
        if chunk.choices[0].index == 0:
            count[0] = 1
        elif chunk.choices[0].index == 1:
            count[1] = 1
        assert hasattr(chunk, "choices")
        assert len(chunk.choices) > 0
        assert hasattr(chunk.choices[0], "text")
        output_chunks.append(chunk.choices[0].text)

    assert len(output_chunks) > 0
    assert sum(count) == 2


@pytest.mark.skip(reason="Temporarily skip this case due to unstable execution")
def test_non_streaming_with_stop_str(openai_client):
    """
    Test non-streaming chat functionality with the local service
    """
    response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        max_tokens=5,
        extra_body={"include_stop_str_in_output": True},
        stream=False,
    )
    # Assertions to check the response structure
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert response.choices[0].message.content.endswith("</s>")

    response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        max_tokens=5,
        extra_body={"include_stop_str_in_output": False},
        stream=False,
    )
    # Assertions to check the response structure
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert not response.choices[0].message.content.endswith("</s>")

    response = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        max_tokens=1024,
        stream=False,
    )
    assert not response.choices[0].text.endswith("</s>")

    response = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        max_tokens=1024,
        extra_body={"include_stop_str_in_output": True},
        stream=False,
    )
    assert response.choices[0].text.endswith("</s>")


def test_streaming_with_stop_str(openai_client):
    """
    Test non-streaming chat functionality with the local service
    """
    response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        max_tokens=5,
        extra_body={"min_tokens": 1, "include_stop_str_in_output": True},
        stream=True,
    )
    # Assertions to check the response structure
    last_token = ""
    for chunk in response:
        last_token = chunk.choices[0].delta.content
    assert last_token.endswith("</s>"), f"last_token did not end with '</s>': {last_token!r}"

    response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        max_tokens=5,
        extra_body={"include_stop_str_in_output": False},
        stream=True,
    )
    # Assertions to check the response structure
    last_token = ""
    for chunk in response:
        last_token = chunk.choices[0].delta.content
    assert last_token != "</s>"

    response_1 = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        max_tokens=10,
        stream=True,
    )
    last_token = ""
    for chunk in response_1:
        last_token = chunk.choices[0].text
    assert not last_token.endswith("</s>")

    response_1 = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        max_tokens=10,
        extra_body={"include_stop_str_in_output": True},
        stream=True,
    )
    last_token = ""
    for chunk in response_1:
        last_token = chunk.choices[0].text
    assert last_token.endswith("</s>")


def test_non_streaming_chat_with_return_token_ids(openai_client, capsys):
    """
    Test return_token_ids option in non-streaming chat functionality with the local service
    """
    #  enable return_token_ids
    response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        max_tokens=5,
        extra_body={"return_token_ids": True},
        stream=False,
    )
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], "message")
    assert hasattr(response.choices[0].message, "prompt_token_ids")
    assert isinstance(response.choices[0].message.prompt_token_ids, list)
    assert hasattr(response.choices[0].message, "completion_token_ids")
    assert isinstance(response.choices[0].message.completion_token_ids, list)

    #  disable return_token_ids
    response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        max_tokens=5,
        extra_body={"return_token_ids": False},
        stream=False,
    )
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], "message")
    assert hasattr(response.choices[0].message, "prompt_token_ids")
    assert response.choices[0].message.prompt_token_ids is None
    assert hasattr(response.choices[0].message, "completion_token_ids")
    assert response.choices[0].message.completion_token_ids is None


def test_streaming_chat_with_return_token_ids(openai_client, capsys):
    """
    Test return_token_ids option in streaming chat functionality with the local service
    """
    # enable return_token_ids
    response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        max_tokens=5,
        extra_body={"return_token_ids": True},
        stream=True,
    )
    is_first_chunk = True
    for chunk in response:
        assert hasattr(chunk, "choices")
        assert len(chunk.choices) > 0
        assert hasattr(chunk.choices[0], "delta")
        assert hasattr(chunk.choices[0].delta, "prompt_token_ids")
        assert hasattr(chunk.choices[0].delta, "completion_token_ids")
        if is_first_chunk:
            is_first_chunk = False
            assert isinstance(chunk.choices[0].delta.prompt_token_ids, list)
            assert chunk.choices[0].delta.completion_token_ids is None
        else:
            assert chunk.choices[0].delta.prompt_token_ids is None
            assert isinstance(chunk.choices[0].delta.completion_token_ids, list)

    # disable return_token_ids
    response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        max_tokens=5,
        extra_body={"return_token_ids": False},
        stream=True,
    )
    for chunk in response:
        assert hasattr(chunk, "choices")
        assert len(chunk.choices) > 0
        assert hasattr(chunk.choices[0], "delta")
        assert hasattr(chunk.choices[0].delta, "prompt_token_ids")
        assert chunk.choices[0].delta.prompt_token_ids is None
        assert hasattr(chunk.choices[0].delta, "completion_token_ids")
        assert chunk.choices[0].delta.completion_token_ids is None


def test_non_streaming_completion_with_return_token_ids(openai_client, capsys):
    """
    Test return_token_ids option in non-streaming completion functionality with the local service
    """
    # enable return_token_ids
    response = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        max_tokens=5,
        extra_body={"return_token_ids": True},
        stream=False,
    )
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], "prompt_token_ids")
    assert isinstance(response.choices[0].prompt_token_ids, list)
    assert hasattr(response.choices[0], "completion_token_ids")
    assert isinstance(response.choices[0].completion_token_ids, list)

    # disable return_token_ids
    response = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        max_tokens=5,
        extra_body={"return_token_ids": False},
        stream=False,
    )
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], "prompt_token_ids")
    assert response.choices[0].prompt_token_ids is None
    assert hasattr(response.choices[0], "completion_token_ids")
    assert response.choices[0].completion_token_ids is None


def test_streaming_completion_with_return_token_ids(openai_client, capsys):
    """
    Test return_token_ids option in streaming completion functionality with the local service
    """
    # enable return_token_ids
    response = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        max_tokens=5,
        extra_body={"return_token_ids": True},
        stream=True,
    )
    is_first_chunk = True
    for chunk in response:
        assert hasattr(chunk, "choices")
        assert len(chunk.choices) > 0
        assert hasattr(chunk.choices[0], "prompt_token_ids")
        assert hasattr(chunk.choices[0], "completion_token_ids")
        if is_first_chunk:
            is_first_chunk = False
            assert isinstance(chunk.choices[0].prompt_token_ids, list)
            assert chunk.choices[0].completion_token_ids is None
        else:
            assert chunk.choices[0].prompt_token_ids is None
            assert isinstance(chunk.choices[0].completion_token_ids, list)

    # disable return_token_ids
    response = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        max_tokens=5,
        extra_body={"return_token_ids": False},
        stream=True,
    )
    for chunk in response:
        assert hasattr(chunk, "choices")
        assert len(chunk.choices) > 0
        assert hasattr(chunk.choices[0], "prompt_token_ids")
        assert chunk.choices[0].prompt_token_ids is None
        assert hasattr(chunk.choices[0], "completion_token_ids")
        assert chunk.choices[0].completion_token_ids is None


def test_non_streaming_chat_with_prompt_token_ids(openai_client, capsys):
    """
    Test prompt_token_ids option in non-streaming chat functionality with the local service
    """
    response = openai_client.chat.completions.create(
        model="default",
        messages=[],
        temperature=1,
        max_tokens=5,
        extra_body={"prompt_token_ids": [5209, 626, 274, 45954, 1071, 3265, 3934, 1869, 93937]},
        stream=False,
    )
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response, "usage")
    assert hasattr(response.usage, "prompt_tokens")
    assert response.usage.prompt_tokens == 9


def test_streaming_chat_with_prompt_token_ids(openai_client, capsys):
    """
    Test prompt_token_ids option in streaming chat functionality with the local service
    """
    response = openai_client.chat.completions.create(
        model="default",
        messages=[],
        temperature=1,
        max_tokens=5,
        extra_body={"prompt_token_ids": [5209, 626, 274, 45954, 1071, 3265, 3934, 1869, 93937]},
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in response:
        assert hasattr(chunk, "choices")
        assert hasattr(chunk, "usage")
        if len(chunk.choices) > 0:
            assert chunk.usage is None
        else:
            assert hasattr(chunk.usage, "prompt_tokens")
            assert chunk.usage.prompt_tokens == 9


def test_non_streaming_completion_with_prompt_token_ids(openai_client, capsys):
    """
    Test prompt_token_ids option in streaming completion functionality with the local service
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


def test_streaming_completion_with_prompt_token_ids(openai_client, capsys):
    """
    Test prompt_token_ids option in non-streaming completion functionality with the local service
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


def test_non_streaming_chat_completion_disable_chat_template(openai_client, capsys):
    """
    Test disable_chat_template option in chat functionality with the local service.
    """
    enabled_response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        max_tokens=10,
        temperature=0.0,
        top_p=0,
        extra_body={"disable_chat_template": False},
        stream=False,
    )
    assert hasattr(enabled_response, "choices")
    assert len(enabled_response.choices) > 0

    # from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer
    # tokenizer = Ernie4_5Tokenizer.from_pretrained("PaddlePaddle/ERNIE-4.5-0.3B-Paddle", trust_remote_code=True)
    # prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Hello, how are you?"}], tokenize=False)
    prompt = "<|begin_of_sentence|>User: Hello, how are you?\nAssistant: "
    disabled_response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,
        top_p=0,
        extra_body={"disable_chat_template": True},
        stream=False,
    )
    assert hasattr(disabled_response, "choices")
    assert len(disabled_response.choices) > 0
    assert enabled_response.choices[0].message.content == disabled_response.choices[0].message.content


def test_non_streaming_chat_with_min_tokens(openai_client, capsys):
    """
    Test min_tokens option in non-streaming chat functionality with the local service
    """
    min_tokens = 1000
    response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        max_tokens=1010,
        extra_body={"min_tokens": min_tokens},
        stream=False,
    )
    assert hasattr(response, "usage")
    assert hasattr(response.usage, "completion_tokens")
    assert response.usage.completion_tokens >= min_tokens


def test_non_streaming_min_max_token_equals_one(openai_client, capsys):
    """
    Test chat/completion when min_tokens equals max_tokens equals 1.
    Verify it returns exactly one token.
    """
    # Test non-streaming chat
    response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1,
        temperature=0.0,
        stream=False,
    )
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], "message")
    assert hasattr(response.choices[0].message, "content")
    # Verify usage shows exactly 1 completion token
    assert hasattr(response, "usage")
    assert response.usage.completion_tokens == 1


def test_non_streaming_chat_with_bad_words(openai_client, capsys):
    """
    Test bad_words option in non-streaming chat functionality with the local service
    """
    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "ernie-4_5-21b-a3b-bf16-paddle")
    else:
        model_path = "./ernie-4_5-21b-a3b-bf16-paddle"
    response_0 = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        top_p=0.0,
        max_tokens=20,
        stream=False,
        extra_body={"return_token_ids": True},
    )

    assert hasattr(response_0, "choices")
    assert len(response_0.choices) > 0
    assert hasattr(response_0.choices[0], "message")
    assert hasattr(response_0.choices[0].message, "completion_token_ids")
    assert isinstance(response_0.choices[0].message.completion_token_ids, list)

    from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer

    tokenizer = Ernie4_5Tokenizer.from_pretrained(model_path, trust_remote_code=True)
    output_tokens_0 = []
    output_ids_0 = []
    for ids in response_0.choices[0].message.completion_token_ids:
        output_tokens_0.append(tokenizer.decode(ids))
        output_ids_0.append(ids)

    # add bad words
    bad_tokens = output_tokens_0[6:10]
    bad_token_ids = output_ids_0[6:10]
    response_1 = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        top_p=0.0,
        max_tokens=20,
        extra_body={"bad_words": bad_tokens, "return_token_ids": True},
        stream=False,
    )
    assert hasattr(response_1, "choices")
    assert len(response_1.choices) > 0
    assert hasattr(response_1.choices[0], "message")
    assert hasattr(response_1.choices[0].message, "completion_token_ids")
    assert isinstance(response_1.choices[0].message.completion_token_ids, list)

    response_2 = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        top_p=0.0,
        max_tokens=20,
        extra_body={"bad_words_token_ids": bad_token_ids, "return_token_ids": True},
        stream=False,
    )
    assert hasattr(response_2, "choices")
    assert len(response_2.choices) > 0
    assert hasattr(response_2.choices[0], "message")
    assert hasattr(response_2.choices[0].message, "completion_token_ids")
    assert isinstance(response_2.choices[0].message.completion_token_ids, list)

    assert not any(ids in response_1.choices[0].message.completion_token_ids for ids in bad_token_ids)
    assert not any(ids in response_2.choices[0].message.completion_token_ids for ids in bad_token_ids)


def test_streaming_chat_with_bad_words(openai_client, capsys):
    """
    Test bad_words option in streaming chat functionality with the local service
    """
    response_0 = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        top_p=0.0,
        max_tokens=20,
        stream=True,
        extra_body={"return_token_ids": True},
    )
    output_tokens_0 = []
    output_ids_0 = []
    is_first_chunk = True
    for chunk in response_0:
        assert hasattr(chunk, "choices")
        assert len(chunk.choices) > 0
        assert hasattr(chunk.choices[0], "delta")
        assert hasattr(chunk.choices[0].delta, "content")
        assert hasattr(chunk.choices[0].delta, "completion_token_ids")
        if is_first_chunk:
            is_first_chunk = False
        else:
            assert isinstance(chunk.choices[0].delta.completion_token_ids, list)
            output_tokens_0.append(chunk.choices[0].delta.content)
            output_ids_0.extend(chunk.choices[0].delta.completion_token_ids)

    # add bad words
    bad_tokens = output_tokens_0[6:10]
    bad_token_ids = output_ids_0[6:10]
    response_1 = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        top_p=0.0,
        max_tokens=20,
        extra_body={"bad_words": bad_tokens, "return_token_ids": True},
        stream=True,
    )
    output_tokens_1 = []
    output_ids_1 = []
    is_first_chunk = True
    for chunk in response_1:
        assert hasattr(chunk, "choices")
        assert len(chunk.choices) > 0
        assert hasattr(chunk.choices[0], "delta")
        assert hasattr(chunk.choices[0].delta, "content")
        assert hasattr(chunk.choices[0].delta, "completion_token_ids")
        if is_first_chunk:
            is_first_chunk = False
        else:
            assert isinstance(chunk.choices[0].delta.completion_token_ids, list)
            output_tokens_1.append(chunk.choices[0].delta.content)
            output_ids_1.extend(chunk.choices[0].delta.completion_token_ids)

    response_2 = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        top_p=0.0,
        max_tokens=20,
        extra_body={"bad_words_token_ids": bad_token_ids, "return_token_ids": True},
        stream=True,
    )
    output_tokens_2 = []
    output_ids_2 = []
    is_first_chunk = True
    for chunk in response_2:
        assert hasattr(chunk, "choices")
        assert len(chunk.choices) > 0
        assert hasattr(chunk.choices[0], "delta")
        assert hasattr(chunk.choices[0].delta, "content")
        assert hasattr(chunk.choices[0].delta, "completion_token_ids")
        if is_first_chunk:
            is_first_chunk = False
        else:
            assert isinstance(chunk.choices[0].delta.completion_token_ids, list)
            output_tokens_2.append(chunk.choices[0].delta.content)
            output_ids_2.extend(chunk.choices[0].delta.completion_token_ids)

    assert not any(ids in output_ids_1 for ids in bad_token_ids)
    assert not any(ids in output_ids_2 for ids in bad_token_ids)


def test_non_streaming_completion_with_bad_words(openai_client, capsys):
    """
    Test bad_words option in non-streaming completion functionality with the local service
    """
    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "ernie-4_5-21b-a3b-bf16-paddle")
    else:
        model_path = "./ernie-4_5-21b-a3b-bf16-paddle"

    response_0 = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        top_p=0.0,
        max_tokens=20,
        stream=False,
        extra_body={"return_token_ids": True},
    )
    assert hasattr(response_0, "choices")
    assert len(response_0.choices) > 0
    assert hasattr(response_0.choices[0], "completion_token_ids")
    assert isinstance(response_0.choices[0].completion_token_ids, list)

    from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer

    tokenizer = Ernie4_5Tokenizer.from_pretrained(model_path, trust_remote_code=True)
    output_tokens_0 = []
    output_ids_0 = []
    for ids in response_0.choices[0].completion_token_ids:
        output_tokens_0.append(tokenizer.decode(ids))
        output_ids_0.append(ids)

    # add bad words
    bad_tokens = output_tokens_0[6:10]
    bad_token_ids = output_ids_0[6:10]
    response_1 = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        top_p=0.0,
        max_tokens=20,
        extra_body={"bad_words": bad_tokens, "return_token_ids": True},
        stream=False,
    )
    assert hasattr(response_1, "choices")
    assert len(response_1.choices) > 0
    assert hasattr(response_1.choices[0], "completion_token_ids")
    assert isinstance(response_1.choices[0].completion_token_ids, list)

    response_2 = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        top_p=0.0,
        max_tokens=20,
        extra_body={"bad_words_token_ids": bad_token_ids, "return_token_ids": True},
        stream=False,
    )
    assert hasattr(response_2, "choices")
    assert len(response_2.choices) > 0
    assert hasattr(response_2.choices[0], "completion_token_ids")
    assert isinstance(response_2.choices[0].completion_token_ids, list)

    assert not any(ids in response_1.choices[0].completion_token_ids for ids in bad_token_ids)
    assert not any(ids in response_2.choices[0].completion_token_ids for ids in bad_token_ids)


def test_streaming_completion_with_bad_words(openai_client, capsys):
    """
    Test bad_words option in streaming completion functionality with the local service
    """
    response_0 = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        top_p=0.0,
        max_tokens=20,
        stream=True,
        extra_body={"return_token_ids": True},
    )
    output_tokens_0 = []
    output_ids_0 = []
    is_first_chunk = True
    for chunk in response_0:
        if is_first_chunk:
            is_first_chunk = False
        else:
            assert hasattr(chunk, "choices")
            assert len(chunk.choices) > 0
            assert hasattr(chunk.choices[0], "text")
            assert hasattr(chunk.choices[0], "completion_token_ids")
            output_tokens_0.append(chunk.choices[0].text)
            output_ids_0.extend(chunk.choices[0].completion_token_ids)

    # add bad words
    bad_token_ids = output_ids_0[6:10]
    bad_tokens = output_tokens_0[6:10]
    response_1 = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        top_p=0.0,
        max_tokens=20,
        extra_body={"bad_words": bad_tokens, "return_token_ids": True},
        stream=True,
    )
    output_tokens_1 = []
    output_ids_1 = []
    is_first_chunk = True
    for chunk in response_1:
        if is_first_chunk:
            is_first_chunk = False
        else:
            assert hasattr(chunk, "choices")
            assert len(chunk.choices) > 0
            assert hasattr(chunk.choices[0], "text")
            assert hasattr(chunk.choices[0], "completion_token_ids")
            output_tokens_1.append(chunk.choices[0].text)
            output_ids_1.extend(chunk.choices[0].completion_token_ids)
    # add bad words token ids
    response_2 = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        top_p=0.0,
        max_tokens=20,
        extra_body={"bad_words_token_ids": bad_token_ids, "return_token_ids": True},
        stream=True,
    )
    output_tokens_2 = []
    output_ids_2 = []
    is_first_chunk = True
    for chunk in response_2:
        if is_first_chunk:
            is_first_chunk = False
        else:
            assert hasattr(chunk, "choices")
            assert len(chunk.choices) > 0
            assert hasattr(chunk.choices[0], "text")
            assert hasattr(chunk.choices[0], "completion_token_ids")
            output_tokens_2.append(chunk.choices[0].text)
            output_ids_2.extend(chunk.choices[0].completion_token_ids)

    assert not any(ids in output_ids_1 for ids in bad_token_ids)
    assert not any(ids in output_ids_2 for ids in bad_token_ids)


def test_streaming_chat_finish_reason(openai_client):
    """
    Test non-streaming chat functionality with the local service
    """
    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "List 3 countries and their capitals."},
        ],
        temperature=1,
        max_tokens=5,
        stream=True,
    )

    for chunk in response:
        last_token = chunk.choices[0].finish_reason
    assert last_token == "length"

    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "List 3 countries and their capitals."},
        ],
        temperature=1,
        max_completion_tokens=5,
        stream=True,
    )

    for chunk in response:
        last_token = chunk.choices[0].finish_reason
    assert last_token == "length"

    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "List 3 countries and their capitals."},
        ],
        temperature=1,
        max_completion_tokens=5,
        stream=True,
        n=2,
    )
    finish_reason_1 = ""
    finish_reason_1 = ""

    for chunk in response:
        last_token = chunk.choices[0].finish_reason
        if last_token:
            if chunk.choices[0].index == 0:
                finish_reason_1 = last_token
            else:
                finish_reason_2 = last_token
    assert finish_reason_1 == "length"
    assert finish_reason_2 == "length"

    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "List 3 countries and their capitals."},
        ],
        temperature=1,
        max_tokens=5,
        stream=True,
        n=2,
    )
    finish_reason_1 = ""
    finish_reason_1 = ""

    for chunk in response:
        last_token = chunk.choices[0].finish_reason
        if last_token:
            if chunk.choices[0].index == 0:
                finish_reason_1 = last_token
            else:
                finish_reason_2 = last_token
    assert finish_reason_1 == "length"
    assert finish_reason_2 == "length"


def test_profile_reset_block_num():
    """测试profile reset_block_num功能，与baseline diff不能超过5%"""
    log_file = "./log/config.log"
    baseline = 31446

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
