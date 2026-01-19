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
    PORTS_TO_CLEAN,
    clean_ports,
    is_port_open,
)


@pytest.fixture(scope="session", autouse=True)
def setup_and_run_server(api_url):
    """
    Pytest fixture that runs once per test session:
    - Cleans ports before tests
    - Starts the API server as a subprocess
    - Waits for server port to open (up to 30 seconds)
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
        "2",
        "--data-parallel-size",
        "2",
        "--max-model-len",
        "65536",
        "--max-num-seqs",
        "32",
        "--quantization",
        "block_wise_fp8",
        "--enable-logprob",
    ]

    # Start subprocess in new process group
    if os.path.exists("log"):
        shutil.rmtree("log")

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
            print(f"Server is up on port {FD_API_PORT}")
            break
        time.sleep(1)
    else:
        print("[TIMEOUT] API server failed to start in 5 minutes. Cleaning up...")
        try:
            os.killpg(process.pid, signal.SIGTERM)
            # clean()
        except Exception as e:
            print(f"Failed to kill process group: {e}")
        raise RuntimeError(f"API server did not start on port {FD_API_PORT}")

    yield  # Run tests

    print("\n===== Post-test server cleanup... =====")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        # clean()
        time.sleep(10)
        print(f"server (pid={process.pid}) terminated")
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


def send_request(url, payload, timeout=60):
    """
    Send a POST request to the specified URL with the given payload.
    """
    headers = {
        "Content-Type": "application/json",
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=timeout)
        print("🟢 Receiving response...\n")
        return res
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out (>{timeout} seconds)")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def get_stream_chunks(response):
    """
    Parse a streaming HTTP response into a list of JSON chunks.

    This helper processes Server-Sent Events (SSE) style responses,
    strips the 'data:' prefix, ignores the '[DONE]' marker, and
    decodes each chunk into a Python dict.

    Args:
        response: HTTP response returned by send_request().

    Returns:
        List[dict]: Parsed stream chunks in arrival order.
    """
    chunks = []

    if response.status_code == 200:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("data: "):
                line = line[len("data: ") :]

            if line.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(line)
                chunks.append(chunk)
            except Exception as e:
                print(f"Failed to parse chunk: {e}, raw line: {line}")
    else:
        print(f"Request failed, status code: {response.status_code}")
        print("Response body:", response.text)

    return chunks


def get_token_list(response):
    """
    Extract generated token strings from a non-streaming response.

    This function reads token-level information from
    `choices[0].logprobs.content` and returns the generated token list
    in order. It is mainly used for stop-sequence validation.

    Args:
        response (dict): JSON-decoded inference response.

    Returns:
        List[str]: Generated token strings.
    """
    token_list = []

    try:
        content_logprobs = response["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"Failed to extract logprobs: {e}")
        return []

    for token_info in content_logprobs:
        token = token_info.get("token")
        if token is not None:
            token_list.append(token)

    return token_list


def extract_logprobs(chunks):
    """
    Extract token-level logprobs from streaming response chunks.

    This helper skips chunks without choices, usage-only chunks,
    and chunks without logprobs, and aggregates token-level
    logprob information in generation order.

    Args:
        chunks (List[dict]): Parsed streaming chunks.

    Returns:
        List[List[dict]]: Structured logprobs for each generated token.
    """
    results = []

    for chunk in chunks:
        choices = chunk.get("choices")
        if not choices:
            continue

        choice = choices[0]
        logprobs = choice.get("logprobs")
        if not logprobs or not logprobs.get("content"):
            continue

        token_infos = []
        for item in logprobs["content"]:
            token_infos.append(
                {
                    "token": item["token"],
                    "logprob": item["logprob"],
                    "top_logprobs": [
                        {
                            "token": tlp["token"],
                            "logprob": tlp["logprob"],
                        }
                        for tlp in item.get("top_logprobs", [])
                    ],
                }
            )

        results.append(token_infos)

    return results


def test_text_diff(api_url):
    """
    Validate deterministic streaming output against a fixed text baseline.

    The test uses fixed decoding parameters and seed, concatenates
    all streamed content, and performs a strict byte-level comparison
    with a stored baseline file.
    """
    payload = {
        "stream": True,
        "seed": 21,
        "top_p": 0,
        "stop": ["</s>", "<eos>", "<|endoftext|>", "<|im_end|>"],
        "chat_template_kwargs": {
            "options": {"thinking_mode": "close"},
        },
        "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
        "messages": [{"role": "user", "content": "解释一下温故而知新"}],
    }

    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)

    result = "".join(x["choices"][0]["delta"]["content"] for x in chunks)

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        base_file = os.path.join(base_path, "21b_ep4_text_baseline.txt")
    else:
        base_file = "21b_ep4_text_baseline.txt"
    with open(base_file, "r", encoding="utf-8") as f:
        baseline = f.read()

    assert result == baseline, f"Text mismatch with baseline\nresult: {result}\nbaseline: {baseline}"


def test_chat_usage_stream(api_url):
    """
    Verify token usage statistics for chat completion in streaming mode.

    The test ensures:
    - Generated content is non-empty
    - completion_tokens respects min/max constraints
    - total_tokens equals prompt_tokens + completion_tokens
    """
    payload = {
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": True,
        },
        "messages": [{"role": "user", "content": "解释一下温故而知新"}],
        "min_tokens": 10,
        "max_tokens": 50,
    }

    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)

    result = "".join(x["choices"][0]["delta"]["content"] for x in chunks[:-1])
    assert result != "", "Empty generation result"

    usage = chunks[-1]["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]

    assert payload["max_tokens"] >= usage["completion_tokens"]
    assert payload["min_tokens"] <= usage["completion_tokens"]
    assert usage["total_tokens"] == total_tokens


def test_chat_usage_non_stream(api_url):
    """
    Verify token usage statistics for chat completion in non-streaming mode.
    """
    payload = {
        "stream": False,
        "messages": [{"role": "user", "content": "解释一下温故而知新"}],
        "temperature": 1.0,
        "seed": 21,
        "top_p": 0,
        "stop": ["</s>", "<eos>", "<|endoftext|>", "<|im_end|>"],
        "min_tokens": 10,
        "max_tokens": 50,
        "chat_template_kwargs": {
            "options": {"thinking_mode": "close"},
        },
        "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
    }

    response = send_request(url=api_url, payload=payload).json()

    usage = response["usage"]
    result = response["choices"][0]["message"]["content"]
    assert result != "", "Empty generation result"

    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"]
    assert payload["min_tokens"] <= usage["completion_tokens"]
    assert usage["total_tokens"] == total_tokens


def test_non_chat_usage_stream(api_url):
    """
    Verify usage statistics for completions API in streaming mode.
    """
    payload = {
        "model": "null",
        "prompt": "你好，你是谁？",
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": True,
        },
        "min_tokens": 10,
        "max_tokens": 50,
        "seed": 566,
        "chat_template_kwargs": {
            "options": {"thinking_mode": "close"},
        },
        "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
    }

    api_url = api_url.replace("chat/completions", "completions")
    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)

    usage = chunks[-1]["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]

    assert payload["max_tokens"] >= usage["completion_tokens"]
    assert payload["min_tokens"] <= usage["completion_tokens"]
    assert usage["total_tokens"] == total_tokens


def test_non_chat_usage_non_stream(api_url):
    """
    Verify usage statistics for completions API in non-streaming mode.
    """
    payload = {
        "model": "null",
        "prompt": "你好，你是谁？",
        "stream": False,
        "min_tokens": 10,
        "max_tokens": 50,
        "seed": 566,
        "chat_template_kwargs": {
            "options": {"thinking_mode": "close"},
        },
        "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
    }

    api_url = api_url.replace("chat/completions", "completions")
    response = send_request(url=api_url, payload=payload).json()

    usage = response["usage"]
    result = response["choices"][0]["text"]
    assert result != "", "Empty generation result"

    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]

    assert payload["max_tokens"] >= usage["completion_tokens"]
    assert payload["min_tokens"] <= usage["completion_tokens"]
    assert usage["total_tokens"] == total_tokens


def test_stop_sequence(api_url):
    """
    Verify that a punctuation stop sequence correctly truncates generation.

    The test validates stop behavior at the token level using logprobs.
    """
    payload = {
        "stream": False,
        "stop": ["。"],
        "messages": [
            {
                "role": "user",
                "content": (
                    "你要严格按照我接下来的话输出，输出冒号后面的内容，"
                    "请输出：这是第一段。果冻这是第二段啦啦啦啦啦。"
                ),
            },
        ],
        "max_tokens": 20,
        "top_p": 0,
        "logprobs": True,
        "top_logprobs": 5,
        "min_tokens": 10,
        "chat_template_kwargs": {
            "options": {"thinking_mode": "close"},
        },
        "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
    }

    response = send_request(url=api_url, payload=payload).json()

    token_list = get_token_list(response)

    assert "第二段" not in token_list
    assert "。" in token_list


def test_stop_sequence1(api_url):
    """
    Verify that generation is not truncated when no stop sequence is provided.
    """
    payload = {
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": (
                    "你要严格按照我接下来的话输出，输出冒号后面的内容，"
                    "请输出：这是第一段。果冻这是第二段啦啦啦啦啦。"
                ),
            },
        ],
        "max_tokens": 20,
        "top_p": 0,
        "logprobs": True,
        "top_logprobs": 5,
        "min_tokens": 10,
        "chat_template_kwargs": {
            "options": {"thinking_mode": "close"},
        },
        "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
    }

    response = send_request(url=api_url, payload=payload).json()

    content = response["choices"][0]["message"]["content"]
    assert "第二段" in content


def test_stop_sequence2(api_url):
    """
    Verify that a custom string stop sequence truncates generation correctly.
    """
    payload = {
        "stream": False,
        "stop": ["这是第二段啦啦"],
        "messages": [
            {
                "role": "user",
                "content": (
                    "你要严格按照我接下来的话输出，输出冒号后面的内容，"
                    "请输出：这是第一段。果冻这是第二段啦啦啦啦啦。"
                ),
            },
        ],
        "max_tokens": 20,
        "top_p": 0,
        "logprobs": True,
        "top_logprobs": 5,
        "min_tokens": 10,
        "chat_template_kwargs": {
            "options": {"thinking_mode": "close"},
        },
        "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
    }

    response = send_request(url=api_url, payload=payload).json()

    content = response["choices"][0]["message"]["content"]

    assert "啦啦啦" not in content


def test_non_stream_with_logprobs(api_url):
    """
    Verify deterministic logprobs output in non-streaming mode.
    """
    payload = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
        "logprobs": True,
        "top_logprobs": 5,
        "seed": 21,
        "min_tokens": 1,
        "chat_template_kwargs": {
            "options": {"thinking_mode": "close"},
        },
        "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
    }

    resp_json = send_request(url=api_url, payload=payload).json()

    logprobs = resp_json["choices"][0]["logprobs"]

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        base_file = os.path.join(base_path, "21b_ep4_logprobs_non_stream_static_baseline.txt")
    else:
        base_file = "21b_ep4_logprobs_non_stream_static_baseline.txt"
    with open(base_file, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    assert logprobs == baseline


def test_stream_with_logprobs(api_url):
    """
    Verify deterministic logprobs output in streaming mode.
    """
    payload = {
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
        "logprobs": True,
        "top_logprobs": 5,
        "min_tokens": 1,
        "seed": 21,
    }

    response = send_request(url=api_url, payload=payload)

    chunks = get_stream_chunks(response)
    logprobs = extract_logprobs(chunks)

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        base_file = os.path.join(base_path, "21b_ep4_logprobs_stream_static_baseline.txt")
    else:
        base_file = "21b_ep4_logprobs_stream_static_baseline.txt"
    with open(base_file, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    assert logprobs == baseline
