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

# Test splitwise deployment WITHOUT Router:
# use local_scheduler, manually construct disaggregate_info,
# send requests to both Prefill and Decode concurrently.
# ENABLE_V1_KVCACHE_SCHEDULER=1, use rdma to transfer cache.

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid

import pytest
import requests
from utils.serving_utils import (
    FD_API_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    check_service_health,
    clean,
)

# Ports for PD disaggregation (no router port needed)
FD_CONNECTOR_PORT = int(os.getenv("FD_CONNECTOR_PORT", 8433))
FD_RDMA_PORT = int(os.getenv("FD_RDMA_PORT", 8623))

# Prefill uses base ports, Decode uses base+1
PORTS_TO_CLEAN = [
    FD_API_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_CONNECTOR_PORT,
    FD_RDMA_PORT,
    FD_API_PORT + 1,
    FD_ENGINE_QUEUE_PORT + 1,
    FD_METRICS_PORT + 1,
    FD_CACHE_QUEUE_PORT + 1,
    FD_CONNECTOR_PORT + 1,
    FD_RDMA_PORT + 1,
]


def _build_disaggregate_info() -> dict:
    """Build disaggregate_info manually, replicating Router's handle_splitwise_request logic."""
    host_ip = os.getenv("FD_HOST_IP", "127.0.0.1")
    return {
        "prefill_ip": host_ip,
        "decode_ip": host_ip,
        "prefill_connector_port": FD_CONNECTOR_PORT,
        "decode_connector_port": FD_CONNECTOR_PORT + 1,
        "decode_device_ids": ["1"],
        "decode_rdma_ports": [FD_RDMA_PORT + 1],
        "transfer_protocol": "rdma",
        "decode_tp_size": 1,
    }


def _send_pd_request(payload: dict, timeout: int = 120):
    """
    Send request to both Prefill and Decode concurrently,
    replicate Router's fan-out forwarding behavior.
    Returns the Decode response (same as Router's return_result_url_index=-1).
    """
    disaggregate_info = _build_disaggregate_info()

    # Inject disaggregate_info and request_id (same as Router)
    payload = payload.copy()
    payload["disaggregate_info"] = disaggregate_info
    if "request_id" not in payload:
        payload["request_id"] = f"test-pd-{uuid.uuid4()}"

    prefill_url = f"http://127.0.0.1:{FD_API_PORT}/v1/chat/completions"
    decode_url = f"http://127.0.0.1:{FD_API_PORT + 1}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}

    # For streaming, use requests with stream=True for decode response
    if payload.get("stream", False):
        # Send to both concurrently (same as Router's fan-out), stream from decode
        import concurrent.futures

        def _post_stream(url):
            return requests.post(url, headers=headers, json=payload, timeout=timeout, stream=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            prefill_future = executor.submit(_post_stream, prefill_url)
            decode_future = executor.submit(_post_stream, decode_url)
            # Return decode streaming response immediately
            decode_resp = decode_future.result()
            # Consume prefill response in background (don't block)
            try:
                prefill_future.result(timeout=timeout)
            except Exception:
                pass
        return decode_resp
    else:
        # Non-streaming: send to both, return decode response
        import concurrent.futures

        def _post(url):
            return requests.post(url, headers=headers, json=payload, timeout=timeout)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            prefill_future = executor.submit(_post, prefill_url)
            decode_future = executor.submit(_post, decode_url)
            # Wait for both, return decode response
            decode_resp = decode_future.result()
            # Also check prefill didn't error (but don't block on it)
            try:
                prefill_future.result(timeout=5)
            except Exception:
                pass
        return decode_resp


@pytest.fixture(scope="session", autouse=True)
def setup_and_run_server():
    """
    Pytest fixture that runs once per test session:
    - Cleans ports before tests
    - Starts Prefill and Decode instances WITHOUT Router
    - Waits for both to be healthy
    - Tears down after all tests finish
    """
    print("Pre-test port cleanup...")
    clean(PORTS_TO_CLEAN)

    print("log dir clean")
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

    base_log_dir = os.getenv("FD_LOG_DIR", "log")

    # Prefill instance
    print("start prefill...")
    env_prefill = os.environ.copy()
    env_prefill["CUDA_VISIBLE_DEVICES"] = "0"
    env_prefill["FD_LOG_DIR"] = os.path.join(base_log_dir, "log_prefill")

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
        "--splitwise-role",
        "prefill",
        "--cache-transfer-protocol",
        "rdma",
        "--rdma-comm-ports",
        str(FD_RDMA_PORT),
        "--pd-comm-port",
        str(FD_CONNECTOR_PORT),
        # No --router flag
    ]

    with open(prefill_log_path, "w") as logfile:
        process_prefill = subprocess.Popen(
            prefill_cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env_prefill,
        )
    time.sleep(1)

    # Decode instance
    print("start decode...")
    env_decode = os.environ.copy()
    env_decode["CUDA_VISIBLE_DEVICES"] = "1"
    env_decode["FD_LOG_DIR"] = os.path.join(base_log_dir, "log_decode")

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
        str(FD_RDMA_PORT + 1),
        "--pd-comm-port",
        str(FD_CONNECTOR_PORT + 1),
        # No --router flag
    ]

    with open(decode_log_path, "w") as logfile:
        process_decode = subprocess.Popen(
            decode_cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env_decode,
        )

    # Wait up to 300 seconds for both instances to be healthy
    for _ in range(60):
        prefill_healthy = check_service_health(f"http://127.0.0.1:{FD_API_PORT}")
        decode_healthy = check_service_health(f"http://127.0.0.1:{FD_API_PORT + 1}")
        if prefill_healthy and decode_healthy:
            print("Prefill and decode servers are both online")
            break
        time.sleep(5)
    else:
        print("[TIMEOUT] Servers failed to start in 5 minutes. Cleaning up...")
        try:
            os.killpg(process_prefill.pid, signal.SIGTERM)
            os.killpg(process_decode.pid, signal.SIGTERM)
            clean(PORTS_TO_CLEAN)
        except Exception as e:
            print(f"Failed to kill process group: {e}")
        raise RuntimeError("Prefill or decode server did not start")

    yield  # Run tests

    print("\n===== Post-test server cleanup... =====")
    try:
        os.killpg(process_prefill.pid, signal.SIGTERM)
        os.killpg(process_decode.pid, signal.SIGTERM)
        clean(PORTS_TO_CLEAN)
        print(f"Prefill server (pid={process_prefill.pid}) terminated")
        print(f"Decode server (pid={process_decode.pid}) terminated")
    except Exception as e:
        print(f"Failed to terminate server: {e}")


@pytest.fixture(scope="session")
def api_url(request):
    """
    Returns the Decode API endpoint URL (where final responses come from).
    """
    return f"http://127.0.0.1:{FD_API_PORT + 1}/v1/chat/completions"


@pytest.fixture
def headers():
    return {"Content-Type": "application/json"}


def get_stream_chunks(response):
    """Parse streaming response into chunk list."""
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
                    print(f"Parse failed: {e}, line: {line}")
    else:
        print(f"Request failed, status: {response.status_code}")
        print("Response:", response.text)

    return chunks


def test_chat_usage_stream(api_url):
    """Test streaming chat with usage"""
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

    response = _send_pd_request(payload)
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
    """Test non-streaming chat with usage"""
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

    response = _send_pd_request(payload).json()
    usage = response["usage"]
    result = response["choices"][0]["message"]["content"]
    assert result != "", "结果为空"
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"], "completion_tokens大于max_tokens"
    assert payload["metadata"]["min_tokens"] <= usage["completion_tokens"], "completion_tokens小于min_tokens"
    assert usage["total_tokens"] == total_tokens, "total_tokens不等于prompt_tokens + completion_tokens"


def test_non_chat_usage_stream(api_url):
    """Test streaming completion (non-chat) with usage"""
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

    # Send to /v1/completions endpoints
    disaggregate_info = _build_disaggregate_info()
    payload = payload.copy()
    payload["disaggregate_info"] = disaggregate_info
    if "request_id" not in payload:
        payload["request_id"] = f"test-pd-{uuid.uuid4()}"

    prefill_url = f"http://127.0.0.1:{FD_API_PORT}/v1/completions"
    decode_url = f"http://127.0.0.1:{FD_API_PORT + 1}/v1/completions"
    headers = {"Content-Type": "application/json"}

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(requests.post, prefill_url, json=payload, headers=headers, timeout=120)
        decode_future = executor.submit(
            requests.post, decode_url, json=payload, headers=headers, timeout=120, stream=True
        )
    response = decode_future.result()

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
    """Test non-streaming completion (non-chat) with usage"""
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

    # Send to /v1/completions endpoints
    disaggregate_info = _build_disaggregate_info()
    payload = payload.copy()
    payload["disaggregate_info"] = disaggregate_info
    if "request_id" not in payload:
        payload["request_id"] = f"test-pd-{uuid.uuid4()}"

    prefill_url = f"http://127.0.0.1:{FD_API_PORT}/v1/completions"
    decode_url = f"http://127.0.0.1:{FD_API_PORT + 1}/v1/completions"
    headers = {"Content-Type": "application/json"}

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(requests.post, prefill_url, json=payload, headers=headers, timeout=120)
        decode_future = executor.submit(requests.post, decode_url, json=payload, headers=headers, timeout=120)
    response = decode_future.result().json()

    usage = response["usage"]
    result = response["choices"][0]["text"]
    print("Decode Response:", result)
    assert result != "", "结果为空"
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"], "completion_tokens大于max_tokens"
    assert payload["metadata"]["min_tokens"] <= usage["completion_tokens"], "completion_tokens小于min_tokens"
    assert usage["total_tokens"] == total_tokens, "total_tokens不等于prompt_tokens + completion_tokens"
