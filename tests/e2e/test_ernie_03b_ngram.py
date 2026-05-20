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

import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest
from utils.serving_utils import (
    FD_API_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    clean,
    extract_logprobs,
    get_stream_chunks,
    is_port_open,
    send_request,
)


@pytest.fixture(scope="session", autouse=True)
def setup_and_run_server():
    print("Pre-test port cleanup...")
    clean()

    if os.path.exists("log") and os.path.isdir("log"):
        shutil.rmtree("log")

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "ERNIE-4.5-0.3B-Paddle")
    else:
        model_path = "baidu/ERNIE-4.5-0.3B-Paddle"

    speculative_config = {
        "method": "ngram",
        "num_speculative_tokens": 5,
        "max_ngram_size": 3,
        "min_ngram_size": 1,
    }

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
        "4096",
        "--max-num-seqs",
        "4",
        "--enable-overlap-schedule",
        "--enable-logprob",
        "--speculative-config",
        json.dumps(speculative_config),
        "--graph-optimization-config",
        '{"use_cudagraph":true}',
    ]

    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    for _ in range(300):
        if is_port_open("127.0.0.1", FD_API_PORT):
            print(f"Server is up on port {FD_API_PORT}")
            break
        time.sleep(1)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            clean()
        except Exception as e:
            print(f"Failed to kill process group: {e}")
        raise RuntimeError(f"API server did not start on port {FD_API_PORT}")

    yield

    print("\n===== Post-test server cleanup... =====")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        clean()
        print(f"server (pid={process.pid}) terminated")
    except Exception as e:
        print(f"Failed to terminate API server: {e}")


@pytest.fixture(scope="session")
def api_url():
    return f"http://0.0.0.0:{FD_API_PORT}/v1/chat/completions"


@pytest.fixture(scope="session")
def metrics_url():
    return f"http://0.0.0.0:{FD_METRICS_PORT}/metrics"


def test_ngram_stream(api_url):
    """Streaming generation returns non-empty result with valid token counts."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "牛顿的三大运动定律是什么？"}],
        "max_tokens": 50,
        "min_tokens": 10,
        "temperature": 0,
        "top_p": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)
    result = "".join(x["choices"][0]["delta"]["content"] for x in chunks[:-1])
    assert result != "", "Generation result is empty"
    usage = chunks[-1]["usage"]
    assert usage["completion_tokens"] <= payload["max_tokens"]
    assert usage["completion_tokens"] >= payload["min_tokens"]
    assert usage["total_tokens"] == usage["completion_tokens"] + usage["prompt_tokens"]


def test_ngram_non_stream(api_url):
    """Non-streaming generation returns non-empty result with valid token counts."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "牛顿的三大运动定律是什么？"}],
        "max_tokens": 50,
        "min_tokens": 10,
        "temperature": 0,
        "top_p": 0,
        "seed": 42,
        "stream": False,
    }
    response = send_request(url=api_url, payload=payload).json()
    result = response["choices"][0]["message"]["content"]
    assert result != "", "Generation result is empty"
    usage = response["usage"]
    assert usage["completion_tokens"] <= payload["max_tokens"]
    assert usage["completion_tokens"] >= payload["min_tokens"]
    assert usage["total_tokens"] == usage["completion_tokens"] + usage["prompt_tokens"]


def test_ngram_speculate_metrics(api_url):
    """speculate_metrics matches the fixed baseline (deterministic with seed=42)."""
    baseline = {
        "accepted_tokens": 100,
        "rejected_tokens": 314,
        "accept_ratio": 0.31000000000000005,
        "average_accept_length": 1.4492753623188406,
        "accepted_tokens_per_head": [69, 13, 9, 3, 3, 3],
        "accept_ratio_per_head": [0.18840579710144928, 0.6923076923076923, 0.3333333333333333, 1.0, 1.0],
    }
    # Prompt with repeated fragments to increase ngram match rate
    content = (
        "请复述以下内容：'牛顿第一定律：物体在不受外力作用时保持静止或匀速直线运动，"
        "牛顿第二定律：F=ma，牛顿第三定律：作用力与反作用力大小相等方向相反。'"
        "然后用自己的话解释牛顿第一定律。"
    )
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 100,
        "min_tokens": 20,
        "temperature": 0,
        "top_p": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)
    # chunks[-1] is the usage chunk; chunks[-2] is the last content chunk containing speculate_metrics
    speculate_metrics = chunks[-2]["choices"][0].get("speculate_metrics")
    # print(f"\n[test_ngram_speculate_metrics] speculate_metrics: {json.dumps(speculate_metrics, indent=2)}")
    assert speculate_metrics == baseline, f"speculate_metrics mismatch\ngot: {speculate_metrics}\nbaseline: {baseline}"


def test_ngram_speculate_metrics_with_logprobs(api_url):
    """speculate_metrics and logprobs coexist correctly when logprobs is enabled."""
    baseline = {
        "accepted_tokens": 100,
        "rejected_tokens": 332,
        "accept_ratio": 0.28,
        "average_accept_length": 1.3888888888888888,
        "accepted_tokens_per_head": [72, 12, 8, 3, 3, 2],
        "accept_ratio_per_head": [0.16666666666666666, 0.6666666666666666, 0.375, 1.0, 0.6666666666666666],
    }
    content = (
        "请复述以下内容：'牛顿第一定律：物体在不受外力作用时保持静止或匀速直线运动，"
        "牛顿第二定律：F=ma，牛顿第三定律：作用力与反作用力大小相等方向相反。'"
        "然后用自己的话解释牛顿第一定律。"
    )
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 100,
        "min_tokens": 20,
        "temperature": 0,
        "top_p": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "logprobs": True,
        "top_logprobs": 5,
    }
    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)

    # logprobs are present in each content chunk
    logprobs_list = extract_logprobs(chunks)
    assert len(logprobs_list) > 0, "No logprobs received"
    for logprobs in logprobs_list:
        assert "content" in logprobs
        for item in logprobs["content"]:
            assert "token" in item
            assert "logprob" in item
            assert "top_logprobs" in item
            assert len(item["top_logprobs"]) <= 5

    # speculate_metrics appears in the last content chunk and matches baseline
    speculate_metrics = chunks[-2]["choices"][0].get("speculate_metrics")
    # print(f"\n[test_ngram_speculate_metrics_with_logprobs] speculate_metrics: {json.dumps(speculate_metrics, indent=2)}")
    assert speculate_metrics == baseline, f"speculate_metrics mismatch\ngot: {speculate_metrics}\nbaseline: {baseline}"
