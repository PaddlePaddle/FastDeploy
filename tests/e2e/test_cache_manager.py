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
from utils.serving_utils import (
    FD_API_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    clean,
    is_port_open,
)


@pytest.fixture(scope="session", autouse=True)
def setup_and_run_server():
    """
    Pytest fixture that runs once per test session:
    - Cleans ports before tests
    - Starts the API server with prefix caching enabled
    - Waits for server to be ready
    - Tears down server after all tests finish
    """
    print("Pre-test port cleanup...")
    clean()

    if os.path.exists("log") and os.path.isdir("log"):
        shutil.rmtree("log")

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "ernie-4_5-21b-a3b-bf16-paddle")
    else:
        model_path = "./ernie-4_5-21b-a3b-bf16-paddle"
    mtp_model_path = os.path.join(model_path, "mtp")
    speculative_config = {"method": "mtp", "num_speculative_tokens": 1, "model": mtp_model_path}

    server_env = os.environ.copy()
    server_env["ENABLE_V1_KVCACHE_MANAGER"] = "1"

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
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "128",
        "--quantization",
        "wint4",
        "--enable-prefix-caching",
        "--swap-space",
        "20",
        "--speculative-config",
        json.dumps(speculative_config),
    ]

    if os.path.exists("log"):
        shutil.rmtree("log")
    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=server_env,
        )

    for _ in range(300):
        if is_port_open("127.0.0.1", FD_API_PORT):
            print(f"Server is up on port {FD_API_PORT}")
            break
        time.sleep(1)
    else:
        print("[TIMEOUT] API server failed to start in 5 minutes. Cleaning up...")
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


@pytest.fixture
def headers():
    return {"Content-Type": "application/json"}


# ── helpers ──────────────────────────────────────────────────────────


def _send_request(url, payload, timeout=60):
    """发送请求并返回响应"""
    try:
        res = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=timeout)
        return res
    except Exception as e:
        print(f"请求失败: {e}")
        return None


def _get_stream_chunks(response):
    """解析流式响应，返回 chunk 列表"""
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
                chunks.append(json.loads(line))
            except Exception:
                pass
    return chunks


def _extract_cached_tokens(chunks):
    """从流式 chunks 中提取 cached_tokens"""
    for chunk in reversed(chunks):
        usage = chunk.get("usage", {})
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        if cached:
            return cached
    return 0


def _parse_metrics(text):
    """解析 Prometheus metrics 文本为 dict"""
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if " " in line:
            key, val = line.rsplit(" ", 1)
            try:
                metrics[key] = float(val)
            except ValueError:
                metrics[key] = val
    return metrics


# ── tests ────────────────────────────────────────────────────────────


def test_cache_cold_start(api_url, headers):
    """冷启动：首次请求不应命中缓存"""
    payload = {
        "model": "default",
        "temperature": 0,
        "seed": 33,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个乐于助人的助手，总是耐心解答用户的问题。"
                    "你具备广泛的知识储备，涵盖科技、历史、文化、艺术等多个领域。"
                    "你善于用通俗易懂的语言解释复杂概念。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请用一段话详细介绍什么是前缀缓存（Prefix Caching）技术。"
                    "包括它的工作原理、主要优势、适用场景，以及在大模型推理中的重要地位。"
                ),
            },
        ],
        "max_tokens": 128,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    resp = _send_request(api_url, payload)
    chunks = _get_stream_chunks(resp)
    cached_tokens = _extract_cached_tokens(chunks)

    assert cached_tokens == 0, f"冷启动不应命中缓存，但 cached_tokens={cached_tokens}"


def test_cache_hit_on_repeat(api_url, headers):
    """重复请求应命中前缀缓存"""
    system_prompt = (
        "你是一个精确、严谨的技术助手，总是给出结构化的回答。"
        "你擅长分析复杂的技术问题，从架构、性能、生态等多个维度给出全面评估。"
        "你的回答总是条理清晰，先概述再分点详述，最后总结。"
        "你避免模糊表述，所有结论都有具体论据支撑。"
        "你对大模型推理框架有深入研究，熟悉 vLLM、SGLang 等竞品的技术方案。"
    )
    user_content = (
        "请从以下角度详细分析 FastDeploy 推理框架的主要功能："
        "1. 模型部署与推理加速方面，FastDeploy 提供了哪些核心能力？比如图优化、算子融合、CUDA Graph 等。"
        "2. 在内存管理和缓存优化方面，有哪些关键技术和策略？比如前缀缓存、KV Cache 多级存储、Swap 机制等。"
        "3. 分布式推理和 PD 分离架构是如何设计和实现的？Router 如何调度请求到 P 节点和 D 节点？"
        "请每个角度至少给出两点具体说明，并与业界主流方案进行对比。"
    )

    payload = {
        "model": "default",
        "temperature": 0,
        "seed": 33,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 128,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    # 第一次请求：写入缓存
    _send_request(api_url, payload)
    time.sleep(2)

    # 第二次请求：应命中缓存
    resp = _send_request(api_url, payload)
    chunks = _get_stream_chunks(resp)
    cached_tokens = _extract_cached_tokens(chunks)

    assert cached_tokens > 0, f"重复请求应命中缓存，但 cached_tokens={cached_tokens}"


def test_cache_shared_prefix(api_url, headers):
    """多轮对话共享前缀应命中缓存"""
    system_prompt = (
        "你是一个高效的AI助手，始终用简洁的中文回答。"
        "你擅长将复杂的技术问题拆解为易于理解的要点，帮助用户快速掌握核心知识。"
        "你对大模型推理、分布式系统、深度学习框架等领域有深入理解。"
    )

    # 第一条请求写入缓存
    payload1 = {
        "model": "default",
        "temperature": 0,
        "seed": 33,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "问题1：什么是KV缓存？它在大模型推理中起什么作用？请详细说明。"},
        ],
        "max_tokens": 64,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    _send_request(api_url, payload1)
    time.sleep(2)

    # 第二条请求共享 system prompt 前缀
    payload2 = {
        "model": "default",
        "temperature": 0,
        "seed": 33,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "问题2：前缀缓存能带来什么收益？它与传统KV缓存相比有哪些改进和优势？"},
        ],
        "max_tokens": 64,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    resp = _send_request(api_url, payload2)
    chunks = _get_stream_chunks(resp)
    cached_tokens = _extract_cached_tokens(chunks)

    assert cached_tokens > 0, f"共享前缀应命中缓存，但 cached_tokens={cached_tokens}"


def test_cache_metrics_endpoint(api_url, metrics_url):
    """验证 /metrics 端点包含缓存相关指标"""
    # 先发几个请求，产生缓存数据
    payload = {
        "model": "default",
        "temperature": 0,
        "seed": 33,
        "messages": [
            {
                "role": "user",
                "content": (
                    "你好，请介绍一下你自己。你是什么模型？有哪些能力？"
                    "可以帮我做哪些事情？比如编程、翻译、写作、分析数据等。"
                ),
            },
        ],
        "max_tokens": 64,
        "stream": False,
    }
    for _ in range(3):
        _send_request(api_url, payload)

    time.sleep(2)

    resp = requests.get(metrics_url, timeout=10)
    assert resp.status_code == 200, f"metrics 端点返回 {resp.status_code}"

    metrics = _parse_metrics(resp.text)

    # 验证缓存相关 key 存在
    cache_keys = [k for k in metrics if "cache" in k.lower() or "hit_" in k.lower()]
    print(f"Cache-related metrics keys: {cache_keys}")
    assert len(cache_keys) > 0, "metrics 中应包含缓存相关指标"


def test_cache_non_stream(api_url, headers):
    """非流式请求的 cached_tokens 验证"""
    system_prompt = (
        "你是一个数学助手，精通代数、几何、微积分、概率统计等各个数学分支。"
        "你总是给出严谨的推导过程，不仅给出答案，还解释背后的数学原理。"
    )
    payload = {
        "model": "default",
        "temperature": 0,
        "seed": 33,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "1+1等于几？请从数学公理体系的角度给出严谨的证明过程。",
            },
        ],
        "max_tokens": 128,
        "stream": False,
    }

    # 第一次：写入缓存
    _send_request(api_url, payload)
    time.sleep(2)

    # 第二次：应命中缓存
    resp = _send_request(api_url, payload)
    data = resp.json()
    cached_tokens = data.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    assert cached_tokens > 0, f"非流式重复请求应命中缓存，但 cached_tokens={cached_tokens}"
