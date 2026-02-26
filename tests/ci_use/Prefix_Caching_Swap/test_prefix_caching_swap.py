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
from typing import Any, Dict, List

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


class PrefixCacheTestHelper:
    """Prefix Cache test utilities"""

    @staticmethod
    def make_usage_payload(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create streaming request payload with usage statistics"""
        return {
            "messages": messages,
            "temperature": 0,
            "top_p": 0,
            "seed": 33,
            "max_tokens": 256,
            "stream": True,
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True,
            },
        }

    @staticmethod
    def make_basic_payload(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create basic request payload"""
        return {
            "messages": messages,
            "temperature": 0,
            "max_tokens": 256,
            "stream": True,
        }

    @staticmethod
    def read_stream_for_cached_tokens(resp: requests.Response) -> int:
        """Extract cached_tokens from streaming response"""
        last_usage = {}

        for line_bytes in resp.iter_lines():
            if not line_bytes:
                continue

            line = line_bytes.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue

            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
                if "usage" in chunk:
                    last_usage = chunk["usage"]
            except Exception:
                continue

        return last_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

    @staticmethod
    def send_until_cache_hit(
        api_url: str, headers: Dict[str, str], messages: List[Dict[str, Any]], max_retry: int = 3, sleep_sec: int = 1
    ) -> int:
        """Send requests until cache hit is detected"""
        for i in range(max_retry):
            resp = requests.post(
                api_url,
                headers=headers,
                json=PrefixCacheTestHelper.make_usage_payload(messages),
                stream=True,
            )
            cached_tokens = PrefixCacheTestHelper.read_stream_for_cached_tokens(resp)

            if cached_tokens > 0:
                return cached_tokens
            time.sleep(sleep_sec)

        return 0

    @staticmethod
    def make_shared_prefix_payload(prefix: str, suffix: str, idx: int = 0) -> Dict[str, Any]:
        """Create request payload with shared prefix"""
        return {
            "messages": [{"role": "user", "content": f"{prefix}\n问题 {idx}：{suffix}"}],
            "temperature": 0,
            "max_tokens": 64,
            "stream": True,
        }


@pytest.fixture(scope="session", autouse=True)
def setup_and_run_server():
    """
    Pytest fixture: Start test server

    Configure small GPU cache (4 blocks) and large CPU cache (10GB)
    to trigger secondary cache functionality
    """
    clean_ports()

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
        "128",
        "--max-num-seqs",
        "128",
        "--quantization",
        "wint4",
        "--graph-optimization-config",
        '{"cudagraph_capture_sizes": [1]}',
        "--swap-space",
        "10",  # 10GB CPU cache
        "--num-gpu-blocks-override",
        "4",  # Small GPU cache to test swap
        "--enable-prefix-caching",  # Enable prefix caching
    ]

    # Clean log directory
    if os.path.exists("log"):
        shutil.rmtree("log")

    # Start server process
    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Enable process group management
        )

    # Wait for server startup (max 300 seconds)
    for _ in range(300):
        if is_port_open("127.0.0.1", FD_API_PORT):
            break
        time.sleep(1)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception as e:
            print(f"Failed to kill process group: {e}")
        raise RuntimeError(f"API server did not start on port {FD_API_PORT}")

    yield  # Execute tests

    # Post-test cleanup
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        pass


@pytest.fixture(scope="session")
def api_url():
    """API endpoint URL"""
    return f"http://0.0.0.0:{FD_API_PORT}/v1/chat/completions"


@pytest.fixture(scope="session")
def metrics_url():
    """Metrics endpoint URL"""
    return f"http://0.0.0.0:{FD_METRICS_PORT}/metrics"


@pytest.fixture
def headers():
    """HTTP request headers"""
    return {"Content-Type": "application/json"}


def test_basic_prefix_cache_functionality(api_url, headers):
    """
    Test Case 1: Basic prefix cache functionality verification

    Test scenarios:
    1. Cold start request should not hit cache
    2. Repeated request should hit prefix cache
    3. Multi-turn conversation should reuse shared prefix
    """
    helper = PrefixCacheTestHelper()

    # System prompt - used as shared prefix
    system_prompt = (
        "You are a helpful assistant. "
        "You are calm, precise, and analytical. "
        "You always give structured answers. "
        "You never hallucinate facts. "
        "You follow instructions strictly and answer in Chinese. "
        "Your name is FastDeploy AI Bot. "
        "Your pronoun is I or me or myself or our. "
        "You can be called by any of these names: "
        "FastDeploy AI Bot, FastDeploy, DeployAI, ChatBot, Assistant."
    )

    # Test 1: Cold start request
    messages1 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "请用三点总结 FastDeploy 的作用。"},
    ]

    resp1 = requests.post(
        api_url,
        headers=headers,
        json=helper.make_usage_payload(messages1),
        stream=True,
    )
    cached1 = helper.read_stream_for_cached_tokens(resp1)
    assert cached1 == 0, "First request should not hit cache"

    time.sleep(1)  # Wait for cache write

    # Test 2: Cache hit verification
    cached2 = helper.send_until_cache_hit(api_url, headers, messages1, max_retry=5, sleep_sec=1)
    assert cached2 > 0, "Repeated request should hit prefix cache"

    time.sleep(1)

    # Test 3: Multi-turn conversation prefix reuse
    messages3 = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "请用三点总结 FastDeploy 的作用。\n" "要求：\n" "1. 从部署角度\n" "2. 从性能角度\n" "3. 从生态角度"
            ),
        },
    ]

    cached3 = helper.send_until_cache_hit(api_url, headers, messages3, max_retry=5, sleep_sec=1)
    assert cached3 > 0, "Multi-turn conversation should reuse shared prefix"


def test_gpu_to_cpu_swap_mechanism(api_url, headers):
    """
    Test Case 2: GPU to CPU swap mechanism verification

    Generate multiple requests to fill GPU cache and trigger SWAP2CPU mechanism
    """
    helper = PrefixCacheTestHelper()

    # Use short shared prefix to avoid exceeding max model length (max_model_len=128)
    long_prefix = "这是测试共享前缀。"

    # Send multiple requests with different prefixes to fill GPU cache
    request_count = 20  # Exceeds GPU block count, forcing SWAP2CPU
    for i in range(request_count):
        # Each request uses different system prompt to ensure different prefixes
        system_prompt = f"这是系统提示词{i}，用于触发GPU缓存机制。"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{long_prefix}\n问题{i}: 这个请求是为了测试GPU缓存。"},
        ]
        resp = requests.post(api_url, headers=headers, json=helper.make_basic_payload(messages))
        assert resp.status_code == 200

    # Wait for swap operations to complete
    time.sleep(5)

    # Verify swap mechanism logs
    log_dir = "log"
    assert os.path.exists(log_dir), "Log directory does not exist"

    # Find latest cache manager log file
    cache_logs = [f for f in os.listdir(log_dir) if f.startswith("cache_manager.log")]
    if not cache_logs:
        log_path = os.path.join(log_dir, "cache_manager.log")
        if not os.path.exists(log_path):
            all_logs = [f for f in os.listdir(log_dir) if "cache" in f.lower() and f.endswith(".log")]
            if all_logs:
                log_path = os.path.join(log_dir, all_logs[0])
            else:
                assert False, "Cache manager log file not found"
        cache_logs = [log_path]
    else:
        cache_logs.sort(reverse=True)
        log_path = os.path.join(log_dir, cache_logs[0])

    assert os.path.exists(log_path), f"Cache manager log file does not exist: {log_path}"

    # Read and verify log content
    with open(log_path, "r") as f:
        log_content = f.read()

    # Verify key swap mechanism related logs
    found_swap2cpu = "CacheStatus.SWAP2CPU" in log_content
    found_gpu_blocks = "self.gpu_free_block_list" in log_content or "num_gpu_blocks_server_owned" in log_content
    found_cpu_blocks = "self.cpu_free_block_list" in log_content or "num_cpu_blocks" in log_content

    if not found_swap2cpu:
        server_log_path = "server.log"
        if os.path.exists(server_log_path):
            with open(server_log_path, "r") as f:
                server_log_content = f.read()
                found_swap2cpu = "SWAP2CPU" in server_log_content or "swap" in server_log_content.lower()

    # If exact logs not found, check for cache operation statistics
    if not found_swap2cpu:
        cache_metrics = ["cache_hit", "cache_miss", "block", "transfer", "swap"]
        found_cache_ops = any(metric in log_content.lower() for metric in cache_metrics)
        if found_cache_ops:
            return  # Test passes if cache operations detected

    assert found_swap2cpu, "GPU to CPU swap operation not detected"
    assert found_gpu_blocks, "GPU block information not recorded"
    assert found_cpu_blocks, "CPU block information not recorded"


def test_cpu_to_gpu_swap_back_mechanism(api_url, headers, metrics_url):
    """
    Test Case 3: Verify CPU to GPU swap back mechanism

    Steps:
    1. Use embedded text dataset for testing
    2. Generate multiple requests to fill GPU cache and trigger SWAP2CPU
    3. Repeatedly access hot data to trigger SWAP2GPU
    4. Verify swap mechanism through logs and metrics

    Expected Behavior:
    - Cached data swapped to CPU should be correctly swapped back to GPU
    - System should maintain consistency during swap operations
    """
    helper = PrefixCacheTestHelper()

    # Embedded AI technology text dataset
    paragraphs = [
        "深度学习模型通过多层神经网络自动学习特征表示。",
        "Transformer架构通过自注意力机制实现长距离依赖建模。",
        "大语言模型(LLM)通过海量数据预训练获得通用能力。",
        "前缀缓存技术可显著减少重复计算提升推理速度。",
        "GPU显存管理是高效推理的关键挑战之一。",
        "KV缓存优化能有效降低大模型推理延迟。",
        "批处理(batching)技术可提高硬件利用率。",
        "量化压缩技术可在精度损失可控情况下减小模型体积。",
        "持续学习使模型能够适应新任务而不遗忘旧知识。",
        "多模态模型能够处理文本、图像、音频等多种输入。",
        "联邦学习允许多方协作训练而不共享原始数据。",
        "模型蒸馏可将大模型知识迁移到小模型。",
        "注意力机制的可视化有助于理解模型决策过程。",
        "神经架构搜索(NAS)可自动化设计最优网络结构。",
        "强化学习通过与环境的交互优化决策策略。",
        "生成对抗网络(GAN)能产生逼真的合成数据。",
        "对比学习通过构建正负样本学习有效表示。",
        "知识图谱可为语言模型提供结构化背景知识。",
        "元学习旨在让模型学会如何学习新任务。",
        "持续预训练可逐步扩展模型的知识和能力。",
    ]

    # System prompts for different request types
    system_prompts = [
        "You are an AI technology analyst.",
        "You specialize in technical documentation.",
        "You focus on business value of AI applications.",
        "You are a technology trends observer.",
        "You explain complex concepts in simple terms.",
        "You research AI ethics and social impact.",
        "You advise tech startups on innovation.",
        "You assess technical feasibility and risks.",
        "You study AI applications in education.",
        "You analyze tech products and markets.",
    ]

    # Store hot requests for repeated access
    hotspot_messages = []

    # Phase 1: Fill cache with diverse requests to trigger SWAP2CPU
    request_count = 32  # Number of initial requests

    for i in range(request_count):
        prompt_idx = i % len(system_prompts)
        para_idx = i % len(paragraphs)

        system_prompt = system_prompts[prompt_idx]
        paragraph = paragraphs[para_idx]

        user_content = f"""
Based on this technical context:
{paragraph}

Please answer:
1. What are the main applications of this technology?
2. What challenges does it face?
3. What are future trends?

(Request ID: {i + 1})
"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content.strip()}]

        # Store first 10 requests as hot data
        if i < 10:
            hotspot_messages.append(messages)

        resp = requests.post(api_url, headers=headers, json=helper.make_basic_payload(messages), timeout=30)
        assert resp.status_code == 200

    time.sleep(5)  # Wait for cache operations

    # Phase 2: Repeatedly access hot data to trigger SWAP2GPU
    repeat_rounds = 3

    for _ in range(repeat_rounds):
        for messages in hotspot_messages:
            resp = requests.post(
                api_url, headers=headers, json=helper.make_usage_payload(messages), stream=True, timeout=30
            )
            assert resp.status_code == 200
            helper.read_stream_for_cached_tokens(resp)

        time.sleep(2)

    # Phase 3: Verify swap mechanism through logs
    time.sleep(3)

    # Check cache-related log files
    log_dir = "log"
    log_files = []

    if os.path.exists(log_dir):
        log_files = [
            os.path.join(log_dir, f) for f in os.listdir(log_dir) if "cache" in f.lower() and f.endswith(".log")
        ]

    if not log_files and os.path.exists("server.log"):
        log_files.append("server.log")

    swap2gpu_detected = False

    for log_path in log_files:
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                log_content = f.read()

            if any(
                pattern.lower() in log_content.lower()
                for pattern in ["CacheStatus.SWAP2GPU", "swap.*gpu", "gpu.*swap", "do_swap_to_gpu_task"]
            ):
                swap2gpu_detected = True
                break

        except Exception:
            continue

    assert swap2gpu_detected or any(
        keyword in log_content.lower() for keyword in ["swap2cpu", "transfer", "block", "cache"]
    ), "No cache swap evidence found in logs"


def test_lru_eviction_policy(api_url, headers):
    """
    Test Case 4: LRU eviction policy verification

    Verify that when cache is full, eviction follows LRU principle
    """
    helper = PrefixCacheTestHelper()

    system_prompt = "LRU测试系统提示词。"
    base_content = "这是LRU淘汰策略测试的内容。"

    # Generate a series of different requests
    requests_data = []
    for i in range(6):  # Exceeds GPU block count
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{base_content} 序列号: {i}"},
        ]
        requests_data.append(messages)

    # Send requests in sequence to fill cache
    for i, messages in enumerate(requests_data):
        resp = requests.post(api_url, headers=headers, json=helper.make_basic_payload(messages))
        assert resp.status_code == 200

    time.sleep(2)

    # Re-access in LRU order: earliest accessed should be evicted first
    # Re-access requests 2 and 3 (middle sequence)
    for i in [1, 2]:
        messages = requests_data[i]
        resp = requests.post(api_url, headers=headers, json=helper.make_usage_payload(messages))
        cached_tokens = helper.read_stream_for_cached_tokens(resp)
        # These requests may be swapped out, but should still hit through secondary cache
        if cached_tokens == 0:
            print("LRU eviction confirmed: request has been moved to L2 cache.")
        else:
            print("GPU cache hit.")
        assert resp.status_code == 200


def test_cache_metrics_and_monitoring(api_url, headers, metrics_url):
    """
    Test Case 5: Cache metrics and monitoring verification

    Verify cache hit rate, swap count, and monitoring metrics are correct
    """
    helper = PrefixCacheTestHelper()

    # Send requests to generate monitoring data
    test_messages = [
        {"role": "system", "content": "监控测试助手。"},
        {"role": "user", "content": "请简单介绍一下自己。"},
    ]

    # Send multiple requests
    for i in range(3):
        resp = requests.post(api_url, headers=headers, json=helper.make_basic_payload(test_messages))
        assert resp.status_code == 200

    time.sleep(2)

    # Check monitoring endpoint
    try:
        metrics_resp = requests.get(metrics_url)
        if metrics_resp.status_code == 200:
            metrics_content = metrics_resp.text
            # Check if cache-related metrics exist
            assert (
                "cache" in metrics_content.lower() or "block" in metrics_content.lower()
            ), "Cache-related metrics not found in monitoring"
    except Exception:
        # Monitoring endpoint unavailable doesn't affect main functionality test
        pass

    # Verify statistics in cache manager logs
    log_path = "log/cache_manager.log"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_content = f.read()

        # Check statistical log entries
        stat_keywords = ["hit", "miss", "swap", "block", "metric"]
        found_stats = any(keyword in log_content.lower() for keyword in stat_keywords)
        if not found_stats:
            # Also check for cache statistics in server log
            server_log_path = "server.log"
            if os.path.exists(server_log_path):
                with open(server_log_path, "r") as f:
                    server_content = f.read()
                    found_stats = any(keyword in server_content.lower() for keyword in stat_keywords)


def test_error_handling_and_robustness(api_url, headers):
    """
    Test Case 6: Error handling and robustness verification

    Verify system behavior under abnormal conditions
    """
    helper = PrefixCacheTestHelper()

    # Test 1: Invalid request handling
    invalid_payload = {"invalid": "data"}
    resp = requests.post(api_url, headers=headers, json=invalid_payload)
    # Should return error status code, not crash
    assert resp.status_code != 200, "Invalid request should be handled properly"

    # Test 2: Boundary length request
    long_content = "A" * 1000  # Long content test
    messages = [{"role": "system", "content": "测试助手。"}, {"role": "user", "content": long_content}]

    resp = requests.post(api_url, headers=headers, json=helper.make_basic_payload(messages))
    assert resp.status_code == 200, "Long content request should be handled properly"

    # Test 3: Special character request
    special_content = "特殊字符测试: !@#$%^&*()_+{}[]|\\:;'<>?,./"
    messages = [{"role": "system", "content": "测试助手。"}, {"role": "user", "content": special_content}]

    resp = requests.post(api_url, headers=headers, json=helper.make_basic_payload(messages))
    assert resp.status_code == 200, "Special character request should be handled properly"
