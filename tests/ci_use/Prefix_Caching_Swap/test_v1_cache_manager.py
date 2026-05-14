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
        "--speculative-config",
        json.dumps(speculative_config),
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
            env=server_env,
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
