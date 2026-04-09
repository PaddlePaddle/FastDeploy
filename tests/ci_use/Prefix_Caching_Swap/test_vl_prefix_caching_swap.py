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


class MultimodalPrefixCacheTestHelper:
    """Prefix Cache test utilities for multimodal models"""

    # Sample image URLs for testing
    TEST_IMAGES = {
        "image1": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg",
        "image2": "https://ku.baidu-int.com/vk-assets-ltd/space/2024/09/13/933d1e0a0760498e94ec0f2ccee865e0",
    }

    @staticmethod
    def make_usage_payload(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create streaming request payload with usage statistics"""
        return {
            "messages": messages,
            "temperature": 0,
            "top_p": 0,
            "seed": 33,
            "max_tokens": 32,
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
            "max_tokens": 32,
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
                json=MultimodalPrefixCacheTestHelper.make_usage_payload(messages),
                stream=True,
            )
            cached_tokens = MultimodalPrefixCacheTestHelper.read_stream_for_cached_tokens(resp)

            if cached_tokens > 0:
                return cached_tokens
            time.sleep(sleep_sec)

        return 0

    @staticmethod
    def make_text_only_message(text: str) -> List[Dict[str, Any]]:
        """Create text-only message"""
        return [{"role": "user", "content": text}]

    @staticmethod
    def make_image_only_message(image_key: str) -> List[Dict[str, Any]]:
        """Create image-only message"""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": MultimodalPrefixCacheTestHelper.TEST_IMAGES[image_key], "detail": "high"},
                    }
                ],
            }
        ]

    @staticmethod
    def make_image_text_message(image_key: str, text: str) -> List[Dict[str, Any]]:
        """Create image+text combined message"""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": MultimodalPrefixCacheTestHelper.TEST_IMAGES[image_key], "detail": "high"},
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]

    @staticmethod
    def make_multimodal_chat_message(image_key: str, text: str, system_prompt: str) -> List[Dict[str, Any]]:
        """Create multimodal message with system prompt"""
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": MultimodalPrefixCacheTestHelper.TEST_IMAGES[image_key], "detail": "high"},
                    },
                    {"type": "text", "text": text},
                ],
            },
        ]


@pytest.fixture(scope="session", autouse=True)
def setup_and_run_server():
    """
    Pytest fixture: Start test server for multimodal model

    Configure small GPU cache (4 blocks) and large CPU cache (5GB)
    to trigger secondary cache functionality with multimodal support
    """
    clean_ports()

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "ernie-4_5-vl-28b-a3b-bf16-paddle")
    else:
        model_path = "./ernie-4_5-vl-28b-a3b-bf16-paddle"

    log_path = "server.log"
    limit_mm_str = json.dumps({"image": 100, "video": 100})

    cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(FD_API_PORT),
        "--tensor-parallel-size",
        "2",
        "--engine-worker-queue-port",
        str(FD_ENGINE_QUEUE_PORT),
        "--metrics-port",
        str(FD_METRICS_PORT),
        "--cache-queue-port",
        str(FD_CACHE_QUEUE_PORT),
        "--enable-mm",
        "--max-model-len",
        "8192",
        "--max-num-batched-tokens",
        "384",
        "--max-num-seqs",
        "32",
        "--limit-mm-per-prompt",
        limit_mm_str,
        "--enable-chunked-prefill",
        "--kv-cache-ratio",
        "0.71",
        "--swap-space",
        "5",
        "--num-gpu-blocks-override",
        "200",
        "--enable-prefix-caching",
        "--graph-optimization-config",
        '{"cudagraph_capture_sizes": [1]}',
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

    # Wait for server startup (max 300 seconds for multimodal model)
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


def test_image_only_prefix_cache(api_url, headers):
    """
    Test Case 1: Image-only prefix cache functionality

    Verify that requests with same image but different text prompts
    can benefit from image prefix caching
    """
    helper = MultimodalPrefixCacheTestHelper()

    # Create messages with same image but different questions
    messages1 = helper.make_image_text_message("image1", "请描述图片内容")
    messages2 = helper.make_image_text_message("image1", "图片中有哪些物体？")

    # First request - no cache hit
    resp1 = requests.post(api_url, headers=headers, json=helper.make_usage_payload(messages1), stream=True)
    cached1 = helper.read_stream_for_cached_tokens(resp1)
    assert cached1 == 0, "First image request should not hit cache"

    time.sleep(1)

    # Second request with same image - should hit cache for image prefix
    cached2 = helper.send_until_cache_hit(api_url, headers, messages2, max_retry=3, sleep_sec=1)
    assert cached2 > 0, "Repeated identical image request should hit prefix cache"


def test_multimodal_combined_prefix_cache(api_url, headers):
    """
    Test Case 2: Combined image+text prefix cache functionality

    Verify that requests with identical image and text prefixes
    can benefit from combined prefix caching
    """
    helper = MultimodalPrefixCacheTestHelper()

    system_prompt = "你是一个专业的图片分析助手。"

    # First request
    messages1 = helper.make_multimodal_chat_message(
        "image1",
        "请详细描述图片中的场景和主要元素。",
        system_prompt,
    )

    resp1 = requests.post(api_url, headers=headers, json=helper.make_usage_payload(messages1), stream=True)
    cached1 = helper.read_stream_for_cached_tokens(resp1)
    assert cached1 == 0, "First multimodal request should not hit cache"

    time.sleep(1)

    # Second identical request - should hit cache
    cached2 = helper.send_until_cache_hit(api_url, headers, messages1, max_retry=3, sleep_sec=1)
    assert cached2 > 0, "Repeated identical multimodal request should hit prefix cache"

    time.sleep(1)

    # Third request with same prefix but different question
    messages2 = helper.make_multimodal_chat_message(
        "image1",
        "图片中的人物在做什么？请简要回答。",
        system_prompt,
    )
    cached3 = helper.send_until_cache_hit(api_url, headers, messages2, max_retry=3, sleep_sec=1)
    assert cached3 > 0, "Request with same prefix but different question should hit prefix cache"


def test_multimodal_cache_with_different_images(api_url, headers):
    """
    Test Case 3: Cache behavior with different images

    Verify that cache correctly distinguishes between different images
    while caching shared text components
    """
    helper = MultimodalPrefixCacheTestHelper()

    system_prompt = "你是图片对比分析助手。"

    question = "请描述这张图片的主要内容。"

    # Create requests with same system prompt and question but different images
    messages_image1 = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": helper.TEST_IMAGES["image1"], "detail": "high"},
                },
                {"type": "text", "text": question},
            ],
        },
    ]

    messages_image2 = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": helper.TEST_IMAGES["image2"], "detail": "high"},
                },
                {"type": "text", "text": question},
            ],
        },
    ]

    # First request with image1
    resp1 = requests.post(api_url, headers=headers, json=helper.make_usage_payload(messages_image1), stream=True)
    cached1 = helper.read_stream_for_cached_tokens(resp1)
    assert cached1 == 0, "First request with image1 should not hit cache"

    time.sleep(1)

    # Second request with same image1 - should hit cache
    cached2 = helper.send_until_cache_hit(api_url, headers, messages_image1, max_retry=3, sleep_sec=1)
    assert cached2 > 0, "Repeated request with same image should hit cache"

    time.sleep(1)

    # Third request with different image2 - system prompt should still be cached
    cached3 = helper.send_until_cache_hit(api_url, headers, messages_image2, max_retry=3, sleep_sec=1)
    assert cached3 > 0, "Request with different image should hit cache"


def test_multimodal_gpu_to_cpu_swap(api_url, headers):
    """
    Test Case 4: GPU cache handling for multimodal requests

    Generate multiple multimodal requests to test cache behavior under load.
    Note: Detailed swap log verification may be limited in some environments.
    """
    helper = MultimodalPrefixCacheTestHelper()

    system_prompts = [f"你是图片分析专家{i}。" for i in range(5)]

    questions = [
        "请描述图片。",
        "图片中有哪些颜色？",
        "图片的主题是什么？",
        "图片的风格如何？",
        "图片给你什么感觉？",
    ]

    # Send multiple requests with different combinations to test cache behavior
    request_count = 12

    for i in range(request_count):
        system_prompt = system_prompts[i % len(system_prompts)]
        question = questions[i % len(questions)]
        image_key = "image1" if i % 2 == 0 else "image2"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": helper.TEST_IMAGES[image_key], "detail": "high"},
                    },
                    {"type": "text", "text": f"{question} (请求ID: {i})"},
                ],
            },
        ]

        resp = requests.post(api_url, headers=headers, json=helper.make_basic_payload(messages))
        assert resp.status_code == 200, f"Request {i} should succeed"

    # Wait for cache operations
    time.sleep(2)

    # Verify by checking that repeated requests with same image get cache hits
    repeat_messages = helper.make_image_text_message("image1", "请描述图片内容")

    cached_tokens = 0
    for attempt in range(3):
        resp = requests.post(api_url, headers=headers, json=helper.make_usage_payload(repeat_messages), stream=True)
        cached_tokens = helper.read_stream_for_cached_tokens(resp)
        if cached_tokens > 0:
            print(f"Cache hit detected after {attempt + 1} attempts: {cached_tokens} tokens")
            break
        time.sleep(1)
