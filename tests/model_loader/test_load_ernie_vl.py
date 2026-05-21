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
import signal
import subprocess
import sys
import time

import openai
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from e2e.utils.serving_utils import (
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
        "32768",
        "--max-num-batched-tokens",
        "384",
        "--max-num-seqs",
        "128",
        "--limit-mm-per-prompt",
        limit_mm_str,
        "--enable-chunked-prefill",
        "--kv-cache-ratio",
        "0.71",
        "--reasoning-parser",
        "ernie-45-vl",
        "--load-choices",
        "default_v1",
    ]

    # Start subprocess in new process group
    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Enables killing full group via os.killpg
        )

    # Wait up to 5 minutes for API server to be ready
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


# ==========================
# OpenAI Client Chat Completion Test
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
            {
                "role": "system",
                "content": "You are a helpful AI assistant.",
            },  # system不是必需，可选
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://ku.baidu-int.com/vk-assets-ltd/space/2024/09/13/933d1e0a0760498e94ec0f2ccee865e0",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "请描述图片内容"},
                ],
            },
        ],
        temperature=1,
        max_tokens=53,
        stream=False,
    )

    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], "message")
    assert hasattr(response.choices[0].message, "content")
