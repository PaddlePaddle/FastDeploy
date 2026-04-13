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

import openai
import pytest
import requests
from e2e.utils.rollout_routing_replay_test_utils import (
    check_routing_replay_chat_completion,
)
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
    print("log dir clean ")
    if os.path.exists("log") and os.path.isdir("log"):
        shutil.rmtree("log")
    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "GLM-4.5-Air")
    else:
        model_path = "./GLM-4.5-Air-Fake"

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
        "4",
        "--data-parallel-size",
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
        "1",
        "--graph-optimization-config",
        '{"use_cudagraph":true}',
        "--load-choices",
        "default_v1",
        "--lm_head-fp32",
        "--routing-replay-config",
        '{"enable_routing_replay":true, "routing_store_type":"local", "local_store_dir":"./R3_tmp/routing_replay_output_glm45air_tp4"}',
    ]
    env = os.environ.copy()
    # Start subprocess in new process group
    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            env=env,
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
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "temperature": 0.6,
        "top_p": 0,  # fix top_p to reduce randomness
        "seed": 13,  # fixed random seed
        "max_tokens": 20,
        "stream": False,
    }


# ==========================
# Test for lm_head_fp32 with fixed payload
# ==========================
def test_lm_head_fp32(api_url, headers, consistent_payload):
    """
    Test that two runs with the same fixed input produce similar outputs.
    """
    # First request
    response = requests.post(api_url, headers=headers, json=consistent_payload, timeout=300)
    assert response.status_code == 200
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    resp_json = response.json()

    # 校验返回内容与概率信息
    assert (
        resp_json["choices"][0]["message"]["content"]
        == "\n<think>我需要回答牛顿的三大运动定律是什么。牛顿的三大运动定律是经典"
    ), f"The response content is not as expected {resp_json['choices'][0]['message']['content']}."


# ==========================
# Test for Rollout Routing Replay
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


def test_r3_accuracy(openai_client):
    moe_layer_num = 45  # GLM45 AIR moe layer num: 45, Fake GLM AIR moe layer num: 1
    check_routing_replay_chat_completion(
        openai_client=openai_client, moe_layer_num=moe_layer_num, model_name="glm45air_tp4"
    )
