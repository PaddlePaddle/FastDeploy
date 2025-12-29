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
import shutil
import signal
import socket
import subprocess
import sys
import time

import openai
import pytest
from utils.serving_utils import (
    FD_API_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    is_port_open,
)

W4AFP8_CONFIGS = [
    {
        "id": "w4afp8_default",
        "load_choices": "default",
        "model_name": "ernie-4_5-21b-a3b-bf16-paddle",
        "model_subdir": None,
        "api_port": FD_API_PORT + 100,
        "engine_queue_port": FD_ENGINE_QUEUE_PORT + 100,
        "metrics_port": FD_METRICS_PORT + 100,
        "cache_queue_port": FD_CACHE_QUEUE_PORT + 100,
    },
    {
        "id": "w4afp8_default_v1",
        "load_choices": "default_v1",
        "model_name": "ERNIE-4.5-21B-A3B-PT",
        "model_subdir": "torch",
        "api_port": FD_API_PORT + 200,
        "engine_queue_port": FD_ENGINE_QUEUE_PORT + 200,
        "metrics_port": FD_METRICS_PORT + 200,
        "cache_queue_port": FD_CACHE_QUEUE_PORT + 200,
    },
]


def clean_ports_for_config(config):
    """Clean ports used by specific W4AFP8 config"""
    ports_to_clean = [
        config["api_port"],
        config["engine_queue_port"],
        config["metrics_port"],
        config["cache_queue_port"],
    ]
    for port in ports_to_clean:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            if result == 0:
                subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
        except Exception as e:
            print(f"清理端口 {port} 时出错: {e}")


def get_model_path(config):
    base_path = os.getenv("MODEL_PATH")
    model_name = config["model_name"]
    model_subdir = config.get("model_subdir")

    if base_path:
        if model_subdir:
            # 例如: MODEL_PATH/torch/ERNIE-4.5-21B-A3B-PT
            model_path = os.path.join(base_path, model_subdir, model_name)
        else:
            # 例如: MODEL_PATH/ernie-4_5-21b-a3b-bf16-paddle
            model_path = os.path.join(base_path, model_name)
    else:
        if model_subdir:
            model_path = os.path.join(".", model_subdir, model_name)
        else:
            model_path = f"./{model_name}"

    return model_path


@pytest.fixture(scope="module", params=W4AFP8_CONFIGS, ids=lambda x: x["id"], autouse=True)
def setup_w4afp8_server(request):
    config = request.param
    config_id = config["id"]
    load_choices = config["load_choices"]
    api_port = config["api_port"]
    engine_queue_port = config["engine_queue_port"]
    metrics_port = config["metrics_port"]
    cache_queue_port = config["cache_queue_port"]

    print(f"\n{'='*60}")
    print(f"Starting W4AFP8 server with config: {config_id}")
    print(f"  load_choices: {load_choices}")
    print(f"  api_port: {api_port}")
    print(f"{'='*60}")

    clean_ports_for_config(config)
    time.sleep(5)

    model_path = get_model_path(config)

    log_path = f"server_{config_id}.log"
    log_dir = f"log_{config_id}"

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(api_port),
        "--tensor-parallel-size",
        "2",
        "--engine-worker-queue-port",
        str(engine_queue_port),
        "--metrics-port",
        str(metrics_port),
        "--cache-queue-port",
        str(cache_queue_port),
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "128",
        "--quantization",
        "w4afp8",
        "--load-choices",
        load_choices,
        "--graph-optimization-config",
        '{"cudagraph_capture_sizes": [1]}',
    ]

    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "FD_LOG_DIR": log_dir},
        )

    for i in range(300):
        if is_port_open("127.0.0.1", api_port):
            print(f"API server [{config_id}] is up on port {api_port}")
            break
        if i % 30 == 0:
            print(f"Waiting for server [{config_id}] to start... ({i}s)")
        time.sleep(1)
    else:
        print(f"[TIMEOUT] API server [{config_id}] failed to start.")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception as e:
            print(f"Failed to kill process group: {e}")
        raise RuntimeError(f"API server [{config_id}] did not start on port {api_port}")

    yield {"process": process, "config": config}

    print(f"\n===== Cleanup W4AFP8 server [{config_id}]... =====")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
        print(f"API server [{config_id}] (pid={process.pid}) terminated")
    except Exception as e:
        print(f"Failed to terminate API server [{config_id}]: {e}")

    clean_ports_for_config(config)
    time.sleep(10)


@pytest.fixture(scope="module")
def openai_client_w4afp8(setup_w4afp8_server):
    """
    Returns OpenAI client for W4AFP8 quantization service.
    """
    config = setup_w4afp8_server["config"]
    api_port = config["api_port"]
    ip = "127.0.0.1"
    client = openai.OpenAI(
        base_url=f"http://{ip}:{api_port}/v1",
        api_key="EMPTY_API_KEY",
    )
    return client


@pytest.fixture(scope="module")
def current_config(setup_w4afp8_server):
    """
    Returns the current server config for the test module.
    """
    return setup_w4afp8_server["config"]


@pytest.fixture
def consistent_payload_w4afp8():
    """
    Returns a fixed payload for consistency testing,
    including a fixed random seed and temperature.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": "北京天安门在哪里?",
            }
        ],
        "temperature": 0.8,
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
# Test Cases
# ==========================
def test_w4afp8_consistency_between_runs(openai_client_w4afp8, consistent_payload_w4afp8, current_config):
    """
    Test that two runs with the same fixed input produce similar outputs.
    This test runs for each W4AFP8 config (default and default_v1).
    """
    config_id = current_config["id"]
    load_choices = current_config["load_choices"]

    print(f"\n[{config_id}] Testing consistency with load_choices={load_choices}")

    resp1 = openai_client_w4afp8.chat.completions.create(
        model="default",
        stream=False,
        max_tokens=256,
        **consistent_payload_w4afp8,
    )
    content1 = resp1.choices[0].message.content

    resp2 = openai_client_w4afp8.chat.completions.create(
        model="default",
        stream=False,
        max_tokens=256,
        **consistent_payload_w4afp8,
    )
    content2 = resp2.choices[0].message.content

    required_keywords = ["北京", "天安门"]
    for keyword in required_keywords:
        assert (
            keyword in content1
        ), f"[{config_id}] First response missing keyword '{keyword}', response content: {content1}"
        assert (
            keyword in content2
        ), f"[{config_id}] Second response missing keyword '{keyword}', response content: {content2}"

    diff_rate = calculate_diff_rate(content1, content2)
    assert diff_rate < 0.05, f"[{config_id}] Output difference too large ({diff_rate:.4%})"

    print(f"[{config_id}] Consistency test passed! Diff rate: {diff_rate:.4%}")
