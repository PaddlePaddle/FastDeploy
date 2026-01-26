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
import subprocess
import sys
import time

import openai
import pytest

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

os.environ.setdefault("DG_NVCC_OVERRIDE_CPP_STANDARD", "17")

W4AFP8_CONFIGS = [
    {
        "id": "w4afp8_default_v1",
        "load_choices": "default_v1",
        "model_name": "ernie-4_5-21b-a3b-bf16-paddle",
        "model_subdir": None,
    },
    {
        "id": "w4afp8_default_v1",
        "load_choices": "default_v1",
        "model_name": "ERNIE-4.5-21B-A3B-PT",
        "model_subdir": "torch",
    },
    {
        "id": "w4afp8_default_v1",
        "load_choices": "default_v1",
        "model_name": "Qwen3-30B-A3B",
        "model_subdir": "torch",
    },
]


def get_model_path(config):
    """Get model path based on config and MODEL_PATH environment variable."""
    base_path = os.getenv("MODEL_PATH")
    model_name = config["model_name"]
    model_subdir = config.get("model_subdir")

    if base_path:
        if model_subdir:
            model_path = os.path.join(base_path, model_subdir, model_name)
        else:
            model_path = os.path.join(base_path, model_name)
    else:
        if model_subdir:
            model_path = os.path.join(".", model_subdir, model_name)
        else:
            model_path = f"./{model_name}"

    return model_path


@pytest.fixture(scope="module", params=W4AFP8_CONFIGS, ids=lambda x: x["id"])
def setup_w4afp8_server(request):
    """
    Setup W4AFP8 server for each config.
    This fixture is parameterized to run with different configurations.
    """
    config = request.param
    config_id = config["id"]
    load_choices = config["load_choices"]

    print(f"\n{'='*60}")
    print(f"Starting W4AFP8 server with config: {config_id}")
    print(f"  load_choices: {load_choices}")
    print(f"  api_port: {FD_API_PORT}")
    print(f"{'='*60}")

    # Clean ports before starting
    clean_ports()
    time.sleep(5)

    model_path = get_model_path(config)

    # Check model path exists
    print(f"Model path: {model_path}")
    if not os.path.exists(model_path):
        pytest.skip(f"Model path does not exist: {model_path}")

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
        str(FD_API_PORT),
        "--tensor-parallel-size",
        "2",
        "--engine-worker-queue-port",
        str(FD_ENGINE_QUEUE_PORT),
        "--metrics-port",
        str(FD_METRICS_PORT),
        "--cache-queue-port",
        str(FD_CACHE_QUEUE_PORT),
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

    print(f"Starting server with command: {' '.join(cmd)}")

    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "FD_LOG_DIR": log_dir},
        )

    print(f"Server process started with PID: {process.pid}")

    # Wait for server to start
    server_started = False
    for i in range(300):
        # Check if process is still alive
        if process.poll() is not None:
            print(f"[ERROR] Server process exited early with code: {process.returncode}")
            break

        if is_port_open("127.0.0.1", FD_API_PORT):
            print(f"API server [{config_id}] is up on port {FD_API_PORT}")
            server_started = True
            break

        if i % 30 == 0:
            print(f"Waiting for server [{config_id}] to start... ({i}s)")
        time.sleep(1)

    if not server_started:
        print(f"[TIMEOUT] API server [{config_id}] failed to start in 5 minutes.")

        # Print log content for debugging
        print(f"\n{'='*60}")
        print(f"Server log [{config_id}]:")
        print(f"{'='*60}")
        try:
            with open(log_path, "r") as f:
                log_content = f.read()
                # Print last 100 lines
                lines = log_content.split("\n")
                print("\n".join(lines[-100:]))
        except Exception as e:
            print(f"Failed to read log: {e}")
        print(f"{'='*60}\n")

        # Cleanup
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception as e:
            print(f"Failed to kill process group: {e}")

        clean_ports()
        raise RuntimeError(f"API server [{config_id}] did not start on port {FD_API_PORT}")

    yield {"process": process, "config": config}

    # Cleanup after test
    print(f"\n===== Cleanup W4AFP8 server [{config_id}]... =====")

    # Graceful shutdown
    try:
        process.terminate()
        process.wait(timeout=30)
        print(f"API server [{config_id}] (pid={process.pid}) terminated gracefully")
    except subprocess.TimeoutExpired:
        print(f"Timeout waiting for server [{config_id}], force killing...")
        try:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        except Exception as e:
            print(f"Failed to force kill: {e}")
    except Exception as e:
        print(f"Failed to terminate API server [{config_id}]: {e}")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except:
            pass

    # Clean ports after shutdown
    clean_ports()
    time.sleep(10)
    print(f"Cleanup [{config_id}] completed")


@pytest.fixture(scope="module")
def openai_client(setup_w4afp8_server):
    """
    Returns OpenAI client for W4AFP8 quantization service.
    Depends on setup_w4afp8_server to ensure server is running.
    """
    client = openai.OpenAI(
        base_url=f"http://127.0.0.1:{FD_API_PORT}/v1",
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
def consistent_payload():
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
def test_w4afp8_consistency_between_runs(openai_client, consistent_payload, current_config):
    """
    Test that two runs with the same fixed input produce similar outputs.
    This test runs for each W4AFP8 config (default and default_v1).
    """
    config_id = current_config["id"]
    load_choices = current_config["load_choices"]

    print(f"\n[{config_id}] Testing consistency with load_choices={load_choices}")

    # First request
    resp1 = openai_client.chat.completions.create(
        model="default",
        stream=False,
        max_tokens=256,
        **consistent_payload,
    )
    content1 = resp1.choices[0].message.content

    # Second request with same parameters
    resp2 = openai_client.chat.completions.create(
        model="default",
        stream=False,
        max_tokens=256,
        **consistent_payload,
    )
    content2 = resp2.choices[0].message.content

    # Check required keywords
    required_keywords = ["北京", "天安门"]
    for keyword in required_keywords:
        assert keyword in content1, (
            f"[{config_id}] First response missing keyword '{keyword}', " f"response content: {content1}"
        )
        assert keyword in content2, (
            f"[{config_id}] Second response missing keyword '{keyword}', " f"response content: {content2}"
        )

    # Check consistency between runs
    diff_rate = calculate_diff_rate(content1, content2)
    print(f"[{config_id}] Diff rate between two runs: {diff_rate:.4%}")

    assert diff_rate < 0.05, (
        f"[{config_id}] Output difference too large ({diff_rate:.4%})\n"
        f"Response 1: {content1}\n"
        f"Response 2: {content2}"
    )

    print(f"[{config_id}] Consistency test passed! Diff rate: {diff_rate:.4%}")
