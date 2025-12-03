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

import pytest
import requests
from e2e.utils.serving_utils import (
    FD_API_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    clean_ports,
    is_port_open,
)

# ==========================
# Shared Helper Functions
# ==========================


def _start_server_process(enable_caching: bool, log_filename: str):

    print(f"\n[Server Setup] Cleaning ports before starting (Caching={'ON' if enable_caching else 'OFF'})...")
    clean_ports()

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "RM_v1008_5")
    else:
        model_path = "./RM_v1008_5"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

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
        "8192",
        "--max-num-seqs",
        "256",
        "--runner",
        "pooling",
        "--convert",
        "embed",
    ]

    if enable_caching:
        cmd.append("--enable-prefix-caching")
    else:
        cmd.append("--no-enable-prefix-caching")

    print(f"[Server Setup] Command: {' '.join(cmd)}")

    with open(log_filename, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    # Wait for server to start
    for _ in range(300):
        if is_port_open("127.0.0.1", FD_API_PORT):
            print(f"[Server Setup] Server is up on port {FD_API_PORT}")
            break
        time.sleep(1)
    else:
        print("[Server Setup] Server failed to start. Cleaning up...")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            pass
        if os.path.exists(log_filename):
            with open(log_filename, "r") as f:
                print(f"Server Log Tail ({log_filename}):\n{f.read()[-500:]}")
        raise RuntimeError(f"Server did not start on port {FD_API_PORT}")

    return process


@pytest.fixture(scope="function")
def reward_api_url():
    """Returns the API endpoint URL for reward."""
    return f"http://0.0.0.0:{FD_API_PORT}/v1/reward"


@pytest.fixture(scope="function")
def headers():
    """Returns common HTTP request headers."""
    return {"Content-Type": "application/json"}


@pytest.fixture(scope="function")
def server_default_caching():
    _start_server_process(enable_caching=True, log_filename="reward_server_caching_on.log")


@pytest.fixture(scope="function")
def server_no_caching():
    _start_server_process(enable_caching=False, log_filename="reward_server_caching_off.log")


def save_score_baseline(score: float, baseline_file: str):
    """Save reward score to baseline file."""
    baseline_data = {"score": score}
    with open(baseline_file, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)
    print(f"Baseline saved to: {baseline_file}")


def check_score_against_baseline(current_score: float, baseline_file: str, threshold: float = 0.01):
    """Check reward score against baseline file."""
    try:
        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
            baseline_score = baseline_data["score"]
    except FileNotFoundError:
        print(f"Baseline file not found: {baseline_file}. Saving current as baseline.")
        save_score_baseline(current_score, baseline_file)
        return

    diff = abs(current_score - baseline_score)
    print(f"Score Difference: {diff:.6f} (Current: {current_score}, Baseline: {baseline_score})")

    if diff >= threshold:
        temp_file = f"{baseline_file}.current"
        save_score_baseline(current_score, temp_file)
        raise AssertionError(
            f"Score differs from baseline by too much (diff={diff:.6f} >= {threshold}):\n"
            f"Current score saved to: {temp_file}"
        )


def _run_test_logic(reward_api_url, headers, baseline_filename):
    payload = {
        "model": "default",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "北京天安门在哪里？"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "北京天安门在中国北京故宫的前面。"}]},
        ],
        "user": "user-123",
        "enable_thinking": False,
    }

    print(f"\n=== Sending request to {reward_api_url} ===")
    response = requests.post(reward_api_url, headers=headers, json=payload, timeout=30)
    assert response.status_code == 200, f"API request failed with status {response.status_code}: {response.text}"

    result = response.json()
    print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")

    assert "data" in result and len(result["data"]) > 0
    score = float(result["data"][0]["score"][0])
    print(f"✓ Reward Score: {score}")

    base_path = os.getenv("MODEL_PATH", "")
    if base_path:
        baseline_file = os.path.join(base_path, baseline_filename)
    else:
        baseline_file = baseline_filename

    check_score_against_baseline(score, baseline_file, threshold=0.01)


def test_reward_model_with_caching(server_default_caching, reward_api_url, headers):
    print("\n>>> Running Test: WITH Prefix Caching")
    _run_test_logic(reward_api_url, headers, baseline_filename="reward_score_baseline.json")


def test_reward_model_without_caching(server_no_caching, reward_api_url, headers):
    print("\n>>> Running Test: WITHOUT Prefix Caching")
    _run_test_logic(reward_api_url, headers, baseline_filename="reward_score_baseline_no_caching.json")
