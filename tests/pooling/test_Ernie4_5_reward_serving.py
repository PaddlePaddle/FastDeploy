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
import socket
import subprocess
import sys
import time

import pytest
import requests

# Read ports from environment variables
FD_API_PORT = int(os.getenv("FD_API_PORT", 8189))
FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8134))
FD_METRICS_PORT = int(os.getenv("FD_METRICS_PORT", 8234))
FD_CACHE_QUEUE_PORT = int(os.getenv("FD_CACHE_QUEUE_PORT", 8334))

PORTS_TO_CLEAN = [FD_API_PORT, FD_ENGINE_QUEUE_PORT, FD_METRICS_PORT, FD_CACHE_QUEUE_PORT]


def is_port_open(host: str, port: int, timeout=1.0):
    """Check if a TCP port is open."""
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except Exception:
        return False


def kill_process_on_port(port: int):
    """Kill processes listening on the given port."""
    try:
        output = subprocess.check_output(f"lsof -i:{port} -t", shell=True).decode().strip()
        for pid in output.splitlines():
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed process on port {port}, pid={pid}")
    except subprocess.CalledProcessError:
        pass


def clean_ports():
    """Clean all ports in PORTS_TO_CLEAN."""
    for port in PORTS_TO_CLEAN:
        kill_process_on_port(port)
    time.sleep(2)


@pytest.fixture(scope="session", autouse=True)
def setup_and_run_reward_server():
    """
    Start reward model API server for testing.
    """
    print("Pre-test port cleanup...")
    clean_ports()

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "RM_v1008_5")
    else:
        model_path = "./RM_v1008_5"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    log_path = "reward_server.log"
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
        "--graph-optimization-config",
        '{"use_cudagraph":false}',
        "--runner",
        "pooling",
        "--convert",
        "embed",
    ]

    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    # Wait for server to start (up to 480 seconds)
    for _ in range(300):
        if is_port_open("127.0.0.1", FD_API_PORT):
            print(f"reward API server is up on port {FD_API_PORT}")
            break
        time.sleep(1)
    else:
        print("reward API server failed to start. Cleaning up...")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception as e:
            print(f"Failed to kill process group: {e}")
        raise RuntimeError(f"reward API server did not start on port {FD_API_PORT}")

    yield

    print("\n===== Post-test reward server cleanup... =====")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        print(f"reward API server (pid={process.pid}) terminated")
    except Exception as e:
        print(f"Failed to terminate reward API server: {e}")


@pytest.fixture(scope="session")
def reward_api_url():
    """Returns the API endpoint URL for reward."""
    return f"http://0.0.0.0:{FD_API_PORT}/v1/reward"


@pytest.fixture
def headers():
    """Returns common HTTP request headers."""
    return {"Content-Type": "application/json"}


# ==========================
# Test Cases
# ==========================


@pytest.fixture
def consistent_payload():
    """
    Returns a fixed payload for reward model consistency testing.
    Reward models evaluate user-assistant conversation pairs.
    """
    return {
        "model": "default",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "北京天安门在哪里？"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "北京天安门位于中国北京市中心，天安门广场北端，故宫博物院的南门。"}
                ],
            },
        ],
        "user": "test-user-123",
    }


def save_score_baseline(score: float, baseline_file: str):
    """
    Save reward score to baseline file.
    """
    baseline_data = {"score": score}
    with open(baseline_file, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)
    print(f"Baseline saved to: {baseline_file}")


def check_score_against_baseline(current_score: float, baseline_file: str, threshold: float = 0.01):
    """
    Check reward score against baseline file.
    """
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


def test_reward_model(reward_api_url, headers):
    """Test reward model scoring using the chat-style payload."""

    payload = {
        "model": "default",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "北京天安门在哪里？"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "北京天安门在中国北京故宫的前面。"}]},
        ],
        "user": "user-123",
    }

    print(f"\n=== Sending request to {reward_api_url} ===")

    # 发送HTTP请求
    response = requests.post(reward_api_url, headers=headers, json=payload, timeout=30)

    assert response.status_code == 200, f"API request failed with status {response.status_code}: {response.text}"

    result = response.json()
    print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")

    assert "data" in result, f"Response missing 'data' field. Got: {result}"
    assert len(result["data"]) > 0, "Response 'data' is empty"

    first_item = result["data"][0]
    assert "score" in first_item, f"Response data item missing 'score' field. Got: {first_item}"

    score_list = first_item["score"]
    assert isinstance(score_list, list), f"Expected 'score' to be a list, got {type(score_list)}"
    assert len(score_list) > 0, "Score list is empty"

    score = float(score_list[0])

    print(f"✓ Reward Score: {score}")

    base_path = os.getenv("MODEL_PATH", "")
    baseline_filename = "reward_score_baseline.json"

    if base_path:
        baseline_file = os.path.join(base_path, baseline_filename)
    else:
        baseline_file = baseline_filename

    check_score_against_baseline(score, baseline_file, threshold=0.0001)
