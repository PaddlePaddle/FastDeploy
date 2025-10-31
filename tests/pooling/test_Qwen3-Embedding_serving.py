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
from typing import List

import numpy as np
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
def setup_and_run_embedding_server():
    """
    Start embedding model API server for testing.
    """
    print("Pre-test port cleanup...")
    clean_ports()

    os.environ["FD_DISABLE_CHUNKED_PREFILL"] = "1"
    os.environ["FD_USE_GET_SAVE_OUTPUT_V1"] = "1"

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "torch", "Qwen3-Embedding-0.6B")
    else:
        model_path = "./Qwen3-Embedding-0.6B"

    if not os.path.exists(model_path):
        pytest.skip(f"Model path not found: {model_path}")

    log_path = "embedding_server.log"
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
    ]

    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    # Wait for server to start (up to 480 seconds)
    for _ in range(480):
        if is_port_open("127.0.0.1", FD_API_PORT):
            print(f"Embedding API server is up on port {FD_API_PORT}")
            break
        time.sleep(1)
    else:
        print("Embedding API server failed to start. Cleaning up...")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception as e:
            print(f"Failed to kill process group: {e}")
        raise RuntimeError(f"Embedding API server did not start on port {FD_API_PORT}")

    yield

    print("\n===== Post-test embedding server cleanup... =====")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        print(f"Embedding API server (pid={process.pid}) terminated")
    except Exception as e:
        print(f"Failed to terminate embedding API server: {e}")


@pytest.fixture(scope="session")
def embedding_api_url():
    """Returns the API endpoint URL for embeddings."""
    return f"http://0.0.0.0:{FD_API_PORT}/v1/embeddings"


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


def save_embedding_baseline(embedding: List[float], baseline_file: str):
    """
    Save embedding vector to baseline file.
    """
    baseline_data = {"embedding": embedding, "dimension": len(embedding)}
    with open(baseline_file, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)
    print(f"Baseline saved to: {baseline_file}")


def compare_embeddings(embedding1: List[float], embedding2: List[float], threshold: float = 0.01) -> float:
    """
    Compare two embedding vectors using mean absolute difference.

    Returns:
        mean_abs_diff: mean absolute difference between two embeddings
    """
    arr1 = np.array(embedding1, dtype=np.float32)
    arr2 = np.array(embedding2, dtype=np.float32)

    # Mean absolute difference
    mean_abs_diff = np.mean(np.abs(arr1 - arr2))

    print(f"Mean Absolute Difference: {mean_abs_diff:.6f}")

    return mean_abs_diff


def check_embedding_against_baseline(embedding: List[float], baseline_file: str, threshold: float = 0.01):
    """
    Check embedding against baseline file.

    Args:
        embedding: Current embedding vector
        baseline_file: Path to baseline file
        threshold: Maximum allowed difference rate (1 - cosine_similarity)
    """
    try:
        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
            baseline_embedding = baseline_data["embedding"]
    except FileNotFoundError:
        raise AssertionError(f"Baseline file not found: {baseline_file}")

    if len(embedding) != len(baseline_embedding):
        raise AssertionError(
            f"Embedding dimension mismatch: current={len(embedding)}, baseline={len(baseline_embedding)}"
        )

    mean_abs_diff = compare_embeddings(embedding, baseline_embedding, threshold)

    if mean_abs_diff >= threshold:
        # Save current embedding for debugging
        temp_file = f"{baseline_file}.current"
        save_embedding_baseline(embedding, temp_file)

        raise AssertionError(
            f"Embedding differs from baseline by too much (mean_abs_diff={mean_abs_diff:.6f} >= {threshold}):\n"
            f"Current embedding saved to: {temp_file}\n"
            f"Please check the differences."
        )


def test_single_text_embedding(embedding_api_url, headers):
    """Test embedding generation for a single text input."""
    payload = {
        "input": "北京天安门在哪里?",
        "model": "Qwen3-Embedding-0.6B",
    }

    resp = requests.post(embedding_api_url, headers=headers, json=payload)
    assert resp.status_code == 200, f"Unexpected status code: {resp.status_code}"

    result = resp.json()
    assert "data" in result, "Response missing 'data' field"
    assert len(result["data"]) == 1, "Expected single embedding result"

    embedding = result["data"][0]["embedding"]
    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) > 0, "Embedding vector should not be empty"
    assert all(isinstance(x, (int, float)) for x in embedding), "Embedding values should be numeric"

    print(f"Single text embedding dimension: {len(embedding)}")

    base_path = os.getenv("MODEL_PATH", "")
    baseline_filename = "Qwen3-Embedding-0.6B-baseline.json"

    if base_path:
        baseline_file = os.path.join(base_path, "torch", baseline_filename)
    else:
        baseline_file = baseline_filename

    if not os.path.exists(baseline_file):
        print("Baseline file not found. Saving current embedding as baseline...")
        save_embedding_baseline(embedding, baseline_file)
    else:
        print(f"Comparing with baseline: {baseline_file}")
        check_embedding_against_baseline(embedding, baseline_file, threshold=0.01)
