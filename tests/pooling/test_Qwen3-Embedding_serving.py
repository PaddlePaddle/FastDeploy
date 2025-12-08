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
from typing import List

import numpy as np
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


@pytest.fixture(scope="session", autouse=True)
def setup_and_run_embedding_server():
    """
    Start embedding model API server for testing.
    """
    print("Pre-test port cleanup...")
    clean_ports()

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "torch", "Qwen3-Embedding-0.6B")
    else:
        model_path = "./Qwen3-Embedding-0.6B"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

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
    baseline_filename = "test-Qwen3-Embedding-0.6B-baseline.json"

    if base_path:
        baseline_file = os.path.join(base_path, "torch", baseline_filename)
    else:
        baseline_file = baseline_filename

    if not os.path.exists(baseline_file):
        print("Baseline file not found. Saving current embedding as baseline...")
        save_embedding_baseline(embedding, baseline_file)
    else:
        print(f"Comparing with baseline: {baseline_file}")
        check_embedding_against_baseline(embedding, baseline_file, threshold=0.02)


def test_multi_text_embedding(embedding_api_url, headers):
    """Test embedding generation for batch text inputs."""
    payload = {
        "model": "default",
        "input": ["北京天安门在哪里?", "上海东方明珠有多高?", "杭州西湖的面积是多少?"],
    }

    resp = requests.post(embedding_api_url, headers=headers, json=payload)
    assert resp.status_code == 200, f"Unexpected status code: {resp.status_code}, response: {resp.text}"

    result = resp.json()
    assert "data" in result, "Response missing 'data' field"
    assert len(result["data"]) == 3, f"Expected 3 embedding results, got {len(result['data'])}"

    # Validate each embedding in the batch
    for idx, item in enumerate(result["data"]):
        assert "embedding" in item, f"Item {idx} missing 'embedding' field"
        assert "index" in item, f"Item {idx} missing 'index' field"
        assert item["index"] == idx, f"Item index mismatch: expected {idx}, got {item['index']}"

        embedding = item["embedding"]
        assert isinstance(embedding, list), f"Embedding {idx} should be a list"
        assert len(embedding) > 0, f"Embedding {idx} vector should not be empty"
        assert all(isinstance(x, (int, float)) for x in embedding), f"Embedding {idx} values should be numeric"

        print(f"Text {idx} embedding dimension: {len(embedding)}")

    # Verify all embeddings have the same dimension
    dimensions = [len(item["embedding"]) for item in result["data"]]
    assert len(set(dimensions)) == 1, f"All embeddings should have same dimension, got: {dimensions}"

    # Compare embeddings with baseline
    base_path = os.getenv("MODEL_PATH", "")
    baseline_filename = "test-Qwen3-Embedding-0.6B-multi-input-baseline.json"

    if base_path:
        baseline_file = os.path.join(base_path, "torch", baseline_filename)
    else:
        baseline_file = baseline_filename

    # Save all embeddings to baseline
    batch_embeddings = [item["embedding"] for item in result["data"]]

    if not os.path.exists(baseline_file):
        print("Batch baseline file not found. Saving current embeddings as baseline...")
        baseline_data = {
            "embeddings": batch_embeddings,
            "dimension": len(batch_embeddings[0]),
            "count": len(batch_embeddings),
            "inputs": payload["input"],
        }
        with open(baseline_file, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2)
        print(f"Batch baseline saved to: {baseline_file}")
    else:
        print(f"Comparing batch with baseline: {baseline_file}")
        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
            baseline_embeddings = baseline_data["embeddings"]

        assert len(batch_embeddings) == len(
            baseline_embeddings
        ), f"Embedding count mismatch: current={len(batch_embeddings)}, baseline={len(baseline_embeddings)}"

        # Compare each embedding
        for idx, (current_emb, baseline_emb) in enumerate(zip(batch_embeddings, baseline_embeddings)):
            print(f"\n--- Comparing embedding {idx}: '{payload['input'][idx]}' ---")
            mean_abs_diff = compare_embeddings(current_emb, baseline_emb, threshold=0.05)

            if mean_abs_diff >= 0.05:
                # Save current batch for debugging
                temp_file = f"{baseline_file}.current"
                print("temp_file", temp_file)
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "embeddings": batch_embeddings,
                            "dimension": len(batch_embeddings[0]),
                            "count": len(batch_embeddings),
                            "inputs": payload["input"],
                        },
                        f,
                        indent=2,
                    )

                raise AssertionError(
                    f"Embedding {idx} differs from baseline by too much "
                    f"(mean_abs_diff={mean_abs_diff:.6f} >= 0.01):\n"
                    f"Current batch saved to: {temp_file}\n"
                    f"Please check the differences."
                )
