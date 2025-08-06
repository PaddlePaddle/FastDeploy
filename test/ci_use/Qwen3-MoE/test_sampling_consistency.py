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
import signal
import socket
import subprocess
import sys
import time

import pytest
import requests


def is_port_open(host: str, port: int, timeout=1.0):
    """Check if a TCP port is open on the given host."""
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except Exception:
        return False


def kill_process_on_port(port: int):
    """Kill processes that are listening on the given port."""
    try:
        output = subprocess.check_output(f"lsof -i:{port} -t", shell=True).decode().strip()
        for pid in output.splitlines():
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed process on port {port}, pid={pid}")
    except subprocess.CalledProcessError:
        pass


def clean_specific_ports(ports_list):
    """Kill all processes occupying the specified ports."""
    for port in ports_list:
        kill_process_on_port(port)


def create_server_process_with_sampling(sampling_class: str, api_port: int, queue_port: int, metrics_port: int):
    """
    Create and start the API server process with specified sampling class and ports.
    Returns the process object.
    """
    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "Qwen3-30B-A3B")
    else:
        model_path = "./Qwen3-30B-A3B"

    log_path = f"server_{sampling_class}_{api_port}.log"
    cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(api_port),
        "--tensor-parallel-size",
        "1",
        "--engine-worker-queue-port",
        str(queue_port),
        "--metrics-port",
        str(metrics_port),
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "50",
        "--quantization",
        "wint4",
    ]

    env = os.environ.copy()
    env["FD_SAMPLING_CLASS"] = sampling_class

    print(f"Starting server with FD_SAMPLING_CLASS={sampling_class} on port {api_port}")

    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    return process


def wait_for_server_ready_on_port(api_port: int, timeout=300):
    """Wait for the API server to be ready on specified port."""
    for _ in range(timeout):
        if is_port_open("127.0.0.1", api_port):
            print(f"API server is up on port {api_port}")
            return True
        time.sleep(1)
    return False


# ==========================
# Fixtures for pytest
# ==========================


@pytest.fixture
def headers():
    """Returns common HTTP request headers."""
    return {"Content-Type": "application/json"}


@pytest.fixture
def consistent_payload():
    """Returns a fixed payload for consistency testing with fixed seed."""
    return {
        "messages": [
            {
                "role": "user",
                "content": "用一句话介绍 PaddlePaddle, 30字以内 /no_think",
            }
        ],
        "temperature": 0.8,
        "seed": 42,  # Fixed seed
        "max_tokens": 50,
    }


@pytest.fixture
def rejection_server():
    """Fixture to manage rejection sampling server lifecycle."""
    sampling_class = "rejection"
    api_port = 8288
    queue_port = 8334
    metrics_port = 8433
    ports_to_clean = [api_port, queue_port, metrics_port]

    # Setup: Clean ports and start server
    clean_specific_ports(ports_to_clean)
    time.sleep(2)
    process = create_server_process_with_sampling(sampling_class, api_port, queue_port, metrics_port)

    # Wait for server to be ready
    if not wait_for_server_ready_on_port(api_port, timeout=300):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            pass
        pytest.fail(f"Server failed to start for {sampling_class}")

    # Yield server info to test
    server_info = {
        "api_url": f"http://0.0.0.0:{api_port}/v1/chat/completions",
        "process": process,
        "sampling_class": sampling_class,
    }

    yield server_info

    # Teardown: Clean up server
    try:
        os.killpg(process.pid, signal.SIGTERM)
        print(f"Server terminated for {sampling_class}")
    except Exception as e:
        print(f"Failed to terminate server: {e}")
    time.sleep(3)


@pytest.fixture
def air_server():
    """Fixture to manage AIR sampling server lifecycle."""
    sampling_class = "air"
    api_port = 8123
    queue_port = 8534
    metrics_port = 8643
    ports_to_clean = [api_port, queue_port, metrics_port]

    # Setup: Clean ports and start server
    clean_specific_ports(ports_to_clean)
    time.sleep(2)
    process = create_server_process_with_sampling(sampling_class, api_port, queue_port, metrics_port)

    # Wait for server to be ready with detailed error reporting
    if not wait_for_server_ready_on_port(api_port, timeout=300):
        # Check log file for debugging
        log_file = f"server_{sampling_class}_{api_port}.log"
        error_msg = f"Server failed to start for {sampling_class}"

        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                lines = f.readlines()
                print(f"Server startup failed. Last 10 lines of {log_file}:")
                for line in lines[-10:]:
                    print(f"  {line.strip()}")

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            pass
        pytest.fail(error_msg)

    # Yield server info to test
    server_info = {
        "api_url": f"http://0.0.0.0:{api_port}/v1/chat/completions",
        "process": process,
        "sampling_class": sampling_class,
    }

    yield server_info

    # Teardown: Clean up server
    try:
        os.killpg(process.pid, signal.SIGTERM)
        print(f"Server terminated for {sampling_class}")
    except Exception as e:
        print(f"Failed to terminate server: {e}")
    time.sleep(3)


# ==========================
# Test cases
# ==========================


def test_seed_consistency_rejection_sampling(rejection_server, headers, consistent_payload):
    """
    Test seed consistency for rejection sampling - multiple runs should produce identical results.
    """
    server_info = rejection_server
    api_url = server_info["api_url"]
    sampling_class = server_info["sampling_class"]
    num_runs = 5

    print(f"\n===== Testing seed consistency for {sampling_class.upper()} sampling =====")

    # Run multiple requests with same seed
    results = []
    print(f"Running {num_runs} requests with fixed seed=42:")

    for i in range(num_runs):
        resp = requests.post(api_url, headers=headers, json=consistent_payload)
        assert resp.status_code == 200, f"Request {i+1} failed with status {resp.status_code}"

        content = resp.json()["choices"][0]["message"]["content"]
        results.append(content)
        print(f"  Run {i+1}: {content[:50]}...")
        time.sleep(1)

    # Check if all results are identical
    reference_result = results[0]
    all_identical = all(result == reference_result for result in results)

    print(f"\n--- {sampling_class.upper()} Sampling Results ---")
    if all_identical:
        print(f" ALL {num_runs} runs produced IDENTICAL results")
        print(f"   Result: {reference_result}")
    else:
        print(" Results are NOT identical:")
        for i, result in enumerate(results):
            status = "yes" if result == reference_result else "no"
            print(f"   Run {i+1} {status}: {result}")

    # Use assertion for pytest compatibility
    assert (
        all_identical
    ), f"Rejection sampling should be consistent with fixed seed. Got {len(set(results))} different outputs: {list(set(results))}"


def test_seed_consistency_air_sampling(air_server, headers, consistent_payload):
    """
    Test seed consistency for AIR sampling - multiple runs should produce identical results.
    """
    server_info = air_server
    api_url = server_info["api_url"]
    sampling_class = server_info["sampling_class"]
    num_runs = 5

    print(f"\n===== Testing seed consistency for {sampling_class.upper()} sampling =====")

    # Run multiple requests with same seed
    results = []
    print(f"Running {num_runs} requests with fixed seed=42:")

    for i in range(num_runs):
        resp = requests.post(api_url, headers=headers, json=consistent_payload)
        assert resp.status_code == 200, f"Request {i+1} failed with status {resp.status_code}"

        content = resp.json()["choices"][0]["message"]["content"]
        results.append(content)
        print(f"  Run {i+1}: {content[:50]}...")
        time.sleep(1)

    # Check if all results are identical
    reference_result = results[0]
    all_identical = all(result == reference_result for result in results)

    print(f"\n--- {sampling_class.upper()} Sampling Results ---")
    if all_identical:
        print(f" ALL {num_runs} runs produced IDENTICAL results")
        print(f"   Result: {reference_result}")
    else:
        print(" Results are NOT identical:")
        for i, result in enumerate(results):
            status = "yes" if result == reference_result else "no"
            print(f"   Run {i+1} {status}: {result}")

    # Use assertion for pytest compatibility
    assert (
        all_identical
    ), f"AIR sampling should be consistent with fixed seed. Got {len(set(results))} different outputs: {list(set(results))}"
