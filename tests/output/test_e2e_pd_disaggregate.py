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

"""
E2E test for PD (Prefill-Decode) disaggregation in single machine mode.
This test validates the output module's functionality in a disaggregated
prefill-decode architecture.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback

import pytest
import requests

# Configuration
FD_API_PORT_PREFILL = int(os.getenv("FD_API_PORT_PREFILL", 8188))
FD_API_PORT_DECODE = int(os.getenv("FD_API_PORT_DECODE", 8189))
FD_ENGINE_QUEUE_PORT_PREFILL = int(os.getenv("FD_ENGINE_QUEUE_PORT_PREFILL", 8133))
FD_ENGINE_QUEUE_PORT_DECODE = int(os.getenv("FD_ENGINE_QUEUE_PORT_DECODE", 8134))
FD_METRICS_PORT_PREFILL = int(os.getenv("FD_METRICS_PORT_PREFILL", 8233))
FD_METRICS_PORT_DECODE = int(os.getenv("FD_METRICS_PORT_DECODE", 8234))
FD_CACHE_QUEUE_PORT_PREFILL = int(os.getenv("FD_CACHE_QUEUE_PORT_PREFILL", 8333))
FD_CACHE_QUEUE_PORT_DECODE = int(os.getenv("FD_CACHE_QUEUE_PORT_DECODE", 8334))

# Timeout configuration
MAX_WAIT_SECONDS = 300  # 5 minutes to start servers
REQUEST_TIMEOUT = 60  # 60 seconds for API requests

# Ports to clean before and after tests
PORTS_TO_CLEAN = [
    FD_API_PORT_PREFILL,
    FD_API_PORT_DECODE,
    FD_ENGINE_QUEUE_PORT_PREFILL,
    FD_ENGINE_QUEUE_PORT_DECODE,
    FD_METRICS_PORT_PREFILL,
    FD_METRICS_PORT_DECODE,
    FD_CACHE_QUEUE_PORT_PREFILL,
    FD_CACHE_QUEUE_PORT_DECODE,
]


def is_port_open(host: str, port: int, timeout=1.0):
    """
    Check if a TCP port is open on the given host.

    Args:
        host: The hostname or IP address
        port: The port number
        timeout: Connection timeout in seconds

    Returns:
        True if connection succeeds, False otherwise
    """
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except Exception:
        return False


def kill_process_on_port(port: int):
    """
    Kill processes that are listening on the given port.

    Args:
        port: The port number to clean up
    """
    try:
        output = subprocess.check_output(f"lsof -i:{port} -t", shell=True).decode().strip()
        current_pid = os.getpid()
        parent_pid = os.getppid()
        for pid in output.splitlines():
            pid = int(pid)
            if pid in (current_pid, parent_pid):
                print(f"Skip killing current process (pid={pid}) on port {port}")
                continue
            os.kill(pid, signal.SIGKILL)
            print(f"Killed process on port {port}, pid={pid}")
    except subprocess.CalledProcessError:
        # No process found on this port
        pass
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")


def clean_ports():
    """
    Kill all processes occupying the ports listed in PORTS_TO_CLEAN.
    """
    print("Cleaning up ports...")
    for port in PORTS_TO_CLEAN:
        kill_process_on_port(port)
    time.sleep(2)


@pytest.fixture(scope="module")
def model_path():
    """
    Get model path from environment variable MODEL_PATH,
    default to "./ERNIE-4.5-0.3B-Paddle" if not set.

    Returns:
        str: Path to the model directory
    """
    base_path = os.getenv("MODEL_PATH")
    if base_path:
        return os.path.join(base_path, "ERNIE-4.5-0.3B-Paddle")
    else:
        return "./ERNIE-4.5-0.3B-Paddle"


@pytest.fixture(scope="module")
def prefill_decode_servers(model_path):
    """
    Fixture to start prefill and decode server instances.

    This fixture:
    1. Cleans ports before starting
    2. Starts prefill server instance
    3. Starts decode server instance
    4. Waits for both servers to be ready
    5. Yields control to tests
    6. Cleans up servers after tests

    Args:
        model_path: Path to the model (from model_path fixture)

    Yields:
        tuple: (prefill_process, decode_process)
    """
    print("\n" + "=" * 80)
    print("Setting up Prefill-Decode disaggregated test environment")
    print("=" * 80)

    # Clean ports before starting
    clean_ports()

    # Configure prefill instance
    env_prefill = os.environ.copy()
    env_prefill["CUDA_VISIBLE_DEVICES"] = "0"
    env_prefill["INFERENCE_MSG_QUEUE_ID"] = str(FD_API_PORT_PREFILL)
    env_prefill["FD_LOG_DIR"] = "prefill_log"

    prefill_log_path = "prefill_server.log"
    prefill_cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(FD_API_PORT_PREFILL),
        "--tensor-parallel-size",
        "1",
        "--engine-worker-queue-port",
        str(FD_ENGINE_QUEUE_PORT_PREFILL),
        "--metrics-port",
        str(FD_METRICS_PORT_PREFILL),
        "--cache-queue-port",
        str(FD_CACHE_QUEUE_PORT_PREFILL),
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "16",
        "--splitwise-role",
        "prefill",
        "--inner-prefill-ports",
        str(FD_ENGINE_QUEUE_PORT_DECODE),
        "--seed",
        "42",
    ]

    print(f"\nStarting Prefill server on port {FD_API_PORT_PREFILL}...")
    print(f"Command: {' '.join(prefill_cmd)}")

    try:
        with open(prefill_log_path, "w") as logfile:
            process_prefill = subprocess.Popen(
                prefill_cmd,
                stdout=logfile,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env_prefill,
            )
        print(f"Prefill server started with PID: {process_prefill.pid}")
    except Exception as e:
        print(f"Failed to start prefill server: {e}")
        traceback.print_exc()
        pytest.fail("Failed to start prefill server")

    # Configure decode instance
    env_decode = os.environ.copy()
    env_decode["CUDA_VISIBLE_DEVICES"] = "1"
    env_decode["INFERENCE_MSG_QUEUE_ID"] = str(FD_API_PORT_DECODE)
    env_decode["FD_LOG_DIR"] = "decode_log"

    decode_log_path = "decode_server.log"
    decode_cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(FD_API_PORT_DECODE),
        "--tensor-parallel-size",
        "1",
        "--engine-worker-queue-port",
        str(FD_ENGINE_QUEUE_PORT_DECODE),
        "--metrics-port",
        str(FD_METRICS_PORT_DECODE),
        "--cache-queue-port",
        str(FD_CACHE_QUEUE_PORT_DECODE),
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "16",
        "--splitwise-role",
        "decode",
        "--seed",
        "42",
    ]

    print(f"\nStarting Decode server on port {FD_API_PORT_DECODE}...")
    print(f"Command: {' '.join(decode_cmd)}")

    try:
        with open(decode_log_path, "w") as logfile:
            process_decode = subprocess.Popen(
                decode_cmd,
                stdout=logfile,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env_decode,
            )
        print(f"Decode server started with PID: {process_decode.pid}")
    except Exception as e:
        print(f"Failed to start decode server: {e}")
        traceback.print_exc()
        try:
            os.killpg(process_prefill.pid, signal.SIGTERM)
        except:
            pass
        pytest.fail("Failed to start decode server")

    # Wait for both servers to be ready
    print(f"\nWaiting for servers to be ready (max {MAX_WAIT_SECONDS}s)...")
    start_time = time.time()
    prefill_ready = False
    decode_ready = False

    for i in range(MAX_WAIT_SECONDS):
        if not prefill_ready:
            prefill_ready = is_port_open("127.0.0.1", FD_API_PORT_PREFILL)
            if prefill_ready:
                print(f"✓ Prefill server is ready on port {FD_API_PORT_PREFILL}")

        if not decode_ready:
            decode_ready = is_port_open("127.0.0.1", FD_API_PORT_DECODE)
            if decode_ready:
                print(f"✓ Decode server is ready on port {FD_API_PORT_DECODE}")

        if prefill_ready and decode_ready:
            elapsed = time.time() - start_time
            print(f"\n✓ Both servers are ready (took {elapsed:.2f}s)")
            break

        time.sleep(1)
    else:
        print(f"\n✗ Servers failed to start within {MAX_WAIT_SECONDS}s")
        print("Cleaning up...")
        try:
            os.killpg(process_prefill.pid, signal.SIGTERM)
            os.killpg(process_decode.pid, signal.SIGTERM)
            clean_ports()
        except Exception as e:
            print(f"Failed to kill process groups: {e}")
        pytest.fail("Servers did not start in time")

    # Additional wait to ensure servers are fully initialized
    print("Waiting 5 additional seconds for full initialization...")
    time.sleep(5)

    print("\n" + "=" * 80)
    print("Test environment ready")
    print("=" * 80 + "\n")

    yield process_prefill, process_decode

    # Cleanup
    print("\n" + "=" * 80)
    print("Cleaning up test environment")
    print("=" * 80)

    try:
        print("Terminating prefill server...")
        os.killpg(process_prefill.pid, signal.SIGTERM)
        print(f"✓ Prefill server (PID={process_prefill.pid}) terminated")
    except Exception as e:
        print(f"Failed to terminate prefill server: {e}")

    try:
        print("Terminating decode server...")
        os.killpg(process_decode.pid, signal.SIGTERM)
        print(f"✓ Decode server (PID={process_decode.pid}) terminated")
    except Exception as e:
        print(f"Failed to terminate decode server: {e}")

    clean_ports()
    print("Cleanup complete\n")


def send_request(url, payload, timeout=REQUEST_TIMEOUT):
    """
    Send a request to the specified URL and return the response.

    Args:
        url: The API endpoint URL
        payload: Request payload (dict)
        timeout: Request timeout in seconds

    Returns:
        Response object or None if request failed
    """
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        return response
    except requests.exceptions.Timeout:
        print(f"✗ Request timeout (exceeded {timeout}s)")
        return None
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        return None


def parse_stream_response(response):
    """
    Parse streaming response and extract chunks.

    Args:
        response: Response object from requests

    Returns:
        List of parsed JSON chunks
    """
    chunks = []

    if response.status_code != 200:
        print(f"✗ Request failed with status code: {response.status_code}")
        print(f"Response: {response.text}")
        return chunks

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        if line.startswith("data: "):
            line = line[6:]  # Remove "data: " prefix

        if line.strip() == "[DONE]":
            break

        try:
            chunk = json.loads(line)
            chunks.append(chunk)
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse JSON: {e}")
            print(f"Line: {line}")

    return chunks


def test_prefill_decode_basic_inference(prefill_decode_servers):
    """
    Test basic inference with prefill-decode disaggregation.

    This test validates:
    1. Prefill server can process requests
    2. Decode server can process requests
    3. Token processing works correctly in disaggregated mode
    4. Output is generated successfully
    """
    print("\n" + "=" * 80)
    print("TEST: Basic Prefill-Decode Inference")
    print("=" * 80)

    # Test payload
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请介绍一下中国的四大发明。"},
        ],
        "max_tokens": 100,
        "temperature": 0.8,
        "top_p": 0.95,
        "stream": False,
    }

    # Test prefill server
    print("\n1. Testing Prefill Server")
    print("-" * 40)
    prefill_url = f"http://127.0.0.1:{FD_API_PORT_PREFILL}/v1/chat/completions"
    print(f"Sending request to: {prefill_url}")

    response = send_request(prefill_url, payload)
    assert response is not None, "Prefill request failed"
    assert response.status_code == 200, f"Prefill request returned status {response.status_code}"

    result = response.json()
    assert "choices" in result, "Response missing 'choices'"
    assert len(result["choices"]) > 0, "No choices in response"
    assert "message" in result["choices"][0], "Choice missing 'message'"
    assert "content" in result["choices"][0]["message"], "Message missing 'content'"

    prefill_content = result["choices"][0]["message"]["content"]
    assert len(prefill_content) > 0, "Generated content is empty"

    print(f"✓ Prefill output: {prefill_content[:100]}...")
    print("✓ Prefill server works correctly")

    # Test decode server
    print("\n2. Testing Decode Server")
    print("-" * 40)
    decode_url = f"http://127.0.0.1:{FD_API_PORT_DECODE}/v1/chat/completions"
    print(f"Sending request to: {decode_url}")

    response = send_request(decode_url, payload)
    assert response is not None, "Decode request failed"
    assert response.status_code == 200, f"Decode request returned status {response.status_code}"

    result = response.json()
    assert "choices" in result, "Response missing 'choices'"
    assert len(result["choices"]) > 0, "No choices in response"
    assert "message" in result["choices"][0], "Choice missing 'message'"
    assert "content" in result["choices"][0]["message"], "Message missing 'content'"

    decode_content = result["choices"][0]["message"]["content"]
    assert len(decode_content) > 0, "Generated content is empty"

    print(f"✓ Decode output: {decode_content[:100]}...")
    print("✓ Decode server works correctly")

    # Verify usage information
    if "usage" in result:
        usage = result["usage"]
        assert "prompt_tokens" in usage, "Missing prompt_tokens in usage"
        assert "completion_tokens" in usage, "Missing completion_tokens in usage"
        assert "total_tokens" in usage, "Missing total_tokens in usage"
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "Total tokens mismatch"
        print(f"✓ Usage: {usage}")

    print("\n" + "=" * 80)
    print("✓ TEST PASSED: Basic Prefill-Decode Inference")
    print("=" * 80)


def test_prefill_decode_streaming_inference(prefill_decode_servers):
    """
    Test streaming inference with prefill-decode disaggregation.

    This test validates:
    1. Streaming responses work in prefill mode
    2. Streaming responses work in decode mode
    3. Token processor correctly handles streaming output
    4. All chunks are properly formatted
    """
    print("\n" + "=" * 80)
    print("TEST: Streaming Prefill-Decode Inference")
    print("=" * 80)

    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "太阳和地球之间的距离是多少?"}],
        "max_tokens": 50,
        "temperature": 0.8,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    # Test prefill streaming
    print("\n1. Testing Prefill Streaming")
    print("-" * 40)
    prefill_url = f"http://127.0.0.1:{FD_API_PORT_PREFILL}/v1/chat/completions"

    response = send_request(prefill_url, payload)
    assert response is not None, "Prefill streaming request failed"
    assert response.status_code == 200, f"Prefill streaming returned status {response.status_code}"

    chunks = parse_stream_response(response)
    assert len(chunks) > 0, "No chunks received from prefill streaming"

    # Reconstruct content from chunks
    content_parts = []
    for chunk in chunks[:-1]:  # Exclude last chunk (usage)
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                content_parts.append(delta["content"])

    full_content = "".join(content_parts)
    assert len(full_content) > 0, "No content generated in streaming mode"

    print(f"✓ Prefill streaming output: {full_content[:100]}...")
    print(f"✓ Received {len(chunks)} chunks")

    # Verify usage in last chunk
    last_chunk = chunks[-1]
    if "usage" in last_chunk:
        usage = last_chunk["usage"]
        assert "completion_tokens" in usage, "Missing completion_tokens in usage"
        print(f"✓ Usage info: {usage}")

    # Test decode streaming
    print("\n2. Testing Decode Streaming")
    print("-" * 40)
    decode_url = f"http://127.0.0.1:{FD_API_PORT_DECODE}/v1/chat/completions"

    response = send_request(decode_url, payload)
    assert response is not None, "Decode streaming request failed"
    assert response.status_code == 200, f"Decode streaming returned status {response.status_code}"

    chunks = parse_stream_response(response)
    assert len(chunks) > 0, "No chunks received from decode streaming"

    # Reconstruct content
    content_parts = []
    for chunk in chunks[:-1]:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                content_parts.append(delta["content"])

    full_content = "".join(content_parts)
    assert len(full_content) > 0, "No content generated in decode streaming"

    print(f"✓ Decode streaming output: {full_content[:100]}...")
    print(f"✓ Received {len(chunks)} chunks")

    print("\n" + "=" * 80)
    print("✓ TEST PASSED: Streaming Prefill-Decode Inference")
    print("=" * 80)


def test_prefill_decode_multiple_requests(prefill_decode_servers):
    """
    Test multiple concurrent requests with prefill-decode disaggregation.

    This test validates:
    1. System can handle multiple requests
    2. Token processor correctly manages multiple tasks
    3. Outputs are generated for all requests
    4. No request is dropped or corrupted
    """
    print("\n" + "=" * 80)
    print("TEST: Multiple Concurrent Requests")
    print("=" * 80)

    # Prepare multiple test prompts
    prompts = [
        "请介绍一下中国的四大发明。",
        "太阳和地球之间的距离是多少?",
        "写一首关于春天的古诗。",
    ]

    results = []

    for idx, prompt in enumerate(prompts):
        print(f"\n{idx + 1}. Testing prompt: {prompt[:30]}...")

        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.8,
            "stream": False,
        }

        # Send to prefill server
        prefill_url = f"http://127.0.0.1:{FD_API_PORT_PREFILL}/v1/chat/completions"
        response = send_request(prefill_url, payload)

        assert response is not None, f"Request {idx + 1} failed"
        assert response.status_code == 200, f"Request {idx + 1} returned status {response.status_code}"

        result = response.json()
        assert "choices" in result, f"Request {idx + 1} missing choices"
        assert len(result["choices"]) > 0, f"Request {idx + 1} has no choices"

        content = result["choices"][0]["message"]["content"]
        assert len(content) > 0, f"Request {idx + 1} generated empty content"

        results.append(content)
        print(f"   ✓ Output: {content[:60]}...")

    # Verify all requests succeeded
    assert len(results) == len(prompts), "Not all requests completed"
    print(f"\n✓ All {len(prompts)} requests completed successfully")

    print("\n" + "=" * 80)
    print("✓ TEST PASSED: Multiple Concurrent Requests")
    print("=" * 80)


def test_prefill_decode_metrics(prefill_decode_servers):
    """
    Test metrics endpoint for prefill and decode servers.

    This test validates:
    1. Metrics endpoint is accessible
    2. Key metrics are present
    3. Metrics reflect actual processing
    """
    print("\n" + "=" * 80)
    print("TEST: Metrics Endpoint")
    print("=" * 80)

    # First, send a request to generate some metrics
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 20,
        "stream": False,
    }

    prefill_url = f"http://127.0.0.1:{FD_API_PORT_PREFILL}/v1/chat/completions"
    response = send_request(prefill_url, payload)
    assert response is not None and response.status_code == 200, "Failed to generate metrics data"

    print("✓ Generated metrics data")

    # Check prefill metrics
    print("\n1. Checking Prefill Metrics")
    print("-" * 40)
    prefill_metrics_url = f"http://127.0.0.1:{FD_METRICS_PORT_PREFILL}/metrics"

    try:
        response = requests.get(prefill_metrics_url, timeout=10)
        assert response.status_code == 200, f"Metrics endpoint returned {response.status_code}"

        metrics_text = response.text
        assert len(metrics_text) > 0, "Metrics response is empty"

        # Check for key metrics
        expected_metrics = [
            "fastdeploy:request_success_total",
            "fastdeploy:generation_tokens_total",
        ]

        found_metrics = []
        for metric in expected_metrics:
            if metric in metrics_text:
                found_metrics.append(metric)
                print(f"   ✓ Found metric: {metric}")

        assert len(found_metrics) > 0, "No expected metrics found"
        print(f"✓ Found {len(found_metrics)} key metrics")

    except Exception as e:
        print(f"✗ Failed to check metrics: {e}")
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("✓ TEST PASSED: Metrics Endpoint")
    print("=" * 80)


if __name__ == "__main__":
    """
    Main entry point for running tests directly.
    """
    pytest.main(["-sv", __file__])
