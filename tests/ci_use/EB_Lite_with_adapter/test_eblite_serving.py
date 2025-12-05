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
import queue
import shutil
import signal
import socket
import subprocess
import sys
import time

import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
print("project_root", project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ci_use.EB_Lite_with_adapter.zmq_client import LLMControlClient, LLMReqClient

env = os.environ.copy()

# Read ports from environment variables; use default values if not set
FD_API_PORT = int(os.getenv("FD_API_PORT", 8188))
FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8133))
FD_METRICS_PORT = int(os.getenv("FD_METRICS_PORT", 8233))
FD_CACHE_QUEUE_PORT = int(os.getenv("FD_CACHE_QUEUE_PORT", 8234))

FD_ENABLE_INTERNAL_ADAPTER = int(os.getenv("FD_ENABLE_INTERNAL_ADAPTER", "1"))
FD_ZMQ_RECV_REQUEST_SERVER_PORT = int(os.getenv("FD_ZMQ_RECV_REQUEST_SERVER_PORT", "8204"))
FD_ZMQ_SEND_RESPONSE_SERVER_PORT = int(os.getenv("FD_ZMQ_SEND_RESPONSE_SERVER_PORT", "8205"))
FD_ZMQ_RECV_REQUEST_SERVER_PORTS = str(os.getenv("FD_ZMQ_RECV_REQUEST_SERVER_PORTS", "8204"))
FD_ZMQ_SEND_RESPONSE_SERVER_PORTS = str(os.getenv("FD_ZMQ_SEND_RESPONSE_SERVER_PORTS", "8205"))
FD_ZMQ_CONTROL_CMD_SERVER_PORTS = int(os.getenv("FD_ZMQ_CONTROL_CMD_SERVER_PORTS", "8206"))
FD_ZMQ_CONTROL_CMD_SERVER_PORT = FD_ZMQ_CONTROL_CMD_SERVER_PORTS

env["FD_ENABLE_INTERNAL_ADAPTER"] = str(FD_ENABLE_INTERNAL_ADAPTER)
env["FD_ZMQ_RECV_REQUEST_SERVER_PORTS"] = str(FD_ZMQ_RECV_REQUEST_SERVER_PORTS)
env["FD_ZMQ_SEND_RESPONSE_SERVER_PORTS"] = str(FD_ZMQ_SEND_RESPONSE_SERVER_PORTS)
env["FD_ZMQ_CONTROL_CMD_SERVER_PORTS"] = str(FD_ZMQ_CONTROL_CMD_SERVER_PORTS)
env["FD_ZMQ_CONTROL_CMD_SERVER_PORT"] = str(FD_ZMQ_CONTROL_CMD_SERVER_PORT)

# List of ports to clean before and after tests
PORTS_TO_CLEAN = [
    FD_API_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_ZMQ_RECV_REQUEST_SERVER_PORTS,
    FD_ZMQ_SEND_RESPONSE_SERVER_PORTS,
    FD_ZMQ_CONTROL_CMD_SERVER_PORT,
]


@pytest.fixture
def zmq_req_client():
    client = LLMReqClient("0.0.0.0", FD_ZMQ_RECV_REQUEST_SERVER_PORT, FD_ZMQ_SEND_RESPONSE_SERVER_PORT)
    return client


@pytest.fixture
def zmq_control_client():
    client = LLMControlClient("0.0.0.0", FD_ZMQ_CONTROL_CMD_SERVER_PORT)
    return client


def is_port_open(host: str, port: int, timeout=1.0):
    """
    Check if a TCP port is open on the given host.
    Returns True if connection succeeds, False otherwise.
    """
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except Exception:
        return False


def kill_process_on_port(port: int):
    """
    Kill processes that are listening on the given port.
    Uses multiple methods to ensure thorough cleanup.
    """
    current_pid = os.getpid()
    parent_pid = os.getppid()

    # Method 1: Use lsof to find processes
    try:
        output = subprocess.check_output(f"lsof -i:{port} -t", shell=True).decode().strip()
        for pid in output.splitlines():
            pid = int(pid)
            if pid in (current_pid, parent_pid):
                print(f"Skip killing current process (pid={pid}) on port {port}")
                continue
            try:
                # First try SIGTERM for graceful shutdown
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                # Then SIGKILL if still running
                os.kill(pid, signal.SIGKILL)
                print(f"Killed process on port {port}, pid={pid}")
            except ProcessLookupError:
                pass  # Process already terminated
    except subprocess.CalledProcessError:
        pass

    # Method 2: Use netstat and fuser as backup
    try:
        # Find processes using netstat and awk
        cmd = f"netstat -tulpn 2>/dev/null | grep :{port} | awk '{{print $7}}' | cut -d'/' -f1"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        for pid in output.splitlines():
            if pid and pid.isdigit():
                pid = int(pid)
                if pid in (current_pid, parent_pid):
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"Killed process (netstat) on port {port}, pid={pid}")
                except ProcessLookupError:
                    pass
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Method 3: Use fuser if available
    try:
        subprocess.run(f"fuser -k {port}/tcp", shell=True, timeout=5)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass


def clean_ports():
    """
    Kill all processes occupying the ports listed in PORTS_TO_CLEAN.
    """
    print(f"Cleaning ports: {PORTS_TO_CLEAN}")
    for port in PORTS_TO_CLEAN:
        kill_process_on_port(port)

    # Double check and retry if ports are still in use
    time.sleep(2)
    for port in PORTS_TO_CLEAN:
        if is_port_open("127.0.0.1", port, timeout=0.1):
            print(f"Port {port} still in use, retrying cleanup...")
            kill_process_on_port(port)
            time.sleep(1)


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
        model_path = os.path.join(base_path, "ernie-4_5-21b-a3b-bf16-paddle")
    else:
        model_path = "./ernie-4_5-21b-a3b-bf16-paddle"

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
        "128",
        "--quantization",
        "wint4",
    ]

    # Start subprocess in new process group
    # 清除log目录
    if os.path.exists("log"):
        shutil.rmtree("log")
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
        clean_ports()
        print(f"API server (pid={process.pid}) terminated")
    except Exception as e:
        print(f"Failed to terminate API server: {e}")


def test_request_and_response(zmq_req_client):
    prompt_token_ids = [5300, 93956, 55791]
    req_id = "test"
    request = {
        "req_id": req_id,
        "request_id": req_id,
        "min_tokens": 1,
        "dp_rank": 0,  # P实例 DP rank, 从当前环境变量里读取
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_len": len(prompt_token_ids),
        "eos_token_ids": [2],
        "stop_token_ids": [2],
        "max_dec_len": 32 * 1024,
        "max_tokens": 32 * 1024,
        "min_dec_len": 1,
        "arrival_time": time.time(),
        "preprocess_start_time": time.time(),
        "preprocess_end_time": time.time(),
        "messages": [],
        "temperature": 0.8,
        "penalty_score": 1.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0,
        "top_p": 0.8,
        "frequency_penalty": 0.0,
    }
    result_queue = queue.Queue()
    zmq_req_client.start(result_queue)
    zmq_req_client.send_request(request)
    zmq_req_client.request_result(req_id)
    has_is_end_result = False
    while True:
        result = result_queue.get()
        if result[0][-1]["finished"]:
            has_is_end_result = True
            break
    assert has_is_end_result is True


def test_control_cmd(zmq_control_client):
    result = zmq_control_client.get_payload()
    assert "unhandled_request_num" in result
    result = zmq_control_client.get_metrics()
    assert result is not None
