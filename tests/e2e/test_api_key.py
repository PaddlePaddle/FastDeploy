"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import os
import signal
import subprocess
import sys
import time
from typing import Optional

import pytest
import requests

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

current_server_process: Optional[subprocess.Popen] = None


def start_api_server(api_key_cli: Optional[list[str]] = None, api_key_env: Optional[str] = None):
    global current_server_process
    clean_ports()

    env = os.environ.copy()
    if api_key_env is not None:
        env["FD_API_KEY"] = api_key_env
    else:
        env.pop("FD_API_KEY", None)
    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "ERNIE-4.5-0.3B-Paddle")
    else:
        model_path = "./ERNIE-4.5-0.3B-Paddle"
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
        "--graph-optimization-config",
        '{"cudagraph_capture_sizes": [1], "use_cudagraph":true}',
    ]

    if api_key_cli is not None:
        for key in api_key_cli:
            cmd.extend(["--api-key", key])

    with open(log_path, "w") as logfile:
        process = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True, env=env)

    for _ in range(300):
        if is_port_open("127.0.0.1", FD_API_PORT):
            print(f"API server started (port: {FD_API_PORT}, cli_key: {api_key_cli}, env_key: {api_key_env})")
            current_server_process = process
            return process
        time.sleep(1)
    else:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise RuntimeError(f"API server failed to start in 5 minutes (port: {FD_API_PORT})")


def stop_api_server():
    global current_server_process
    if current_server_process and current_server_process.poll() is None:
        try:
            os.killpg(current_server_process.pid, signal.SIGTERM)
            current_server_process.wait(timeout=10)
            print(f"API server stopped (pid: {current_server_process.pid})")
        except Exception as e:
            print(f"Failed to stop server: {e}")
    current_server_process = None
    clean_ports()


@pytest.fixture(scope="function", autouse=True)
def teardown_server():
    yield
    stop_api_server()
    os.environ.pop("FD_API_KEY", None)


@pytest.fixture(scope="function")
def api_url():
    return f"http://0.0.0.0:{FD_API_PORT}/v1/chat/completions"


@pytest.fixture
def common_headers():
    return {"Content-Type": "application/json"}


@pytest.fixture
def valid_auth_headers():
    return {"Content-Type": "application/json", "Authorization": "Bearer {api_key}"}


@pytest.fixture
def test_payload():
    return {"messages": [{"role": "user", "content": "hello"}], "temperature": 0.9, "max_tokens": 100}


def test_api_key_cli_only(api_url, common_headers, valid_auth_headers, test_payload):
    test_api_key = ["cli_test_key_123", "cli_test_key_456"]
    start_api_server(api_key_cli=test_api_key)

    response = requests.post(api_url, json=test_payload, headers=common_headers)
    assert response.status_code == 401
    assert "error" in response.json()
    assert "unauthorized" in response.json()["error"].lower()

    invalid_headers = valid_auth_headers.copy()
    invalid_headers["Authorization"] = invalid_headers["Authorization"].format(api_key="wrong_key")
    response = requests.post(api_url, json=test_payload, headers=invalid_headers)
    assert response.status_code == 401

    valid_headers = valid_auth_headers.copy()
    valid_headers["Authorization"] = valid_headers["Authorization"].format(api_key=test_api_key[0])
    response = requests.post(api_url, json=test_payload, headers=valid_headers)
    assert response.status_code == 200

    valid_headers = valid_auth_headers.copy()
    valid_headers["Authorization"] = valid_headers["Authorization"].format(api_key=test_api_key[1])
    response = requests.post(api_url, json=test_payload, headers=valid_headers)
    assert response.status_code == 200


def test_api_key_env_only(api_url, common_headers, valid_auth_headers, test_payload):
    test_api_key = "env_test_key_456,env_test_key_789"
    start_api_server(api_key_env=test_api_key)

    response = requests.post(api_url, json=test_payload, headers=common_headers)
    assert response.status_code == 401

    valid_headers = valid_auth_headers.copy()
    valid_headers["Authorization"] = valid_headers["Authorization"].format(api_key="env_test_key_456")
    response = requests.post(api_url, json=test_payload, headers=valid_headers)
    assert response.status_code == 200

    valid_headers = valid_auth_headers.copy()
    valid_headers["Authorization"] = valid_headers["Authorization"].format(api_key="env_test_key_789")
    response = requests.post(api_url, json=test_payload, headers=valid_headers)
    assert response.status_code == 200


def test_api_key_cli_priority_over_env(api_url, valid_auth_headers, test_payload):
    cli_key = ["cli_priority_key_789"]
    env_key = "env_low_priority_key_000"
    start_api_server(api_key_cli=cli_key, api_key_env=env_key)

    env_headers = valid_auth_headers.copy()
    env_headers["Authorization"] = env_headers["Authorization"].format(api_key=env_key)
    response = requests.post(api_url, json=test_payload, headers=env_headers)
    assert response.status_code == 401

    cli_headers = valid_auth_headers.copy()
    cli_headers["Authorization"] = cli_headers["Authorization"].format(api_key=cli_key[0])
    response = requests.post(api_url, json=test_payload, headers=cli_headers)
    assert response.status_code == 200


def test_api_key_not_set(api_url, common_headers, valid_auth_headers, test_payload):
    start_api_server(api_key_cli=None, api_key_env=None)

    response = requests.post(api_url, json=test_payload, headers=common_headers)
    assert response.status_code == 200

    cli_headers = valid_auth_headers.copy()
    cli_headers["Authorization"] = cli_headers["Authorization"].format(api_key="some_api_key")
    response = requests.post(api_url, json=test_payload, headers=cli_headers)
    assert response.status_code == 200
