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
import shutil
import signal
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
    clean_ports,
    is_port_open,
)

os.environ["FD_USE_MACHETE"] = "0"


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
    print("log dir clean ")
    if os.path.exists("log") and os.path.isdir("log"):
        shutil.rmtree("log")

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "ernie-4_5-vl-28b-a3b-bf16-paddle")
    else:
        model_path = "./ernie-4_5-vl-28b-a3b-bf16-paddle"

    log_path = "server.log"
    limit_mm_str = json.dumps({"image": 100, "video": 100})

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
        "--enable-mm",
        "--max-model-len",
        "8192",
        "--max-num-batched-tokens",
        "172",
        "--max-num-seqs",
        "64",
        "--limit-mm-per-prompt",
        limit_mm_str,
        "--enable-chunked-prefill",
        "--kv-cache-ratio",
        "0.71",
        "--quantization",
        "wint4",
        "--reasoning-parser",
        "ernie-45-vl",
        "--graph-optimization-config",
        '{"graph_opt_level": 1, "use_cudagraph": true, "full_cuda_graph": false}',
    ]

    # Start subprocess in new process group
    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Enables killing full group via os.killpg
        )

    # Wait up to 10 minutes for API server to be ready
    for _ in range(10 * 60):
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
        print(f"API server (pid={process.pid}) terminated")
        clean_ports()
    except Exception as e:
        print(f"Failed to terminate API server: {e}")


# ==========================
# OpenAI Client additional chat/completions test
# ==========================
@pytest.fixture
def openai_client():
    ip = "0.0.0.0"
    service_http_port = str(FD_API_PORT)
    client = openai.Client(
        base_url=f"http://{ip}:{service_http_port}/v1",
        api_key="EMPTY_API_KEY",
    )
    return client


def test_non_streaming_chat_with_return_token_ids(openai_client, capsys):
    """
    Test return_token_ids option in non-streaming chat functionality with the local service
    """
    # 设定 return_token_ids
    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},  # system不是必需，可选
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "请描述图片内容"},
                ],
            },
        ],
        temperature=1,
        max_tokens=53,
        extra_body={"return_token_ids": True},
        stream=False,
    )
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], "message")
    assert hasattr(response.choices[0].message, "prompt_token_ids")
    assert isinstance(response.choices[0].message.prompt_token_ids, list)
    assert hasattr(response.choices[0].message, "completion_token_ids")
    assert isinstance(response.choices[0].message.completion_token_ids, list)

    # 不设定 return_token_ids
    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},  # system不是必需，可选
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "请描述图片内容"},
                ],
            },
        ],
        temperature=1,
        max_tokens=53,
        extra_body={"return_token_ids": False},
        stream=False,
    )
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], "message")
    assert hasattr(response.choices[0].message, "prompt_token_ids")
    assert response.choices[0].message.prompt_token_ids is None
    assert hasattr(response.choices[0].message, "completion_token_ids")
    assert response.choices[0].message.completion_token_ids is None


def test_streaming_chat_with_return_token_ids(openai_client, capsys):
    """
    Test return_token_ids option in streaming chat functionality with the local service
    """
    # enable return_token_ids
    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},  # system不是必需，可选
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "请描述图片内容"},
                ],
            },
        ],
        temperature=1,
        max_tokens=53,
        extra_body={"return_token_ids": True},
        stream=True,
    )
    is_first_chunk = True
    for chunk in response:
        assert hasattr(chunk, "choices")
        assert len(chunk.choices) > 0
        assert hasattr(chunk.choices[0], "delta")
        assert hasattr(chunk.choices[0].delta, "prompt_token_ids")
        assert hasattr(chunk.choices[0].delta, "completion_token_ids")
        if is_first_chunk:
            is_first_chunk = False
            assert isinstance(chunk.choices[0].delta.prompt_token_ids, list)
            assert chunk.choices[0].delta.completion_token_ids is None
        else:
            assert chunk.choices[0].delta.prompt_token_ids is None
            assert isinstance(chunk.choices[0].delta.completion_token_ids, list)

    # disable return_token_ids
    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},  # system不是必需，可选
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "请描述图片内容"},
                ],
            },
        ],
        temperature=1,
        max_tokens=53,
        extra_body={"return_token_ids": False},
        stream=True,
    )
    for chunk in response:
        assert hasattr(chunk, "choices")
        assert len(chunk.choices) > 0
        assert hasattr(chunk.choices[0], "delta")
        assert hasattr(chunk.choices[0].delta, "prompt_token_ids")
        assert chunk.choices[0].delta.prompt_token_ids is None
        assert hasattr(chunk.choices[0].delta, "completion_token_ids")
        assert chunk.choices[0].delta.completion_token_ids is None


def test_chat_with_thinking(openai_client, capsys):
    """
    Test enable_thinking & reasoning_max_tokens option in non-streaming chat functionality with the local service
    """
    # enable thinking, non-streaming
    response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Explain gravity in a way that a five-year-old child can understand."}],
        temperature=1,
        stream=False,
        max_tokens=10,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    assert response.choices[0].message.reasoning_content is not None

    # disable thinking, non-streaming
    response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Explain gravity in a way that a five-year-old child can understand."}],
        temperature=1,
        stream=False,
        max_tokens=10,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    assert response.choices[0].message.reasoning_content is None
    assert "</think>" not in response.choices[0].message.content

    # test logic
    reasoning_max_tokens = None
    response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Explain gravity in a way that a five-year-old child can understand."}],
        temperature=1,
        stream=False,
        max_tokens=20,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_max_tokens": reasoning_max_tokens,
        },
    )
    assert response.choices[0].message.reasoning_content is not None

    # enable thinking, streaming
    reasoning_max_tokens = 3
    response = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Explain gravity in a way that a five-year-old child can understand."}],
        temperature=1,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_max_tokens": reasoning_max_tokens,
            "return_token_ids": True,
        },
        stream=True,
        max_tokens=10,
    )
    completion_tokens = 1
    reasoning_tokens = 0
    total_tokens = 0
    for chunk_id, chunk in enumerate(response):
        if chunk_id == 0:  # the first chunk is an extra chunk
            continue
        delta_message = chunk.choices[0].delta
        if delta_message.content != "" and delta_message.reasoning_content == "":
            completion_tokens += len(delta_message.completion_token_ids)
        elif delta_message.reasoning_content != "" and delta_message.content == "":
            reasoning_tokens += len(delta_message.completion_token_ids)
        total_tokens += len(delta_message.completion_token_ids)
    assert completion_tokens + reasoning_tokens == total_tokens
    assert reasoning_tokens <= reasoning_max_tokens


def test_thinking_logic_flag(openai_client, capsys):
    """
    Test the interaction between token calculation logic and conditional thinking.
    This test covers:
    1. Default max_tokens calculation when not provided.
    2. Capping of max_tokens when it exceeds model limits.
    3. Default reasoning_max_tokens calculation when not provided.
    4. Activation of thinking based on the final state of reasoning_max_tokens.
    """

    response_case_1 = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Explain gravity briefly."}],
        temperature=1,
        stream=False,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    assert response_case_1.choices[0].message.reasoning_content is not None

    response_case_2 = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Explain gravity in a way that a five-year-old child can understand."}],
        temperature=1,
        stream=False,
        max_tokens=20,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_max_tokens": 5,
        },
    )
    assert response_case_2.choices[0].message.reasoning_content is not None

    response_case_3 = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Explain gravity in a way that a five-year-old child can understand."}],
        temperature=1,
        stream=False,
        max_tokens=20,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert response_case_3.choices[0].message.reasoning_content is None
