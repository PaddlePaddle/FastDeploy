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
import re
import signal
import subprocess
import sys
import time

import openai
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
        "32768",
        "--max-num-batched-tokens",
        "384",
        "--max-num-seqs",
        "128",
        "--limit-mm-per-prompt",
        limit_mm_str,
        "--enable-chunked-prefill",
        "--kv-cache-ratio",
        "0.71",
        "--quantization",
        "wint4",
        "--reasoning-parser",
        "ernie-45-vl",
        "--guided-decoding-backend",
        "auto",
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
    except Exception as e:
        print(f"Failed to terminate API server: {e}")


@pytest.fixture(scope="session")
def api_url(request):
    """
    Returns the API endpoint URL for chat completions.
    """
    return f"http://0.0.0.0:{FD_API_PORT}/v1/chat/completions"


@pytest.fixture(scope="session")
def metrics_url(request):
    """
    Returns the metrics endpoint URL.
    """
    return f"http://0.0.0.0:{FD_METRICS_PORT}/metrics"


@pytest.fixture
def headers():
    """
    Returns common HTTP request headers.
    """
    return {"Content-Type": "application/json"}


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
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://ku.baidu-int.com/vk-assets-ltd/space/2024/09/13/933d1e0a0760498e94ec0f2ccee865e0",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "请描述图片内容"},
                ],
            }
        ],
        "temperature": 0.8,
        "top_p": 0,  # fix top_p to reduce randomness
        "seed": 13,  # fixed random seed
    }


# ==========================
# Consistency test for repeated runs with fixed payload
# ==========================
def test_consistency_between_runs(api_url, headers, consistent_payload):
    """
    Test that result is same as the base result.
    """
    # request
    resp1 = requests.post(api_url, headers=headers, json=consistent_payload)
    assert resp1.status_code == 200
    result1 = resp1.json()
    content1 = (
        result1["choices"][0]["message"]["reasoning_content"]
        + "</think>"
        + result1["choices"][0]["message"]["content"]
    )
    file_res_temp = "ernie-4_5-vl"
    f_o = open(file_res_temp, "a")
    f_o.writelines(content1)
    f_o.close()

    # base result
    base_path = os.getenv("MODEL_PATH")
    if base_path:
        base_file = os.path.join(base_path, "ernie-4_5-vl-base-tp2-dev")
    else:
        base_file = "ernie-4_5-vl-base-tp2-dev"
    with open(base_file, "r") as f:
        content2 = f.read()

    # Verify that result is same as the base result
    assert content1 == content2


# ==========================
# OpenAI Client Chat Completion Test
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


# Non-streaming test
def test_non_streaming_chat(openai_client):
    """Test non-streaming chat functionality with the local service"""
    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant.",
            },  # system不是必需，可选
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://ku.baidu-int.com/vk-assets-ltd/space/2024/09/13/933d1e0a0760498e94ec0f2ccee865e0",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "请描述图片内容"},
                ],
            },
        ],
        temperature=1,
        max_tokens=53,
        stream=False,
    )

    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], "message")
    assert hasattr(response.choices[0].message, "content")


# Streaming test
def test_streaming_chat(openai_client, capsys):
    """Test streaming chat functionality with the local service"""
    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant.",
            },  # system不是必需，可选
            {"role": "user", "content": "List 3 countries and their capitals."},
            {
                "role": "assistant",
                "content": "China(Beijing), France(Paris), Australia(Canberra).",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://ku.baidu-int.com/vk-assets-ltd/space/2024/09/13/933d1e0a0760498e94ec0f2ccee865e0",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "请描述图片内容"},
                ],
            },
        ],
        temperature=1,
        max_tokens=512,
        stream=True,
    )

    output = []
    for chunk in response:
        if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
            output.append(chunk.choices[0].delta.content)
    assert len(output) > 2


# ==========================
# OpenAI Client additional chat/completions test
# ==========================


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


def streaming_chat_base(openai_client, chat_param):
    """
    Test streaming chat base functionality with the local service
    """
    assert isinstance(chat_param, dict), f"{chat_param} should be a dict"
    assert "messages" in chat_param, f"{chat_param} should contain messages"

    response = openai_client.chat.completions.create(
        model="default",
        stream=True,
        **chat_param,
    )

    output = []
    for chunk in response:
        if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
            output.append(chunk.choices[0].delta.content)
    assert len(output) > 2
    return "".join(output)


def non_streaming_chat_base(openai_client, chat_param):
    """
    Test non streaming chat base functionality with the local service
    """
    assert isinstance(chat_param, dict), f"{chat_param} should be a dict"
    assert "messages" in chat_param, f"{chat_param} should contain messages"

    response = openai_client.chat.completions.create(
        model="default",
        stream=False,
        **chat_param,
    )

    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], "message")
    assert hasattr(response.choices[0].message, "content")
    return response.choices[0].message.content


@pytest.mark.skip(reason="Temporarily skip this case due to unstable execution")
def test_structured_outputs_json_schema(openai_client):
    """
    Test structured outputs json_schema functionality with the local service
    """
    chat_param = {
        "temperature": 1,
        "max_tokens": 1024,
    }

    # json_object
    json_chat_param = {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
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
                    {"type": "text", "text": "请描述图片内容,使用json格式输出结果"},
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }
    json_chat_param.update(chat_param)

    outputs = []
    outputs.append(streaming_chat_base(openai_client, json_chat_param))
    outputs.append(non_streaming_chat_base(openai_client, json_chat_param))

    json_chat_param["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    outputs.append(streaming_chat_base(openai_client, json_chat_param))
    outputs.append(non_streaming_chat_base(openai_client, json_chat_param))

    for response in outputs:
        try:
            json.loads(response)
            is_valid = True
        except ValueError:
            is_valid = False

        assert is_valid, f"json_object response: {response} is not a valid json"

    # json_schema
    from enum import Enum

    from pydantic import BaseModel

    class BookType(str, Enum):
        romance = "Romance"
        historical = "Historical"
        adventure = "Adventure"
        mystery = "Mystery"
        dystopian = "Dystopian"

    class BookDescription(BaseModel):
        author: str
        title: str
        genre: BookType

    json_schema_param = {
        "messages": [
            {
                "role": "user",
                "content": "Generate a JSON describing a literary work, including author, title and book type.",
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "book-description", "schema": BookDescription.model_json_schema()},
        },
    }
    json_schema_param.update(chat_param)
    response = streaming_chat_base(openai_client, json_schema_param)
    try:
        json_schema_response = json.loads(response)
        is_valid = True
    except ValueError:
        is_valid = False

    assert is_valid, f"json_schema streaming response: {response} is not a valid json"
    assert (
        "author" in json_schema_response and "title" in json_schema_response and "genre" in json_schema_response
    ), f"json_schema streaming response: {response} is not a valid book-description"
    assert json_schema_response["genre"] in {
        genre.value for genre in BookType
    }, f"json_schema streaming response: {json_schema_response['genre']} is not a valid book-type"

    response = non_streaming_chat_base(openai_client, json_schema_param)
    try:
        json_schema_response = json.loads(response)
        is_valid = True
    except ValueError:
        is_valid = False

    assert is_valid, f"json_schema non_streaming response: {response} is not a valid json"
    assert (
        "author" in json_schema_response and "title" in json_schema_response and "genre" in json_schema_response
    ), f"json_schema non_streaming response: {response} is not a valid book-description"
    assert json_schema_response["genre"] in {
        genre.value for genre in BookType
    }, f"json_schema non_streaming response: {json_schema_response['genre']} is not a valid book-type"


@pytest.mark.skip(reason="Temporarily skip this case due to unstable execution")
def test_structured_outputs_structural_tag(openai_client):
    """
    Test structured outputs structural_tag functionality with the local service
    """
    content_str = """
        You have the following function available:

        {
            "name": "get_current_date",
            "description": "Get current date and time for given timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone to get current date/time, e.g.: Asia/Shanghai",
                    }
                },
                "required": ["timezone"],
            }
        }

        If you choose to call only this function, reply in this format:
        <{start_tag}={function_name}>{parameters}{end_tag}
        where:

        start_tag => `<function`
        parameters => JSON dictionary with parameter names as keys
        end_tag => `</function>`

        Example:
        <function=example_function>{"param": "value"}</function>

        Note:
        - Function call must follow specified format
        - Required parameters must be specified
        - Only one function can be called at a time
        - Place entire function call response on a single line

        You are an AI assistant. Answer the following question.
    """

    structural_tag_param = {
        "temperature": 1,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "system",
                "content": content_str,
            },
            {
                "role": "user",
                "content": "You're traveling to Shanghai today",
            },
        ],
        "response_format": {
            "type": "structural_tag",
            "structures": [
                {
                    "begin": "<function=get_current_date>",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "Timezone to get current date/time, e.g.: Asia/Shanghai",
                            }
                        },
                        "required": ["timezone"],
                    },
                    "end": "</function>",
                }
            ],
            "triggers": ["<function="],
        },
    }

    expect_str1 = "get_current_date"
    expect_str2 = "Asia/Shanghai"
    response = streaming_chat_base(openai_client, structural_tag_param)
    assert expect_str1 in response, f"structural_tag streaming response: {response} is not as expected"
    assert expect_str2 in response, f"structural_tag streaming response: {response} is not as expected"

    response = non_streaming_chat_base(openai_client, structural_tag_param)
    assert expect_str1 in response, f"structural_tag non_streaming response: {response} is not as expected"
    assert expect_str2 in response, f"structural_tag non_streaming response: {response} is not as expected"


def test_structured_outputs_choice(openai_client):
    """
    Test structured outputs choice functionality with the local service
    """
    choice_param = {
        "temperature": 1,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "What is the landmark building in Shenzhen?"}],
        "extra_body": {
            "guided_choice": ["Ping An Finance Centre", "China Resources Headquarters", "KK100", "Diwang Mansion"]
        },
    }

    response = streaming_chat_base(openai_client, choice_param)
    assert response in [
        "Ping An Finance Centre",
        "China Resources Headquarters",
        "KK100",
        "Diwang Mansion",
    ], f"choice streaming response: {response} is not as expected"
    response = non_streaming_chat_base(openai_client, choice_param)
    assert response in [
        "Ping An Finance Centre",
        "China Resources Headquarters",
        "KK100",
        "Diwang Mansion",
    ], f"choice non_streaming response: {response} is not as expected"


def test_structured_outputs_regex(openai_client):
    """
    Test structured outputs regex functionality with the local service
    """
    regex_param = {
        "temperature": 1,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "Generate a standard format web address including protocol and domain.\n",
            }
        ],
        "extra_body": {"guided_regex": r"^https:\/\/www\.[a-zA-Z]+\.com\/?$\n"},
    }

    import re

    response = streaming_chat_base(openai_client, regex_param)
    assert re.fullmatch(
        r"^https:\/\/www\.[a-zA-Z]+\.com\/?$\n", response
    ), f"regex streaming response: {response} is not as expected"
    response = non_streaming_chat_base(openai_client, regex_param)
    assert re.fullmatch(
        r"^https:\/\/www\.[a-zA-Z]+\.com\/?$\n", response
    ), f"regex non_streaming response: {response} is not as expected"


def test_structured_outputs_grammar(openai_client):
    """
    Test structured outputs grammar functionality with the local service
    """
    html_h1_grammar = """
        root ::= html_statement

        html_statement ::= "<h1" style_attribute? ">" text "</h1>"

        style_attribute ::= " style=" dq style_value dq

        style_value ::= (font_style ("; " font_weight)?) | (font_weight ("; " font_style)?)

        font_style ::= "font-family: '" font_name "'"

        font_weight ::= "font-weight: " weight_value

        font_name ::= "Arial" | "Times New Roman" | "Courier New"

        weight_value ::= "normal" | "bold"

        text ::= [A-Za-z0-9 ]+

        dq ::= ["]
    """

    grammar_param = {
        "temperature": 1,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "Generate HTML code for this heading in bold Times New Roman font: ERNIE Bot",
            }
        ],
        "extra_body": {"guided_grammar": html_h1_grammar},
    }

    import re

    pattern = r'^<h1( style="[^"]*")?>[A-Za-z0-9 ]+</h1>$'
    response = streaming_chat_base(openai_client, grammar_param)
    assert re.fullmatch(pattern, response), f"grammar streaming response: {response} is not as expected"
    response = non_streaming_chat_base(openai_client, grammar_param)
    assert re.fullmatch(pattern, response), f"grammar non_streaming response: {response} is not as expected"


def test_profile_reset_block_num():
    """测试profile reset_block_num功能，与baseline diff不能超过5%"""
    log_file = "./log/config.log"
    baseline = 40000

    if not os.path.exists(log_file):
        pytest.fail(f"Log file not found: {log_file}")

    with open(log_file, "r") as f:
        log_lines = f.readlines()

    target_line = None
    for line in log_lines:
        if "Reset block num" in line:
            target_line = line.strip()
            break

    if target_line is None:
        pytest.fail("日志中没有Reset block num信息")

    match = re.search(r"total_block_num:(\d+)", target_line)
    if not match:
        pytest.fail(f"Failed to extract total_block_num from line: {target_line}")

    try:
        actual_value = int(match.group(1))
    except ValueError:
        pytest.fail(f"Invalid number format: {match.group(1)}")

    lower_bound = baseline * (1 - 0.05)
    upper_bound = baseline * (1 + 0.05)
    print(f"Reset total_block_num: {actual_value}. baseline: {baseline}")

    assert lower_bound <= actual_value <= upper_bound, (
        f"Reset total_block_num {actual_value} 与 baseline {baseline} diff需要在5%以内"
        f"Allowed range: [{lower_bound:.1f}, {upper_bound:.1f}]"
    )


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
