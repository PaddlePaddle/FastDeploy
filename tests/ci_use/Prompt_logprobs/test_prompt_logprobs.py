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


import json
import os
import shutil
import signal
import subprocess
import sys
import time

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

URL = f"http://0.0.0.0:{FD_API_PORT}/v1/chat/completions"
COMPLETIONS_URL = URL.replace("/v1/chat/completions", "/v1/completions")


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
    FD_CONTROLLER_PORT = int(os.getenv("FD_CONTROLLER_PORT", 8633))
    clean_ports([FD_API_PORT, FD_ENGINE_QUEUE_PORT, FD_METRICS_PORT, FD_CACHE_QUEUE_PORT, FD_CONTROLLER_PORT])

    env = os.environ.copy()
    env["FD_USE_GET_SAVE_OUTPUT_V1"] = "1"

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "ERNIE-4.5-0.3B-Paddle")
    else:
        model_path = "/MODELDATA/ERNIE-4.5-0.3B-Paddle"

    log_path = "server.log"
    cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(FD_API_PORT),
        "--max-model-len",
        "65536",
        "--max-logprobs",
        "10",
        "--no-enable-prefix-caching",
        "--enable-logprob",
    ]

    # Start subprocess in new process group
    # 清除log目录
    if os.path.exists("log"):
        shutil.rmtree("log")
    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Enables killing full group via os.killpg
            env=env,
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
        print(f"API server (pid={process.pid}) terminated")
    except Exception as e:
        print(f"Failed to terminate API server: {e}")


def test_unstream_with_prompt_logprobs():
    """
    测试非流式响应prompt_logprobs字段为正整数时,正确返回
    """
    data = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
        "prompt_logprobs": 3,
    }

    response = send_request(URL, data)
    resp_json = response.json()
    print(json.dumps(resp_json, ensure_ascii=False))

    # 校验返回内容与概率信息
    assert resp_json["choices"][0]["message"]["content"] == "牛顿的"
    assert resp_json["usage"]["prompt_tokens"] == 22
    assert resp_json["usage"]["completion_tokens"] == 3
    assert resp_json["usage"]["total_tokens"] == 25

    for i, prompt_logprobs in enumerate(resp_json["choices"][0]["prompt_logprobs"]):
        if i == 0:
            assert prompt_logprobs is None
        else:
            top = sorted(prompt_logprobs.values(), key=lambda x: x["rank"], reverse=False)
            assert top[0]["rank"] == 1
            assert len(top) in {data["prompt_logprobs"], data["prompt_logprobs"] + 1}
            for i in range(len(top)):
                assert top[i]["logprob"] < 0
                assert top[i]["decoded_token"].encode("utf-8")


def test_unstream_with_prompt_logprobs_zero():
    """
    测试非流式响应prompt_logprobs字段为0时返回结果是否正确
    """
    data = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
        "prompt_logprobs": 0,
        "return_token_ids": True,
    }

    response = send_request(URL, data)
    # print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    resp_json = response.json()

    # 校验返回内容与概率信息
    assert resp_json["choices"][0]["message"]["content"] == "牛顿的"
    assert resp_json["usage"]["prompt_tokens"] == 22
    assert resp_json["usage"]["completion_tokens"] == 3
    assert resp_json["usage"]["total_tokens"] == 25

    for i, prompt_logprobs in enumerate(resp_json["choices"][0]["prompt_logprobs"]):
        if i == 0:
            assert prompt_logprobs is None
        else:
            top = list(prompt_logprobs.values())
            token_id = int(list(prompt_logprobs.keys())[0])
            assert top[0]["decoded_token"] is not None
            assert top[0]["logprob"] < 0
            assert top[0]["rank"] >= 1
            assert token_id in resp_json["choices"][0]["message"]["prompt_token_ids"]


def test_unstream_with_prompt_logprobs_none():
    """
    测试非流式响应prompt_logprobs字段为0时返回结果是否正确
    """
    data = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
        "return_token_ids": True,
    }

    response = send_request(URL, data)
    # print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    resp_json = response.json()

    # 校验返回内容与概率信息
    assert resp_json["choices"][0]["message"]["content"] == "牛顿的"
    assert resp_json["usage"]["prompt_tokens"] == 22
    assert resp_json["usage"]["completion_tokens"] == 3
    assert resp_json["usage"]["total_tokens"] == 25
    assert resp_json["choices"][0]["prompt_logprobs"] is None


def test_unstream_with_prompt_logprobs_n():
    """
    测试非流式响应组合n参数，返回内容正常
    """
    data = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
        "prompt_logprobs": 3,
        "n": 3,
    }

    response = send_request(URL, data)
    # print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    resp_json = response.json()

    for j in range(data["n"]):
        for i, prompt_logprobs in enumerate(resp_json["choices"][j]["prompt_logprobs"]):
            if i == 0:
                assert prompt_logprobs is None
            else:
                top = sorted(prompt_logprobs.values(), key=lambda x: x["rank"], reverse=False)
                assert top[0]["rank"] == 1
                assert len(top) in {data["prompt_logprobs"], data["prompt_logprobs"] + 1}
                for i in range(len(top)):
                    assert top[i]["logprob"] < 0
                    assert top[i]["decoded_token"].encode("utf-8")


def test_stream_with_prompt_logprobs():
    """
    测试流式响应prompt_logprobs字段为正整数时,正确返回
    """
    data = {
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
        "prompt_logprobs": 3,
        "return_token_ids": True,
    }

    response = send_request(URL, data)

    result_chunk = {}
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8").removeprefix("data: ")
        if decoded == "[DONE]":
            break

        result_chunk = json.loads(decoded)
        completion_token_ids = result_chunk["choices"][0]["delta"].get("completion_token_ids")
        if completion_token_ids:
            assert "prompt_logprobs" not in result_chunk["choices"][0]
        else:
            for i, prompt_logprobs in enumerate(result_chunk["choices"][0]["prompt_logprobs"]):
                if i == 0:
                    assert prompt_logprobs is None
                else:
                    top = list(prompt_logprobs.values())
                    token_id = int(list(prompt_logprobs.keys())[0])
                    assert top[0]["decoded_token"] is not None
                    assert top[0]["logprob"] < 0
                    assert top[0]["rank"] >= 1
                    assert token_id in result_chunk["choices"][0]["delta"]["prompt_token_ids"]


def test_unstream_with_prompt_logprobs_completions():
    """
    测试completions接口非流式响应prompt_logprobs字段为正整数时,正确返回
    """
    data = {"stream": False, "prompt": "牛顿的三大运动定律是什么？", "max_tokens": 3, "prompt_logprobs": 3}

    response = send_request(COMPLETIONS_URL, data)
    resp_json = response.json()
    # print(json.dumps(resp_json, indent=2, ensure_ascii=False))

    for i, prompt_logprobs in enumerate(resp_json["choices"][0]["prompt_logprobs"]):
        if i == 0:
            assert prompt_logprobs is None
        else:
            top = sorted(prompt_logprobs.values(), key=lambda x: x["rank"], reverse=False)
            assert top[0]["rank"] == 1
            assert len(top) in {data["prompt_logprobs"], data["prompt_logprobs"] + 1}
            for i in range(len(top)):
                assert top[i]["logprob"] < 0
                assert top[i]["decoded_token"].encode("utf-8")


def test_unstream_with_prompt_logprobs_zero_completions():
    """
    测试completions非流式响应prompt_logprobs字段为0时返回结果是否正确
    """
    data = {
        "stream": False,
        "prompt": "牛顿的三大运动定律是什么？",
        "max_tokens": 3,
        "prompt_logprobs": 0,
        "return_token_ids": True,
    }

    response = send_request(COMPLETIONS_URL, data)
    # print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    resp_json = response.json()

    for i, prompt_logprobs in enumerate(resp_json["choices"][0]["prompt_logprobs"]):
        if i == 0:
            assert prompt_logprobs is None
        else:
            top = list(prompt_logprobs.values())
            token_id = int(list(prompt_logprobs.keys())[0])
            assert top[0]["decoded_token"] is not None
            assert top[0]["logprob"] < 0
            assert top[0]["rank"] >= 1
            assert token_id in resp_json["choices"][0]["prompt_token_ids"]


def test_unstream_with_prompt_logprobs_chunk():
    """
    测试chunk切分的能力是否正常
    """
    data = {
        "stream": False,
        "prompt": [10] * (32 * 1024),
        "max_tokens": 1,
        "prompt_logprobs": 1,
    }
    response = send_request(COMPLETIONS_URL, data)
    resp_json = response.json()

    # 校验返回内容与概率信息
    assert resp_json["choices"][0]["text"] is not None
    # assert resp_json["usage"]["prompt_tokens"] == 7
    assert resp_json["usage"]["completion_tokens"] == 1
    for i, prompt_logprobs in enumerate(resp_json["choices"][0]["prompt_logprobs"]):
        if i == 0:
            assert prompt_logprobs is None
        else:
            top = sorted(prompt_logprobs.values(), key=lambda x: x["rank"], reverse=False)
            assert top[0]["rank"] == 1
            assert len(top) in {data["prompt_logprobs"], data["prompt_logprobs"] + 1}
            for i in range(len(top)):
                assert top[i]["logprob"] < 0
                assert top[i]["decoded_token"].encode("utf-8")


def test_unstream_with_prompt_logprobs_chunk_chat():
    """
    测试chunk切分的能力是否正常
    """
    data = {
        "stream": False,
        "messages": [
            {"role": "user", "content": "!hello! " * (8 * 1024)},
        ],
        "max_tokens": 1,
        "prompt_logprobs": 1,
    }
    # 构建请求并发送
    response = send_request(URL, data)
    resp_json = response.json()

    # 校验返回内容与概率信息
    assert resp_json["choices"][0]["message"]["content"] is not None
    # assert resp_json["usage"]["prompt_tokens"] == 7
    assert resp_json["usage"]["completion_tokens"] == 1
    for i, prompt_logprobs in enumerate(resp_json["choices"][0]["prompt_logprobs"]):
        if i == 0:
            assert prompt_logprobs is None
        else:
            top = sorted(prompt_logprobs.values(), key=lambda x: x["rank"], reverse=False)
            assert top[0]["rank"] == 1
            assert len(top) in {data["prompt_logprobs"], data["prompt_logprobs"] + 1}
            for i in range(len(top)):
                assert top[i]["logprob"] < 0
                assert top[i]["decoded_token"].encode("utf-8")


def test_unstream_with_prompt_logprobs_none_completions():
    """
    测试completions非流式响应prompt_logprobs字段为0时返回结果是否正确
    """
    data = {"stream": False, "prompt": "牛顿的三大运动定律是什么？", "max_tokens": 3, "return_token_ids": True}

    response = send_request(COMPLETIONS_URL, data)
    # print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    resp_json = response.json()

    # 校验返回内容与概率信息
    assert resp_json["choices"][0]["text"] is not None
    assert resp_json["usage"]["prompt_tokens"] == 7
    assert resp_json["usage"]["completion_tokens"] == 3
    assert resp_json["choices"][0]["prompt_logprobs"] is None


def test_unstream_with_prompt_logprobs_n_completions():
    """
    测试completions非流式响应组合n参数，返回结果是否正确
    """
    data = {"stream": False, "prompt": "牛顿的三大运动定律是什么？", "max_tokens": 3, "prompt_logprobs": 3, "n": 3}

    response = send_request(COMPLETIONS_URL, data)
    # print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    resp_json = response.json()

    for j in range(data["n"]):
        for i, prompt_logprobs in enumerate(resp_json["choices"][j]["prompt_logprobs"]):
            if i == 0:
                assert prompt_logprobs is None
            else:
                top = sorted(prompt_logprobs.values(), key=lambda x: x["rank"], reverse=False)
                assert top[0]["rank"] == 1
                assert len(top) in {data["prompt_logprobs"], data["prompt_logprobs"] + 1}
                for i in range(len(top)):
                    assert top[i]["logprob"] < 0
                    assert top[i]["decoded_token"].encode("utf-8")


def test_stream_with_prompt_logprobs_completions():
    """
    测试completions流式响应prompt_logprobs字段为正整数时,正确返回
    """
    data = {
        "stream": True,
        "prompt": "牛顿的三大运动定律是什么？",
        "max_tokens": 3,
        "prompt_logprobs": 3,
        "return_token_ids": True,
    }

    response = send_request(COMPLETIONS_URL, data)

    result_chunk = {}
    # first_packet = True
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8").removeprefix("data: ")
        if decoded == "[DONE]":
            break

        result_chunk = json.loads(decoded)
        completion_token_ids = result_chunk["choices"][0].get("completion_token_ids")
        if completion_token_ids:
            # if not first_packet:
            assert result_chunk["choices"][0]["prompt_logprobs"] is None
        else:
            for i, prompt_logprobs in enumerate(result_chunk["choices"][0]["prompt_logprobs"]):
                if i == 0:
                    assert prompt_logprobs is None
                else:
                    top = list(prompt_logprobs.values())
                    token_id = int(list(prompt_logprobs.keys())[0])
                    assert top[0]["decoded_token"] is not None
                    assert top[0]["logprob"] < 0
                    assert top[0]["rank"] >= 1
                    assert token_id in result_chunk["choices"][0]["prompt_token_ids"]
            # first_packet = False


def test_unstream_with_prompt_logprobs_list_completions():
    """
    测试completions非流式响应组合list prompt，返回结果是否正确
    """
    data = {
        "stream": False,
        "prompt": ["牛顿的三大运动定律是什么？", "什么是机器学习？"],
        "max_tokens": 10,
        "prompt_logprobs": 3,
        "n": 3,
    }

    response = send_request(COMPLETIONS_URL, data)
    # print(json.dumps(response.json(), ensure_ascii=False))
    resp_json = response.json()

    for j in range(data["n"] * len(data["prompt"])):
        for i, prompt_logprobs in enumerate(resp_json["choices"][j]["prompt_logprobs"]):
            if i == 0:
                assert prompt_logprobs is None
            else:
                top = sorted(prompt_logprobs.values(), key=lambda x: x["rank"], reverse=False)
                assert top[0]["rank"] == 1
                assert len(top) in {data["prompt_logprobs"], data["prompt_logprobs"] + 1}
                for i in range(len(top)):
                    assert top[i]["logprob"] < 0
                    assert top[i]["decoded_token"].encode("utf-8")


def test_unstream_with_prompt_logprobs_no_decode_completions():
    """
    测试completions非流式响应组合关闭decode
    """
    data = {
        "stream": False,
        "prompt": ["牛顿的三大运动定律是什么？"],
        "max_tokens": 10,
        "prompt_logprobs": 1,
        "include_logprobs_decode_token": False,
    }

    response = send_request(COMPLETIONS_URL, data)
    # print(json.dumps(response.json(), ensure_ascii=False))
    resp_json = response.json()

    for i, prompt_logprobs in enumerate(resp_json["choices"][0]["prompt_logprobs"]):
        if i == 0:
            assert prompt_logprobs is None
        else:
            top = sorted(prompt_logprobs.values(), key=lambda x: x["rank"], reverse=False)
            assert top[0]["rank"] == 1
            assert len(top) in {data["prompt_logprobs"], data["prompt_logprobs"] + 1}
            for i in range(len(top)):
                assert top[i]["logprob"] < 0
                assert top[i]["decoded_token"] is None


def test_unstream_with_prompt_logprobs_no_decode():
    """
    测试completions非流式响应组合关闭decode
    """
    data = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 10,
        "logprobs": True,
        "top_logprobs": 3,
        "prompt_logprobs": 1,
        "include_logprobs_decode_token": False,
    }

    response = send_request(URL, data)
    # print(json.dumps(response.json(), ensure_ascii=False))
    resp_json = response.json()

    for i, prompt_logprobs in enumerate(resp_json["choices"][0]["prompt_logprobs"]):
        if i == 0:
            assert prompt_logprobs is None
        else:
            top = sorted(prompt_logprobs.values(), key=lambda x: x["rank"], reverse=False)
            assert top[0]["rank"] == 1
            assert len(top) in {data["prompt_logprobs"], data["prompt_logprobs"] + 1}
            for i in range(len(top)):
                assert top[i]["logprob"] < 0
                assert top[i]["decoded_token"] is None
    for i, logprobs in enumerate(resp_json["choices"][0]["logprobs"]["content"]):
        # assert logprobs is not None
        assert len(logprobs["top_logprobs"]) == data["top_logprobs"]
        assert logprobs["token"] in ("", None)
        assert logprobs["logprob"] < 0


def test_error_with_prompt_logprobs():
    """
    测试prompt_logprobs的校验信息
    """
    data = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
        "prompt_logprobs": 15,
    }

    response = send_request(URL, data)
    resp_json = response.json()

    assert (
        "Number of prompt_logprobs requested (15) exceeds maximum allowed value (10)" in resp_json["error"]["message"]
    )


def send_request(url, payload, timeout=600, stream=False):
    """
    向指定URL发送POST请求，并返回响应结果。
    """
    headers = {
        "Content-Type": "application/json",
    }

    try:
        res = requests.post(url, headers=headers, json=payload, stream=stream, timeout=timeout)
        return res
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时（超过 {timeout} 秒）")
        # base_logger.error(f"❌ 请求超时（超过 {timeout} 秒）")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败：{e}")
        # base_logger.error(f"❌ 请求失败：{e}")
        return None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-sv"]))
