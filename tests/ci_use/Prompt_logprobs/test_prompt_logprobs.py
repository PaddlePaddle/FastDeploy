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

import numpy
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
    # print(json.dumps(resp_json, ensure_ascii=False))

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


def test_logprobs_with_prompt_logprobs_diff():
    """
    测试prompt_logprobs与logprobs的一致性
    """
    data = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 1024,
        "logprobs": True,
        "top_logprobs": 0,
        "return_token_ids": True,
        "temperature": 1,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 33,
    }

    # 构建请求并发送
    response_short = send_request(URL, data)
    resp_json_short = response_short.json()
    print(json.dumps(resp_json_short, ensure_ascii=False))
    prompt_token_ids = resp_json_short["choices"][0]["message"]["prompt_token_ids"]
    completion_token_ids = resp_json_short["choices"][0]["message"]["completion_token_ids"]
    logprobs = resp_json_short["choices"][0]["logprobs"]["content"]
    # assert completions_token_ids
    data2 = {
        "stream": False,
        "messages": [
            {"role": "user", "content": ""},
        ],
        "max_tokens": 1,
        "prompt_logprobs": 0,
        "return_token_ids": True,
        "temperature": 1,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 33,
        "prompt_token_ids": prompt_token_ids + completion_token_ids,
    }

    # 构建请求并发送
    response_long = send_request(URL, data2)
    resp_json_long = response_long.json()
    print(json.dumps(resp_json_long, ensure_ascii=False))
    prompt_logprobs = resp_json_long["choices"][0].get("prompt_logprobs")
    completion_prompt_logprobs = prompt_logprobs[len(prompt_token_ids) :]

    print("======对比1请求的logprob和2请求的后半部分prompt_logprobs======>")

    with open("output_logprobs.log", "w", encoding="utf-8") as f:
        for i in range(len(completion_token_ids)):
            output_token_ids = completion_token_ids[i]
            line = (
                f"{i}, {output_token_ids}, "
                f'logprob={logprobs[i]["logprob"]}, '
                f'prompt_logprob={completion_prompt_logprobs[i][str(output_token_ids)]["logprob"]}\n'
            )
            f.write(line)

    print("====== 校验绝对误差 abs(logprob - prompt_logprob) <= 10 ======")

    MAX_ABS_ERROR = 1.0

    for i in range(len(completion_token_ids)):
        token_id = completion_token_ids[i]
        logprob = logprobs[i]["logprob"]
        prompt_logprob = completion_prompt_logprobs[i][str(token_id)]["logprob"]
        # numpy.testing.assert_allclose(numpy.array(logprob), numpy.array(prompt_logprob))
        numpy.testing.assert_allclose(
            numpy.array(prompt_logprob),
            numpy.array(logprob),
            rtol=3e-1,
            atol=1e-3,
        )
        abs_error = abs(logprob - prompt_logprob)

        assert abs_error <= MAX_ABS_ERROR, (
            f"[ABS_ERROR_TOO_LARGE] "
            f"index={i}, token_id={token_id}, "
            f"logprob={logprob}, "
            f"prompt_logprob={prompt_logprob}, "
            f"abs_error={abs_error}"
        )

        print("✅  所有 token 的绝对误差均 <= 1")


def test_prompt_logprobs_accuracy():
    """
    测试prompt_logprobs的精度,计算一致
    """
    data1 = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "top_p": 1.0,
        "temperature": 0,
        "max_tokens": 10,
        "n": 1,
        "seed": 1,
        "return_token_ids": True,
        "prompt_logprobs": 3,
        "top_k": -1,
    }

    # 构建请求并发送
    response_short = send_request(URL, data1)
    resp_json_short = response_short.json()
    print(json.dumps(resp_json_short, ensure_ascii=False))
    prompt_token_ids = resp_json_short["choices"][0]["message"]["prompt_token_ids"]
    completion_token_ids = resp_json_short["choices"][0]["message"]["completion_token_ids"]
    prompt_short_logprobs = resp_json_short["choices"][0]["prompt_logprobs"]
    # print(json.dumps(prompt_short_logprobs, ensure_ascii=False))

    print("-----------------------prompt_short_logprobs------------------------------------")
    prompt_and_completion_token_ids = prompt_token_ids + completion_token_ids
    data2 = {
        "stream": False,
        "messages": [
            {"role": "user", "content": ""},
        ],
        "top_p": 1.0,
        "temperature": 0,
        "max_tokens": 10,
        "n": 1,
        "seed": 1,
        "prompt_logprobs": 3,
        "top_k": -1,
        "prompt_token_ids": prompt_and_completion_token_ids,
    }
    # 构建请求并发送
    response_long = send_request(URL, data2)
    resp_json_long = response_long.json()
    prompt_long_logprobs = resp_json_long["choices"][0]["prompt_logprobs"]
    print("-----------------------prompt_long_logprobs------------------------------------")
    print(json.dumps(prompt_long_logprobs, ensure_ascii=False))

    for i in range(len(prompt_short_logprobs)):
        assert prompt_long_logprobs[i] == prompt_short_logprobs[i], f"prompt_logprobs mismatch at token index {i}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-sv"]))
