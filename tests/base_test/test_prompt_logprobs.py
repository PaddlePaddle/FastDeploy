#!/bin/env python3
# -*- coding: utf-8 -*-
# @author xujing43
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
Checking for prompt_logprobs
"""

import json
import os

import numpy
from utils import send_request

URL_HOST = os.getenv("URL_HOST", "10.174.137.88")
URL_PORT = os.getenv("URL_PORT", "8801")

URL = f"http://{URL_HOST}:{URL_PORT}/v1/chat/completions"
print(f"FD URL: {URL}")
COMPLETIONS_URL = URL.replace("/v1/chat/completions", "/v1/completions")


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
        "max_completion_tokens": 3,
        "prompt_logprobs": 3,
    }

    # 构建请求并发送
    response = send_request(URL, data)
    resp_json = response.json()
    print(json.dumps(response.json(), ensure_ascii=False))
    # 校验返回内容与概率信息
    # assert resp_json["choices"][0]["message"]["content"] == "牛顿的"
    assert resp_json["usage"]["completion_tokens"] == 3

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
        "max_completion_tokens": 3,
        "prompt_logprobs": 0,
        "return_token_ids": True,
    }

    # 构建请求并发送
    response = send_request(URL, data)
    # print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    resp_json = response.json()

    # 校验返回内容与概率信息
    # assert resp_json["choices"][0]["message"]["content"] == "牛顿的"
    assert resp_json["usage"]["completion_tokens"] == 3

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
        "max_completion_tokens": 3,
        "return_token_ids": True,
    }

    # 构建请求并发送
    response = send_request(URL, data)
    # print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    resp_json = response.json()

    # 校验返回内容与概率信息
    # assert resp_json["choices"][0]["message"]["content"] == "牛顿的"
    assert resp_json["usage"]["completion_tokens"] == 3
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
        "max_completion_tokens": 3,
        "prompt_logprobs": 3,
        "n": 3,
    }

    # 构建请求并发送
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
        "max_completion_tokens": 3,
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
    data = {"stream": False, "prompt": "牛顿的三大运动定律是什么？", "max_completion_tokens": 3, "prompt_logprobs": 3}

    # 构建请求并发送
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
        "max_completion_tokens": 3,
        "prompt_logprobs": 0,
        "return_token_ids": True,
    }

    # 构建请求并发送
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
        "max_completion_tokens": 1,
        "prompt_logprobs": 1,
    }
    # 构建请求并发送
    response = send_request(URL, data)
    resp_json = response.json()
    # print(json.dumps(resp_json, ensure_ascii=False))

    # 校验返回内容与概率信息
    assert resp_json["choices"][0]["message"]["content"] is not None
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
        "max_completion_tokens": 1,
        "prompt_logprobs": 1,
    }
    # 构建请求并发送
    response = send_request(COMPLETIONS_URL, data)
    resp_json = response.json()
    print(json.dumps(resp_json, ensure_ascii=False))

    # 校验返回内容与概率信息
    assert resp_json["choices"][0]["text"] is not None
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
    data = {
        "stream": False,
        "prompt": "牛顿的三大运动定律是什么？",
        "max_completion_tokens": 3,
        "return_token_ids": True,
    }

    # 构建请求并发送
    response = send_request(COMPLETIONS_URL, data)
    # print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    resp_json = response.json()

    # 校验返回内容与概率信息
    assert resp_json["choices"][0]["text"] is not None
    assert resp_json["usage"]["completion_tokens"] == 3
    assert resp_json["choices"][0]["prompt_logprobs"] is None


def test_unstream_with_prompt_logprobs_n_completions():
    """
    测试completions非流式响应组合n参数，返回结果是否正确
    """
    data = {
        "stream": False,
        "prompt": "牛顿的三大运动定律是什么？",
        "max_completion_tokens": 3,
        "prompt_logprobs": 3,
        "n": 3,
    }

    # 构建请求并发送
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
        "max_completion_tokens": 3,
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
        print(result_chunk)
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
        "max_completion_tokens": 10,
        "prompt_logprobs": 3,
        "n": 3,
    }

    # 构建请求并发送
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
        "max_completion_tokens": 10,
        "prompt_logprobs": 1,
        "include_logprobs_decode_token": False,
    }

    # 构建请求并发送
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
        "max_completion_tokens": 10,
        "logprobs": True,
        "top_logprobs": 3,
        "prompt_logprobs": 1,
        "include_logprobs_decode_token": False,
    }

    # 构建请求并发送
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
        "max_completion_tokens": 3,
        "prompt_logprobs": 25,
    }

    # 构建请求并发送
    response = send_request(URL, data)
    resp_json = response.json()
    print(json.dumps(resp_json, ensure_ascii=False))

    assert (
        "Number of prompt_logprobs requested (25) exceeds maximum allowed value (20)" in resp_json["error"]["message"]
    )


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
        "max_completion_tokens": 1024,
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
        "max_completion_tokens": 1,
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
        "max_completion_tokens": 10,
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
        "max_completion_tokens": 10,
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
