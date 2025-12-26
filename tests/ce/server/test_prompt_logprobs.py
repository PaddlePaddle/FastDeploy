#!/bin/env python3
# -*- coding: utf-8 -*-
# @author xujing43
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
Checking for /v1/completions parameters
"""

import json

from core import TEMPLATE, URL, build_request_payload, send_request

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
        "max_tokens": 3,
        "prompt_logprobs": 3,
    }

    # 构建请求并发送
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload)
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

    # 构建请求并发送
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload)
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

    # 构建请求并发送
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload)
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

    # 构建请求并发送
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload)
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

    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload)

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

    # 构建请求并发送
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(COMPLETIONS_URL, payload)
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

    # 构建请求并发送
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(COMPLETIONS_URL, payload)
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
    data = {"stream": False, "prompt": [10] * (32 * 1024), "max_tokens": 1, "return_token_ids": True}

    # 构建请求并发送
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(COMPLETIONS_URL, payload)
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    resp_json = response.json()

    # 校验返回内容与概率信息
    assert resp_json["choices"][0]["text"] is not None
    # assert resp_json["usage"]["prompt_tokens"] == 7
    assert resp_json["usage"]["completion_tokens"] == 1
    assert resp_json["choices"][0]["prompt_logprobs"] is None


def test_unstream_with_prompt_logprobs_none_completions():
    """
    测试completions非流式响应prompt_logprobs字段为0时返回结果是否正确
    """
    data = {"stream": False, "prompt": "牛顿的三大运动定律是什么？", "max_tokens": 3, "return_token_ids": True}

    # 构建请求并发送
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(COMPLETIONS_URL, payload)
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

    # 构建请求并发送
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(COMPLETIONS_URL, payload)
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
        # "return_token_ids":True
    }

    payload = build_request_payload(TEMPLATE, data)
    response = send_request(COMPLETIONS_URL, payload)

    result_chunk = {}
    first_packet = True
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8").removeprefix("data: ")
        if decoded == "[DONE]":
            break

        result_chunk = json.loads(decoded)
        # completion_token_ids = result_chunk["choices"][0].get("completion_token_ids")
        # if completion_token_ids:
        if not first_packet:
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
                    # assert token_id in result_chunk["choices"][0]["prompt_token_ids"]
            first_packet = False


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

    # 构建请求并发送
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(COMPLETIONS_URL, payload)
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

    # 构建请求并发送
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(COMPLETIONS_URL, payload)
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

    # 构建请求并发送
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload)
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


if __name__ == "__main__":
    # chat接口返回
    test_unstream_with_prompt_logprobs()
    test_unstream_with_prompt_logprobs_zero()
    test_unstream_with_prompt_logprobs_none()
    test_unstream_with_prompt_logprobs_n()
    test_stream_with_prompt_logprobs()
    # chunk切分检查
    test_unstream_with_prompt_logprobs_chunk()
    # completions接口返回
    test_unstream_with_prompt_logprobs_completions()
    test_unstream_with_prompt_logprobs_zero_completions()
    test_unstream_with_prompt_logprobs_none_completions()
    test_unstream_with_prompt_logprobs_n_completions()
    test_stream_with_prompt_logprobs_completions()
    # list[str]返回
    test_unstream_with_prompt_logprobs_list_completions()
    # 关闭decode
    test_unstream_with_prompt_logprobs_no_decode_completions()
    test_unstream_with_prompt_logprobs_no_decode()
