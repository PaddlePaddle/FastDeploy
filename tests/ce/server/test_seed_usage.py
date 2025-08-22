#!/bin/env python3
# -*- coding: utf-8 -*-
# @author ZhangYulongg
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import json

from core import TEMPLATE, URL, build_request_payload, get_stream_chunks, send_request


def test_seed_stream():
    """测试payload seed参数"""
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "seed": 26,
        "max_tokens": 50,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }

    payload = build_request_payload(TEMPLATE, data)
    response_1 = send_request(url=URL, payload=payload, stream=True)
    # print(response_1.text)
    chunks_1 = get_stream_chunks(response_1)
    # print(chunks_1)
    # for idx, chunk in enumerate(chunks_1):
    #     print(f"\nchunk[{idx}]:\n{json.dumps(chunk, indent=2, ensure_ascii=False)}")
    resul_1 = "".join([x["choices"][0]["delta"]["content"] for x in chunks_1[:-1]])
    logprobs_1 = [json.dumps(x["choices"][0]["logprobs"]["content"][0], ensure_ascii=False) for x in chunks_1[1:-1]]
    # print(resul_1)
    # print(logprobs_1, type(logprobs_1[0]))

    response_2 = send_request(url=URL, payload=payload, stream=True)
    chunks_2 = get_stream_chunks(response_2)
    resul_2 = "".join([x["choices"][0]["delta"]["content"] for x in chunks_2[:-1]])
    logprobs_2 = [json.dumps(x["choices"][0]["logprobs"]["content"][0], ensure_ascii=False) for x in chunks_2[1:-1]]
    # print(resul_2)

    assert resul_1 == resul_2, "top_p=0, 固定seed, 两次请求结果不一致"
    for idx, (l1, l2) in enumerate(zip(logprobs_1, logprobs_2)):
        assert l1 == l2, f"top_p=0, 固定seed, logprobs[{idx}]不一致"


def test_chat_usage_stream():
    """测试payload max_tokens参数"""
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 50,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "metadata": {"min_tokens": 10},
    }

    payload = build_request_payload(TEMPLATE, data)
    response = send_request(url=URL, payload=payload, stream=True)
    chunks = get_stream_chunks(response)
    # for idx, chunk in enumerate(chunks):
    #     print(f"\nchunk[{idx}]:\n{json.dumps(chunk, indent=2, ensure_ascii=False)}")

    usage = chunks[-1]["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert data["max_tokens"] >= usage["completion_tokens"], f"completion_tokens大于max_tokens, usage: {usage}"
    assert (
        data["metadata"]["min_tokens"] <= usage["completion_tokens"]
    ), f"completion_tokens小于min_tokens, usage: {usage}"
    assert (
        usage["total_tokens"] == total_tokens
    ), f"total_tokens不等于prompt_tokens + completion_tokens, usage: {usage}"


def test_chat_usage_non_stream():
    """测试非流式 usage"""
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 50,
        "stream": False,
        "metadata": {"min_tokens": 10},
    }

    payload = build_request_payload(TEMPLATE, data)

    response = send_request(url=URL, payload=payload).json()
    # print(response)
    # chunks = get_stream_chunks(response)
    # for idx, chunk in enumerate(chunks):
    #     print(f"\nchunk[{idx}]:\n{json.dumps(chunk, indent=2, ensure_ascii=False)}")

    usage = response["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert data["max_tokens"] >= usage["completion_tokens"], f"completion_tokens大于max_tokens, usage: {usage}"
    assert (
        data["metadata"]["min_tokens"] <= usage["completion_tokens"]
    ), f"completion_tokens小于min_tokens, usage: {usage}"
    assert (
        usage["total_tokens"] == total_tokens
    ), f"total_tokens不等于prompt_tokens + completion_tokens, usage: {usage}"


def test_non_chat_usage_stream():
    """测试completions 流式 usage"""
    data = {
        "prompt": "牛顿的三大运动定律是什么？",
        "max_tokens": 50,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "metadata": {"min_tokens": 10},
    }
    completion_url = URL.replace("chat/completions", "completions")

    payload = build_request_payload(TEMPLATE, data)

    response = send_request(url=completion_url, payload=payload, stream=True)
    chunks = get_stream_chunks(response)
    # for idx, chunk in enumerate(chunks):
    #     print(f"\nchunk[{idx}]:\n{json.dumps(chunk, indent=2, ensure_ascii=False)}")

    usage = chunks[-1]["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert data["max_tokens"] >= usage["completion_tokens"], f"completion_tokens大于max_tokens, usage: {usage}"
    assert (
        data["metadata"]["min_tokens"] <= usage["completion_tokens"]
    ), f"completion_tokens小于min_tokens, usage: {usage}"
    assert (
        usage["total_tokens"] == total_tokens
    ), f"total_tokens不等于prompt_tokens + completion_tokens, usage: {usage}"


def test_non_chat_usage_non_stream():
    """测试completions 非流式 usage"""
    data = {
        "prompt": "牛顿的三大运动定律是什么？",
        "max_tokens": 50,
        "stream": False,
        "metadata": {"min_tokens": 10},
    }
    completion_url = URL.replace("chat/completions", "completions")

    payload = build_request_payload(TEMPLATE, data)

    response = send_request(url=completion_url, payload=payload).json()
    # print(response)
    # chunks = get_stream_chunks(response)
    # for idx, chunk in enumerate(chunks):
    #     print(f"\nchunk[{idx}]:\n{json.dumps(chunk, indent=2, ensure_ascii=False)}")

    usage = response["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert data["max_tokens"] >= usage["completion_tokens"], f"completion_tokens大于max_tokens, usage: {usage}"
    assert (
        data["metadata"]["min_tokens"] <= usage["completion_tokens"]
    ), f"completion_tokens小于min_tokens, usage: {usage}"
    assert (
        usage["total_tokens"] == total_tokens
    ), f"total_tokens不等于prompt_tokens + completion_tokens, usage: {usage}"


if __name__ == "__main__":
    test_seed_stream()
