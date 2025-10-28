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


def test_completion_stream_prompt_tokens_completion_tokens():
    """
    /v1/completions接口, stream=True
    return "prompt_tokens"和"reasoning_content"
    """
    data = {
        "prompt": "你是谁",
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "max_tokens": 50,
        "return_token_ids": True,
    }

    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(COMPLETIONS_URL, payload, stream=True)
    for line in resp.iter_lines(decode_unicode=True):
        if line.strip() == "data: [DONE]":
            break
        if line.strip() == "" or not line.startswith("data: "):
            continue
        line = line[len("data: ") :]
        response_data = json.loads(line)

        choice = response_data["choices"][0]
        if "prompt_token_ids" in choice and choice["prompt_token_ids"] is not None:
            prompt_tokens = choice["prompt_tokens"]
            assert data["prompt"] in prompt_tokens, "prompt_tokens取值结果不正确"
        else:
            completion_tokens = choice["completion_tokens"]
            reasoning_content = choice["reasoning_content"]
            text = choice["text"]
            assert reasoning_content or text in completion_tokens, "completion_tokens取值结果不正确"
        if "finish_reason" in line.strip():
            break


def test_completion_prompt_tokens_completion_tokens_return_token_ids():
    """
    /v1/completions接口,非流式接口
    return "prompt_tokens"和"reasoning_content"
    """
    data = {"stream": False, "prompt": "你是谁", "max_tokens": 50, "return_token_ids": True}
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(COMPLETIONS_URL, payload).json()

    prompt_tokens = resp["choices"][0]["prompt_tokens"]
    assert data["prompt"] in prompt_tokens, "prompt_tokens取值结果不正确"

    completion_tokens = resp["choices"][0]["completion_tokens"]
    reasoning_content = resp["choices"][0]["reasoning_content"]
    text = resp["choices"][0]["text"]
    assert reasoning_content or text in completion_tokens, "completion_tokens取值结果不正确"


def test_completion_prompt_tokens_completion_tokens():
    """
    /v1/completions接口,无return_token_ids参数
    非流式接口中,无return token ids 属性"prompt_tokens"和"reasoning_content"值为null
    """
    data = {"stream": False, "prompt": "你是谁", "max_tokens": 50}
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(COMPLETIONS_URL, payload).json()

    prompt_tokens = resp["choices"][0]["prompt_tokens"]
    assert prompt_tokens is None, "prompt_tokens取值结果不正确"

    completion_tokens = resp["choices"][0]["completion_tokens"]
    assert completion_tokens is None, "completion_tokens取值结果不正确"


def test_stream_prompt_tokens_completion_tokens():
    """
    /v1/chat/completions接口,"stream": True
    返回属性"prompt_tokens"和"reasoning_content"
    """
    data = {
        "messages": [{"role": "user", "content": "你是谁"}],
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "max_tokens": 50,
        "return_token_ids": True,
    }

    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload, stream=True)
    for line in resp.iter_lines(decode_unicode=True):
        if line.strip() == "data: [DONE]":
            break
        if line.strip() == "" or not line.startswith("data: "):
            continue
        line = line[len("data: ") :]
        response_data = json.loads(line)

        choice = response_data["choices"][0]
        if "prompt_token_ids" in choice["delta"] and choice["delta"]["prompt_token_ids"] is not None:
            prompt_tokens = choice["delta"]["prompt_tokens"]
            assert data["messages"][0]["content"] in prompt_tokens, "prompt_tokens取值结果不正确"
        else:
            completion_tokens = choice["delta"]["completion_tokens"]
            reasoning_content = choice["delta"]["reasoning_content"]
            content = choice["delta"]["content"]
            assert reasoning_content or content in completion_tokens, "completion_tokens取值结果不正确"
        if "finish_reason" in line.strip():
            break


def test_prompt_tokens_completion_tokens_return_token_ids():
    """
    /v1/chat/completions接口,非流式接口
    返回属性"prompt_tokens"和"reasoning_content"
    """
    data = {
        "stream": False,
        "messages": [{"role": "user", "content": "你是谁"}],
        "max_tokens": 50,
        "return_token_ids": True,
        "logprobs": False,
        "top_logprobs": None,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()

    prompt_tokens = resp["choices"][0]["message"]["prompt_tokens"]
    assert data["messages"][0]["content"] in prompt_tokens, "prompt_tokens取值结果不正确"

    completion_tokens = resp["choices"][0]["message"]["completion_tokens"]
    reasoning_content = resp["choices"][0]["message"]["reasoning_content"]
    text = resp["choices"][0]["message"]["content"]
    assert reasoning_content or text in completion_tokens, "completion_tokens取值结果不正确"


def test_prompt_tokens_completion_tokens():
    """
    /v1/chat/completions接口,无return_token_ids参数
    无return token ids 属性"prompt_tokens"和"reasoning_content"值为null
    """
    data = {
        "stream": False,
        "messages": [{"role": "user", "content": "你是谁"}],
        "max_tokens": 50,
        "logprobs": False,
        "top_logprobs": None,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()

    prompt_tokens = resp["choices"][0]["message"]["prompt_tokens"]
    assert prompt_tokens is None, "prompt_tokens取值结果不正确"

    completion_tokens = resp["choices"][0]["message"]["completion_tokens"]
    assert completion_tokens is None, "completion_tokens取值结果不正确"
