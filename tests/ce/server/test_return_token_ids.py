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


def test_completion_stream_text_after_process_raw_prediction():
    """
    /v1/completions接口, stream=True
    返回属性"text_after_process"和"reasoning_content"
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
            text_after_process = choice["text_after_process"]
            assert data["prompt"] in text_after_process, "text_after_process取值结果不正确"
        else:
            raw_prediction = choice["raw_prediction"]
            reasoning_content = choice["reasoning_content"]
            text = choice["text"]
            assert reasoning_content or text in raw_prediction, "raw_prediction取值结果不正确"
        if "finish_reason" in line.strip():
            break


def test_completion_text_after_process_raw_predictio_return_token_ids():
    """
    /v1/completions接口,非流式接口
    返回属性"text_after_process"和"reasoning_content"
    """
    data = {"stream": False, "prompt": "你是谁", "max_tokens": 50, "return_token_ids": True}
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(COMPLETIONS_URL, payload).json()

    text_after_process = resp["choices"][0]["text_after_process"]
    assert data["prompt"] in text_after_process, "text_after_process取值结果不正确"

    raw_prediction = resp["choices"][0]["raw_prediction"]
    reasoning_content = resp["choices"][0]["reasoning_content"]
    text = resp["choices"][0]["text"]
    assert reasoning_content or text in raw_prediction, "raw_prediction取值结果不正确"


def test_completion_text_after_process_raw_prediction():
    """
    /v1/completions接口,无return_token_ids参数
    非流式接口中,无return token ids 属性"text_after_process"和"reasoning_content"值为null
    """
    data = {"stream": False, "prompt": "你是谁", "max_tokens": 50}
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(COMPLETIONS_URL, payload).json()

    text_after_process = resp["choices"][0]["text_after_process"]
    assert text_after_process is None, "text_after_process取值结果不正确"

    raw_prediction = resp["choices"][0]["raw_prediction"]
    assert raw_prediction is None, "raw_prediction取值结果不正确"


def test_stream_text_after_process_raw_prediction():
    """
    /v1/chat/completions接口,"stream": True
    返回属性"text_after_process"和"reasoning_content"
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
            text_after_process = choice["delta"]["text_after_process"]
            assert data["messages"][0]["content"] in text_after_process, "text_after_process取值结果不正确"
        else:
            raw_prediction = choice["delta"]["raw_prediction"]
            reasoning_content = choice["delta"]["reasoning_content"]
            content = choice["delta"]["content"]
            assert reasoning_content or content in raw_prediction, "raw_prediction取值结果不正确"
        if "finish_reason" in line.strip():
            break


def test_text_after_process_raw_prediction_return_token_ids():
    """
    /v1/chat/completions接口,非流式接口
    返回属性"text_after_process"和"reasoning_content"
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

    text_after_process = resp["choices"][0]["message"]["text_after_process"]
    assert data["messages"][0]["content"] in text_after_process, "text_after_process取值结果不正确"

    raw_prediction = resp["choices"][0]["message"]["raw_prediction"]
    reasoning_content = resp["choices"][0]["message"]["reasoning_content"]
    text = resp["choices"][0]["message"]["content"]
    assert reasoning_content or text in raw_prediction, "raw_prediction取值结果不正确"


def test_text_after_process_raw_prediction():
    """
    /v1/chat/completions接口,无return_token_ids参数
    无return token ids 属性"text_after_process"和"reasoning_content"值为null
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

    text_after_process = resp["choices"][0]["message"]["text_after_process"]
    assert text_after_process is None, "text_after_process取值结果不正确"

    raw_prediction = resp["choices"][0]["message"]["raw_prediction"]
    assert raw_prediction is None, "raw_prediction取值结果不正确"
