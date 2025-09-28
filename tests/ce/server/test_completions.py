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


def test_completion_total_tokens():
    data = {
        "prompt": "你是谁",
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }

    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(COMPLETIONS_URL, payload, stream=True)
    last_data = None
    for line in resp.iter_lines(decode_unicode=True):
        if line.strip() == "data: [DONE]":
            break
        if line.strip() == "" or not line.startswith("data: "):
            continue
        line = line[len("data: ") :]
        last_data = json.loads(line)
    usage = last_data["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert "total_tokens" in usage, "total_tokens 不存在"
    assert usage["total_tokens"] == total_tokens, "total_tokens计数不正确"


def test_completion_echo_stream_one_prompt_rti():
    """
    测试echo参数在流式回复中，且设置为仅回复一个prompt
    """
    data = {
        "prompt": "水果的营养价值是如何的？",
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "echo": True,
        "max_tokens": 2,
        "return_token_ids": True,
    }

    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(COMPLETIONS_URL, payload, stream=True)
    # 初始化计数器
    counter = 0
    second_data = None
    for line in resp.iter_lines(decode_unicode=True):
        if line.strip() == "data: [DONE]":
            break
        if line.strip() == "" or not line.startswith("data: "):
            continue
        line = line[len("data: ") :]
        stream_data = json.loads(line)
        counter += 1
        if counter == 2:  # 当计数器为2时，保存第二包数据
            second_data = stream_data
            break  # 如果只需要第二包数据，可以在这里直接退出循环
    text = second_data["choices"][0]["text"]
    assert data["prompt"] in text, "echo回显不正确"
    position = text.find(data["prompt"])
    assert position == 0, "echo回显没有在靠前的位置"


def test_completion_echo_stream_one_prompt():
    """
    测试echo参数在流式回复中，且设置为仅回复一个prompt
    """
    data = {
        "prompt": "水果的营养价值是如何的？",
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "echo": True,
        "max_tokens": 2,
    }

    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(COMPLETIONS_URL, payload, stream=True)
    # 初始化计数器
    counter = 0
    second_data = None
    for line in resp.iter_lines(decode_unicode=True):
        if line.strip() == "data: [DONE]":
            break
        if line.strip() == "" or not line.startswith("data: "):
            continue
        line = line[len("data: ") :]
        stream_data = json.loads(line)
        counter += 1
        if counter == 1:  # 当计数器为1时，保存第一包数据
            second_data = stream_data
            break  # 如果只需要第二包数据，可以在这里直接退出循环
    text = second_data["choices"][0]["text"]
    assert data["prompt"] in text, "echo回显不正确"
    position = text.find(data["prompt"])
    assert position == 0, "echo回显没有在靠前的位置"


def test_completion_echo_stream_more_prompt():
    """
    测试echo参数在流式回复中，且设置为回复多个prompt
    """
    data = {
        "prompt": ["水果的营养价值是如何的？", "水的化学式是什么？"],
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "echo": True,
        "max_tokens": 2,
        "return_token_ids": True,
    }

    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(COMPLETIONS_URL, payload, stream=True)
    # 初始化字典来存储每个index的第二包数据
    second_data_by_index = {0: None, 1: None}
    # 初始化字典来记录每个index的包计数
    packet_count_by_index = {0: 0, 1: 0}

    for line in resp.iter_lines(decode_unicode=True):
        if line.strip() == "data: [DONE]":
            break
        if line.strip() == "" or not line.startswith("data: "):
            continue
        line = line[len("data: ") :]
        stream_data = json.loads(line)

        for choice in stream_data.get("choices", []):
            index = choice.get("index")
            if index in packet_count_by_index:
                packet_count_by_index[index] += 1
                if packet_count_by_index[index] == 2:
                    second_data_by_index[index] = choice
                    if all(value is not None for value in second_data_by_index.values()):
                        break
    text_0 = second_data_by_index[0]["text"]
    text_1 = second_data_by_index[1]["text"]
    assert data["prompt"][0] in text_0, "echo回显不正确"
    assert data["prompt"][1] in text_1, "echo回显不正确"
    position_0 = text_0.find(data["prompt"][0])
    assert position_0 == 0, "prompt[0]的echo回显没有在靠前的位置"
    position_1 = text_1.find(data["prompt"][1])
    assert position_1 == 0, "prompt[1]的echo回显没有在靠前的位置"


def test_completion_echo_one_prompt():
    """
    测试echo参数在非流式回复中，且设置为仅发送一个prompt
    """
    data = {
        "stream": False,
        "prompt": "水果的营养价值是如何的？",
        "echo": True,
        "max_tokens": 100,
    }
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(COMPLETIONS_URL, payload)
    response = response.json()

    text = response["choices"][0]["text"]
    assert data["prompt"] in text, "echo回显不正确"
    position = text.find(data["prompt"])
    assert position == 0, "echo回显没有在靠前的位置"


def test_completion_echo_more_prompt():
    """
    测试echo参数在非流式回复中，且设置为发送多个prompt
    """
    data = {
        "stream": False,
        "prompt": ["水果的营养价值是如何的？", "水的化学式是什么？"],
        "echo": True,
        "max_tokens": 100,
    }
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(COMPLETIONS_URL, payload).json()

    text_0 = response["choices"][0]["text"]
    text_1 = response["choices"][1]["text"]
    assert data["prompt"][0] in text_0, "echo回显不正确"
    assert data["prompt"][1] in text_1, "echo回显不正确"
    position_0 = text_0.find(data["prompt"][0])
    assert position_0 == 0, "prompt[0]的echo回显没有在靠前的位置"
    position_1 = text_1.find(data["prompt"][1])
    assert position_1 == 0, "prompt[1]的echo回显没有在靠前的位置"


def test_completion_finish_length():
    """
    非流式回复中,因达到max_token截断检查finish_reasoning参数
    """
    data = {"stream": False, "prompt": "水果的营养价值是如何的？", "max_tokens": 10}

    payload = build_request_payload(TEMPLATE, data)
    response = send_request(COMPLETIONS_URL, payload).json()

    finish_reason = response["choices"][0]["finish_reason"]
    assert finish_reason == "length", "达到max_token时，finish_reason不为length"


def test_completion_finish_stop():
    """
    非流式回复中,模型自然回复完成，检查finish_reasoning参数
    """
    data = {"stream": False, "prompt": "简短的回答我：苹果是水果吗？"}

    payload = build_request_payload(TEMPLATE, data)
    response = send_request(COMPLETIONS_URL, payload).json()

    finish_reason = response["choices"][0]["finish_reason"]
    assert finish_reason == "stop", "无任何中介，finish_reason不为stop"
