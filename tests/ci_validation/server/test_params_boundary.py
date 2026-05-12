#!/bin/env python3
# -*- coding: utf-8 -*-
# @author xujing43
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
Boundary value checking for API parameters
"""


from core import TEMPLATE, URL, build_request_payload, send_request


def test_max_min_1_token():
    data = {
        "stream": False,
        "messages": [{"role": "user", "content": "非洲的首都是？"}],
        "max_tokens": 1,
        "metadata": {"min_tokens": 1},
    }
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload).json()

    response_object = response["object"]
    assert "error" not in response_object, f"响应中包含错误信息: {response_object}"
    completion_tokens = response["usage"]["completion_tokens"]
    assert completion_tokens == 1, f"实际生成的token数为: {completion_tokens}, 应该为1"
    finish_reason = response["choices"][0]["finish_reason"]
    assert finish_reason == "length", f"内容不可能完整生成, 但实际finish_reason为: {response}"
