#!/bin/env python3
# -*- coding: utf-8 -*-
# @author xujing43
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
Checking for /v1/completions parameters
"""

import json

from core import (
    TEMPLATE,
    URL,
    build_request_payload,
    send_request,
)

URL = URL.replace("/v1/chat/completions", "/v1/completions")

def test_completion_total_tokens():
    data = {
        "prompt": "你是谁",
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload, stream=True)
    last_data = None
    for line in resp.iter_lines(decode_unicode=True):
        if line.strip() == "data: [DONE]":
            break
        if line.strip() == "" or not line.startswith("data: "):
            continue
        line = line[len("data: "):]
        last_data = json.loads(line)
    usage = last_data["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert "total_tokens" in usage, "total_tokens 不存在"
    assert usage["total_tokens"]== total_tokens, "total_tokens计数不正确"
    