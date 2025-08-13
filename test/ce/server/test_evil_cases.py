#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
边缘检测 ，攻击性测试
"""


import pytest
from core import TEMPLATE, URL, build_request_payload, send_request


def test_missing_messages_field():
    """缺失 messages 字段，服务应返回合理错误，而非崩溃"""
    data = {
        "stream": False,
        "max_tokens": 10,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()

    assert "detail" in resp, "返回中未包含 detail 错误信息字段"
    assert any("messages" in err.get("loc", []) for err in resp["detail"]), "未检测到 messages 字段缺失的报错"
    assert any("Field required" in err.get("msg", "") for err in resp["detail"]), "未检测到 'Field required' 错误提示"


def test_malformed_messages_format():
    """messages 为非列表，应报错而非崩溃"""
    data = {
        "stream": False,
        "messages": "我是一个非法的消息结构",
        "max_tokens": 10,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert "detail" in resp, "非法结构未被识别"
    assert any("messages" in err.get("loc", []) for err in resp["detail"]), "未检测到 messages 字段结构错误"
    assert any(
        "Input should be a valid list" in err.get("msg", "") for err in resp["detail"]
    ), "未检测到 'Input should be a valid list' 错误提示"


def test_extremely_large_max_tokens():
    """设置极大 max_tokens，观察模型内存/容错行为"""
    data = {
        "stream": False,
        "messages": [{"role": "user", "content": "1+1=?"}],
        "max_tokens": 10000000,
    }
    payload = build_request_payload(TEMPLATE, data)
    try:
        resp = send_request(URL, payload).json()
        assert "error" in resp or resp["usage"]["completion_tokens"] < 10000000
    except Exception:
        pytest.fail("设置极大 max_tokens 时服务崩溃")


def test_null_metadata():
    """metadata = null"""
    data = {
        "stream": False,
        "messages": [{"role": "user", "content": "介绍下你自己"}],
        "max_tokens": 10,
        "metadata": None,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert "error" not in resp, "metadata=null 应被容忍而不是报错"


def test_top_p_exceed_1():
    """top_p 超过1，违反规定，服务应报错"""
    data = {
        "stream": False,
        "messages": [{"role": "user", "content": "非洲的首都是？"}],
        "top_p": 1.5,
        "max_tokens": 10,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert resp.get("detail").get("object") == "error", "top_p > 1 应触发校验异常"
    assert "top_p value can only be defined" in resp.get("detail").get("message", ""), "未返回预期的 top_p 错误信息"


def test_mixed_valid_invalid_fields():
    """混合合法字段与非法字段，看是否污染整个请求"""
    data = {
        "stream": False,
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 10,
        "invalid_field": "this_should_be_ignored_or_warned",
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert "error" not in resp, "非法字段不应导致请求失败"


def test_stop_seq_exceed_num():
    """stop 字段包含超过 FD_MAX_STOP_SEQS_NUM 个元素，服务应报错"""
    data = {
        "stream": False,
        "messages": [{"role": "user", "content": "非洲的首都是？"}],
        "top_p": 0,
        "stop": ["11", "22", "33", "44", "55", "66", "77"],
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert resp.get("detail").get("object") == "error", "stop 超出个数应触发异常"
    assert "exceeds the limit max_stop_seqs_num" in resp.get("detail").get("message", ""), "未返回预期的报错信息"


def test_stop_seq_exceed_length():
    """stop 中包含长度超过 FD_STOP_SEQS_MAX_LEN 的元素，服务应报错"""
    data = {
        "stream": False,
        "messages": [{"role": "user", "content": "非洲的首都是？"}],
        "top_p": 0,
        "stop": ["11", "今天天气比明天好多了，请问你会出门还是和我一起玩"],
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert resp.get("detail").get("object") == "error", "stop 超出长度应触发异常"
    assert "exceeds the limit stop_seqs_max_len" in resp.get("detail").get("message", ""), "未返回预期的报错信息"
