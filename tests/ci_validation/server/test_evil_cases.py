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

    assert "error" in resp, "返回中未包含 error 错误信息字段"
    error = resp["error"]
    assert "Field required" in error.get("message", ""), "未检测到 messages 字段缺失的报错"
    assert error.get("code") == "missing_required_parameter", "code 字段不正确"

    # assert "detail" in resp, "返回中未包含 detail 错误信息字段"
    # assert any("messages" in err.get("loc", []) for err in resp["detail"]), "未检测到 messages 字段缺失的报错"
    # assert any("Field required" in err.get("msg", "") for err in resp["detail"]), "未检测到 'Field required' 错误提示"


def test_malformed_messages_format():
    """messages 为非列表，应报错而非崩溃"""
    data = {
        "stream": False,
        "messages": "我是一个非法的消息结构",
        "max_tokens": 10,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert "error" in resp, "非法结构未被识别"
    err = resp["error"]
    assert err.get("param") == "list[any]", f"param 字段错误: {err.get('param')}"
    assert err.get("message") == "Input should be a valid list", "错误提示不符合预期"
    # assert "detail" in resp, "非法结构未被识别"
    # assert any("messages" in err.get("loc", []) for err in resp["detail"]), "未检测到 messages 字段结构错误"
    # assert any(
    #     "Input should be a valid list" in err.get("msg", "") for err in resp["detail"]
    # ), "未检测到 'Input should be a valid list' 错误提示"


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
    assert "error" in resp, "未返回 error 字段"

    err = resp["error"]
    assert err.get("param") == "top_p", f"param 字段错误: {err.get('param')}"
    assert err.get("message") == "Input should be less than or equal to 1", "错误提示不符合预期"
    # assert resp.get("detail").get("object") == "error", "top_p > 1 应触发校验异常"
    # assert "top_p value can only be defined" in resp.get("detail").get("message", ""), "未返回预期的 top_p 错误信息"


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
    assert resp.get("error").get("type") == "invalid_request_error", "stop 超出个数应触发异常"
    assert "exceeds the limit max_stop_seqs_num" in resp.get("error").get("message", ""), "未返回预期的报错信息"


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
    assert resp.get("error").get("type") == "invalid_request_error", "stop 超出长度应触发异常"
    assert "exceeds the limit stop_seqs_max_len" in resp.get("error").get("message", ""), "未返回预期的报错信息"


def test_multilingual_input():
    """测试多语言混合输入是否能够被正确处理"""
    data = {
        "messages": [
            {
                "role": "user",
                "content": "这是一个包含多种语言的输入：Hello, 世界！Bonjour, le monde! Hola, el mundo! こんにちは、世界！",
            }
        ],
        "stream": False,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()

    # 验证响应是否包含有效的回复
    assert "choices" in resp, "未收到有效的回复"
    assert len(resp["choices"]) > 0, "回复为空"
    assert "message" in resp["choices"][0], "回复中未包含消息内容"
    assert "content" in resp["choices"][0]["message"], "回复中未包含内容字段"
    # 验证模型是否能够正确处理多语言输入
    response_content = resp["choices"][0]["message"]["content"]
    assert response_content.strip() != "", "模型未生成任何内容"
    print("多语言混合输入测试通过！")


def test_too_long_input():
    """测试超长输入是否被正确处理"""
    data = {"messages": [{"role": "user", "content": "a，" * 200000}], "stream": False}  # 超过最大输入长度
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    # assert resp["detail"].get("object") == "error", "超长输入未被识别为错误"
    assert "Input text is too long" in resp["error"].get("message", ""), "未检测到最大长度限制错误"


def test_empty_input():
    """测试空输入是否被正确处理"""
    data = {"messages": [{"role": "user", "content": ""}], "stream": False}  # 空输入
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert "error" not in resp.get("object"), "空输入被识别为错误"
    assert len(resp["choices"][0]["message"]["content"]) > 0, "内容为空时，回复为空"


def test_prompt_only_spaces():
    """messages content 为纯空格字符串，服务正常返回"""
    data = {
        "messages": [
            {
                "role": "user",
                "content": "    ",  # 纯空格
            }
        ],
        "stream": False,
        "max_tokens": 10,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert resp.get("object") == "chat.completion", "应返回 chat.completion 对象"
    response_content = resp["choices"][0]["message"]["content"]
    assert len(response_content) > 0, "messages content为空，未正常生成回复"


def test_illegal_characters():
    """测试非法字符输入是否被正确处理"""
    data = {"messages": [{"role": "user", "content": "非洲的首都是？：\x00\x01\x02"}], "stream": False}  # 非法字符
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert len(resp["choices"][0]["message"]["content"]) > 0, "非法字符输入影响模型回复"


def test_sql_injection():
    """测试 SQL 注入攻击是否被正确处理"""
    data = {
        "messages": [
            {"role": "user", "content": "SELECT * FROM users WHERE username = 'admin' OR '1'='1';"}  # SQL 注入攻击
        ],
        "stream": False,
        "max_tokens": 50,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert "message" in resp["choices"][0], "回复中未包含消息内容"
    assert "content" in resp["choices"][0]["message"], "回复中未包含内容字段"
    response_content = resp["choices"][0]["message"]["content"]
    assert len(response_content) > 0, "SQL 注入攻击影响模型回复"


def test_xss_attack():
    """测试 XSS 攻击是否被正确处理"""
    data = {
        "messages": [{"role": "user", "content": "<script>alert('XSS');</script>"}],  # XSS 攻击
        "stream": False,
        "max_tokens": 50,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert "message" in resp["choices"][0], "回复中未包含消息内容"
    assert "content" in resp["choices"][0]["message"], "回复中未包含内容字段"
    response_content = resp["choices"][0]["message"]["content"]
    assert len(response_content) > 0, "XSS 攻击未被正确处理"


def test_stop_empty_string():
    """测试 stop 参数为空字符串时的行为"""
    data = {
        "stream": False,
        "messages": [{"role": "user", "content": "非洲的首都是？"}],
        "max_tokens": 10,
        "stop": "",  # 空字符串
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert resp.get("object") == "chat.completion", "应返回 chat.completion 对象"
    assert len(resp.get("choices", [])[0].get("message", {}).get("content", "")) > 0, "应生成有效文本"


def test_stop_multiple_strings():
    """测试 stop 参数为多个字符串时的行为"""
    data = {
        "stream": False,
        "messages": [{"role": "user", "content": "非洲的首都是？"}],
        "max_tokens": 50,
        "stop": ["。", "！", "？"],  # 多个停止条件
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert resp.get("object") == "chat.completion", "应返回 chat.completion 对象"
    generated_text = resp.get("choices")[0].get("message", {}).get("content", "")
    assert any(stop in generated_text for stop in data["stop"]), "生成文本应包含 stop 序列之一"


def test_stop_with_special_characters():
    """测试 stop 参数为包含特殊字符的字符串时的行为"""
    data = {
        "stream": False,
        "messages": [{"role": "user", "content": "非洲的首都是？"}],
        "max_tokens": 50,
        "stop": "!@#$%^&*()",  # 包含特殊字符
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert resp.get("object") == "chat.completion", "应返回 chat.completion 对象"
    generated_text = resp.get("choices")[0].get("message", {}).get("content", "")
    assert any(char in generated_text for char in data["stop"]), "生成文本应包含 stop 序列中的特殊字符之一"


def test_stop_with_newlines():
    """测试 stop 参数为包含换行符的字符串时的行为"""
    data = {
        "stream": False,
        "messages": [{"role": "user", "content": "非洲的首都是？"}],
        "max_tokens": 50,
        "stop": "\n\n",  # 包含换行符
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert resp.get("object") == "chat.completion", "应返回 chat.completion 对象"
    generated_text = resp.get("choices")[0].get("message", {}).get("content", "")
    assert data["stop"] in generated_text, "生成文本应包含 stop 序列"


def test_model_empty():
    """model 参数为空，不影响服务"""
    data = {
        "messages": [
            {
                "role": "user",
                "content": "非洲的首都是？",
            }
        ],
        "stream": False,
        "max_tokens": 10,
        "model": "",  # 空模型
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert resp.get("object") == "chat.completion", "应返回 chat.completion 对象"
    response_content = resp["choices"][0]["message"]["content"]
    assert len(response_content) > 0, "模型名为空，未正常生成回复"


def test_model_invalid():
    """model 参数为不存在的模型，不影响服务"""
    data = {
        "messages": [
            {
                "role": "user",
                "content": "非洲的首都是？",
            }
        ],
        "stream": False,
        "max_tokens": 10,
        "model": "non-existent-model",  # 不存在的模型
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert resp.get("object") == "chat.completion", "不存在的 model 应触发校验异常"
    # assert "non-existent-model" in resp.get("model"), "未返回预期的 model 信息"
    assert len(resp.get("choices")[0].get("message").get("content")) > 0, "模型名为不存在的 model，未正常生成回复"


def test_model_with_special_characters():
    """model 参数为非法格式（例如包含特殊字符），不影响服务"""
    data = {
        "messages": [
            {
                "role": "user",
                "content": "非洲的首都是？",
            }
        ],
        "stream": False,
        "max_tokens": 10,
        "model": "!@#",  # 包含特殊字符
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert resp.get("object") == "chat.completion", "不存在的 model 应触发校验异常"
    # assert "!@#" in resp.get("model"), "未返回预期的 model 信息"
    assert (
        len(resp.get("choices")[0].get("message").get("content")) > 0
    ), "模型名为model 参数为非法格式，未正常生成回复"


def test_max_tokens_negative():
    """max_tokens 为负数，服务应报错"""
    data = {
        "messages": [
            {
                "role": "user",
                "content": "非洲的首都是？",
            }
        ],
        "stream": False,
        "max_tokens": -10,  # 负数
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    # assert resp.get("detail").get("object") == "error", "max_tokens < 0 未触发校验异常"
    assert "max_tokens can be defined [1," in resp.get("error").get("message"), "未返回预期的 max_tokens 错误信息"


def test_max_tokens_min():
    """测试 max_tokens 达到异常值0 时的行为"""
    data = {
        "messages": [
            {
                "role": "user",
                "content": "非洲的首都是？",
            }
        ],
        "stream": False,
        "max_tokens": 0,  # 最小值
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    # assert resp.get("detail").get("object") == "error", "max_tokens未0时API未拦截住"
    assert "max_tokens can be defined [1," in resp.get("error").get(
        "message"
    ), "max_tokens未0时API未拦截住,未返回预期的 max_tokens 错误信息"


def test_max_tokens_non_integer():
    """max_tokens 为非整数，服务应报错"""
    data = {
        "messages": [
            {
                "role": "user",
                "content": "非洲的首都是？",
            }
        ],
        "stream": False,
        "max_tokens": 10.5,  # 非整数
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    assert (
        resp.get("error").get("message") == "Input should be a valid integer, got a number with a fractional part"
    ), "未返回预期的 max_tokens 为非整数的错误信息"


def test_error_structure():
    """校验返回 error 结构，而不关心具体内容"""
    data = {
        "stream": False,
        "messages": "我是一个非法的消息结构",
        "max_tokens": 10,
    }

    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()

    # 基本校验：必须有 error
    assert "error" in resp, "返回结果缺少 error 字段"
    err = resp["error"]
    assert isinstance(err, dict), "error 不是字典"

    # 校验结构字段存在
    for field in ["message", "type", "param", "code"]:
        assert field in err, f"error 缺少 {field} 字段"

    # message 不为 None
    assert err["message"] is not None, "error.message 不应为 null"
