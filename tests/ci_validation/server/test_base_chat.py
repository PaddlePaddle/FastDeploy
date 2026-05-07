#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
some basic check for fd web api
"""

import json

import requests
from core import TEMPLATE, URL, build_request_payload, get_token_list, send_request


def test_stream_response():
    data = {
        "stream": True,
        "messages": [
            {"role": "system", "content": "你是一个知识渊博的 AI 助手"},
            {"role": "user", "content": "讲讲爱因斯坦的相对论"},
        ],
        "max_tokens": 10,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload, stream=True)

    output = ""
    for line in resp.iter_lines(decode_unicode=True):
        if line.strip() == "" or not line.startswith("data: "):
            continue
        line = line[len("data: ") :]
        if line.strip() == "[DONE]":
            break
        chunk = json.loads(line)
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        output += delta.get("content", "")

    print("Stream输出:", output)
    assert "相对论" in output or len(output) > 0


def test_system_prompt_effect():
    data = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "请用一句话回答"},
            {"role": "user", "content": "什么是人工智能？"},
        ],
        "max_tokens": 30,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    content = resp["choices"][0]["message"]["content"]
    print("内容输出:", content)
    assert len(content) < 50


def test_logprobs_enabled():
    data = {
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5,
        "messages": [{"role": "user", "content": "非洲的首都是？"}],
        "max_tokens": 3,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    logprob_data = resp["choices"][0].get("logprobs")
    print("LogProbs:", logprob_data)
    assert logprob_data is not None
    content_logprobs = logprob_data.get("content", [])
    assert isinstance(content_logprobs, list)
    assert all("token" in item for item in content_logprobs)


def test_stop_sequence():
    data = {
        "stream": False,
        "stop": ["。"],
        "messages": [
            {
                "role": "user",
                "content": "你要严格按照我接下来的话输出，输出冒号后面的内容，请输出：这是第一段。果冻这是第二段啦啦啦啦啦。",
            },
        ],
        "max_tokens": 20,
        "top_p": 0,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    content = resp["choices"][0]["message"]["content"]
    token_list = get_token_list(resp)
    print("截断输出:", content)
    assert "第二段" not in content
    assert "第二段" not in token_list
    assert "。" in token_list, "没有找到。符号"


def test_stop_sequence1():
    """
    不加stop看看是否有影响
    """
    data = {
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": "你要严格按照我接下来的话输出，输出冒号后面的内容，请输出：这是第一段。果冻这是第二段啦啦啦啦啦。",
            },
        ],
        "max_tokens": 20,
        "top_p": 0,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    content = resp["choices"][0]["message"]["content"]
    print("截断输出:", content)
    assert "第二段" in content


def test_stop_sequence2():
    """
    stop token长度测试
    """
    data = {
        "stream": False,
        "stop": ["这是第二段啦啦"],
        "messages": [
            {
                "role": "user",
                "content": "你要严格按照我接下来的话输出，输出冒号后面的内容，请输出：这是第一段。果冻这是第二段啦啦啦啦啦。",
            },
        ],
        "max_tokens": 50,
        "top_p": 0,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    content = resp["choices"][0]["message"]["content"]
    # token_list = get_token_list(resp)
    print("截断输出:", content)
    assert "啦啦啦" not in content


# def test_stop_sequence3():
#     """
#     stop token 数量测试
#     """
#     data = {
#         "stream": False,
#         "stop": ["。", "果冻", "果", "冻", "第二", "二"],
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "你要严格按照我接下来的话输出，输出冒号后面的内容，请输出：这是第一段。果冻这是第二段啦啦啦啦啦。",
#             },
#         ],
#         "max_tokens": 50,
#         "top_p": 0,
#     }
#     payload = build_request_payload(TEMPLATE, data)
#     resp = send_request(URL, payload).json()
#     content = resp["choices"][0]["message"]["content"]
#     print("截断输出:", content)
#     assert "啦啦啦" not in content


def test_sampling_parameters():
    data = {
        "stream": False,
        "temperature": 0,
        "top_p": 0,
        "messages": [
            {"role": "user", "content": "1+1=？,直接回答答案"},
        ],
        "max_tokens": 50,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    answer = resp["choices"][0]["message"]["content"]
    print("Sampling输出:", answer)
    assert any(ans in answer for ans in ["2", "二"])


def test_multi_turn_conversation():
    data = {
        "stream": False,
        "messages": [
            {"role": "user", "content": "牛顿是谁？"},
            {"role": "assistant", "content": "牛顿是一位物理学家。"},
            {"role": "user", "content": "他提出了什么理论？"},
        ],
        "max_tokens": 30,
    }
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload).json()
    content = resp["choices"][0]["message"]["content"]
    print("多轮记忆:", content)
    assert "三大运动定律" in content or "万有引力" in content


def test_bad_words_filtering():
    banned_tokens = ["香蕉"]

    data = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "你是一个助手，回答简洁清楚"},
            {"role": "user", "content": "请输出冒号后面的字: 我爱吃果冻，和苹果，香蕉，和荔枝"},
        ],
        "top_p": 0,
        "max_tokens": 69,
        "bad_words": banned_tokens,
    }

    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload).json()
    content = response["choices"][0]["message"]["content"]
    print("生成内容:", content)
    token_list = get_token_list(response)

    for word in banned_tokens:
        assert word not in token_list, f"bad_word '{word}' 不应出现在生成结果中"

    print("test_bad_words_filtering 正例验证通过")


def test_bad_words_filtering1():
    banned_tokens = ["和", "呀"]

    data = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "你是一个助手，回答简洁清楚"},
            {"role": "user", "content": "请输出冒号后面的字: 我爱吃果冻，和苹果，香蕉，和荔枝"},
        ],
        "top_p": 0,
        "max_tokens": 69,
        "bad_words": banned_tokens,
    }

    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload).json()

    content = response["choices"][0]["message"]["content"]
    print("生成内容:", content)

    for word in banned_tokens:
        assert word not in content, f"bad_word '{word}' 不应出现在生成结果中"

    print("test_bad_words_filtering1 通过：生成结果未包含被禁词")

    # 正例验证
    word = "呀"
    data = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "你是一个助手，回答简洁清楚"},
            {"role": "user", "content": "请输出冒号后面的字，一模一样: 我爱吃果冻，苹果，香蕉，和荔枝呀呀呀"},
        ],
        "top_p": 0,
        "max_tokens": 69,
    }

    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload).json()

    content = response["choices"][0]["message"]["content"]
    print("生成内容:", content)
    token_list = get_token_list(response)
    assert word in token_list, f"'{word}' 应出现在生成结果中"

    print("test_bad_words_filtering1 正例验证通过")


def test_n_parameters():
    """
    n参数测试 n=2
    """
    # 1. 构建请求
    data = {
        "stream": False,
        "messages": [
            {"role": "user", "content": "牛顿是谁？"},
        ],
        "max_tokens": 30,
        "n": 2,
    }
    payload = build_request_payload(TEMPLATE, data)

    # 2. 发送请求
    resp = send_request(URL, payload).json()

    # 3. 检查返回choices数量
    choices = resp.get("choices", [])
    assert len(choices) == 2, f"n参数为2，输出必须是2条数据，但实际返回 {len(choices)} 条"

    # 4. 检查每条内容开头是否符合预期
    expected_start = "牛顿是英国著名的物理学家"
    for i, choice in enumerate(choices):
        content = choice["message"]["content"]
        print(f"Choice {i} 内容:", content)
        assert content.startswith(expected_start), f"第{i}条输出内容开头不匹配"

    print("test_n_parameters 验证通过")


def test_n_parameters1():
    """
    n参数测试 n=3
    """
    # 1. 构建请求
    data = {
        "stream": False,
        "messages": [
            {"role": "user", "content": "牛顿是谁？"},
        ],
        "max_tokens": 30,
        "n": 3,
    }
    payload = build_request_payload(TEMPLATE, data)

    # 2. 发送请求
    resp = send_request(URL, payload).json()

    # 3. 检查返回choices数量
    choices = resp.get("choices", [])
    assert len(choices) == 3, f"n参数为3，输出必须是3条数据，但实际返回 {len(choices)} 条"

    # 4. 检查每条内容开头是否符合预期
    expected_start = "牛顿是英国著名的物理学家"
    for i, choice in enumerate(choices):
        content = choice["message"]["content"]
        print(f"Choice {i} 内容:", content)
        assert content.startswith(expected_start), f"第{i}条输出内容开头不匹配"

    print("test_n_parameters 验证通过")


def test_n_parameters2():
    """
    n参数测试 n=6
    """
    # 1. 构建请求
    data = {
        "stream": False,
        "messages": [
            {"role": "user", "content": "牛顿是谁？"},
        ],
        "max_tokens": 30,
        "n": 6,
    }
    payload = build_request_payload(TEMPLATE, data)

    # 2. 发送请求
    resp = send_request(URL, payload).json()

    # 3. 检查返回choices数量
    choices = resp.get("choices", [])
    assert len(choices) == 6, f"n参数为6，输出必须是6条数据，但实际返回 {len(choices)} 条"

    # 4. 检查每条内容开头是否符合预期
    expected_start = "牛顿是英国著名的物理学家"
    for i, choice in enumerate(choices):
        content = choice["message"]["content"]
        print(f"Choice {i} 内容:", content)
        assert content.startswith(expected_start), f"第{i}条输出内容开头不匹配"

    print("test_n_parameters 验证通过")


def test_n_parameters_stream():
    """
    n参数测试（流式输出 n=3）
    """
    data = {
        "stream": True,
        "messages": [
            {"role": "user", "content": "牛顿是谁？"},
        ],
        "max_tokens": 30,
        "n": 3,
    }
    payload = build_request_payload(TEMPLATE, data)

    with requests.post(URL, json=payload, stream=True) as resp:
        assert resp.status_code == 200, f"请求失败，状态码 {resp.status_code}"

        # 初始化3个缓存
        partial_contents = ["", "", ""]

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: ") :]
            if data_str.strip() == "[DONE]":
                break

            try:
                data_json = json.loads(data_str)
            except Exception as e:
                print("解析异常：", e, line)
                continue

            choices = data_json.get("choices", [])
            for choice in choices:
                idx = choice.get("index", 0)
                delta = choice.get("delta", {}).get("content", "")
                if idx < len(partial_contents):
                    partial_contents[idx] += delta

        # 检查流式聚合结果
        assert len(partial_contents) == 3, "应产生3个流式输出"
        expected_start = "牛顿是英国著名的物理学家"
        for i, content in enumerate(partial_contents):
            print(f"Choice {i} 最终内容:", content)
            assert content.startswith(expected_start), f"第{i}条输出开头不匹配"

    print("✅ test_n_parameters_stream 验证通过")
