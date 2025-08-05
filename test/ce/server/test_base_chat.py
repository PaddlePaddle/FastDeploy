#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
some basic check for fd web api
"""

import json

from core import (
    TEMPLATE,
    URL,
    build_request_payload,
    get_probs_list,
    get_token_list,
    send_request,
)


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


def test_repetition_early_stop():
    """
    用于验证 repetition early stop 功能是否生效：
    设置 window_size=6，threshold=0.93，输入内容设计成易重复，观察模型是否提前截断输出。
    threshold = 0.93
    window_size = 6 这个必须是启动模型的时候加上这个参数 负责不能用！！！！
    """

    data = {
        "stream": False,
        "messages": [
            {"role": "user", "content": "输出'我爱吃果冻' 10次"},
        ],
        "max_tokens": 10000,
        "temperature": 0.8,
        "top_p": 0,
    }

    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload).json()
    content = response["choices"][0]["message"]["content"]

    print("🧪 repetition early stop 输出内容:\n", content)
    probs_list = get_probs_list(response)

    threshold = 0.93
    window_size = 6

    assert len(probs_list) >= window_size, "列表长度不足 window_size"

    # 条件 1：末尾 6 个都 > threshold
    tail = probs_list[-window_size:]
    assert all(v > threshold for v in tail), "末尾 window_size 个数不全大于阈值"

    # 条件 2：前面不能有连续 >=6 个值 > threshold
    head = probs_list[:-window_size]
    count = 0
    for v in head:
        if v > threshold:
            count += 1
            assert count < window_size, f"在末尾之前出现了连续 {count} 个大于阈值的数"
        else:
            count = 0

    print("repetition early stop 功能验证通过")
