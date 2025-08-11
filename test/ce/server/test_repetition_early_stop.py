#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


from core import TEMPLATE, URL, build_request_payload, get_probs_list, send_request


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
