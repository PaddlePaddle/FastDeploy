#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


import concurrent.futures
from collections import Counter

from core import TEMPLATE, URL, build_request_payload, send_request


def test_concurrency():
    """
    并发测试：
    同时发起 10 条请求，校验返回码是否为 5 个 200 和 5 个 429。
    --max-num-seqs 128 \
  --tensor-parallel-size 1 \
  --max-concurrency 5 \
    """

    data = {
        "stream": False,
        "messages": [
            {"role": "user", "content": "1+1=？ 直接回答"},
        ],
        "max_tokens": 1000,
        "temperature": 0.8,
        "top_p": 0,
    }

    def send_one_request(i):
        payload = build_request_payload(TEMPLATE, data)
        response = send_request(URL, payload)
        print(f"请求 {i} 返回码: {response.status_code}")
        return response.status_code

    # 并发执行 10 个请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(send_one_request, i) for i in range(10)]
        results = [f.result() for f in futures]

    # 统计返回码
    counter = Counter(results)
    count_200 = counter.get(200, 0)
    count_429 = counter.get(429, 0)

    print(f"统计结果: 200={count_200}, 429={count_429}, 全部结果={results}")

    # 校验必须是 5 个 200 和 5 个 429
    assert count_200 == 5, f"200 数量错误: {count_200}"
    assert count_429 == 5, f"429 数量错误: {count_429}"

    print("并发请求校验通过")
