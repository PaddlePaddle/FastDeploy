#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


import concurrent.futures
from collections import Counter

from core import TEMPLATE, URL, build_request_payload, send_request


def test_waiting_time():
    """
    并发测试：
    同时发起 1333 条请求。
    校验所有返回码统计，且数量总和必须等于 1333。
    额外校验：200 的数量必须小于 1333。
    --max-num-seqs 128 \
  --tensor-parallel-size 1 \
  --max-concurrency 5000 \
  --max-waiting-time 1 \
    """

    data = {
        "stream": False,
        "messages": [
            {"role": "user", "content": "1+1=？ 直接回答"},
        ],
        "max_tokens": 10000,
        "metadata": {
            "min_tokens": 99,
        },
        "temperature": 0.8,
        "top_p": 0,
    }

    def send_one_request(i):
        payload = build_request_payload(TEMPLATE, data)
        response = send_request(URL, payload)
        print(f"请求 {i} 返回码: {response.status_code}")
        return response.status_code

    # 并发执行 1333 个请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=1333) as executor:
        futures = [executor.submit(send_one_request, i) for i in range(1333)]
        results = [f.result() for f in futures]

    # 统计所有返回码
    counter = Counter(results)
    print("返回码统计结果:")
    for code, cnt in sorted(counter.items()):
        print(f"  {code}: {cnt}")

    # 校验返回总数
    total = sum(counter.values())
    assert total == 1333, f"返回数量不一致，总数={total}, 期望=1333"

    # 校验 200 数量必须小于 1333
    count_200 = counter.get(200, 0)
    assert count_200 < 1333, f"200 数量错误，应小于1333，实际={count_200}"
    assert count_200 >= 1024, f"200 数量错误，应大于等于1024，实际={count_200}"

    print("并发请求校验通过")
