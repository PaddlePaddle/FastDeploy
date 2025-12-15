#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

from core import TEMPLATE, URL, build_request_payload, send_request


def test_incremental_image_reasoning_consistency():
    """
    多模态增量推理一致性检查：
    第一次请求携带图片和文本，并打开 return_token_ids。
    第二次请求拼入 prompt_token_ids，要求输出一致，否则校验失败。
    """

    # 第一次请求数据
    data_1st = {
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
                        }
                    },
                    {"type": "text", "text": "图中的文物属于哪个年代？"}
                ],
            }
        ],
        "return_token_ids": True,
    }

    print("==== 开始第一次请求 ====")
    payload_1st = build_request_payload(TEMPLATE, data_1st)
    res1 = send_request(URL, payload_1st)
    print(f"第一次请求返回码: {res1.status_code}")

    if res1.status_code != 200:
        raise AssertionError(f"首次请求失败: {res1.text}")

    result1 = res1.json()
    choice1 = result1["choices"][0]
    msg1 = choice1.get("message", {})

    content1 = msg1.get("content", "")
    reasoning1 = msg1.get("reasoning_content", "")
    tokens1 = msg1.get("prompt_token_ids", [])
    print(f"第一次请求结果: {tokens1}")

    if not tokens1:
        raise AssertionError("首次请求未返回 prompt_token_ids！")

    print(f"第一次 content = {content1}")
    print(f"第一次 reasoning_content = {reasoning1}")
    print(f"第一次 prompt_token_ids 长度 = {len(tokens1)}")

    # 构造第二次请求
    print("\n==== 开始第二次请求（携带 prompt_token_ids）====")
    data_2nd = {
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
                        }
                    },
                ],

            }
        ],
        "prompt_token_ids": tokens1,
    }

    payload_2nd = build_request_payload(TEMPLATE, data_2nd)
    res2 = send_request(URL, payload_2nd)
    print(f"第二次请求返回码: {res2.status_code}")

    if res2.status_code != 200:
        raise AssertionError(f"二次请求失败: {res2.text}")

    result2 = res2.json()
    choice2 = result2["choices"][0]
    msg2 = choice2.get("message", {})

    content2 = msg2.get("content", "")
    reasoning2 = msg2.get("reasoning_content", "")

    print(f"第二次 content = {content2}")
    print(f"第二次 reasoning_content = {reasoning2}")

    # 一致性校验
    assert content1 == content2, "content 不一致，增量推理一致性校验失败!"
    assert reasoning1 == reasoning2, "reasoning_content 不一致，增量推理一致性校验失败!"

    print("\n 一致性校验通过！增量推理行为正常。")


if __name__ == "__main__":
    test_incremental_image_reasoning_consistency()
