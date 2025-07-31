#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

from core import TEMPLATE, URL, build_request_payload, send_request


def demo():
    data = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
    }
    payload = build_request_payload(TEMPLATE, data)
    req = send_request(URL, payload)
    print(req.json())
    req = req.json()

    assert req["usage"]["prompt_tokens"] == 22
    assert req["usage"]["total_tokens"] == 25
    assert req["usage"]["completion_tokens"] == 3


def test_demo():
    data = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
    }
    payload = build_request_payload(TEMPLATE, data)
    req = send_request(URL, payload)
    print(req.json())
    req = req.json()

    assert req["usage"]["prompt_tokens"] == 22
    assert req["usage"]["total_tokens"] == 25
    assert req["usage"]["completion_tokens"] == 5


if __name__ == "__main__":
    demo()
