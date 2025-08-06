#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
ServeTest
"""


TOKEN_LOGPROB = {
    "model": "default",
    "temperature": 0,
    "top_p": 0,
    "seed": 33,
    "stream": True,
    "logprobs": True,
    "top_logprobs": 5,
    "max_tokens": 10000,
}


TEMPLATES = {
    "TOKEN_LOGPROB": TOKEN_LOGPROB,
    # "ANOTHER_TEMPLATE": ANOTHER_TEMPLATE
}
