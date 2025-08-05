#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import os
import sys

from .logger import Logger

base_logger = Logger(loggername="FDSentry", save_level="channel", log_path="./fd_logs").get_logger()
base_logger.setLevel("INFO")

from .request_template import TEMPLATES
from .utils import (
    build_request_payload,
    get_logprobs_list,
    get_probs_list,
    get_stream_chunks,
    get_token_list,
    send_request,
)

__all__ = [
    "build_request_payload",
    "send_request",
    "TEMPLATES",
    "get_stream_chunks",
    "get_token_list",
    "get_logprobs_list",
    "get_probs_list",
]

# 检查环境变量是否存在
URL = os.environ.get("URL")
TEMPLATE = os.environ.get("TEMPLATE")

missing_vars = []
if not URL:
    missing_vars.append("URL")
if not TEMPLATE:
    missing_vars.append("TEMPLATE")

if not URL:
    msg = (
        f"❌ 缺少环境变量：{', '.join(missing_vars)}，请先设置，例如：\n"
        f"   export URL=http://localhost:8000/v1/chat/completions\n"
        f"   export TEMPLATE=TOKEN_LOGPROB"
    )
    base_logger.error(msg)
    sys.exit(33)  # 终止程序

if not TEMPLATE:
    base_logger.warning("⚠️ 未设置 TEMPLATE，请确保在用例中显式传入请求模板。")
