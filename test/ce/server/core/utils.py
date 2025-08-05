#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import json
import math

import requests
from core import TEMPLATES, base_logger


def build_request_payload(template_name: str, case_data: dict) -> dict:
    """
    基于模板构造请求 payload，按优先级依次合并：
    template < payload 参数 < case_data，后者会覆盖前者的同名字段。

    :param template_name: 模板变量名，例如 "TOKEN_LOGPROB"
    :return: 构造后的完整请求 payload dict
    """
    template = TEMPLATES[template_name]
    print(template)
    final_payload = template.copy()
    final_payload.update(case_data)

    return final_payload


def send_request(url, payload, timeout=600, stream=False):
    """
    向指定URL发送POST请求，并返回响应结果。

    Args:
        url (str): 请求的目标URL。
        payload (dict): 请求的负载数据，应该是一个字典类型。
        timeout (int, optional): 请求的超时时间，默认为600秒。
        stream (bool, optional): 是否以流的方式下载响应内容，默认为False。

    Returns:
        response: 请求的响应结果，如果请求失败则返回None。
    """
    headers = {
        "Content-Type": "application/json",
    }
    base_logger.info("🔄 正在请求模型接口...")

    try:
        res = requests.post(url, headers=headers, json=payload, stream=stream, timeout=timeout)
        base_logger.info("🟢 接收响应中...\n")
        return res
    except requests.exceptions.Timeout:
        base_logger.error(f"❌ 请求超时（超过 {timeout} 秒）")
        return None
    except requests.exceptions.RequestException as e:
        base_logger.error(f"❌ 请求失败：{e}")
        return None


def get_stream_chunks(response):
    """解析流式返回，生成 chunk List[dict]"""
    chunks = []

    if response.status_code == 200:
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data: "):
                    line = line[len("data: ") :]

                if line.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except Exception as e:
                    base_logger.error(f"解析失败: {e}, 行内容: {line}")
    else:
        base_logger.error(f"请求失败，状态码: {response.status_code}")
        base_logger.error("返回内容：", response.text)

    return chunks


def get_token_list(response):
    """解析 response 中的 token 文本列表"""
    token_list = []

    try:
        content_logprobs = response["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        base_logger.error(f"解析失败：{e}")
        return []

    for token_info in content_logprobs:
        token = token_info.get("token")
        if token is not None:
            token_list.append(token)

    base_logger.info(f"Token List:{token_list}")
    return token_list


def get_logprobs_list(response):
    """解析 response 中的 token 文本列表"""
    logprobs_list = []

    try:
        content_logprobs = response["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        base_logger.error(f"解析失败：{e}")
        return []

    for token_info in content_logprobs:
        token = token_info.get("logprob")
        if token is not None:
            logprobs_list.append(token)

    base_logger.info(f"Logprobs List:{logprobs_list}")
    return logprobs_list


def get_probs_list(response):
    """解析 response 中的 token 文本列表"""
    probs_list = []

    try:
        content_logprobs = response["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        base_logger.error(f"解析失败：{e}")
        return []

    for token_info in content_logprobs:
        token = token_info.get("logprob")
        if token is not None:
            probs_list.append(math.exp(token))

    base_logger.info(f"probs List:{probs_list}")
    return probs_list
