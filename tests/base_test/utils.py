# -*- coding: utf-8 -*-
"""
FD 服务 Chat Completion 全参数测试
重点：针对服务能力，不检查模型是否聪明。
usage:
    python -m pytest test_maoyan.py
"""
import json
import os

import requests

# 从环境变量读取，默认回退值（可选）
URL_HOST = os.getenv("URL_HOST", "10.174.136.93")
URL_PORT = os.getenv("URL_PORT", "8180")

URL = f"http://{URL_HOST}:{URL_PORT}/v1/chat/completions"
print(f"FD URL: {URL}")


MODEL = "default"  # 内部 FD 固定模型名


# ===================== 工具封装 =====================


def send_request(url, payload, timeout=600, stream=False):
    """
    向指定URL发送POST请求，并返回响应结果。
    """
    headers = {
        "Content-Type": "application/json",
    }

    try:
        res = requests.post(url, headers=headers, json=payload, stream=stream, timeout=timeout)
        return res
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时（超过 {timeout} 秒）")
        # base_logger.error(f"❌ 请求超时（超过 {timeout} 秒）")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败：{e}")
        # base_logger.error(f"❌ 请求失败：{e}")
        return None


def get_content_from_choice(choice):
    """兼容 Chat(delta/message) 和 Completion(text) 模式内容提取"""
    if "message" in choice:
        return choice["message"].get("content", "")
    if "text" in choice:
        return choice["text"]
    if "delta" in choice:
        return choice["delta"].get("content", "")
    return ""


def send_request_stream(url: str, payload: dict):
    """
    统一流式请求函数
    返回: (chunks: list, text: str, usage: dict, error_msg: str)
    """
    chunks, texts, usage = [], {}, None

    try:
        # 1. 发起请求，注意必须有 stream=True
        resp = requests.post(url, json=payload, stream=True, timeout=60)

        # 2. 如果状态码不是 200，说明是业务错误（如参数校验非法）
        if resp.status_code != 200:
            error_body = resp.text
            print(f"\n[API Error] Status: {resp.status_code}, Body: {error_body}")
            return [], "", None, error_body

        # 3. 正常解析流式数据
        for line in resp.iter_lines(decode_unicode=True):
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue

            seg = line[len("data:") :].strip()
            if seg == "[DONE]":
                break

            try:
                chunk = json.loads(seg)
                chunks.append(chunk)

                # 提取 Usage
                if "usage" in chunk and chunk["usage"]:
                    usage = chunk["usage"]

                # 提取文本
                for choice in chunk.get("choices", []):
                    idx = choice.get("index", 0)
                    content = get_content_from_choice(choice)
                    if content:
                        texts[idx] = texts.get(idx, "") + content
            except json.JSONDecodeError:
                continue

    except Exception as e:
        # 网络层异常（如超时、断连）
        err_obj = {"error": {"message": f"Network/Runtime Error: {str(e)}", "type": "runtime_error"}}
        return [], "", None, json.dumps(err_obj)

    # 4. 汇总文本结果
    if not texts:
        text_result = ""
    elif len(texts) == 1:
        text_result = next(iter(texts.values()))
    else:
        text_result = texts

    return chunks, text_result, usage, None  # 成功时，error_msg 为 None
