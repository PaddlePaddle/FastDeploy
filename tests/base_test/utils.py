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


# def send_request_stream(url: str, payload: dict):
#     """
#     统一拉取 SSE 并返回
#     1. 完整 chunk 列表
#     2. 最终拼接文本
#     3. 末包 usage
#     """
#     try:
#         resp = send_request(url, payload)
#         resp.raise_for_status()
#     except requests.exceptions.HTTPError as e:
#         try:
#             # 从响应中尝试获取 JSON 错误信息
#             error_content = e.response.text
#             error_data = json.loads(error_content)
#             error_message = json.dumps(error_data)
#         except ValueError:
#             # 如果无法解析 JSON，返回简化的错误信息
#             error_message = f"HTTP error occurred: {e.response.status_code} {e.response.reason}"
#         print(error_message)
#         return [], "", None, error_message
#     except Exception as e:
#         # 处理其他异常并返回错误信息
#         error_message = f"Request failed: {e}"
#         print(error_message)
#         return [], "", None, error_message

#     chunks, texts, usage = [], {}, None
#     for line in resp.iter_lines(decode_unicode=True):
#         line = line.strip()
#         if not line or not line.startswith("data: "):
#             continue

#         seg = line[len("data: "):].strip()
#         if seg == "[DONE]":
#             break

#         chunk = json.loads(seg)
#         chunks.append(chunk)

#         for choice in chunk.get("choices", []):
#             idx = choice.get("index", 0)
#             delta = choice.get("delta", {})

#             # 累加文本
#             if "content" in delta:
#                 texts[idx] = texts.get(idx, "") + delta["content"]

#             # 末包 usage（通常只在最后一个 chunk 有）
#             if choice.get("finish_reason"):
#                 usage = chunk.get("usage")

#     # ===== 关键：单条时返回字符串 =====
#     if not texts:
#         text_or_texts = ""
#     elif len(texts) == 1:
#         text_or_texts = next(iter(texts.values()))
#     else:
#         text_or_texts = texts

#     return chunks, text_or_texts, usage


# # def send_request_stream(url: str, payload: dict):
#     """
#     统一拉取 SSE 并返回 3 个值（确保解包 (_, text, _) 不报错）
#     1. chunks (list)
#     2. text_result (str/dict)
#     3. usage (dict/None)
#     """
#     chunks, texts, usage = [], {}, None

#     try:
#         # 这里的 send_request 必须支持 stream=True
#         resp = send_request(url, payload)
#         resp.raise_for_status()

#         for line in resp.iter_lines(decode_unicode=True):
#             line = line.strip()
#             # 兼容性处理：匹配 data: 或 data: {
#             if not line or not line.startswith("data:"):
#                 continue

#             # 移除 "data:" 前缀并处理 [DONE] 标记
#             seg = line[len("data:"):].strip()
#             if seg == "[DONE]":
#                 break

#             try:
#                 chunk = json.loads(seg)
#                 chunks.append(chunk)

#                 # 1. 提取流式文本内容
#                 for choice in chunk.get("choices", []):
#                     idx = choice.get("index", 0)
#                     delta = choice.get("delta", {})
#                     if "content" in delta:
#                         texts[idx] = texts.get(idx, "") + delta["content"]

#                 # 2. 提取 Usage (通常在最后一个数据包)
#                 if "usage" in chunk and chunk["usage"]:
#                     usage = chunk["usage"]

#             except json.JSONDecodeError:
#                 continue

#     except Exception as e:
#         # 核心修复点：即使出错也只返回 3 个值，防止 unpack 报错
#         print(f"\n[ERROR] Streaming Request Failed: {e}")
#         return [], "", None

#     # 3. 汇总文本结果
#     if not texts:
#         text_result = ""
#     elif len(texts) == 1:
#         text_result = next(iter(texts.values()))
#     else:
#         text_result = texts

#     return chunks, text_result, usage


# def send_request_stream(url: str, payload: dict):
#     chunks, texts, usage = [], {}, None

#     try:
#         resp = send_request(url, payload)
#         resp.raise_for_status()

#         for line in resp.iter_lines(decode_unicode=True):
#             line = line.strip()
#             if not line or not line.startswith("data:"):
#                 continue

#             seg = line[len("data:"):].strip()
#             if seg == "[DONE]":
#                 break

#             try:
#                 chunk = json.loads(seg)
#                 chunks.append(chunk)

#                 for choice in chunk.get("choices", []):
#                     idx = choice.get("index", 0)

#                     # --- 核心修复：兼容 Completion 模式 (choice["text"]) 和 Chat 模式 (choice["delta"]["content"]) ---
#                     content = ""
#                     if "text" in choice:
#                         content = choice["text"]  # 你的模型走这里
#                     elif "delta" in choice and "content" in choice["delta"]:
#                         content = choice["delta"]["content"]

#                     if content:
#                         texts[idx] = texts.get(idx, "") + content

#                 if "usage" in chunk and chunk["usage"]:
#                     usage = chunk["usage"]

#             except json.JSONDecodeError:
#                 continue
#     except Exception as e:
#         print(f"Error: {e}")
#         return [], "", None

#     # 拼接文本逻辑保持不变
#     if not texts:
#         text_result = ""
#     elif len(texts) == 1:
#         text_result = next(iter(texts.values()))
#     else:
#         text_result = texts

#     return chunks, text_result, usage


# def get_content_from_choice(choice):
#     """
#     核心兼容逻辑：从 choice 结构中提取文本内容
#     支持：Chat 模式 (message), Completion 模式 (text), 流式模式 (delta)
#     """
#     if "message" in choice: # 非流式 Chat
#         return choice["message"].get("content", "")
#     if "text" in choice:    # Completion 模式
#         return choice["text"]
#     if "delta" in choice:   # 流式 Chat
#         return choice["delta"].get("content", "")
#     return ""

# def send_request_stream(url: str, payload: dict):
#     chunks, texts, usage = [], {}, None
#     try:
#         # 强制底层请求使用 stream=True
#         resp = requests.post(url, json=payload, stream=True, timeout=60)
#         resp.raise_for_status()

#         for line in resp.iter_lines(decode_unicode=True):
#             line = line.strip()
#             if not line or not line.startswith("data:"):
#                 continue

#             seg = line[len("data:"):].strip()
#             if seg == "[DONE]":
#                 break

#             try:
#                 chunk = json.loads(seg)
#                 chunks.append(chunk)

#                 # 提取 Usage (通常在末包根节点)
#                 if "usage" in chunk and chunk["usage"]:
#                     usage = chunk["usage"]

#                 # 提取内容
#                 for choice in chunk.get("choices", []):
#                     idx = choice.get("index", 0)
#                     content = get_content_from_choice(choice)
#                     if content:
#                         texts[idx] = texts.get(idx, "") + content
#             except json.JSONDecodeError:
#                 continue
#     except Exception as e:
#         print(f"\n[Stream Error] {e}")
#         return [], "", None

#     # 合并结果
#     if not texts:
#         text_result = ""
#     elif len(texts) == 1:
#         text_result = next(iter(texts.values()))
#     else:
#         text_result = texts

#     return chunks, text_result, usage


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
