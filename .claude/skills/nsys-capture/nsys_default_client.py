#!/usr/bin/env python3
"""
nsys 抓取默认请求脚本（文生文，流式输出）
用法：python3 nsys_default_client.py <host> <port>
"""

import sys

try:
    from openai import OpenAI
except ImportError:
    print("[nsys_client] 错误：未安装 openai 库，请执行 pip install openai", file=sys.stderr)
    sys.exit(1)

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = sys.argv[2] if len(sys.argv) > 2 else "8080"

client = OpenAI(
    base_url=f"http://{HOST}:{PORT}/v1",
    api_key="placeholder",
)

PROMPT = (
    "请详细介绍深度学习技术的发展历史与主要里程碑，"
    "包括早期神经网络、卷积神经网络、循环神经网络、注意力机制、Transformer 架构的演进，"
    "以及近年来大语言模型的兴起与影响。要求不少于 600 字，语言流畅，逻辑清晰。"
)

print(f"[nsys_client] 发送请求 → http://{HOST}:{PORT}")
print(f"[nsys_client] Prompt: {PROMPT[:60]}...")
print("-" * 60)

try:
    resp = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=2048,
        stream=True,
    )
    for chunk in resp:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()
    print("-" * 60)
    print("[nsys_client] 请求完成")

except KeyboardInterrupt:
    print("\n[nsys_client] 用户中断")
except Exception as e:
    # 流式连接中断（RemoteProtocolError）是正常现象（服务在 nvprof_stop 后退出）
    print(f"\n[nsys_client] 连接结束（{type(e).__name__}: {e}）")
