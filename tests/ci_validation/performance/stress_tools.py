#!/bin/env python3
# -*- coding: utf-8 -*-
# @author: DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import asyncio
import json
import os
import time
from collections import Counter
from statistics import mean, median

import aiohttp
from tqdm import tqdm

# ============ 配置 ============
API_URL = os.environ.get("URL", "http://localhost:8000/v1/chat/completions")
MAX_CONCURRENCY = 200  # 最大并发协程数
TOTAL_REQUESTS = 300000  # 总请求数
TIMEOUT = 1800  # 每个请求超时时间（秒）
DATA_FILE = "math_15k.jsonl"  # 请求数据文件


# ============ 数据加载 ============
async def load_data():
    data = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                # RL 要求
                data.append(obj["src"][0] if "src" in obj else obj.get("content", line.strip()))
            except json.JSONDecodeError:
                data.append(line.strip())
    if not data:
        raise ValueError(f"{DATA_FILE} 为空或格式不正确")
    return data


# ============ 请求发送 ============
async def send_request(session, payload):
    start_time = time.perf_counter()
    try:
        async with session.post(API_URL, json=payload) as resp:
            try:
                _ = await resp.json()
            except Exception:
                _ = await resp.text()
            latency = time.perf_counter() - start_time
            return resp.status == 200, latency, resp.status, None if resp.status == 200 else _
    except Exception as e:
        latency = time.perf_counter() - start_time
        return False, latency, None, f"{type(e).__name__}: {e}"


# ============ Worker ============
async def worker(name, session, prompts, counter, latencies, pbar, queue):
    while True:
        i = await queue.get()
        if i is None:  # 毒丸退出
            queue.task_done()
            break

        payload = {
            "model": "eb",
            "messages": [{"role": "user", "content": prompts[i % len(prompts)]}],
            "max_prompt_len": 2048,
            "max_dec_len": 1024,
            "min_dec_len": 32,
            "top_p": 1.0,
            "temperature": 1.0,
            "repetition_penalty": 1.0,
            "rollout_quant_type": "weight_only_int8",
            "disable_chat_template": True,
        }

        success, latency, status, error = await send_request(session, payload)
        if success:
            counter["success"] += 1
            latencies.append(latency)
        else:
            # print(f"Request failed ({status}): {error}")
            counter["fail"] += 1
            counter[f"error_{error or 'client'}"] += 1

        pbar.update(1)
        queue.task_done()


# ============ 主流程 ============
async def run_load_test():
    prompts = await load_data()
    queue = asyncio.Queue(maxsize=MAX_CONCURRENCY * 5)  # 限制队列大小，降低内存占用
    counter = Counter()
    latencies = []

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY * 2)  # 限制TCP连接
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        with tqdm(total=TOTAL_REQUESTS, desc="压测进度") as pbar:
            # 启动 Worker
            workers = [
                asyncio.create_task(worker(f"W{i}", session, prompts, counter, latencies, pbar, queue))
                for i in range(MAX_CONCURRENCY)
            ]

            # 边生产边消费
            for i in range(TOTAL_REQUESTS):
                await queue.put(i)

            # 发送毒丸让 worker 退出
            for _ in workers:
                await queue.put(None)

            await queue.join()
            await asyncio.gather(*workers)

    generate_report(counter, latencies)


# ============ 报告输出 ============
def generate_report(counter, latencies):
    print("\n====== 压测报告 ======")
    total = counter["success"] + counter["fail"]
    print(f"总请求数: {total}")
    print(f"成功数: {counter['success']}")
    print(f"失败数: {counter['fail']}")
    for k, v in counter.items():
        if k.startswith("error_"):
            print(f"{k}: {v}")
    if latencies:
        print(f"平均延迟: {mean(latencies):.4f}s")
        print(f"中位延迟: {median(latencies):.4f}s")
        print(f"最快: {min(latencies):.4f}s")
        print(f"最慢: {max(latencies):.4f}s")
    print("=====================")


if __name__ == "__main__":
    asyncio.run(run_load_test())
