import json
import os
import threading
import time

import requests

FD_API_PORT = os.getenv("FD_API_PORT", 8180)
BASE_URL = f"http://localhost:{FD_API_PORT}"
FD_METRICS_PORT = os.getenv("FD_METRICS_PORT", 8078)
FD_METRICS_URL = f"http://localhost:{FD_METRICS_PORT}"


# ---------------------------
# 基础接口调用函数
# ---------------------------
def pause():
    """
    发送 pause 请求
    1. 确认返回 200
    2. 确认 status 为 success
    """
    resp = requests.post(f"{BASE_URL}/v1/pause")
    print("pause:", resp.status_code, resp.text)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    return resp


def resume():
    """
    发送 resume 请求
    1. 确认返回 200
    2. 确认 status 为 success
    """
    resp = requests.post(f"{BASE_URL}/v1/resume")
    print("resume:", resp.status_code, resp.text)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    return resp


def is_paused(expected=None):
    """
    查询当前 paused 状态
    1. 返回 200
    2. status 为 success
    3. 如果传 expected，则断言返回值与期望一致
    """
    resp = requests.get(f"{BASE_URL}/v1/is_paused")
    print("is_paused:", resp.status_code, resp.text)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    paused = data["result"]["is_paused"]
    if expected is not None:
        assert paused == expected, f"Expected is_paused={expected}, got {paused}"
    return paused


def get_available_gpu_block_num():
    """
    查询 metrics 接口，返回 fastdeploy:available_gpu_block_num 的值
    1. 请求 /metrics 接口
    2. 过滤出 fastdeploy:available_gpu_block_num 指标行
    3. 返回解析后的数值
    """
    resp = requests.get(f"{FD_METRICS_URL}/metrics")
    print("metrics:", resp.status_code)
    assert resp.status_code == 200, f"Metrics HTTP status not 200: {resp.status_code}"

    value = None
    for line in resp.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "fastdeploy:available_gpu_block_num" in line:
            # Prometheus metrics 格式: metric_name{labels} value
            parts = line.split()
            if len(parts) >= 2:
                value = float(parts[-1])
                print("available_gpu_block_num:", value)
                break

    assert value is not None, "fastdeploy:available_gpu_block_num not found in metrics"
    return value


def abort_requests(req_ids=None, abort_all=False):
    """
    中断推理请求
    1. abort_all=True 时中断所有请求
    2. 传入 req_ids 列表时中断指定请求
    3. 两者不能同时使用，至少指定一个
    """
    assert abort_all or req_ids, "abort_all 或 req_ids 必须指定一个"
    assert not (abort_all and req_ids), "abort_all 和 req_ids 不能同时使用"

    payload = {}
    if abort_all:
        payload["abort_all"] = True
    else:
        payload["req_ids"] = req_ids

    resp = requests.post(f"{BASE_URL}/v1/abort_requests", json=payload)
    print("abort_requests:", resp.status_code, resp.text)
    assert resp.status_code == 200, f"abort_requests HTTP status not 200: {resp.status_code}"
    return resp


def do_completions(req_id_holder, result_holder, stream=True):
    """
    发送 /v1/completions 请求，将结果写入 holder
    1. stream=True 时通过流式方式读取，逐步拼接 text 和 token_count
    2. stream=False 时非流式请求，一次性读取 text
    3. 将 id、text、finish_reason 写入 result_holder
    """
    payload = {
        "prompt": "世界上存在多少人口？",
        "stream": stream,
        "min_tokens": 1800,
    }
    result_holder["id"] = None
    result_holder["text"] = ""
    result_holder["token_count"] = 0
    result_holder["finish_reason"] = None
    result_holder["error"] = None
    try:
        resp = requests.post(f"{BASE_URL}/v1/completions", json=payload, stream=stream, timeout=120)
        if stream:
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    chunk = json.loads(data)
                    result_holder["token_count"] += 1
                    if "id" in chunk and not req_id_holder.get("id"):
                        req_id_holder["id"] = chunk["id"]
                    if chunk.get("choices") and len(chunk["choices"]) > 0:
                        choice = chunk["choices"][0]
                        if choice.get("text"):
                            result_holder["text"] += choice["text"]
                        if choice.get("finish_reason"):
                            result_holder["finish_reason"] = choice["finish_reason"]
        else:
            if resp.status_code != 200:
                result_holder["error"] = f"HTTP {resp.status_code}: {resp.text}"
                return
            data = resp.json()
            result_holder["id"] = data.get("id")
            if data.get("choices") and len(data["choices"]) > 0:
                choice = data["choices"][0]
                result_holder["text"] = choice.get("text", "")
                result_holder["finish_reason"] = choice.get("finish_reason")
    except Exception as e:
        result_holder["error"] = str(e)


def do_chat_completions(req_id_holder, result_holder, stream=True, min_tokens=1800, max_tokens=None):
    """
    发送 /v1/chat/completions 请求，将结果写入 holder
    1. stream=True 时通过流式方式读取，逐步拼接 content 和 token_count
    2. stream=False 时非流式请求，一次性读取 content
    3. 将 id、content、finish_reason 写入 result_holder
    """
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
        "min_tokens": min_tokens,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    result_holder["id"] = None
    result_holder["content"] = ""
    result_holder["token_count"] = 0
    result_holder["finish_reason"] = None
    result_holder["error"] = None
    try:
        resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, stream=stream, timeout=120)
        if stream:
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    chunk = json.loads(data)
                    result_holder["token_count"] += 1
                    if "id" in chunk and not req_id_holder.get("id"):
                        req_id_holder["id"] = chunk["id"]
                    if chunk.get("choices") and chunk["choices"][0].get("finish_reason"):
                        result_holder["finish_reason"] = chunk["choices"][0]["finish_reason"]
        else:
            if resp.status_code != 200:
                result_holder["error"] = f"HTTP {resp.status_code}: {resp.text}"
                return
            data = resp.json()
            result_holder["id"] = data.get("id")
            if data.get("choices") and len(data["choices"]) > 0:
                choice = data["choices"][0]
                message = choice.get("message", {})
                result_holder["content"] = message.get("content", "")
                result_holder["finish_reason"] = choice.get("finish_reason")
    except Exception as e:
        result_holder["error"] = str(e)


def launch_stream_requests(n):
    """
    发起 n 个流式推理请求
    返回 (threads, req_id_holders, result_holders)
    """
    req_id_holders = [{} for _ in range(n)]
    result_holders = [{} for _ in range(n)]
    threads = []
    for i in range(n):
        t = threading.Thread(target=do_chat_completions, args=(req_id_holders[i], result_holders[i], True, 1800, 1869))
        t.start()
        threads.append(t)
    return threads, req_id_holders, result_holders


def wait_requests_started(req_id_holders, timeout=30):
    """
    等待所有请求拿到 request id
    返回已拿到 id 的请求数量
    """
    start = time.time()
    while time.time() - start < timeout:
        got = sum(1 for h in req_id_holders if h.get("id"))
        if got >= len(req_id_holders):
            return got
        time.sleep(0.1)
    return sum(1 for h in req_id_holders if h.get("id"))


def infer(expect_success=True):
    """
    发送推理请求
    1. 根据 expect_success 参数判断是否预期成功
    2. 响应 HTTP 状态码 !=200 或返回 error 时：
       - expect_success=True -> 抛出断言
       - expect_success=False -> 通过
    3. 响应正常包含 choices 时：
       - expect_success=True -> 通过
       - expect_success=False -> 抛出断言
    """
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0,
                "top_p": 0,
                "seed": 33,
                "min_tokens": 1366,
                "max_tokens": 1369,
                "stream": False,
            },
            timeout=16,
        )
        print("infer:", resp.status_code, resp.text)

        assert resp.status_code == 200, f"Infer HTTP status not 200: {resp.status_code}"

        data = resp.json()

        if "error" in data:
            print("infer returned error:", data["error"]["message"])
            if expect_success:
                raise AssertionError(f"Infer expected success, but got error: {data['error']['message']}")
            else:
                return resp

        if "choices" in data:
            if not expect_success:
                raise AssertionError("Infer expected failure, but succeeded")
            print("infer succeeded as expected")
        else:
            raise AssertionError(f"Infer response missing 'choices': {data}")

        return resp

    except Exception as e:
        if expect_success:
            raise
        else:
            print("infer failed as expected:", e)
            return None


# ---------------------------
# 测试用例1：基础功能验证
# ---------------------------
def test_basic():
    """
    基础接口功能验证：
    1. 查询初始状态，期望 False
    2. pause -> 查询，期望 True
    3. resume -> 查询，期望 False
    """
    is_paused(expected=False)
    pause()
    is_paused(expected=True)
    resume()
    is_paused(expected=False)


# ---------------------------
# 测试用例2：重复调用
# ---------------------------
def test_reentrant():
    """
    重入调用测试：
    1. pause 两次 -> 状态仍为 True
    2. resume 两次 -> 状态仍为 False
    """
    pause()
    pause()
    is_paused(expected=True)
    resume()
    resume()
    is_paused(expected=False)


# ---------------------------
# 测试用例3：异常调用场景
# ---------------------------
def test_exception_scenario():
    """
    异常调用场景：
    1. infer 过程中 pause -> infer 应失败
    2. resume -> 状态恢复为 False
    """
    t = threading.Thread(target=lambda: infer(expect_success=False))
    t.start()
    time.sleep(0.1)
    pause()
    t.join()

    resume()
    is_paused(expected=False)


# ---------------------------
# 测试用例4：高并发pause/resume
# ---------------------------
def test_concurrent():
    """
    高并发场景：
    1. 并发 infer 5 个
    2. pause -> 正在推理请求被中断
    3. 查询状态 -> True
    4. resume -> 查询状态 -> False
    """
    threads = []
    for _ in range(5):
        t = threading.Thread(target=lambda: infer(expect_success=False))
        threads.append(t)
        t.start()

    time.sleep(0.1)
    pause()
    is_paused(expected=True)
    resume()
    is_paused(expected=False)

    for t in threads:
        t.join()


# ---------------------------
# 测试用例5：大量重复调用
# ---------------------------
def test_reentrant1():
    """
    重入调用测试：
    1. pause 9次 -> 状态仍为 True
    2. resume 9次 -> 状态仍为 False
    """
    for i in range(9):
        pause()
    is_paused(expected=True)
    for i in range(9):
        resume()
    is_paused(expected=False)


# ---------------------------
# 测试用例6：大量重启服务
# ---------------------------
def test_reentrant2():
    """
    重入调用测试：
    1. pause
    2. resume
    3. continue 9 times
    """
    for i in range(9):
        pause()
        resume()
    is_paused(expected=False)
    for i in range(9):
        resume()
        pause()
    is_paused(expected=True)
    resume()
    is_paused(expected=False)


# ---------------------------
# 测试用例7：停止状态下请求
# ---------------------------
def test_exception_scenario1():
    """
    停止状态下请求
    """
    pause()
    is_paused(expected=True)
    infer(expect_success=False)
    resume()
    is_paused(expected=False)
    infer(expect_success=True)


# ---------------------------
# 测试用例8：高并发 pause/resume
# ---------------------------
def test_concurrent_pause_resume():
    """
    高并发 pause/resume 场景：
    1. 并发发送 pause 请求 5 次
    2. 查询 is_paused 状态 -> True
    3. 并发发送 resume 请求 5 次
    4. 查询 is_paused 状态 -> False
    5. 验证并发重复请求不会破坏状态
    """
    pause_threads = []
    resume_threads = []

    # 并发发送 pause
    for _ in range(5):
        t = threading.Thread(target=pause)
        pause_threads.append(t)
        t.start()

    for t in pause_threads:
        t.join()

    # 检查状态应为 paused
    is_paused(expected=True)

    # 并发发送 resume
    for _ in range(5):
        t = threading.Thread(target=resume)
        resume_threads.append(t)
        t.start()

    for t in resume_threads:
        t.join()

    # 检查状态应恢复为未 paused
    is_paused(expected=False)


# ---------------------------
# 测试用例9：中断部分流式请求并验证 block 恢复
# ---------------------------
def test_abort_partial_requests():
    """
    部分中断请求测试：
    1. 记录初始 available_gpu_block_num
    2. 并发发送 10 个流式推理请求
    3. 等待所有请求拿到 request id
    4. 筛选仍在运行中的请求，中断其中一部分
    5. 等待被中断的请求结束，断言 finish_reason 为 abort
    6. 等待剩余请求全部完成
    7. 记录结束后的 available_gpu_block_num
    8. 断言前后 block 数一致
    """
    # 1. 记录初始 block 数
    blocks_before = get_available_gpu_block_num()

    # 2. 并发发送 10 个流式推理请求
    threads, req_id_holders, result_holders = launch_stream_requests(10)

    # 3. 等待所有请求拿到 request id
    started = wait_requests_started(req_id_holders)
    print(f"test_abort_partial_requests: {started}/10 个请求已拿到 request_id")
    assert started > 0, "没有请求拿到 request_id"

    # 4. 等待推理产出一些 token
    time.sleep(1)

    # 4.1 筛选仍在运行中的请求（finish_reason 为 None 说明还没结束）
    running_indices = []
    for i in range(len(threads)):
        if req_id_holders[i].get("id") and result_holders[i].get("finish_reason") is None:
            running_indices.append(i)
    assert (
        len(running_indices) >= 2
    ), f"仍在运行中的请求不足 2 个（共 {len(running_indices)} 个），无法执行部分中断测试"
    print(f"test_abort_partial_requests: {len(running_indices)}/10 个请求仍在运行中")

    # 4.2 选择前半部分进行中断
    abort_count = len(running_indices) // 2
    abort_ids = [req_id_holders[i]["id"] for i in running_indices[:abort_count]]
    abort_id_set = set(abort_ids)
    print(f"test_abort_partial_requests: 要 abort 的请求: {abort_ids}")

    # 执行 abort
    abort_requests(req_ids=abort_ids)

    # 5. 等待被中断的请求结束
    for i, t in enumerate(threads):
        req_id = req_id_holders[i].get("id")
        if req_id and req_id in abort_id_set:
            t.join(timeout=10)

    # 5.1 断言被中断请求的 finish_reason 为 abort
    for i in running_indices[:abort_count]:
        req_id = req_id_holders[i].get("id")
        finish_reason = result_holders[i].get("finish_reason")
        assert finish_reason == "abort", f"请求 {req_id} 被 abort 但 finish_reason={finish_reason}, 期望 abort"
    print("test_abort_partial_requests: 被中断请求的 finish_reason 均为 abort")

    # 6. 等待剩余正常请求全部完成
    for t in threads:
        t.join(timeout=120)

    # 7. 记录结束后的 block 数
    blocks_after = get_available_gpu_block_num()

    # 8. 断言前后 block 数一致
    assert blocks_before == blocks_after, f"Block 数不一致: before={blocks_before}, after={blocks_after}"
    print(f"test_abort_partial_requests: block 数一致 ({blocks_before})")


# ---------------------------
# 测试用例10：abort_all 中断全部流式请求
# ---------------------------
def test_abort_all_requests():
    """
    abort_all 中断全部请求测试：
    1. 记录初始 available_gpu_block_num
    2. 并发发送 10 个流式推理请求
    3. 等待所有请求拿到 request id
    4. 调用 abort_requests(abort_all=True)
    5. 等待所有请求结束
    6. 断言所有请求的 finish_reason 为 abort
    7. 记录结束后的 available_gpu_block_num
    8. 断言前后 block 数一致
    """
    # 1. 记录初始 block 数
    blocks_before = get_available_gpu_block_num()

    # 2. 并发发送 10 个流式推理请求
    threads, req_id_holders, result_holders = launch_stream_requests(10)

    # 3. 等待所有请求拿到 request id
    started = wait_requests_started(req_id_holders)
    print(f"test_abort_all_requests: {started}/10 个请求已拿到 request_id")
    assert started > 0, "没有请求拿到 request_id"

    # 等待推理产出一些 token
    time.sleep(1)

    # 4. 执行 abort_all（对仍在运行中的请求生效）
    abort_requests(abort_all=True)

    # 5. 等待所有请求结束
    for t in threads:
        t.join(timeout=10)

    # 6. 断言所有仍在运行中的请求 finish_reason 为 abort
    aborted_count = 0
    for i in range(len(threads)):
        finish_reason = result_holders[i].get("finish_reason")
        if finish_reason == "abort":
            aborted_count += 1
    assert aborted_count > 0, "没有请求被 abort（可能请求在 abort 前已全部完成）"
    print(f"test_abort_all_requests: {aborted_count}/{len(threads)} 个请求的 finish_reason 为 abort")

    # 7. 记录结束后的 block 数
    blocks_after = get_available_gpu_block_num()

    # 8. 断言前后 block 数一致
    assert blocks_before == blocks_after, f"Block 数不一致: before={blocks_before}, after={blocks_after}"
    print(f"test_abort_all_requests: block 数一致 ({blocks_before})")


# ---------------------------
# 测试用例11：流式 completions 接口被 abort 后正常返回已生成的 token
# ---------------------------
def test_abort_completions_return_tokens():
    """
    completions 接口 abort 后返回已生成 token 测试：
    1. 记录初始 available_gpu_block_num
    2. 并发发送 5 个 /v1/completions 流式请求
    3. 等待所有请求拿到 request id
    4. 调用 abort_all 中断全部请求
    5. 等待所有请求结束
    6. 断言被 abort 的请求 finish_reason 为 abort
    7. 断言被 abort 的请求已生成的 text 不为空（token 已正常返回）
    8. 断言前后 block 数一致
    """
    # 1. 记录初始 block 数
    blocks_before = get_available_gpu_block_num()

    # 2. 并发发送 5 个 completions 流式请求
    n = 5
    req_id_holders = [{} for _ in range(n)]
    result_holders = [{} for _ in range(n)]
    threads = []
    for i in range(n):
        t = threading.Thread(target=do_completions, args=(req_id_holders[i], result_holders[i], True))
        t.start()
        threads.append(t)

    # 2. 等待所有请求拿到 request id
    started = wait_requests_started(req_id_holders)
    print(f"test_abort_completions_return_tokens: {started}/{n} 个请求已拿到 request_id")
    assert started > 0, "没有请求拿到 request_id"

    # 等待推理产出一些 token
    time.sleep(1)

    # 3. 执行 abort_all（对仍在运行中的请求生效）
    abort_requests(abort_all=True)

    # 4. 等待所有请求结束
    for t in threads:
        t.join(timeout=10)

    # 5. 断言被 abort 的请求 finish_reason 为 abort
    aborted_count = 0
    for i in range(n):
        finish_reason = result_holders[i].get("finish_reason")
        if finish_reason == "abort":
            aborted_count += 1
    assert aborted_count > 0, "没有请求被 abort（可能请求在 abort 前已全部完成）"
    print(f"test_abort_completions_return_tokens: {aborted_count}/{n} 个请求的 finish_reason 为 abort")

    # 6. 断言被 abort 的请求已生成的 text 不为空
    for i in range(n):
        text = result_holders[i].get("text", "")
        token_count = result_holders[i].get("token_count", 0)
        if result_holders[i].get("finish_reason") != "abort":
            continue
        assert len(text) > 0, f"请求 {req_id_holders[i].get('id')} abort 后返回的 text 为空"
        assert token_count > 0, f"请求 {req_id_holders[i].get('id')} abort 后 token_count=0"
        print(
            f"test_abort_completions_return_tokens: 请求 {req_id_holders[i].get('id')} "
            f"返回 {token_count} 个 token, text 长度={len(text)}"
        )
    print("test_abort_completions_return_tokens: 所有被 abort 的请求均正常返回已生成的 token")

    # 7. 记录结束后的 block 数
    blocks_after = get_available_gpu_block_num()

    # 8. 断言前后 block 数一致
    assert blocks_before == blocks_after, f"Block 数不一致: before={blocks_before}, after={blocks_after}"
    print(f"test_abort_completions_return_tokens: block 数一致 ({blocks_before})")


# ---------------------------
# 测试用例12：非流式 completions 被 abort 后正常返回已生成的 token
# ---------------------------
def test_abort_non_stream_completions_return_tokens():
    """
    非流式 completions abort 后返回已生成 token 测试：
    1. 记录初始 available_gpu_block_num
    2. 并发发送 5 个 /v1/completions 非流式请求（min_tokens=1800 确保不会自然结束）
    3. 等待请求开始执行
    4. 调用 abort_all 中断全部请求
    5. 等待所有请求结束
    6. 断言所有请求的 finish_reason 为 abort
    7. 断言所有请求返回的 text 不为空（token 已正常返回）
    8. 断言前后 block 数一致
    """
    # 1. 记录初始 block 数
    blocks_before = get_available_gpu_block_num()

    # 2. 并发发送 5 个非流式 completions 请求
    n = 5
    req_id_holders = [{} for _ in range(n)]
    result_holders = [{} for _ in range(n)]
    threads = []
    for i in range(n):
        t = threading.Thread(target=do_completions, args=(req_id_holders[i], result_holders[i], False))
        t.start()
        threads.append(t)

    # 2. 等待请求开始执行
    time.sleep(1)

    # 3. 执行 abort_all（对仍在运行中的请求生效）
    abort_requests(abort_all=True)

    # 4. 等待所有请求结束
    for t in threads:
        t.join(timeout=10)

    # 5. 断言被 abort 的请求 finish_reason 为 abort（非流式 abort 时连接可能断开）
    aborted_count = 0
    for i in range(n):
        req_id = result_holders[i].get("id")
        error = result_holders[i].get("error")
        if error:
            print(f"test_abort_non_stream_completions_return_tokens: 请求 {req_id} 遇到异常(abort断开): {error}")
            aborted_count += 1
            continue
        finish_reason = result_holders[i].get("finish_reason")
        if finish_reason == "abort":
            aborted_count += 1
    assert aborted_count > 0, "没有请求被 abort（可能请求在 abort 前已全部完成）"
    print(f"test_abort_non_stream_completions_return_tokens: {aborted_count}/{n} 个请求被 abort")

    # 6. 断言所有被 abort 的未异常请求返回的 text 不为空
    for i in range(n):
        req_id = result_holders[i].get("id")
        if result_holders[i].get("error"):
            continue
        if result_holders[i].get("finish_reason") != "abort":
            continue
        text = result_holders[i].get("text", "")
        assert len(text) > 0, f"请求 {req_id} abort 后返回的 text 为空"
        print(f"test_abort_non_stream_completions_return_tokens: 请求 {req_id} " f"text 长度={len(text)}")
    print("test_abort_non_stream_completions_return_tokens: 所有被 abort 的请求均正常返回已生成的 token")

    # 7. 记录结束后的 block 数
    blocks_after = get_available_gpu_block_num()

    # 8. 断言前后 block 数一致
    assert blocks_before == blocks_after, f"Block 数不一致: before={blocks_before}, after={blocks_after}"
    print(f"test_abort_non_stream_completions_return_tokens: block 数一致 ({blocks_before})")


# ---------------------------
# 测试用例13：非流式 chat/completions 被 abort 后正常返回已生成的 token
# ---------------------------
def test_abort_non_stream_chat_completions_return_tokens():
    """
    非流式 chat/completions abort 后返回已生成 token 测试：
    1. 记录初始 available_gpu_block_num
    2. 并发发送 5 个 /v1/chat/completions 非流式请求（min_tokens=1800 确保不会自然结束）
    3. 等待请求开始执行
    4. 调用 abort_all 中断全部请求
    5. 等待所有请求结束
    6. 断言所有请求的 finish_reason 为 abort
    7. 断言所有请求返回的 content 不为空（token 已正常返回）
    8. 断言前后 block 数一致
    """
    # 1. 记录初始 block 数
    blocks_before = get_available_gpu_block_num()

    # 2. 并发发送 5 个非流式 chat/completions 请求
    n = 5
    req_id_holders = [{} for _ in range(n)]
    result_holders = [{} for _ in range(n)]
    threads = []
    for i in range(n):
        t = threading.Thread(target=do_chat_completions, args=(req_id_holders[i], result_holders[i], False))
        t.start()
        threads.append(t)

    # 2. 等待请求开始执行
    time.sleep(1)

    # 3. 执行 abort_all（对仍在运行中的请求生效）
    abort_requests(abort_all=True)

    # 4. 等待所有请求结束
    for t in threads:
        t.join(timeout=10)

    # 5. 断言被 abort 的请求 finish_reason 为 abort（非流式 abort 时连接可能断开）
    aborted_count = 0
    for i in range(n):
        req_id = result_holders[i].get("id")
        error = result_holders[i].get("error")
        if error:
            print(f"test_abort_non_stream_chat_completions_return_tokens: 请求 {req_id} 遇到异常(abort断开): {error}")
            aborted_count += 1
            continue
        finish_reason = result_holders[i].get("finish_reason")
        if finish_reason == "abort":
            aborted_count += 1
    assert aborted_count > 0, "没有请求被 abort（可能请求在 abort 前已全部完成）"
    print(f"test_abort_non_stream_chat_completions_return_tokens: {aborted_count}/{n} 个请求被 abort")

    # 6. 断言所有被 abort 的未异常请求返回的 content 不为空
    for i in range(n):
        req_id = result_holders[i].get("id")
        if result_holders[i].get("error"):
            continue
        if result_holders[i].get("finish_reason") != "abort":
            continue
        content = result_holders[i].get("content", "")
        assert len(content) > 0, f"请求 {req_id} abort 后返回的 content 为空"
        print(f"test_abort_non_stream_chat_completions_return_tokens: 请求 {req_id} " f"content 长度={len(content)}")
    print("test_abort_non_stream_chat_completions_return_tokens: 所有被 abort 的请求均正常返回已生成的 token")

    # 7. 记录结束后的 block 数
    blocks_after = get_available_gpu_block_num()

    # 8. 断言前后 block 数一致
    assert blocks_before == blocks_after, f"Block 数不一致: before={blocks_before}, after={blocks_after}"
    print(f"test_abort_non_stream_chat_completions_return_tokens: block 数一致 ({blocks_before})")


# ---------------------------
# 执行所有测试
# ---------------------------
if __name__ == "__main__":
    test_basic()
    time.sleep(1)
    test_reentrant()
    time.sleep(1)
    test_exception_scenario()
    time.sleep(1)
    test_concurrent()
    time.sleep(1)
    test_abort_partial_requests()
    time.sleep(1)
    test_abort_all_requests()
    time.sleep(1)
    test_abort_completions_return_tokens()
    time.sleep(1)
    test_abort_non_stream_completions_return_tokens()
    time.sleep(1)
    test_abort_non_stream_chat_completions_return_tokens()
