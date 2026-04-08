import os
import threading
import time

import requests

FD_API_PORT = os.getenv("FD_API_PORT", 8180)
BASE_URL = f"http://localhost:{FD_API_PORT}"


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
