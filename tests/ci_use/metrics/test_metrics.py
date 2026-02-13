import asyncio
import os
import shutil
import signal
import subprocess
import sys
import time

import httpx
import pytest
import requests

tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, tests_dir)

from e2e.utils.serving_utils import (
    FD_API_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_CONTROLLER_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    clean_ports,
    is_port_open,
)


@pytest.fixture(scope="session", autouse=True)
def setup_and_run_server():
    """
    Pytest fixture that runs once per test session:
    - Cleans ports before tests
    - Starts the API server as a subprocess
    - Waits for server port to open (up to 30 seconds)
    - Tears down server after all tests finish
    """
    print("Pre-test port cleanup...")
    clean_ports([FD_API_PORT, FD_ENGINE_QUEUE_PORT, FD_METRICS_PORT, FD_CACHE_QUEUE_PORT, FD_CONTROLLER_PORT])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "6,7"
    env["ENABLE_V1_KVCACHE_SCHEDULER"] = "1"

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "TP2")
    else:
        model_path = "./TP2"

    log_path = "server.log"
    cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(FD_API_PORT),
        "--tensor-parallel-size",
        "2",
        "--engine-worker-queue-port",
        str(FD_ENGINE_QUEUE_PORT),
        "--metrics-port",
        str(FD_METRICS_PORT),
        "--cache-queue-port",
        str(FD_CACHE_QUEUE_PORT),
        "--controller-port",
        str(FD_CONTROLLER_PORT),
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.9",
        "--load-strategy",
        "ipc_snapshot",
        "--dynamic-load-weight",
    ]

    # Start subprocess in new process group
    # 清除log目录
    if os.path.exists("log"):
        shutil.rmtree("log")
    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Enables killing full group via os.killpg
            env=env,
        )

    # Wait up to 300 seconds for API server to be ready
    for _ in range(300):
        if is_port_open("127.0.0.1", FD_API_PORT):
            print(f"API server is up on port {FD_API_PORT}")
            break
        time.sleep(1)
    else:
        print("[TIMEOUT] API server failed to start in 5 minutes. Cleaning up...")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception as e:
            print(f"Failed to kill process group: {e}")
        raise RuntimeError(f"API server did not start on port {FD_API_PORT}")

    yield  # Run tests

    print("\n===== Post-test server cleanup... =====")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        print(f"API server (pid={process.pid}) terminated")
    except Exception as e:
        print(f"Failed to terminate API server: {e}")


async def send_inference(idx, client: httpx.AsyncClient):
    try:
        url = f"http://0.0.0.0:{FD_API_PORT}/v1/chat/completions"
        data = {
            "model": "dummy",
            "messages": [{"role": "user", "content": f"hello {idx}"}],
            "metadata": {"min_tokens": 1000},
        }
        resp = await client.post(url, json=data, timeout=20)
        return resp.status_code
    except Exception as e:
        print(f"infer {idx} error:", e)
        return None


async def run_concurrent_inference(n):
    async with httpx.AsyncClient() as client:
        tasks = [send_inference(i, client) for i in range(n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results


def async_concurrency(n=10):
    print(f"Launching {n} concurrent async inference requests...")
    t0 = time.time()
    results = asyncio.run(run_concurrent_inference(n))
    print("Done in", time.time() - t0, "seconds")
    print("Status codes:", results)


def parse_prometheus_to_dict(metrics_text: str):
    """转换为dict格式"""
    result = {}
    for line in metrics_text.split("\n"):
        line = line.strip()
        # 跳过注释和空行
        if not line or line.startswith("#"):
            continue

        if "{" in line:  # 有 label
            metric_name = line.split("{", 1)[0]
            labels_str = line[line.index("{") + 1 : line.index("}")]
            value = float(line.split("}")[1].strip())

            # 解析 labels
            labels = {}
            for kv in labels_str.split(","):
                if "=" not in kv:
                    continue
                k, v = kv.split("=")
                labels[k] = v.strip('"')

            # 存储
            if metric_name not in result:
                result[metric_name] = []
            result[metric_name].append({"labels": labels, "value": value})

        else:  # 无 label
            metric_name, value_str = line.split()
            result[metric_name] = float(value_str)

    return result


def get_metrics_dict(metrics_url):
    """获取metrics指标数据"""
    resp = requests.get(metrics_url, timeout=5)

    assert resp.status_code == 200, f"Unexpected status code: {resp.status_code}"
    assert "text/plain" in resp.headers["Content-Type"], "Content-Type is not text/plain"

    # Parse Prometheus metrics data
    metrics_data = resp.text
    metrics_dict = parse_prometheus_to_dict(metrics_data)

    return metrics_dict


def poll_metrics_until(metrics_url, predicate=None, timeout=10, interval=0.3):
    """
    轮询 metrics 直到 predicate(metrics_dict) 返回 True 或超时。
    返回最后一次拉取到的 metrics_dict。
    """
    deadline = time.time() + timeout
    while True:
        metrics = get_metrics_dict(metrics_url)
        if predicate is None or predicate(metrics):
            return metrics
        if time.time() >= deadline:
            return metrics
        time.sleep(interval)


def test_metrics_during_inference():
    """
    正常推理场景：并发 10 个请求，max_num_seqs=1。
    验证 enqueued -> waiting -> running 各状态计数的正确性，以及无 preemption。

    请求在系统中的流转路径：
      1. 请求进入 _recv_request_loop -> enqueued.inc(1)
      2. 调度器取走请求 -> enqueued.dec(N)，进入 resource_manager.waiting
      3. schedule() 将 waiting 中的请求分配资源 -> running
      4. update_metrics() 刷新 num_requests_running / num_requests_waiting / num_requests_preempted
    """
    metrics_url = f"http://0.0.0.0:{FD_METRICS_PORT}/metrics"
    base_metrics = get_metrics_dict(metrics_url)
    async_concurrency(n=10)

    # 等待所有 10 个请求到达服务端
    def all_requests_arrived(m):
        total = m.get("fastdeploy:requests_number_total", 0) - base_metrics.get("fastdeploy:requests_number_total", 0)
        return total >= 10

    metrics = poll_metrics_until(metrics_url, all_requests_arrived, timeout=10)

    running = metrics["fastdeploy:num_requests_running"]
    waiting = metrics["fastdeploy:num_requests_waiting"]
    preempted = metrics["fastdeploy:num_requests_preempted"]
    enqueued = metrics["fastdeploy:num_requests_enqueued"]
    total = metrics["fastdeploy:requests_number_total"] - base_metrics["fastdeploy:requests_number_total"]

    assert total == 10, f"server should have received all 10 requests, got {total}"

    # max_num_seqs=1，最多只有 1 个请求处于 running
    assert running <= 1, f"at most 1 request should be running (max_num_seqs=1), got {running}"

    # enqueued(已入队未调度) + waiting(已调度等资源) 覆盖剩余非 running 请求
    # 由于异步时序，部分请求可能还在 enqueued，部分已到 waiting
    non_running = enqueued + waiting
    assert non_running >= 0, f"enqueued({enqueued}) + waiting({waiting}) should be non-negative"
    assert (
        running + non_running <= total
    ), f"running({running}) + enqueued({enqueued}) + waiting({waiting}) should not exceed total({total})"

    # max_num_seqs=1 不会触发抢占
    assert preempted == 0, f"no preemption should be triggered, got {preempted}"


def test_metrics_with_clear_and_reset():
    """
    权重清除场景：验证 reset_scheduler 接口调用后，metrics 指标是否被正确重置。
    """
    metrics_url = f"http://0.0.0.0:{FD_METRICS_PORT}/metrics"
    base_metrics = get_metrics_dict(metrics_url)

    # 1. 发送请求并等待所有请求到达
    async_concurrency(n=10)

    def all_requests_arrived(m):
        total = m.get("fastdeploy:requests_number_total", 0) - base_metrics.get("fastdeploy:requests_number_total", 0)
        return total >= 10

    poll_metrics_until(metrics_url, all_requests_arrived, timeout=10)

    # 2. 调用 clear_load_weight — 清除模型权重（RL 动态加载场景）
    clear_url = f"http://0.0.0.0:{FD_API_PORT}/clear_load_weight"
    print("Calling clear_load_weight...")
    r = requests.get(clear_url, timeout=30)
    assert r.status_code == 200, f"clear_load_weight failed: {r.status_code}"

    # 3. 调用 reset_scheduler — 清空调度队列并重置资源管理器指标
    reset_url = f"http://0.0.0.0:{FD_CONTROLLER_PORT}/controller/reset_scheduler"
    print("Calling reset_scheduler...")
    r = requests.post(reset_url, json={"reset": True}, timeout=30)
    assert r.status_code == 200, f"reset_scheduler failed: {r.status_code}"

    # 4. 检查 enqueued/running/waiting/preempted 指标应被 reset_metrics() 清零
    metrics = poll_metrics_until(metrics_url, predicate=None, timeout=10)

    enqueued = metrics["fastdeploy:num_requests_enqueued"]
    running = metrics["fastdeploy:num_requests_running"]
    waiting = metrics["fastdeploy:num_requests_waiting"]
    preempted = metrics["fastdeploy:num_requests_preempted"]
    total = metrics["fastdeploy:requests_number_total"] - base_metrics["fastdeploy:requests_number_total"]

    assert total == 10, f"server should have received all 10 requests, got {total}"
    assert enqueued == 0, f"after reset_scheduler, num_requests_running should be 0, got {running}"
    assert running == 0, f"after reset_scheduler, num_requests_running should be 0, got {running}"
    assert waiting == 0, f"after reset_scheduler, num_requests_waiting should be 0, got {waiting}"
    assert preempted == 0, f"after reset_scheduler, num_requests_preempted should be 0, got {preempted}"


if __name__ == "__main__":
    test_metrics_with_clear_and_reset()
