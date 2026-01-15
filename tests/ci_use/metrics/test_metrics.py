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
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
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


def test_metrics_with_too_many_requests():
    metrics_url = f"http://0.0.0.0:{FD_METRICS_PORT}/metrics"
    base_metrics = get_metrics_dict(metrics_url)
    async_concurrency(n=10)
    time.sleep(0.3)

    metrics = get_metrics_dict(metrics_url)
    running = metrics["fastdeploy:num_requests_running"]
    waiting = metrics["fastdeploy:num_requests_waiting"]
    preempted = metrics["fastdeploy:num_requests_preempted"]
    enqueued = metrics["fastdeploy:num_requests_enqueued"]
    total = metrics["fastdeploy:requests_number_total"] - base_metrics["fastdeploy:requests_number_total"]
    assert total == 10, f"server should have received all 10 requests, got {total}"
    assert running == 1, "only one request should be running, for max_num_seqs = 1"
    assert enqueued + waiting == 9, "the other requests should all be waiting"
    assert preempted == 0, "no preemption should be triggered"


def test_metrics_with_clear_and_reset():
    """
    Test the metrics monitoring endpoint.
    """
    metrics_url = f"http://0.0.0.0:{FD_METRICS_PORT}/metrics"
    base_metrics = get_metrics_dict(metrics_url)

    # 1. send requests
    async_concurrency(n=10)

    time.sleep(0.3)

    # 2. clear_load_weight
    clear_url = f"http://0.0.0.0:{FD_API_PORT}/clear_load_weight"
    print("Calling clear_load_weight...")
    r = requests.get(clear_url, timeout=30)
    assert r.status_code == 200, f"clear_load_weight failed: {r.status_code}"

    # 3. reset_scheduler
    reset_url = f"http://0.0.0.0:{FD_CONTROLLER_PORT}/controller/reset_scheduler"
    print("Calling reset_scheduler...")
    r = requests.post(reset_url, json={"reset": True}, timeout=30)
    assert r.status_code == 200, f"reset_scheduler failed: {r.status_code}"

    # 4. check metrics after reset_scheduler
    metrics = get_metrics_dict(metrics_url)
    running = metrics["fastdeploy:num_requests_running"]
    waiting = metrics["fastdeploy:num_requests_waiting"]
    preempted = metrics["fastdeploy:num_requests_preempted"]
    enqueued = metrics["fastdeploy:num_requests_enqueued"]
    total = metrics["fastdeploy:requests_number_total"] - base_metrics["fastdeploy:requests_number_total"]

    assert total == 10, f"server should have received all 10 requests, got {total}"
    assert running == 0, f"after reset_scheduler, num_requests_running should be 0, got {running}"
    assert waiting == 0, f"after reset_scheduler, num_requests_waiting should be 0, got {waiting}"
    assert preempted == 0, f"after reset_scheduler, num_requests_preempted should be 0, got {preempted}"
    assert enqueued == 0, f"after reset_scheduler, num_requests_enqueued should be 0, got {enqueued}"


if __name__ == "__main__":
    test_metrics_with_clear_and_reset()
