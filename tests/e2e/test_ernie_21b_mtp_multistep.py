# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest
from utils.baseline_manager import BaselineManager
from utils.serving_utils import (
    FD_API_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    clean,
    extract_logprobs,
    get_stream_chunks,
    is_port_open,
    send_request,
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
    clean()

    print("log dir clean ")
    if os.path.exists("log") and os.path.isdir("log"):
        shutil.rmtree("log")

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "ernie-4_5-21b-a3b-bf16-paddle")
    else:
        model_path = "./ernie-4_5-21b-a3b-bf16-paddle"
    mtp_model_path = os.path.join(model_path, "mtp")
    speculative_config = {"method": "mtp", "num_speculative_tokens": 3, "num_model_steps": 3, "model": mtp_model_path}

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
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "128",
        "--quantization",
        "wint4",
        "--enable-logprob",
        "--speculative-config",
        json.dumps(speculative_config),
        "--graph-optimization-config",
        '{"use_cudagraph":true,  "use_unique_memory_pool":true, "draft_model_use_cudagraph":true}',
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
        )

    # Wait up to 300 seconds for API server to be ready
    for _ in range(300):
        if is_port_open("127.0.0.1", FD_API_PORT):
            print(f"Server is up on port {FD_API_PORT}")
            break
        time.sleep(1)
    else:
        print("[TIMEOUT] API server failed to start in 5 minutes. Cleaning up...")
        try:
            os.killpg(process.pid, signal.SIGTERM)
            clean()
        except Exception as e:
            print(f"Failed to kill process group: {e}")
        raise RuntimeError(f"API server did not start on port {FD_API_PORT}")

    yield  # Run tests

    print("\n===== Post-test server cleanup... =====")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        clean()
        print(f"server (pid={process.pid}) terminated")
    except Exception as e:
        print(f"Failed to terminate API server: {e}")


@pytest.fixture(scope="session")
def api_url(request):
    """
    Returns the API endpoint URL for chat completions.
    """
    return f"http://0.0.0.0:{FD_API_PORT}/v1/chat/completions"


@pytest.fixture(scope="session")
def metrics_url(request):
    """
    Returns the metrics endpoint URL.
    """
    return f"http://0.0.0.0:{FD_METRICS_PORT}/metrics"


@pytest.fixture
def headers():
    """
    Returns common HTTP request headers.
    """
    return {"Content-Type": "application/json"}


def test_prefix_cache_text(api_url):
    payload = {
        "model": "null",
        "messages": [
            {
                "role": "user",
                "content": "国外项目风险管理研究起步较早，理论体系成熟。早期研究集中于保险与金融领域，后逐步扩展至工程项目、"
                "公共管理等多领域。在理论层面，COSO《企业风险管理——整合框架》和ISO31000标准为风险管理提供了系统性"
                "指导，强调风险识别、评估、应对与监控的全流程管理。风险识别方法包括故障树分析、事件树分析等；风险评估"
                "则广泛应用VaR模型、蒙特卡洛模拟等量化工具。应对策略涵盖规避、转移、减轻和接受等，并衍生出风险共享、"
                "升级等复杂策略。此外，组织文化、管理层支持等因素对风险管理有效性影响显著。近年来，随着科技发展，"
                "人工智能、大数据等技术被引入风险管理，推动其向智能化、自动化方向发展。请介绍一下国外关于项目风险管理"
                "的文献研究综述，300字以内",
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "temperature": 0.8,
        "seed": 21,
        "top_p": 0,
        "logprobs": True,
        "top_logprobs": 3,
    }

    print("fastdeploy answer is :")

    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)
    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)
    result = "".join([x["choices"][0]["delta"]["content"] for x in chunks[:-1]])

    print("\nresult:\n", result)
    logprobs = extract_logprobs(chunks)
    # req_id = chunks[-1]["id"]
    # entropy = extract_last_entropy("log/data_processor.log", req_id)
    speculate_metrics = chunks[-2]["choices"][0]["speculate_metrics"]
    print("chunks:", chunks[-1])
    print("speculate_metrics:", speculate_metrics)
    # print("entropy:", entropy)

    # print("\nlogprobs:\n", logprobs)

    resp2 = send_request(url=api_url, payload=payload)
    chunks2 = get_stream_chunks(resp2)
    # req_id_2 = chunks2[-1]["id"]
    result_2 = "".join([x["choices"][0]["delta"]["content"] for x in chunks2[:-1]])
    logprobs_2 = extract_logprobs(chunks2)
    speculate_metrics_2 = chunks2[-2]["choices"][0]["speculate_metrics"]
    # entropy_2 = extract_last_entropy("log/data_processor.log", req_id_2)
    # speculate_metrics_2["entropy"] = entropy_2
    print("chunks2:", chunks2[-1])
    print("speculate_metrics_2:", speculate_metrics_2)
    # print("entropy_2:", entropy_2)

    base_path = os.getenv("MODEL_PATH")
    baseline_manager = BaselineManager(base_path)
    # mtp accept ratio
    if os.getenv("BASELINE") == "1":
        baseline_manager.save("base_21b_step3", result)
        baseline_manager.save("base_21b_mtp_metrics_step3", speculate_metrics_2)
        baseline_manager.save("base_21b_logprobs_step3_new", logprobs_2)

    baseline_result = baseline_manager.load("base_21b_step3")
    baseline_mtp_metrics = baseline_manager.load("base_21b_mtp_metrics_step3")
    baseline_logprobs = baseline_manager.load("base_21b_logprobs_step3_new")

    assert logprobs == logprobs_2, (
        "logprobs 前后不一致\n"
        f"logprobs_1: {json.dumps(logprobs, ensure_ascii=False, indent=2)}\n"
        f"logprobs_2: {json.dumps(logprobs_2, ensure_ascii=False, indent=2)}"
    )
    assert baseline_logprobs == logprobs_2, (
        "logprobs 与baseline不一致\n"
        f"logprobs_1: {json.dumps(baseline_logprobs, ensure_ascii=False, indent=2)}\n"
        f"logprobs_2: {json.dumps(logprobs_2, ensure_ascii=False, indent=2)}"
    )
    # assert abs(entropy - entropy_2) < 1e-12, (
    #     "entropy 前后不一致\n"
    #     f"entropy_1: {req_id}:{entropy}\n"
    #     f"entropy_2: {req_id_2}:{entropy_2}"
    # )
    assert speculate_metrics_2 == baseline_mtp_metrics, (
        f"speculate_metrics存在diff，"
        f"speculate_metrics_2: {speculate_metrics_2}\n "
        f"baseline: {baseline_mtp_metrics}"
    )
    assert result == baseline_result, f"与baseline存在diff，result: {result}\n baseline: {baseline_result}"
    assert result_2 == baseline_result, f"与baseline存在diff，result: {result_2}\n baseline: {baseline_result}"

    prompt_tokens = chunks2[-1]["usage"]["prompt_tokens"]
    cached_tokens = chunks2[-1]["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cached_tokens == prompt_tokens // 64 * 64, "cached_tokens数量有问题"
