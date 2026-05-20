# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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


def _build_speculate_metrics_baseline(
    accepted_tokens,
    rejected_tokens,
    accept_ratio,
    average_accept_length,
    accepted_tokens_per_head,
    accept_ratio_per_head,
):
    """
    Build a tolerance-based baseline for speculate metrics.

    Integer counters remain strict, while floating-point fields and
    per-head metric arrays use approximate comparison to reduce
    environment-sensitive test flakiness.
    """
    return {
        "accepted_tokens": accepted_tokens,
        "rejected_tokens": rejected_tokens,
        "accept_ratio": pytest.approx(accept_ratio, abs=0.02),
        "average_accept_length": pytest.approx(average_accept_length, abs=0.1),
        "accepted_tokens_per_head": pytest.approx(accepted_tokens_per_head, abs=2),
        "accept_ratio_per_head": pytest.approx(accept_ratio_per_head, abs=0.05),
    }


BASELINE_SPECULATE_METRICS = _build_speculate_metrics_baseline(
    accepted_tokens=100,
    rejected_tokens=176,
    accept_ratio=0.54,
    average_accept_length=2.1739130434782608,
    accepted_tokens_per_head=[46, 25, 15, 8, 6, 0],
    accept_ratio_per_head=[0.5434782608695652, 0.6, 0.5333333333333333, 0.75, 0.0],
)
BASELINE_SPECULATE_METRICS_WITH_LOGPROBS = _build_speculate_metrics_baseline(
    accepted_tokens=100,
    rejected_tokens=182,
    accept_ratio=0.53,
    average_accept_length=2.127659574468085,
    accepted_tokens_per_head=[47, 29, 16, 5, 3, 0],
    accept_ratio_per_head=[0.6170212765957447, 0.5517241379310345, 0.3125, 0.6, 0.0],
)


def _assert_speculate_metrics_match(actual, baseline, label):
    """Per-field comparison against a tolerance-based baseline.

    Avoids whole-dict ``==`` so that an AssertionError isn't masked by a
    TypeError when json.dumps tries to serialize pytest.approx wrappers in
    the failure message.
    """
    missing = set(baseline) - set(actual)
    extra = set(actual) - set(baseline)
    assert not missing and not extra, (
        f"[{label}] speculate_metrics keys mismatch: missing={missing}, extra={extra}, "
        f"got_keys={sorted(actual.keys())}"
    )
    for key, expected in baseline.items():
        got = actual[key]
        assert got == expected, f"[{label}] field '{key}' mismatch:\n" f"  got:      {got}\n" f"  expected: {expected}"


@pytest.fixture(scope="session", autouse=True)
def setup_and_run_server():
    """
    Pytest fixture that runs once per test session:
    - Cleans ports before tests
    - Starts the API server in hybrid MTP-Ngram mode as a subprocess
    - Waits for server port to open (up to 5 minutes)
    - Tears down server after all tests finish
    """
    print("Pre-test port cleanup...")
    clean()

    if os.path.exists("log") and os.path.isdir("log"):
        shutil.rmtree("log")

    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "ernie-4_5-21b-a3b-bf16-paddle")
    else:
        model_path = "./ernie-4_5-21b-a3b-bf16-paddle"
    mtp_model_path = os.path.join(model_path, "mtp")

    speculative_config = {
        "method": "mtp",
        "model": mtp_model_path,
        "num_speculative_tokens": 5,
        "num_model_steps": 3,
        "mtp_strategy": "with_ngram",
        "max_ngram_size": 3,
        "min_ngram_size": 1,
    }

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
        "1",
        "--engine-worker-queue-port",
        str(FD_ENGINE_QUEUE_PORT),
        "--metrics-port",
        str(FD_METRICS_PORT),
        "--cache-queue-port",
        str(FD_CACHE_QUEUE_PORT),
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "8",
        "--quantization",
        "wint4",
        "--enable-overlap-schedule",
        "--enable-logprob",
        "--speculative-config",
        json.dumps(speculative_config),
        "--graph-optimization-config",
        '{"use_cudagraph":true, "use_unique_memory_pool":true, "draft_model_use_cudagraph":true}',
    ]

    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    for _ in range(300):
        if is_port_open("127.0.0.1", FD_API_PORT):
            print(f"Server is up on port {FD_API_PORT}")
            break
        time.sleep(1)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            clean()
        except Exception as e:
            print(f"Failed to kill process group: {e}")
        raise RuntimeError(f"API server did not start on port {FD_API_PORT}")

    yield

    print("\n===== Post-test server cleanup... =====")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        clean()
        print(f"server (pid={process.pid}) terminated")
    except Exception as e:
        print(f"Failed to terminate API server: {e}")


@pytest.fixture(scope="session")
def api_url():
    return f"http://0.0.0.0:{FD_API_PORT}/v1/chat/completions"


@pytest.fixture(scope="session")
def metrics_url():
    return f"http://0.0.0.0:{FD_METRICS_PORT}/metrics"


def test_mtp_ngram_stream(api_url):
    """Hybrid MTP-Ngram streaming generation returns non-empty result with valid token counts."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "牛顿的三大运动定律是什么？"}],
        "max_tokens": 50,
        "min_tokens": 10,
        "temperature": 0,
        "top_p": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)
    result = "".join(x["choices"][0]["delta"]["content"] for x in chunks[:-1])
    assert result != "", "Generation result is empty"
    usage = chunks[-1]["usage"]
    assert usage["completion_tokens"] <= payload["max_tokens"]
    assert usage["completion_tokens"] >= payload["min_tokens"]
    assert usage["total_tokens"] == usage["completion_tokens"] + usage["prompt_tokens"]


def test_mtp_ngram_non_stream(api_url):
    """Hybrid MTP-Ngram non-streaming generation returns non-empty result with valid token counts."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "牛顿的三大运动定律是什么？"}],
        "max_tokens": 50,
        "min_tokens": 10,
        "temperature": 0,
        "top_p": 0,
        "seed": 42,
        "stream": False,
    }
    response = send_request(url=api_url, payload=payload).json()
    result = response["choices"][0]["message"]["content"]
    assert result != "", "Generation result is empty"
    usage = response["usage"]
    assert usage["completion_tokens"] <= payload["max_tokens"]
    assert usage["completion_tokens"] >= payload["min_tokens"]
    assert usage["total_tokens"] == usage["completion_tokens"] + usage["prompt_tokens"]


def test_mtp_ngram_speculate_metrics(api_url):
    """speculate_metrics contains the MTP + ngram acceptance stats for a repeated-prompt request."""
    # Prompt with repeated fragments to increase ngram match rate
    content = (
        "国外项目风险管理研究起步较早，理论体系成熟。早期研究集中于保险与金融领域，后逐步扩展至工程项目、"
        "公共管理等多领域。在理论层面，COSO《企业风险管理——整合框架》和ISO31000标准为风险管理提供了系统性"
        "指导，强调风险识别、评估、应对与监控的全流程管理。风险识别方法包括故障树分析、事件树分析等；风险评估"
        "则广泛应用VaR模型、蒙特卡洛模拟等量化工具。应对策略涵盖规避、转移、减轻和接受等，并衍生出风险共享、"
        "升级等复杂策略。此外，组织文化、管理层支持等因素对风险管理有效性影响显著。近年来，随着科技发展，"
        "人工智能、大数据等技术被引入风险管理，推动其向智能化、自动化方向发展。请介绍一下国外关于项目风险管理"
        "的文献研究综述，300字以内"
    )
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 100,
        "min_tokens": 20,
        "temperature": 0,
        "top_p": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)
    # chunks[-1] is the usage chunk; chunks[-2] is the last content chunk containing speculate_metrics
    speculate_metrics = chunks[-2]["choices"][0].get("speculate_metrics")
    # print(f"\n[test_mtp_ngram_speculate_metrics] speculate_metrics: {json.dumps(speculate_metrics, indent=2)}")
    assert speculate_metrics is not None, "speculate_metrics is missing from last chunk"

    # accepted_tokens_per_head length should equal num_speculative_tokens + 1 (head 0 is the verified token).
    # num_speculative_tokens=5 → 6 entries.
    accepted_per_head = speculate_metrics.get("accepted_tokens_per_head")
    assert accepted_per_head is not None, "accepted_tokens_per_head is missing"
    assert len(accepted_per_head) == 6, (
        f"Expected 6 entries in accepted_tokens_per_head (num_speculative_tokens=5 + 1), "
        f"got {len(accepted_per_head)}: {accepted_per_head}"
    )
    # Monotonically non-increasing (each subsequent draft head is harder to accept)
    for i in range(len(accepted_per_head) - 1):
        assert (
            accepted_per_head[i] >= accepted_per_head[i + 1]
        ), f"accepted_tokens_per_head is not monotonically non-increasing at index {i}: {accepted_per_head}"

    # Cross-field consistency: total accepted == sum of per-head
    assert speculate_metrics["accepted_tokens"] == sum(accepted_per_head), (
        f"accepted_tokens ({speculate_metrics['accepted_tokens']}) != "
        f"sum(accepted_tokens_per_head) ({sum(accepted_per_head)})"
    )

    # Baseline comparison (tolerance-based) against values captured in the reference environment.
    if BASELINE_SPECULATE_METRICS is not None:
        _assert_speculate_metrics_match(
            speculate_metrics, BASELINE_SPECULATE_METRICS, label="test_mtp_ngram_speculate_metrics"
        )


def test_mtp_ngram_speculate_metrics_with_logprobs(api_url):
    """speculate_metrics and logprobs coexist correctly when hybrid mode + logprobs are both enabled."""
    content = (
        "国外项目风险管理研究起步较早，理论体系成熟。早期研究集中于保险与金融领域，后逐步扩展至工程项目、"
        "公共管理等多领域。在理论层面，COSO《企业风险管理——整合框架》和ISO31000标准为风险管理提供了系统性"
        "指导，强调风险识别、评估、应对与监控的全流程管理。风险识别方法包括故障树分析、事件树分析等；风险评估"
        "则广泛应用VaR模型、蒙特卡洛模拟等量化工具。应对策略涵盖规避、转移、减轻和接受等，并衍生出风险共享、"
        "升级等复杂策略。此外，组织文化、管理层支持等因素对风险管理有效性影响显著。近年来，随着科技发展，"
        "人工智能、大数据等技术被引入风险管理，推动其向智能化、自动化方向发展。请介绍一下国外关于项目风险管理"
        "的文献研究综述，300字以内"
    )
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 100,
        "min_tokens": 20,
        "temperature": 0,
        "top_p": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "logprobs": True,
        "top_logprobs": 5,
    }
    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)

    # logprobs are present in each content chunk
    logprobs_list = extract_logprobs(chunks)
    assert len(logprobs_list) > 0, "No logprobs received"
    for logprobs in logprobs_list:
        assert "content" in logprobs
        for item in logprobs["content"]:
            assert "token" in item
            assert "logprob" in item
            assert "top_logprobs" in item
            assert len(item["top_logprobs"]) <= 5

    # speculate_metrics appears in the last content chunk and is consistent
    speculate_metrics = chunks[-2]["choices"][0].get("speculate_metrics")
    # print(
    #     f"\n[test_mtp_ngram_speculate_metrics_with_logprobs] "
    #     f"speculate_metrics: {json.dumps(speculate_metrics, indent=2)}"
    # )
    assert speculate_metrics is not None, "speculate_metrics is missing from last chunk"

    accepted_per_head = speculate_metrics.get("accepted_tokens_per_head")
    assert accepted_per_head is not None, "accepted_tokens_per_head is missing"
    assert len(accepted_per_head) == 6
    assert speculate_metrics["accepted_tokens"] == sum(accepted_per_head)

    # Baseline comparison (tolerance-based) against values captured in the reference environment.
    if BASELINE_SPECULATE_METRICS_WITH_LOGPROBS is not None:
        _assert_speculate_metrics_match(
            speculate_metrics,
            BASELINE_SPECULATE_METRICS_WITH_LOGPROBS,
            label="test_mtp_ngram_speculate_metrics_with_logprobs",
        )
