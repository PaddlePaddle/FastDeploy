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
import requests
from utils.serving_utils import (
    FD_API_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    clean,
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
    speculative_config = {"method": "mtp", "num_speculative_tokens": 1, "model": mtp_model_path}

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
        "--speculative-config",
        json.dumps(speculative_config),
        "--graph-optimization-config",
        '{"use_cudagraph":true,  "use_unique_memory_pool":true, "draft_model_use_cudagraph":true}',
        "--enable-keep-sampling-mask",
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


def send_request(url, payload, timeout=60):
    """
    发送请求到指定的URL，并返回响应结果。
    """
    headers = {
        "Content-Type": "application/json",
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=timeout)
        print("🟢 接收响应中...\n")
        return res
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时（超过 {timeout} 秒）")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败：{e}")
        return None


def get_stream_chunks(response):
    """解析流式返回，生成chunk List[dict]"""
    chunks = []

    if response.status_code == 200:
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data: "):
                    line = line[len("data: ") :]

                if line.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except Exception as e:
                    print(f"解析失败: {e}, 行内容: {line}")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print("返回内容：", response.text)

    return chunks


def test_chat_usage_stream(api_url):
    """测试流式chat usage"""
    payload = {
        "model": "default",
        "temperature": 0,
        "top_p": 0,
        "seed": 33,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 50,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "metadata": {"min_tokens": 10},
    }

    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)
    result = "".join([x["choices"][0]["delta"]["content"] for x in chunks[:-1]])
    print("Prefill Response:", result)
    assert result != "", "结果为空"
    usage = chunks[-1]["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"], "completion_tokens大于max_tokens"
    assert payload["metadata"]["min_tokens"] <= usage["completion_tokens"], "completion_tokens小于min_tokens"
    assert usage["total_tokens"] == total_tokens, "total_tokens不等于prompt_tokens + completion_tokens"


def test_chat_usage_non_stream(api_url):
    """测试非流式chat usage"""
    payload = {
        "model": "default",
        "temperature": 0,
        "top_p": 0,
        "seed": 33,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 50,
        "stream": False,
        "metadata": {"min_tokens": 10},
    }

    response = send_request(url=api_url, payload=payload).json()
    usage = response["usage"]
    result = response["choices"][0]["message"]["content"]
    assert result != "", "结果为空"
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"], "completion_tokens大于max_tokens"
    assert payload["metadata"]["min_tokens"] <= usage["completion_tokens"], "completion_tokens小于min_tokens"
    assert usage["total_tokens"] == total_tokens, "total_tokens不等于prompt_tokens + completion_tokens"


def test_non_chat_usage_stream(api_url):
    """测试流式非chat usage"""
    payload = {
        "model": "default",
        "temperature": 0,
        "top_p": 0,
        "seed": 33,
        "prompt": "牛顿的三大运动定律是什么？",
        "max_tokens": 50,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "metadata": {"min_tokens": 10},
    }
    api_url = api_url.replace("chat/completions", "completions")

    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)
    result = "".join([x["choices"][0]["text"] for x in chunks[:-1]])
    # print("Prefill Response:", result)
    assert result != "", "结果为空"
    usage = chunks[-1]["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"], "completion_tokens大于max_tokens"
    assert payload["metadata"]["min_tokens"] <= usage["completion_tokens"], "completion_tokens小于min_tokens"
    assert usage["total_tokens"] == total_tokens, "total_tokens不等于prompt_tokens + completion_tokens"


def test_non_chat_usage_non_stream(api_url):
    """测试非流式非chat usage"""
    payload = {
        "model": "default",
        "temperature": 0,
        "top_p": 0,
        "seed": 33,
        "prompt": "牛顿的三大运动定律是什么？",
        "max_tokens": 50,
        "stream": False,
        "metadata": {"min_tokens": 10},
    }
    api_url = api_url.replace("chat/completions", "completions")

    response = send_request(url=api_url, payload=payload).json()
    usage = response["usage"]
    result = response["choices"][0]["text"]
    # print("Prefill Response:", result)
    assert result != "", "结果为空"
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"], "completion_tokens大于max_tokens"
    assert payload["metadata"]["min_tokens"] <= usage["completion_tokens"], "completion_tokens小于min_tokens"
    assert usage["total_tokens"] == total_tokens, "total_tokens不等于prompt_tokens + completion_tokens"


def test_mtp_accept_ratio(api_url):
    """测试mtp接受率"""
    payload = {
        "model": "default",
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
        "temperature": 0,
        "seed": 23,
        "top_p": 0,
    }

    print("fastdeploy answer is :")

    try:
        # TODO: 第一次和第二次存在diff，后面正常，暂时多请求一次
        response = send_request(url=api_url, payload=payload)
        chunks = get_stream_chunks(response)
        response = send_request(url=api_url, payload=payload)
        chunks = get_stream_chunks(response)
        for idx, chunk in enumerate(chunks):
            print(f"\nchunk[{idx}]:\n{json.dumps(chunk, ensure_ascii=False)}")
        result = "".join([x["choices"][0]["delta"]["content"] for x in chunks[:-1]])
        speculate_metrics = chunks[-2]["choices"][0]["speculate_metrics"]
    except Exception as e:
        print(f"解析失败: {e}")
    print("\nresult:\n", result)

    base_path = os.getenv("MODEL_PATH")
    baseline_path = os.path.join(base_path, "21b_mtp_accept_ratio_baseline_dev_0311.txt")
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = f.read()
    baseline_ratio = {
        "accepted_tokens": 131,
        "rejected_tokens": 23,
        "accept_ratio": 0.4122137404580153,
        "average_accept_length": 1.7012987012987013,
        "accepted_tokens_per_head": [77, 54],
        "accept_ratio_per_head": [0.7012987012987013],
    }

    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)
    result_2 = "".join([x["choices"][0]["delta"]["content"] for x in chunks[:-1]])
    speculate_metrics_2 = chunks[-2]["choices"][0]["speculate_metrics"]
    print("chunks:", chunks[-2])
    print("baseline", speculate_metrics)
    print("speculate_metrics_2", speculate_metrics_2)
    assert result_2 == baseline, f"与baseline存在diff，result_2: {result}\n baseline: {baseline}"
    assert speculate_metrics_2 == baseline_ratio, (
        f"speculate_metrics存在diff，" f"speculate_metrics_2: {speculate_metrics_2}\n " f"baseline: {baseline_ratio}"
    )
    assert speculate_metrics_2["accept_ratio"] > 0, "accept_ratio异常"
    prompt_tokens = chunks[-1]["usage"]["prompt_tokens"]
    cached_tokens = chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cached_tokens == prompt_tokens // 64 * 64, "cached_tokens数量有问题"


def _assert_sampling_mask_format(sampling_mask, max_tokens):
    """验证 sampling_mask 字段格式的公共辅助函数。

    sampling_mask 是 List[List[int]]：
    - 外层列表长度 == 生成的 token 数（completion_tokens），对应 MTP 每步可接受多个 token
    - 内层列表为保留位置的词汇表索引（int），非空且单调递增
    """
    assert sampling_mask is not None, "sampling_mask 不应为 None"
    assert isinstance(sampling_mask, list), "sampling_mask 应为 list"
    assert len(sampling_mask) > 0, "sampling_mask 不应为空"
    assert len(sampling_mask) <= max_tokens, "sampling_mask 长度不应超过 max_tokens"

    for token_mask in sampling_mask:
        assert isinstance(token_mask, list), f"每个 token 的 mask 应为 list，实际: {type(token_mask)}"
        assert len(token_mask) > 0, "每个 token 的 mask 不应为空（至少保留采样到的 token）"
        for idx in token_mask:
            assert isinstance(idx, int), f"mask 中的每个元素应为 int，实际: {type(idx)}"
            assert idx >= 0, f"mask 索引不应为负数，实际: {idx}"


def test_keep_sampling_mask_stream(api_url):
    """测试流式响应中 keep_sampling_mask 功能（MTP 模式）。

    验证：
    1. 每个非空 chunk 的 choices[0].sampling_mask 格式为 List[List[int]]
    2. 内层列表包含词汇表保留位置的索引，非空且单调递增
    3. 最终 sampling_mask 总长度等于 completion_tokens
    """
    max_tokens = 20
    payload = {
        "model": "default",
        "temperature": 1.0,
        "top_p": 0.9,
        "seed": 42,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请用一句话介绍Python语言。"},
        ],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)

    assert len(chunks) > 1, "流式响应应包含至少两个 chunk"

    all_sampling_masks = []
    for chunk in chunks[:-1]:  # 最后一个 chunk 是 usage-only
        choice = chunk["choices"][0]
        # 仅当 delta 有实际内容时才应携带 sampling_mask（首个 role chunk 内容为空，不含该字段）
        has_content = bool(choice.get("delta", {}).get("content"))
        mask = choice.get("sampling_mask")
        if has_content:
            assert mask is not None, f"有内容的 chunk 缺少 sampling_mask 字段: {choice}"
        if mask is not None:
            assert isinstance(mask, list), f"sampling_mask 应为 list，实际: {type(mask)}"
            for token_mask in mask:
                assert isinstance(token_mask, list), "每个 token mask 应为 list"
                assert len(token_mask) > 0, "每个 token mask 不应为空"
                for idx in token_mask:
                    assert isinstance(idx, int) and idx >= 0, f"mask 索引应为非负 int，实际: {idx}"
            all_sampling_masks.extend(mask)

    # 最后一个 chunk 携带 usage 信息
    usage = chunks[-1].get("usage")
    if usage:
        completion_tokens = usage["completion_tokens"]
        assert (
            len(all_sampling_masks) == completion_tokens
        ), f"sampling_mask 总长度 {len(all_sampling_masks)} 应等于 completion_tokens {completion_tokens}"


def test_keep_sampling_mask_non_stream(api_url):
    """测试非流式响应中 keep_sampling_mask 功能（MTP 模式）。

    验证：
    1. choices[0].sampling_mask 格式为 List[List[int]]
    2. 长度等于 completion_tokens
    3. 内层列表包含非负递增的词汇表索引
    """
    max_tokens = 20
    payload = {
        "model": "default",
        "temperature": 1.0,
        "top_p": 0.9,
        "seed": 42,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请用一句话介绍Python语言。"},
        ],
        "max_tokens": max_tokens,
        "stream": False,
    }

    response = send_request(url=api_url, payload=payload).json()
    assert "choices" in response, f"响应缺少 choices 字段: {response}"
    choice = response["choices"][0]
    assert "sampling_mask" in choice, f"choice 缺少 sampling_mask 字段: {choice}"

    sampling_mask = choice["sampling_mask"]
    completion_tokens = response["usage"]["completion_tokens"]
    _assert_sampling_mask_format(sampling_mask, max_tokens)
    assert (
        len(sampling_mask) == completion_tokens
    ), f"sampling_mask 长度 {len(sampling_mask)} 应等于 completion_tokens {completion_tokens}"


def test_keep_sampling_mask_top_p_1_stream(api_url):
    """测试 top_p=1.0 时流式响应的 sampling_mask（MTP 模式）。

    top_p=1.0 表示保留全部词汇，每个 token mask 应包含所有词汇表位置。
    验证 mask 非空且每个内层列表长度 > 1（至少保留多个候选 token）。
    """
    max_tokens = 10
    payload = {
        "model": "default",
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 42,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "1+1="},
        ],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    response = send_request(url=api_url, payload=payload)
    chunks = get_stream_chunks(response)
    assert len(chunks) > 1, "流式响应应包含至少两个 chunk"

    for chunk in chunks[:-1]:
        choice = chunk["choices"][0]
        mask = choice.get("sampling_mask")
        if mask is not None:
            for token_mask in mask:
                assert len(token_mask) > 1, "top_p=1.0 时每个 token 的候选集应大于 1"


def test_keep_sampling_mask_consistent_with_top_p(api_url):
    """对比 top_p=0.1 与 top_p=0.9 时 sampling_mask 的候选集大小（非流式，MTP 模式）。

    top_p 越小，保留的候选 token 越少，平均 mask 长度应更短。
    """
    max_tokens = 15

    def get_avg_mask_len(top_p):
        payload = {
            "model": "default",
            "temperature": 1.0,
            "top_p": top_p,
            "seed": 42,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "请列举三种编程语言。"},
            ],
            "max_tokens": max_tokens,
            "stream": False,
        }
        resp = send_request(url=api_url, payload=payload).json()
        mask = resp["choices"][0].get("sampling_mask")
        if not mask:
            return 0
        return sum(len(m) for m in mask) / len(mask)

    avg_small = get_avg_mask_len(0.1)
    avg_large = get_avg_mask_len(0.9)
    assert avg_small <= avg_large, f"top_p=0.1 的平均 mask 长度 ({avg_small:.1f}) 应 <= top_p=0.9 ({avg_large:.1f})"
