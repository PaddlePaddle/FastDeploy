"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

# This file is modified from https://github.com/vllm-project/vllm/blob/main/benchmarks/backend_request_func.py


import asyncio
import copy
import io
import json
import logging
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=int(os.environ.get("AIOHTTP_TIMEOUT", 6 * 60 * 60)))


def _get_control_url(api_url: str, endpoint: str) -> str:
    parsed = urlsplit(api_url)
    return urlunsplit((parsed.scheme, parsed.netloc, endpoint, "", ""))


async def _close_session(session: aiohttp.ClientSession, api_url: str, session_id: str) -> None:
    url = _get_control_url(api_url, "/close_session")
    async with session.post(url=url, json={"session_id": session_id}) as response:
        if response.status != 200:
            raise RuntimeError(
                f"Failed to close session {session_id}: {response.status} {await response.text()}"
            )


@dataclass
class RequestFuncInput:
    """Input for requesting LLMs via API"""

    no: int
    prompt: str
    history_QA: Optional[dict]
    hyper_parameters: dict
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None
    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False
    language: Optional[str] = None
    debug: bool = False
    pd_metrics: bool = False
    response_format: Optional[dict] = None
    random_flag: bool = False
    json_data: Optional[dict] = None
    prompt_token_ids: Optional[list] = None
    tokenizer_model: str = None
    tokenizer_path: str = None
    stream: bool = True
    session_id: Optional[str] = None
    turn_idx: Optional[int] = None
    enable_session_control: bool = False
    session_control_url: Optional[str] = None


@dataclass
class RequestFuncOutput:
    """Output for requesting LLMs via API"""

    no: int = 0
    request_id: str = ""
    generated_text: str = ""
    reasoning_content: str = ""
    success: bool = False
    has_arrival_time: bool = False
    latency: float = 0.0
    end_timestamp: float = 0.0  # 模型完全返回的时间戳（秒, perf_counter基准）
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    arrival_time: list = field(default_factory=list)  # arrival_time
    itl: list = field(default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    prompt_tokens: int = 0  # 推理侧返回输入token数
    reasoning_tokens: int = 0  # 思考长度
    res_ttft: int = 0  # 包含思考首token时延
    error: str = ""
    metrics: dict = field(default_factory=dict)
    tool_calls: list = field(default_factory=list)
    output_ids: list = field(default_factory=list)
    # 本轮请求之前的环境交互等待时长（秒），来自数据集里对应 tool 消息的 env_interact_time
    env_wait_time: float = 0.0


@dataclass
class SessionMetrics:
    """多轮对话指标"""

    session_no: int
    session_e2e_time: float
    pure_llm_time: float
    input_tokens: int
    output_tokens: int
    tool_calls: int


def safe_cost(a, b):
    """时间差计算"""
    if a is None or b is None:
        return None
    return a - b


def metrics_summary(metrics, token_timestamps):
    """Summarize metrics"""
    if not metrics or len(token_timestamps) < 2:
        return {}

    m0 = metrics[0]
    m_last = metrics[-1]

    summary = {}

    arrival_time = m0.get("arrival_time")
    inference_start_time = m0.get("inference_start_time")

    # prefill 总耗时
    summary["prefill_cost_time"] = safe_cost(m0.get("send_request_output_to_decode_time"), arrival_time)
    # prefill准备总耗时
    summary["prefill_prepare_cost_time"] = safe_cost(inference_start_time, arrival_time)
    # 预处理耗时
    summary["preprocess_cost_time"] = safe_cost(m0.get("scheduler_recv_req_time"), arrival_time)
    # 请求缓存耗时
    summary["cache_in_scheduler_cost_time"] = safe_cost(
        m0.get("engine_get_req_time"), m0.get("scheduler_recv_req_time")
    )
    # 申请 decode资源耗时
    summary["ask_decode_resource_cost_time"] = safe_cost(
        m0.get("ask_decode_resource_finish_time"), m0.get("ask_decode_resource_start_time")
    )
    # scheduler调度耗时
    summary["schedule_cost_time"] = safe_cost(
        m0.get("inference_start_time"), m0.get("ask_decode_resource_finish_time")
    )
    # prefill 的首 token 推理耗时
    summary["prefill_first_token_infer_cost_time"] = safe_cost(
        m0.get("engine_recv_first_token_time"), inference_start_time
    )
    # prefill 等待 cache 传输耗时
    summary["wait_sending_cache_cost_time"] = safe_cost(
        m0.get("send_request_output_to_decode_time"), m0.get("wait_for_sending_cache_time")
    )
    # decode分配资源耗时
    summary["decode_preallocate_cost_time"] = safe_cost(
        m_last.get("decode_preallocate_req_time"), m_last.get("decode_recv_req_time")
    )
    # decode准备推理耗时
    summary["decode_prepare_cost_time"] = safe_cost(
        m_last.get("decode_inference_start_time"), m_last.get("decode_recv_first_token_time")
    )
    # decode次token推理耗时
    summary["decode_second_token_infer_cost_time"] = safe_cost(
        m_last.get("decode_recv_second_token_time"), m_last.get("decode_inference_start_time")
    )
    # 返回首 token 链路耗时
    summary["first_token_transmission_cost_time"] = safe_cost(
        token_timestamps[0], m_last.get("decode_recv_first_token_time")
    )
    # 返回次 token 链路耗时
    summary["second_token_transmission_cost_time"] = safe_cost(
        token_timestamps[1], m_last.get("decode_recv_second_token_time")
    )

    # MIX 模式下，scheduler调度耗时
    summary["mixed_schedule_cost_time"] = safe_cost(m0.get("inference_start_time"), m0.get("engine_get_req_time"))
    # MIX 模式下，返回首 token 链路耗时
    summary["mixed_first_token_transmission_cost_time"] = safe_cost(
        token_timestamps[0], m0.get("engine_recv_first_token_time")
    )

    summary["gpu_cache_token_num"] = m0.get("gpu_cache_token_num")
    summary["cpu_cache_token_num"] = m0.get("cpu_cache_token_num")
    summary["storage_cache_token_num"] = m0.get("storage_cache_token_num")
    summary["cpu_cache_prepare_time"] = m0.get("cpu_cache_prepare_time")
    summary["storage_cache_prepare_time"] = m0.get("storage_cache_prepare_time")

    return summary


def load_tokenizer(model, actor_tokenizer_path):
    """加载tokenizer"""
    from ernie_tokenizer import Ernie5Tokenizer, ErnieBotTokenizer
    from paddleformers.transformers import AutoTokenizer

    from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer

    vocab_file_names = ["tokenizer.model", "spm.model", "ernie_token_100k.model"]

    try:
        if model == "eb":
            for i in range(len(vocab_file_names)):
                if os.path.exists(os.path.join(actor_tokenizer_path, vocab_file_names[i])):
                    ErnieBotTokenizer.resource_files_names["vocab_file"] = vocab_file_names[i]
                    break
            tokenizer = ErnieBotTokenizer.from_pretrained(actor_tokenizer_path)
        elif model == "eb_mm":
            for vocab_file in vocab_file_names:
                full_path = os.path.join(actor_tokenizer_path, vocab_file)
                if os.path.exists(full_path):
                    Ernie4_5Tokenizer.resource_files_names["vocab_file"] = vocab_file
            # for i in range(len(vocab_file_names)):
            #     if os.path.exists(os.path.join(actor_tokenizer_path, vocab_file_names[i])):
            #         Ernie45Tokenizer.resource_files_names["vocab_file"] = vocab_file_names[i]
            #         break
            tokenizer = Ernie4_5Tokenizer.from_pretrained(actor_tokenizer_path)
            # tokenizer.ignored_index = -100
        elif model == "eb5":
            for i in range(len(vocab_file_names)):
                if os.path.exists(os.path.join(actor_tokenizer_path, vocab_file_names[i])):
                    Ernie5Tokenizer.resource_files_names["vocab_file"] = vocab_file_names[i]
                    break
            tokenizer = Ernie5Tokenizer.from_pretrained(actor_tokenizer_path)
        else:
            print("tokenizer: AUTO")
            tokenizer = AutoTokenizer.from_pretrained(actor_tokenizer_path, padding_side="left", use_fast=True)
    except Exception as e:
        tokenizer = None
        logging.warning(f"Load tokenizer error: {e}")

    return tokenizer


async def handle_non_stream_response(
    response,
    output,
    st,
):
    """
    处理非流式返回
    """
    text = await response.text()

    timestamp = time.perf_counter()
    data = json.loads(text)
    # print("data:", data)

    request_id = data.get("id", "None")

    usage = data.get("usage", {})

    output.output_tokens = usage.get("completion_tokens", 0)
    output.prompt_tokens = usage.get("prompt_tokens", 0)

    if output.prompt_len == 0:
        if usage.get("prompt_tokens_details", {}):
            output.prompt_len = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

    choices = data.get("choices", [])

    if choices:
        message = choices[0].get("message", {})

        output.generated_text = message.get("content", "") or ""
        output.reasoning_content = message.get("reasoning_content", "") or ""

        completion_token_ids = message.get("completion_token_ids", [])
        if completion_token_ids:
            output.output_ids.extend(completion_token_ids)

        # tool calls
        tool_calls = message.get("tool_calls") or []

        for tc in tool_calls:
            func = tc.get("function", {})

            try:
                args = json.loads(func.get("arguments", "{}"))
            except Exception:
                args = {}

            output.tool_calls.append(
                {
                    "id": tc.get("id"),
                    "name": func.get("name"),
                    "arguments": args,
                }
            )

    latency = timestamp - st

    # 非流式没有ttft
    output.ttft = latency
    output.res_ttft = latency

    output.end_timestamp = timestamp
    output.latency = latency
    # 非流式没有stream chunk
    # 非流式兼容stream benchmark逻辑
    # arrival_time:
    output.arrival_time = []

    has_text = bool(output.generated_text) or bool(output.reasoning_content)

    has_tool = bool(output.tool_calls)

    if not has_text and not has_tool:
        output.success = False
        output.error = "No generated text found!"
    else:
        output.success = True

    return data, request_id


async def async_request_eb_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
    session: aiohttp.ClientSession | None = None,
) -> RequestFuncOutput:
    """Request an LLM using EB OpenAI"""
    api_url = request_func_input.api_url
    assert api_url.endswith(("completions", "profile")), "OpenAI Chat Completions API URL must end with 'completions'."

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession(
            trust_env=True,
            read_bufsize=10 * 1024 * 1024,
            timeout=AIOHTTP_TIMEOUT,
        )

    content = [{"type": "text", "text": request_func_input.prompt}]
    if request_func_input.multi_modal_content:
        content.append(request_func_input.multi_modal_content)
    # print("######json_data:", request_func_input.json_data)
    payload = {
        "model": request_func_input.model,
        "messages": request_func_input.history_QA,
        "stream": request_func_input.stream,
        "collect_metrics": request_func_input.pd_metrics,
    }

    if request_func_input.output_len is not None:
        payload["max_tokens"] = request_func_input.output_len

    # 流式模式返回usage
    if request_func_input.stream:
        payload["stream_options"] = {
            "include_usage": True,
            "continuous_usage_stats": True,
            # sglang 扩展：解析器攒批时补发只带 usage 的空 delta 包，否则这些
            # token 会被算进 TTFT，解码速度被高估。不支持的服务端会忽略该字段。
            "step_usage_chunks": "first_token",
        }
    if request_func_input.json_data:
        json_data = request_func_input.json_data

        if json_data.get("max_tokens"):
            payload["max_tokens"] = json_data["max_tokens"]

        if json_data.get("min_tokens"):
            payload["min_tokens"] = json_data["min_tokens"]
    if request_func_input.response_format:
        payload["response_format"] = request_func_input.response_format

    # Random-length input/output knob.
    if request_func_input.random_flag:
        payload["max_tokens"] = request_func_input.output_len
        payload["min_tokens"] = request_func_input.output_len

    # When the prompt is a list of token ids, route through prompt_token_ids
    # regardless of random_flag.
    if isinstance(request_func_input.prompt, list):
        request_func_input.prompt_token_ids = request_func_input.prompt
        request_func_input.prompt = ""

    # 支持传入prompt_token_ids
    if request_func_input.prompt_token_ids:
        # 不走messages
        payload["messages"] = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        payload["prompt_token_ids"] = request_func_input.prompt_token_ids
        payload["return_token_ids"] = True
        # print("use_token_ids:", payload)

    # 超参由yaml传入
    payload.update(request_func_input.hyper_parameters)

    # tools信息，yaml优先级最高
    json_data = request_func_input.json_data or {}
    hyper = request_func_input.hyper_parameters or {}

    tools = None
    tool_choice = None

    if hyper.get("tools"):
        tools = hyper.get("tools")
        tool_choice = hyper.get("tool_choice", "auto")
    elif json_data.get("tools"):
        tools = json_data.get("tools")
        tool_choice = json_data.get("tool_choice", "auto")

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    # 随机输入开关
    if request_func_input.random_flag:
        payload["max_tokens"] = request_func_input.output_len
        metadata = payload.get("metadata", {})
        metadata["min_tokens"] = request_func_input.output_len
        payload["metadata"] = metadata

    if request_func_input.ignore_eos:
        payload["ignore_eos"] = request_func_input.ignore_eos

    if request_func_input.session_id:
        payload["session_id"] = request_func_input.session_id

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }

    if request_func_input.session_id is not None:
        headers["X-SMG-Routing-Key"] = f"{request_func_input.session_id}"
    if request_func_input.session_id is not None and request_func_input.turn_idx is not None:
        headers["X-Request-Id"] = f"{request_func_input.session_id}:{request_func_input.turn_idx}"

    output = RequestFuncOutput()
    output.prompt_len = 0
    output.no = request_func_input.no
    payload["no"] = request_func_input.no
    if request_func_input.debug:
        print(f"payload:{json.dumps(payload, ensure_ascii=False)}")
    metrics_list = []
    request_id = "None"

    ttft = 0.0
    res_ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    last_chunk_timestamp = st
    token_timestamps = []
    tool_call_buffer = {}
    # 用于 buffer burst 修正：累计上一次"真增量 chunk"结束时服务端已输出的 token 数。
    last_output_len = 0
    try:
        async with session.post(url=api_url, json=payload, headers=headers, read_bufsize=10 * 1024 * 1024) as response:
            data = {}
            if response.status == 200:
                # 默认流式模式
                if request_func_input.stream:
                    # Reader loop 保持极简：只记录收包时间和原始 chunk，避免解析/拼接影响 ITL。
                    stream_chunks = []
                    async for chunk_bytes in response.content:
                        timestamp = time.perf_counter()
                        wall_timestamp = time.time()
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        if chunk_bytes in (b"data: [DONE]", b"[DONE]"):
                            break
                        stream_chunks.append((chunk_bytes, timestamp, wall_timestamp))

                    generated_text_parts = []
                    reasoning_content_parts = []
                    for chunk_bytes, timestamp, wall_timestamp in stream_chunks:
                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk == "[DONE]":
                            continue
                        # print("####chunk:", chunk, type(chunk))
                        data = json.loads(chunk)

                        # 新增：捕获服务端流式 error
                        if "error" in data:
                            err = data["error"]

                            output.success = False
                            output.error = err.get("message", str(err))

                            # 可选：保存更多信息
                            output.error_type = err.get("type")
                            output.error_code = err.get("code")

                            print("####server error:", json.dumps(err, ensure_ascii=False))

                            break
                        # print("####data:", json.dumps(data, indent=2, ensure_ascii=False))

                        if "metrics" in data:
                            metrics_list.append(data["metrics"])

                        if request_id == "None" and "id" in data:
                            request_id = data["id"]

                        if choices := data.get("choices"):
                            content = choices[0]["delta"].get("content")
                            reason_content = choices[0]["delta"].get("reasoning_content")
                            tool_calls = choices[0]["delta"].get("tool_calls")
                            completion_token_ids = choices[0]["delta"].get("completion_token_ids", [])
                            # 服务端已生成的累计 token 数：优先用 usage(continuous_usage_stats)，
                            # 否则退化为按 token_ids 长度推断。
                            cur_completion_tokens = (data.get("usage") or {}).get("completion_tokens")
                            if cur_completion_tokens is None and completion_token_ids:
                                cur_completion_tokens = len(output.output_ids) + len(completion_token_ids)
                            has_token_chunk = bool(content or reason_content or tool_calls or completion_token_ids)
                            # reasoning/tool-call parser 攒批时 delta 为空，服务端只发 usage 心跳包
                            # （sglang 侧开关 stream_options.step_usage_chunks）。token 数增长
                            # 就是一次真实的 token 到达，必须计入 TTFT/ITL，否则被攒批吞掉的 token
                            # 会被算进 TTFT，解码区间被压缩、解码速度虚高。
                            if not has_token_chunk and cur_completion_tokens is not None:
                                has_token_chunk = cur_completion_tokens > last_output_len
                            if tool_calls:
                                for tc in tool_calls:
                                    idx = tc.get("index", 0)

                                    if idx not in tool_call_buffer:
                                        tool_call_buffer[idx] = {
                                            "id": tc.get("id"),
                                            "name": "",
                                            "arguments": "",
                                        }

                                    if tc.get("id"):
                                        tool_call_buffer[idx]["id"] = tc["id"]

                                    func = tc.get("function", {})

                                    if func.get("name"):
                                        tool_call_buffer[idx]["name"] = func["name"]

                                    if func.get("arguments"):
                                        tool_call_buffer[idx]["arguments"] += func["arguments"]

                            # 过滤 role / finish / usage 等空包，只用真正 token 包统计 TTFT/ITL。
                            if has_token_chunk:
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft
                                    # cached_tokens
                                    usage = data.get("usage") or {}

                                    if usage.get("prompt_tokens_details"):
                                        output.prompt_len = usage.get("prompt_tokens_details", {}).get(
                                            "cached_tokens", 0
                                        )
                                    else:
                                        output.prompt_len = 0

                                    # 首 token 也要更新 last_output_len，用于后续 burst 摊分
                                    if cur_completion_tokens is not None:
                                        last_output_len = cur_completion_tokens
                                    else:
                                        last_output_len = 1

                                # Decoding phase
                                else:
                                    # buffer burst 修正：如果服务端把多个 token 合并到同一个流式 chunk 里，
                                    # 直接把整段间隔记成单个 ITL 会高估解码间隔。这里参考 sglang 官方
                                    # bench_serving 的做法：用真实 token 增量摊分该 chunk 的等待时间。
                                    chunk_gap = timestamp - most_recent_timestamp
                                    if cur_completion_tokens is not None:
                                        num_new_tokens = cur_completion_tokens - last_output_len
                                        if num_new_tokens <= 0:
                                            # 没有真实新 token（罕见的 usage/状态包），跳过 ITL 记录
                                            most_recent_timestamp = timestamp
                                            continue
                                        adjust_itl = chunk_gap / num_new_tokens
                                        output.itl.extend([adjust_itl] * num_new_tokens)
                                        last_output_len = cur_completion_tokens
                                    else:
                                        # 拿不到 completion_tokens 时退化为旧逻辑
                                        output.itl.append(chunk_gap)

                                most_recent_timestamp = timestamp
                                token_timestamps.append(wall_timestamp)

                            # response首token
                            if res_ttft == 0.0:
                                if content:
                                    res_ttft = choices[0].get("arrival_time", timestamp - st)
                                    output.res_ttft = res_ttft
                                    usage = data.get("usage") or {}
                                    output.reasoning_tokens = max(usage.get("completion_tokens", 0) - 1, 0)

                            if content:
                                generated_text_parts.append(content)
                            if reason_content:
                                reasoning_content_parts.append(reason_content)
                            if completion_token_ids:
                                output.output_ids.extend(completion_token_ids)
                            # print(f"####content:{data}")
                            arrival = choices[0].get("arrival_time")
                            if arrival is not None and has_token_chunk:
                                output.has_arrival_time = True
                                output.arrival_time.append(arrival)
                        elif usage := data.get("usage", {}):
                            output.output_tokens = usage.get("completion_tokens", 0)
                            output.prompt_tokens = usage.get("prompt_tokens", 0)
                            prompt_tokens_details = usage.get("prompt_tokens_details") or {}
                            if output.prompt_len == 0:
                                output.prompt_len = prompt_tokens_details.get("cached_tokens", 0)
                            # 收尾 usage 包：把最后一批"生成了但没随 delta 下发"的 token 补进 ITL，
                            # 否则分子(总 token)覆盖不到分母(ITL 区间)，解码速度会被高估。
                            tail_tokens = output.output_tokens - last_output_len
                            if tail_tokens > 0 and ttft > 0.0:
                                tail_gap = timestamp - most_recent_timestamp
                                if tail_gap > 0:
                                    output.itl.extend([tail_gap / tail_tokens] * tail_tokens)
                                last_output_len = output.output_tokens
                                most_recent_timestamp = timestamp
                                token_timestamps.append(wall_timestamp)

                        last_chunk_timestamp = timestamp

                    output.generated_text = "".join(generated_text_parts)
                    output.reasoning_content = "".join(reasoning_content_parts)
                    # 在流式结束时，记录最后一个非 DONE chunk 收到的时间戳
                    output.end_timestamp = last_chunk_timestamp
                    # 截断case
                    usage = data.get("usage", {})
                    output.output_tokens = usage.get("completion_tokens", 0)
                    output.prompt_tokens = usage.get("prompt_tokens", 0)
                    if output.prompt_len == 0:
                        prompt_details = usage.get("prompt_tokens_details") or {}
                        output.prompt_len = prompt_details.get("cached_tokens", 0)

                    if tool_call_buffer:
                        for _, tc in tool_call_buffer.items():
                            try:
                                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                            except:
                                args = {}

                            output.tool_calls.append({"id": tc["id"], "name": tc["name"], "arguments": args})

                    # 如果没有thinking内容，则response首token等于ttft
                    if not output.reasoning_content:
                        output.res_ttft = output.ttft
                    # 新增metrics统计，计算首token过滤空包
                    output.metrics = metrics_summary(metrics_list, token_timestamps[1:])

                    has_text = bool(output.generated_text) or bool(output.reasoning_content)
                    has_tool = getattr(output, "tool_calls", None)

                    # 如果前面已经有服务端错误，保留原错误
                    if output.error:
                        output.success = False
                    # 兼容思考内容超长截断的情况，此时回复内容为空
                    elif not has_text and not has_tool:
                        output.success = False
                        output.reasoning_tokens = output.output_tokens
                        output.error = "No generated text found!"
                    else:
                        output.success = True
                    output.latency = most_recent_timestamp - st
                else:
                    # 非流式模式
                    data, request_id = await handle_non_stream_response(
                        response=response,
                        output=output,
                        st=st,
                    )
            else:
                error_text = await response.text()
                print(
                    "####error response:",
                    error_text,
                    "####payload:",
                    payload,
                )
                output.error = error_text or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
    finally:
        if own_session:
            await session.close()

    output.request_id = request_id

    # 保存失败请求结果
    if not output.success or output.output_tokens == 0:
        with open("error_output.txt", "a") as f:
            f.write(str(output) + "\n")
        print("####error response:", output)
    if pbar:
        pbar.update(1)
    if request_func_input.debug:
        print("#####final_output:", output)
    return output


async def simple_tool_call(model_output, tool_url: str, timeout=60):
    """调用工具函数"""

    import httpx

    tool_id = None

    if getattr(model_output, "tool_calls", None):
        tc = model_output.tool_calls[0]
        tool_name = tc["name"]
        args = tc.get("arguments", {})
        tool_id = tc.get("id")
    else:
        # 取消正则逻辑
        return "", False, "", tool_id
        # match = re.search(r"<tool_call>(.*?)</tool_call>", model_output.generated_text, re.S)
        # if not match:
        #     return "", False, "", tool_id
        #
        # block = match.group(1).strip()
        # lines = block.splitlines()
        # tool_name = lines[0].strip()
        #
        # key = re.search(r"<arg_key>(.*?)</arg_key>", block)
        # val = re.search(r"<arg_value>(.*?)</arg_value>", block)
        #
        # args = {key.group(1): val.group(1)} if key and val else {}

    if not tool_name:
        return "", False, "", tool_id

    headers = {"Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                tool_url,
                headers=headers,
                json={"tool_name": tool_name, "arguments": args},
            )

        resp.raise_for_status()
        obj = resp.json()

        return obj.get("result", resp.text), "result" in obj, tool_name, tool_id

    except Exception as e:
        print(f"[TOOL ERROR] {tool_name}: {repr(e)}")
        return str(e), False, tool_name, tool_id


async def _async_request_eb_openai_chat_completions_multi_turn(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
):
    # 只有显式指定 enable_tools=True 时才走工具调用逻辑，否则走SWE模式（直接用数据集拼接多轮）
    json_data = request_func_input.json_data or {}
    hyper = request_func_input.hyper_parameters or {}
    enable_tools = bool(json_data.get("enable_tools") or hyper.get("enable_tools"))

    outputs = []

    tool_call_count = 0
    llm_time = 0.0
    tool_time = 0.0
    input_tokens = 0
    output_tokens = 0

    ori_history = request_func_input.history_QA
    user_count = sum(msg.get("role") == "user" for msg in ori_history)
    print("START", request_func_input.no, "user对话轮数:", user_count, flush=True)
    history = []
    prompt_no = 0
    max_prompt_len = (
        hyper.get("max_prompt_len") if hyper.get("max_prompt_len") is not None else json_data.get("max_prompt_len")
    )
    print("max_prompt_len:", max_prompt_len)
    input_ids_all = []
    # FD每轮 completion_token_ids
    output_ids = []
    use_token_ids = bool(request_func_input.tokenizer_model and request_func_input.tokenizer_path)
    tokenizer = None

    if use_token_ids:
        print("token ids 拼接模式")
        enable_tools = False
        print("tokenizer_model:", request_func_input.tokenizer_model)
        print("tokenizer_path:", request_func_input.tokenizer_path)
        tokenizer = load_tokenizer(
            request_func_input.tokenizer_model,
            request_func_input.tokenizer_path,
        )
    else:
        print("messages 明文拼接模式")

    # 只创建一次 session
    session_start = time.perf_counter()
    session_uuid = uuid.uuid4().hex
    connector = aiohttp.TCPConnector(
        limit=0,
        limit_per_host=0,
        keepalive_timeout=60,
    )

    async with aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        read_bufsize=10 * 1024 * 1024,
        timeout=AIOHTTP_TIMEOUT,
    ) as session:
        for i, message in enumerate(ori_history):
            if message["role"] == "user" or message["role"] == "tool":
                # 模拟真实 rollout 里工具执行 / 环境交互的等待时间：
                # 数据集里 tool 消息可能带 env_interact_time（秒）， 表示上一步 assistant 生成完到 tool 结果返回的耗时。
                turn_env_wait = 0.0
                env_delay = message.get("env_interact_time") if isinstance(message, dict) else None
                if env_delay and os.environ.get("DISABLE_ENV_TOOL_WAIT", "").lower() not in ("1", "true", "yes", "on"):
                    try:
                        delay_sec = float(env_delay)
                    except (TypeError, ValueError):
                        delay_sec = 0.0
                    if delay_sec > 0:
                        turn_env_wait = delay_sec
                        await asyncio.sleep(delay_sec)

                history.append(message)
                round_input = copy.deepcopy(request_func_input)
                round_input.history_QA = history
                round_input.no = f"{round_input.no}_{prompt_no}"
                round_input.session_id = f"{session_uuid}:{request_func_input.no}"
                round_input.turn_idx = prompt_no
                if use_token_ids:
                    if len(input_ids_all) == 0:
                        # 拼接token_ids模式，首轮token_ids
                        spliced_text = tokenizer.apply_chat_template(
                            history,
                            tokenize=False,
                            split_special_tokens=False,
                            add_special_tokens=False,
                        )
                        # 转换为token ids
                        tokens = tokenizer.tokenize(spliced_text)
                        prompt_token_ids = tokenizer.convert_tokens_to_ids(tokens)
                        input_ids_all.extend(prompt_token_ids)
                        round_input.prompt_token_ids = input_ids_all
                    else:
                        prompt_length = len(input_ids_all) + len(output_ids)
                        if max_prompt_len and prompt_length >= max_prompt_len:
                            # 超长截断
                            print(
                                f"[SESSION STOP] {round_input.no} reach max_prompt_len={max_prompt_len}, stop session"
                            )
                            break
                        # 拼接token_ids模式，后续轮
                        input_ids_all.extend(output_ids)
                        user_prompt = message["content"]
                        # 拼接user_prompt
                        if round_input.tokenizer_model == "eb5":
                            # EB5模型
                            user_prompt = (
                                f"\n\n<|im_start|>user\n{user_prompt}<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
                            )
                        else:
                            # 0.3B模型,2 </s>，拼接时会被替换成100272 <|end_of_sentence|>
                            input_ids_all[-1] = 100272
                            user_prompt = f"User: {user_prompt}\nAssistant: "
                        prompt_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(user_prompt))
                        input_ids_all.extend(prompt_token_ids)
                        round_input.prompt_token_ids = input_ids_all
                # 复用 session
                s0 = time.perf_counter()
                output = await async_request_eb_openai_chat_completions(
                    round_input,
                    pbar=None,
                    session=session,
                )
                s1 = time.perf_counter()
                llm_time += s1 - s0
                output.env_wait_time = turn_env_wait

                outputs.append(output)

                if not output.success:
                    session_end = time.perf_counter()
                    metrics = SessionMetrics(
                        session_no=request_func_input.no,
                        session_e2e_time=session_end - session_start,
                        pure_llm_time=llm_time,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        tool_calls=tool_call_count,
                    )
                    return outputs, metrics

                # llm_cost = s1 - s0
                input_tokens = output.prompt_tokens
                output_tokens += output.output_tokens

                # 更新output_ids
                output_ids = output.output_ids

                if max_prompt_len and input_tokens >= max_prompt_len:
                    # 后验超长截断
                    print(f"[SESSION STOP] {round_input.no} reach max_prompt_len={max_prompt_len}, stop session")
                    break

                if enable_tools:
                    # 循环调用工具
                    max_loop = json_data.get("max_loop", 10)
                    tool_url = json_data.get("tool_url", "")
                    max_prompt_len = json_data.get("max_prompt_len")
                    if not tool_url:
                        raise ValueError("tool_url is empty.")
                    for _ in range(max_loop):
                        t0 = time.perf_counter()
                        tool_result, is_tool_result, tool_name, tool_id = await simple_tool_call(
                            output,
                            tool_url,
                        )
                        t1 = time.perf_counter()
                        tool_time += t1 - t0
                        # print(f"#### tool_time: {t1 - t0:.3f}")
                        # print(f"#### tool_result: {tool_result}")
                        # print(f"#### is_tool_result: {is_tool_result}")

                        # 工具调用失败
                        if tool_name and not is_tool_result:
                            print(f"[SESSION FAIL] tool call failed: {tool_name}")

                            output.success = False

                            session_end = time.perf_counter()
                            session_e2e_time = session_end - session_start
                            tool_call_count += 1

                            metrics = SessionMetrics(
                                session_no=request_func_input.no,
                                session_e2e_time=session_e2e_time,
                                pure_llm_time=llm_time,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                tool_calls=tool_call_count,
                            )

                            return outputs, metrics

                        if not is_tool_result:
                            history.append(
                                {
                                    "role": "assistant",
                                    "content": output.generated_text,
                                }
                            )
                            break

                        assistant_msg = {
                            "role": "assistant",
                            "content": output.generated_text,
                        }

                        if getattr(output, "tool_calls", None):
                            assistant_msg["tool_calls"] = [
                                {
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                                    },
                                }
                                for tc in output.tool_calls
                            ]

                        history.append(assistant_msg)

                        history.append(
                            {
                                "role": "tool",
                                "content": json.dumps(tool_result, ensure_ascii=False),
                                "tool_call_id": tool_id or tool_name,
                            }
                        )
                        tool_call_count += 1

                        round_input.history_QA = history

                        s0 = time.perf_counter()
                        output = await async_request_eb_openai_chat_completions(
                            round_input,
                            pbar=None,
                            session=session,
                        )
                        s1 = time.perf_counter()
                        llm_time += s1 - s0

                        outputs.append(output)

                        if not output.success:
                            session_end = time.perf_counter()
                            metrics = SessionMetrics(
                                session_no=request_func_input.no,
                                session_e2e_time=session_end - session_start,
                                pure_llm_time=llm_time,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                tool_calls=tool_call_count,
                            )
                            return outputs, metrics

                        input_tokens = output.prompt_tokens
                        output_tokens += output.output_tokens
                        # 若session输入长度超过max_prompt_len，则停止session
                        if max_prompt_len and input_tokens >= max_prompt_len:
                            print(
                                f"[SESSION STOP] {round_input.no} reach max_prompt_len={max_prompt_len}, stop session"
                            )
                            session_end = time.perf_counter()
                            metrics = SessionMetrics(
                                session_no=request_func_input.no,
                                session_e2e_time=session_end - session_start,
                                pure_llm_time=llm_time,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                tool_calls=tool_call_count,
                            )
                            return outputs, metrics
                    else:
                        print(f"Warning {prompt_no} exceed max_loop={max_loop}, force stop tool loop")

                else:
                    # 无tools（SWE模式）：不追加模型实际返回，直接用数据集里的assistant回复拼接多轮
                    pass

                prompt_no += 1
            elif message["role"] == "assistant":
                if enable_tools:
                    # 工具调用模式：跳过数据集里的assistant消息，使用模型实际返回
                    continue
                else:
                    # SWE模式：直接用数据集里的assistant回复拼接多轮
                    history.append(message)
            else:
                history.append(message)

    session_end = time.perf_counter()
    session_e2e_time = session_end - session_start

    if pbar:
        pbar.update(1)

    metrics = SessionMetrics(
        session_no=request_func_input.no,
        session_e2e_time=session_e2e_time,
        pure_llm_time=llm_time,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=tool_call_count,
    )

    return outputs, metrics


async def async_request_eb_openai_chat_completions_multi_turn(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
):
    if not request_func_input.enable_session_control:
        return await _async_request_eb_openai_chat_completions_multi_turn(request_func_input, pbar)

    request_func_input.session_id = request_func_input.session_id or uuid.uuid4().hex
    result = None
    try:
        result = await _async_request_eb_openai_chat_completions_multi_turn(request_func_input, pbar)
    finally:
        try:
            async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
                await _close_session(
                    session,
                    request_func_input.session_control_url or request_func_input.api_url,
                    request_func_input.session_id,
                )
        except Exception:
            close_error = traceback.format_exc()
            logging.error("Failed to close session %s: %s", request_func_input.session_id, close_error)
            if result and result[0]:
                result[0][-1].success = False
                result[0][-1].error += close_error

    return result


async def async_request_eb_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """Request an LLM using EB OpenAI"""
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    async with aiohttp.ClientSession(
        trust_env=True, read_bufsize=10 * 1024 * 1024, timeout=AIOHTTP_TIMEOUT
    ) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "stream": True,
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True,
            },
        }
        # 超参由yaml传入
        payload.update(request_func_input.hyper_parameters)

        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos

        if request_func_input.debug:
            print("payload:", json.dumps(payload, ensure_ascii=False))

        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        output.no = request_func_input.no

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk != "[DONE]":
                            # print("####chunk:", chunk, chunk.usage)
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("text")

                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                generated_text += text or ""

                                most_recent_timestamp = timestamp
                                output.arrival_time.append(choices[0].get("arrival_time", timestamp))
                            elif usage := data.get("usage"):
                                output.prompt_tokens = usage.get("prompt_tokens")
                                output.output_tokens = usage.get("completion_tokens")
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT." "This response will be marked as failed!"
                        )

                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st

                    if output.generated_text == "":
                        output.success = False
                        output.error = "No generated text found!"
                    else:
                        output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if request_func_input.debug:
            print(f"final_output:{output}")

    if pbar:
        pbar.update(1)
    return output


async def async_request_tgi(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """Request an LLM using the TGI API"""
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        params = {
            "max_new_tokens": request_func_input.output_len,
            "do_sample": True,
            "temperature": 0.01,  # TGI does not accept 0.0 temperature.
            "top_p": 0.99,  # TGI does not accept 1.0 top_p.
            "truncate": request_func_input.prompt_len,
            "ignore_eos_token": request_func_input.ignore_eos,
        }
        payload = {
            "inputs": request_func_input.prompt,
            "parameters": params,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        if request_func_input.ignore_eos:
            output.output_tokens = request_func_input.output_len
        else:
            output.output_tokens = None

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk_bytes = chunk_bytes.decode("utf-8")

                        # NOTE: Sometimes TGI returns a ping response without
                        # any data, we should skip it.
                        if chunk_bytes.startswith(":"):
                            continue
                        chunk = chunk_bytes.removeprefix("data:")

                        data = json.loads(chunk)
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp
                        output.arrival_time.append(data["arrival_time"])

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.generated_text = data["generated_text"]
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """Request an LLM using TRT's llm_server"""
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        if request_func_input.ignore_eos:
            payload["min_length"] = request_func_input.output_len
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data:")

                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = timestamp - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True

                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_deepspeed_mii(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """Request an LLM using Deepspeed MII"""
    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:

        payload = {
            "prompt": request_func_input.prompt,
            "max_tokens": request_func_input.output_len,
            "temperature": 0.01,  # deepspeed-mii does not accept 0.0 temp.
            "top_p": 1.0,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        # NOTE: DeepSpeed-MII doesn't support streaming as of Jan 28 2024,
        # will use 0 as placeholder.
        # See https://github.com/microsoft/DeepSpeed-MII/pull/311
        output.ttft = 0

        st = time.perf_counter()
        try:
            async with session.post(url=request_func_input.api_url, json=payload) as response:
                if response.status == 200:
                    parsed_resp = await response.json()
                    output.latency = time.perf_counter() - st
                    if "choices" in parsed_resp:
                        output.generated_text = parsed_resp["choices"][0]["text"]
                    elif "text" in parsed_resp:
                        output.generated_text = parsed_resp["text"][0]
                    else:
                        output.error = "Unexpected response format: " "neither 'choices' nor 'text' found"
                        output.success = False
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """Request an LLM using OpenAI"""
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": (request_func_input.model_name if request_func_input.model_name else request_func_input.model),
            "prompt": request_func_input.prompt,
            # "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            # "stream_options": {
            #    "include_usage": True,
            # },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos

        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk != "[DONE]":
                            # print("####chunk:", chunk, type(chunk))
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("text")
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT." "This response will be marked as failed!"
                        )
                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_audio(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """Request an LLM using OpenAI"""
    # Lazy import without PlaceholderModule to avoid vllm dep.
    import soundfile

    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("transcriptions", "translations")
    ), "OpenAI Chat Completions API URL must end with 'transcriptions' "
    "or `translations`."

    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        content = [{"type": "text", "text": request_func_input.prompt}]
        payload = {
            "model": (request_func_input.model_name if request_func_input.model_name else request_func_input.model),
            "temperature": 0.0,
            "max_completion_tokens": request_func_input.output_len,
            "stream": True,
            "language": "en",
            # Flattened due to multipart/form-data
            "stream_include_usage": True,
            "stream_continuous_usage_stats": True,
        }
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        # Send audio file
        def to_bytes(y, sr):
            buffer = io.BytesIO()
            soundfile.write(buffer, y, sr, format="WAV")
            buffer.seek(0)
            return buffer

        with to_bytes(*request_func_input.multi_modal_content["audio"]) as f:
            form = aiohttp.FormData()
            form.add_field("file", f, content_type="audio/wav")
            for key, value in payload.items():
                form.add_field(key, str(value))

            output = RequestFuncOutput()
            output.prompt_len = request_func_input.prompt_len

            generated_text = ""
            ttft = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st
            try:
                async with session.post(url=api_url, data=form, headers=headers) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                            if chunk != "[DONE]":
                                timestamp = time.perf_counter()
                                data = json.loads(chunk)

                                if choices := data.get("choices"):
                                    content = choices[0]["delta"].get("content")
                                    # First token
                                    if ttft == 0.0:
                                        ttft = timestamp - st
                                        output.ttft = ttft

                                    # Decoding phase
                                    else:
                                        output.itl.append(timestamp - most_recent_timestamp)

                                    generated_text += content or ""
                                elif usage := data.get("usage"):
                                    output.output_tokens = usage.get("completion_tokens")

                                most_recent_timestamp = timestamp

                        output.generated_text = generated_text
                        output.success = True
                        output.latency = most_recent_timestamp - st
                    else:
                        output.error = response.reason or ""
                        output.success = False
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_eb_openai_completions,
    "openai-chat": async_request_eb_openai_chat_completions,
    "openai-chat-multi-turn": async_request_eb_openai_chat_completions_multi_turn,
    "openai-audio": async_request_openai_audio,
    "tensorrt-llm": async_request_trt_llm,
    "scalellm": async_request_openai_completions,
    "sglang": async_request_openai_completions,
}

OPENAI_COMPATIBLE_BACKENDS = [
    k
    for k, v in ASYNC_REQUEST_FUNCS.items()
    if v
    in (
        async_request_openai_completions,
        async_request_eb_openai_chat_completions,
    )
]
