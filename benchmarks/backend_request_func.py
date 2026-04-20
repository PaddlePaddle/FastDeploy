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


import copy
import io
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


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


@dataclass
class RequestFuncOutput:
    """Output for requesting LLMs via API"""

    no: int = 0
    request_id: str = ""
    generated_text: str = ""
    reasoning_content: str = ""
    success: bool = False
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
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": True,
        },
        "max_tokens": request_func_input.output_len,
        "collect_metrics": request_func_input.pd_metrics,
    }
    if request_func_input.json_data:
        json_data = request_func_input.json_data

        if json_data.get("max_tokens"):
            payload["max_tokens"] = json_data["max_tokens"]

        if json_data.get("min_tokens"):
            payload["min_tokens"] = json_data["min_tokens"]
    if request_func_input.response_format:
        payload["response_format"] = request_func_input.response_format

    # 随机输入开关
    if request_func_input.random_flag:
        payload["max_tokens"] = request_func_input.output_len
        payload["min_tokens"] = request_func_input.output_len
        # 随机token_ids场景
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

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }

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
    token_timestamps = []
    tool_call_buffer = {}
    try:
        async with session.post(url=api_url, json=payload, headers=headers, read_bufsize=10 * 1024 * 1024) as response:
            data = {}
            if response.status == 200:
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue

                    chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                    if chunk != "[DONE]":
                        # print("####chunk:", chunk, type(chunk))
                        timestamp = time.perf_counter()
                        data = json.loads(chunk)
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
                            if tool_calls:
                                for tc in tool_calls:
                                    idx = tc.get("index", 0)

                                    if idx not in tool_call_buffer:
                                        tool_call_buffer[idx] = {
                                            "id": tc.get("id"),
                                            "name": "",
                                            "arguments": "",
                                        }

                                    func = tc.get("function", {})

                                    if "name" in func:
                                        tool_call_buffer[idx]["name"] = func["name"]

                                    if "arguments" in func:
                                        tool_call_buffer[idx]["arguments"] += func["arguments"]

                            # First token
                            if ttft == 0.0:
                                ttft = timestamp - st
                                output.ttft = ttft
                                # cached_tokens
                                if data["usage"] and data["usage"].get("prompt_tokens_details", {}):
                                    output.prompt_len = (
                                        data["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                                    )
                                else:
                                    output.prompt_len = 0

                            # Decoding phase
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)

                            # response首token
                            if res_ttft == 0.0:
                                if content:
                                    res_ttft = choices[0].get("arrival_time", timestamp)
                                    output.res_ttft = res_ttft
                                    usage = data.get("usage") or {}
                                    output.reasoning_tokens = max(usage.get("completion_tokens", 0) - 1, 0)

                            output.generated_text += content or ""
                            output.reasoning_content += reason_content or ""
                            if completion_token_ids:
                                output.output_ids.extend(completion_token_ids)
                            # print(f"####content:{data}")
                            output.arrival_time.append(choices[0].get("arrival_time", timestamp))
                        elif usage := data.get("usage", {}):
                            output.output_tokens = usage.get("completion_tokens", 0)
                            output.prompt_tokens = usage.get("prompt_tokens", 0)
                            if output.prompt_len == 0:
                                if data["usage"] and data["usage"].get("prompt_tokens_details", {}):
                                    output.prompt_len = (
                                        data["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                                    )

                        most_recent_timestamp = timestamp
                        token_timestamps.append(time.time())

                # output.generated_text = generated_text
                # 在流式结束时，记录最后一个 chunk 收到的时间戳
                output.end_timestamp = most_recent_timestamp
                # 截断case也记录usage
                usage = data.get("usage", {})
                if usage:
                    output.output_tokens = usage.get("completion_tokens", 0)
                    output.prompt_tokens = usage.get("prompt_tokens", 0)
                    if output.prompt_len == 0:
                        prompt_details = usage.get("prompt_tokens_details", {})
                        if prompt_details:
                            output.prompt_len = prompt_details.get("cached_tokens", 0)

                if tool_call_buffer:
                    for _, tc in tool_call_buffer.items():
                        try:
                            args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                        except:
                            args = {}

                        output.tool_calls.append({"id": tc["id"], "name": tc["name"], "arguments": args})

                # 新增metrics统计，计算首token过滤空包
                output.metrics = metrics_summary(metrics_list, token_timestamps[1:])

                has_text = output.generated_text.strip() or output.reasoning_content.strip()
                has_tool = getattr(output, "tool_calls", None)

                # 兼容思考内容超长截断的情况，此时回复内容为空
                if not has_text and not has_tool:
                    output.success = False
                    output.reasoning_tokens = output.output_tokens
                    output.error = "No generated text found!"
                else:
                    output.success = True
                output.latency = most_recent_timestamp - st
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
    if pbar:
        pbar.update(1)
    if request_func_input.debug:
        print("#####final_output:", output)
    return output


async def simple_tool_call(model_output, tool_url: str, timeout=60):
    """调用工具函数"""
    import re

    import httpx

    tool_id = None

    if getattr(model_output, "tool_calls", None):
        tc = model_output.tool_calls[0]
        tool_name = tc["name"]
        args = tc.get("arguments", {})
        tool_id = tc.get("id")
    else:
        match = re.search(r"<tool_call>(.*?)</tool_call>", model_output.generated_text, re.S)
        if not match:
            return "", False, "", tool_id

        block = match.group(1).strip()
        lines = block.splitlines()
        tool_name = lines[0].strip()

        key = re.search(r"<arg_key>(.*?)</arg_key>", block)
        val = re.search(r"<arg_value>(.*?)</arg_value>", block)

        args = {key.group(1): val.group(1)} if key and val else {}

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


async def async_request_eb_openai_chat_completions_multi_turn(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
):
    # yaml中或数据集中带tools才走工具调用逻辑
    json_data = request_func_input.json_data or {}
    hyper = request_func_input.hyper_parameters or {}
    enable_tools = bool(json_data.get("tools") or hyper.get("tools"))

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
                history.append(message)
                round_input = copy.deepcopy(request_func_input)
                round_input.history_QA = history
                round_input.no = f"{round_input.no}_{prompt_no}"
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
                input_tokens += output.prompt_tokens
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

                        input_tokens += output.prompt_tokens
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
                    # 无tools
                    history.append(
                        {
                            "role": "assistant",
                            "content": output.generated_text,
                        }
                    )

                prompt_no += 1
            elif message["role"] == "assistant":
                continue
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
