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

from __future__ import annotations

import os
import sys
import threading
import types
from pathlib import Path

import paddle
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ["FD_USE_GET_SAVE_OUTPUT_V1"] = "1"
if not hasattr(paddle, "compat"):
    paddle.compat = types.SimpleNamespace(enable_torch_proxy=lambda scope=None: None)

from fastdeploy.engine.sampling_params import GuidedDecodingParams, SamplingParams
from fastdeploy.entrypoints.llm import LLM
from fastdeploy.worker.output import LogprobsLists, LogprobsTensors


class DummyTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab = list(range(vocab_size))


class DummyDataProcessor:
    def __init__(self, vocab_size: int):
        self.tokenizer = DummyTokenizer(vocab_size)

    def process_logprob_response(self, token_ids, clean_up_tokenization_spaces: bool = False):
        return f"tok_{token_ids[0]}"

    def process_response(self, result):
        return result

    def process_response_dict_streaming(
        self, response_dict, stream: bool, enable_thinking: bool, include_stop_str_in_output: bool
    ):
        text = "".join(f"tok_{token_id}" for token_id in response_dict["outputs"]["token_ids"])
        if enable_thinking:
            text = f"think:{text}"
        return {"outputs": {"text": text}}


class DummyModelConfig:
    def __init__(self, max_logprobs: int, enable_logprob: bool):
        self.max_logprobs = max_logprobs
        self.enable_logprob = enable_logprob
        self.ori_vocab_size = 0


class DummyCacheConfig:
    def __init__(self, enable_prefix_caching: bool):
        self.enable_prefix_caching = enable_prefix_caching


class DummyCfg:
    def __init__(
        self,
        max_logprobs: int,
        enable_logprob: bool,
        enable_prefix_caching: bool,
        master_ip: str,
        is_master: bool,
    ):
        self.model_config = DummyModelConfig(max_logprobs, enable_logprob)
        self.cache_config = DummyCacheConfig(enable_prefix_caching)
        self.master_ip = master_ip
        self._is_master = is_master

    def _check_master(self):
        return self._is_master


class DummyEngine:
    def __init__(
        self,
        vocab_size: int,
        max_logprobs: int,
        enable_logprob: bool,
        enable_prefix_caching: bool,
        master_ip: str,
        is_master: bool,
    ):
        self.cfg = DummyCfg(max_logprobs, enable_logprob, enable_prefix_caching, master_ip, is_master)
        self.data_processor = DummyDataProcessor(vocab_size)
        self.requests: list[tuple[dict, SamplingParams, dict]] = []
        self.on_add_requests = None

    def add_requests(self, tasks, sampling_params, **kwargs):
        self.requests.append((tasks, sampling_params, kwargs))
        if self.on_add_requests:
            self.on_add_requests(tasks, sampling_params, kwargs)


class DummyOutputs:
    def __init__(self, token_ids, top_logprobs=None):
        self.token_ids = token_ids
        self.top_logprobs = top_logprobs
        self.logprobs = None
        self.text = None


class DummyResult:
    def __init__(self, request_id: str, outputs: DummyOutputs, finished: bool, prompt_logprobs=None):
        self.request_id = request_id
        self.outputs = outputs
        self.finished = finished
        self.prompt_logprobs = prompt_logprobs
        self.prompt = None


def build_llm(
    vocab_size: int = 10,
    max_logprobs: int = 10,
    enable_logprob: bool = False,
    enable_prefix_caching: bool = False,
    master_ip: str = "127.0.0.1",
    is_master: bool = True,
) -> LLM:
    llm = LLM.__new__(LLM)
    llm.llm_engine = DummyEngine(vocab_size, max_logprobs, enable_logprob, enable_prefix_caching, master_ip, is_master)
    llm.master_node_ip = master_ip
    llm.req_output = {}
    llm.mutex = threading.Lock()
    return llm


def test_validate_tools_accepts_and_rejects_invalid():
    llm = build_llm()
    valid_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather by city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }

    assert llm._validate_tools(None) is None
    assert llm._validate_tools([]) is None

    validated_single = llm._validate_tools(valid_tool)
    assert isinstance(validated_single, list)
    assert validated_single[0]["function"]["name"] == "get_weather"

    validated_list = llm._validate_tools([valid_tool, valid_tool])
    assert len(validated_list) == 2

    with pytest.raises(ValueError):
        llm._validate_tools("invalid")

    with pytest.raises(ValueError):
        llm._validate_tools([valid_tool, 123])

    with pytest.raises(ValueError):
        llm._validate_tools({"type": "function", "function": {"description": "missing name"}})


def test_make_logprob_dict_uses_sample_rank_and_topk():
    logprob_dict = LLM._make_logprob_dict(
        logprobs=[-0.1, -0.2, -0.3],
        logprob_token_ids=[101, 102, 103],
        decoded_tokens=["A", "B", "C"],
        rank=7,
        num_logprobs=-1,
    )

    assert logprob_dict[101].rank == 7
    assert logprob_dict[102].rank == 1
    assert logprob_dict[103].rank == 2
    assert logprob_dict[103].decoded_token == "C"
    assert logprob_dict[101].logprob == -0.1


def test_build_sample_logprobs_filters_topk_and_handles_empty():
    llm = build_llm()
    logprobs_lists = LogprobsLists(
        logprob_token_ids=[[1, 2, 3, 4]],
        logprobs=[[-1.0, -2.0, -3.0, -4.0]],
        sampled_token_ranks=[1],
    )

    result = llm._build_sample_logprobs(logprobs_lists, topk_logprobs=2)
    assert len(result) == 1
    assert set(result[0].keys()) == {1, 2, 3}
    assert result[0][1].decoded_token == "tok_1"

    empty_lists = LogprobsLists(logprob_token_ids=[], logprobs=[], sampled_token_ranks=[])
    assert llm._build_sample_logprobs(empty_lists, topk_logprobs=2) is None


def test_build_prompt_logprobs_uses_paddle_tensors():
    llm = build_llm()
    token_ids = paddle.to_tensor([[10, 11]], dtype="int64")
    logprobs = paddle.to_tensor([[-0.1, -0.2]], dtype="float32")
    ranks = paddle.to_tensor([5], dtype="int64")

    prompt_tensors = LogprobsTensors(
        logprob_token_ids=token_ids,
        logprobs=logprobs,
        selected_token_ranks=ranks,
    )

    result = llm._build_prompt_logprobs(prompt_tensors, num_prompt_logprobs=1)
    assert result[0] is None
    assert result[1][10].rank == 5
    assert result[1][11].rank == 1
    assert result[1][10].decoded_token == "tok_10"


def test_add_request_validates_logprob_constraints():
    llm = build_llm(vocab_size=4, max_logprobs=3, enable_logprob=False)
    sampling_params = SamplingParams(logprobs=2)
    with pytest.raises(ValueError, match="logprobs is only supported"):
        llm._add_request(prompts=["hello"], sampling_params=sampling_params)

    llm_prefix_cache = build_llm(vocab_size=4, max_logprobs=3, enable_logprob=True, enable_prefix_caching=True)
    sampling_params = SamplingParams(prompt_logprobs=1)
    with pytest.raises(ValueError, match="prefix caching"):
        llm_prefix_cache._add_request(prompts=["hello"], sampling_params=sampling_params)

    llm_vocab_limit = build_llm(vocab_size=2, max_logprobs=5, enable_logprob=True)
    sampling_params = SamplingParams()
    with pytest.raises(ValueError, match="exceeds vocabulary size"):
        llm_vocab_limit._add_request(prompts=["hello"], sampling_params=sampling_params)


def test_add_request_adds_guided_decoding_and_tools():
    llm = build_llm(vocab_size=8, max_logprobs=8, enable_logprob=True)
    sampling_params = SamplingParams()
    sampling_params.guided_decoding = GuidedDecodingParams(json={"city": "beijing"})
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Resolve city",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]

    req_ids = llm._add_request(prompts=["Hello"], sampling_params=sampling_params, tools=tools)
    assert len(req_ids) == 1
    tasks, _, _ = llm.llm_engine.requests[0]
    assert tasks["prompt"] == "Hello"
    assert tasks["guided_json"] == {"city": "beijing"}
    assert tasks["tools"] == tools
    assert "request_id" in tasks


def test_generate_sets_prompt_and_filters_logprobs():
    llm = build_llm(vocab_size=6, max_logprobs=6, enable_logprob=True)
    llm.default_sampling_params = SamplingParams(max_tokens=4)
    llm.chat_template = "unused"

    def on_add_requests(tasks, sampling_params, kwargs):
        request_id = tasks["request_id"]
        top_logprobs = LogprobsLists(
            logprob_token_ids=[[1, 2, 3]],
            logprobs=[[-1.0, -2.0, -3.0]],
            sampled_token_ranks=[1],
        )
        prompt_logprobs = LogprobsTensors(
            logprob_token_ids=paddle.to_tensor([[4, 5]], dtype="int64"),
            logprobs=paddle.to_tensor([[-0.4, -0.5]], dtype="float32"),
            selected_token_ranks=paddle.to_tensor([2], dtype="int64"),
        )
        outputs = DummyOutputs(token_ids=[1, 2], top_logprobs=top_logprobs)
        llm.req_output[request_id] = DummyResult(
            request_id=request_id,
            outputs=outputs,
            finished=True,
            prompt_logprobs=prompt_logprobs,
        )

    llm.llm_engine.on_add_requests = on_add_requests

    sampling_params = SamplingParams(logprobs=1, prompt_logprobs=1)
    outputs = llm.generate(prompts="hi", sampling_params=sampling_params, use_tqdm=True)
    assert outputs[0].prompt == "hi"
    assert outputs[0].outputs.logprobs[0][1].decoded_token == "tok_1"
    assert outputs[0].prompt_logprobs[1][4].rank == 2


def test_chat_stream_emits_incremental_output_with_thinking():
    llm = build_llm(vocab_size=8, max_logprobs=8, enable_logprob=True)
    llm.mutex = threading.Lock()
    llm.req_output = {}
    llm.default_sampling_params = SamplingParams(max_tokens=4)
    llm.chat_template = "unused"

    def on_add_requests(tasks, sampling_params, kwargs):
        request_id = tasks["request_id"]
        top_logprobs = LogprobsLists(
            logprob_token_ids=[[7, 8]],
            logprobs=[[-0.7, -0.8]],
            sampled_token_ranks=[1],
        )
        outputs = DummyOutputs(token_ids=[7], top_logprobs=top_logprobs)
        llm.req_output[request_id] = DummyResult(request_id=request_id, outputs=outputs, finished=True)

    llm.llm_engine.on_add_requests = on_add_requests

    stream = llm.chat(
        messages=[[{"role": "user", "content": "hello"}]],
        sampling_params=SamplingParams(logprobs=1),
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": True},
        stream=True,
    )
    streamed_results = list(stream)
    assert streamed_results[-1][0].outputs.text == "think:tok_7"
    assert streamed_results[-1][0].outputs.logprobs is not None
    assert streamed_results[-1][0].prompt == [{"role": "user", "content": "hello"}]


def test_generate_and_chat_reject_invalid_inputs():
    llm = build_llm(is_master=False, master_ip="10.0.0.1")
    llm.default_sampling_params = SamplingParams(max_tokens=2)
    with pytest.raises(ValueError, match="master node"):
        llm.generate(prompts="hi", sampling_params=SamplingParams(), use_tqdm=False)

    llm = build_llm()
    llm.default_sampling_params = SamplingParams(max_tokens=2)
    llm.chat_template = "unused"
    with pytest.raises(ValueError, match="prompts must be a input dict"):
        llm.generate(prompts={"bad": "payload"}, sampling_params=SamplingParams(), use_tqdm=False)

    with pytest.raises(RuntimeError, match="Failed to validate 'tools'"):
        llm.chat(
            messages=[[{"role": "user", "content": "hi"}]],
            sampling_params=SamplingParams(),
            tools="invalid",
            use_tqdm=False,
        )


def test_init_wires_engine_and_thread(monkeypatch):
    import fastdeploy.entrypoints.llm as llm_module

    class DummyEngineArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyEngineForInit:
        def __init__(self):
            self.cfg = types.SimpleNamespace(
                model_config=types.SimpleNamespace(max_model_len=64),
                master_ip="192.168.0.1",
                _check_master=lambda: True,
            )
            self.started = False

        def start(self):
            self.started = True

    class DummyThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

    dummy_engine = DummyEngineForInit()
    captured = {}

    def fake_from_engine_args(engine_args):
        captured["engine_args"] = engine_args
        return dummy_engine

    def fake_retrieve(model, revision):
        captured["model"] = (model, revision)
        return "resolved-model"

    def fake_load_chat_template(chat_template, model):
        captured["chat_template"] = (chat_template, model)
        return "resolved-template"

    def fake_import_tool_parser(plugin):
        captured["plugin"] = plugin

    monkeypatch.setattr(llm_module, "EngineArgs", DummyEngineArgs)
    monkeypatch.setattr(llm_module.LLMEngine, "from_engine_args", staticmethod(fake_from_engine_args))
    monkeypatch.setattr(llm_module, "retrive_model_from_server", fake_retrieve)
    monkeypatch.setattr(llm_module, "load_chat_template", fake_load_chat_template)
    monkeypatch.setattr(llm_module.ToolParserManager, "import_tool_parser", fake_import_tool_parser)
    monkeypatch.setattr(llm_module.threading, "Thread", DummyThread)

    llm = llm_module.LLM(
        model="raw-model",
        revision="dev",
        tokenizer="tokenizer",
        enable_logprob=True,
        tool_parser_plugin="plugin",
    )

    assert llm.llm_engine is dummy_engine
    assert llm.default_sampling_params.max_tokens == 64
    assert llm.master_node_ip == "192.168.0.1"
    assert llm.chat_template == "resolved-template"
    assert captured["plugin"] == "plugin"


def test_generate_stream_with_default_sampling_and_prompt_dict():
    llm = build_llm(vocab_size=5, max_logprobs=5, enable_logprob=True)
    llm.default_sampling_params = SamplingParams(logprobs=1, max_tokens=2)

    def on_add_requests(tasks, sampling_params, kwargs):
        request_id = tasks["request_id"]
        outputs = DummyOutputs(token_ids=[9])
        llm.req_output[request_id] = DummyResult(request_id=request_id, outputs=outputs, finished=True)

    llm.llm_engine.on_add_requests = on_add_requests

    stream = llm.generate(prompts=[1, 2, 3], sampling_params=None, use_tqdm=False, stream=True)
    streamed = list(stream)
    assert streamed[-1][0].prompt == [1, 2, 3]

    outputs = llm.generate(prompts={"prompt": "hi"}, sampling_params=SamplingParams(), use_tqdm=True)
    assert outputs[0].prompt["prompt"] == "hi"
    assert "request_id" in outputs[0].prompt


def test_generate_sampling_params_length_mismatch():
    llm = build_llm()
    llm.default_sampling_params = SamplingParams(max_tokens=2)
    with pytest.raises(ValueError, match="same length"):
        llm.generate(
            prompts=["a"],
            sampling_params=[SamplingParams(), SamplingParams()],
            use_tqdm=False,
        )


def test_chat_wraps_messages_and_uses_default_template():
    llm = build_llm(enable_logprob=True)
    llm.default_sampling_params = SamplingParams(logprobs=1, max_tokens=2)
    llm.chat_template = "default-template"

    def on_add_requests(tasks, sampling_params, kwargs):
        request_id = tasks["request_id"]
        outputs = DummyOutputs(token_ids=[5])
        llm.req_output[request_id] = DummyResult(request_id=request_id, outputs=outputs, finished=True)

    llm.llm_engine.on_add_requests = on_add_requests

    outputs = llm.chat(
        messages=[{"role": "user", "content": "hi"}],
        sampling_params=None,
        use_tqdm=True,
        stream=False,
    )
    assert outputs[0].outputs.text is None
    assert llm.llm_engine.requests[0][2]["chat_template"] == "default-template"


def test_chat_rejects_non_master_and_length_mismatch():
    llm = build_llm(is_master=False, master_ip="10.0.0.2")
    llm.default_sampling_params = SamplingParams(max_tokens=2)
    with pytest.raises(ValueError, match="master node"):
        llm.chat(messages=[[{"role": "user", "content": "hi"}]], sampling_params=SamplingParams(), use_tqdm=False)

    llm = build_llm()
    llm.default_sampling_params = SamplingParams(max_tokens=2)
    with pytest.raises(ValueError, match="same length"):
        llm.chat(
            messages=[[{"role": "user", "content": "hi"}]],
            sampling_params=[SamplingParams(), SamplingParams()],
            use_tqdm=False,
        )


def test_add_request_handles_logprobs_bounds_and_stream_flag():
    llm = build_llm(vocab_size=4, max_logprobs=-1, enable_logprob=True)
    sampling_params = SamplingParams(logprobs=1)
    llm._add_request(prompts=["hello"], sampling_params=sampling_params)

    llm = build_llm(vocab_size=5, max_logprobs=2, enable_logprob=True)
    with pytest.raises(ValueError, match="Number of logprobs\\(-1\\)"):
        llm._add_request(prompts=["hello"], sampling_params=SamplingParams(logprobs=-1))

    llm = build_llm(vocab_size=5, max_logprobs=5, enable_logprob=True)
    with pytest.raises(ValueError, match="streaming"):
        llm._add_request(
            prompts=["hello"],
            sampling_params=SamplingParams(prompt_logprobs=1),
            stream=True,
        )
