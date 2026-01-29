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

import textwrap
import threading
from types import SimpleNamespace

import paddle
import pytest

import fastdeploy.entrypoints.llm as llm_module
import fastdeploy.envs as envs
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

    def process_response_dict_streaming(self, response_dict, stream, enable_thinking, include_stop_str_in_output):
        tokens = "".join(f"tok_{tid}" for tid in response_dict["outputs"]["token_ids"])
        return {"outputs": {"text": f"think:{tokens}" if enable_thinking else tokens}}


class DummyResult:
    def __init__(self, request_id, token_ids, top_logprobs=None, prompt_logprobs=None, finished=True):
        self.request_id = request_id
        self.outputs = SimpleNamespace(token_ids=token_ids, top_logprobs=top_logprobs, logprobs=None)
        self.prompt_logprobs = prompt_logprobs
        self.finished = finished
        self.added = False

    def add(self, other):
        self.added = True


def _make_engine(vocab_size=5, max_logprobs=5, enable_logprob=True, enable_prefix_caching=False, is_master=True):
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            max_logprobs=max_logprobs,
            enable_logprob=enable_logprob,
            ori_vocab_size=vocab_size,
            max_model_len=8,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=enable_prefix_caching),
        master_ip="127.0.0.1",
        _check_master=lambda: is_master,
    )
    engine = SimpleNamespace(cfg=cfg, data_processor=DummyDataProcessor(vocab_size), requests=[])
    engine.add_requests = lambda tasks, sampling_params, **kwargs: engine.requests.append(
        (tasks, sampling_params, kwargs)
    )
    engine.start = lambda: None
    return engine


def _make_llm(engine):
    llm = LLM.__new__(LLM)
    llm.llm_engine = engine
    llm.default_sampling_params = SamplingParams(max_tokens=2)
    llm.mutex = threading.Lock()
    llm.req_output = {}
    llm.master_node_ip = engine.cfg.master_ip
    llm.chat_template = "template"
    return llm


def test_init_tool_parser_plugin(monkeypatch):
    captured = {}
    engine = _make_engine()

    class DummyThread:
        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            return None

    monkeypatch.setattr(llm_module, "deprecated_kwargs_warning", lambda **_: None)
    monkeypatch.setattr(llm_module, "retrive_model_from_server", lambda model, rev: model)
    monkeypatch.setattr(
        llm_module.ToolParserManager, "import_tool_parser", lambda name: captured.setdefault("p", name)
    )
    monkeypatch.setattr(llm_module, "EngineArgs", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(llm_module.LLMEngine, "from_engine_args", lambda engine_args: engine)
    monkeypatch.setattr(llm_module, "load_chat_template", lambda template, model: "tmpl")
    monkeypatch.setattr(llm_module.threading, "Thread", DummyThread)
    llm = LLM(model="m", tool_parser_plugin="plugin")
    assert captured["p"] == "plugin"
    assert llm.master_node_ip == "127.0.0.1"


def test_receive_output_merges():
    llm = _make_llm(_make_engine())
    first = DummyResult("r1", [1])
    second = DummyResult("r1", [2])
    results = iter([{"r1": [first, second]}, SystemExit()])

    def _get_generated_result():
        nxt = next(results)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    llm.llm_engine._get_generated_result = _get_generated_result
    with pytest.raises(SystemExit):
        llm._receive_output()
    assert first.added is True


def test_receive_output_logs_exception(caplog):
    llm = _make_llm(_make_engine())
    calls = iter([RuntimeError("boom"), SystemExit()])

    def _get_generated_result():
        nxt = next(calls)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    llm.llm_engine._get_generated_result = _get_generated_result
    with pytest.raises(SystemExit):
        llm._receive_output()
    assert "Unexcepted error happened" in caplog.text


def test_generate_and_chat_branches():
    llm = _make_llm(_make_engine(is_master=False))
    llm._check_master = lambda: False
    with pytest.raises(ValueError, match="master node"):
        llm.generate("hi")

    llm = _make_llm(_make_engine())
    llm._check_master = lambda: True
    llm._run_engine_stream = lambda *_, **__: "streamed"
    llm._add_request = lambda **_: ["r1"]
    with pytest.raises(ValueError, match="input dict"):
        llm.generate({"x": 1}, sampling_params=SamplingParams(max_tokens=1), use_tqdm=False)
    assert llm.generate("hi", sampling_params=SamplingParams(max_tokens=1), use_tqdm=False, stream=True) == "streamed"

    llm._check_master = lambda: False
    with pytest.raises(ValueError, match="master node"):
        llm.chat(messages=[[{"role": "user", "content": "hi"}]], sampling_params=SamplingParams(), use_tqdm=False)

    llm._check_master = lambda: True
    llm._validate_tools = lambda *_: (_ for _ in ()).throw(ValueError("bad tools"))
    with pytest.raises(RuntimeError, match="Failed to validate"):
        llm.chat(
            messages=[[{"role": "user", "content": "hi"}]], tools=1, sampling_params=SamplingParams(), use_tqdm=False
        )
    assert (
        llm.chat(
            messages=[[{"role": "user", "content": "hi"}]],
            sampling_params=SamplingParams(max_tokens=1),
            use_tqdm=False,
            stream=True,
        )
        == "streamed"
    )


def test_add_request_validations_and_guided_decoding(monkeypatch):
    llm = _make_llm(_make_engine())
    with pytest.raises(ValueError, match="both None"):
        llm._add_request(prompts=None, sampling_params=SamplingParams())
    with pytest.raises(TypeError, match="Invalid type"):
        llm._add_request(prompts=[object()], sampling_params=SamplingParams())

    llm = _make_llm(_make_engine(enable_logprob=False))
    with pytest.raises(ValueError, match="enable_logprob"):
        llm._add_request(prompts=["hi"], sampling_params=SamplingParams(logprobs=1))

    monkeypatch.setattr(envs, "FD_USE_GET_SAVE_OUTPUT_V1", True)
    llm = _make_llm(_make_engine(max_logprobs=1, enable_logprob=True, vocab_size=5))
    with pytest.raises(ValueError, match=r"Number of logprobs\(-1\)"):
        llm._add_request(prompts=["hi"], sampling_params=SamplingParams(logprobs=-1))
    llm = _make_llm(_make_engine(enable_logprob=False))
    with pytest.raises(ValueError, match="prompt_logprobs"):
        llm._add_request(prompts=["hi"], sampling_params=SamplingParams(prompt_logprobs=1))
    llm = _make_llm(_make_engine(max_logprobs=1, enable_logprob=True, vocab_size=5))
    with pytest.raises(ValueError, match=r"prompt_logprobs\(-1\)"):
        llm._add_request(prompts=["hi"], sampling_params=SamplingParams(prompt_logprobs=-1))

    llm = _make_llm(_make_engine())
    params = SamplingParams(guided_decoding=GuidedDecodingParams(regex="hi"))
    llm._add_request(prompts=["hi"], sampling_params=params)
    tasks, _, _ = llm.llm_engine.requests[0]
    assert tasks["guided_regex"] == "hi"


def test_build_sample_logprobs_and_errors():
    llm = _make_llm(_make_engine())
    logprobs = LogprobsLists(logprob_token_ids=[[1, 2]], logprobs=[[-0.1, -0.2]], sampled_token_ranks=[0])
    assert llm._build_sample_logprobs(logprobs, -2) is None
    llm._decode_token = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))
    assert llm._build_sample_logprobs(logprobs, 0) is None


def test_run_engine_and_streaming(monkeypatch):
    llm = _make_llm(_make_engine(vocab_size=3, enable_logprob=True))
    llm._build_sample_logprobs = lambda *_: [{1: None}]
    llm._build_prompt_logprobs = lambda *_: [None]

    top_logprobs = LogprobsLists(logprob_token_ids=[[1]], logprobs=[[-0.1]], sampled_token_ranks=[0])
    prompt_logprobs = LogprobsTensors(
        paddle.to_tensor([[1]], dtype=paddle.int64),
        paddle.to_tensor([[-0.1]], dtype=paddle.float32),
        paddle.to_tensor([1], dtype=paddle.int64),
    )
    result = DummyResult("r1", [1], top_logprobs=top_logprobs, prompt_logprobs=prompt_logprobs, finished=True)
    llm.req_output["r1"] = result

    class DummyTqdm:
        last_instance = None

        def __init__(self, **_):
            self.updated = 0
            self.closed = False
            DummyTqdm.last_instance = self

        def update(self, n):
            self.updated += n

        def close(self):
            self.closed = True
            return None

    llm_module.tqdm = DummyTqdm
    out = llm._run_engine(["r1"], use_tqdm=True, topk_logprobs=-1, num_prompt_logprobs=-1)
    assert out[0] is result

    current = DummyResult(
        "r2",
        [1, 2],
        top_logprobs=top_logprobs,
        prompt_logprobs=None,
        finished=True,
    )
    if "r2" in llm.req_output:
        llm.req_output.pop("r2")

    def fake_sleep(_):
        if "r2" not in llm.req_output:
            llm.req_output["r2"] = current
        return None

    monkeypatch.setattr(llm_module.time, "sleep", fake_sleep)
    it = llm._run_engine_stream(
        ["r2"],
        prompts=["hi"],
        use_tqdm=True,
        topk_logprobs=1,
        chat_template_kwargs={"enable_thinking": True},
    )
    batches = list(it)
    assert batches[0][0].prompt == "hi"
    assert DummyTqdm.last_instance.closed is True


def test_validate_tools_empty_and_main_block():
    llm = _make_llm(_make_engine())
    assert llm._validate_tools([]) is None

    src = llm_module.__file__
    text = src and open(src, "r", encoding="utf-8").read()
    marker = 'if __name__ == "__main__":'
    start = text.index(marker)
    line_no = text[:start].count("\n") + 1
    block = text[start:].split(marker, 1)[1].lstrip("\n")
    code = "\n" * line_no + textwrap.dedent(block)

    class DummyLLM:
        def __init__(self, *args, **kwargs):
            return None

        def generate(self, *args, **kwargs):
            return ["ok"]

    class DummySamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    exec(compile(code, src, "exec"), {"LLM": DummyLLM, "SamplingParams": DummySamplingParams, "__name__": "__main__"})


def test_create_incremental_result_scalar_prompt():
    llm = _make_llm(_make_engine())
    result = DummyResult("r3", [1, 2], finished=True)
    out = llm._create_incremental_result(result, 0, 0, "hi")
    assert out.prompt == "hi"
