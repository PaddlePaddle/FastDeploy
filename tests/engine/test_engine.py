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

import time
from types import SimpleNamespace

import numpy as np
import paddle
import pytest

import fastdeploy.engine.engine as engine_module
from fastdeploy import envs
from fastdeploy.engine.engine import LLMEngine
from fastdeploy.utils import EngineError


def _make_signal(value):
    return SimpleNamespace(value=value, clear=lambda: None)


class DummyIPCSignal:
    def __init__(self, name, array, dtype, suffix, create):
        self.name = name
        self.value = array
        self.dtype = dtype
        self.suffix = suffix
        self.create = create

    def clear(self):
        return None


class DummyProcess:
    def __init__(self, pid=1):
        self.pid = pid

    def start(self):
        return None

    def join(self):
        return None


class DummyScheduler:
    def __init__(self):
        self.requests = None

    def put_requests(self, requests):
        self.requests = requests

    def get_results(self):
        return "ok"


class DummyTokenizer:
    def __init__(self):
        self.vocab = {"</think>": 1, "<|IMAGE_PLACEHOLDER|>": 2, "\n": 3}

    def get_vocab(self):
        return self.vocab


class DummyDataProcessor:
    def __init__(self, prompt_token_ids):
        self.prompt_token_ids = prompt_token_ids
        self.kwargs = None
        self.tokenizer = DummyTokenizer()
        self.eos_token_id_len = 1
        self.pad_token_id = 0

    def process_request(self, request, max_model_len, **kwargs):
        self.kwargs = kwargs
        request.prompt_token_ids = self.prompt_token_ids
        return request

    def process_response(self, _result):
        return SimpleNamespace(to_dict=lambda: {"outputs": {"text": "ok", "reasoning_content": ""}})


def _make_cfg(max_model_len=10, splitwise_role="mixed", data_parallel_size=1):
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=1,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            gpu_memory_utilization=0.9,
            block_size=16,
            enc_dec_block_num=0,
            kv_cache_ratio=0.5,
            num_cpu_blocks=0,
            max_encoder_cache=0,
            kvcache_storage_backend=None,
            cache_transfer_protocol=None,
        ),
        scheduler_config=SimpleNamespace(
            splitwise_role=splitwise_role,
            max_num_seqs=2,
            max_num_batched_tokens=8,
            name="splitwise",
        ),
        parallel_config=SimpleNamespace(
            device_ids="0",
            engine_worker_queue_port=[12345, 12346],
            tensor_parallel_size=1,
            data_parallel_size=data_parallel_size,
            expert_parallel_size=1,
            chunked_moe_size=1,
            enable_expert_parallel=False,
            enable_chunked_moe=False,
            disable_custom_all_reduce=False,
            use_internode_ll_two_stage=False,
            disable_sequence_parallel_moe=False,
            shutdown_comm_group_if_worker_idle=False,
        ),
        model_config=SimpleNamespace(
            max_model_len=max_model_len,
            quantization={},
            model="dummy",
            runner=None,
            convert=None,
            override_pooler_config=None,
            logprobs_mode=None,
            max_logprobs=None,
            model_impl=None,
            enable_logprob=False,
            lm_head_fp32=False,
            enable_entropy=False,
            num_hidden_layers=2,
        ),
        structured_outputs_config=SimpleNamespace(
            guided_decoding_backend="none",
            logits_processors=None,
            disable_any_whitespace=False,
            reasoning_parser=None,
        ),
        load_config=SimpleNamespace(load_strategy="lazy", rsync_config={}, dynamic_load_weight=False, load_choices=""),
        speculative_config=SimpleNamespace(to_json_string=lambda: "{}"),
        graph_opt_config=SimpleNamespace(to_json_string=lambda: "{}"),
        early_stop_config=SimpleNamespace(to_json_string=lambda: "{}"),
        plas_attention_config=SimpleNamespace(to_json_string=lambda: "{}"),
        eplb_config=SimpleNamespace(to_json_string=lambda: "{}"),
        routing_replay_config=SimpleNamespace(to_json_string=lambda: "{}"),
        ips=None,
        master_ip="127.0.0.1",
        host_ip="127.0.0.1",
        register_info=None,
        node_rank=0,
        worker_num_per_node=1,
        nnode=1,
    )
    cfg.print = lambda: None
    return cfg


def _make_engine_for_add_requests(prompt_token_ids, max_model_len=10):
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = _make_cfg(max_model_len=max_model_len)
    engine.engine = SimpleNamespace(
        data_processor=DummyDataProcessor(prompt_token_ids),
        scheduler=DummyScheduler(),
    )
    engine.guided_decoding_checker = None
    engine._has_guided_input = lambda _request: False
    return engine


def _make_start_engine(cfg):
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.do_profile = 0
    engine.is_started = False
    engine.engine = SimpleNamespace(
        start=lambda: None,
        create_data_processor=lambda: None,
        start_cache_service=lambda _device_ids, _suffix: ["cache"],
        start_zmq_service=lambda _pid: None,
        data_processor=DummyDataProcessor([1, 2]),
    )
    engine.data_processor = engine.engine.data_processor
    engine._start_worker_service = lambda: SimpleNamespace(pid=7)
    engine.launch_components = lambda: None
    engine.check_worker_initialize_status = lambda: False

    def _init_worker_signals():
        engine.worker_ready_signal = _make_signal(np.zeros([cfg.worker_num_per_node], dtype=np.int32))
        engine.loaded_model_signal = _make_signal(np.ones([1], dtype=np.int32))

    engine._init_worker_signals = _init_worker_signals
    return engine


def _make_engine_with_processor(cfg, prompt_token_ids=None):
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    processor = DummyDataProcessor(prompt_token_ids or [1, 2])
    engine.engine = SimpleNamespace(
        data_processor=processor,
        scheduler=DummyScheduler(),
        resource_manager=SimpleNamespace(cache_manager=SimpleNamespace()),
    )
    engine.data_processor = processor
    return engine


def _make_worker_proc(stdout, poll_seq):
    seq = list(poll_seq)

    def _poll():
        return seq.pop(0) if seq else None

    return SimpleNamespace(stdout=iter(stdout), poll=_poll)


def test_add_requests_paths(monkeypatch):
    engine = _make_engine_for_add_requests([1, 2], max_model_len=10)
    monkeypatch.setattr(envs, "FD_STOP_SEQS_MAX_LEN", 2)
    with pytest.raises(EngineError):
        engine.add_requests({"request_id": "r1", "max_tokens": 5, "stop_seqs_len": [3]})

    monkeypatch.setattr(envs, "FD_MAX_STOP_SEQS_NUM", 1)
    with pytest.raises(EngineError):
        engine.add_requests({"request_id": "r1b", "max_tokens": 5, "stop_seqs_len": [1, 1]})

    engine = _make_engine_for_add_requests([1] * 9, max_model_len=10)
    with pytest.raises(EngineError):
        engine.add_requests({"request_id": "r1c", "max_tokens": 5, "min_tokens": 2})

    engine = _make_engine_for_add_requests([1, 2], max_model_len=10)
    engine._has_guided_input = lambda _req: True
    engine.guided_decoding_checker = SimpleNamespace(schema_format=lambda request: (request, None))
    engine.add_requests({"request_id": "r2", "max_tokens": 5})
    assert engine.engine.scheduler.requests[0].request_id == "r2"

    engine = _make_engine_for_add_requests([1, 2], max_model_len=10)
    engine._has_guided_input = lambda _req: True
    with pytest.raises(EngineError):
        engine.add_requests({"request_id": "r2b", "max_tokens": 5})


def test_engine_init_and_worker_ready(monkeypatch):
    cfg = _make_cfg()
    cfg.cache_config.num_gpu_blocks_override = None

    class DummyEngineService:
        def __init__(self, _cfg):
            self.data_processor = SimpleNamespace()

    monkeypatch.setattr(engine_module, "EngineService", DummyEngineService)
    monkeypatch.setattr(engine_module.main_process_metrics, "set_cache_config_info", lambda obj: None)
    monkeypatch.setattr(engine_module.tracing, "trace_set_thread_info", lambda _name: None)

    engine = LLMEngine(cfg)
    engine._finalizer.detach()
    assert engine.do_profile == 1

    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = _make_cfg()
    engine.cfg.worker_num_per_node = 2
    engine.worker_ready_signal = _make_signal(np.ones([2], dtype=np.int32))
    assert engine._worker_processes_ready() is True
    engine.worker_ready_signal = _make_signal(np.zeros([2], dtype=np.int32))
    assert engine._worker_processes_ready() is False


def test_init_worker_signals(monkeypatch):
    cfg = _make_cfg(splitwise_role="prefill", data_parallel_size=2)
    cfg.cache_config.enable_prefix_caching = True
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.ipc_signal_suffix = "suffix"
    engine.do_profile = 1

    monkeypatch.setattr(engine_module, "IPCSignal", DummyIPCSignal)
    monkeypatch.setattr(envs, "FD_ENABLE_MULTI_API_SERVER", False)
    monkeypatch.setattr(paddle, "is_compiled_with_custom_device", lambda _name: False)
    engine._init_worker_signals()

    assert engine.worker_ready_signal.name == "worker_ready_signal"
    assert engine.launched_cache_manager_signal.name == "launched_cache_manager_signal"
    assert engine.launched_expert_service_signal.name == "launched_expert_service_signal"
    assert engine.loaded_model_signal.name == "loaded_model_signal"
    assert engine.get_profile_block_num_signal.name == "get_profile_block_num"

    monkeypatch.setattr(paddle, "is_compiled_with_custom_device", lambda _name: True)
    engine._init_worker_signals()
    assert engine.get_profile_block_num_signal.value.shape[0] == cfg.worker_num_per_node


def test_exit_sub_services_cleanup(monkeypatch):
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = _make_cfg()
    engine.engine = SimpleNamespace(
        resource_manager=SimpleNamespace(
            cache_manager=SimpleNamespace(
                shm_cache_task_flag_broadcast=_make_signal([0]),
                cache_ready_signal=_make_signal([0]),
            )
        )
    )
    engine.cache_manager_processes = [DummyProcess(pid=1)]
    engine.worker_ready_signal = _make_signal([0])
    engine.loaded_model_signal = _make_signal([0])
    engine.get_profile_block_num_signal = _make_signal([0])
    engine.worker_proc = SimpleNamespace(pid=2)
    engine.zmq_server = SimpleNamespace(close=lambda: None)
    engine.dp_processed = [DummyProcess(pid=3)]
    engine.dp_engine_worker_queue_server = [SimpleNamespace(cleanup=lambda: None)]

    monkeypatch.setattr(engine_module.os, "getpgid", lambda _pid: 1)
    monkeypatch.setattr(engine_module.os, "killpg", lambda _pgid, _sig: (_ for _ in ()).throw(OSError("kill")))
    engine._exit_sub_services()
    assert engine.worker_ready_signal.value is not None


def test_start_paths(monkeypatch):
    cfg = _make_cfg(splitwise_role="prefill")
    engine = _make_start_engine(cfg)

    monkeypatch.setattr(engine_module.time, "sleep", lambda _t: None)
    monkeypatch.setattr(engine_module.current_platform, "is_intel_hpu", lambda: False)
    monkeypatch.setattr(envs, "FD_ENABLE_INTERNAL_ADAPTER", True)
    monkeypatch.setattr(envs, "FD_ZMQ_RECV_REQUEST_SERVER_PORTS", "1000,1001")
    monkeypatch.setattr(envs, "FD_ZMQ_SEND_RESPONSE_SERVER_PORTS", "2000,2001")

    assert engine.start() is False
    assert envs.FD_ZMQ_RECV_REQUEST_SERVER_PORT == "1000"
    assert envs.FD_ZMQ_SEND_RESPONSE_SERVER_PORT == "2000"

    engine = _make_start_engine(cfg)
    engine._init_worker_signals = lambda: setattr(engine, "loaded_model_signal", _make_signal(np.zeros([1])))
    assert engine.start() is False

    cfg = _make_cfg(splitwise_role="mixed")
    cfg.cache_config.enable_prefix_caching = True
    engine = _make_start_engine(cfg)
    engine.check_worker_initialize_status = lambda: True
    cache_calls = []
    engine.engine.start_cache_service = lambda device_ids, suffix: cache_calls.append((device_ids, suffix)) or [
        "cache"
    ]
    assert engine.start() is True
    assert cache_calls


def test_generate_paths():
    engine = LLMEngine.__new__(LLMEngine)
    engine._format_and_add_data = lambda _prompts: "rid"

    class DummyResp:
        def __init__(self, finished):
            self.finished = finished

    calls = {"count": 0}

    def _process_response(_res):
        calls["count"] += 1
        if calls["count"] == 1:
            return None
        return SimpleNamespace(to_dict=lambda: {"outputs": {"text": "ok", "reasoning_content": ""}})

    engine._get_generated_tokens = lambda _rid: iter([DummyResp(False), DummyResp(True)])
    engine.engine = SimpleNamespace(
        data_processor=SimpleNamespace(process_response=_process_response),
        check_and_free_block_tables=lambda: None,
    )
    outputs = list(engine.generate({"prompt": "hi"}, stream=True))
    assert outputs[-1]["outputs"]["text"] == ""
    outputs_sync = list(engine.generate({"prompt": "hi"}, stream=False))
    assert outputs_sync[-1]["outputs"]["text"] == "ok"

    engine = LLMEngine.__new__(LLMEngine)
    engine._format_and_add_data = lambda _prompts: (_ for _ in ()).throw(ValueError("bad"))
    with pytest.raises(EngineError):
        list(engine.generate({"prompt": "hi"}, stream=False))

    engine = LLMEngine.__new__(LLMEngine)
    engine.engine = SimpleNamespace(
        scheduler=DummyScheduler(),
        worker_healthy_live_signal=_make_signal([time.time() - 100]),
    )
    assert engine._get_generated_result() == "ok"
    ok, reason = engine.check_health(time_interval_threashold=30)
    assert ok is False
    assert reason == "Worker Service Not Healthy"


def test_format_and_add_data_context():
    engine = _make_engine_with_processor(_make_cfg())
    captured = {}
    engine.add_requests = lambda prompts: captured.setdefault("prompts", prompts)
    prompts = {"context": [{"role": "system", "utterance": "s"}, {"role": "user", "utterance": "u"}]}
    req_id = engine._format_and_add_data(prompts)
    assert req_id == prompts["request_id"]
    assert prompts["prompt"] == ["u"]
    assert captured["prompts"] is prompts
    assert engine._format_and_add_data({"request_id": "fixed-id"}) == "fixed-id"


def test_start_worker_service_builds_command(monkeypatch):
    cfg = _make_cfg(splitwise_role="prefill")
    cfg.ips = ["10.0.0.1", "10.0.0.2"]
    cfg.nnode = 2
    cfg.cache_config.num_gpu_blocks_override = 2
    cfg.structured_outputs_config.logits_processors = ["lp1", "lp2"]
    engine = _make_engine_with_processor(cfg)
    engine.do_profile = 0

    monkeypatch.setattr(engine_module.current_platform, "is_iluvatar", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(engine_module.os, "setsid", lambda: 0)
    monkeypatch.setattr(envs, "ENABLE_V1_KVCACHE_SCHEDULER", True)

    captured = {}

    def _fake_popen(cmd, stdout, shell, preexec_fn):
        captured["cmd"] = cmd
        return DummyProcess(pid=10)

    monkeypatch.setattr(engine_module.subprocess, "Popen", _fake_popen)
    proc = engine._start_worker_service()
    assert isinstance(proc, DummyProcess)
    assert "--logits-processors lp1 lp2" in captured["cmd"]
    assert "--nnodes 2" in captured["cmd"]
    assert "--devices" not in captured["cmd"]
    assert "FLAGS_use_pd_disaggregation_per_chunk=1" in captured["cmd"]
    assert "FLAGS_fmt_write_cache_completed_signal=1" in captured["cmd"]


def test_launch_components_paths(monkeypatch):
    cfg = _make_cfg(splitwise_role="prefill")
    engine = _make_engine_with_processor(cfg)
    engine.engine.split_connector = SimpleNamespace(start_receiver=lambda: None)
    called = {}
    engine.engine.scheduler.start = lambda *args: called.setdefault("args", args)
    engine.launch_components()
    assert called["args"][0] == "prefill"

    cfg = _make_cfg(splitwise_role="mixed", data_parallel_size=2)
    cfg.scheduler_config.name = "dp"
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.engine = SimpleNamespace(scheduler=SimpleNamespace(start=lambda *args: None))
    engine.launched_expert_service_signal = _make_signal(np.array([0, 0], dtype=np.int32))

    monkeypatch.setattr(envs, "FD_ENABLE_MULTI_API_SERVER", False)
    monkeypatch.setattr(engine_module.multiprocessing, "Queue", lambda: "queue")

    class DummyCtx:
        def Process(self, *args, **kwargs):
            return DummyProcess(pid=11)

    monkeypatch.setattr(engine_module.multiprocessing, "get_context", lambda _name: DummyCtx())
    monkeypatch.setattr(engine_module, "EngineWorkerQueue", lambda **_kwargs: object())

    for use_shm in (True, False):
        monkeypatch.setattr(envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", use_shm)

        def _sleep(_t):
            engine.launched_expert_service_signal.value[1] = 1

        monkeypatch.setattr(engine_module.time, "sleep", _sleep)
        engine.launch_components()
        assert len(engine.dp_processed) == 1


def test_check_worker_initialize_status_cases(monkeypatch):
    def run_case(stdout, poll_seq, status, ready, expect, join_error=False):
        engine = LLMEngine.__new__(LLMEngine)
        engine.cfg = _make_cfg()
        engine.worker_init_status = status
        engine.worker_proc = _make_worker_proc(stdout, poll_seq)
        engine._worker_processes_ready = lambda: ready
        monkeypatch.setattr(engine_module.time, "sleep", lambda _t: None)
        if join_error:
            monkeypatch.setattr(
                engine_module.threading.Thread,
                "join",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError()),
            )
        return engine.check_worker_initialize_status()

    assert run_case([], [1], {}, False, False) is False
    assert run_case([], [None, 1], {}, True, False) is False
    assert run_case([b"line"], [None, None], {"finished": True}, True, True) is True
    assert run_case([b"Loading checkpoint shards: 100", b"Start load layer 2"], [None, None], {}, True, True) is True
    assert run_case([b""], [None, None], {"weight_loadding": 1.0, "layer_loadding": 1.0}, True, True, True) is True
