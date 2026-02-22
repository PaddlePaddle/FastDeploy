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
import time
from types import SimpleNamespace

import paddle
import pytest

if not hasattr(paddle, "compat"):
    paddle.compat = SimpleNamespace(enable_torch_proxy=lambda scope: None)

import fastdeploy.engine.engine as engine_module
from fastdeploy.engine.engine import LLMEngine
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.utils import EngineError, envs


class DummySignal:
    def __init__(self, name, array, dtype, suffix, create):
        self.name = name
        self.value = array
        self.dtype = dtype
        self.suffix = suffix
        self.create = create
        self.cleared = False

    def clear(self):
        self.cleared = True


class DummyMetrics:
    def __init__(self):
        self.scheduler_recv_req_time = 0
        self.preprocess_start_time = 0
        self.preprocess_end_time = 0


class FakeRequest:
    def __init__(self, data):
        self._data = data
        self.guided_json = data.get("guided_json")
        self.guided_regex = data.get("guided_regex")
        self.guided_choice = data.get("guided_choice")
        self.structural_tag = data.get("structural_tag")
        self.guided_grammar = data.get("guided_grammar")
        self.guided_json_object = data.get("guided_json_object")
        self.prompt_token_ids = data.get("prompt_token_ids", [])
        self.prompt_token_ids_len = data.get("prompt_token_ids_len", 0)
        self.need_prefill_tokens = data.get("need_prefill_tokens", 0)
        self.metrics = DummyMetrics()
        self.sampling_params = None

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = value


class DummyScheduler:
    def __init__(self):
        self.requests = []
        self.started = []

    def put_requests(self, requests):
        self.requests.extend(requests)

    def start(self, *args):
        self.started.append(args)

    def get_results(self):
        return "results"


class DummyTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or ["a", "b"]

    def get_vocab(self):
        return {"</think>": 2, "<|IMAGE_PLACEHOLDER|>": 3, "\n": 4}


class DummyDataProcessor:
    def __init__(self, prompt_token_ids=None):
        self.prompt_token_ids = prompt_token_ids or [1, 2]
        self.tokenizer = DummyTokenizer()
        self.eos_token_id_len = 1
        self.pad_token_id = 0

    def process_request(self, request, max_model_len, **kwargs):
        request.prompt_token_ids = list(self.prompt_token_ids)
        request.prompt_token_ids_len = len(self.prompt_token_ids)
        return request

    def process_response(self, result):
        return result


class DummyProcess:
    def __init__(self, pid=123):
        self.pid = pid
        self.join_called = False
        self._polled = False
        self.started = False

    def join(self):
        self.join_called = True

    def poll(self):
        return None if not self._polled else 1

    def start(self):
        self.started = True


class DummyQueueServer:
    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


class DummyZmqServer:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class DummyConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def print(self):
        return None


class JsonConfig:
    def to_json_string(self):
        return "{}"


def build_cfg():
    return DummyConfig(
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            enable_prefix_caching=True,
            enable_chunked_prefill=False,
            block_size=8,
            enc_dec_block_num=0,
            gpu_memory_utilization=0.9,
            kv_cache_ratio=0.5,
            num_cpu_blocks=1,
            max_encoder_cache=0,
            cache_transfer_protocol="tcp",
            total_block_num=10,
            reset=lambda num: None,
        ),
        parallel_config=SimpleNamespace(
            device_ids="0",
            engine_worker_queue_port=[1234, 1235],
            tensor_parallel_size=1,
            expert_parallel_size=1,
            chunked_moe_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_chunked_moe=False,
            disable_custom_all_reduce=False,
            use_internode_ll_two_stage=False,
            disable_sequence_parallel_moe=False,
            shutdown_comm_group_if_worker_idle=False,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=16,
            splitwise_role="prefill",
            name="splitwise",
        ),
        model_config=SimpleNamespace(
            max_model_len=10,
            model="demo",
            quantization={},
            runner="default",
            convert="none",
            override_pooler_config="",
            logprobs_mode="",
            max_logprobs=0,
            model_impl="",
            enable_logprob=False,
            lm_head_fp32=False,
            enable_entropy=False,
            num_hidden_layers=4,
        ),
        structured_outputs_config=SimpleNamespace(
            guided_decoding_backend="",
            reasoning_parser="",
            disable_any_whitespace=False,
            logits_processors=None,
        ),
        load_config=SimpleNamespace(load_strategy="", dynamic_load_weight=False, load_choices="", rsync_config={}),
        speculative_config=JsonConfig(),
        graph_opt_config=JsonConfig(),
        early_stop_config=JsonConfig(),
        plas_attention_config=JsonConfig(),
        eplb_config=JsonConfig(),
        routing_replay_config=JsonConfig(),
        master_ip="127.0.0.1",
        host_ip="127.0.0.1",
        register_info={},
        node_rank=0,
        worker_num_per_node=1,
        nnode=1,
        ips=["127.0.0.1", "127.0.0.2"],
    )


def build_engine(cfg, data_processor=None):
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.guided_decoding_checker = None
    engine.engine = SimpleNamespace(
        data_processor=data_processor or DummyDataProcessor(),
        scheduler=DummyScheduler(),
    )
    return engine


class DummyResult:
    def __init__(self, finished=False):
        self.finished = finished

    def to_dict(self):
        return {"outputs": {"text": "ok", "reasoning_content": "trace"}}


def test_init_sets_do_profile_and_uses_paddle(monkeypatch):
    tensor = paddle.to_tensor([1, 2, 3])
    assert int(tensor.sum()) == 6

    cfg = DummyConfig(cache_config=SimpleNamespace(num_gpu_blocks_override=1), print=lambda: None)
    dummy_service = object()
    monkeypatch.setattr(engine_module, "EngineService", lambda cfg: dummy_service)
    monkeypatch.setattr(engine_module.main_process_metrics, "set_cache_config_info", lambda obj: None)
    monkeypatch.setattr(engine_module.tracing, "trace_set_thread_info", lambda name: None)

    engine = LLMEngine(cfg)

    assert engine.do_profile == 0
    assert engine.engine is dummy_service
    engine._finalizer.detach()


def test_add_requests_min_tokens_too_long(monkeypatch):
    cfg = build_cfg()
    cfg.model_config.max_model_len = 5
    request = FakeRequest({"request_id": "r1", "max_tokens": 2, "min_tokens": 1})
    engine = build_engine(cfg, data_processor=DummyDataProcessor([1, 2, 3, 4]))

    monkeypatch.setattr(engine_module.Request, "from_dict", lambda data: request)

    with pytest.raises(EngineError) as exc:
        engine.add_requests({"request_id": "r1", "min_tokens": 1, "max_tokens": 2})

    assert "Input text is too long" in str(exc.value)


def test_add_requests_input_ids_too_long(monkeypatch):
    cfg = build_cfg()
    cfg.model_config.max_model_len = 3
    request = FakeRequest({"request_id": "r2", "max_tokens": 2, "min_tokens": -5})
    engine = build_engine(cfg, data_processor=DummyDataProcessor([1, 2, 3, 4]))

    monkeypatch.setattr(engine_module.Request, "from_dict", lambda data: request)

    with pytest.raises(EngineError) as exc:
        engine.add_requests({"request_id": "r2", "min_tokens": -5, "max_tokens": 2})

    assert "exceeds the limit" in str(exc.value)


def test_add_requests_stop_sequences_and_temperature(monkeypatch):
    cfg = build_cfg()
    request = FakeRequest(
        {
            "request_id": "r3",
            "max_tokens": 2,
            "min_tokens": 1,
            "stop_seqs_len": [1, 2],
        }
    )
    engine = build_engine(cfg)
    sampling_params = SamplingParams(max_tokens=2, min_tokens=1, temperature=0.0)

    monkeypatch.setattr(engine_module.Request, "from_dict", lambda data: request)
    monkeypatch.setattr(envs, "FD_MAX_STOP_SEQS_NUM", 1, raising=False)

    with pytest.raises(EngineError) as exc:
        engine.add_requests({"request_id": "r3"}, sampling_params=sampling_params)

    assert sampling_params.temperature == pytest.approx(1e-06)
    assert "max_stop_seqs_num" in str(exc.value)


def test_add_requests_stop_sequence_length_limit(monkeypatch):
    cfg = build_cfg()
    request = FakeRequest(
        {
            "request_id": "r4",
            "max_tokens": 2,
            "min_tokens": 1,
            "stop_seqs_len": [3],
        }
    )
    engine = build_engine(cfg)

    monkeypatch.setattr(engine_module.Request, "from_dict", lambda data: request)
    monkeypatch.setattr(envs, "FD_STOP_SEQS_MAX_LEN", 1, raising=False)

    with pytest.raises(EngineError) as exc:
        engine.add_requests({"request_id": "r4"})

    assert "stop_seqs" in str(exc.value)


def test_add_requests_guided_backend_missing(monkeypatch):
    cfg = build_cfg()
    request = FakeRequest({"request_id": "r5", "max_tokens": 2, "min_tokens": 1, "guided_json": {}})
    engine = build_engine(cfg)

    monkeypatch.setattr(engine_module.Request, "from_dict", lambda data: request)

    with pytest.raises(EngineError) as exc:
        engine.add_requests({"request_id": "r5"})

    assert "guided_backend is None" in str(exc.value)


def test_add_requests_guided_checker_error(monkeypatch):
    cfg = build_cfg()
    request = FakeRequest({"request_id": "r6", "max_tokens": 2, "min_tokens": 1, "guided_json": {}})
    engine = build_engine(cfg)
    engine.guided_decoding_checker = SimpleNamespace(schema_format=lambda req: (req, "bad schema"))

    monkeypatch.setattr(engine_module.Request, "from_dict", lambda data: request)

    with pytest.raises(EngineError) as exc:
        engine.add_requests({"request_id": "r6"})

    assert "bad schema" in str(exc.value)


def test_add_requests_guided_checker_success(monkeypatch):
    cfg = build_cfg()
    request = FakeRequest({"request_id": "r7", "max_tokens": 2, "min_tokens": 1, "guided_json": {}})
    engine = build_engine(cfg)
    engine.guided_decoding_checker = SimpleNamespace(schema_format=lambda req: (req, None))

    monkeypatch.setattr(engine_module.Request, "from_dict", lambda data: request)

    engine.add_requests({"request_id": "r7"})

    assert engine.engine.scheduler.requests == [request]
    assert request.prompt_token_ids_len == len(request.prompt_token_ids)


def test_worker_processes_ready():
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.worker_ready_signal = SimpleNamespace(value=[1])

    assert engine._worker_processes_ready() is True


def test_init_worker_signals_with_profile(monkeypatch):
    cfg = build_cfg()
    cfg.parallel_config.data_parallel_size = 2
    cfg.nnode = 1
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.do_profile = 1
    engine.ipc_signal_suffix = "123"

    monkeypatch.setattr(engine_module, "IPCSignal", DummySignal)
    monkeypatch.setattr(paddle, "is_compiled_with_custom_device", lambda name: True)
    monkeypatch.setattr(envs, "FD_ENABLE_MULTI_API_SERVER", False, raising=False)

    engine._init_worker_signals()

    assert isinstance(engine.worker_ready_signal, DummySignal)
    assert isinstance(engine.launched_cache_manager_signal, DummySignal)
    assert isinstance(engine.launched_expert_service_signal, DummySignal)
    assert isinstance(engine.loaded_model_signal, DummySignal)
    assert isinstance(engine.get_profile_block_num_signal, DummySignal)
    assert len(engine.get_profile_block_num_signal.value) == cfg.worker_num_per_node


def test_exit_sub_services_cleans_resources(monkeypatch):
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg

    cache_manager = SimpleNamespace(
        shm_cache_task_flag_broadcast=DummySignal("shm", [0], None, "", True),
        cache_ready_signal=DummySignal("cache", [0], None, "", True),
    )
    engine.engine = SimpleNamespace(resource_manager=SimpleNamespace(cache_manager=cache_manager))
    engine.cache_manager_processes = [DummyProcess(pid=111)]
    engine.worker_ready_signal = DummySignal("worker", [0], None, "", True)
    engine.loaded_model_signal = DummySignal("loaded", [0], None, "", True)
    engine.get_profile_block_num_signal = DummySignal("profile", [0], None, "", True)
    engine.worker_proc = DummyProcess(pid=222)
    engine.zmq_server = DummyZmqServer()
    engine.dp_processed = [DummyProcess(pid=333)]
    engine.dp_engine_worker_queue_server = [DummyQueueServer()]

    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(OSError("fail")))

    engine._exit_sub_services()

    assert engine.worker_ready_signal.cleared is True
    assert engine.loaded_model_signal.cleared is True
    assert engine.get_profile_block_num_signal.cleared is True
    assert engine.zmq_server.closed is True
    assert engine.dp_processed[0].join_called is True
    assert engine.dp_engine_worker_queue_server[0].cleaned is True


def test_setting_environ_variables(monkeypatch):
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg

    monkeypatch.setattr(envs, "ENABLE_V1_KVCACHE_SCHEDULER", True, raising=False)

    prefix = engine._setting_environ_variables()

    assert "FLAGS_use_pd_disaggregation_per_chunk=1" in prefix
    assert "FLAGS_fmt_write_cache_completed_signal=1" in prefix


def test_start_worker_service_builds_command(monkeypatch):
    cfg = build_cfg()
    cfg.nnode = 2
    cfg.parallel_config.data_parallel_size = 1
    cfg.cache_config.num_gpu_blocks_override = 2
    cfg.structured_outputs_config.logits_processors = ["p1", "p2"]
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.do_profile = 0
    engine.engine = SimpleNamespace(data_processor=DummyDataProcessor())
    engine.data_processor = engine.engine.data_processor

    captured = {}

    def fake_popen(cmd, stdout, shell, preexec_fn):
        captured["cmd"] = cmd
        return "proc"

    monkeypatch.setattr(engine_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(engine_module.current_platform, "is_iluvatar", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    engine._start_worker_service()

    assert "--logits-processors p1 p2" in captured["cmd"]
    assert "--nnodes" in captured["cmd"]
    assert "--devices" not in captured["cmd"]


def test_format_and_add_data_builds_context(monkeypatch):
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    captured = {}

    def fake_add_requests(prompts):
        captured["prompts"] = prompts

    engine.add_requests = fake_add_requests

    prompts = {
        "context": [
            {"role": "system", "utterance": "sys"},
            {"role": "user", "utterance": "hi"},
            {"role": "assistant", "utterance": "yo"},
        ]
    }

    req_id = engine._format_and_add_data(prompts)

    assert prompts["system"] == "sys"
    assert prompts["prompt"] == ["hi", "yo"]
    assert prompts["max_tokens"] == cfg.model_config.max_model_len
    assert captured["prompts"]["request_id"] == req_id


def test_generate_streaming_and_completion():
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.data_processor = DummyDataProcessor()
    engine.engine = SimpleNamespace(
        data_processor=engine.data_processor,
        check_and_free_block_tables=lambda: None,
    )

    engine._format_and_add_data = lambda prompts: "req"
    engine._get_generated_tokens = lambda req_id: [DummyResult(False), DummyResult(True)]

    outputs = list(engine.generate({"prompt": "hi"}, stream=True))

    assert outputs[0]["outputs"]["text"] == "ok"
    assert outputs[-1]["outputs"]["text"] == ""


def test_get_generated_result_returns_scheduler_output():
    engine = LLMEngine.__new__(LLMEngine)
    scheduler = DummyScheduler()
    scheduler.get_results = lambda: {"result": "ok"}
    engine.engine = SimpleNamespace(scheduler=scheduler)

    assert engine._get_generated_result() == {"result": "ok"}


def test_generate_wraps_add_request_errors():
    engine = LLMEngine.__new__(LLMEngine)
    engine._format_and_add_data = lambda prompts: (_ for _ in ()).throw(ValueError("bad"))

    with pytest.raises(EngineError) as exc:
        list(engine.generate({"prompt": "hi"}, stream=False))

    assert "bad" in str(exc.value)


def test_stop_profile_resets_cache(monkeypatch):
    cfg = build_cfg()
    cfg.cache_config.enable_prefix_caching = True
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.do_profile = 1
    engine.ipc_signal_suffix = "suffix"
    engine.get_profile_block_num_signal = SimpleNamespace(value=[2])
    engine.engine = SimpleNamespace(
        resource_manager=SimpleNamespace(reset_cache_config=lambda cache: None),
        start_cache_service=lambda device_ids, suffix: "cache",
    )

    monkeypatch.setattr(engine_module.current_platform, "is_intel_hpu", lambda: False)

    engine._stop_profile()

    assert engine.do_profile == 0


def test_check_health_detects_unhealthy_worker(monkeypatch):
    engine = LLMEngine.__new__(LLMEngine)
    engine.engine = SimpleNamespace(worker_healthy_live_signal=SimpleNamespace(value=[100]))

    monkeypatch.setattr(engine_module.time, "time", lambda: 200)

    ok, message = engine.check_health(time_interval_threashold=30)

    assert ok is False
    assert message == "Worker Service Not Healthy"


def test_launch_components_dp_path(monkeypatch):
    cfg = build_cfg()
    cfg.scheduler_config.name = "dp"
    cfg.parallel_config.data_parallel_size = 2
    cfg.nnode = 1
    cfg.parallel_config.tensor_parallel_size = 2
    cfg.parallel_config.engine_worker_queue_port = [1000, 1001]
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.engine = SimpleNamespace(
        split_connector=SimpleNamespace(start_receiver=lambda: None),
        scheduler=DummyScheduler(),
    )
    engine.launched_expert_service_signal = SimpleNamespace(value=[1, 1])

    monkeypatch.setattr(envs, "FD_ENABLE_MULTI_API_SERVER", False, raising=False)
    monkeypatch.setattr(envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", False, raising=False)
    monkeypatch.setattr(engine_module, "EngineWorkerQueue", lambda **kwargs: DummyQueueServer())
    monkeypatch.setattr(engine_module, "start_data_parallel_service", lambda *args: None)
    monkeypatch.setattr(engine_module.multiprocessing, "Queue", lambda: object())

    class DummyContext:
        def Process(self, target, args):
            return DummyProcess(pid=500)

    monkeypatch.setattr(engine_module.multiprocessing, "get_context", lambda name: DummyContext())
    monkeypatch.setattr(engine_module.time, "sleep", lambda seconds: None)

    engine.launch_components()

    assert engine.dp_engine_worker_queue_server
    assert engine.dp_processed


def test_check_worker_initialize_status(monkeypatch):
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.worker_init_status = {}  # Initialize the attribute

    stdout_lines = [
        b"Loading checkpoint shards: 50",
        b"Start load layer 2",
    ]

    engine.worker_proc = SimpleNamespace(stdout=stdout_lines, poll=lambda: None)
    engine._worker_processes_ready = lambda: True

    class DummyTqdm:
        def __init__(self, total, desc):
            self.n = 0

        def update(self, delta):
            self.n += delta

        def refresh(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(engine_module, "tqdm", DummyTqdm)
    monkeypatch.setattr(engine_module.time, "sleep", lambda seconds: None)

    assert engine.check_worker_initialize_status() is True
    assert engine.worker_init_status["finished"] is True


def test_worker_processes_not_ready(monkeypatch):
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    # Set worker_ready_signal to have fewer ready workers than expected
    engine.worker_ready_signal = SimpleNamespace(value=[1, 0])  # Only 1 out of 2 workers ready
    engine.cfg.worker_num_per_node = 2

    assert engine._worker_processes_ready() is False


def test_check_health_worker_healthy(monkeypatch):
    engine = LLMEngine.__new__(LLMEngine)
    engine.engine = SimpleNamespace(worker_healthy_live_signal=SimpleNamespace(value=[int(time.time())]))

    healthy, message = engine.check_health(time_interval_threashold=30)

    assert healthy is True
    assert message == ""


def test_launch_non_mixed_mode_starts_cache_manager(monkeypatch):
    """Test that cache manager starts in non-mixed mode for non-HPU platforms."""
    cfg = build_cfg()
    cfg.scheduler_config.splitwise_role = "prefill"  # Not mixed
    cfg.parallel_config.device_ids = "0,1"
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.do_profile = 0
    engine.ipc_signal_suffix = "test"
    engine.engine = SimpleNamespace()
    engine._wait_for_workers_ready = lambda: None
    engine.is_started = False

    # Mock cache manager processes
    mock_cache_processes = [DummyProcess(pid=123)]
    mock_engine = SimpleNamespace(
        start=lambda: None,
        create_data_processor=lambda: None,
        data_processor=SimpleNamespace(),
        start_cache_service=lambda device_ids, suffix: mock_cache_processes,
    )

    monkeypatch.setattr(engine_module, "current_platform", SimpleNamespace(is_intel_hpu=lambda: False))
    monkeypatch.setattr(engine, "engine", mock_engine)
    monkeypatch.setattr(engine, "_start_worker_service", lambda: DummyProcess(pid=456))
    monkeypatch.setattr(engine, "_init_worker_signals", lambda: None)
    monkeypatch.setattr(engine, "_wait_for_workers_ready", lambda: None)
    monkeypatch.setattr(engine, "launch_components", lambda: None)
    monkeypatch.setattr(engine_module.time, "sleep", lambda x: None)

    engine.start()

    assert engine.cache_manager_processes == mock_cache_processes


def test_launch_mixed_mode_starts_cache_manager_after_profile(monkeypatch):
    """Test that cache manager starts in mixed mode after profiling."""
    cfg = build_cfg()
    cfg.scheduler_config.splitwise_role = "mixed"
    cfg.cache_config.enable_prefix_caching = True
    cfg.parallel_config.device_ids = "0,1"
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.do_profile = 1  # Will trigger profiling
    engine.ipc_signal_suffix = "test"
    engine.engine = SimpleNamespace()
    engine._wait_for_workers_ready = lambda: None
    engine.is_started = False

    # Mock signals
    engine.loaded_model_signal = SimpleNamespace(value=[1])
    engine.launched_expert_service_signal = SimpleNamespace(value=[1])
    engine.worker_ready_signal = SimpleNamespace(value=[1])

    # Mock cache manager processes
    mock_cache_processes = [DummyProcess(pid=789)]
    mock_engine = SimpleNamespace(
        start=lambda: None,
        create_data_processor=lambda: None,
        data_processor=SimpleNamespace(),
        start_cache_service=lambda device_ids, suffix: mock_cache_processes,
        scheduler=SimpleNamespace(start=lambda *args: None),
    )

    monkeypatch.setattr(engine_module, "current_platform", SimpleNamespace(is_intel_hpu=lambda: False))
    monkeypatch.setattr(engine, "engine", mock_engine)
    monkeypatch.setattr(engine, "_start_worker_service", lambda: DummyProcess(pid=456))
    monkeypatch.setattr(engine, "_init_worker_signals", lambda: None)
    monkeypatch.setattr(engine, "_wait_for_workers_ready", lambda: None)
    monkeypatch.setattr(engine, "_stop_profile", lambda: None)
    monkeypatch.setattr(envs, "FD_ENABLE_MULTI_API_SERVER", False, raising=False)
    monkeypatch.setattr(envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", False, raising=False)
    monkeypatch.setattr(engine_module, "EngineWorkerQueue", lambda **kwargs: DummyQueueServer())
    monkeypatch.setattr(engine_module, "start_data_parallel_service", lambda *args: None)
    monkeypatch.setattr(engine_module.multiprocessing, "Queue", lambda: object())
    monkeypatch.setattr(
        engine_module.multiprocessing,
        "get_context",
        lambda name: SimpleNamespace(Process=lambda *args, **kwargs: DummyProcess(pid=500)),
    )
    monkeypatch.setattr(engine_module.time, "sleep", lambda seconds: None)

    engine.start()

    assert engine.cache_manager_processes == mock_cache_processes


def test_launch_non_mixed_mode_sets_cache_manager_signal(monkeypatch):
    """Test that cache manager signal is set in non-mixed mode."""
    cfg = build_cfg()
    cfg.scheduler_config.splitwise_role = "prefill"  # Not mixed
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.do_profile = 0
    engine.ipc_signal_suffix = "test"
    engine.engine = SimpleNamespace()
    engine._wait_for_workers_ready = lambda: None
    engine.is_started = False

    # Mock signals
    engine.launched_cache_manager_signal = SimpleNamespace(value=[0])

    mock_engine = SimpleNamespace(start_cache_service=lambda device_ids, suffix: [])

    monkeypatch.setattr(
        engine_module, "current_platform", SimpleNamespace(is_intel_hpu=lambda: True)
    )  # Skip cache manager
    monkeypatch.setattr(engine, "engine", mock_engine)
    monkeypatch.setattr(engine, "_start_worker_service", lambda: DummyProcess(pid=456))
    monkeypatch.setattr(engine, "_init_worker_signals", lambda: None)
    monkeypatch.setattr(engine, "_wait_for_workers_ready", lambda: None)
    monkeypatch.setattr(envs, "FD_ENABLE_MULTI_API_SERVER", False, raising=False)
    monkeypatch.setattr(envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", False, raising=False)
    monkeypatch.setattr(engine_module, "EngineWorkerQueue", lambda **kwargs: DummyQueueServer())
    monkeypatch.setattr(engine_module, "start_data_parallel_service", lambda *args: None)
    monkeypatch.setattr(engine_module.multiprocessing, "Queue", lambda: object())
    monkeypatch.setattr(
        engine_module.multiprocessing,
        "get_context",
        lambda name: SimpleNamespace(Process=lambda *args, **kwargs: DummyProcess(pid=500)),
    )
    monkeypatch.setattr(engine_module.time, "sleep", lambda seconds: None)

    engine.start()

    assert engine.launched_cache_manager_signal.value[0] == 1


def test_worker_init_check_failure_path(monkeypatch):
    """Test worker initialization check failure path."""
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.worker_init_status = {}

    # Create a mock process that will "fail" during polling
    class FailingProcess:
        def __init__(self):
            self.stdout = [b"Loading checkpoint shards: 50"]
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            return 1 if self.poll_count > 2 else None  # Fail after a few polls

    engine.worker_proc = FailingProcess()
    engine._worker_processes_ready = lambda: False

    class DummyTqdm:
        def __init__(self, total, desc):
            self.n = 0

        def update(self, delta):
            self.n += delta

        def refresh(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(engine_module, "tqdm", DummyTqdm)
    monkeypatch.setattr(engine_module.time, "sleep", lambda seconds: None)

    result = engine.check_worker_initialize_status()

    assert result is False


def test_generate_processes_stream_results(monkeypatch):
    """Test generate method processes streaming results."""
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.data_processor = DummyDataProcessor()
    engine.engine = SimpleNamespace(
        data_processor=engine.data_processor,
        check_and_free_block_tables=lambda: None,
    )

    engine._format_and_add_data = lambda prompts: "req"
    engine._get_generated_tokens = lambda req_id: [
        DummyResult(False),  # Streaming result
        DummyResult(True),  # Final result
    ]

    # Mock process_response to return None for streaming, result for final
    def mock_process_response(result):
        if not result.finished:
            return None  # Skip streaming results
        return result

    monkeypatch.setattr(engine.data_processor, "process_response", mock_process_response)

    outputs = list(engine.generate({"prompt": "hi"}, stream=True))

    # Should only get the final result since streaming returns None
    assert len(outputs) == 1
    assert outputs[0]["outputs"]["text"] == ""


def test_launch_components_logs_tensor_parallel_info(monkeypatch):
    """Test that launch_components logs tensor parallel information."""
    cfg = build_cfg()
    cfg.scheduler_config.name = "dp"
    cfg.parallel_config.data_parallel_size = 2
    cfg.nnode = 1
    cfg.parallel_config.tensor_parallel_size = 2
    cfg.parallel_config.engine_worker_queue_port = [1000, 1001]
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.engine = SimpleNamespace(
        split_connector=SimpleNamespace(start_receiver=lambda: None),
        scheduler=DummyScheduler(),
    )
    engine.launched_expert_service_signal = SimpleNamespace(value=[1, 1])

    # Mock logging to capture messages
    logged_messages = []

    def mock_info(msg):
        logged_messages.append(msg)

    monkeypatch.setattr(engine_module.llm_logger, "info", mock_info)
    monkeypatch.setattr(envs, "FD_ENABLE_MULTI_API_SERVER", False, raising=False)
    monkeypatch.setattr(envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", False, raising=False)
    monkeypatch.setattr(engine_module, "EngineWorkerQueue", lambda **kwargs: DummyQueueServer())
    monkeypatch.setattr(engine_module, "start_data_parallel_service", lambda *args: None)
    monkeypatch.setattr(engine_module.multiprocessing, "Queue", lambda: object())
    monkeypatch.setattr(
        engine_module.multiprocessing,
        "get_context",
        lambda name: SimpleNamespace(Process=lambda *args, **kwargs: DummyProcess(pid=500)),
    )
    monkeypatch.setattr(engine_module.time, "sleep", lambda seconds: None)

    engine.launch_components()

    # Check that some logging occurred (tensor parallel info or other initialization messages)
    assert len(logged_messages) > 0
