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
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from fastdeploy.engine.engine import LLMEngine


def _make_cfg(**overrides):
    """Minimal cfg-like object matching LLMEngine expectations."""
    model_cfg = SimpleNamespace(
        model="/fake/model",
        model_type="ernie",
        max_model_len=2048,
        num_hidden_layers=2,
        quantization="{}",
        runner="default",
        convert=None,
        override_pooler_config=None,
        logprobs_mode="none",
        max_logprobs=0,
        enable_logprob=False,
        lm_head_fp32=False,
        moe_gate_fp32=False,
        enable_entropy=False,
        model_impl="default",
    )
    parallel_cfg = SimpleNamespace(
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
        device_ids="0",
        data_parallel_size=1,
        expert_parallel_size=1,
        chunked_moe_size=0,
        engine_worker_queue_port=[6778],
        enable_expert_parallel=False,
        enable_chunked_moe=False,
        disable_custom_all_reduce=False,
        use_internode_ll_two_stage=False,
        disable_sequence_parallel_moe=False,
        shutdown_comm_group_if_worker_idle=False,
    )
    scheduler_cfg = SimpleNamespace(
        max_num_seqs=256,
        max_num_batched_tokens=4096,
        splitwise_role="mixed",
        name="local",
        enable_overlap_schedule=False,
    )
    cache_cfg = SimpleNamespace(
        num_gpu_blocks_override=None,
        gpu_memory_utilization=0.9,
        block_size=16,
        enc_dec_block_num=0,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        kv_cache_ratio=1.0,
        kvcache_storage_backend=None,
        num_cpu_blocks=0,
        max_encoder_cache=0,
        cache_transfer_protocol="tcp",
        total_block_num=100,
    )
    load_cfg = SimpleNamespace(
        load_strategy="auto",
        rsync_config={},
        dynamic_load_weight=False,
        load_choices="auto",
    )
    speculative_cfg = SimpleNamespace(
        model_type="main",
        to_json_string=lambda: "{}",
    )
    graph_opt_cfg = SimpleNamespace(to_json_string=lambda: "{}")
    structured_outputs_cfg = SimpleNamespace(
        guided_decoding_backend=None,
        logits_processors=None,
        reasoning_parser="none",
        disable_any_whitespace=False,
    )
    early_stop_cfg = SimpleNamespace(to_json_string=lambda: "{}")
    eplb_cfg = SimpleNamespace(to_json_string=lambda: "{}")
    routing_replay_cfg = SimpleNamespace(to_json_string=lambda: "{}")
    plas_attention_cfg = SimpleNamespace(to_json_string=lambda: "{}")

    cfg = SimpleNamespace(
        model_config=model_cfg,
        parallel_config=parallel_cfg,
        scheduler_config=scheduler_cfg,
        cache_config=cache_cfg,
        load_config=load_cfg,
        speculative_config=speculative_cfg,
        graph_opt_config=graph_opt_cfg,
        structured_outputs_config=structured_outputs_cfg,
        early_stop_config=early_stop_cfg,
        eplb_config=eplb_cfg,
        routing_replay_config=routing_replay_cfg,
        plas_attention_config=plas_attention_cfg,
        worker_num_per_node=1,
        master_ip="127.0.0.1",
        host_ip="127.0.0.1",
        ips=None,
        nnode=1,
        register_info=None,
        node_rank=0,
        print=lambda: None,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_engine(**cfg_overrides):
    """Create an LLMEngine bypassing __init__."""
    engine = object.__new__(LLMEngine)
    engine.cfg = _make_cfg(**cfg_overrides)
    engine.running = True
    engine.is_started = False
    engine.do_profile = 0
    engine.engine = SimpleNamespace(scheduler=SimpleNamespace(get_results=lambda: []))
    engine.guided_decoding_checker = None
    engine.ipc_signal_suffix = 6778
    return engine


class TestLLMEngine:
    """Pytest-style tests for LLMEngine methods."""

    # ── _has_guided_input ──────────────────────────────────────────────

    def test_has_guided_input_none(self):
        e = _make_engine()
        req = SimpleNamespace(
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=None,
        )
        assert e._has_guided_input(req) is False

    def test_has_guided_input_json(self):
        e = _make_engine()
        req = SimpleNamespace(
            guided_json='{"type":"object"}',
            guided_regex=None,
            guided_choice=None,
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=None,
        )
        assert e._has_guided_input(req) is True

    def test_has_guided_input_regex(self):
        e = _make_engine()
        req = SimpleNamespace(
            guided_json=None,
            guided_regex=r"\d+",
            guided_choice=None,
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=None,
        )
        assert e._has_guided_input(req) is True

    def test_has_guided_input_choice(self):
        e = _make_engine()
        req = SimpleNamespace(
            guided_json=None,
            guided_regex=None,
            guided_choice=["yes", "no"],
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=None,
        )
        assert e._has_guided_input(req) is True

    def test_has_guided_input_structural_tag(self):
        e = _make_engine()
        req = SimpleNamespace(
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            structural_tag="<json>",
            guided_grammar=None,
            guided_json_object=None,
        )
        assert e._has_guided_input(req) is True

    def test_has_guided_input_grammar(self):
        e = _make_engine()
        req = SimpleNamespace(
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            structural_tag=None,
            guided_grammar="expr",
            guided_json_object=None,
        )
        assert e._has_guided_input(req) is True

    def test_has_guided_input_json_object(self):
        e = _make_engine()
        req = SimpleNamespace(
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=True,
        )
        assert e._has_guided_input(req) is True

    # ── _setting_environ_variables ─────────────────────────────────────

    def test_environ_returns_string_with_critical_vars(self):
        e = _make_engine()
        result = e._setting_environ_variables()
        assert isinstance(result, str)
        assert "OMP_NUM_THREADS=" in result
        assert "NCCL_ALGO=Ring" in result
        assert "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" in result
        assert "SOT_LOG_LEVEL=" in result

    def test_environ_splitwise_prefill_adds_disagg(self):
        e = _make_engine()
        e.cfg.scheduler_config.splitwise_role = "prefill"
        result = e._setting_environ_variables()
        assert "FLAGS_use_pd_disaggregation" in result

    def test_environ_mixed_no_disagg(self):
        e = _make_engine()
        e.cfg.scheduler_config.splitwise_role = "mixed"
        result = e._setting_environ_variables()
        assert "FLAGS_use_pd_disaggregation=1" not in result

    # ── _worker_processes_ready ────────────────────────────────────────

    def test_worker_ready_all(self):
        e = _make_engine()
        e.worker_ready_signal = SimpleNamespace(value=np.ones(1, dtype=np.int32))
        assert e._worker_processes_ready() is True

    def test_worker_not_ready(self):
        e = _make_engine()
        e.worker_ready_signal = SimpleNamespace(value=np.zeros(1, dtype=np.int32))
        assert e._worker_processes_ready() is False

    def test_worker_partial_multi(self):
        e = _make_engine()
        e.cfg.worker_num_per_node = 4
        e.worker_ready_signal = SimpleNamespace(value=np.array([1, 1, 0, 1], dtype=np.int32))
        assert e._worker_processes_ready() is False

    def test_worker_all_multi(self):
        e = _make_engine()
        e.cfg.worker_num_per_node = 3
        e.worker_ready_signal = SimpleNamespace(value=np.array([1, 1, 1], dtype=np.int32))
        assert e._worker_processes_ready() is True

    # ── check_health ───────────────────────────────────────────────────

    def test_health_ok_signal_zero(self):
        e = _make_engine()
        e.engine.worker_healthy_live_signal = SimpleNamespace(value=np.array([0.0]))
        healthy, msg = e.check_health()
        assert healthy is True

    def test_health_ok_recent(self):
        e = _make_engine()
        e.engine.worker_healthy_live_signal = SimpleNamespace(value=np.array([time.time()]))
        healthy, _ = e.check_health()
        assert healthy is True

    def test_health_stale(self):
        e = _make_engine()
        e.engine.worker_healthy_live_signal = SimpleNamespace(value=np.array([time.time() - 60]))
        healthy, msg = e.check_health(time_interval_threashold=30)
        assert healthy is False
        assert "Not Healthy" in msg

    # ── _format_and_add_data ───────────────────────────────────────────

    def test_format_generates_request_id(self):
        e = _make_engine()
        calls = []
        e.add_requests = lambda t, **kw: calls.append(t)
        prompts = {"prompt": "Hello"}
        req_id = e._format_and_add_data(prompts)
        uuid.UUID(req_id)
        assert prompts["request_id"] == req_id
        assert len(calls) == 1

    def test_format_preserves_request_id(self):
        e = _make_engine()
        e.add_requests = lambda t, **kw: None
        prompts = {"prompt": "Hello", "request_id": "my-id"}
        assert e._format_and_add_data(prompts) == "my-id"

    def test_format_sets_max_tokens_default(self):
        e = _make_engine()
        e.add_requests = lambda t, **kw: None
        prompts = {"prompt": "Hello"}
        e._format_and_add_data(prompts)
        assert prompts["max_tokens"] == e.cfg.model_config.max_model_len

    def test_format_context_extraction(self):
        e = _make_engine()
        e.add_requests = lambda t, **kw: None
        prompts = {
            "context": [
                {"role": "system", "utterance": "Helper"},
                {"role": "user", "utterance": "Hi"},
                {"role": "assistant", "utterance": "Hey"},
                {"role": "user", "utterance": "Bye"},
            ]
        }
        e._format_and_add_data(prompts)
        assert prompts["system"] == "Helper"
        assert prompts["prompt"] == ["Hi", "Hey", "Bye"]

    # ── _init_worker_signals ───────────────────────────────────────────

    def test_init_signals_basic(self, monkeypatch):
        e = _make_engine()
        monkeypatch.setattr(
            "fastdeploy.engine.engine.IPCSignal",
            lambda **kw: SimpleNamespace(
                value=np.zeros(kw.get("array", np.zeros(1)).shape, dtype=kw.get("dtype", np.int32)), clear=lambda: None
            ),
        )
        e._init_worker_signals()
        assert hasattr(e, "worker_ready_signal")
        assert hasattr(e, "loaded_model_signal")

    def test_init_signals_with_profile(self, monkeypatch):
        e = _make_engine()
        e.do_profile = 1
        monkeypatch.setattr(
            "fastdeploy.engine.engine.IPCSignal",
            lambda **kw: SimpleNamespace(
                value=np.zeros(kw.get("array", np.zeros(1)).shape, dtype=kw.get("dtype", np.int32)), clear=lambda: None
            ),
        )
        e._init_worker_signals()
        assert hasattr(e, "get_profile_block_num_signal")

    def test_init_signals_with_prefix_caching(self, monkeypatch):
        e = _make_engine()
        e.cfg.cache_config.enable_prefix_caching = True
        monkeypatch.setattr(
            "fastdeploy.engine.engine.IPCSignal",
            lambda **kw: SimpleNamespace(
                value=np.zeros(kw.get("array", np.zeros(1)).shape, dtype=kw.get("dtype", np.int32)), clear=lambda: None
            ),
        )
        e._init_worker_signals()
        assert hasattr(e, "launched_cache_manager_signal")

    def test_init_signals_dp_gt_1(self, monkeypatch):
        e = _make_engine()
        e.cfg.parallel_config.data_parallel_size = 2
        monkeypatch.setattr(
            "fastdeploy.engine.engine.IPCSignal",
            lambda **kw: SimpleNamespace(
                value=np.zeros(kw.get("array", np.zeros(1)).shape, dtype=kw.get("dtype", np.int32)), clear=lambda: None
            ),
        )
        monkeypatch.setattr("fastdeploy.engine.engine.envs.FD_ENABLE_MULTI_API_SERVER", False)
        e._init_worker_signals()
        assert hasattr(e, "launched_expert_service_signal")

    # ── _exit_sub_services ─────────────────────────────────────────────

    def test_exit_sets_running_false(self, monkeypatch):
        e = _make_engine()
        e.worker_ready_signal = SimpleNamespace(clear=lambda: None)
        e.loaded_model_signal = SimpleNamespace(clear=lambda: None)
        monkeypatch.setattr("fastdeploy.engine.engine.os.getpgid", lambda pid: 12345)
        monkeypatch.setattr("fastdeploy.engine.engine.os.killpg", lambda pgid, sig: None)
        e._exit_sub_services()
        assert e.running is False

    def test_exit_kills_worker(self, monkeypatch):
        e = _make_engine()
        e.worker_ready_signal = SimpleNamespace(clear=lambda: None)
        e.loaded_model_signal = SimpleNamespace(clear=lambda: None)
        killed = []
        monkeypatch.setattr("fastdeploy.engine.engine.os.getpgid", lambda pid: pid)
        monkeypatch.setattr("fastdeploy.engine.engine.os.killpg", lambda pgid, sig: killed.append(pgid))
        e.worker_proc = SimpleNamespace(pid=99999)
        e._exit_sub_services()
        assert 99999 in killed

    # ── _stop_profile ──────────────────────────────────────────────────

    def test_stop_profile(self):
        e = _make_engine()
        e.do_profile = 1
        e.get_profile_block_num_signal = SimpleNamespace(value=np.array([100], dtype=np.int32))
        reset_calls = []
        e.engine.resource_manager = SimpleNamespace(reset_cache_config=lambda cfg: None)
        e.cfg.cache_config = SimpleNamespace(
            reset=lambda n: reset_calls.append(n),
            enable_prefix_caching=False,
        )
        e._stop_profile()
        assert e.do_profile == 0
        assert reset_calls == [100]

    # ── _get_generated_result ──────────────────────────────────────────

    def test_get_result_delegates(self):
        e = _make_engine()
        e.engine.scheduler = SimpleNamespace(get_results=lambda: ["r1"])
        assert e._get_generated_result() == ["r1"]

    # ── from_engine_args ───────────────────────────────────────────────

    def test_from_engine_args(self, monkeypatch):
        monkeypatch.setattr("fastdeploy.engine.engine.EngineService", lambda cfg: SimpleNamespace())
        monkeypatch.setattr("fastdeploy.engine.engine.main_process_metrics.set_cache_config_info", lambda **kw: None)
        monkeypatch.setattr("fastdeploy.engine.engine.tracing.trace_set_thread_info", lambda s: None)
        args = SimpleNamespace(create_engine_config=lambda: _make_cfg())
        engine = LLMEngine.from_engine_args(args)
        assert isinstance(engine, LLMEngine)
        assert engine.do_profile == 1

    def test_from_engine_args_no_profile(self, monkeypatch):
        monkeypatch.setattr("fastdeploy.engine.engine.EngineService", lambda cfg: SimpleNamespace())
        monkeypatch.setattr("fastdeploy.engine.engine.main_process_metrics.set_cache_config_info", lambda **kw: None)
        monkeypatch.setattr("fastdeploy.engine.engine.tracing.trace_set_thread_info", lambda s: None)
        cfg = _make_cfg()
        cfg.cache_config.num_gpu_blocks_override = 100
        args = SimpleNamespace(create_engine_config=lambda: cfg)
        engine = LLMEngine.from_engine_args(args)
        assert engine.do_profile == 0

    # ── launch_components ──────────────────────────────────────────────

    def test_launch_splitwise_starts_receiver(self):
        e = _make_engine()
        e.cfg.scheduler_config.splitwise_role = "prefill"
        e.cfg.scheduler_config.name = "splitwise"
        started = []
        e.engine.split_connector = SimpleNamespace(start_receiver=lambda: None)
        e.engine.scheduler = SimpleNamespace(start=lambda *a, **kw: started.append(True))
        e.launch_components()
        assert hasattr(e, "splitwise_receive_thread")
        assert len(started) == 1

    def test_launch_local_no_splitwise(self):
        e = _make_engine()
        e.cfg.scheduler_config.splitwise_role = "mixed"
        e.cfg.scheduler_config.name = "local"
        started = []
        e.engine.scheduler = SimpleNamespace(start=lambda: started.append(True))
        e.launch_components()
        assert len(started) == 0

    # ── add_requests (validation) ──────────────────────────────────────

    def test_add_requests_input_too_long(self, monkeypatch):
        from fastdeploy.utils import EngineError

        e = _make_engine()
        e.cfg.model_config.max_model_len = 2048
        req = SimpleNamespace(
            prompt_token_ids=list(range(3000)),
            prompt_token_ids_len=3000,
            need_prefill_tokens=3000,
            metrics=SimpleNamespace(scheduler_recv_req_time=0, preprocess_start_time=0, preprocess_end_time=0),
            get=lambda k: {"max_tokens": 100, "min_tokens": 0, "request_id": "x", "stop_seqs_len": None}.get(k),
            set=lambda k, v: None,
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=None,
        )
        monkeypatch.setattr("fastdeploy.engine.engine.Request.from_dict", lambda d: req)
        e.engine.data_processor = SimpleNamespace(process_request=lambda r, *a, **kw: r)
        with pytest.raises(EngineError):
            e.add_requests({"prompt": "x" * 3000})

    def test_add_requests_min_tokens_exceeds(self, monkeypatch):
        from fastdeploy.utils import EngineError

        e = _make_engine()
        e.cfg.model_config.max_model_len = 100
        vals = {"max_tokens": 50, "min_tokens": 95, "request_id": "x", "stop_seqs_len": None}
        req = SimpleNamespace(
            prompt_token_ids=list(range(10)),
            prompt_token_ids_len=10,
            need_prefill_tokens=10,
            metrics=SimpleNamespace(scheduler_recv_req_time=0, preprocess_start_time=0, preprocess_end_time=0),
            get=lambda k: vals.get(k),
            set=lambda k, v: None,
            sampling_params=None,
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=None,
        )
        monkeypatch.setattr("fastdeploy.engine.engine.Request.from_dict", lambda d: req)
        e.engine.data_processor = SimpleNamespace(process_request=lambda r, *a, **kw: r)
        with pytest.raises(EngineError):
            e.add_requests({"prompt": "hi"})

    def test_add_requests_stop_seqs_too_many(self, monkeypatch):
        from fastdeploy.utils import EngineError

        e = _make_engine()
        e.cfg.model_config.max_model_len = 2048
        vals = {"max_tokens": 100, "min_tokens": 0, "request_id": "x", "stop_seqs_len": list(range(200))}
        req = SimpleNamespace(
            prompt_token_ids=list(range(10)),
            prompt_token_ids_len=10,
            need_prefill_tokens=10,
            metrics=SimpleNamespace(scheduler_recv_req_time=0, preprocess_start_time=0, preprocess_end_time=0),
            get=lambda k: vals.get(k),
            set=lambda k, v: setattr(req, k, v),
            sampling_params=None,
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=None,
        )
        monkeypatch.setattr("fastdeploy.engine.engine.Request.from_dict", lambda d: req)
        monkeypatch.setattr("fastdeploy.engine.engine.envs.FD_MAX_STOP_SEQS_NUM", 10)
        e.engine.data_processor = SimpleNamespace(process_request=lambda r, *a, **kw: r)
        with pytest.raises(EngineError):
            e.add_requests({"prompt": "hi"})

    def test_add_requests_stop_seq_too_long(self, monkeypatch):
        from fastdeploy.utils import EngineError

        e = _make_engine()
        e.cfg.model_config.max_model_len = 2048
        vals = {"max_tokens": 100, "min_tokens": 0, "request_id": "x", "stop_seqs_len": [500]}
        req = SimpleNamespace(
            prompt_token_ids=list(range(10)),
            prompt_token_ids_len=10,
            need_prefill_tokens=10,
            metrics=SimpleNamespace(scheduler_recv_req_time=0, preprocess_start_time=0, preprocess_end_time=0),
            get=lambda k: vals.get(k),
            set=lambda k, v: setattr(req, k, v),
            sampling_params=None,
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=None,
        )
        monkeypatch.setattr("fastdeploy.engine.engine.Request.from_dict", lambda d: req)
        monkeypatch.setattr("fastdeploy.engine.engine.envs.FD_MAX_STOP_SEQS_NUM", 100)
        monkeypatch.setattr("fastdeploy.engine.engine.envs.FD_STOP_SEQS_MAX_LEN", 10)
        e.engine.data_processor = SimpleNamespace(process_request=lambda r, *a, **kw: r)
        with pytest.raises(EngineError):
            e.add_requests({"prompt": "hi"})

    def test_add_requests_guided_no_backend(self, monkeypatch):
        from fastdeploy.utils import EngineError

        e = _make_engine()
        e.cfg.model_config.max_model_len = 2048
        e.guided_decoding_checker = None
        vals = {"max_tokens": 100, "min_tokens": 0, "request_id": "x", "stop_seqs_len": None}
        req = SimpleNamespace(
            prompt_token_ids=list(range(10)),
            prompt_token_ids_len=10,
            need_prefill_tokens=10,
            metrics=SimpleNamespace(scheduler_recv_req_time=0, preprocess_start_time=0, preprocess_end_time=0),
            get=lambda k: vals.get(k),
            set=lambda k, v: setattr(req, k, v),
            sampling_params=None,
            guided_json='{"type":"object"}',
            guided_regex=None,
            guided_choice=None,
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=None,
        )
        monkeypatch.setattr("fastdeploy.engine.engine.Request.from_dict", lambda d: req)
        e.engine.data_processor = SimpleNamespace(process_request=lambda r, *a, **kw: r)
        with pytest.raises(EngineError):
            e.add_requests({"prompt": "hi"})

    def test_add_requests_happy_path(self, monkeypatch):
        e = _make_engine()
        e.cfg.model_config.max_model_len = 2048
        put_calls = []
        vals = {"max_tokens": 100, "min_tokens": 0, "request_id": "x", "stop_seqs_len": None}
        req = SimpleNamespace(
            prompt_token_ids=list(range(10)),
            prompt_token_ids_len=10,
            need_prefill_tokens=10,
            metrics=SimpleNamespace(scheduler_recv_req_time=0, preprocess_start_time=0, preprocess_end_time=0),
            get=lambda k: vals.get(k),
            set=lambda k, v: setattr(req, k, v),
            sampling_params=None,
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=None,
        )
        monkeypatch.setattr("fastdeploy.engine.engine.Request.from_dict", lambda d: req)
        e.engine.data_processor = SimpleNamespace(process_request=lambda r, *a, **kw: r)
        e.engine.scheduler = SimpleNamespace(put_requests=lambda reqs: put_calls.extend(reqs))
        e.add_requests({"prompt": "hi"})
        assert len(put_calls) == 1

    def test_add_requests_with_sampling_params(self, monkeypatch):

        e = _make_engine()
        e.cfg.model_config.max_model_len = 2048
        put_calls = []
        vals = {"max_tokens": 100, "min_tokens": 0, "request_id": "x", "stop_seqs_len": None}
        req = SimpleNamespace(
            prompt_token_ids=list(range(10)),
            prompt_token_ids_len=10,
            need_prefill_tokens=10,
            metrics=SimpleNamespace(scheduler_recv_req_time=0, preprocess_start_time=0, preprocess_end_time=0),
            get=lambda k: vals.get(k),
            set=lambda k, v: setattr(req, k, v),
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=None,
        )
        sp = SimpleNamespace(temperature=0.0)
        monkeypatch.setattr("fastdeploy.engine.engine.Request.from_dict", lambda d: req)
        monkeypatch.setattr("fastdeploy.engine.engine.asdict", lambda x: {"temperature": 0.0})
        e.engine.data_processor = SimpleNamespace(process_request=lambda r, *a, **kw: r)
        e.engine.scheduler = SimpleNamespace(put_requests=lambda reqs: put_calls.extend(reqs))
        e.add_requests({"prompt": "hi"}, sampling_params=sp)
        assert sp.temperature == 1e-06  # clamped from 0
        assert req.sampling_params is sp

    # ── _start_worker_service ──────────────────────────────────────────

    def test_start_worker_service_builds_cmd(self, monkeypatch):

        e = _make_engine()
        e.data_processor = SimpleNamespace(
            tokenizer=SimpleNamespace(
                vocab={"<pad>": 0, "hello": 1},
                get_vocab=lambda: {"<think>": 5, "</think>": 6, "<|IMAGE_PLACEHOLDER|>": -1, "\n": 10},
                encode=lambda s, add_special_tokens=False: [10],
                think_truncate_prompt="...",
                tokenize=lambda s: ["..."],
                convert_tokens_to_ids=lambda t: [99],
            ),
            eos_token_id_len=1,
            pad_token_id=0,
        )
        e.engine.data_processor = e.data_processor
        e.engine.mm_max_tokens_per_item = None
        captured = []
        monkeypatch.setattr(
            "fastdeploy.engine.engine.subprocess.Popen",
            lambda cmd, **kw: SimpleNamespace(pid=1234) if captured.append(cmd) or True else None,
        )
        monkeypatch.setattr("fastdeploy.engine.engine.current_platform.is_iluvatar", lambda: False)
        e._start_worker_service()
        assert "--max_model_len 2048" in captured[0]
        assert "--tensor_parallel_size 1" in captured[0]

    # ── generate ───────────────────────────────────────────────────────

    def test_generate_stream(self):
        e = _make_engine()
        e.add_requests = lambda t, **kw: None
        results = [
            SimpleNamespace(finished=False),
            SimpleNamespace(finished=True),
        ]
        e._get_generated_tokens = lambda rid: iter(results)
        e.engine.data_processor = SimpleNamespace(
            process_response=lambda r: SimpleNamespace(
                to_dict=lambda: {"outputs": {"text": "hi", "reasoning_content": ""}}
            )
        )
        e.engine.check_and_free_block_tables = lambda: None
        outputs = list(e.generate({"prompt": "test"}, stream=True))
        assert len(outputs) == 2

    def test_generate_non_stream(self):
        e = _make_engine()
        e.add_requests = lambda t, **kw: None
        results = [SimpleNamespace(finished=True)]
        e._get_generated_tokens = lambda rid: iter(results)
        e.engine.data_processor = SimpleNamespace(
            process_response=lambda r: SimpleNamespace(to_dict=lambda: {"outputs": {"text": "done"}})
        )
        e.engine.check_and_free_block_tables = lambda: None
        outputs = list(e.generate({"prompt": "test"}, stream=False))
        assert len(outputs) == 1
        assert outputs[0]["outputs"]["text"] == "done"

    def test_generate_error_raises(self, monkeypatch):
        from fastdeploy.utils import EngineError

        e = _make_engine()
        e.add_requests = None  # will fail
        monkeypatch.setattr(
            "fastdeploy.engine.engine.Request.from_dict",
            lambda d: (_ for _ in ()).throw(ValueError("bad")),
        )
        with pytest.raises(EngineError):
            list(e.generate({"prompt": "x"}, stream=False))

    # ── launch_components (DP path) ────────────────────────────────────

    def test_launch_dp_scheduler(self, monkeypatch):
        e = _make_engine()
        e.cfg.scheduler_config.name = "dp"
        e.cfg.scheduler_config.splitwise_role = "mixed"
        started = []
        e.engine.scheduler = SimpleNamespace(start=lambda *a, **kw: started.append(a))
        e.launch_components()
        assert len(started) == 1

    def test_launch_dp_multi_creates_processes(self, monkeypatch):

        e = _make_engine()
        e.cfg.scheduler_config.name = "local"
        e.cfg.scheduler_config.splitwise_role = "mixed"
        e.cfg.parallel_config.data_parallel_size = 2
        e.cfg.parallel_config.engine_worker_queue_port = [6778, 6779]
        monkeypatch.setattr("fastdeploy.engine.engine.envs.FD_ENABLE_MULTI_API_SERVER", False)
        monkeypatch.setattr("fastdeploy.engine.engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", False)

        e.launched_expert_service_signal = SimpleNamespace(value=np.zeros(2, dtype=np.int32))
        # Make signal value[1] immediately 1 so while loop exits
        e.launched_expert_service_signal.value[1] = 1

        mock_proc = SimpleNamespace(start=lambda: None, pid=111)
        mock_ctx = SimpleNamespace(Process=lambda target, args: mock_proc)
        monkeypatch.setattr("fastdeploy.engine.engine.multiprocessing.get_context", lambda kind: mock_ctx)
        monkeypatch.setattr(
            "fastdeploy.engine.engine.EngineWorkerQueue",
            lambda **kw: SimpleNamespace(),
        )
        monkeypatch.setattr("fastdeploy.engine.engine.copy.deepcopy", lambda x: x)

        e.launch_components()
        assert len(e.dp_processed) == 1
        assert e.dp_processed[0].pid == 111

    # ── _exit_sub_services (cache manager path) ────────────────────────

    def test_exit_cleans_cache_manager(self, monkeypatch):
        e = _make_engine()
        e.worker_ready_signal = SimpleNamespace(clear=lambda: None)
        e.loaded_model_signal = SimpleNamespace(clear=lambda: None)
        killed_pids = []
        monkeypatch.setattr("fastdeploy.engine.engine.os.getpgid", lambda pid: pid)
        monkeypatch.setattr("fastdeploy.engine.engine.os.killpg", lambda pgid, sig: killed_pids.append(pgid))
        cache_cleared = []
        e.engine.resource_manager = SimpleNamespace(
            cache_manager=SimpleNamespace(
                shm_cache_task_flag_broadcast=SimpleNamespace(clear=lambda: cache_cleared.append("broadcast")),
                cache_ready_signal=SimpleNamespace(clear=lambda: cache_cleared.append("ready")),
            )
        )
        e.cache_manager_processes = [SimpleNamespace(pid=5555)]
        e._exit_sub_services()
        assert 5555 in killed_pids
        assert "broadcast" in cache_cleared
        assert "ready" in cache_cleared

    def test_exit_cleans_dp_processes(self, monkeypatch):
        e = _make_engine()
        e.worker_ready_signal = SimpleNamespace(clear=lambda: None)
        e.loaded_model_signal = SimpleNamespace(clear=lambda: None)
        monkeypatch.setattr("fastdeploy.engine.engine.os.getpgid", lambda pid: pid)
        monkeypatch.setattr("fastdeploy.engine.engine.os.killpg", lambda pgid, sig: None)
        joined = []
        e.dp_processed = [SimpleNamespace(pid=7777, join=lambda: joined.append(True))]
        e.dp_engine_worker_queue_server = [SimpleNamespace(cleanup=lambda: None)]
        e._exit_sub_services()
        assert len(joined) == 1

    def test_exit_cleans_zmq(self, monkeypatch):
        e = _make_engine()
        e.worker_ready_signal = SimpleNamespace(clear=lambda: None)
        e.loaded_model_signal = SimpleNamespace(clear=lambda: None)
        closed = []
        e.zmq_server = SimpleNamespace(close=lambda: closed.append(True))
        e._exit_sub_services()
        assert len(closed) == 1

    # ── _stop_profile (prefix caching path) ────────────────────────────

    def test_stop_profile_starts_cache_service(self, monkeypatch):
        e = _make_engine()
        e.do_profile = 1
        e.get_profile_block_num_signal = SimpleNamespace(value=np.array([42], dtype=np.int32))
        e.cfg.cache_config.enable_prefix_caching = True
        e.cfg.scheduler_config.splitwise_role = "mixed"
        cache_started = []
        e.engine.resource_manager = SimpleNamespace(reset_cache_config=lambda cfg: None)
        e.engine.start_cache_service = lambda dev, suf: cache_started.append(True)
        e.cfg.cache_config.reset = lambda n: None
        monkeypatch.setattr("fastdeploy.engine.engine.current_platform.is_intel_hpu", lambda: False)
        e._stop_profile()
        assert e.do_profile == 0
        assert len(cache_started) == 1

    # ── check_worker_initialize_status ─────────────────────────────────

    def test_check_worker_status_success(self, monkeypatch):

        e = _make_engine()
        e.cfg.model_config.num_hidden_layers = 2
        e.worker_init_status = {}
        # Simulate stdout with weight loading and layer loading lines
        lines = [
            b"Loading checkpoint shards: 100\n",
            b"Start load layer 0\n",
            b"Start load layer 1\n",
        ]
        e.worker_proc = SimpleNamespace(
            stdout=iter(lines),
            poll=lambda: None,
        )
        # Make worker ready immediately so progress loops exit
        e.worker_ready_signal = SimpleNamespace(value=np.ones(1, dtype=np.int32))
        e.cfg.worker_num_per_node = 1
        result = e.check_worker_initialize_status()
        assert result is True

    def test_check_worker_status_proc_dies(self, monkeypatch):
        e = _make_engine()
        e.cfg.model_config.num_hidden_layers = 2
        e.worker_init_status = {}
        e.worker_proc = SimpleNamespace(
            stdout=iter([]),
            poll=lambda: 1,  # process exited
        )
        e.worker_ready_signal = SimpleNamespace(value=np.zeros(1, dtype=np.int32))
        e.cfg.worker_num_per_node = 1
        result = e.check_worker_initialize_status()
        assert result is False

    # ── _start_worker_service (sp_model path + iluvatar) ───────────────

    def test_start_worker_sp_model_path(self, monkeypatch):
        e = _make_engine()
        e.data_processor = SimpleNamespace(
            tokenizer=SimpleNamespace(
                sp_model=["a", "b", "c"],  # has sp_model
                get_vocab=lambda: {"<think>": -1, "</think>": -1, "<|IMAGE_PLACEHOLDER|>": -1, "\n": -1},
                encode=lambda s, add_special_tokens=False: {"input_ids": [10]},
                think_truncate_prompt="...",
                tokenize=lambda s: ["..."],
                convert_tokens_to_ids=lambda t: [99],
            ),
            eos_token_id_len=1,
            pad_token_id=0,
        )
        e.engine.data_processor = e.data_processor
        e.engine.mm_max_tokens_per_item = {"image": 256}
        e.cfg.structured_outputs_config.logits_processors = ["proc1"]
        captured = []
        monkeypatch.setattr(
            "fastdeploy.engine.engine.subprocess.Popen",
            lambda cmd, **kw: SimpleNamespace(pid=2222) if captured.append(cmd) or True else None,
        )
        monkeypatch.setattr("fastdeploy.engine.engine.current_platform.is_iluvatar", lambda: False)
        e._start_worker_service()
        assert "--ori_vocab_size 3" in captured[0]
        assert "--logits-processors proc1" in captured[0]
        assert "--mm_max_tokens_per_item" in captured[0]

    def test_start_worker_iluvatar_strips_devices(self, monkeypatch):

        e = _make_engine()
        e.data_processor = SimpleNamespace(
            tokenizer=SimpleNamespace(
                vocab={"<pad>": 0},
                get_vocab=lambda: {"<think>": -1, "</think>": -1, "<|IMAGE_PLACEHOLDER|>": -1, "\n": -1},
                encode=lambda s, add_special_tokens=False: [10],
                think_truncate_prompt="...",
                tokenize=lambda s: ["..."],
                convert_tokens_to_ids=lambda t: [99],
            ),
            eos_token_id_len=1,
            pad_token_id=0,
        )
        e.engine.data_processor = e.data_processor
        e.engine.mm_max_tokens_per_item = None
        captured = []
        monkeypatch.setattr(
            "fastdeploy.engine.engine.subprocess.Popen",
            lambda cmd, **kw: SimpleNamespace(pid=3333) if captured.append(cmd) or True else None,
        )
        monkeypatch.setattr("fastdeploy.engine.engine.current_platform.is_iluvatar", lambda: True)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        e._start_worker_service()
        assert f"--devices {e.cfg.parallel_config.device_ids}" not in captured[0]

    def test_start_worker_nnode_gt_1(self, monkeypatch):
        e = _make_engine()
        e.cfg.nnode = 2
        e.cfg.ips = ["10.0.0.1", "10.0.0.2"]
        e.data_processor = SimpleNamespace(
            tokenizer=SimpleNamespace(
                vocab={"<pad>": 0},
                get_vocab=lambda: {"<think>": -1, "</think>": -1, "<|IMAGE_PLACEHOLDER|>": -1, "\n": -1},
                encode=lambda s, add_special_tokens=False: [10],
                think_truncate_prompt="...",
                tokenize=lambda s: ["..."],
                convert_tokens_to_ids=lambda t: [99],
            ),
            eos_token_id_len=1,
            pad_token_id=0,
        )
        e.engine.data_processor = e.data_processor
        e.engine.mm_max_tokens_per_item = None
        captured = []
        monkeypatch.setattr(
            "fastdeploy.engine.engine.subprocess.Popen",
            lambda cmd, **kw: SimpleNamespace(pid=4444) if captured.append(cmd) or True else None,
        )
        monkeypatch.setattr("fastdeploy.engine.engine.current_platform.is_iluvatar", lambda: False)
        e._start_worker_service()
        assert "--nnodes 2" in captured[0]

    def test_start_worker_store_true_flags(self, monkeypatch):
        e = _make_engine()
        e.cfg.cache_config.num_gpu_blocks_override = 200
        e.cfg.cache_config.kvcache_storage_backend = "rocksdb"
        e.cfg.parallel_config.enable_expert_parallel = True
        e.cfg.cache_config.enable_prefix_caching = True
        e.data_processor = SimpleNamespace(
            tokenizer=SimpleNamespace(
                vocab={"<pad>": 0},
                get_vocab=lambda: {"<think>": -1, "</think>": -1, "<|IMAGE_PLACEHOLDER|>": -1, "\n": -1},
                encode=lambda s, add_special_tokens=False: [10],
                think_truncate_prompt="...",
                tokenize=lambda s: ["..."],
                convert_tokens_to_ids=lambda t: [99],
            ),
            eos_token_id_len=1,
            pad_token_id=0,
        )
        e.engine.data_processor = e.data_processor
        e.engine.mm_max_tokens_per_item = None
        captured = []
        monkeypatch.setattr(
            "fastdeploy.engine.engine.subprocess.Popen",
            lambda cmd, **kw: SimpleNamespace(pid=5555) if captured.append(cmd) or True else None,
        )
        monkeypatch.setattr("fastdeploy.engine.engine.current_platform.is_iluvatar", lambda: False)
        e._start_worker_service()
        assert "--enable_expert_parallel" in captured[0]
        assert "--enable_prefix_caching" in captured[0]
        assert "--num_gpu_blocks_override 200" in captured[0]
        assert "--kvcache_storage_backend rocksdb" in captured[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
