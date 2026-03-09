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
