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

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ns(**kw):
    return SimpleNamespace(**kw)


class _FakeSignal:
    """Lightweight stand-in for IPCSignal (no shared memory)."""

    def __init__(self, value=None):
        self.value = value if value is not None else np.zeros([1], dtype=np.int32)
        self.cleared = False

    def clear(self):
        self.cleared = True


class _Recorder:
    """Records calls for later assertions."""

    def __init__(self):
        self.calls = []

    def __call__(self, *a, **kw):
        self.calls.append((a, kw))


def _make_cfg(**overrides):
    """Minimal EngineService cfg matching attribute access patterns."""
    parallel = _ns(
        data_parallel_size=1,
        local_data_parallel_id=0,
        tensor_parallel_size=1,
        local_engine_worker_queue_port=12345,
        engine_worker_queue_port=[12345],
        device_ids="0",
        enable_expert_parallel=False,
        expert_parallel_size=1,
        chunked_moe_size=1,
        disable_custom_all_reduce=False,
        use_internode_ll_two_stage=False,
        disable_sequence_parallel_moe=False,
    )
    model = _ns(
        model="test-model",
        max_model_len=2048,
        num_hidden_layers=2,
        enable_mm=False,
        quantization=None,
        enable_logprob=False,
        lm_head_fp32=False,
        moe_gate_fp32=False,
        enable_entropy=False,
        runner="default",
        convert="default",
        override_pooler_config=None,
        logprobs_mode="default",
        max_logprobs=5,
    )
    cache = _ns(
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        block_size=16,
        gpu_memory_utilization=0.9,
        enc_dec_block_num=0,
        num_gpu_blocks_override=None,
        local_cache_queue_port=0,
        max_block_num_per_seq=128,
        kv_cache_ratio=1.0,
        cache_transfer_protocol="shm",
        kvcache_storage_backend=None,
        num_cpu_blocks=0,
    )
    scheduler = _ns(
        max_num_seqs=32,
        max_num_batched_tokens=4096,
        splitwise_role="mixed",
        name="local",
        enable_overlap_schedule=False,
    )
    cfg = _ns(
        parallel_config=parallel,
        model_config=model,
        cache_config=cache,
        scheduler_config=scheduler,
        master_ip="127.0.0.1",
        host_ip="127.0.0.1",
        worker_num_per_node=1,
        max_prefill_batch=1,
        max_num_partial_prefills=1,
        nnode=1,
        ips=None,
        node_rank=0,
        router_config=_ns(router=None, api_server_host="localhost", api_server_port=8080),
        register_info={},
        structured_outputs_config=_ns(
            guided_decoding_backend="off",
            disable_any_whitespace=False,
            reasoning_parser="default",
            logits_processors=None,
        ),
        load_config=_ns(
            load_strategy="default",
            rsync_config={},
            dynamic_load_weight=False,
            load_choices="default",
        ),
        early_stop_config=_ns(to_json_string=lambda: "{}"),
        speculative_config=_ns(method="none", to_json_string=lambda: "{}"),
        graph_opt_config=_ns(to_json_string=lambda: "{}"),
        plas_attention_config=_ns(to_json_string=lambda: "{}"),
        eplb_config=_ns(enable_eplb=False, to_json_string=lambda: "{}"),
        limit_mm_per_prompt=None,
        mm_processor_kwargs=None,
        tool_parser=None,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_engine(monkeypatch, **cfg_overrides):
    """Create EngineService bypassing __init__ with essential attributes."""
    from fastdeploy.engine.common_engine import EngineService

    eng = object.__new__(EngineService)
    eng.cfg = _make_cfg(**cfg_overrides)
    eng.use_async_llm = False
    eng.running = True
    eng.is_paused = False
    eng._pause_cond = threading.Condition()
    eng.llm_logger = _ns(
        info=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        exception=lambda *a, **kw: None,
    )

    # Resource manager with stop_flags
    eng.resource_manager = _ns(
        stop_flags=np.array([True, True, True, True], dtype=bool),
        check_and_free_block_tables=lambda: None,
        cache_manager=_ns(
            launch_cache_manager=lambda **kw: [],
        ),
    )

    # Scheduler
    eng.scheduler = _ns(
        put_requests=lambda *a: [],
        get_requests=lambda **kw: [],
        put_results=lambda *a: None,
        get_results=lambda: [],
        start=lambda *a, **kw: None,
        reset=lambda: None,
        name="local",
    )

    # IPC signals
    eng.exist_task_signal = _FakeSignal()
    eng.exist_swapped_task_signal = _FakeSignal()
    eng.exist_prefill_task_signal = _FakeSignal()
    eng.worker_healthy_live_signal = _FakeSignal()
    eng.cache_ready_signal = _FakeSignal()
    eng.swap_space_ready_signal = _FakeSignal()
    eng.cache_transfer_inited_signal = _FakeSignal()
    eng.model_weights_status_signal = _FakeSignal()
    eng.prefix_tree_status_signal = _FakeSignal()
    eng.kv_cache_status_signal = _FakeSignal()
    eng.worker_ready_signal = _FakeSignal(np.array([0], dtype=np.int32))
    eng.loaded_model_signal = _FakeSignal()

    # Token processor
    eng.token_processor = _ns(
        clear_data=lambda: None,
        number_of_tasks=0,
        number_of_input_tokens=0,
    )

    # Engine worker queue
    eng.engine_worker_queue = _ns(
        clear_data=lambda: None,
        put_tasks=lambda *a: None,
        exist_tasks=lambda: False,
    )

    # Split connector
    eng.split_connector = _ns(
        start_receiver=lambda: None,
    )

    # Partial chunked tokens (from __init__)
    eng.partial_chunked_tokens = [0, eng.cfg.scheduler_config.max_num_batched_tokens]

    # Ctrl worker output queues
    eng._ctrl_worker_output_queues = []

    return eng


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCommonEngine:
    """Lean pytest-style tests for EngineService methods."""

    # -- task_is_finished / all_tasks_finished --

    def test_task_is_finished_true(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.resource_manager.stop_flags = np.array([True, False], dtype=bool)
        assert eng.task_is_finished(0)

    def test_task_is_finished_false(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.resource_manager.stop_flags = np.array([True, False], dtype=bool)
        assert not eng.task_is_finished(1)

    def test_all_tasks_finished_true(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.resource_manager.stop_flags = np.array([True, True], dtype=bool)
        assert eng.all_tasks_finished()

    def test_all_tasks_finished_false(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.resource_manager.stop_flags = np.array([True, False], dtype=bool)
        assert not eng.all_tasks_finished()

    # -- check_and_free_block_tables --

    def test_check_and_free_block_tables(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        rec = _Recorder()
        eng.resource_manager.check_and_free_block_tables = rec
        eng.check_and_free_block_tables()
        assert len(rec.calls) == 1

    # -- _get_scheduler_unhandled_request_num --

    def test_get_scheduler_unhandled_request_num_callable(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.scheduler.get_unhandled_request_num = lambda: 5
        assert eng._get_scheduler_unhandled_request_num() == 5

    def test_get_scheduler_unhandled_request_num_not_callable(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.scheduler.get_unhandled_request_num = "not_callable"
        assert eng._get_scheduler_unhandled_request_num() == 0

    def test_get_scheduler_unhandled_request_num_negative(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.scheduler.get_unhandled_request_num = lambda: -3
        assert eng._get_scheduler_unhandled_request_num() == 0

    def test_get_scheduler_unhandled_request_num_exception(self, monkeypatch):
        eng = _make_engine(monkeypatch)

        def _raise():
            raise RuntimeError("boom")

        eng.scheduler.get_unhandled_request_num = _raise
        assert eng._get_scheduler_unhandled_request_num() == 0

    # -- check_health --

    def test_check_health_healthy(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.worker_healthy_live_signal.value[0] = time.time()
        ok, msg = eng.check_health()
        assert ok is True
        assert msg == ""

    def test_check_health_unhealthy(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.worker_healthy_live_signal.value = np.array([time.time() - 60], dtype=np.float64)
        ok, msg = eng.check_health(time_interval_threashold=30)
        assert ok is False
        assert "Not Healthy" in msg

    def test_check_health_zero_signal(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.worker_healthy_live_signal.value[0] = 0
        ok, msg = eng.check_health()
        assert ok is True

    # -- _worker_processes_ready --

    def test_worker_processes_ready_true(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.worker_num_per_node = 2
        eng.worker_ready_signal.value = np.array([1, 1], dtype=np.int32)
        assert eng._worker_processes_ready() is True

    def test_worker_processes_ready_false(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.worker_num_per_node = 2
        eng.worker_ready_signal.value = np.array([1, 0], dtype=np.int32)
        assert eng._worker_processes_ready() is False

    # -- _control_resume / _control_is_paused / _control_update_weights --

    def test_control_resume_when_paused(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.is_paused = True
        ctrl = _ns(request_id="r1")
        result = eng._control_resume(ctrl)
        assert eng.is_paused is False
        assert result is None

    def test_control_resume_when_not_paused(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.is_paused = False
        ctrl = _ns(request_id="r1")
        result = eng._control_resume(ctrl)
        assert result is None

    def test_control_is_paused_true(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.is_paused = True
        ctrl = _ns(request_id="r1")
        result = eng._control_is_paused(ctrl)
        assert result == {"is_paused": True}

    def test_control_is_paused_false(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.is_paused = False
        ctrl = _ns(request_id="r1")
        result = eng._control_is_paused(ctrl)
        assert result == {"is_paused": False}

    def test_control_update_weights_not_paused_raises(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.is_paused = False
        ctrl = _ns(request_id="r1")
        with pytest.raises(Exception, match="Pause"):
            eng._control_update_weights(ctrl)

    def test_control_update_weights_paused_calls_worker(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.is_paused = True
        called = []
        eng._call_worker = lambda cr, t: called.append(cr.request_id)
        ctrl = _ns(request_id="r1")
        eng._control_update_weights(ctrl)
        assert called == ["r1"]

    # -- run_control_method --

    def test_run_control_method_unknown(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append((rid, r)))
        ctrl = _ns(request_id="r1", method="nonexistent", params={}, get_method=lambda: "nonexistent")
        eng.run_control_method(ctrl)
        assert len(sent) == 1
        assert sent[0][0] == "r1"

    def test_run_control_method_success(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append((rid, r)))
        eng._control_test = lambda cr: {"ok": True}
        ctrl = _ns(request_id="r2", method="test", params={}, get_method=lambda: "test")
        eng.run_control_method(ctrl)
        assert len(sent) == 1
        assert sent[0][0] == "r2"

    def test_run_control_method_handler_raises(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append((rid, r)))

        def _boom(cr):
            raise ValueError("test error")

        eng._control_boom = _boom
        ctrl = _ns(request_id="r3", method="boom", params={}, get_method=lambda: "boom")
        eng.run_control_method(ctrl)
        assert len(sent) == 1
        assert sent[0][0] == "r3"

    # -- _send_error_response --

    def test_send_error_response_standard(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append((rid, r)))
        eng._send_error_response("req-1", "something broke", 503)
        assert len(sent) == 1
        assert sent[0][0] == "req-1"

    def test_send_error_response_internal_adapter(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append((rid, r)))
        eng._send_error_response("req-2", "adapter error")
        assert len(sent) == 1
        assert sent[0][0] is None  # internal adapter sends None as rid

    # -- _decode_token --

    def test_decode_token_return_text_enabled(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        eng.data_processor = _ns(
            ids2tokens=lambda tids, rid: ("hello", [1, 2, 3], None),
            decode_status={"req1": [0, 2]},
        )
        delta, tids = eng._decode_token([1, 2, 3], "req1", is_end=False)
        assert delta == "hello"
        assert tids == [1, 2]

    def test_decode_token_return_text_end_cleanup(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        eng.data_processor = _ns(
            ids2tokens=lambda tids, rid: ("world", [10, 20], None),
            decode_status={"req2": [0, 1]},
        )
        delta, tids = eng._decode_token([10], "req2", is_end=True)
        assert delta == "world"
        assert "req2" not in eng.data_processor.decode_status

    def test_decode_token_return_text_disabled(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", False)
        delta, tids = eng._decode_token([1, 2], "req3", is_end=False)
        assert delta == ""
        assert tids == [1, 2]

    def test_decode_token_empty_delta(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        eng.data_processor = _ns(
            ids2tokens=lambda tids, rid: ("", [], None),
            decode_status={"req4": [0, 0]},
        )
        delta, tids = eng._decode_token([5], "req4", is_end=False)
        assert delta == ""
        assert tids == []

    # -- clear_data --

    def test_clear_data_success(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        tp_cleared = []
        ewq_cleared = []
        eng.token_processor.clear_data = lambda: tp_cleared.append(1)
        eng.engine_worker_queue.clear_data = lambda: ewq_cleared.append(1)
        eng.send_response_server = _ns(req_dict={})
        eng.recv_request_server = _ns(req_dict={})
        assert eng.clear_data() is True
        assert len(tp_cleared) == 1
        assert len(ewq_cleared) == 1

    def test_clear_data_with_cache_task_queue(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.send_response_server = _ns(req_dict={})
        eng.recv_request_server = _ns(req_dict={})
        cache_cleared = []
        eng.cache_task_queue = _ns(clear_transfer_task=lambda: cache_cleared.append(1))
        assert eng.clear_data() is True
        assert len(cache_cleared) == 1

    def test_clear_data_handles_exception(self, monkeypatch):
        eng = _make_engine(monkeypatch)

        def _boom():
            raise RuntimeError("clear failed")

        eng.token_processor.clear_data = _boom
        assert eng.clear_data() is False

    # -- _setting_environ_variables --

    def test_setting_environ_variables_basic(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        result = eng._setting_environ_variables()
        assert "FLAGS_use_append_attn=1" in result
        assert "OMP_NUM_THREADS=3" in result
        assert "NCCL_ALGO=Ring" in result

    def test_setting_environ_variables_splitwise_prefill(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "prefill"
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        result = eng._setting_environ_variables()
        assert "FLAGS_use_pd_disaggregation=1" in result

    def test_setting_environ_variables_splitwise_v1(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "prefill"
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        result = eng._setting_environ_variables()
        assert "FLAGS_use_pd_disaggregation_per_chunk=1" in result

    def test_setting_environ_variables_mm(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = True
        result = eng._setting_environ_variables()
        assert "FLAGS_max_partition_size=1024" in result

    # -- update_requests_chunk_size --

    def test_update_requests_chunk_size_disabled(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.cache_config.enable_chunked_prefill = False
        reqs = [_ns(prompt_token_ids_len=100)]
        eng.update_requests_chunk_size(reqs)
        assert not hasattr(reqs[0], "prefill_chunk_info")

    def test_update_requests_chunk_size_empty(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.cache_config.enable_chunked_prefill = True
        eng.update_requests_chunk_size([])

    def test_update_requests_chunk_size_single(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.cache_config.enable_chunked_prefill = True
        eng.cfg.cache_config.block_size = 16
        eng.cfg.scheduler_config.max_num_batched_tokens = 128
        eng.partial_chunked_tokens = [0, 128]

        chunk_info = {}
        req = _ns(
            prompt_token_ids_len=64,
            set=lambda key, val: chunk_info.update({key: val}),
        )
        eng.update_requests_chunk_size([req])
        assert "prefill_chunk_info" in chunk_info
        total = sum(chunk_info["prefill_chunk_info"])
        assert total == 64

    def test_update_requests_chunk_size_multi(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.cache_config.enable_chunked_prefill = True
        eng.cfg.cache_config.block_size = 16
        eng.cfg.scheduler_config.max_num_batched_tokens = 256
        eng.cfg.max_num_partial_prefills = 2
        eng.partial_chunked_tokens = [0, 256, 128]

        chunks = [{}, {}]
        reqs = [
            _ns(
                prompt_token_ids_len=100,
                set=lambda key, val, i=i: chunks[i].update({key: val}),
            )
            for i in range(2)
        ]
        eng.update_requests_chunk_size(reqs)
        for c in chunks:
            assert "prefill_chunk_info" in c

    # -- _exit_sub_services --

    def test_exit_sub_services_basic(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng._exit_sub_services()
        assert eng.running is False
        assert eng.exist_task_signal.cleared
        assert eng.exist_swapped_task_signal.cleared
        assert eng.worker_healthy_live_signal.cleared
        assert eng.cache_ready_signal.cleared

    def test_exit_sub_services_with_zmq(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        closed = []
        eng.send_response_server = _ns(close=lambda: closed.append("send"))
        eng.recv_request_server = _ns(close=lambda: closed.append("recv"))
        eng._exit_sub_services()
        assert "send" in closed
        assert "recv" in closed

    def test_exit_sub_services_async_llm(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        eng.worker_proc = None
        eng.worker_ready_signal = _FakeSignal()
        eng.loaded_model_signal = _FakeSignal()
        eng._exit_sub_services()
        assert eng.running is False
        assert eng.worker_ready_signal.cleared
        assert eng.loaded_model_signal.cleared

    def test_exit_sub_services_with_engine_worker_queue_server(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        cleaned = []
        eng.engine_worker_queue_server = _ns(cleanup=lambda: cleaned.append(1))
        eng._exit_sub_services()
        assert len(cleaned) == 1

    # -- start_cache_service --

    def test_start_cache_service(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        launch_kw = {}
        eng.resource_manager.cache_manager.launch_cache_manager = lambda **kw: launch_kw.update(kw) or []
        result = eng.start_cache_service(["0", "1"], 12345)
        assert result == []
        assert launch_kw["tensor_parallel_size"] == 1
        assert launch_kw["device_ids"] == ["0", "1"]

    # -- _init_worker_monitor_signals --

    def test_init_worker_monitor_signals(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.IPCSignal",
            lambda **kw: _FakeSignal(kw.get("array")),
        )
        eng._init_worker_monitor_signals()
        assert hasattr(eng, "exist_task_signal")
        assert hasattr(eng, "exist_swapped_task_signal")
        assert hasattr(eng, "exist_prefill_task_signal")
        assert hasattr(eng, "worker_healthy_live_signal")
        assert hasattr(eng, "cache_ready_signal")
        assert hasattr(eng, "swap_space_ready_signal")
        assert hasattr(eng, "cache_transfer_inited_signal")
        assert hasattr(eng, "model_weights_status_signal")
        assert hasattr(eng, "prefix_tree_status_signal")
        assert hasattr(eng, "kv_cache_status_signal")

    # -- _init_worker_signals --

    def test_init_worker_signals_basic(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.ipc_signal_suffix = 12345
        eng.do_profile = 0
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.IPCSignal",
            lambda **kw: _FakeSignal(kw.get("array")),
        )
        eng._init_worker_signals()
        assert hasattr(eng, "worker_ready_signal")
        assert hasattr(eng, "loaded_model_signal")
        assert not hasattr(eng, "get_profile_block_num_signal")

    def test_init_worker_signals_with_profile(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.ipc_signal_suffix = 12345
        eng.do_profile = 1
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.IPCSignal",
            lambda **kw: _FakeSignal(kw.get("array")),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.paddle.is_compiled_with_custom_device", lambda x: False)
        eng._init_worker_signals()
        assert hasattr(eng, "get_profile_block_num_signal")

    def test_init_worker_signals_prefix_caching(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.ipc_signal_suffix = 12345
        eng.do_profile = 0
        eng.cfg.cache_config.enable_prefix_caching = True
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.IPCSignal",
            lambda **kw: _FakeSignal(kw.get("array")),
        )
        eng._init_worker_signals()
        assert hasattr(eng, "launched_cache_manager_signal")

    def test_init_worker_signals_expert_parallel(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.ipc_signal_suffix = 12345
        eng.do_profile = 0
        eng.cfg.parallel_config.enable_expert_parallel = True
        eng.cfg.parallel_config.data_parallel_size = 2
        eng.cfg.nnode = 1
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.IPCSignal",
            lambda **kw: _FakeSignal(kw.get("array")),
        )
        eng._init_worker_signals()
        assert hasattr(eng, "launched_expert_service_signal")

    # -- launch_components --

    def test_launch_components_mixed(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.scheduler_config.name = "local"
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.envs.FD_ENABLE_MULTI_API_SERVER",
            False,
        )
        eng.launch_components()
        assert not hasattr(eng, "splitwise_receive_thread")

    def test_launch_components_prefill(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "prefill"
        eng.cfg.scheduler_config.name = "local"
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.envs.FD_ENABLE_MULTI_API_SERVER",
            False,
        )
        eng.launch_components()
        assert hasattr(eng, "splitwise_receive_thread")

    def test_launch_components_splitwise_scheduler(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "prefill"
        eng.cfg.scheduler_config.name = "splitwise"
        started = []
        eng.scheduler.start = lambda *a, **kw: started.append(a)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.envs.FD_ENABLE_MULTI_API_SERVER",
            False,
        )
        eng.launch_components()
        assert len(started) == 1
        assert started[0][0] == "prefill"

    def test_launch_components_dp_scheduler(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.scheduler_config.name = "dp"
        started = []
        eng.scheduler.start = lambda *a, **kw: started.append(a)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.envs.FD_ENABLE_MULTI_API_SERVER",
            False,
        )
        eng.launch_components()
        assert len(started) == 1

    # -- _stop_profile --

    def test_stop_profile(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.do_profile = 1
        eng.get_profile_block_num_signal = _FakeSignal(np.array([100], dtype=np.int32))
        eng.worker_proc = None
        reset_calls = []
        eng.cfg.cache_config.reset = lambda n: reset_calls.append(n)
        eng.resource_manager.reset_cache_config = lambda cc: None
        eng.cfg.cache_config.enable_prefix_caching = False
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.ipc_signal_suffix = 12345
        eng._stop_profile()
        assert eng.do_profile == 0
        assert reset_calls == [100]

    # -- _register_to_router --

    def test_register_to_router_disabled(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.router_config.router = None
        eng._register_to_router()

    # -- start --

    def test_start_sets_running(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = False
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER",
            False,
        )
        eng.token_processor.tasks_queue = None
        eng.token_processor.run = lambda: None
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.router_config.router = None
        eng._schedule_request_to_worker = lambda: None
        eng.start()
        assert eng.running is True
        assert hasattr(eng, "insert_task_to_worker_thread")

    # -- start_worker_queue_service (no queue) --

    def test_start_worker_queue_service(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM",
            False,
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.EngineWorkerQueue",
            lambda **kw: _ns(
                get_server_port=lambda: 12345,
                cleanup=lambda: None,
            ),
        )
        eng.start_worker_queue_service(start_queue=False)
        assert hasattr(eng, "engine_worker_queue")

    # -- check_health edge case (int signal) --

    def test_check_health_int_signal_recent(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.worker_healthy_live_signal.value = np.array([int(time.time())], dtype=np.int32)
        ok, msg = eng.check_health(time_interval_threashold=30)
        assert ok is True

    # -- _control_pause --

    def test_control_pause_happy_path(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        eng.cfg.scheduler_config.name = "local"
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=lambda *a: None, clear_data=lambda: None)
        eng.resource_manager.log_status = lambda: None
        eng.resource_manager.preempted_all = lambda: []
        eng.resource_manager.cache_manager = _ns(reset=lambda: None)
        eng.scheduler.get_inflight_requests = lambda: []
        ctrl = _ns(request_id="r-pause")
        result = eng._control_pause(ctrl)
        assert result is None
        assert eng.is_paused is True

    def test_control_pause_already_paused(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        eng.cfg.scheduler_config.name = "local"
        eng.is_paused = True
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=lambda *a: None, clear_data=lambda: None)
        eng.resource_manager.log_status = lambda: None
        eng.resource_manager.preempted_all = lambda: []
        eng.resource_manager.cache_manager = _ns(reset=lambda: None)
        eng.scheduler.get_inflight_requests = lambda: []
        result = eng._control_pause(_ns(request_id="r2"))
        assert result is None

    def test_control_pause_with_running_reqs(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        eng.cfg.scheduler_config.name = "local"
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=lambda *a: None, clear_data=lambda: None)
        eng.resource_manager.log_status = lambda: None
        preempted_tasks = [_ns(task_type="PREEMPTED")]
        eng.resource_manager.preempted_all = lambda: preempted_tasks
        eng.resource_manager.get_real_bsz = lambda: None
        eng.resource_manager.real_bsz = 1
        eng.resource_manager.wait_worker_inflight_requests_finish = lambda timeout: None
        eng.resource_manager.cache_manager = _ns(reset=lambda: None)
        eng.scheduler.get_inflight_requests = lambda: []
        eng._control_pause(_ns(request_id="r3"))
        assert eng.is_paused is True

    def test_control_pause_with_inflight_requests(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        eng.cfg.scheduler_config.name = "local"
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=lambda *a: None, clear_data=lambda: None)
        eng.resource_manager.log_status = lambda: None
        eng.resource_manager.preempted_all = lambda: []
        eng.resource_manager.cache_manager = _ns(reset=lambda: None)
        sent_errors = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent_errors.append(rid))
        inflight = [_ns(request_id="req-inflight")]
        eng.scheduler.get_inflight_requests = lambda: inflight
        eng._control_pause(_ns(request_id="r4"))
        assert len(sent_errors) == 1

    def test_control_pause_not_v1_raises(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        with pytest.raises(Exception, match="pause only supported"):
            eng._control_pause(_ns(request_id="r5"))

    def test_control_pause_not_local_raises(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        eng.cfg.scheduler_config.name = "dp"
        with pytest.raises(Exception, match="pause only supported in local"):
            eng._control_pause(_ns(request_id="r6"))

    # -- insert_tasks --

    def test_insert_tasks_mixed_role(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.cache_config.enable_chunked_prefill = False
        eng.cfg.model_config.enable_mm = False
        eng.resource_manager.check_and_free_block_tables = lambda: None
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        put_calls = []
        eng.engine_worker_queue = _ns(put_tasks=lambda t: put_calls.append(t))
        task = _ns(
            request_id="r1",
            trace_carrier=None,
            prompt_token_ids_len=32,
            metrics=_ns(
                inference_start_time=0,
                scheduler_recv_req_time=time.time(),
                add_req_to_resource_manager_time=0,
                ask_decode_resource_start_time=0,
                ask_decode_resource_finish_time=0,
            ),
            disaggregate_info=None,
            has_been_preempted_before=False,
            set=lambda k, v: None,
            user="test",
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *a: None
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", lambda *a: None
        )
        result = eng.insert_tasks([task])
        assert result is True
        assert len(put_calls) == 1

    def test_insert_tasks_exceeds_batch(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.cache_config.enable_chunked_prefill = False
        eng.cfg.model_config.enable_mm = False
        eng.resource_manager.stop_flags = np.array([True, False, False, False], dtype=bool)
        eng.resource_manager.check_and_free_block_tables = lambda: None
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        put_calls = []
        eng.engine_worker_queue = _ns(put_tasks=lambda t: put_calls.append(t))
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *a: None
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", lambda *a: None
        )
        tasks = [
            _ns(
                request_id=f"r{i}",
                trace_carrier=None,
                prompt_token_ids_len=32,
                metrics=_ns(
                    inference_start_time=0,
                    scheduler_recv_req_time=time.time(),
                    add_req_to_resource_manager_time=0,
                    ask_decode_resource_start_time=0,
                    ask_decode_resource_finish_time=0,
                ),
                disaggregate_info=None,
                has_been_preempted_before=False,
                set=lambda k, v: None,
                user="test",
            )
            for i in range(3)
        ]
        result = eng.insert_tasks(tasks)
        assert result is True
        # Only 1 task should pass (available_batch=1)
        assert len(put_calls[0][0]) == 1

    def test_insert_tasks_allocation_fails(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.resource_manager.check_and_free_block_tables = lambda: None
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: []
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *a: None
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", lambda *a: None
        )
        task = _ns(
            request_id="r1",
            trace_carrier=None,
            prompt_token_ids_len=32,
            metrics=_ns(
                inference_start_time=0,
                scheduler_recv_req_time=time.time(),
                add_req_to_resource_manager_time=0,
            ),
            disaggregate_info=None,
            has_been_preempted_before=False,
            set=lambda k, v: None,
            user="test",
        )
        from fastdeploy.engine.common_engine import EngineError

        with pytest.raises(EngineError):
            eng.insert_tasks([task])

    def test_insert_tasks_prefill_role_success(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "prefill"
        eng.cfg.cache_config.enable_chunked_prefill = False
        eng.cfg.model_config.enable_mm = False
        eng.resource_manager.check_and_free_block_tables = lambda: None
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        eng.split_connector = _ns(
            check_decode_allocated=lambda t: (True, None),
            send_cache_info_to_messager=lambda *a: None,
        )
        put_calls = []
        eng.engine_worker_queue = _ns(put_tasks=lambda t: put_calls.append(t))
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *a: None
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", lambda *a: None
        )
        task = _ns(
            request_id="r-pf",
            trace_carrier=None,
            prompt_token_ids_len=32,
            metrics=_ns(
                inference_start_time=0,
                scheduler_recv_req_time=time.time(),
                add_req_to_resource_manager_time=0,
                ask_decode_resource_start_time=0,
                ask_decode_resource_finish_time=0,
            ),
            disaggregate_info=_ns(foo=1),
            has_been_preempted_before=False,
            set=lambda k, v: None,
            user="test",
        )
        result = eng.insert_tasks([task])
        assert result is True

    def test_insert_tasks_prefill_decode_fails(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "prefill"
        eng.resource_manager.check_and_free_block_tables = lambda: None
        eng.split_connector = _ns(
            check_decode_allocated=lambda t: (False, "D failed"),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *a: None
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", lambda *a: None
        )
        put_results = []
        eng.scheduler.put_results = lambda r: put_results.extend(r)
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        eng.engine_worker_queue = _ns(put_tasks=lambda t: None)
        task = _ns(
            request_id="r-fail",
            trace_carrier=None,
            prompt_token_ids_len=32,
            metrics=_ns(
                inference_start_time=0,
                scheduler_recv_req_time=time.time(),
                add_req_to_resource_manager_time=0,
                ask_decode_resource_start_time=0,
                ask_decode_resource_finish_time=0,
            ),
            disaggregate_info=None,
            has_been_preempted_before=False,
            set=lambda k, v: None,
            user="test",
        )
        # Task removed due to decode alloc failure, allocate gets empty→raises
        from fastdeploy.engine.common_engine import EngineError

        with pytest.raises(EngineError):
            eng.insert_tasks([task])
        assert len(put_results) == 1

    def test_insert_tasks_with_preempted(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.cache_config.enable_chunked_prefill = False
        eng.cfg.model_config.enable_mm = False
        eng.resource_manager.check_and_free_block_tables = lambda: None
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        put_calls = []
        eng.engine_worker_queue = _ns(put_tasks=lambda t: put_calls.append(t))
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *a: None
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", lambda *a: None
        )
        task = _ns(
            request_id="r-pre",
            trace_carrier=None,
            prompt_token_ids_len=32,
            metrics=_ns(
                inference_start_time=0,
                scheduler_recv_req_time=time.time(),
                add_req_to_resource_manager_time=0,
            ),
            disaggregate_info=None,
            has_been_preempted_before=True,
            set=lambda k, v: None,
            user="test",
        )
        result = eng.insert_tasks([task])
        assert result is True

    def test_insert_tasks_decode_role(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "decode"
        eng.cfg.cache_config.enable_chunked_prefill = False
        eng.cfg.model_config.enable_mm = False
        eng.resource_manager.check_and_free_block_tables = lambda: None
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        eng.split_connector = _ns(send_cache_info_to_prefill=lambda *a: None)
        put_calls = []
        eng.engine_worker_queue = _ns(put_tasks=lambda t: put_calls.append(t))
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *a: None
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", lambda *a: None
        )
        task = _ns(
            request_id="r-dec",
            trace_carrier=None,
            prompt_token_ids_len=32,
            metrics=_ns(
                inference_start_time=0,
                scheduler_recv_req_time=time.time(),
                add_req_to_resource_manager_time=0,
                decode_inference_start_time=0,
            ),
            disaggregate_info=_ns(foo=1),
            has_been_preempted_before=False,
            set=lambda k, v: None,
            user="test",
        )
        result = eng.insert_tasks([task])
        assert result is True

    # -- _insert_prefilled_requests --

    def test_insert_prefilled_requests_happy(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.speculative_config = _ns(method="none")
        eng.cfg.scheduler_config.splitwise_role = "decode"
        eng.resource_manager.req_dict = {"req1": 0}
        eng.resource_manager.tasks_list = [
            _ns(
                prompt_token_ids=[0],
                num_cached_tokens=0,
                metrics=_ns(decode_recv_req_time=0, decode_preallocate_req_time=0, decode_inference_start_time=0),
            )
        ]
        eng.resource_manager.stop_flags = np.array([False], dtype=bool)
        eng.token_processor = _ns(
            tokens_counter={}, clear_data=lambda: None, number_of_tasks=0, number_of_input_tokens=0
        )
        put_calls = []
        eng.engine_worker_queue = _ns(put_tasks=lambda t: put_calls.append(t))
        eng.resource_manager.real_bsz = 1
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        req_out = _ns(
            request_id="req1",
            outputs=_ns(token_ids=[42], draft_token_ids=None),
            error_code=200,
            error_msg=None,
            num_cached_tokens=5,
            metrics=_ns(decode_recv_req_time=0, decode_preallocate_req_time=0, decode_inference_start_time=0),
        )
        result = eng._insert_prefilled_requests([req_out])
        assert result is True
        assert len(put_calls) == 1

    def test_insert_prefilled_requests_error(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.speculative_config = _ns(method="none")
        eng.cfg.scheduler_config.splitwise_role = "decode"
        eng.resource_manager.req_dict = {"req-e": 0}
        eng.resource_manager.tasks_list = [
            _ns(
                prompt_token_ids=[0],
                num_cached_tokens=0,
                metrics=_ns(decode_recv_req_time=0, decode_preallocate_req_time=0, decode_inference_start_time=0),
            )
        ]
        eng.resource_manager.stop_flags = np.array([False], dtype=bool)
        eng.resource_manager._recycle_block_tables = lambda r: None
        eng.token_processor = _ns(tokens_counter={"req-e": 1}, clear_data=lambda: None, number_of_tasks=0)
        put_results = []
        eng.scheduler.put_results = lambda r: put_results.extend(r)
        eng.engine_worker_queue = _ns(put_tasks=lambda t: None)
        eng.resource_manager.real_bsz = 1
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        req_out = _ns(
            request_id="req-e",
            outputs=_ns(token_ids=[42], draft_token_ids=None),
            error_code=500,
            error_msg="prefill error",
            num_cached_tokens=0,
            metrics=_ns(decode_recv_req_time=0, decode_preallocate_req_time=0, decode_inference_start_time=0),
        )
        eng._insert_prefilled_requests([req_out])
        assert eng.resource_manager.stop_flags[0]  # noqa: E712
        assert len(put_results) == 1

    def test_insert_prefilled_requests_internal_adapter_eos(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.speculative_config = _ns(method="none")
        eng.cfg.scheduler_config.splitwise_role = "decode"
        eng.resource_manager.req_dict = {"req-eos": 0}
        eng.resource_manager.tasks_list = [
            _ns(
                prompt_token_ids=[0],
                num_cached_tokens=0,
                metrics=_ns(decode_recv_req_time=0, decode_preallocate_req_time=0, decode_inference_start_time=0),
            )
        ]
        eng.resource_manager.stop_flags = np.array([False], dtype=bool)
        eng.resource_manager._recycle_block_tables = lambda r: None
        eng.token_processor = _ns(tokens_counter={"req-eos": 1}, clear_data=lambda: None, number_of_tasks=0)
        eng.engine_worker_queue = _ns(put_tasks=lambda t: None)
        eng.resource_manager.real_bsz = 1
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        req_out = _ns(
            request_id="req-eos",
            outputs=_ns(token_ids=[], draft_token_ids=None),
            error_code=200,
            error_msg=None,
            num_cached_tokens=0,
            metrics=_ns(decode_recv_req_time=0, decode_preallocate_req_time=0, decode_inference_start_time=0),
        )
        eng._insert_prefilled_requests([req_out])
        # EOS triggers recycle, stop_flags set True
        assert eng.resource_manager.stop_flags[0]  # noqa: E712

    # -- _start_worker_service --

    def test_start_worker_service_builds_cmd(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.data_processor = _ns(
            tokenizer=_ns(
                vocab={"a": 0, "b": 1},
                get_vocab=lambda: {"<think>": 5, "</think>": 6, "\n": 10},
                encode=lambda *a, **kw: [10],
            ),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        eng.mm_max_tokens_per_item = None
        eng.do_profile = 0
        eng.ipc_signal_suffix = 12345
        captured = {}

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd
            return _ns(pid=9999, poll=lambda: None, stdout=iter([]))

        monkeypatch.setattr("fastdeploy.engine.common_engine.subprocess.Popen", fake_popen)
        result = eng._start_worker_service()
        assert "cmd" in captured
        assert "--max_num_seqs" in captured["cmd"]
        assert "--model test-model" in captured["cmd"]
        assert result.pid == 9999

    def test_start_worker_service_sp_model_path(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.data_processor = _ns(
            tokenizer=_ns(
                sp_model=type("SP", (), {"__len__": lambda s: 50})(),
                get_vocab=lambda: {},
                encode=lambda *a, **kw: [10],
            ),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        eng.mm_max_tokens_per_item = None
        eng.do_profile = 0
        eng.ipc_signal_suffix = 12345
        captured = {}

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd
            return _ns(pid=9999, poll=lambda: None, stdout=iter([]))

        monkeypatch.setattr("fastdeploy.engine.common_engine.subprocess.Popen", fake_popen)
        eng._start_worker_service()
        assert "--ori_vocab_size 50" in captured["cmd"]

    def test_start_worker_service_store_true_flags(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.cache_config.enable_prefix_caching = True
        eng.cfg.cache_config.enable_chunked_prefill = True
        eng.data_processor = _ns(
            tokenizer=_ns(
                vocab={"a": 0},
                get_vocab=lambda: {},
                encode=lambda *a, **kw: [10],
            ),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        eng.mm_max_tokens_per_item = None
        eng.do_profile = 1
        eng.ipc_signal_suffix = 12345
        captured = {}

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd
            return _ns(pid=9999, poll=lambda: None, stdout=iter([]))

        monkeypatch.setattr("fastdeploy.engine.common_engine.subprocess.Popen", fake_popen)
        eng._start_worker_service()
        assert "--enable_prefix_caching" in captured["cmd"]
        assert "--enable_chunked_prefill" in captured["cmd"]
        assert "--do_profile" in captured["cmd"]

    def test_start_worker_service_nnode_gt_1(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.nnode = 2
        eng.cfg.ips = ["10.0.0.1", "10.0.0.2"]
        eng.data_processor = _ns(
            tokenizer=_ns(
                vocab={"a": 0},
                get_vocab=lambda: {},
                encode=lambda *a, **kw: [10],
            ),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        eng.mm_max_tokens_per_item = None
        eng.do_profile = 0
        eng.ipc_signal_suffix = 12345
        captured = {}

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd
            return _ns(pid=9999, poll=lambda: None, stdout=iter([]))

        monkeypatch.setattr("fastdeploy.engine.common_engine.subprocess.Popen", fake_popen)
        eng._start_worker_service()
        assert "--nnodes 2" in captured["cmd"]

    def test_start_worker_service_mm_tokens(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.data_processor = _ns(
            tokenizer=_ns(
                vocab={"a": 0},
                get_vocab=lambda: {},
                encode=lambda *a, **kw: [10],
            ),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        eng.mm_max_tokens_per_item = {"image": 128}
        eng.do_profile = 0
        eng.ipc_signal_suffix = 12345
        captured = {}

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd
            return _ns(pid=9999, poll=lambda: None, stdout=iter([]))

        monkeypatch.setattr("fastdeploy.engine.common_engine.subprocess.Popen", fake_popen)
        eng._start_worker_service()
        assert "--mm_max_tokens_per_item" in captured["cmd"]

    def test_start_worker_service_logits_processors(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.structured_outputs_config.logits_processors = ["proc1", "proc2"]
        eng.data_processor = _ns(
            tokenizer=_ns(
                vocab={"a": 0},
                get_vocab=lambda: {},
                encode=lambda *a, **kw: [10],
            ),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        eng.mm_max_tokens_per_item = None
        eng.do_profile = 0
        eng.ipc_signal_suffix = 12345
        captured = {}

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd
            return _ns(pid=9999, poll=lambda: None, stdout=iter([]))

        monkeypatch.setattr("fastdeploy.engine.common_engine.subprocess.Popen", fake_popen)
        eng._start_worker_service()
        assert "--logits-processors proc1 proc2" in captured["cmd"]

    def test_start_worker_service_gpu_blocks_override(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.cache_config.num_gpu_blocks_override = 512
        eng.data_processor = _ns(
            tokenizer=_ns(
                vocab={"a": 0},
                get_vocab=lambda: {},
                encode=lambda *a, **kw: [10],
            ),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        eng.mm_max_tokens_per_item = None
        eng.do_profile = 0
        eng.ipc_signal_suffix = 12345
        captured = {}

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd
            return _ns(pid=9999, poll=lambda: None, stdout=iter([]))

        monkeypatch.setattr("fastdeploy.engine.common_engine.subprocess.Popen", fake_popen)
        eng._start_worker_service()
        assert "--num_gpu_blocks_override 512" in captured["cmd"]

    # -- check_worker_initialize_status --

    def test_check_worker_status_success(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.worker_num_per_node = 1
        eng.worker_ready_signal = _FakeSignal(np.array([0], dtype=np.int32))
        eng.worker_init_status = {}
        stdout_lines = [
            b"Loading checkpoint shards: 100\n",
            b"Start load layer 0\n",
            b"Start load layer 1\n",
        ]
        eng.worker_proc = _ns(
            stdout=iter(stdout_lines),
            poll=lambda: None,
        )

        # Simulate worker becoming ready after a short delay
        def set_ready():
            time.sleep(0.1)
            eng.worker_ready_signal.value[0] = 1

        threading.Thread(target=set_ready, daemon=True).start()
        result = eng.check_worker_initialize_status()
        assert result is True

    def test_check_worker_status_proc_dies(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.worker_num_per_node = 1
        eng.cfg.model_config.num_hidden_layers = 2
        eng.worker_ready_signal = _FakeSignal(np.array([0], dtype=np.int32))
        eng.worker_init_status = {}
        eng.worker_proc = _ns(
            stdout=iter([]),
            poll=lambda: 1,  # process exited
        )
        result = eng.check_worker_initialize_status()
        assert result is False

    # -- _schedule_request_to_worker (v0) --

    def test_schedule_request_to_worker_v0_one_iter(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.resource_manager.available_batch = lambda: 1
        eng.resource_manager.available_block_num = lambda: 100
        eng.resource_manager.abort_req_ids_set = set()
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *a: None
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", lambda *a: None
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.num_requests_waiting", _ns(dec=lambda *a: None)
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.num_requests_running", _ns(inc=lambda *a: None)
        )
        eng.engine_worker_queue = _ns(
            exist_tasks=lambda: False,
            num_cache_infos=lambda: 0,
            put_tasks=lambda *a: None,
        )
        eng.split_connector = _ns(current_request_ids=[], send_splitwise_tasks=lambda *a: None)
        task = _ns(
            request_id="v0-req",
            trace_carrier=None,
            prompt_token_ids_len=32,
            metrics=_ns(
                engine_get_req_time=0,
                inference_start_time=0,
                scheduler_recv_req_time=time.time(),
                ask_decode_resource_start_time=0,
                ask_decode_resource_finish_time=0,
                add_req_to_resource_manager_time=0,
            ),
            disaggregate_info=None,
            has_been_preempted_before=False,
            set=lambda k, v: None,
            user="test",
        )

        call_count = [0]

        def fake_get_requests(**kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return [task]
            eng.running = False
            return []

        eng.scheduler.get_requests = fake_get_requests
        eng.resource_manager.check_and_free_block_tables = lambda: None
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        eng._schedule_request_to_worker()
        assert call_count[0] >= 1

    def test_schedule_request_to_worker_v0_no_batch(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        call_count = [0]

        def fake_avail():
            call_count[0] += 1
            if call_count[0] >= 2:
                eng.running = False
            return 0

        eng.resource_manager.available_batch = fake_avail
        eng._schedule_request_to_worker()
        assert call_count[0] >= 2

    # -- _schedule_request_to_worker_v1 --

    def test_schedule_request_to_worker_v1_mixed_happy(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *a: None
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", lambda *a: None
        )
        eng.engine_worker_queue = _ns(
            exist_tasks=lambda: False,
            put_tasks=lambda *a: None,
        )
        eng.resource_manager.waiting = []
        eng.resource_manager.available_batch = lambda: 1
        eng.resource_manager.available_block_num = lambda: 100
        eng.resource_manager.abort_req_ids_set = set()
        eng.resource_manager.add_request = lambda t: None

        task = _ns(
            request_id="v1-req",
            trace_carrier=None,
            prompt_token_ids_len=32,
            metrics=_ns(
                engine_get_req_time=0,
                inference_start_time=0,
                scheduler_recv_req_time=time.time(),
                add_req_to_resource_manager_time=0,
            ),
            has_been_preempted_before=False,
            task_type=None,
            user="test",
        )

        sched_call = [0]

        def fake_get_requests(**kw):
            return [task]

        eng.scheduler.get_requests = fake_get_requests

        def fake_schedule():
            sched_call[0] += 1
            if sched_call[0] >= 2:
                eng.running = False
            return [], []

        eng.resource_manager.schedule = fake_schedule
        eng._schedule_request_to_worker_v1()
        assert sched_call[0] >= 1

    def test_schedule_request_to_worker_v1_tasks_and_errors(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *a: None
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", lambda *a: None
        )
        put_calls = []
        eng.engine_worker_queue = _ns(
            exist_tasks=lambda: False,
            put_tasks=lambda *a: put_calls.append(a),
        )
        eng.resource_manager.waiting = []
        eng.resource_manager.get_real_bsz = lambda: None
        eng.resource_manager.real_bsz = 1
        eng.resource_manager.abort_req_ids_set = set()
        eng.resource_manager.available_batch = lambda: 1
        eng.resource_manager.available_block_num = lambda: 100
        eng.resource_manager.add_request = lambda t: None

        eng.scheduler.get_requests = lambda **kw: []

        sched_task = _ns(
            request_id="sched-1",
            task_type=None,
            trace_carrier=None,
            metrics=_ns(
                inference_start_time=0,
                scheduler_recv_req_time=time.time(),
                decode_inference_start_time=0,
            ),
            has_been_preempted_before=False,
            user="test",
        )
        sent_errors = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent_errors.append(rid))

        call_count = [0]

        def fake_schedule():
            call_count[0] += 1
            if call_count[0] == 1:
                return [sched_task], [("err-req", "some error")]
            eng.running = False
            return [], []

        eng.resource_manager.schedule = fake_schedule
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        eng._schedule_request_to_worker_v1()
        assert len(put_calls) >= 1
        assert len(sent_errors) >= 1

    def test_schedule_request_to_worker_v1_shutdown(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False)
        eng.resource_manager.waiting = []

        def fake_schedule():
            raise RuntimeError("cannot schedule new futures after shutdown")

        eng.resource_manager.schedule = fake_schedule
        eng._schedule_request_to_worker_v1()  # should break cleanly

    def test_schedule_request_to_worker_v1_decode_preempted(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "decode"
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *a: None
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", lambda *a: None
        )
        put_calls = []
        eng.engine_worker_queue = _ns(
            exist_tasks=lambda: False,
            put_tasks=lambda *a: put_calls.append(a),
        )
        eng.resource_manager.available_batch = lambda: 1
        eng.resource_manager.available_block_num = lambda: 100
        eng.resource_manager.abort_req_ids_set = set()
        eng.resource_manager.get_real_bsz = lambda: None
        eng.resource_manager.real_bsz = 1

        from fastdeploy.engine.common_engine import RequestType

        sched_task = _ns(
            request_id="dec-pre",
            task_type=RequestType.PREEMPTED,
            trace_carrier=None,
            metrics=_ns(
                inference_start_time=0,
                scheduler_recv_req_time=time.time(),
                decode_inference_start_time=0,
            ),
            has_been_preempted_before=False,
            user="test",
        )
        put_results = []
        eng.scheduler.put_results = lambda r: put_results.extend(r)
        eng.scheduler.get_requests = lambda **kw: []

        call_count = [0]

        def fake_schedule():
            call_count[0] += 1
            if call_count[0] == 1:
                return [sched_task], []
            eng.running = False
            return [], []

        eng.resource_manager.schedule = fake_schedule
        eng._schedule_request_to_worker_v1()
        assert len(put_results) >= 1

    def test_schedule_request_to_worker_v1_prefill_tasks(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *a: None
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", lambda *a: None
        )
        put_calls = []
        eng.engine_worker_queue = _ns(
            exist_tasks=lambda: False,
            put_tasks=lambda *a: put_calls.append(a),
        )
        eng.resource_manager.waiting = []
        eng.resource_manager.get_real_bsz = lambda: None
        eng.resource_manager.real_bsz = 1
        eng.resource_manager.abort_req_ids_set = set()
        eng.resource_manager.available_batch = lambda: 1
        eng.resource_manager.available_block_num = lambda: 100
        eng.resource_manager.add_request = lambda t: None

        from fastdeploy.engine.common_engine import RequestType

        sched_task = _ns(
            request_id="pf-task_0",
            task_type=RequestType.PREFILL,
            trace_carrier={"trace": "data"},
            metrics=_ns(
                inference_start_time=0,
                scheduler_recv_req_time=time.time(),
                decode_inference_start_time=0,
            ),
            has_been_preempted_before=False,
            user="test",
        )
        # Not a Request instance → no has_been_preempted_before check on isinstance
        eng.scheduler.get_requests = lambda **kw: []

        call_count = [0]

        def fake_schedule():
            call_count[0] += 1
            if call_count[0] == 1:
                return [sched_task], []
            eng.running = False
            return [], []

        eng.resource_manager.schedule = fake_schedule
        eng._schedule_request_to_worker_v1()
        assert len(put_calls) >= 1

    # -- start_zmq_service --

    def test_start_zmq_service_ipc(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        created_servers = []

        class FakeZmqIpc:
            def __init__(self, **kw):
                self.kw = kw
                created_servers.append(self)

            def recv_result_handle(self):
                pass

        monkeypatch.setattr("fastdeploy.engine.common_engine.ZmqIpcServer", FakeZmqIpc)
        eng.running = True
        eng._insert_zmq_task_to_scheduler = lambda: None
        eng._zmq_send_generated_tokens = lambda: None
        eng.start_zmq_service(api_server_pid=54321)
        assert eng.api_server_pid == 54321
        assert len(created_servers) == 2

    def test_start_zmq_service_none_pid(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.start_zmq_service(api_server_pid=None)
        assert not hasattr(eng, "api_server_pid")

    def test_start_zmq_service_internal_adapter(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ZMQ_RECV_REQUEST_SERVER_PORT", 5555)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ZMQ_SEND_RESPONSE_SERVER_PORT", 5556)
        created_servers = []

        class FakeZmqTcp:
            def __init__(self, **kw):
                created_servers.append(self)

            def recv_result_handle(self):
                pass

        class FakeInternalAdapter:
            def __init__(self, **kw):
                pass

        monkeypatch.setattr("fastdeploy.engine.common_engine.ZmqTcpServer", FakeZmqTcp)
        monkeypatch.setattr("fastdeploy.engine.common_engine.InternalAdapter", FakeInternalAdapter)
        eng.running = True
        eng._insert_zmq_task_to_scheduler = lambda: None
        eng._zmq_send_generated_tokens = lambda: None
        eng.start_zmq_service(api_server_pid=54321)
        assert len(created_servers) == 2
        assert hasattr(eng, "internal_adapter")

    # -- _insert_zmq_task_to_scheduler --

    def test_insert_zmq_task_normal_request(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.requests_number", _ns(inc=lambda *a: None)
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.num_requests_waiting", _ns(inc=lambda *a: None)
        )
        eng.guided_decoding_checker = None

        call_count = [0]
        received_data = {
            "request_id": "zmq-req-1",
            "user": "tester",
        }

        fake_request = _ns(
            request_id="zmq-req-1",
            metrics=_ns(scheduler_recv_req_time=0),
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.Request.from_dict",
            lambda d: fake_request,
        )

        def fake_receive(block):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return None, received_data
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_json_once=fake_receive)
        eng.scheduler.put_requests = lambda t: []
        eng._insert_zmq_task_to_scheduler()
        assert call_count[0] == 2

    def test_insert_zmq_task_abort_request(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)

        from fastdeploy.engine.common_engine import RequestStatus

        abort_data = {
            "request_id": "abort-1",
            "status": RequestStatus.ABORT.value,
        }
        call_count = [0]

        def fake_receive(block):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return None, abort_data
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_json_once=fake_receive)
        eng.resource_manager.abort_req_ids_set = set()
        eng._insert_zmq_task_to_scheduler()
        assert "abort-1" in eng.resource_manager.abort_req_ids_set

    def test_insert_zmq_task_control_request(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)

        ctrl_data = {"request_id": "ctrl-1", "method": "is_paused", "params": {}}
        monkeypatch.setattr("fastdeploy.engine.common_engine.ControlRequest.is_control_request", lambda d: True)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.ControlRequest.from_dict",
            lambda d: _ns(request_id="ctrl-1", method="is_paused", params={}, get_method=lambda: "is_paused"),
        )
        ctrl_called = []
        eng.run_control_method = lambda cr: ctrl_called.append(cr.method)

        call_count = [0]

        def fake_receive(block):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return None, ctrl_data
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_json_once=fake_receive)
        eng._insert_zmq_task_to_scheduler()
        assert len(ctrl_called) == 1

    def test_insert_zmq_task_error_reconnects(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        eng.api_server_pid = 12345

        call_count = [0]
        reconnected = []

        class FakeZmqIpc:
            def __init__(self, **kw):
                reconnected.append(1)

            def receive_json_once(self, block):
                eng.running = False
                return "Context was terminated", None

        def fake_receive(block):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return "Non-context error", None
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_json_once=fake_receive)
        monkeypatch.setattr("fastdeploy.engine.common_engine.ZmqIpcServer", FakeZmqIpc)
        eng._insert_zmq_task_to_scheduler()
        assert len(reconnected) >= 1

    def test_insert_zmq_task_paused_drops(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.is_paused = True
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.requests_number", _ns(inc=lambda *a: None)
        )

        fake_request = _ns(
            request_id="paused-req",
            metrics=_ns(scheduler_recv_req_time=0),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.Request.from_dict", lambda d: fake_request)
        sent_errors = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent_errors.append(rid))

        call_count = [0]

        def fake_receive(block):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return None, {"request_id": "paused-req", "user": "u"}
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_json_once=fake_receive)
        eng._insert_zmq_task_to_scheduler()
        assert len(sent_errors) >= 1

    # -- _zmq_send_generated_tokens --

    def test_zmq_send_generated_tokens_non_adapter(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", False)

        req_output = _ns(
            request_id="zmq-out-1",
            outputs=_ns(token_ids=[42], decode_type=1, text="hello"),
            finished=False,
        )
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append((rid, r)))

        call_count = [0]

        def fake_get_results():
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return {"zmq-out-1": [req_output]}
            eng.running = False
            return {}

        eng.scheduler.get_results = fake_get_results
        eng._zmq_send_generated_tokens()
        assert len(sent) >= 1
        assert sent[0][0] == "zmq-out-1"

    def test_zmq_send_generated_tokens_adapter(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", False)

        req_output = _ns(
            request_id="zmq-a-1",
            outputs=_ns(token_ids=[42], decode_type=1, text="world"),
            finished=False,
        )
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append((rid, r)))

        call_count = [0]

        def fake_get_results():
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return [[req_output]]
            eng.running = False
            return []

        eng.scheduler.get_results = fake_get_results
        eng._zmq_send_generated_tokens()
        assert len(sent) >= 1
        assert sent[0][0] is None  # adapter sends None as rid

    def test_zmq_send_generated_tokens_with_decode(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        eng.data_processor = _ns(
            ids2tokens=lambda tids, rid: ("decoded", [99], None),
            decode_status={"zmq-dec": [0, 1]},
        )
        req_output = _ns(
            request_id="zmq-dec",
            outputs=_ns(token_ids=[99], decode_type=0, text=""),
            finished=False,
        )
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append((rid, r)))

        call_count = [0]

        def fake_get_results():
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return {"zmq-dec": [req_output]}
            eng.running = False
            return {}

        eng.scheduler.get_results = fake_get_results
        eng._zmq_send_generated_tokens()
        assert len(sent) >= 1

    def test_zmq_send_generated_tokens_finished_empty_tokens(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        eng.data_processor = _ns(
            ids2tokens=lambda tids, rid: ("", [], None),
            decode_status={},
        )
        req_output = _ns(
            request_id="zmq-fin",
            outputs=_ns(token_ids=[1], decode_type=0, text=""),
            finished=True,
        )
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append((rid, r)))

        call_count = [0]

        def fake_get_results():
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return {"zmq-fin": [req_output]}
            eng.running = False
            return {}

        eng.scheduler.get_results = fake_get_results
        eng._zmq_send_generated_tokens()
        # Finished but empty tokens → still appended
        assert len(sent) >= 1

    # -- start with v1 scheduler --

    def test_start_v1_scheduler(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        eng.token_processor.tasks_queue = None
        eng.token_processor.run = lambda: None
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.router_config.router = None
        eng._schedule_request_to_worker_v1 = lambda: None
        eng.start()
        assert eng.running is True
        assert hasattr(eng, "insert_task_to_worker_thread")

    def test_start_decode_role(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        eng.token_processor.tasks_queue = None
        eng.token_processor.run = lambda: None
        eng.cfg.scheduler_config.splitwise_role = "decode"
        eng.cfg.router_config.router = None
        eng._schedule_request_to_worker = lambda: None
        decode_called = []
        eng._decode_process_splitwise_requests = lambda: decode_called.append(1)
        eng.start()
        assert len(decode_called) == 1

    # -- start_worker_queue_service with start_queue=True --

    def test_start_worker_queue_service_with_queue(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.host_ip = "127.0.0.1"
        eng.cfg.master_ip = "127.0.0.1"
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", False)

        created = []

        class FakeEWQ:
            def __init__(self, **kw):
                self.kw = kw
                created.append(self)

            def get_server_port(self):
                return 55555

            def cleanup(self):
                pass

        monkeypatch.setattr("fastdeploy.engine.common_engine.EngineWorkerQueue", FakeEWQ)
        eng.start_worker_queue_service(start_queue=True)
        # Should create both server and client EWQ instances
        assert len(created) == 2
        assert hasattr(eng, "engine_worker_queue_server")

    def test_start_worker_queue_service_prefix_caching(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.host_ip = "127.0.0.1"
        eng.cfg.master_ip = "127.0.0.1"
        eng.cfg.cache_config.enable_prefix_caching = True
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", False)

        class FakeEWQ:
            def __init__(self, **kw):
                pass

            def get_server_port(self):
                return 55555

            def cleanup(self):
                pass

        class FakeECQ:
            def __init__(self, **kw):
                pass

            def get_server_port(self):
                return 55556

        monkeypatch.setattr("fastdeploy.engine.common_engine.EngineWorkerQueue", FakeEWQ)
        monkeypatch.setattr("fastdeploy.engine.common_engine.EngineCacheQueue", FakeECQ)
        eng.start_worker_queue_service(start_queue=False)
        assert hasattr(eng, "cache_task_queue")

    def test_start_worker_queue_service_shm_mode(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.host_ip = "127.0.0.1"
        eng.cfg.master_ip = "127.0.0.1"
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", True)

        class FakeEWQ:
            def __init__(self, **kw):
                self.kw = kw

            def get_server_port(self):
                return 55555

            def cleanup(self):
                pass

        monkeypatch.setattr("fastdeploy.engine.common_engine.EngineWorkerQueue", FakeEWQ)
        eng.start_worker_queue_service(start_queue=True)
        # SHM mode uses /dev/shm path
        assert hasattr(eng, "engine_worker_queue")

    # -- _exit_sub_services async_llm with worker_proc --

    def test_exit_sub_services_async_worker_proc(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        eng.worker_ready_signal = _FakeSignal()
        eng.loaded_model_signal = _FakeSignal()
        killed = []
        eng.worker_proc = _ns(pid=12345)
        monkeypatch.setattr("os.getpgid", lambda pid: pid)
        monkeypatch.setattr("os.killpg", lambda pgid, sig: killed.append(pgid))
        eng._exit_sub_services()
        assert 12345 in killed

    def test_exit_sub_services_async_cache_manager(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        eng.worker_proc = None
        eng.worker_ready_signal = _FakeSignal()
        eng.loaded_model_signal = _FakeSignal()
        killed = []
        eng.cache_manager_processes = [_ns(pid=111)]
        eng.resource_manager.cache_manager = _ns(
            shm_cache_task_flag_broadcast=_FakeSignal(),
            cache_ready_signal=_FakeSignal(),
        )
        monkeypatch.setattr("os.getpgid", lambda pid: pid)
        monkeypatch.setattr("os.killpg", lambda pgid, sig: killed.append(pgid))
        eng._exit_sub_services()
        assert 111 in killed

    def test_exit_sub_services_async_cache_task_queue(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        eng.worker_proc = None
        eng.worker_ready_signal = _FakeSignal()
        eng.loaded_model_signal = _FakeSignal()
        cleaned = []
        eng.cache_task_queue = _ns(cleanup=lambda: cleaned.append(1))
        eng._exit_sub_services()
        assert len(cleaned) == 1

    def test_exit_sub_services_async_dp_processed(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        eng.worker_proc = None
        eng.worker_ready_signal = _FakeSignal()
        eng.loaded_model_signal = _FakeSignal()
        eng.get_profile_block_num_signal = _FakeSignal()
        joined = []
        cleaned = []
        eng.dp_processed = [_ns(pid=222, join=lambda: joined.append(1))]
        eng.dp_engine_worker_queue_server = [_ns(cleanup=lambda: cleaned.append(1))]
        eng._exit_sub_services()
        assert len(joined) == 1
        assert len(cleaned) == 1

    # -- _register_to_router enabled --

    def test_register_to_router_enabled(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.router_config.router = "http://router:8080"
        threads_started = []
        original_thread_init = threading.Thread.__init__

        def track_thread(self_thread, *a, **kw):
            original_thread_init(self_thread, *a, **kw)
            threads_started.append(kw.get("target"))

        monkeypatch.setattr(threading.Thread, "__init__", track_thread)
        eng._register_to_router()
        assert len(threads_started) >= 1

    # -- _call_worker --

    def test_call_worker_timeout(self, monkeypatch):
        import asyncio

        eng = _make_engine(monkeypatch)

        class FakeQueue:
            def __init__(self, name):
                self.name = name

            async def get(self, timeout=None):
                await asyncio.sleep(10)

        eng._ctrl_worker_output_queues = [FakeQueue("q0")]
        ctrl = _ns(request_id="cw-1")
        eng.engine_worker_queue = _ns(put_tasks=lambda *a: None)
        with pytest.raises(Exception, match="Timeouted"):
            eng._call_worker(ctrl, timeout=0.01)

    # -- _wait_all_control_responses success --

    def test_call_worker_success(self, monkeypatch):

        eng = _make_engine(monkeypatch)

        class FakeQueue:
            def __init__(self, name):
                self.name = name

            async def get(self, timeout=None):
                return _ns(
                    payload=_ns(request_id="cw-ok", error_code=200, error_message=None, result={"status": "ok"})
                )

        eng._ctrl_worker_output_queues = [FakeQueue("q0")]
        eng.engine_worker_queue = _ns(put_tasks=lambda *a: None)
        ctrl = _ns(request_id="cw-ok")
        results = eng._call_worker(ctrl, timeout=5)
        assert results == [{"status": "ok"}]

    # -- start_worker_service flow --

    def test_start_worker_service_flow(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        eng.do_profile = 0
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.IPCSignal",
            lambda **kw: _FakeSignal(kw.get("array")),
        )
        eng.ipc_signal_suffix = 12345
        eng.cfg.cache_config.enable_prefix_caching = False
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.data_processor = _ns(
            tokenizer=_ns(
                vocab={"a": 0},
                get_vocab=lambda: {},
                encode=lambda *a, **kw: [10],
            ),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        eng.mm_max_tokens_per_item = None

        # Simulate worker becoming ready
        def fake_popen(cmd, **kw):
            return _ns(pid=9999, poll=lambda: None, stdout=iter([]))

        monkeypatch.setattr("fastdeploy.engine.common_engine.subprocess.Popen", fake_popen)

        # Make loaded_model_signal appear ready immediately
        ready_signal = _FakeSignal(np.array([1], dtype=np.int32))

        def fake_init_signals():
            eng.worker_ready_signal = _FakeSignal(np.array([1], dtype=np.int32))
            eng.loaded_model_signal = ready_signal

        eng._init_worker_signals = fake_init_signals

        # Bypass check_worker_initialize_status
        def fake_check():
            eng.worker_init_status["finished"] = True
            return True

        eng.check_worker_initialize_status = fake_check
        monkeypatch.setattr("fastdeploy.engine.common_engine.time.sleep", lambda s: None)

        eng.start_worker_service(async_llm_pid=None)
        assert hasattr(eng, "worker_proc")

    # -- launch_components with expert parallel and DP --

    def test_launch_components_expert_parallel_dp(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.scheduler_config.name = "local"
        eng.cfg.parallel_config.enable_expert_parallel = True
        eng.cfg.parallel_config.data_parallel_size = 2
        eng.cfg.parallel_config.engine_worker_queue_port = [12345, 12346]
        eng.cfg.nnode = 1
        eng.launched_expert_service_signal = _FakeSignal(np.array([0, 1], dtype=np.int32))
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.envs.FD_ENABLE_MULTI_API_SERVER",
            False,
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM",
            False,
        )

        class FakeEWQ:
            def __init__(self, **kw):
                pass

            def cleanup(self):
                pass

        monkeypatch.setattr("fastdeploy.engine.common_engine.EngineWorkerQueue", FakeEWQ)

        started_procs = []

        class FakeProcess:
            def __init__(self, target=None, args=None):
                self.pid = 999
                self._target = target

            def start(self):
                started_procs.append(1)
                # Simulate instant ready
                eng.launched_expert_service_signal.value[1] = 1

        monkeypatch.setattr("fastdeploy.engine.common_engine.multiprocessing.Process", FakeProcess)
        monkeypatch.setattr("fastdeploy.engine.expert_service.start_data_parallel_service", lambda *a: None)
        eng.launch_components()
        assert len(started_procs) == 1
        assert hasattr(eng, "dp_processed")

    # -- __init__ --

    def test_init_v0_scheduler(self, monkeypatch):
        from fastdeploy.engine.common_engine import EngineService

        cfg = _make_cfg()
        cfg.scheduler_config.scheduler = lambda: _ns(
            put_requests=lambda *a: [],
            get_requests=lambda **kw: [],
            put_results=lambda *a: None,
            get_results=lambda: [],
            start=lambda *a, **kw: None,
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_CACHE_TASK", "0")
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.ResourceManager",
            lambda *a, **kw: _ns(scheduler_metrics_logger=None, cache_manager=_ns()),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.FMQ", lambda: _ns(queue=lambda name, role: _ns()))
        monkeypatch.setattr(
            EngineService,
            "start_worker_queue_service",
            lambda self, sq: setattr(self, "engine_worker_queue", _ns()),
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.SplitwiseConnector", lambda *a, **kw: _ns(start_receiver=lambda: None)
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.TokenProcessor",
            lambda *a, **kw: _ns(
                set_resource_manager=lambda rm: None,
                set_scheduler_metrics_logger=lambda sml: None,
            ),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.SchedulerMetricsLogger", lambda *a, **kw: _ns())
        monkeypatch.setattr("fastdeploy.engine.common_engine.IPCSignal", lambda **kw: _FakeSignal(kw.get("array")))
        eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        assert eng.is_paused is False
        assert eng.mm_max_tokens_per_item is None
        assert eng.guided_decoding_checker is None
        assert eng.bos_client is None

    def test_init_v1_scheduler_async(self, monkeypatch):
        from fastdeploy.engine.common_engine import EngineService

        cfg = _make_cfg()
        cfg.scheduler_config.scheduler = lambda: _ns(
            put_requests=lambda *a: [],
            get_requests=lambda **kw: [],
            put_results=lambda *a: None,
            get_results=lambda: [],
            start=lambda *a, **kw: None,
        )
        cfg.cache_config.num_gpu_blocks_override = None
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_CACHE_TASK", "0")
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.ResourceManagerV1",
            lambda *a, **kw: _ns(scheduler_metrics_logger=None, cache_manager=_ns()),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.FMQ", lambda: _ns(queue=lambda name, role: _ns()))
        monkeypatch.setattr(
            EngineService,
            "start_worker_queue_service",
            lambda self, sq: setattr(self, "engine_worker_queue", _ns()),
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.SplitwiseConnector", lambda *a, **kw: _ns(start_receiver=lambda: None)
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.TokenProcessor",
            lambda *a, **kw: _ns(
                set_resource_manager=lambda rm: None,
                set_scheduler_metrics_logger=lambda sml: None,
            ),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.SchedulerMetricsLogger", lambda *a, **kw: _ns())
        monkeypatch.setattr("fastdeploy.engine.common_engine.IPCSignal", lambda **kw: _FakeSignal(kw.get("array")))
        eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        assert eng.use_async_llm is True
        assert eng.do_profile == 1
        assert eng.worker_proc is None

    def test_init_dp_gt_1(self, monkeypatch):
        from fastdeploy.engine.common_engine import EngineService

        cfg = _make_cfg()
        cfg.parallel_config.data_parallel_size = 2
        cfg.scheduler_config.scheduler = lambda: _ns()
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_CACHE_TASK", "0")
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.ResourceManager",
            lambda *a, **kw: _ns(scheduler_metrics_logger=None),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.FMQ", lambda: _ns(queue=lambda name, role: _ns()))
        monkeypatch.setattr(
            EngineService,
            "start_worker_queue_service",
            lambda self, sq: setattr(self, "engine_worker_queue", _ns()),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.SplitwiseConnector", lambda *a, **kw: _ns())
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.TokenProcessor",
            lambda *a, **kw: _ns(
                set_resource_manager=lambda rm: None,
                set_scheduler_metrics_logger=lambda sml: None,
            ),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.SchedulerMetricsLogger", lambda *a, **kw: _ns())
        monkeypatch.setattr("fastdeploy.engine.common_engine.IPCSignal", lambda **kw: _FakeSignal(kw.get("array")))
        monkeypatch.setattr("fastdeploy.engine.common_engine.get_logger", lambda *a, **kw: _ns(info=lambda *a: None))
        eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        assert eng.cfg.parallel_config.data_parallel_size == 2

    def test_init_guided_decoding(self, monkeypatch):
        from fastdeploy.engine.common_engine import EngineService

        cfg = _make_cfg()
        cfg.structured_outputs_config.guided_decoding_backend = "xgrammar"
        cfg.scheduler_config.scheduler = lambda: _ns()
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_CACHE_TASK", "0")
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.ResourceManager",
            lambda *a, **kw: _ns(scheduler_metrics_logger=None),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.FMQ", lambda: _ns(queue=lambda name, role: _ns()))
        monkeypatch.setattr(
            EngineService,
            "start_worker_queue_service",
            lambda self, sq: setattr(self, "engine_worker_queue", _ns()),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.SplitwiseConnector", lambda *a, **kw: _ns())
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.TokenProcessor",
            lambda *a, **kw: _ns(
                set_resource_manager=lambda rm: None,
                set_scheduler_metrics_logger=lambda sml: None,
            ),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.SchedulerMetricsLogger", lambda *a, **kw: _ns())
        monkeypatch.setattr("fastdeploy.engine.common_engine.IPCSignal", lambda **kw: _FakeSignal(kw.get("array")))
        checker = _ns()
        monkeypatch.setattr("fastdeploy.engine.common_engine.schema_checker", lambda *a, **kw: checker)
        eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        assert eng.guided_decoding_checker is checker

    # -- _schedule_request_to_worker v0 splitwise path --

    def test_schedule_v0_splitwise_decode_skips(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "decode"
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        eng.resource_manager.available_batch = lambda: 1
        eng.resource_manager.available_block_num = lambda: 100
        eng.resource_manager.check_and_free_block_tables = lambda: None
        eng.resource_manager.abort_req_ids_set = set()
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, num_cache_infos=lambda: 0)
        eng.split_connector = _ns(
            current_request_ids=[],
            has_splitwise_tasks=lambda: False,
            send_splitwise_tasks=lambda *a: None,
        )
        task = _ns(
            request_id="v0-dec",
            metrics=_ns(
                engine_get_req_time=0,
                ask_decode_resource_start_time=0,
            ),
            user="test",
        )
        call_count = [0]

        def fake_get_requests(**kw):
            call_count[0] += 1
            if call_count[0] >= 2:
                eng.running = False
            return [task]

        eng.scheduler.get_requests = fake_get_requests
        eng._schedule_request_to_worker()
        assert call_count[0] >= 2  # decode skips insert, loops again

    def test_schedule_v0_exist_prefill_signal(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        eng.resource_manager.available_batch = lambda: 1
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, num_cache_infos=lambda: 0)
        eng.split_connector = _ns(
            current_request_ids=[],
            has_splitwise_tasks=lambda: True,
        )
        eng.exist_prefill_task_signal = _FakeSignal(np.array([1], dtype=np.int32))
        call_count = [0]

        def fake_avail():
            call_count[0] += 1
            if call_count[0] >= 3:
                eng.running = False
            return 1

        eng.resource_manager.available_batch = fake_avail
        eng._schedule_request_to_worker()
        assert call_count[0] >= 3

    def test_schedule_v0_num_cache_infos(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        eng.resource_manager.available_batch = lambda: 1
        call_count = [0]

        def num_cache():
            call_count[0] += 1
            if call_count[0] >= 2:
                eng.running = False
            return 1

        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, num_cache_infos=num_cache)
        eng.split_connector = _ns(current_request_ids=[])
        eng._schedule_request_to_worker()
        assert call_count[0] >= 2

    def test_schedule_v0_split_connector_ids(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        eng.resource_manager.available_batch = lambda: 1
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, num_cache_infos=lambda: 0)
        call_count = [0]

        class FakeConn:
            @property
            def current_request_ids(self):
                nonlocal call_count
                call_count[0] += 1
                if call_count[0] >= 2:
                    eng.running = False
                return ["some_id"]

        eng.split_connector = FakeConn()
        eng._schedule_request_to_worker()
        assert call_count[0] >= 2

    # -- _schedule_request_to_worker_v1 fetch_request (decode role) --

    def test_schedule_v1_decode_fetch(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "decode"
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        eng.resource_manager.available_batch = lambda: 1
        eng.resource_manager.available_block_num = lambda: 100
        eng.resource_manager.abort_req_ids_set = set()
        eng.scheduler.get_requests = lambda **kw: []
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=lambda *a: None)

        call_count = [0]

        def fake_schedule():
            call_count[0] += 1
            if call_count[0] >= 2:
                eng.running = False
            return [], []

        eng.resource_manager.schedule = fake_schedule
        eng._schedule_request_to_worker_v1()
        assert call_count[0] >= 1

    # -- _zmq_send_generated_tokens adapter with decode_type=0 --

    def test_zmq_send_tokens_adapter_decode(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        eng.data_processor = _ns(
            ids2tokens=lambda tids, rid: ("tok", [42], None),
            decode_status={"a-dec": [0, 1]},
        )
        req_output = _ns(
            request_id="a-dec",
            outputs=_ns(token_ids=[42], decode_type=0, text=""),
            finished=False,
        )
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append((rid, r)))
        call_count = [0]

        def fake_get_results():
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return [[req_output]]
            eng.running = False
            return []

        eng.scheduler.get_results = fake_get_results
        eng._zmq_send_generated_tokens()
        assert len(sent) >= 1

    def test_zmq_send_tokens_non_request_output(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        non_output = _ns(outputs=None, finished=False)
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append((rid, r)))
        call_count = [0]

        def fake_get_results():
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return {"req-no": [non_output]}
            eng.running = False
            return {}

        eng.scheduler.get_results = fake_get_results
        eng._zmq_send_generated_tokens()
        assert len(sent) >= 1

    # -- _insert_zmq_task with guided_decoding and v1_abort --

    def test_insert_zmq_task_guided_decoding_error(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.requests_number", _ns(inc=lambda *a: None)
        )

        fake_request = _ns(
            request_id="gd-req",
            metrics=_ns(scheduler_recv_req_time=0),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.Request.from_dict", lambda d: fake_request)
        eng.guided_decoding_checker = _ns(schema_format=lambda req: (req, "bad schema"))
        sent_errors = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent_errors.append(rid))
        eng.scheduler.put_requests = lambda t: []

        call_count = [0]

        def fake_receive(block):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return None, {"request_id": "gd-req", "user": "u"}
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_json_once=fake_receive)
        eng._insert_zmq_task_to_scheduler()
        assert len(sent_errors) >= 1

    def test_insert_zmq_task_v1_abort_in_resource(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)

        from fastdeploy.engine.common_engine import RequestStatus

        abort_data = {"request_id": "v1-ab", "status": RequestStatus.ABORT.value}
        eng.resource_manager.abort_req_ids_set = set()
        eng.resource_manager.requests = {"v1-ab": _ns(request_id="v1-ab")}
        preempt_task = _ns(request_id="v1-ab")
        eng.resource_manager._prepare_preempt_task = lambda req: preempt_task
        eng.resource_manager.real_bsz = 1
        put_calls = []
        eng.engine_worker_queue = _ns(put_tasks=lambda t: put_calls.append(t))
        call_count = [0]

        def fake_receive(block):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return None, abort_data
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_json_once=fake_receive)
        eng._insert_zmq_task_to_scheduler()
        assert "v1-ab" in eng.resource_manager.abort_req_ids_set
        assert len(put_calls) >= 1

    def test_insert_zmq_task_v1_abort_not_in_resource(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)

        from fastdeploy.engine.common_engine import RequestStatus

        abort_data = {"request_id": "v1-ab2", "status": RequestStatus.ABORT.value}
        eng.resource_manager.abort_req_ids_set = set()
        eng.resource_manager.requests = {}
        recycled = []
        eng.scheduler._recycle = lambda rid: recycled.append(rid)
        call_count = [0]

        def fake_receive(block):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return None, abort_data
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_json_once=fake_receive)
        eng._insert_zmq_task_to_scheduler()
        assert "v1-ab2" in recycled

    def test_insert_zmq_task_request_error(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.requests_number", _ns(inc=lambda *a: None)
        )

        def bad_from_dict(d):
            raise ValueError("bad request format")

        monkeypatch.setattr("fastdeploy.engine.common_engine.Request.from_dict", bad_from_dict)
        eng.guided_decoding_checker = None
        sent_errors = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent_errors.append(rid))
        eng.scheduler.put_requests = lambda t: []
        call_count = [0]

        class _AttrDict(dict):
            """Dict subclass with attribute access — mirrors real data after from_dict fails."""

            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(name)

        def fake_receive(block):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return None, _AttrDict(request_id="bad-req", user="u")
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_json_once=fake_receive)
        eng._insert_zmq_task_to_scheduler()
        assert len(sent_errors) >= 1

    def test_insert_zmq_task_with_trace_carrier(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *a: None
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.requests_number", _ns(inc=lambda *a: None)
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.num_requests_waiting", _ns(inc=lambda *a: None)
        )

        fake_request = _ns(request_id="tc-req", metrics=_ns(scheduler_recv_req_time=0))
        monkeypatch.setattr("fastdeploy.engine.common_engine.Request.from_dict", lambda d: fake_request)
        eng.guided_decoding_checker = None
        eng.scheduler.put_requests = lambda t: []
        call_count = [0]

        def fake_receive(block):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return None, {"request_id": "tc-req_0", "user": "u", "trace_carrier": {"x": "y"}}
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_json_once=fake_receive)
        eng._insert_zmq_task_to_scheduler()

    def test_insert_zmq_task_pyobj_mode(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = True
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.requests_number", _ns(inc=lambda *a: None)
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.num_requests_waiting", _ns(inc=lambda *a: None)
        )

        fake_request = _ns(request_id="py-req", metrics=_ns(scheduler_recv_req_time=0))
        monkeypatch.setattr("fastdeploy.engine.common_engine.Request.from_dict", lambda d: fake_request)
        eng.guided_decoding_checker = None
        eng.scheduler.put_requests = lambda t: []
        call_count = [0]

        def fake_receive_pyobj(block):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return None, {"request_id": "py-req", "user": "u"}
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_pyobj_once=fake_receive_pyobj)
        eng._insert_zmq_task_to_scheduler()

    # -- update_requests_chunk_size more paths --

    def test_update_requests_chunk_size_large_overflow(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.cache_config.enable_chunked_prefill = True
        eng.cfg.cache_config.block_size = 16
        eng.cfg.scheduler_config.max_num_batched_tokens = 64
        eng.cfg.max_num_partial_prefills = 2
        eng.partial_chunked_tokens = [0, 64, 32]
        chunks = [{}, {}]
        reqs = [
            _ns(
                prompt_token_ids_len=50,
                set=lambda key, val, i=i: chunks[i].update({key: val}),
            )
            for i in range(2)
        ]
        eng.update_requests_chunk_size(reqs)
        assert "prefill_chunk_info" in chunks[0]

    # -- _start_worker_service tokenizer edge cases --

    def test_start_worker_service_line_break_nested(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.data_processor = _ns(
            tokenizer=_ns(
                vocab={"a": 0},
                get_vocab=lambda: {},
                encode=lambda *a, **kw: {"input_ids": [[10]]},  # dict with nested list
            ),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        eng.mm_max_tokens_per_item = None
        eng.do_profile = 0
        eng.ipc_signal_suffix = 12345
        captured = {}

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd
            return _ns(pid=9999, poll=lambda: None, stdout=iter([]))

        monkeypatch.setattr("fastdeploy.engine.common_engine.subprocess.Popen", fake_popen)
        eng._start_worker_service()
        assert "--line_break_id 10" in captured["cmd"]

    # -- _stop_profile with prefix caching --

    def test_stop_profile_prefix_caching(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.do_profile = 1
        eng.get_profile_block_num_signal = _FakeSignal(np.array([100], dtype=np.int32))
        eng.worker_proc = None
        eng.cfg.cache_config.reset = lambda n: None
        eng.resource_manager.reset_cache_config = lambda cc: None
        eng.cfg.cache_config.enable_prefix_caching = True
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.ipc_signal_suffix = 12345
        cache_started = []
        eng.start_cache_service = lambda d, s: cache_started.append(1) or []
        eng._stop_profile()
        assert eng.do_profile == 0
        assert len(cache_started) == 1

    def test_stop_profile_splitwise_role(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.do_profile = 1
        eng.get_profile_block_num_signal = _FakeSignal(np.array([100], dtype=np.int32))
        eng.worker_proc = None
        eng.cfg.cache_config.reset = lambda n: None
        eng.resource_manager.reset_cache_config = lambda cc: None
        eng.cfg.cache_config.enable_prefix_caching = False
        eng.cfg.scheduler_config.splitwise_role = "prefill"
        eng.ipc_signal_suffix = 12345
        cache_started = []
        eng.start_cache_service = lambda d, s: cache_started.append(1) or []
        eng._stop_profile()
        assert len(cache_started) == 1

    # -- _exit_sub_services more paths --

    def test_exit_sub_services_cache_task_queue_manager(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        eng.worker_proc = None
        eng.worker_ready_signal = _FakeSignal()
        eng.loaded_model_signal = _FakeSignal()
        shut = []
        eng.cache_task_queue = _ns(manager=_ns(shutdown=lambda: shut.append(1)))
        eng._exit_sub_services()
        assert len(shut) == 1

    def test_exit_sub_services_recv_control_cmd(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        closed = []
        eng.send_response_server = _ns(close=lambda: closed.append("send"))
        eng.recv_request_server = _ns(close=lambda: closed.append("recv"))
        eng.recv_control_cmd_server = _ns(close=lambda: closed.append("ctrl"))
        eng._exit_sub_services()
        assert "ctrl" in closed

    # -- insert_tasks with trace_carrier --

    def test_insert_tasks_with_trace_carrier(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.cache_config.enable_chunked_prefill = False
        eng.cfg.model_config.enable_mm = False
        eng.resource_manager.check_and_free_block_tables = lambda: None
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        eng.engine_worker_queue = _ns(put_tasks=lambda *a: None)
        traced = []
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context",
            lambda *a: traced.append(a),
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context",
            lambda *a: {"trace": "ctx"},
        )
        task = _ns(
            request_id="tc_0",
            trace_carrier={"span": "123"},
            prompt_token_ids_len=32,
            metrics=_ns(
                inference_start_time=0,
                scheduler_recv_req_time=time.time(),
                add_req_to_resource_manager_time=0,
            ),
            disaggregate_info=None,
            has_been_preempted_before=False,
            set=lambda k, v: None,
            user="test",
        )
        result = eng.insert_tasks([task])
        assert result is True
        assert len(traced) >= 1

    # -- _zmq_send_generated_tokens --

    def test_zmq_send_generated_tokens_internal_adapter(self, monkeypatch):
        """Internal-adapter branch: results is list-of-lists."""
        from fastdeploy.engine.request import RequestOutput

        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", False)

        out1 = RequestOutput(
            request_id="r1", finished=False, outputs=_ns(tool_calls=None, decode_type=1, token_ids=[10, 20])
        )
        out2 = RequestOutput(
            request_id="r2", finished=True, outputs=_ns(tool_calls=None, decode_type=0, token_ids=[30])
        )
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, c: sent.append((rid, c)))

        call_count = [0]

        def fake_get_results():
            call_count[0] += 1
            if call_count[0] == 1:
                return [[out1, out2]]
            eng.running = False
            return []

        eng.scheduler.get_results = fake_get_results
        eng._zmq_send_generated_tokens()
        assert len(sent) == 1
        assert sent[0][0] is None  # internal adapter sends rid=None
        from fastdeploy.engine.request import RequestOutput

        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)

        eng.data_processor = _ns(
            ids2tokens=lambda tids, rid: ("hello", [1, 2, 3, 4], None),
            decode_status={"r1": (1, 3)},
        )

        out = RequestOutput(
            request_id="r1", finished=True, outputs=_ns(tool_calls=None, decode_type=0, token_ids=[42], text="")
        )
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, c: sent.append((rid, c)))

        call_count = [0]

        def fake_get_results():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"r1": [out]}
            eng.running = False
            return {}

        eng.scheduler.get_results = fake_get_results
        eng._zmq_send_generated_tokens()
        assert len(sent) == 1
        assert out.outputs.text == "hello"
        assert "r1" not in eng.data_processor.decode_status  # is_end=True deletes

    def test_decode_token_return_text_empty_delta(self, monkeypatch):
        """Empty delta_text → token_ids becomes []."""
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        eng.data_processor = _ns(
            ids2tokens=lambda tids, rid: ("", [10, 20], None),
            decode_status={"r1": (0, 2)},
        )
        text, tids = eng._decode_token([42], "r1", is_end=False)
        assert text == ""
        assert tids == []

    def test_register_to_router_skips_when_no_router(self, monkeypatch):
        """Router=None → skip registering."""
        eng = _make_engine(monkeypatch)
        eng.cfg.router_config.router = None
        # Should not raise or start any thread
        eng._register_to_router()

    def test_register_to_router_starts_thread(self, monkeypatch):
        """Router configured → starts registration thread (daemon)."""
        eng = _make_engine(monkeypatch)
        eng.cfg.router_config.router = "http://fake-router:9090"
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.check_service_health",
            lambda url: False,  # never healthy → loop sleeps and thread is daemon
        )
        eng._register_to_router()

    # -- _exit_sub_services cache manager cleanup --

    def test_exit_sub_services_cache_manager_cleanup(self, monkeypatch):
        """Cache manager processes cleanup path."""
        import os

        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        killed_pids = []
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: killed_pids.append(pgid))
        eng.worker_proc = _ns(pid=999)
        eng.cache_manager_processes = [_ns(pid=1001), _ns(pid=1002)]
        eng.resource_manager.cache_manager = _ns(
            shm_cache_task_flag_broadcast=_FakeSignal(),
            cache_ready_signal=_FakeSignal(),
        )
        eng.engine_worker_queue_server = None
        eng._exit_sub_services()
        assert 999 in killed_pids
        assert 1001 in killed_pids
        assert 1002 in killed_pids

    # -- _insert_zmq_task_to_scheduler: internal adapter decode early return --

    def test_insert_zmq_internal_adapter_decode_returns(self, monkeypatch):
        """Internal adapter + decode role → returns immediately."""
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        eng.cfg.scheduler_config.splitwise_role = "decode"
        eng._insert_zmq_task_to_scheduler()  # should return immediately

    # -- _zmq_send_generated_tokens: decode_type zero with return text --

    def test_zmq_send_tokens_decode_type_zero_with_text(self, monkeypatch):
        """decode_type==0 with FD_ENABLE_RETURN_TEXT → calls _decode_token."""
        from fastdeploy.engine.request import RequestOutput

        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        eng.data_processor = _ns(
            ids2tokens=lambda tids, rid: ("word", tids, None),
            decode_status={"r1": (0, 2)},
        )

        out = RequestOutput(
            request_id="r1",
            finished=False,
            outputs=_ns(tool_calls=None, decode_type=0, token_ids=[10, 20]),
        )
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, contents: sent.append((rid, contents)))

        call_count = [0]

        def fake_get_results():
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return {"r1": [out]}
            eng.running = False
            return {}

        eng.scheduler.get_results = fake_get_results
        eng._zmq_send_generated_tokens()
        assert len(sent) >= 1

    # -- _insert_zmq_task_to_scheduler: pyobj path (enable_mm=True) --

    def test_insert_zmq_task_pyobj_path(self, monkeypatch):
        """enable_mm=True → uses receive_pyobj_once."""
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = True
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.requests_number", _ns(inc=lambda *a: None)
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.num_requests_waiting", _ns(inc=lambda *a: None)
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", True)

        fake_request = _ns(
            request_id="mm-req",
            metrics=_ns(scheduler_recv_req_time=0),
        )
        eng.guided_decoding_checker = None
        eng.scheduler.put_requests = lambda t: []
        call_count = [0]

        def fake_receive(block):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return None, fake_request
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_pyobj_once=fake_receive)
        eng._insert_zmq_task_to_scheduler()

    # -- _insert_zmq_task_to_scheduler: error reconnect (non-term error) --

    def test_insert_zmq_error_reconnect_ipc(self, monkeypatch):
        """Non-termination error → recreates IPC server and continues."""
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        eng.api_server_pid = 12345
        created_servers = []

        def _new_receive(block):
            eng.running = False
            return "Context was terminated", None

        def fake_ipc(name, mode):
            s = _ns(receive_json_once=_new_receive)
            created_servers.append(name)
            return s

        monkeypatch.setattr("fastdeploy.engine.common_engine.ZmqIpcServer", fake_ipc)

        eng.recv_request_server = _ns(receive_json_once=lambda b: ("Connection reset", None))
        eng._insert_zmq_task_to_scheduler()
        assert len(created_servers) >= 1

    # -- _insert_zmq_task_to_scheduler: control request path --

    def test_insert_zmq_control_request(self, monkeypatch):
        """Control requests are dispatched to run_control_method."""
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.ControlRequest.is_control_request",
            staticmethod(lambda d: d.get("is_control", False)),
        )
        ctrl_calls = []
        monkeypatch.setattr("fastdeploy.engine.common_engine.ControlRequest.from_dict", lambda d: _ns(**d))
        eng.run_control_method = lambda cr: ctrl_calls.append(cr.request_id)

        call_count = [0]

        def fake_receive(block):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return None, {"is_control": True, "request_id": "ctrl-1"}
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_json_once=fake_receive)
        eng._insert_zmq_task_to_scheduler()
        assert "ctrl-1" in ctrl_calls

    # -- _insert_zmq_task_to_scheduler: paused engine drops request --

    def test_insert_zmq_paused_drops_request(self, monkeypatch):
        """Paused engine sends error response for received request."""
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = False
        eng.is_paused = True
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.requests_number", _ns(inc=lambda *a: None)
        )

        fake_req = _ns(request_id="paused-req", metrics=_ns(scheduler_recv_req_time=0))
        monkeypatch.setattr("fastdeploy.engine.common_engine.Request.from_dict", lambda d: fake_req)

        sent_errors = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent_errors.append(rid))

        call_count = [0]

        def fake_receive(block):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return None, {"request_id": "paused-req", "user": "u"}
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_json_once=fake_receive)
        eng._insert_zmq_task_to_scheduler()
        assert "paused-req" in sent_errors

    # -- _insert_zmq_task_to_scheduler: guided decoding error --

    def test_insert_zmq_guided_decoding_error(self, monkeypatch):
        """Guided decoding checker returns error → sends error response."""
        eng = _make_engine(monkeypatch)
        eng.cfg.model_config.enable_mm = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.requests_number", _ns(inc=lambda *a: None)
        )

        fake_req = _ns(request_id="gd-req", metrics=_ns(scheduler_recv_req_time=0))
        monkeypatch.setattr("fastdeploy.engine.common_engine.Request.from_dict", lambda d: fake_req)

        eng.guided_decoding_checker = _ns(schema_format=lambda r: (r, "invalid schema"))
        sent_errors = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent_errors.append(rid))
        eng.scheduler.put_requests = lambda t: []

        call_count = [0]

        def fake_receive(block):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return None, {"request_id": "gd-req", "user": "u"}
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_json_once=fake_receive)
        eng._insert_zmq_task_to_scheduler()
        assert "gd-req" in sent_errors

    # -- _exit_sub_services: cache_task_queue cleanup --

    def test_exit_sub_services_cache_task_queue_cleanup(self, monkeypatch):
        """cache_task_queue with cleanup() method gets called."""
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        eng.worker_proc = None
        eng.cache_task_queue = _ns(cleanup=_Recorder())
        eng.engine_worker_queue_server = None
        eng._exit_sub_services()
        assert len(eng.cache_task_queue.cleanup.calls) == 1

    def test_exit_sub_services_get_profile_signal(self, monkeypatch):
        """get_profile_block_num_signal.clear() gets called."""
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        eng.worker_proc = None
        eng.get_profile_block_num_signal = _FakeSignal()
        eng.engine_worker_queue_server = None
        eng._exit_sub_services()
        assert eng.get_profile_block_num_signal.cleared

    def test_schedule_request_to_worker_v0_gets_tasks(self, monkeypatch):
        """v0 scheduler: gets tasks, inserts them, updates metrics."""
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)
        monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.num_requests_waiting", _ns(dec=lambda *a: None)
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.num_requests_running", _ns(inc=lambda *a: None)
        )

        eng.resource_manager.available_batch = lambda: 1
        eng.resource_manager.available_block_num = lambda: 100
        eng.resource_manager.abort_req_ids_set = set()
        eng.split_connector = _ns(
            has_splitwise_tasks=lambda: False,
            current_request_ids=[],
        )
        eng.engine_worker_queue = _ns(
            exist_tasks=lambda: False,
            num_cache_infos=lambda: 0,
            put_tasks=lambda *a: None,
        )
        eng.exist_prefill_task_signal = _FakeSignal(np.array([0], dtype=np.int32))

        task = _ns(
            request_id="v0-1",
            metrics=_ns(
                engine_get_req_time=0,
                inference_start_time=0,
                add_req_to_resource_manager_time=0,
                scheduler_recv_req_time=time.time(),
            ),
            prompt_token_ids_len=16,
            trace_carrier=None,
            disaggregate_info=None,
            has_been_preempted_before=False,
            set=lambda k, v: None,
            user="u",
        )

        call_count = [0]

        def fake_get_requests(**kw):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                return [task]
            eng.running = False
            return []

        eng.scheduler.get_requests = fake_get_requests
        # Mock insert_tasks to avoid complex resource allocation
        eng.insert_tasks = lambda tasks, cid: True
        eng._schedule_request_to_worker()

    # -- _decode_process_splitwise_requests --

    def test_decode_process_splitwise_allocate_non_v1(self, monkeypatch):
        """Allocate path: Request tasks with non-v1 kvcache scheduler."""
        from fastdeploy.engine.request import Request

        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)

        req = Request(request_id="split-alloc-1")
        req.prompt_token_ids_len = 16
        req.metrics = _ns(decode_recv_req_time=0, decode_preallocate_req_time=0)

        call_count = [0]
        inserted = []

        def fake_disagg_empty():
            nonlocal call_count
            call_count[0] += 1
            return call_count[0] > 1  # non-empty on first call only

        eng.engine_worker_queue.disaggregate_queue_empty = fake_disagg_empty
        eng.engine_worker_queue.get_disaggregated_tasks = lambda: [("batch", [req])]
        eng.resource_manager.is_resource_sufficient = lambda n: True
        eng.insert_tasks = lambda tasks, *a, **kw: inserted.extend(tasks)
        eng.split_connector = _ns(send_cache_info_to_prefill=lambda t: None)
        eng.enable_decode_cache_task = False
        eng.scheduler.has_request = lambda rid: True
        eng.cfg.splitwise_version = "v0"
        eng._insert_prefilled_requests = lambda reqs: None

        eng._decode_process_splitwise_requests()
        time.sleep(0.15)
        eng.running = False
        time.sleep(0.05)
        assert len(inserted) >= 1

    def test_decode_process_splitwise_alloc_v1_kvcache(self, monkeypatch):
        """Allocate path: Request tasks with v1 kvcache scheduler."""
        from fastdeploy.engine.request import Request

        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)

        req = Request(request_id="split-v1-1")
        req.prompt_token_ids_len = 16
        req.metrics = _ns(decode_recv_req_time=0, decode_preallocate_req_time=0)

        call_count = [0]
        cache_sent = []

        def fake_disagg_empty():
            nonlocal call_count
            call_count[0] += 1
            return call_count[0] > 1

        eng.engine_worker_queue.disaggregate_queue_empty = fake_disagg_empty
        eng.engine_worker_queue.get_disaggregated_tasks = lambda: [("batch", [req])]
        eng.resource_manager.preallocate_resource_in_d = lambda t: True
        eng.split_connector = _ns(send_cache_info_to_prefill=lambda t: cache_sent.extend(t))
        eng.enable_decode_cache_task = False
        eng.scheduler.has_request = lambda rid: True
        eng.cfg.splitwise_version = "v0"
        eng._insert_prefilled_requests = lambda reqs: None

        eng._decode_process_splitwise_requests()
        time.sleep(0.15)
        eng.running = False
        time.sleep(0.05)
        assert len(cache_sent) >= 1

    def test_decode_process_splitwise_alloc_fail_no_cache(self, monkeypatch):
        """Allocate fail without cache task → sends error via split_connector."""
        from fastdeploy.engine.request import Request

        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)

        req = Request(request_id="split-fail-1")
        req.prompt_token_ids_len = 16
        req.metrics = _ns(decode_recv_req_time=0, decode_preallocate_req_time=0)

        call_count = [0]
        fail_sent = []

        def fake_disagg_empty():
            nonlocal call_count
            call_count[0] += 1
            return call_count[0] > 1

        eng.engine_worker_queue.disaggregate_queue_empty = fake_disagg_empty
        eng.engine_worker_queue.get_disaggregated_tasks = lambda: [("batch", [req])]
        eng.resource_manager.preallocate_resource_in_d = lambda t: False
        eng.split_connector = _ns(send_cache_info_to_prefill=lambda t: fail_sent.extend(t))
        eng.enable_decode_cache_task = False
        eng.scheduler.has_request = lambda rid: True
        eng.cfg.splitwise_version = "v0"
        eng._insert_prefilled_requests = lambda reqs: None

        eng._decode_process_splitwise_requests()
        time.sleep(0.15)
        eng.running = False
        time.sleep(0.05)
        assert len(fail_sent) >= 1
        assert fail_sent[0].error_msg == "Not enough resources"

    def test_decode_process_splitwise_prefilled_non_v1(self, monkeypatch):
        """Prefilled path: RequestOutput tasks with non-v1 kvcache."""
        from fastdeploy.engine.request import RequestOutput

        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)

        out = RequestOutput(request_id="pf-1", outputs=_ns(tool_calls=None, token_ids=[10]))
        out.metrics = _ns(decode_recv_first_token_time=0)

        call_count = [0]
        prefill_inserted = []

        def fake_disagg_empty():
            nonlocal call_count
            call_count[0] += 1
            return call_count[0] > 1

        eng.engine_worker_queue.disaggregate_queue_empty = fake_disagg_empty
        eng.engine_worker_queue.get_disaggregated_tasks = lambda: [("batch", [out])]
        eng.scheduler.has_request = lambda rid: True
        eng.cfg.splitwise_version = "v0"
        eng._insert_prefilled_requests = lambda reqs: prefill_inserted.extend(reqs)
        eng.split_connector = _ns(send_cache_info_to_prefill=lambda t: None)
        eng.enable_decode_cache_task = False

        eng._decode_process_splitwise_requests()
        time.sleep(0.15)
        eng.running = False
        time.sleep(0.05)
        assert len(prefill_inserted) >= 1

    def test_decode_process_splitwise_prefilled_v1_kvcache(self, monkeypatch):
        """Prefilled path: v1 kvcache, normal token processing."""
        from fastdeploy.engine.request import RequestOutput

        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)

        out = RequestOutput(request_id="pf-v1-1", outputs=_ns(tool_calls=None, token_ids=[10]))
        out.metrics = _ns(decode_recv_first_token_time=0)
        out.error_code = 200

        call_count = [0]
        added = []

        def fake_disagg_empty():
            nonlocal call_count
            call_count[0] += 1
            return call_count[0] > 1

        eng.engine_worker_queue.disaggregate_queue_empty = fake_disagg_empty
        eng.engine_worker_queue.get_disaggregated_tasks = lambda: [("batch", [out])]
        eng.scheduler.has_request = lambda rid: True
        eng.resource_manager.add_prefilled_request = lambda ro: added.append(ro)
        eng.cfg.splitwise_version = "v0"
        eng.split_connector = _ns(send_cache_info_to_prefill=lambda t: None)
        eng.enable_decode_cache_task = False
        eng.token_processor = _ns(tokens_counter={}, clear_data=lambda: None)

        eng._decode_process_splitwise_requests()
        time.sleep(0.15)
        eng.running = False
        time.sleep(0.05)
        assert len(added) >= 1

    def test_decode_process_splitwise_prefilled_v1_error(self, monkeypatch):
        """Prefilled path: v1 kvcache, error_code != 200 → recycle."""
        from fastdeploy.engine.request import RequestOutput

        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)

        out = RequestOutput(request_id="pf-err-1", outputs=_ns(tool_calls=None, token_ids=[10]))
        out.metrics = _ns(decode_recv_first_token_time=0)
        out.error_code = 500
        out.error_msg = "Prefill failed"

        call_count = [0]
        recycled = []

        def fake_disagg_empty():
            nonlocal call_count
            call_count[0] += 1
            return call_count[0] > 1

        eng.engine_worker_queue.disaggregate_queue_empty = fake_disagg_empty
        eng.engine_worker_queue.get_disaggregated_tasks = lambda: [("batch", [out])]
        eng.scheduler.has_request = lambda rid: True
        eng.resource_manager.pre_recycle_resource = lambda rid: recycled.append(rid)
        eng.cfg.splitwise_version = "v0"
        eng.split_connector = _ns(send_cache_info_to_prefill=lambda t: None)
        eng.enable_decode_cache_task = False
        eng.token_processor = _ns(tokens_counter={}, clear_data=lambda: None)

        eng._decode_process_splitwise_requests()
        time.sleep(0.15)
        eng.running = False
        time.sleep(0.05)
        assert "pf-err-1" in recycled

    def test_decode_process_splitwise_prefilled_v1_eos(self, monkeypatch):
        """Prefilled path: v1 kvcache + internal adapter, empty token_ids → recycle."""
        from fastdeploy.engine.request import RequestOutput

        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)

        out = RequestOutput(request_id="pf-eos-1", outputs=_ns(tool_calls=None, token_ids=[]))
        out.metrics = _ns(decode_recv_first_token_time=0)
        out.error_code = 200

        call_count = [0]
        recycled = []

        def fake_disagg_empty():
            nonlocal call_count
            call_count[0] += 1
            return call_count[0] > 1

        eng.engine_worker_queue.disaggregate_queue_empty = fake_disagg_empty
        eng.engine_worker_queue.get_disaggregated_tasks = lambda: [("batch", [out])]
        eng.scheduler.has_request = lambda rid: True
        eng.resource_manager.pre_recycle_resource = lambda rid: recycled.append(rid)
        eng.cfg.splitwise_version = "v0"
        eng.split_connector = _ns(send_cache_info_to_prefill=lambda t: None)
        eng.enable_decode_cache_task = False
        eng.token_processor = _ns(tokens_counter={}, clear_data=lambda: None)

        eng._decode_process_splitwise_requests()
        time.sleep(0.15)
        eng.running = False
        time.sleep(0.05)
        assert "pf-eos-1" in recycled

    # -- _register_to_router inner loop --

    def test_register_to_router_inner_success(self, monkeypatch):
        """Inner _register function: successful registration."""
        eng = _make_engine(monkeypatch)
        eng.cfg.router_config.router = "http://router:8000"
        eng.cfg.router_config.api_server_host = "localhost"
        eng.cfg.router_config.api_server_port = 8080
        eng.cfg.register_info = {"host": "localhost"}

        monkeypatch.setattr("fastdeploy.engine.common_engine.check_service_health", lambda url: True)

        call_count = [0]

        class FakeResp:
            ok = True

        def fake_post(url, json=None, timeout=None):
            nonlocal call_count
            call_count[0] += 1
            return FakeResp()

        monkeypatch.setattr("fastdeploy.engine.common_engine.requests.post", fake_post)
        # Replace time.sleep to stop loop after first iteration
        sleep_count = [0]
        original_sleep = time.sleep

        def fast_sleep(secs):
            nonlocal sleep_count
            sleep_count[0] += 1
            if sleep_count[0] >= 2:
                raise KeyboardInterrupt  # break out of while True
            original_sleep(0.001)

        monkeypatch.setattr("fastdeploy.engine.common_engine.time.sleep", fast_sleep)

        # Capture thread target and run it directly
        captured = []

        class FakeThread:
            def __init__(self, target=None, daemon=None):
                self.target = target
                captured.append(target)

            def start(self):
                try:
                    self.target()
                except KeyboardInterrupt:
                    pass

        monkeypatch.setattr("fastdeploy.engine.common_engine.threading.Thread", FakeThread)
        eng._register_to_router()
        assert call_count[0] >= 1

    def test_register_to_router_inner_health_fail(self, monkeypatch):
        """Inner _register function: health check fails, waits and retries."""
        eng = _make_engine(monkeypatch)
        eng.cfg.router_config.router = "http://router:8000"
        eng.cfg.router_config.api_server_host = "localhost"
        eng.cfg.router_config.api_server_port = 8080
        eng.cfg.register_info = {"host": "localhost"}

        health_count = [0]

        def fake_health(url):
            nonlocal health_count
            health_count[0] += 1
            return False

        monkeypatch.setattr("fastdeploy.engine.common_engine.check_service_health", fake_health)

        sleep_count = [0]

        def fast_sleep(secs):
            nonlocal sleep_count
            sleep_count[0] += 1
            if sleep_count[0] >= 3:
                raise KeyboardInterrupt
            time.sleep(0.001)

        monkeypatch.setattr("fastdeploy.engine.common_engine.time.sleep", fast_sleep)

        class FakeThread:
            def __init__(self, target=None, daemon=None):
                self.target = target

            def start(self):
                try:
                    self.target()
                except KeyboardInterrupt:
                    pass

        monkeypatch.setattr("fastdeploy.engine.common_engine.threading.Thread", FakeThread)
        eng._register_to_router()
        assert health_count[0] >= 1

    def test_register_to_router_inner_post_fail(self, monkeypatch):
        """Inner _register function: post returns non-ok response."""
        eng = _make_engine(monkeypatch)
        eng.cfg.router_config.router = "http://router:8000"
        eng.cfg.router_config.api_server_host = "localhost"
        eng.cfg.router_config.api_server_port = 8080
        eng.cfg.register_info = {"host": "localhost"}

        monkeypatch.setattr("fastdeploy.engine.common_engine.check_service_health", lambda url: True)

        class FailResp:
            ok = False
            status_code = 500
            text = "Internal Error"

        monkeypatch.setattr("fastdeploy.engine.common_engine.requests.post", lambda **kw: FailResp())

        sleep_count = [0]

        def fast_sleep(secs):
            nonlocal sleep_count
            sleep_count[0] += 1
            if sleep_count[0] >= 2:
                raise KeyboardInterrupt
            time.sleep(0.001)

        monkeypatch.setattr("fastdeploy.engine.common_engine.time.sleep", fast_sleep)

        class FakeThread:
            def __init__(self, target=None, daemon=None):
                self.target = target

            def start(self):
                try:
                    self.target()
                except KeyboardInterrupt:
                    pass

        monkeypatch.setattr("fastdeploy.engine.common_engine.threading.Thread", FakeThread)
        eng._register_to_router()
        # test reaches log error path — no crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
