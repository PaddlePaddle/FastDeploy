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


def _ns(**kw):
    return SimpleNamespace(**kw)


class _FakeSignal:
    def __init__(self, value=None):
        self.value = value if value is not None else np.zeros([1], dtype=np.int32)
        self.cleared = False

    def clear(self):
        self.cleared = True


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *a, **kw):
        self.calls.append((a, kw))


def _make_cfg(**overrides):
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
        load_config=_ns(load_strategy="default", rsync_config={}, dynamic_load_weight=False, load_choices="default"),
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
    eng.resource_manager = _ns(
        stop_flags=np.array([True, True, True, True], dtype=bool),
        check_and_free_block_tables=lambda: None,
        cache_manager=_ns(
            launch_cache_manager=lambda **kw: [],
            shm_cache_task_flag_broadcast=_FakeSignal(),
            cache_ready_signal=_FakeSignal(),
        ),
    )
    eng.scheduler = _ns(
        put_requests=lambda *a: [],
        get_requests=lambda **kw: [],
        put_results=lambda *a: None,
        get_results=lambda: [],
        start=lambda *a, **kw: None,
        reset=lambda: None,
        name="local",
    )
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
    eng.token_processor = _ns(
        clear_data=lambda: None,
        number_of_tasks=0,
        number_of_input_tokens=0,
    )
    eng.engine_worker_queue = _ns(
        clear_data=lambda: None,
        put_tasks=lambda *a: None,
        exist_tasks=lambda: False,
    )
    eng.split_connector = _ns(start_receiver=lambda: None)
    eng.partial_chunked_tokens = [0, eng.cfg.scheduler_config.max_num_batched_tokens]
    eng._ctrl_worker_output_queues = []
    return eng


def _make_task(rid="r1", preempted=False, disagg=None, carrier=None):
    return _ns(
        request_id=rid,
        trace_carrier=carrier,
        prompt_token_ids_len=32,
        metrics=_ns(
            inference_start_time=0,
            scheduler_recv_req_time=time.time(),
            add_req_to_resource_manager_time=0,
            ask_decode_resource_start_time=0,
            ask_decode_resource_finish_time=0,
            decode_inference_start_time=0,
            engine_get_req_time=0,
            decode_recv_req_time=0,
            decode_preallocate_req_time=0,
        ),
        disaggregate_info=disagg,
        has_been_preempted_before=preempted,
        set=lambda k, v: None,
        user="test",
    )


def _patch_tracing(monkeypatch):
    monkeypatch.setattr("fastdeploy.engine.common_engine.trace_print", lambda *a, **kw: None)
    monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *a, **kw: None)
    monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *a: None)
    monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", lambda *a: None)
    monkeypatch.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", lambda *a: None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQueriesAndHealth:
    def test_task_finished_and_all_finished(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.resource_manager.stop_flags = np.array([True, False], dtype=bool)
        assert eng.task_is_finished(0)
        assert not eng.task_is_finished(1)
        assert not eng.all_tasks_finished()
        eng.resource_manager.stop_flags[:] = True
        assert eng.all_tasks_finished()

    def test_check_and_free_block_tables(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        rec = _Recorder()
        eng.resource_manager.check_and_free_block_tables = rec
        eng.check_and_free_block_tables()
        assert len(rec.calls) == 1

    def test_scheduler_unhandled_request_num(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        # callable, positive
        eng.scheduler.get_unhandled_request_num = lambda: 5
        assert eng._get_scheduler_unhandled_request_num() == 5
        # not callable
        eng.scheduler.get_unhandled_request_num = "nope"
        assert eng._get_scheduler_unhandled_request_num() == 0
        # negative clamped
        eng.scheduler.get_unhandled_request_num = lambda: -3
        assert eng._get_scheduler_unhandled_request_num() == 0
        # exception
        eng.scheduler.get_unhandled_request_num = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        assert eng._get_scheduler_unhandled_request_num() == 0

    def test_check_health(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        # zero signal → healthy
        eng.worker_healthy_live_signal.value[0] = 0
        ok, msg = eng.check_health()
        assert ok
        # recent signal → healthy
        eng.worker_healthy_live_signal.value = np.array([time.time()], dtype=np.float64)
        ok, msg = eng.check_health()
        assert ok
        # stale signal → unhealthy
        eng.worker_healthy_live_signal.value = np.array([time.time() - 60], dtype=np.float64)
        ok, msg = eng.check_health(time_interval_threashold=30)
        assert not ok
        assert "Not Healthy" in msg

    def test_worker_processes_ready(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.worker_num_per_node = 2
        eng.worker_ready_signal.value = np.array([1, 0], dtype=np.int32)
        assert not eng._worker_processes_ready()
        eng.worker_ready_signal.value = np.array([1, 1], dtype=np.int32)
        assert eng._worker_processes_ready()


class TestControl:
    def test_resume_and_is_paused(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        # is_paused reports state
        eng.is_paused = True
        assert eng._control_is_paused(_ns(request_id="r")) == {"is_paused": True}
        # resume clears pause
        eng._control_resume(_ns(request_id="r"))
        assert not eng.is_paused
        assert eng._control_is_paused(_ns(request_id="r")) == {"is_paused": False}
        # resume when not paused is noop
        eng._control_resume(_ns(request_id="r"))
        assert not eng.is_paused

    def test_update_weights(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        # must be paused first
        eng.is_paused = False
        with pytest.raises(Exception, match="Pause"):
            eng._control_update_weights(_ns(request_id="r"))
        # paused → calls worker
        eng.is_paused = True
        called = []
        eng._call_worker = lambda cr, t: called.append(cr.request_id)
        eng._control_update_weights(_ns(request_id="r"))
        assert called == ["r"]

    def test_run_control_method(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append((rid, r)))
        # unknown method
        eng.run_control_method(_ns(request_id="r1", method="x", params={}, get_method=lambda: "x"))
        assert sent[0][0] == "r1"
        # success
        eng._control_ok = lambda cr: {"ok": True}
        eng.run_control_method(_ns(request_id="r2", method="ok", params={}, get_method=lambda: "ok"))
        assert sent[1][0] == "r2"
        # handler raises
        eng._control_err = lambda cr: (_ for _ in ()).throw(ValueError("bad"))
        eng.run_control_method(_ns(request_id="r3", method="err", params={}, get_method=lambda: "err"))
        assert sent[2][0] == "r3"

    def test_control_pause(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        eng.cfg.scheduler_config.name = "local"
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=lambda *a: None, clear_data=lambda: None)
        eng.resource_manager.log_status = lambda: None
        eng.resource_manager.preempted_all = lambda: []
        eng.resource_manager.cache_manager = _ns(reset=lambda: None)
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append(rid))
        # with inflight requests
        eng.scheduler.get_inflight_requests = lambda: [_ns(request_id="inf-1")]
        eng._control_pause(_ns(request_id="p1"))
        assert eng.is_paused
        assert len(sent) == 1
        # not v1 raises
        eng.is_paused = False
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        with pytest.raises(Exception, match="pause only supported"):
            eng._control_pause(_ns(request_id="p2"))
        # not local raises
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        eng.cfg.scheduler_config.name = "dp"
        with pytest.raises(Exception, match="pause only supported in local"):
            eng._control_pause(_ns(request_id="p3"))

    def test_control_pause_with_running_reqs(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        eng.cfg.scheduler_config.name = "local"
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=lambda *a: None, clear_data=lambda: None)
        eng.resource_manager.log_status = lambda: None
        eng.resource_manager.preempted_all = lambda: [_ns(task_type="PREEMPTED")]
        eng.resource_manager.get_real_bsz = lambda: None
        eng.resource_manager.real_bsz = 1
        eng.resource_manager.wait_worker_inflight_requests_finish = lambda timeout: None
        eng.resource_manager.cache_manager = _ns(reset=lambda: None)
        eng.scheduler.get_inflight_requests = lambda: []
        eng._control_pause(_ns(request_id="p"))
        assert eng.is_paused

    def test_control_pause_already_paused_and_queue_drain(self, monkeypatch):
        monkeypatch.setattr("fastdeploy.engine.common_engine.time.sleep", lambda s: None)
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        eng.cfg.scheduler_config.name = "local"
        eng.is_paused = True  # already paused → covers "already paused" log
        exist_calls = [0]

        def mock_exist():
            exist_calls[0] += 1
            return exist_calls[0] < 3  # True twice then False → exercises loop

        eng.engine_worker_queue = _ns(exist_tasks=mock_exist, put_tasks=lambda *a: None, clear_data=lambda: None)
        eng.resource_manager.log_status = lambda: None
        eng.resource_manager.preempted_all = lambda: []
        eng.resource_manager.cache_manager = _ns(reset=lambda: None)
        eng.scheduler.get_inflight_requests = lambda: []
        eng._control_pause(_ns(request_id="p2"))
        assert eng.is_paused
        assert exist_calls[0] >= 3

    def test_standard_and_adapter(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append(rid))
        # standard mode
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        eng._send_error_response("req-1", "err", 503)
        assert sent[-1] == "req-1"
        # adapter mode
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        eng._send_error_response("req-2", "err")
        assert sent[-1] is None


class TestMiscUtilities:
    def test_decode_token_paths(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        # return text disabled
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", False)
        delta, tids = eng._decode_token([1, 2], "r", is_end=False)
        assert delta == "" and tids == [1, 2]
        # return text enabled, non-empty delta
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        eng.data_processor = _ns(
            ids2tokens=lambda t, r: ("hello", [1, 2, 3], None),
            decode_status={"r1": [0, 2]},
        )
        delta, tids = eng._decode_token([1, 2, 3], "r1", is_end=False)
        assert delta == "hello" and tids == [1, 2]
        # is_end cleans up
        eng.data_processor.decode_status["r2"] = [0, 1]
        eng.data_processor.ids2tokens = lambda t, r: ("end", [10], None)
        eng._decode_token([10], "r2", is_end=True)
        assert "r2" not in eng.data_processor.decode_status
        # empty delta
        eng.data_processor.ids2tokens = lambda t, r: ("", [], None)
        eng.data_processor.decode_status["r3"] = [0, 0]
        delta, tids = eng._decode_token([5], "r3", is_end=False)
        assert delta == "" and tids == []

    def test_clear_data(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.send_response_server = _ns(req_dict={})
        eng.recv_request_server = _ns(req_dict={})
        assert eng.clear_data()
        # with cache_task_queue
        eng.cache_task_queue = _ns(clear_transfer_task=lambda: None)
        assert eng.clear_data()
        # exception path
        eng.token_processor.clear_data = lambda: (_ for _ in ()).throw(RuntimeError("fail"))
        assert not eng.clear_data()

    def test_setting_environ_variables(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        result = eng._setting_environ_variables()
        assert "FLAGS_use_append_attn=1" in result
        assert "OMP_NUM_THREADS=3" in result
        # splitwise prefill v0
        eng.cfg.scheduler_config.splitwise_role = "prefill"
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        assert "FLAGS_use_pd_disaggregation=1" in eng._setting_environ_variables()
        # splitwise prefill v1
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        assert "FLAGS_use_pd_disaggregation_per_chunk=1" in eng._setting_environ_variables()
        # multimodal
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.model_config.enable_mm = True
        assert "FLAGS_max_partition_size=1024" in eng._setting_environ_variables()

    def test_update_requests_chunk_size(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        # disabled → noop
        eng.cfg.cache_config.enable_chunked_prefill = False
        reqs = [_ns(prompt_token_ids_len=100)]
        eng.update_requests_chunk_size(reqs)
        assert not hasattr(reqs[0], "prefill_chunk_info")
        # empty list → noop
        eng.cfg.cache_config.enable_chunked_prefill = True
        eng.update_requests_chunk_size([])
        # single request
        eng.cfg.cache_config.block_size = 16
        eng.cfg.scheduler_config.max_num_batched_tokens = 128
        eng.partial_chunked_tokens = [0, 128]
        chunk_info = {}
        req = _ns(prompt_token_ids_len=64, set=lambda k, v: chunk_info.update({k: v}))
        eng.update_requests_chunk_size([req])
        assert sum(chunk_info["prefill_chunk_info"]) == 64
        # multiple requests
        eng.cfg.scheduler_config.max_num_batched_tokens = 256
        eng.cfg.max_num_partial_prefills = 2
        eng.partial_chunked_tokens = [0, 256, 128]
        chunks = [{}, {}]
        reqs = [_ns(prompt_token_ids_len=100, set=lambda k, v, i=i: chunks[i].update({k: v})) for i in range(2)]
        eng.update_requests_chunk_size(reqs)
        for c in chunks:
            assert "prefill_chunk_info" in c

    def test_chunk_size_remainder_distribution(self, monkeypatch):
        """Trigger the second-pass loop that distributes remaining batched tokens."""
        eng = _make_engine(monkeypatch)
        eng.cfg.cache_config.enable_chunked_prefill = True
        eng.cfg.cache_config.block_size = 16
        eng.cfg.scheduler_config.max_num_batched_tokens = 256
        eng.cfg.max_num_partial_prefills = 2
        # partial_chunked_tokens[2]=32 forces small initial allocation, leaving remainder
        eng.partial_chunked_tokens = [0, 256, 32]
        chunks = [{}, {}]
        reqs = [_ns(prompt_token_ids_len=100, set=lambda k, v, i=i: chunks[i].update({k: v})) for i in range(2)]
        eng.update_requests_chunk_size(reqs)
        for c in chunks:
            total = sum(c["prefill_chunk_info"])
            assert total == 100  # all tokens distributed


class TestExitSubServices:
    def test_exit_basic_and_zmq(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        closed = []
        eng.send_response_server = _ns(close=lambda: closed.append("send"))
        eng.recv_request_server = _ns(close=lambda: closed.append("recv"))
        eng._exit_sub_services()
        assert not eng.running
        assert eng.exist_task_signal.cleared
        assert eng.exist_swapped_task_signal.cleared
        assert "send" in closed and "recv" in closed

    def test_exit_async_llm(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        eng.worker_proc = None
        eng.worker_ready_signal = _FakeSignal()
        eng.loaded_model_signal = _FakeSignal()
        eng._exit_sub_services()
        assert eng.worker_ready_signal.cleared and eng.loaded_model_signal.cleared

    def test_exit_engine_worker_queue_server(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        cleaned = []
        eng.engine_worker_queue_server = _ns(cleanup=lambda: cleaned.append(1))
        eng._exit_sub_services()
        assert len(cleaned) == 1

    def test_exit_async_worker_and_cache(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        killed = []
        eng.worker_proc = _ns(pid=100)
        monkeypatch.setattr("os.getpgid", lambda pid: pid)
        monkeypatch.setattr("os.killpg", lambda pgid, sig: killed.append(pgid))
        eng.worker_ready_signal = _FakeSignal()
        eng.loaded_model_signal = _FakeSignal()
        cache_p = _ns(pid=200)
        eng.cache_manager_processes = [cache_p]
        eng.resource_manager.cache_manager.shm_cache_task_flag_broadcast = _FakeSignal()
        eng.resource_manager.cache_manager.cache_ready_signal = _FakeSignal()
        eng._exit_sub_services()
        assert 100 in killed and 200 in killed

    def test_exit_cache_task_queue_variants(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        eng.worker_proc = None
        eng.worker_ready_signal = _FakeSignal()
        eng.loaded_model_signal = _FakeSignal()
        # cleanup method
        cleaned = []
        eng.cache_task_queue = _ns(cleanup=lambda: cleaned.append(1))
        eng._exit_sub_services()
        assert len(cleaned) == 1
        # manager.shutdown
        eng2 = _make_engine(monkeypatch)
        eng2.use_async_llm = True
        eng2.worker_proc = None
        eng2.worker_ready_signal = _FakeSignal()
        eng2.loaded_model_signal = _FakeSignal()
        shutdown = []
        eng2.cache_task_queue = _ns(manager=_ns(shutdown=lambda: shutdown.append(1)))
        eng2._exit_sub_services()
        assert len(shutdown) == 1

    def test_exit_with_recv_control_cmd(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        closed = []
        eng.recv_control_cmd_server = _ns(close=lambda: closed.append(1))
        eng._exit_sub_services()
        assert len(closed) == 1

    def test_exit_get_profile_signal(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        eng.worker_proc = None
        eng.worker_ready_signal = _FakeSignal()
        eng.loaded_model_signal = _FakeSignal()
        eng.get_profile_block_num_signal = _FakeSignal()
        eng._exit_sub_services()
        assert eng.get_profile_block_num_signal.cleared


class TestSetup:
    def test_create_data_processor(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        processor = _ns(
            get_mm_max_tokens_per_item=lambda ml: None,
            create_processor=lambda: None,
        )
        # mm_max_tokens_per_item is None → skip postprocess
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.InputPreprocessor",
            lambda *a, **kw: _ns(create_processor=lambda: processor),
        )
        eng.create_data_processor()
        assert eng.mm_max_tokens_per_item is None
        # mm_max_tokens_per_item is not None → calls postprocess
        processor.get_mm_max_tokens_per_item = lambda ml: {"image": 128}
        eng.cfg.get_max_chunk_tokens = lambda mm: 256
        eng.cfg.cache_config.postprocess = lambda mt, ms: None
        eng.create_data_processor()
        assert eng.mm_max_tokens_per_item == {"image": 128}

    def test_start_cache_service(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        kw = {}
        eng.resource_manager.cache_manager.launch_cache_manager = lambda **k: kw.update(k) or []
        eng.start_cache_service(["0", "1"], 12345)
        assert kw["device_ids"] == ["0", "1"]

    def test_init_worker_monitor_signals(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.IPCSignal", lambda **kw: _FakeSignal(kw.get("array")))
        eng._init_worker_monitor_signals()
        for attr in [
            "exist_task_signal",
            "exist_swapped_task_signal",
            "exist_prefill_task_signal",
            "worker_healthy_live_signal",
            "cache_ready_signal",
            "swap_space_ready_signal",
            "cache_transfer_inited_signal",
            "model_weights_status_signal",
            "prefix_tree_status_signal",
            "kv_cache_status_signal",
        ]:
            assert hasattr(eng, attr)

    def test_init_worker_signals_variants(self, monkeypatch):
        monkeypatch.setattr("fastdeploy.engine.common_engine.IPCSignal", lambda **kw: _FakeSignal(kw.get("array")))
        monkeypatch.setattr("fastdeploy.engine.common_engine.paddle.is_compiled_with_custom_device", lambda x: False)
        # basic (no profile)
        eng = _make_engine(monkeypatch)
        eng.ipc_signal_suffix = 12345
        eng.do_profile = 0
        eng._init_worker_signals()
        assert hasattr(eng, "worker_ready_signal") and hasattr(eng, "loaded_model_signal")
        assert not hasattr(eng, "get_profile_block_num_signal")
        # with profile
        eng.do_profile = 1
        eng._init_worker_signals()
        assert hasattr(eng, "get_profile_block_num_signal")
        # prefix caching → launched_cache_manager_signal
        eng2 = _make_engine(monkeypatch)
        eng2.ipc_signal_suffix = 12345
        eng2.do_profile = 0
        eng2.cfg.cache_config.enable_prefix_caching = True
        eng2._init_worker_signals()
        assert hasattr(eng2, "launched_cache_manager_signal")
        # expert parallel → launched_expert_service_signal
        eng3 = _make_engine(monkeypatch)
        eng3.ipc_signal_suffix = 12345
        eng3.do_profile = 0
        eng3.cfg.parallel_config.enable_expert_parallel = True
        eng3.cfg.parallel_config.data_parallel_size = 2
        eng3.cfg.nnode = 1
        eng3._init_worker_signals()
        assert hasattr(eng3, "launched_expert_service_signal")

    def test_launch_mixed_and_prefill(self, monkeypatch):
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_MULTI_API_SERVER", False)
        # mixed → no splitwise thread
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.scheduler_config.name = "local"
        eng.launch_components()
        assert not hasattr(eng, "splitwise_receive_thread")
        # prefill → splitwise thread started
        eng2 = _make_engine(monkeypatch)
        eng2.cfg.scheduler_config.splitwise_role = "prefill"
        eng2.cfg.scheduler_config.name = "local"
        eng2.launch_components()
        assert hasattr(eng2, "splitwise_receive_thread")

    def test_launch_splitwise_and_dp_scheduler(self, monkeypatch):
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_MULTI_API_SERVER", False)
        # splitwise scheduler
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "prefill"
        eng.cfg.scheduler_config.name = "splitwise"
        started = []
        eng.scheduler.start = lambda *a, **kw: started.append(a)
        eng.launch_components()
        assert started[0][0] == "prefill"
        # dp scheduler
        eng2 = _make_engine(monkeypatch)
        eng2.cfg.scheduler_config.splitwise_role = "mixed"
        eng2.cfg.scheduler_config.name = "dp"
        started2 = []
        eng2.scheduler.start = lambda *a, **kw: started2.append(a)
        eng2.launch_components()
        assert len(started2) == 1

    def test_stop_profile(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.do_profile = 1
        eng.get_profile_block_num_signal = _FakeSignal(np.array([100], dtype=np.int32))
        eng.worker_proc = None
        reset_calls = []
        eng.cfg.cache_config.reset = lambda n: reset_calls.append(n)
        eng.resource_manager.reset_cache_config = lambda cc: None
        eng.ipc_signal_suffix = 12345
        eng._stop_profile()
        assert eng.do_profile == 0 and reset_calls == [100]
        # with prefix caching → starts cache service
        eng2 = _make_engine(monkeypatch)
        eng2.do_profile = 1
        eng2.get_profile_block_num_signal = _FakeSignal(np.array([50], dtype=np.int32))
        eng2.worker_proc = None
        eng2.cfg.cache_config.reset = lambda n: None
        eng2.resource_manager.reset_cache_config = lambda cc: None
        eng2.cfg.cache_config.enable_prefix_caching = True
        eng2.ipc_signal_suffix = 12345
        eng2._stop_profile()
        assert hasattr(eng2, "cache_manager_processes")


class TestStartAndRegister:
    def test_disabled(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.router_config.router = None
        eng._register_to_router()  # noop

    def test_enabled_starts_thread(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.router_config.router = "http://router:8080"
        eng._register_to_router()
        assert hasattr(eng, "_register_to_router")

    def test_start_sets_running(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        eng.token_processor.tasks_queue = None
        eng.token_processor.run = lambda: None
        eng.cfg.router_config.router = None
        eng._schedule_request_to_worker = lambda: None
        eng.start()
        assert eng.running and hasattr(eng, "insert_task_to_worker_thread")

    def test_start_v1_scheduler(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        eng.token_processor.tasks_queue = None
        eng.token_processor.run = lambda: None
        eng.cfg.router_config.router = None
        eng._schedule_request_to_worker_v1 = lambda: None
        eng.start()
        assert hasattr(eng, "insert_task_to_worker_thread")

    def test_start_decode_role(self, monkeypatch):
        eng = _make_engine(monkeypatch)
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


class TestWorkerService:
    def test_queue_variants(self, monkeypatch):
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.EngineWorkerQueue",
            lambda **kw: _ns(get_server_port=lambda: 12345, cleanup=lambda: None),
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.EngineCacheQueue",
            lambda **kw: _ns(get_server_port=lambda: 9999),
        )
        # no queue
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", False)
        eng = _make_engine(monkeypatch)
        eng.start_worker_queue_service(start_queue=False)
        assert hasattr(eng, "engine_worker_queue")
        # with queue
        eng2 = _make_engine(monkeypatch)
        eng2.start_worker_queue_service(start_queue=True)
        assert hasattr(eng2, "engine_worker_queue_server")
        # prefix caching → cache_task_queue
        eng3 = _make_engine(monkeypatch)
        eng3.cfg.cache_config.enable_prefix_caching = True
        eng3.start_worker_queue_service(start_queue=True)
        assert hasattr(eng3, "cache_task_queue")
        # SHM mode
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", True)
        eng4 = _make_engine(monkeypatch)
        eng4.start_worker_queue_service(start_queue=True)
        assert hasattr(eng4, "engine_worker_queue")

    def _make_cmd_engine(self, monkeypatch, **kw):
        eng = _make_engine(monkeypatch)
        eng.data_processor = _ns(
            tokenizer=_ns(vocab={"a": 0}, get_vocab=lambda: {}, encode=lambda *a, **k: [10]),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        eng.mm_max_tokens_per_item = None
        eng.do_profile = 0
        eng.ipc_signal_suffix = 12345
        for k2, v2 in kw.items():
            setattr(eng, k2, v2)
        return eng

    def _capture_popen(self, monkeypatch):
        captured = {}

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd
            return _ns(pid=9999, poll=lambda: None, stdout=iter([]))

        monkeypatch.setattr("fastdeploy.engine.common_engine.subprocess.Popen", fake_popen)
        return captured

    def test_basic_and_flags(self, monkeypatch):
        captured = self._capture_popen(monkeypatch)
        eng = self._make_cmd_engine(monkeypatch)
        eng._start_worker_service()
        assert "--max_num_seqs" in captured["cmd"]
        assert "--model test-model" in captured["cmd"]
        # store_true flags
        eng2 = self._make_cmd_engine(monkeypatch)
        eng2.cfg.cache_config.enable_prefix_caching = True
        eng2.do_profile = 1
        eng2._start_worker_service()
        assert "--enable_prefix_caching" in captured["cmd"]
        assert "--do_profile" in captured["cmd"]

    def test_sp_model_and_nnode(self, monkeypatch):
        captured = self._capture_popen(monkeypatch)
        eng = self._make_cmd_engine(monkeypatch)
        eng.data_processor.tokenizer = _ns(
            sp_model=type("SP", (), {"__len__": lambda s: 50})(),
            get_vocab=lambda: {},
            encode=lambda *a, **k: [10],
        )
        eng._start_worker_service()
        assert "--ori_vocab_size 50" in captured["cmd"]
        # nnode > 1
        eng2 = self._make_cmd_engine(monkeypatch)
        eng2.cfg.nnode = 2
        eng2.cfg.ips = ["10.0.0.1", "10.0.0.2"]
        eng2._start_worker_service()
        assert "--nnodes 2" in captured["cmd"]

    def test_optional_args(self, monkeypatch):
        captured = self._capture_popen(monkeypatch)
        eng = self._make_cmd_engine(monkeypatch)
        eng.mm_max_tokens_per_item = {"image": 128}
        eng.cfg.structured_outputs_config.logits_processors = ["proc1"]
        eng.cfg.cache_config.num_gpu_blocks_override = 512
        eng._start_worker_service()
        assert "--mm_max_tokens_per_item" in captured["cmd"]
        assert "--logits-processors proc1" in captured["cmd"]
        assert "--num_gpu_blocks_override 512" in captured["cmd"]

    def test_line_break_nested(self, monkeypatch):
        captured = self._capture_popen(monkeypatch)
        eng = self._make_cmd_engine(monkeypatch)
        eng.data_processor.tokenizer = _ns(
            vocab={"a": 0},
            get_vocab=lambda: {},
            encode=lambda *a, **k: {"input_ids": [[10]]},
        )
        eng._start_worker_service()
        assert "--line_break_id 10" in captured["cmd"]

    def test_check_status_success(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.worker_num_per_node = 1
        eng.worker_ready_signal = _FakeSignal(np.array([0], dtype=np.int32))
        eng.worker_init_status = {}
        eng.worker_proc = _ns(
            stdout=iter([b"Loading checkpoint shards: 100\n", b"Start load layer 0\n"]),
            poll=lambda: None,
        )

        def set_ready():
            time.sleep(0.1)
            eng.worker_ready_signal.value[0] = 1

        threading.Thread(target=set_ready, daemon=True).start()
        assert eng.check_worker_initialize_status()

    def test_proc_dies(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.cfg.worker_num_per_node = 1
        eng.worker_ready_signal = _FakeSignal(np.array([0], dtype=np.int32))
        eng.worker_init_status = {}
        eng.worker_proc = _ns(stdout=iter([]), poll=lambda: 1)
        assert not eng.check_worker_initialize_status()

    def test_timeout(self, monkeypatch):
        import asyncio as _asyncio

        eng = _make_engine(monkeypatch)

        class FakeQueue:
            def __init__(self, name):
                self.name = name

            async def get(self, timeout=None):
                await _asyncio.sleep(10)

        eng._ctrl_worker_output_queues = [FakeQueue("q0")]
        eng.engine_worker_queue = _ns(put_tasks=lambda *a: None)
        with pytest.raises(Exception, match="Timeouted"):
            eng._call_worker(_ns(request_id="cw-t"), timeout=0.01)

    def test_call_worker_success(self, monkeypatch):
        eng = _make_engine(monkeypatch)

        class FakeQueue:
            def __init__(self, name):
                self.name = name

            async def get(self, timeout=None):
                return _ns(payload=_ns(request_id="cw-ok", error_code=200, error_message=None, result={"ok": True}))

        eng._ctrl_worker_output_queues = [FakeQueue("q0")]
        eng.engine_worker_queue = _ns(put_tasks=lambda *a: None)
        results = eng._call_worker(_ns(request_id="cw-ok"), timeout=5)
        assert results == [{"ok": True}]

    def test_full_flow(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        eng.do_profile = 0
        monkeypatch.setattr("fastdeploy.engine.common_engine.IPCSignal", lambda **kw: _FakeSignal(kw.get("array")))
        eng.ipc_signal_suffix = 12345
        eng.cfg.cache_config.enable_prefix_caching = False
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.data_processor = _ns(
            tokenizer=_ns(vocab={"a": 0}, get_vocab=lambda: {}, encode=lambda *a, **kw: [10]),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        eng.mm_max_tokens_per_item = None

        def fake_popen(cmd, **kw):
            return _ns(pid=9999, poll=lambda: None, stdout=iter([]))

        monkeypatch.setattr("fastdeploy.engine.common_engine.subprocess.Popen", fake_popen)

        ready_signal = _FakeSignal(np.array([1], dtype=np.int32))

        def fake_init_signals():
            eng.worker_ready_signal = _FakeSignal(np.array([1], dtype=np.int32))
            eng.loaded_model_signal = ready_signal

        eng._init_worker_signals = fake_init_signals
        eng.check_worker_initialize_status = lambda: True
        monkeypatch.setattr("fastdeploy.engine.common_engine.time.sleep", lambda s: None)
        eng.start_worker_service(async_llm_pid=None)
        assert hasattr(eng, "worker_proc")

    def test_flow_with_profiling(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        eng.do_profile = 1
        monkeypatch.setattr("fastdeploy.engine.common_engine.IPCSignal", lambda **kw: _FakeSignal(kw.get("array")))
        eng.ipc_signal_suffix = 12345
        eng.cfg.cache_config.enable_prefix_caching = False
        eng.data_processor = _ns(
            tokenizer=_ns(vocab={"a": 0}, get_vocab=lambda: {}, encode=lambda *a, **kw: [10]),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        eng.mm_max_tokens_per_item = None

        def fake_popen(cmd, **kw):
            return _ns(pid=9999, poll=lambda: None, stdout=iter([]))

        monkeypatch.setattr("fastdeploy.engine.common_engine.subprocess.Popen", fake_popen)

        ready_signal = _FakeSignal(np.array([1], dtype=np.int32))
        profile_signal = _FakeSignal(np.array([100], dtype=np.int32))

        def fake_init_signals():
            eng.worker_ready_signal = _FakeSignal(np.array([1], dtype=np.int32))
            eng.loaded_model_signal = ready_signal
            eng.get_profile_block_num_signal = profile_signal

        eng._init_worker_signals = fake_init_signals
        eng.check_worker_initialize_status = lambda: True
        eng.cfg.cache_config.reset = lambda n: None
        eng.resource_manager.reset_cache_config = lambda cc: None
        monkeypatch.setattr("fastdeploy.engine.common_engine.time.sleep", lambda s: None)
        eng.start_worker_service(async_llm_pid=None)
        assert eng.do_profile == 0

    def test_flow_loads_model_dies(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.use_async_llm = True
        eng.do_profile = 0
        monkeypatch.setattr("fastdeploy.engine.common_engine.IPCSignal", lambda **kw: _FakeSignal(kw.get("array")))
        eng.ipc_signal_suffix = 12345
        eng.cfg.cache_config.enable_prefix_caching = False
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.data_processor = _ns(
            tokenizer=_ns(vocab={"a": 0}, get_vocab=lambda: {}, encode=lambda *a, **kw: [10]),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        eng.mm_max_tokens_per_item = None

        def fake_popen(cmd, **kw):
            return _ns(pid=9999, poll=lambda: 1, stdout=iter([]))

        monkeypatch.setattr("fastdeploy.engine.common_engine.subprocess.Popen", fake_popen)

        def fake_init_signals():
            eng.worker_ready_signal = _FakeSignal(np.array([0], dtype=np.int32))
            eng.loaded_model_signal = _FakeSignal(np.array([0], dtype=np.int32))

        eng._init_worker_signals = fake_init_signals
        eng.check_worker_initialize_status = lambda: False
        monkeypatch.setattr("fastdeploy.engine.common_engine.time.sleep", lambda s: None)
        result = eng.start_worker_service(async_llm_pid=None)
        assert result is False


class TestInsertTasks:
    def test_single_task_not_list(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.cache_config.enable_chunked_prefill = False
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        eng.engine_worker_queue = _ns(put_tasks=lambda t: None)
        assert eng.insert_tasks(_make_task())  # non-list → wraps into list

    def test_mixed_success(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.cache_config.enable_chunked_prefill = False
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        put_calls = []
        eng.engine_worker_queue = _ns(put_tasks=lambda t: put_calls.append(t))
        assert eng.insert_tasks([_make_task()])
        assert len(put_calls) == 1

    def test_exceeds_batch(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.cache_config.enable_chunked_prefill = False
        eng.resource_manager.stop_flags = np.array([True, False, False], dtype=bool)
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        put_calls = []
        eng.engine_worker_queue = _ns(put_tasks=lambda t: put_calls.append(t))
        eng.insert_tasks([_make_task(f"r{i}") for i in range(3)])
        assert len(put_calls[0][0]) == 1

    def test_allocation_fails(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: []
        from fastdeploy.engine.common_engine import EngineError

        with pytest.raises(EngineError):
            eng.insert_tasks([_make_task()])

    def test_prefill_role(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "prefill"
        eng.cfg.cache_config.enable_chunked_prefill = False
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        eng.split_connector = _ns(
            check_decode_allocated=lambda t: (True, None),
            send_cache_info_to_messager=lambda *a: None,
        )
        eng.engine_worker_queue = _ns(put_tasks=lambda t: None)
        assert eng.insert_tasks([_make_task(disagg=_ns(foo=1))])

    def test_prefill_decode_fails(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "prefill"
        eng.split_connector = _ns(check_decode_allocated=lambda t: (False, "D fail"))
        put_results = []
        eng.scheduler.put_results = lambda r: put_results.extend(r)
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        eng.engine_worker_queue = _ns(put_tasks=lambda t: None)
        from fastdeploy.engine.common_engine import EngineError

        with pytest.raises(EngineError):
            eng.insert_tasks([_make_task()])
        assert len(put_results) == 1

    def test_decode_role_and_preempted(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "decode"
        eng.cfg.cache_config.enable_chunked_prefill = False
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        eng.split_connector = _ns(send_cache_info_to_prefill=lambda *a: None)
        eng.engine_worker_queue = _ns(put_tasks=lambda t: None)
        assert eng.insert_tasks([_make_task(disagg=_ns(x=1))])
        # preempted task path
        eng2 = _make_engine(monkeypatch)
        eng2.cfg.scheduler_config.splitwise_role = "mixed"
        eng2.cfg.cache_config.enable_chunked_prefill = False
        eng2.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng2.resource_manager.real_bsz = 1
        eng2.engine_worker_queue = _ns(put_tasks=lambda t: None)
        assert eng2.insert_tasks([_make_task(preempted=True)])

    def test_with_trace_carrier(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.cfg.cache_config.enable_chunked_prefill = False
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        eng.engine_worker_queue = _ns(put_tasks=lambda t: None)
        task = _make_task(carrier={"traceparent": "00-abc"})
        assert eng.insert_tasks([task])

    def test_happy_error_and_eos(self, monkeypatch):
        def _setup(eng):
            eng.cfg.speculative_config = _ns(method="none")
            eng.cfg.scheduler_config.splitwise_role = "decode"
            eng.resource_manager.real_bsz = 1
            eng.resource_manager._recycle_block_tables = lambda r: None

        # happy path
        eng = _make_engine(monkeypatch)
        _setup(eng)
        eng.resource_manager.req_dict = {"r1": 0}
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
        eng.engine_worker_queue = _ns(put_tasks=lambda t: put_calls.append(1))
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        req = _ns(
            request_id="r1",
            outputs=_ns(token_ids=[42], draft_token_ids=None),
            error_code=200,
            error_msg=None,
            num_cached_tokens=5,
            metrics=_ns(decode_recv_req_time=0, decode_preallocate_req_time=0, decode_inference_start_time=0),
        )
        assert eng._insert_prefilled_requests([req])
        assert len(put_calls) == 1
        # error path
        eng2 = _make_engine(monkeypatch)
        _setup(eng2)
        eng2.resource_manager.req_dict = {"re": 0}
        eng2.resource_manager.tasks_list = [
            _ns(
                prompt_token_ids=[0],
                num_cached_tokens=0,
                metrics=_ns(decode_recv_req_time=0, decode_preallocate_req_time=0, decode_inference_start_time=0),
            )
        ]
        eng2.resource_manager.stop_flags = np.array([False], dtype=bool)
        eng2.token_processor = _ns(tokens_counter={"re": 1}, clear_data=lambda: None, number_of_tasks=0)
        results = []
        eng2.scheduler.put_results = lambda r: results.extend(r)
        eng2.engine_worker_queue = _ns(put_tasks=lambda t: None)
        req_e = _ns(
            request_id="re",
            outputs=_ns(token_ids=[1], draft_token_ids=None),
            error_code=500,
            error_msg="fail",
            num_cached_tokens=0,
            metrics=_ns(decode_recv_req_time=0, decode_preallocate_req_time=0, decode_inference_start_time=0),
        )
        eng2._insert_prefilled_requests([req_e])
        assert eng2.resource_manager.stop_flags[0]
        # eos (internal adapter)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        eng3 = _make_engine(monkeypatch)
        _setup(eng3)
        eng3.resource_manager.req_dict = {"reos": 0}
        eng3.resource_manager.tasks_list = [
            _ns(
                prompt_token_ids=[0],
                num_cached_tokens=0,
                metrics=_ns(decode_recv_req_time=0, decode_preallocate_req_time=0, decode_inference_start_time=0),
            )
        ]
        eng3.resource_manager.stop_flags = np.array([False], dtype=bool)
        eng3.token_processor = _ns(tokens_counter={"reos": 1}, clear_data=lambda: None, number_of_tasks=0)
        eng3.engine_worker_queue = _ns(put_tasks=lambda t: None)
        req_eos = _ns(
            request_id="reos",
            outputs=_ns(token_ids=[], draft_token_ids=None),
            error_code=200,
            error_msg=None,
            num_cached_tokens=0,
            metrics=_ns(decode_recv_req_time=0, decode_preallocate_req_time=0, decode_inference_start_time=0),
        )
        eng3._insert_prefilled_requests([req_eos])
        assert eng3.resource_manager.stop_flags[0]


class TestSchedule:
    def test_one_iteration(self, monkeypatch):
        _patch_tracing(monkeypatch)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.num_requests_waiting.dec", lambda n: None
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.num_requests_running.inc", lambda n: None
        )
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.resource_manager.available_batch = lambda: 1
        eng.resource_manager.available_block_num = lambda: 100
        eng.resource_manager.abort_req_ids_set = set()
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        eng.split_connector = _ns(current_request_ids=[], has_splitwise_tasks=lambda: False)
        eng.engine_worker_queue = _ns(
            exist_tasks=lambda: False,
            put_tasks=lambda *a: None,
            num_cache_infos=lambda: 0,
        )
        call_count = [0]
        task = _make_task()

        def get_reqs(**kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return [task]
            eng.running = False
            return []

        eng.scheduler.get_requests = get_reqs
        eng._schedule_request_to_worker()
        assert call_count[0] >= 1

    def test_no_batch_skips(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        call_count = [0]
        eng.resource_manager.available_batch = lambda: 0

        def stop():
            call_count[0] += 1
            if call_count[0] > 2:
                eng.running = False
            return 0

        eng.resource_manager.available_batch = stop
        eng._schedule_request_to_worker()

    def test_splitwise_decode_skips(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "decode"
        eng.resource_manager.available_batch = lambda: 1
        eng.resource_manager.available_block_num = lambda: 100
        eng.resource_manager.abort_req_ids_set = set()
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, num_cache_infos=lambda: 0)
        eng.split_connector = _ns(current_request_ids=[], has_splitwise_tasks=lambda: False)
        task = _ns(
            request_id="v0-dec", metrics=_ns(engine_get_req_time=0, ask_decode_resource_start_time=0), user="test"
        )
        call_count = [0]

        def get_reqs(**kw):
            call_count[0] += 1
            if call_count[0] >= 2:
                eng.running = False
            return [task]

        eng.scheduler.get_requests = get_reqs
        eng._schedule_request_to_worker()
        assert call_count[0] >= 2

    def test_exist_prefill_signal(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.resource_manager.available_batch = lambda: 1
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, num_cache_infos=lambda: 0)
        eng.split_connector = _ns(current_request_ids=[], has_splitwise_tasks=lambda: True)
        eng.exist_prefill_task_signal = _FakeSignal(np.array([1], dtype=np.int32))
        call_count = [0]

        def avail():
            call_count[0] += 1
            if call_count[0] >= 3:
                eng.running = False
            return 1

        eng.resource_manager.available_batch = avail
        eng._schedule_request_to_worker()
        assert call_count[0] >= 3

    def test_num_cache_infos(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.resource_manager.available_batch = lambda: 1
        call_count = [0]

        def ncache():
            call_count[0] += 1
            if call_count[0] >= 2:
                eng.running = False
            return 1

        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, num_cache_infos=ncache)
        eng.split_connector = _ns(current_request_ids=[])
        eng._schedule_request_to_worker()
        assert call_count[0] >= 2

    def test_split_connector_ids(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
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

    def test_mixed_happy(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.resource_manager.available_batch = lambda: 1
        eng.resource_manager.available_block_num = lambda: 100
        eng.resource_manager.abort_req_ids_set = set()
        eng.resource_manager.waiting = []
        eng.resource_manager.schedule = lambda: ([_make_task()], [])
        eng.resource_manager.get_real_bsz = lambda: None
        eng.resource_manager.real_bsz = 1
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=lambda *a: None)
        eng.scheduler.get_requests = lambda **kw: [_make_task()]
        call_count = [0]
        orig_schedule = eng.resource_manager.schedule

        def one_iter():
            call_count[0] += 1
            if call_count[0] > 1:
                eng.running = False
                return [], []
            return orig_schedule()

        eng.resource_manager.schedule = one_iter
        eng._schedule_request_to_worker_v1()

    def test_shutdown_runtime_error(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.resource_manager.waiting = []
        eng.resource_manager.schedule = lambda: ([], [])
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False)
        call_count = [0]

        def schedule():
            call_count[0] += 1
            if call_count[0] > 1:
                raise RuntimeError("cannot schedule new futures after shutdown")
            return [], []

        eng.resource_manager.schedule = schedule
        eng._schedule_request_to_worker_v1()

    def test_error_tasks_send_response(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.resource_manager.waiting = []
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append(rid))
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False)
        call_count = [0]

        def schedule():
            call_count[0] += 1
            if call_count[0] == 1:
                return [], [("req-err", "something failed")]
            eng.running = False
            return [], []

        eng.resource_manager.schedule = schedule
        eng._schedule_request_to_worker_v1()
        assert len(sent) >= 1

    def test_decode_fetch(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "decode"
        eng.resource_manager.available_batch = lambda: 1
        eng.resource_manager.available_block_num = lambda: 100
        eng.resource_manager.abort_req_ids_set = set()
        eng.resource_manager.waiting = []
        eng.scheduler.get_requests = lambda **kw: []
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=lambda *a: None)
        call_count = [0]

        def schedule():
            call_count[0] += 1
            if call_count[0] >= 2:
                eng.running = False
            return [], []

        eng.resource_manager.schedule = schedule
        eng._schedule_request_to_worker_v1()
        assert call_count[0] >= 1

    def test_prefill_with_tasks(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "prefill"
        eng.resource_manager.waiting = []
        eng.resource_manager.available_batch = lambda: 1
        eng.resource_manager.available_block_num = lambda: 100
        eng.resource_manager.abort_req_ids_set = set()
        eng.resource_manager.get_real_bsz = lambda: None
        eng.resource_manager.real_bsz = 1
        eng.split_connector = _ns(
            check_decode_allocated=lambda t: (True, None),
            send_cache_info_to_messager=lambda *a: None,
        )
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=lambda *a: None)
        eng.scheduler.get_requests = lambda **kw: [_make_task()]
        call_count = [0]

        def schedule():
            call_count[0] += 1
            if call_count[0] == 1:
                return [_make_task()], []
            eng.running = False
            return [], []

        eng.resource_manager.schedule = schedule
        eng._schedule_request_to_worker_v1()
        assert call_count[0] >= 1

    def test_decode_preempted_and_prefill_tasks(self, monkeypatch):
        from fastdeploy.engine.request import Request, RequestType

        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "decode"
        eng.resource_manager.waiting = []
        eng.resource_manager.get_real_bsz = lambda: None
        eng.resource_manager.real_bsz = 1
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=lambda *a: None)

        preempted = Request.__new__(Request)
        preempted.request_id = "r-preempt"
        preempted.task_type = RequestType.PREEMPTED
        preempted.has_been_preempted_before = True
        preempted.trace_carrier = None
        preempted.user = "test"
        preempted.metrics = _ns(
            scheduler_recv_req_time=time.time(),
            inference_start_time=0,
            decode_inference_start_time=0,
            add_req_to_resource_manager_time=0,
        )

        prefill = Request.__new__(Request)
        prefill.request_id = "r-prefill"
        prefill.task_type = RequestType.PREFILL
        prefill.has_been_preempted_before = False
        prefill.trace_carrier = None
        prefill.user = "test"
        prefill.metrics = _ns(
            scheduler_recv_req_time=time.time(),
            inference_start_time=0,
            decode_inference_start_time=0,
            add_req_to_resource_manager_time=0,
        )

        put_results_calls = []
        eng.scheduler.put_results = lambda r: put_results_calls.append(r)

        call_count = [0]

        def schedule():
            call_count[0] += 1
            if call_count[0] == 1:
                return [preempted, prefill], []
            eng.running = False
            return [], []

        eng.resource_manager.schedule = schedule
        eng._schedule_request_to_worker_v1()
        assert len(put_results_calls) >= 1

    def test_prefill_role_rescheduled(self, monkeypatch):
        from fastdeploy.engine.request import Request, RequestType

        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.resource_manager.waiting = []
        eng.resource_manager.get_real_bsz = lambda: None
        eng.resource_manager.real_bsz = 1
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=lambda *a: None)

        rescheduled = Request.__new__(Request)
        rescheduled.request_id = "r-resched"
        rescheduled.task_type = RequestType.PREFILL
        rescheduled.has_been_preempted_before = True
        rescheduled.trace_carrier = None
        rescheduled.user = "test"
        rescheduled.metrics = _ns(
            scheduler_recv_req_time=time.time(),
            inference_start_time=0,
            add_req_to_resource_manager_time=0,
        )

        call_count = [0]

        def schedule():
            call_count[0] += 1
            if call_count[0] == 1:
                return [rescheduled], []
            eng.running = False
            return [], []

        eng.resource_manager.schedule = schedule
        eng._schedule_request_to_worker_v1()

    def test_error_tasks_none_failed(self, monkeypatch):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "mixed"
        eng.resource_manager.waiting = []
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append(rid))
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        eng.engine_worker_queue = _ns(exist_tasks=lambda: False)
        call_count = [0]

        def schedule():
            call_count[0] += 1
            if call_count[0] == 1:
                return [], [("req-skip", None), ("req-real", "actual error")]
            eng.running = False
            return [], []

        eng.resource_manager.schedule = schedule
        eng._schedule_request_to_worker_v1()
        assert "req-real" in sent
        assert "req-skip" not in sent


class TestZmq:
    def test_start_zmq_none_pid(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.start_zmq_service(api_server_pid=None)
        assert not hasattr(eng, "recv_request_server")

    def test_start_zmq_ipc_and_adapter(self, monkeypatch):
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.ZmqIpcServer",
            lambda **kw: _ns(
                recv_result_handle=lambda: None,
                close=lambda: None,
            ),
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.ZmqTcpServer",
            lambda **kw: _ns(
                recv_result_handle=lambda: None,
                close=lambda: None,
            ),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.time.sleep", lambda s: None)
        # IPC mode
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        eng = _make_engine(monkeypatch)
        eng.running = False
        eng.start_zmq_service(api_server_pid=1234)
        assert hasattr(eng, "recv_request_server")
        # adapter mode
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        eng2 = _make_engine(monkeypatch)
        eng2.running = False
        eng2.cfg.parallel_config.local_data_parallel_id = 0
        monkeypatch.setattr("fastdeploy.engine.common_engine.InternalAdapter", lambda **kw: _ns())
        eng2.start_zmq_service(api_server_pid=5678)
        assert hasattr(eng2, "internal_adapter")

    def _setup_zmq_engine(self, monkeypatch, recv_data_seq):
        _patch_tracing(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.main_process_metrics.requests_number.inc", lambda: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.num_requests_waiting.inc", lambda n: None
        )
        eng = _make_engine(monkeypatch)
        idx = [0]

        def recv_json(block):
            if idx[0] >= len(recv_data_seq):
                eng.running = False
                return "Context was terminated", None
            item = recv_data_seq[idx[0]]
            idx[0] += 1
            return item

        eng.recv_request_server = _ns(receive_json_once=recv_json)
        eng.send_response_server = _ns(send_response=lambda rid, r: None)
        return eng

    def test_normal_request(self, monkeypatch):
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.Request.from_dict",
            lambda d: _make_task(d["request_id"]),
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.ControlRequest.is_control_request",
            lambda d: False,
        )
        eng = self._setup_zmq_engine(
            monkeypatch,
            [
                (None, {"request_id": "zmq-1", "status": None}),
            ],
        )
        eng.guided_decoding_checker = None
        eng._insert_zmq_task_to_scheduler()

    def test_abort_request(self, monkeypatch):
        from fastdeploy.engine.request import RequestStatus

        monkeypatch.setattr("fastdeploy.engine.common_engine.ControlRequest.is_control_request", lambda d: False)
        eng = self._setup_zmq_engine(
            monkeypatch,
            [
                (None, {"request_id": "abort-1", "status": RequestStatus.ABORT.value}),
            ],
        )
        eng.resource_manager.abort_req_ids_set = set()
        eng._insert_zmq_task_to_scheduler()
        assert "abort-1" in eng.resource_manager.abort_req_ids_set

    def test_control_request(self, monkeypatch):
        monkeypatch.setattr("fastdeploy.engine.common_engine.ControlRequest.is_control_request", lambda d: True)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.ControlRequest.from_dict",
            lambda d: _ns(request_id="ctrl-1", method="test", params={}, get_method=lambda: "test"),
        )
        eng = self._setup_zmq_engine(
            monkeypatch,
            [
                (None, {"request_id": "ctrl-1", "is_control": True}),
            ],
        )
        eng._control_test = lambda cr: {"ok": True}
        eng.run_control_method = lambda cr: None
        eng._insert_zmq_task_to_scheduler()

    def test_paused_drops(self, monkeypatch):
        monkeypatch.setattr("fastdeploy.engine.common_engine.ControlRequest.is_control_request", lambda d: False)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.Request.from_dict",
            lambda d: _make_task(d["request_id"]),
        )
        eng = self._setup_zmq_engine(
            monkeypatch,
            [
                (None, {"request_id": "p-1", "status": None}),
            ],
        )
        eng.is_paused = True
        eng.guided_decoding_checker = None
        dropped = []
        eng._send_error_response = lambda *a: dropped.append(1)
        eng._insert_zmq_task_to_scheduler()
        assert len(dropped) >= 1

    def test_guided_decoding_error(self, monkeypatch):
        _patch_tracing(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.main_process_metrics.requests_number.inc", lambda: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.num_requests_waiting.inc", lambda n: None
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.ControlRequest.is_control_request", lambda d: False)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.Request.from_dict",
            lambda d: _ns(
                request_id="gd-1",
                prompt_token_ids_len=10,
                user="test",
                metrics=_ns(scheduler_recv_req_time=0, engine_get_req_time=0),
                schema=_ns(json_schema='{"bad": true}'),
                trace_carrier=None,
            ),
        )
        eng = _make_engine(monkeypatch)
        idx = [0]

        def recv_json(block):
            if idx[0] >= 1:
                eng.running = False
                return "Context was terminated", None
            idx[0] += 1
            return None, {"request_id": "gd-1", "status": None}

        eng.recv_request_server = _ns(receive_json_once=recv_json)
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append(rid))
        eng.guided_decoding_checker = _ns(schema_format=lambda req: (req, "bad schema"))
        eng._insert_zmq_task_to_scheduler()
        assert len(sent) >= 1

    def test_v1_abort_removes(self, monkeypatch):
        _patch_tracing(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.main_process_metrics.requests_number.inc", lambda: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.num_requests_waiting.inc", lambda n: None
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.ControlRequest.is_control_request", lambda d: False)
        from fastdeploy.engine.request import RequestStatus

        eng = _make_engine(monkeypatch)
        idx = [0]

        def recv_json(block):
            if idx[0] >= 1:
                eng.running = False
                return "Context was terminated", None
            idx[0] += 1
            return None, {"request_id": "v1-abort", "status": RequestStatus.ABORT.value}

        eng.recv_request_server = _ns(receive_json_once=recv_json)
        eng.resource_manager.abort_req_ids_set = set()
        eng.resource_manager.abort = lambda rid: None
        eng._insert_zmq_task_to_scheduler()
        assert "v1-abort" in eng.resource_manager.abort_req_ids_set

    def test_zmq_v1_data_processor_and_error_reconnect(self, monkeypatch):
        _patch_tracing(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.main_process_metrics.requests_number.inc", lambda: None)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.main_process_metrics.num_requests_waiting.inc", lambda n: None
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.ControlRequest.is_control_request", lambda d: False)
        eng = _make_engine(monkeypatch)
        call_count = [0]

        def recv_pyobj(block):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Some ZMQ error", None  # non-termination error → reconnect
            eng.running = False
            return "Context was terminated", None

        eng.recv_request_server = _ns(receive_pyobj_once=recv_pyobj)

        # Reconnected server also stops the loop
        def make_stop_server(**kw):
            def stop_recv(block):
                eng.running = False
                return "Context was terminated", None

            return _ns(receive_pyobj_once=stop_recv)

        monkeypatch.setattr("fastdeploy.engine.common_engine.ZmqIpcServer", make_stop_server)
        eng.cfg.model_config.enable_mm = True  # triggers receive_pyobj_once path
        eng.api_server_pid = 1234
        eng._insert_zmq_task_to_scheduler()
        assert call_count[0] >= 1

    def _make_ro(self, rid, tids, finished=False, decode_type=1):
        from fastdeploy.engine.request import CompletionOutput, RequestOutput

        co = CompletionOutput.__new__(CompletionOutput)
        co.token_ids = tids
        co.decode_type = decode_type
        co.text = ""
        ro = RequestOutput.__new__(RequestOutput)
        ro.request_id = rid
        ro.outputs = co
        ro.finished = finished
        return ro

    def test_non_adapter_inner_loop(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        eng.data_processor = _ns(
            ids2tokens=lambda t, r: ("txt", [1], None),
            decode_status={"r1": [0, 1]},
        )
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append(rid))
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", False)
        call_count = [0]

        def get_results():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"r1": [self._make_ro("r1", [1], decode_type=1)]}
            eng.running = False
            return {}

        eng.scheduler.get_results = get_results
        eng._zmq_send_generated_tokens()
        assert len(sent) >= 1

    def test_non_adapter_decode_type_0(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        eng.data_processor = _ns(
            ids2tokens=lambda t, r: ("hello", [1, 2], None),
            decode_status={"r1": [0, 1]},
        )
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append(rid))
        call_count = [0]

        def get_results():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"r1": [self._make_ro("r1", [1, 2], decode_type=0)]}
            eng.running = False
            return {}

        eng.scheduler.get_results = get_results
        eng._zmq_send_generated_tokens()
        assert len(sent) >= 1

    def test_non_adapter_finished_empty_tokens(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", False)
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append(rid))
        ro = self._make_ro("r2", [], finished=True, decode_type=1)
        call_count = [0]

        def get_results():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"r2": [ro]}
            eng.running = False
            return {}

        eng.scheduler.get_results = get_results
        eng._zmq_send_generated_tokens()
        assert len(sent) >= 1

    def test_adapter_inner_loop(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        eng.data_processor = _ns(
            ids2tokens=lambda t, r: ("tok", [42], None),
            decode_status={"a1": [0, 1]},
        )
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append(rid))
        call_count = [0]

        def get_results():
            call_count[0] += 1
            if call_count[0] == 1:
                return [[self._make_ro("a1", [42], decode_type=0)]]
            eng.running = False
            return []

        eng.scheduler.get_results = get_results
        eng._zmq_send_generated_tokens()
        assert len(sent) >= 1

    def test_adapter_decode_type_0(self, monkeypatch):
        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        eng.data_processor = _ns(
            ids2tokens=lambda tids, rid: ("tok", [42], None),
            decode_status={"a-dec": [0, 1]},
        )
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append((rid, r)))
        call_count = [0]

        def get_results():
            call_count[0] += 1
            if call_count[0] == 1:
                return [[self._make_ro("a-dec", [42], decode_type=0)]]
            eng.running = False
            return []

        eng.scheduler.get_results = get_results
        eng._zmq_send_generated_tokens()
        assert len(sent) >= 1

    def test_non_request_output(self, monkeypatch):

        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append((rid, r)))
        # Use a non-RequestOutput object
        non_output = _ns(outputs=None, finished=False)
        call_count = [0]

        def get_results():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"req-no": [non_output]}
            eng.running = False
            return {}

        eng.scheduler.get_results = get_results
        eng._zmq_send_generated_tokens()
        assert len(sent) >= 1

    def test_non_adapter_accumulate_warning(self, monkeypatch):

        eng = _make_engine(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", False)
        sent = []
        eng.send_response_server = _ns(send_response=lambda rid, r: sent.append(rid))
        ro = self._make_ro("r-acc", [], finished=False, decode_type=1)
        call_count = [0]

        def get_results():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"r-acc": [ro]}
            eng.running = False
            return {}

        eng.scheduler.get_results = get_results
        eng._zmq_send_generated_tokens()
        # Empty tokens, not finished → should NOT send (accumulate path)
        assert len(sent) == 0


class TestDecodeProcessSplitwise:
    def _make_decode_engine(self, monkeypatch, v1=False):
        _patch_tracing(monkeypatch)
        eng = _make_engine(monkeypatch)
        eng.cfg.scheduler_config.splitwise_role = "decode"
        eng.cfg.splitwise_version = "v1" if v1 else "v0"
        eng.enable_decode_cache_task = False
        eng.resource_manager.is_resource_sufficient = lambda n: True
        eng.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        eng.resource_manager.real_bsz = 1
        eng.resource_manager.preallocate_resource_in_d = lambda t: True
        eng.resource_manager.pre_recycle_resource = lambda rid: None
        eng.resource_manager.add_prefilled_request = lambda ro: None
        eng.split_connector = _ns(send_cache_info_to_prefill=lambda *a: None)
        eng.token_processor.tokens_counter = {}
        eng.engine_worker_queue = _ns(
            disaggregate_queue_empty=lambda: True,
            get_disaggregated_tasks=lambda: [],
            put_tasks=lambda *a: None,
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", v1)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        return eng

    def _real_request(self, rid="req-1"):
        from fastdeploy.engine.request import Request

        req = Request.__new__(Request)
        req.request_id = rid
        req.prompt_token_ids_len = 10
        req.metrics = _ns(decode_recv_req_time=0, decode_preallocate_req_time=0)
        req.error_msg = ""
        return req

    def _real_request_output(self, rid="ro-1", token_ids=None, error_code=200):
        from fastdeploy.engine.request import CompletionOutput, RequestOutput

        co = CompletionOutput.__new__(CompletionOutput)
        co.token_ids = [42] if token_ids is None else token_ids
        ro = RequestOutput.__new__(RequestOutput)
        ro.request_id = rid
        ro.outputs = co
        ro.finished = False
        ro.error_code = error_code
        ro.error_msg = ""
        ro.metrics = _ns(decode_recv_first_token_time=0)
        return ro

    def _run_one_iter(self, eng, items):
        """Run one iteration of the decode loop, then stop."""
        call_count = [0]

        def queue_empty():
            return call_count[0] > 0

        def get_tasks():
            call_count[0] += 1
            eng.running = False
            return items

        eng.engine_worker_queue.disaggregate_queue_empty = queue_empty
        eng.engine_worker_queue.get_disaggregated_tasks = get_tasks
        eng._decode_process_splitwise_requests()

    def test_fetch_request_objects(self, monkeypatch):
        eng = self._make_decode_engine(monkeypatch, v1=False)
        req = self._real_request("r-fetch")
        inserted = []
        eng.insert_tasks = lambda t, **kw: inserted.append(t)
        self._run_one_iter(eng, [(0, [req])])
        assert len(inserted) >= 1

    def test_fetch_request_output_objects_v1(self, monkeypatch):
        eng = self._make_decode_engine(monkeypatch, v1=True)
        ro = self._real_request_output("ro-fetch", [10, 20])
        eng.scheduler.has_request = lambda rid: True
        added = []
        eng.resource_manager.add_prefilled_request = lambda r: added.append(r)
        self._run_one_iter(eng, [(0, [ro])])
        assert len(added) >= 1

    def test_allocate_v1_success(self, monkeypatch):
        eng = self._make_decode_engine(monkeypatch, v1=True)
        req = self._real_request("r-v1alloc")
        sent = []
        eng.split_connector.send_cache_info_to_prefill = lambda tasks: sent.append(tasks)
        self._run_one_iter(eng, [(0, [req])])
        assert len(sent) >= 1

    def test_allocate_fail_no_cache_task(self, monkeypatch):
        eng = self._make_decode_engine(monkeypatch, v1=True)
        eng.resource_manager.preallocate_resource_in_d = lambda t: False
        eng.enable_decode_cache_task = False
        req = self._real_request("r-fail")
        eng.split_connector.send_cache_info_to_prefill = lambda tasks: None
        self._run_one_iter(eng, [(0, [req])])
        assert req.error_msg == "Not enough resources"

    def test_prefilled_v1_error_code(self, monkeypatch):
        eng = self._make_decode_engine(monkeypatch, v1=True)
        ro = self._real_request_output("ro-err", [1], error_code=500)
        ro.error_msg = "prefill failed"
        eng.scheduler.has_request = lambda rid: True
        recycled = []
        eng.resource_manager.pre_recycle_resource = lambda rid: recycled.append(rid)
        self._run_one_iter(eng, [(0, [ro])])
        assert "ro-err" in recycled

    def test_prefilled_v1_adapter_eos(self, monkeypatch):
        eng = self._make_decode_engine(monkeypatch, v1=True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        ro = self._real_request_output("ro-eos", token_ids=[])
        eng.scheduler.has_request = lambda rid: True
        eng.token_processor.tokens_counter = {"ro-eos": 1}
        recycled = []
        eng.resource_manager.pre_recycle_resource = lambda rid: recycled.append(rid)
        self._run_one_iter(eng, [(0, [ro])])
        assert "ro-eos" in recycled
        assert ro.finished is True

    def test_prefilled_waiting_no_has_request(self, monkeypatch):
        eng = self._make_decode_engine(monkeypatch, v1=False)
        ro = self._real_request_output("ro-wait")
        eng.scheduler.has_request = lambda rid: False
        prefilled_inserted = []
        eng._insert_prefilled_requests = lambda reqs: prefilled_inserted.extend(reqs)
        self._run_one_iter(eng, [(0, [ro])])
        assert len(prefilled_inserted) == 0


class TestInit:
    """Tests that exercise EngineService.__init__() directly."""

    def _patch_init_deps(self, monkeypatch, v1=False, guided=False, dp_gt_1=False):
        from fastdeploy.engine.common_engine import EngineService

        # Prevent weakref finalizer from running during gc (tested separately)
        monkeypatch.setattr(EngineService, "_exit_sub_services", lambda self: None)

        cfg = _make_cfg()
        cfg.scheduler_config.scheduler = lambda: _ns(
            put_requests=lambda *a: [],
            get_requests=lambda **kw: [],
            put_results=lambda *a: None,
            get_results=lambda: [],
            start=lambda *a, **kw: None,
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", v1)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_CACHE_TASK", "0")
        rm_cls = "ResourceManagerV1" if v1 else "ResourceManager"
        monkeypatch.setattr(
            f"fastdeploy.engine.common_engine.{rm_cls}",
            lambda *a, **kw: _ns(
                scheduler_metrics_logger=None,
                cache_manager=_ns(
                    shm_cache_task_flag_broadcast=_FakeSignal(),
                    cache_ready_signal=_FakeSignal(),
                ),
            ),
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
            lambda *a, **kw: _ns(set_resource_manager=lambda rm: None, set_scheduler_metrics_logger=lambda sml: None),
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.SchedulerMetricsLogger", lambda *a, **kw: _ns())
        monkeypatch.setattr("fastdeploy.engine.common_engine.IPCSignal", lambda **kw: _FakeSignal(kw.get("array")))
        if dp_gt_1:
            cfg.parallel_config.data_parallel_size = 2
            monkeypatch.setattr(
                "fastdeploy.engine.common_engine.get_logger", lambda *a, **kw: _ns(info=lambda *a: None)
            )
        if guided:
            cfg.structured_outputs_config.guided_decoding_backend = "xgrammar"
            monkeypatch.setattr("fastdeploy.engine.common_engine.schema_checker", lambda *a, **kw: _ns())
        return cfg

    def test_init_v0(self, monkeypatch):
        from fastdeploy.engine.common_engine import EngineService

        cfg = self._patch_init_deps(monkeypatch, v1=False)
        eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        assert eng.is_paused is False
        assert eng.mm_max_tokens_per_item is None
        assert eng.guided_decoding_checker is None

    def test_init_v1_async(self, monkeypatch):
        from fastdeploy.engine.common_engine import EngineService

        cfg = self._patch_init_deps(monkeypatch, v1=True)
        eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        assert eng.use_async_llm is True
        assert eng.do_profile == 1
        assert eng.worker_proc is None

    def test_init_dp_gt_1(self, monkeypatch):
        from fastdeploy.engine.common_engine import EngineService

        cfg = self._patch_init_deps(monkeypatch, dp_gt_1=True)
        eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        assert eng.cfg.parallel_config.data_parallel_size == 2

    def test_init_guided_decoding(self, monkeypatch):
        from fastdeploy.engine.common_engine import EngineService

        cfg = self._patch_init_deps(monkeypatch, guided=True)
        eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        assert eng.guided_decoding_checker is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
