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

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

_noop = lambda *a, **kw: None
_ns = SimpleNamespace


class _Sig:
    def __init__(self, v=None):
        self.value = v if v is not None else np.zeros([1], dtype=np.int32)
        self.cleared = False

    def clear(self):
        self.cleared = True


def _make_cfg(**kw):
    c = _ns(
        parallel_config=_ns(
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
        ),
        model_config=_ns(
            model="test",
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
        ),
        cache_config=_ns(
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
        ),
        scheduler_config=_ns(
            max_num_seqs=32,
            max_num_batched_tokens=4096,
            splitwise_role="mixed",
            name="local",
            enable_overlap_schedule=False,
        ),
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
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _eng(mp, **co):
    from fastdeploy.engine.common_engine import EngineService

    e = object.__new__(EngineService)
    e.cfg = _make_cfg(**co)
    e.use_async_llm = e.is_paused = False
    e.running = True
    e._pause_cond = threading.Condition()
    log = _ns(info=_noop, debug=_noop, error=_noop, warning=_noop, exception=_noop)
    e.llm_logger = log
    e.resource_manager = _ns(
        stop_flags=np.array([True] * 4, dtype=bool),
        check_and_free_block_tables=_noop,
        cache_manager=_ns(
            launch_cache_manager=lambda **kw: [], shm_cache_task_flag_broadcast=_Sig(), cache_ready_signal=_Sig()
        ),
    )
    e.scheduler = _ns(
        put_requests=lambda *a: [],
        get_requests=lambda **kw: [],
        put_results=_noop,
        get_results=lambda: [],
        start=_noop,
        reset=_noop,
        name="local",
    )
    for s in (
        "exist_task",
        "exist_swapped_task",
        "exist_prefill_task",
        "worker_healthy_live",
        "cache_ready",
        "swap_space_ready",
        "cache_transfer_inited",
        "model_weights_status",
        "prefix_tree_status",
        "kv_cache_status",
        "loaded_model",
    ):
        setattr(e, f"{s}_signal", _Sig())
    e.worker_ready_signal = _Sig(np.array([0], dtype=np.int32))
    e.token_processor = _ns(clear_data=_noop, number_of_tasks=0, number_of_input_tokens=0)
    e.engine_worker_queue = _ns(clear_data=_noop, put_tasks=_noop, exist_tasks=lambda: False)
    e.split_connector = _ns(start_receiver=_noop)
    e.partial_chunked_tokens = [0, e.cfg.scheduler_config.max_num_batched_tokens]
    e._ctrl_worker_output_queues = []
    return e


def _task(rid="r1", preempted=False, disagg=None, carrier=None):
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


def _ptr(mp):
    mp.setattr("fastdeploy.engine.common_engine.trace_print", _noop)
    mp.setattr("fastdeploy.engine.common_engine.tracing.trace_report_span", _noop)
    mp.setattr("fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", _noop)
    mp.setattr("fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", _noop)
    mp.setattr("fastdeploy.engine.common_engine.tracing.trace_set_thread_info", _noop)


def _zmq_recv(eng, items):
    """Feed items to _insert_zmq_task_to_scheduler, then stop."""
    idx = [0]

    def recv(block):
        if idx[0] >= len(items):
            eng.running = False
            return "Context was terminated", None
        r = items[idx[0]]
        idx[0] += 1
        return r

    eng.recv_request_server = _ns(receive_json_once=recv)
    eng._insert_zmq_task_to_scheduler()


# ---------------------------------------------------------------------------


class TestInit:
    def _deps(self, mp, v1=False, guided=False, dp=1):
        from fastdeploy.engine.common_engine import EngineService

        mp.setattr(EngineService, "_exit_sub_services", lambda self: None)
        cfg = _make_cfg()
        cfg.scheduler_config.scheduler = lambda: _ns(
            put_requests=lambda *a: [],
            get_requests=lambda **kw: [],
            put_results=_noop,
            get_results=lambda: [],
            start=_noop,
        )
        mp.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", v1)
        mp.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_CACHE_TASK", "0")
        rm = "ResourceManagerV1" if v1 else "ResourceManager"
        mp.setattr(
            f"fastdeploy.engine.common_engine.{rm}",
            lambda *a, **kw: _ns(
                scheduler_metrics_logger=None,
                cache_manager=_ns(shm_cache_task_flag_broadcast=_Sig(), cache_ready_signal=_Sig()),
            ),
        )
        mp.setattr("fastdeploy.engine.common_engine.FMQ", lambda: _ns(queue=lambda n, r: _ns()))
        mp.setattr(
            EngineService, "start_worker_queue_service", lambda self, sq: setattr(self, "engine_worker_queue", _ns())
        )
        mp.setattr("fastdeploy.engine.common_engine.SplitwiseConnector", lambda *a, **kw: _ns(start_receiver=_noop))
        mp.setattr(
            "fastdeploy.engine.common_engine.TokenProcessor",
            lambda *a, **kw: _ns(set_resource_manager=_noop, set_scheduler_metrics_logger=_noop),
        )
        mp.setattr("fastdeploy.engine.common_engine.SchedulerMetricsLogger", lambda *a, **kw: _ns())
        mp.setattr("fastdeploy.engine.common_engine.IPCSignal", lambda **kw: _Sig(kw.get("array")))
        if dp > 1:
            cfg.parallel_config.data_parallel_size = dp
            mp.setattr("fastdeploy.engine.common_engine.get_logger", lambda *a, **kw: _ns(info=_noop))
        if guided:
            cfg.structured_outputs_config.guided_decoding_backend = "xgrammar"
            mp.setattr("fastdeploy.engine.common_engine.schema_checker", lambda *a, **kw: _ns())
        return cfg

    def test_init_v0_and_v1(self, monkeypatch):
        from fastdeploy.engine.common_engine import EngineService

        cfg = self._deps(monkeypatch, v1=False)
        e = EngineService(cfg, start_queue=False, use_async_llm=False)
        assert e.is_paused is False and e.guided_decoding_checker is None
        cfg = self._deps(monkeypatch, v1=True)
        e = EngineService(cfg, start_queue=False, use_async_llm=True)
        assert e.use_async_llm is True and e.do_profile == 1

    def test_init_options(self, monkeypatch):
        from fastdeploy.engine.common_engine import EngineService

        cfg = self._deps(monkeypatch, dp=2)
        e = EngineService(cfg, start_queue=False, use_async_llm=False)
        assert e.cfg.parallel_config.data_parallel_size == 2
        cfg = self._deps(monkeypatch, guided=True)
        e = EngineService(cfg, start_queue=False, use_async_llm=False)
        assert e.guided_decoding_checker is not None


class TestLifecycle:
    def test_start_and_register(self, monkeypatch):
        e = _eng(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        e.token_processor.tasks_queue = None
        e.token_processor.run = _noop
        e.cfg.router_config.router = None
        e._schedule_request_to_worker = _noop
        e.start()
        assert e.running and hasattr(e, "insert_task_to_worker_thread")
        # v1
        e2 = _eng(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        e2.token_processor.tasks_queue = None
        e2.token_processor.run = _noop
        e2.cfg.router_config.router = None
        e2._schedule_request_to_worker_v1 = _noop
        e2.start()
        # decode role
        e3 = _eng(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        e3.token_processor.tasks_queue = None
        e3.token_processor.run = _noop
        e3.cfg.scheduler_config.splitwise_role = "decode"
        e3.cfg.router_config.router = None
        e3._schedule_request_to_worker = _noop
        dc = []
        e3._decode_process_splitwise_requests = lambda: dc.append(1)
        e3.start()
        assert len(dc) == 1

    def test_worker_service(self, monkeypatch):
        EWQ = lambda **kw: _ns(get_server_port=lambda: 12345, cleanup=_noop)
        monkeypatch.setattr("fastdeploy.engine.common_engine.EngineWorkerQueue", EWQ)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.EngineCacheQueue", lambda **kw: _ns(get_server_port=lambda: 9999)
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", False)
        # queue service
        e = _eng(monkeypatch)
        e.start_worker_queue_service(start_queue=True)
        assert hasattr(e, "engine_worker_queue_server")
        e2 = _eng(monkeypatch)
        e2.cfg.cache_config.enable_prefix_caching = True
        e2.start_worker_queue_service(start_queue=True)
        assert hasattr(e2, "cache_task_queue")
        # cmd building
        cap = {}
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.subprocess.Popen",
            lambda cmd, **kw: (cap.update(cmd=cmd), _ns(pid=9, poll=lambda: None, stdout=iter([])))[1],
        )
        dp = _ns(
            tokenizer=_ns(vocab={"a": 0}, get_vocab=lambda: {}, encode=lambda *a, **k: [10]),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        e3 = _eng(monkeypatch)
        e3.data_processor = dp
        e3.mm_max_tokens_per_item = None
        e3.do_profile = 0
        e3.ipc_signal_suffix = 12345
        e3._start_worker_service()
        assert "--max_num_seqs" in cap["cmd"]
        # sp_model + think tokens + line_break via encode
        dp_sp = _ns(
            tokenizer=_ns(
                sp_model=[0] * 50,
                vocab={"a": 0},
                get_vocab=lambda: {"<think>": 10, "</think>": 11},
                encode=lambda *a, **k: {"input_ids": [[42]]},
            ),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        e3b = _eng(monkeypatch)
        e3b.data_processor = dp_sp
        e3b.mm_max_tokens_per_item = None
        e3b.do_profile = 0
        e3b.ipc_signal_suffix = 12345
        e3b._start_worker_service()
        assert "--think_start_id 10" in cap["cmd"]
        # line_break via .input_ids attr
        dp_lb = _ns(
            tokenizer=_ns(
                vocab={"a": 0},
                get_vocab=lambda: {},
                encode=lambda *a, **k: _ns(input_ids=[99]),
            ),
            eos_token_id_len=1,
            pad_token_id=0,
            image_patch_id=-1,
        )
        e3c = _eng(monkeypatch)
        e3c.data_processor = dp_lb
        e3c.mm_max_tokens_per_item = None
        e3c.do_profile = 0
        e3c.ipc_signal_suffix = 12345
        e3c._start_worker_service()
        assert "--line_break_id 99" in cap["cmd"]
        # nnode > 1
        e3d = _eng(monkeypatch)
        e3d.data_processor = dp
        e3d.mm_max_tokens_per_item = None
        e3d.do_profile = 0
        e3d.ipc_signal_suffix = 12345
        e3d.cfg.nnode = 2
        e3d.cfg.ips = ["10.0.0.1", "10.0.0.2"]
        e3d._start_worker_service()
        assert "--nnodes 2" in cap["cmd"]
        # full flow with profiling
        e4 = _eng(monkeypatch)
        e4.use_async_llm = True
        e4.do_profile = 1
        monkeypatch.setattr("fastdeploy.engine.common_engine.IPCSignal", lambda **kw: _Sig(kw.get("array")))
        e4.ipc_signal_suffix = 12345
        e4.data_processor = dp
        e4.mm_max_tokens_per_item = None

        def init4():
            e4.worker_ready_signal = _Sig(np.array([1], dtype=np.int32))
            e4.loaded_model_signal = _Sig(np.array([1], dtype=np.int32))
            e4.get_profile_block_num_signal = _Sig(np.array([100], dtype=np.int32))

        e4._init_worker_signals = init4
        e4.check_worker_initialize_status = lambda: True
        e4.cfg.cache_config.reset = _noop
        e4.resource_manager.reset_cache_config = _noop
        monkeypatch.setattr("fastdeploy.engine.common_engine.time.sleep", _noop)
        e4.start_worker_service(async_llm_pid=None)
        assert e4.do_profile == 0
        # worker dies
        e5 = _eng(monkeypatch)
        e5.use_async_llm = True
        e5.do_profile = 0
        e5.ipc_signal_suffix = 12345
        e5.data_processor = dp
        e5.mm_max_tokens_per_item = None
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.subprocess.Popen",
            lambda cmd, **kw: _ns(pid=9, poll=lambda: 1, stdout=iter([])),
        )

        def init5():
            e5.worker_ready_signal = _Sig(np.array([0], dtype=np.int32))
            e5.loaded_model_signal = _Sig(np.array([0], dtype=np.int32))

        e5._init_worker_signals = init5
        e5.check_worker_initialize_status = lambda: False
        assert e5.start_worker_service(async_llm_pid=None) is False

    def test_exit_sub_services(self, monkeypatch):
        e = _eng(monkeypatch)
        closed = []
        e.send_response_server = _ns(close=lambda: closed.append("s"))
        e.recv_request_server = _ns(close=lambda: closed.append("r"))
        e._exit_sub_services()
        assert not e.running and e.exist_task_signal.cleared and "s" in closed
        # async + worker + cache
        e2 = _eng(monkeypatch)
        e2.use_async_llm = True
        killed = []
        e2.worker_proc = _ns(pid=100)
        monkeypatch.setattr("os.getpgid", lambda pid: pid)
        monkeypatch.setattr("os.killpg", lambda pgid, sig: killed.append(pgid))
        e2.worker_ready_signal = _Sig()
        e2.loaded_model_signal = _Sig()
        e2.cache_manager_processes = [_ns(pid=200)]
        e2.resource_manager.cache_manager.shm_cache_task_flag_broadcast = _Sig()
        e2.resource_manager.cache_manager.cache_ready_signal = _Sig()
        e2.cache_task_queue = _ns(cleanup=_noop)
        e2.recv_control_cmd_server = _ns(close=_noop)
        e2.get_profile_block_num_signal = _Sig()
        e2._exit_sub_services()
        assert 100 in killed and 200 in killed
        # cache_task_queue with manager (no cleanup attr)
        e3 = _eng(monkeypatch)
        e3.use_async_llm = True
        e3.worker_proc = None
        e3.worker_ready_signal = _Sig()
        e3.loaded_model_signal = _Sig()
        shut = []
        e3.cache_task_queue = _ns(manager=_ns(shutdown=lambda: shut.append(1)))
        e3._exit_sub_services()
        assert len(shut) == 1
        # dp_processed join + cleanup
        e4 = _eng(monkeypatch)
        e4.use_async_llm = True
        e4.worker_proc = None
        e4.worker_ready_signal = _Sig()
        e4.loaded_model_signal = _Sig()
        joined = []
        cleaned = []
        e4.dp_processed = [_ns(pid=300, join=lambda: joined.append(1))]
        e4.dp_engine_worker_queue_server = [_ns(cleanup=lambda: cleaned.append(1))]
        e4._exit_sub_services()
        assert len(joined) == 1 and len(cleaned) == 1
        # worker kill exception (OSError)
        e5 = _eng(monkeypatch)
        e5.use_async_llm = True
        e5.worker_proc = _ns(pid=999)
        monkeypatch.setattr("os.getpgid", lambda pid: (_ for _ in ()).throw(OSError("no pid")))
        e5.worker_ready_signal = _Sig()
        e5.loaded_model_signal = _Sig()
        e5._exit_sub_services()  # should not raise
        # engine_worker_queue_server cleanup
        e6 = _eng(monkeypatch)
        qc = []
        e6.engine_worker_queue_server = _ns(cleanup=lambda: qc.append(1))
        e6._exit_sub_services()
        assert len(qc) == 1

    def test_signals_and_setup(self, monkeypatch):
        monkeypatch.setattr("fastdeploy.engine.common_engine.IPCSignal", lambda **kw: _Sig(kw.get("array")))
        monkeypatch.setattr("fastdeploy.engine.common_engine.paddle.is_compiled_with_custom_device", lambda x: False)
        # monitor signals
        e = _eng(monkeypatch)
        e._init_worker_monitor_signals()
        assert hasattr(e, "exist_task_signal")
        # worker signals: basic + profile + prefix caching
        e.ipc_signal_suffix = 12345
        e.do_profile = 0
        e._init_worker_signals()
        assert hasattr(e, "worker_ready_signal")
        e.do_profile = 1
        e._init_worker_signals()
        assert hasattr(e, "get_profile_block_num_signal")
        e2 = _eng(monkeypatch)
        e2.ipc_signal_suffix = 12345
        e2.do_profile = 0
        e2.cfg.cache_config.enable_prefix_caching = True
        e2._init_worker_signals()
        assert hasattr(e2, "launched_cache_manager_signal")
        # expert parallel
        e3 = _eng(monkeypatch)
        e3.ipc_signal_suffix = 12345
        e3.do_profile = 0
        e3.cfg.parallel_config.enable_expert_parallel = True
        e3.cfg.parallel_config.data_parallel_size = 2
        e3.cfg.nnode = 1
        e3._init_worker_signals()
        assert hasattr(e3, "launched_expert_service_signal")
        # create_data_processor
        e4 = _eng(monkeypatch)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.InputPreprocessor",
            lambda *a, **kw: _ns(create_processor=lambda: _ns(get_mm_max_tokens_per_item=lambda ml: {"image": 128})),
        )
        e4.cfg.get_max_chunk_tokens = lambda mm: 256
        e4.cfg.cache_config.postprocess = _noop
        e4.create_data_processor()
        assert e4.mm_max_tokens_per_item == {"image": 128}
        # launch_components
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_MULTI_API_SERVER", False)
        e5 = _eng(monkeypatch)
        e5.cfg.scheduler_config.splitwise_role = "prefill"
        e5.cfg.scheduler_config.name = "local"
        e5.launch_components()
        assert hasattr(e5, "splitwise_receive_thread")
        # launch_components: splitwise scheduler
        e5b = _eng(monkeypatch)
        e5b.cfg.scheduler_config.splitwise_role = "prefill"
        e5b.cfg.scheduler_config.name = "splitwise"
        e5b.scheduler.start = lambda role, host, info: None
        e5b.launch_components()
        # launch_components: dp scheduler
        e5c = _eng(monkeypatch)
        e5c.cfg.scheduler_config.splitwise_role = "mixed"
        e5c.cfg.scheduler_config.name = "dp"
        e5c.scheduler.start = lambda rank: None
        e5c.launch_components()
        # stop_profile
        e6 = _eng(monkeypatch)
        e6.do_profile = 1
        e6.get_profile_block_num_signal = _Sig(np.array([100], dtype=np.int32))
        e6.worker_proc = None
        e6.cfg.cache_config.reset = _noop
        e6.resource_manager.reset_cache_config = _noop
        e6.ipc_signal_suffix = 12345
        e6._stop_profile()
        assert e6.do_profile == 0
        # stop_profile with prefix_caching → starts cache service
        e6b = _eng(monkeypatch)
        e6b.do_profile = 1
        e6b.get_profile_block_num_signal = _Sig(np.array([100], dtype=np.int32))
        e6b.worker_proc = None
        e6b.cfg.cache_config.reset = _noop
        e6b.cfg.cache_config.enable_prefix_caching = True
        e6b.resource_manager.reset_cache_config = _noop
        e6b.ipc_signal_suffix = 12345
        started_cache = []
        e6b.start_cache_service = lambda d, s: (started_cache.append(1), [])[1]
        e6b._stop_profile()
        assert len(started_cache) == 1
        # stop_profile: worker_proc dies during profiling
        e6c = _eng(monkeypatch)
        e6c.do_profile = 1
        e6c.get_profile_block_num_signal = _Sig(np.array([0], dtype=np.int32))
        e6c.worker_proc = _ns(poll=lambda: 1)
        with pytest.raises(RuntimeError, match="Worker process failed"):
            e6c._stop_profile()
        # check_worker_initialize_status: success
        e7 = _eng(monkeypatch)
        e7.cfg.worker_num_per_node = 1
        e7.worker_ready_signal = _Sig(np.array([0], dtype=np.int32))
        e7.worker_init_status = {}
        e7.worker_proc = _ns(stdout=iter([b"Loading checkpoint shards: 100\n"]), poll=lambda: None)

        def ready7():
            time.sleep(0.05)
            e7.worker_ready_signal.value[0] = 1

        threading.Thread(target=ready7, daemon=True).start()
        assert e7.check_worker_initialize_status()
        # check_worker_initialize_status: layer_loading (covers L2219, 2237)
        e7b = _eng(monkeypatch)
        e7b.cfg.worker_num_per_node = 1
        e7b.cfg.model_config.num_hidden_layers = 4
        e7b.worker_ready_signal = _Sig(np.array([0], dtype=np.int32))
        e7b.worker_init_status = {}
        e7b.worker_proc = _ns(
            stdout=iter([b"Start load layer 0\n", b"Start load layer 1\n", b"set state for layer 2\n"]),
            poll=lambda: None,
        )

        def ready7b():
            time.sleep(0.1)
            e7b.worker_ready_signal.value[0] = 1

        threading.Thread(target=ready7b, daemon=True).start()
        assert e7b.check_worker_initialize_status()
        # check_worker_initialize_status: proc dies
        e8 = _eng(monkeypatch)
        e8.cfg.worker_num_per_node = 1
        e8.worker_ready_signal = _Sig(np.array([0], dtype=np.int32))
        e8.worker_init_status = {}
        e8.worker_proc = _ns(stdout=iter([]), poll=lambda: 1)
        assert not e8.check_worker_initialize_status()
        # custom device path in _init_worker_signals
        monkeypatch.setattr("fastdeploy.engine.common_engine.paddle.is_compiled_with_custom_device", lambda x: True)
        e9 = _eng(monkeypatch)
        e9.ipc_signal_suffix = 12345
        e9.do_profile = 0
        e9._init_worker_signals()
        assert hasattr(e9, "worker_ready_signal")

    def test_stop_profile(self, monkeypatch):
        monkeypatch.setattr("fastdeploy.engine.common_engine.time.sleep", _noop)
        # normal: worker alive, signal set after 2 iterations
        e = _eng(monkeypatch)
        prof_sig = _Sig(np.array([0], dtype=np.int32))
        sp_cc = [0]

        def _sp_poll():
            sp_cc[0] += 1
            if sp_cc[0] >= 2:
                prof_sig.value[0] = 42
            return None

        e.get_profile_block_num_signal = prof_sig
        e.worker_proc = _ns(poll=_sp_poll)
        e.cfg.cache_config.reset = _noop
        e.cfg.cache_config.enable_prefix_caching = True
        e.resource_manager.reset_cache_config = _noop
        e.ipc_signal_suffix = "test"
        e.start_cache_service = lambda *a, **kw: []
        e._stop_profile()
        assert e.do_profile == 0
        # worker died: poll returns non-None → RuntimeError
        e2 = _eng(monkeypatch)
        e2.get_profile_block_num_signal = _Sig(np.array([0], dtype=np.int32))
        e2.worker_proc = _ns(poll=lambda: 1)
        with pytest.raises(RuntimeError, match="Worker process failed"):
            e2._stop_profile()


class TestQueryControlMisc:
    def test_health_and_queries(self, monkeypatch):
        e = _eng(monkeypatch)
        e.resource_manager.stop_flags = np.array([True, False], dtype=bool)
        assert e.task_is_finished(0) and not e.task_is_finished(1)
        assert not e.all_tasks_finished()
        e.resource_manager.stop_flags[:] = True
        assert e.all_tasks_finished()
        # unhandled request num
        e.scheduler.get_unhandled_request_num = lambda: 5
        assert e._get_scheduler_unhandled_request_num() == 5
        e.scheduler.get_unhandled_request_num = "nope"
        assert e._get_scheduler_unhandled_request_num() == 0
        e.scheduler.get_unhandled_request_num = lambda: -3
        assert e._get_scheduler_unhandled_request_num() == 0
        # check_health
        e.worker_healthy_live_signal.value[0] = 0
        ok, _ = e.check_health()
        assert ok
        e.worker_healthy_live_signal.value = np.array([time.time() - 60], dtype=np.float64)
        ok, msg = e.check_health(time_interval_threashold=30)
        assert not ok and "Not Healthy" in msg
        # worker_processes_ready
        e.cfg.worker_num_per_node = 2
        e.worker_ready_signal.value = np.array([1, 0], dtype=np.int32)
        assert not e._worker_processes_ready()
        e.worker_ready_signal.value = np.array([1, 1], dtype=np.int32)
        assert e._worker_processes_ready()

    def test_control_api(self, monkeypatch):
        e = _eng(monkeypatch)
        e.is_paused = True
        assert e._control_is_paused(_ns(request_id="r")) == {"is_paused": True}
        e._control_resume(_ns(request_id="r"))
        assert not e.is_paused
        # update_weights requires pause
        with pytest.raises(Exception, match="Pause"):
            e._control_update_weights(_ns(request_id="r"))
        e.is_paused = True
        called = []
        e._call_worker = lambda cr, t: called.append(cr.request_id)
        e._control_update_weights(_ns(request_id="r"))
        assert called == ["r"]
        # run_control_method unknown
        sent = []
        e.send_response_server = _ns(send_response=lambda rid, r: sent.append(rid))
        e.run_control_method(_ns(request_id="r1", method="x", params={}, get_method=lambda: "x"))
        assert sent[-1] == "r1"
        # v1 pause
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        e.cfg.scheduler_config.name = "local"
        e.is_paused = False
        e.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=_noop, clear_data=_noop)
        e.resource_manager.log_status = _noop
        e.resource_manager.preempted_all = lambda: []
        e.resource_manager.cache_manager = _ns(reset=_noop)
        e.scheduler.get_inflight_requests = lambda: []
        e._control_pause(_ns(request_id="p"))
        assert e.is_paused
        # already-paused logs and returns
        e._control_pause(_ns(request_id="p1b"))
        assert e.is_paused
        # inflight requests aborted during pause
        e.is_paused = False
        inflight = [_ns(request_id="inf1")]
        e.scheduler.get_inflight_requests = lambda: inflight
        e._control_pause(_ns(request_id="p1c"))
        assert e.is_paused
        # non-local scheduler raises
        e.is_paused = False
        e.cfg.scheduler_config.name = "dp"
        with pytest.raises(Exception, match="local scheduler"):
            e._control_pause(_ns(request_id="p1d"))
        # non-v1 pause raises
        e.is_paused = False
        e.cfg.scheduler_config.name = "local"
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        with pytest.raises(Exception, match="pause only supported"):
            e._control_pause(_ns(request_id="p2"))
        # run_control_method exception path — handler raises
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        e.is_paused = True
        e._call_worker = lambda cr, t: (_ for _ in ()).throw(RuntimeError("fail"))
        e.run_control_method(
            _ns(request_id="ex", method="update_weights", params={}, get_method=lambda: "update_weights")
        )
        assert sent[-1] == "ex"
        # pause with running requests
        e.is_paused = False
        e.cfg.scheduler_config.name = "local"
        et_cc = [0]
        e.engine_worker_queue = _ns(
            exist_tasks=lambda: (et_cc.__setitem__(0, et_cc[0] + 1) or et_cc[0] <= 1), put_tasks=_noop
        )
        e.resource_manager.preempted_all = lambda: [_ns(request_id="rr1")]
        e.resource_manager.get_real_bsz = _noop
        e.resource_manager.real_bsz = 1
        e.resource_manager.wait_worker_inflight_requests_finish = _noop
        e.resource_manager.cache_manager = _ns(reset=_noop)
        e.scheduler.get_inflight_requests = lambda: []
        e._control_pause(_ns(request_id="p_run"))
        assert e.is_paused
        # run_control_method success path (known method completes normally)
        e.run_control_method(_ns(request_id="sp", method="is_paused", params={}, get_method=lambda: "is_paused"))
        assert sent[-1] == "sp"
        # pause timeout — exist_tasks always True (L1315-1319)
        monkeypatch.setattr("fastdeploy.engine.common_engine.time.sleep", _noop)
        e.is_paused = False
        e.resource_manager.log_status = _noop
        e.engine_worker_queue = _ns(exist_tasks=lambda: True)
        with pytest.raises(Exception, match="timeout"):
            e._control_pause(_ns(request_id="p_to"))

    def test_misc_utils(self, monkeypatch):
        e = _eng(monkeypatch)
        # decode_token: text disabled
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", False)
        d, t = e._decode_token([1, 2], "r", is_end=False)
        assert d == "" and t == [1, 2]
        # text enabled
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        e.data_processor = _ns(ids2tokens=lambda t, r: ("hi", [1, 2, 3], None), decode_status={"r1": [0, 2]})
        d, _ = e._decode_token([1, 2, 3], "r1", is_end=False)
        assert d == "hi"
        e.data_processor.decode_status["r2"] = [0, 1]
        e.data_processor.ids2tokens = lambda t, r: ("end", [10], None)
        e._decode_token([10], "r2", is_end=True)
        assert "r2" not in e.data_processor.decode_status
        # clear_data
        e.send_response_server = _ns(req_dict={})
        e.recv_request_server = _ns(req_dict={})
        assert e.clear_data()
        e.token_processor.clear_data = lambda: (_ for _ in ()).throw(RuntimeError("x"))
        assert not e.clear_data()
        # _setting_environ_variables
        e2 = _eng(monkeypatch)
        r = e2._setting_environ_variables()
        assert "FLAGS_use_append_attn=1" in r
        e2.cfg.scheduler_config.splitwise_role = "prefill"
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        assert "FLAGS_use_pd_disaggregation=1" in e2._setting_environ_variables()
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        assert "FLAGS_use_pd_disaggregation_per_chunk=1" in e2._setting_environ_variables()
        # _send_error_response
        e3 = _eng(monkeypatch)
        sent = []
        e3.send_response_server = _ns(send_response=lambda rid, r: sent.append(rid))
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        e3._send_error_response("r1", "err", 503)
        assert sent[-1] == "r1"
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        e3._send_error_response("r2", "err")
        assert sent[-1] is None
        # _get_scheduler_unhandled_request_num
        e4 = _eng(monkeypatch)
        assert e4._get_scheduler_unhandled_request_num() == 0  # not callable
        e4.scheduler.get_unhandled_request_num = lambda: 5
        assert e4._get_scheduler_unhandled_request_num() == 5
        e4.scheduler.get_unhandled_request_num = lambda: -3
        assert e4._get_scheduler_unhandled_request_num() == 0  # clamped
        e4.scheduler.get_unhandled_request_num = lambda: (_ for _ in ()).throw(RuntimeError)
        assert e4._get_scheduler_unhandled_request_num() == 0  # exception path
        # resume when not paused
        e5 = _eng(monkeypatch)
        e5.is_paused = False
        assert e5._control_resume(_ns(request_id="r")) is None
        assert not e5.is_paused

    def test_call_worker(self, monkeypatch):
        """Cover _call_worker + _wait_all_control_responses (L1398-1434)."""
        import asyncio

        e = _eng(monkeypatch)
        # set up mock ctrl queues that return a ControlResponse-like object
        resp = _ns(request_id="cw1", error_code=200, error_message="", result={"ok": True})
        payload_msg = _ns(payload=resp)

        async def _get(timeout=0):
            return payload_msg

        q = _ns(get=_get, name="q0")
        e._ctrl_worker_output_queues = [q]
        e.engine_worker_queue = _ns(put_tasks=_noop)
        cr = _ns(request_id="cw1")
        result = e._call_worker(cr, timeout=10)
        assert result == [{"ok": True}]

        # timeout path
        async def _get_slow(timeout=0):
            await asyncio.sleep(999)

        q2 = _ns(get=_get_slow, name="q1")
        e2 = _eng(monkeypatch)
        e2._ctrl_worker_output_queues = [q2]
        e2.engine_worker_queue = _ns(put_tasks=_noop)
        with pytest.raises(Exception, match="Timeouted"):
            e2._call_worker(_ns(request_id="cw2"), timeout=0.01)

        # error_code != 200
        resp_err = _ns(request_id="cw3", error_code=500, error_message="bad", result=None)
        payload_err = _ns(payload=resp_err)

        async def _get_err(timeout=0):
            return payload_err

        q3 = _ns(get=_get_err, name="q2")
        e3 = _eng(monkeypatch)
        e3._ctrl_worker_output_queues = [q3]
        e3.engine_worker_queue = _ns(put_tasks=_noop)
        with pytest.raises(Exception, match="Call Worker error"):
            e3._call_worker(_ns(request_id="cw3"), timeout=10)

        # None message
        async def _get_none(timeout=0):
            return None

        q4 = _ns(get=_get_none, name="q3")
        e4 = _eng(monkeypatch)
        e4._ctrl_worker_output_queues = [q4]
        e4.engine_worker_queue = _ns(put_tasks=_noop)
        with pytest.raises(Exception, match="Timeouted"):
            e4._call_worker(_ns(request_id="cw4"), timeout=10)

        # Exception from get
        async def _get_exc(timeout=0):
            raise ConnectionError("broken")

        q5 = _ns(get=_get_exc, name="q4")
        e5 = _eng(monkeypatch)
        e5._ctrl_worker_output_queues = [q5]
        e5.engine_worker_queue = _ns(put_tasks=_noop)
        with pytest.raises(Exception, match="Call Worker error"):
            e5._call_worker(_ns(request_id="cw5"), timeout=10)

        # mismatched request_id (skipped) + matching
        resp_skip = _ns(request_id="old", error_code=200, error_message="", result="skip")
        resp_match = _ns(request_id="cw6", error_code=200, error_message="", result="match")
        call_count = [0]

        async def _get_multi(timeout=0):
            call_count[0] += 1
            if call_count[0] == 1:
                return _ns(payload=resp_skip)
            return _ns(payload=resp_match)

        q6a = _ns(get=_get_multi, name="q5a")

        async def _get_match(timeout=0):
            return _ns(payload=resp_match)

        q6b = _ns(get=_get_match, name="q5b")
        e6 = _eng(monkeypatch)
        e6._ctrl_worker_output_queues = [q6a, q6b]
        e6.engine_worker_queue = _ns(put_tasks=_noop)
        # q6a returns mismatched "old" (skipped), q6b returns matching "cw6"
        result6 = e6._call_worker(_ns(request_id="cw6"), timeout=10)
        assert "match" in result6

    def test_chunk_size(self, monkeypatch):
        e = _eng(monkeypatch)
        e.cfg.cache_config.enable_chunked_prefill = False
        reqs = [_ns(prompt_token_ids_len=100)]
        e.update_requests_chunk_size(reqs)
        assert not hasattr(reqs[0], "prefill_chunk_info")
        e.cfg.cache_config.enable_chunked_prefill = True
        e.cfg.cache_config.block_size = 16
        e.cfg.scheduler_config.max_num_batched_tokens = 128
        e.partial_chunked_tokens = [0, 128]
        ci = {}
        req = _ns(prompt_token_ids_len=64, set=lambda k, v: ci.update({k: v}))
        e.update_requests_chunk_size([req])
        assert sum(ci["prefill_chunk_info"]) == 64
        # multiple with remainder
        e.cfg.scheduler_config.max_num_batched_tokens = 256
        e.cfg.max_num_partial_prefills = 2
        e.partial_chunked_tokens = [0, 256, 32]
        cs = [{}, {}]
        rs = [_ns(prompt_token_ids_len=100, set=lambda k, v, i=i: cs[i].update({k: v})) for i in range(2)]
        e.update_requests_chunk_size(rs)
        for c in cs:
            assert sum(c["prefill_chunk_info"]) == 100


class TestInsertTasks:
    def test_insert_basic(self, monkeypatch):
        _ptr(monkeypatch)
        e = _eng(monkeypatch)
        e.cfg.scheduler_config.splitwise_role = "mixed"
        e.cfg.cache_config.enable_chunked_prefill = False
        e.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        e.resource_manager.real_bsz = 1
        put = []
        e.engine_worker_queue = _ns(put_tasks=lambda t: put.append(t))
        assert e.insert_tasks(_task())  # non-list wraps
        assert e.insert_tasks([_task()])  # list happy
        # exceeds batch
        e.resource_manager.stop_flags = np.array([True, False, False], dtype=bool)
        e.insert_tasks([_task(f"r{i}") for i in range(3)])
        assert len(put[-1][0]) == 1
        # allocation fails
        e2 = _eng(monkeypatch)
        e2.resource_manager.allocate_resources_for_new_tasks = lambda t: []
        from fastdeploy.engine.common_engine import EngineError

        with pytest.raises(EngineError):
            e2.insert_tasks([_task()])

    def test_insert_splitwise(self, monkeypatch):
        _ptr(monkeypatch)
        # prefill role
        e = _eng(monkeypatch)
        e.cfg.scheduler_config.splitwise_role = "prefill"
        e.cfg.cache_config.enable_chunked_prefill = False
        e.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        e.resource_manager.real_bsz = 1
        e.split_connector = _ns(check_decode_allocated=lambda t: (True, None), send_cache_info_to_messager=_noop)
        e.engine_worker_queue = _ns(put_tasks=_noop)
        assert e.insert_tasks([_task(disagg=_ns(foo=1))])
        # prefill fail path
        from fastdeploy.engine.common_engine import EngineError as EE

        e1b = _eng(monkeypatch)
        e1b.cfg.scheduler_config.splitwise_role = "prefill"
        e1b.cfg.cache_config.enable_chunked_prefill = False
        e1b.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        e1b.resource_manager.real_bsz = 1
        e1b.split_connector = _ns(
            check_decode_allocated=lambda t: (False, "no blocks"), send_cache_info_to_messager=_noop
        )
        e1b.engine_worker_queue = _ns(put_tasks=_noop)
        e1b.scheduler.put_results = _noop
        with pytest.raises(EE):
            e1b.insert_tasks([_task(disagg=_ns(foo=1))])
        # decode role
        e2 = _eng(monkeypatch)
        e2.cfg.scheduler_config.splitwise_role = "decode"
        e2.cfg.cache_config.enable_chunked_prefill = False
        e2.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        e2.resource_manager.real_bsz = 1
        e2.split_connector = _ns(send_cache_info_to_prefill=_noop)
        e2.engine_worker_queue = _ns(put_tasks=_noop)
        assert e2.insert_tasks([_task(disagg=_ns(x=1))])
        # preempted + carrier
        e3 = _eng(monkeypatch)
        e3.cfg.scheduler_config.splitwise_role = "mixed"
        e3.cfg.cache_config.enable_chunked_prefill = False
        e3.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        e3.resource_manager.real_bsz = 1
        e3.engine_worker_queue = _ns(put_tasks=_noop)
        assert e3.insert_tasks([_task(preempted=True, carrier={"traceparent": "00-abc"})])

    def test_prefilled_requests(self, monkeypatch):
        def _setup(e):
            e.cfg.speculative_config = _ns(method="none")
            e.cfg.scheduler_config.splitwise_role = "decode"
            e.resource_manager.real_bsz = 1
            e.resource_manager._recycle_block_tables = _noop

        met = _ns(decode_recv_req_time=0, decode_preallocate_req_time=0, decode_inference_start_time=0)
        # happy path
        e = _eng(monkeypatch)
        _setup(e)
        e.resource_manager.req_dict = {"r1": 0}
        e.resource_manager.tasks_list = [_ns(prompt_token_ids=[0], num_cached_tokens=0, metrics=met)]
        e.resource_manager.stop_flags = np.array([False], dtype=bool)
        e.token_processor = _ns(tokens_counter={}, clear_data=_noop, number_of_tasks=0, number_of_input_tokens=0)
        put = []
        e.engine_worker_queue = _ns(put_tasks=lambda t: put.append(1))
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        ro = _ns(
            request_id="r1",
            outputs=_ns(token_ids=[42], draft_token_ids=None),
            error_code=200,
            error_msg=None,
            num_cached_tokens=5,
            metrics=met,
        )
        assert e._insert_prefilled_requests([ro])
        assert len(put) == 1
        # error path
        e2 = _eng(monkeypatch)
        _setup(e2)
        e2.resource_manager.req_dict = {"re": 0}
        e2.resource_manager.tasks_list = [_ns(prompt_token_ids=[0], num_cached_tokens=0, metrics=met)]
        e2.resource_manager.stop_flags = np.array([False], dtype=bool)
        e2.token_processor = _ns(tokens_counter={"re": 1}, clear_data=_noop, number_of_tasks=0)
        e2.scheduler.put_results = _noop
        e2.engine_worker_queue = _ns(put_tasks=_noop)
        ro_e = _ns(
            request_id="re",
            outputs=_ns(token_ids=[1], draft_token_ids=None),
            error_code=500,
            error_msg="fail",
            num_cached_tokens=0,
            metrics=met,
        )
        e2._insert_prefilled_requests([ro_e])
        # adapter first-token-is-EOS
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        e3 = _eng(monkeypatch)
        _setup(e3)
        e3.resource_manager.req_dict = {"re2": 0}
        e3.resource_manager.tasks_list = [_ns(prompt_token_ids=[0], num_cached_tokens=0, metrics=met)]
        e3.resource_manager.stop_flags = np.array([False], dtype=bool)
        e3.resource_manager._recycle_block_tables = _noop
        e3.token_processor = _ns(
            tokens_counter={"re2": 1}, clear_data=_noop, number_of_tasks=0, number_of_input_tokens=0
        )
        e3.engine_worker_queue = _ns(put_tasks=_noop)
        ro_eos = _ns(
            request_id="re2",
            outputs=_ns(token_ids=[], draft_token_ids=None),
            error_code=200,
            error_msg=None,
            num_cached_tokens=0,
            metrics=met,
        )
        e3._insert_prefilled_requests([ro_eos])
        assert e3.resource_manager.stop_flags[0]
        assert e2.resource_manager.stop_flags[0]


class TestSchedule:
    def test_v0_scheduling(self, monkeypatch):
        _ptr(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.main_process_metrics.num_requests_waiting.dec", _noop)
        monkeypatch.setattr("fastdeploy.engine.common_engine.main_process_metrics.num_requests_running.inc", _noop)
        e = _eng(monkeypatch)
        e.cfg.scheduler_config.splitwise_role = "mixed"
        e.resource_manager.available_batch = lambda: 1
        e.resource_manager.available_block_num = lambda: 100
        e.resource_manager.abort_req_ids_set = set()
        e.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        e.resource_manager.real_bsz = 1
        e.split_connector = _ns(current_request_ids=[], has_splitwise_tasks=lambda: False)
        e.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=_noop, num_cache_infos=lambda: 0)
        cc = [0]

        def get_reqs(**kw):
            cc[0] += 1
            if cc[0] == 1:
                return [_task()]
            e.running = False
            return []

        e.scheduler.get_requests = get_reqs
        e._schedule_request_to_worker()
        assert cc[0] >= 1
        # no-batch skips
        e2 = _eng(monkeypatch)
        sc = [0]
        e2.resource_manager.available_batch = lambda: (
            (sc.__setitem__(0, sc[0] + 1) or 0) if sc[0] < 3 else (setattr(e2, "running", False) or 0)
        )
        e2._schedule_request_to_worker()
        # exist_prefill_signal
        e3 = _eng(monkeypatch)
        e3.cfg.scheduler_config.splitwise_role = "mixed"
        e3.resource_manager.available_batch = lambda: 1
        e3.engine_worker_queue = _ns(exist_tasks=lambda: False, num_cache_infos=lambda: 0)
        e3.split_connector = _ns(current_request_ids=[], has_splitwise_tasks=lambda: True)
        e3.exist_prefill_task_signal = _Sig(np.array([1], dtype=np.int32))
        pc = [0]

        def avail3():
            pc[0] += 1
            if pc[0] >= 3:
                e3.running = False
            return 1

        e3.resource_manager.available_batch = avail3
        e3._schedule_request_to_worker()
        # exist_tasks true → sleep+continue, then proceed
        e4 = _eng(monkeypatch)
        e4.cfg.scheduler_config.splitwise_role = "mixed"
        et4 = [0]
        e4.resource_manager.available_batch = lambda: 1
        e4.resource_manager.available_block_num = lambda: 100
        e4.resource_manager.abort_req_ids_set = set()
        e4.engine_worker_queue = _ns(
            exist_tasks=lambda: (et4.__setitem__(0, et4[0] + 1) or et4[0] <= 1),
            put_tasks=_noop,
            num_cache_infos=lambda: 0,
        )
        e4.split_connector = _ns(current_request_ids=[], has_splitwise_tasks=lambda: False)
        vc4 = [0]

        def gr4(**kw):
            vc4[0] += 1
            if vc4[0] >= 1:
                e4.running = False
            return []

        e4.scheduler.get_requests = gr4
        e4._schedule_request_to_worker()
        # decode role → skip
        e5 = _eng(monkeypatch)
        e5.cfg.scheduler_config.splitwise_role = "decode"
        e5.resource_manager.available_batch = lambda: 1
        e5.resource_manager.available_block_num = lambda: 100
        e5.resource_manager.abort_req_ids_set = set()
        e5.engine_worker_queue = _ns(exist_tasks=lambda: False, num_cache_infos=lambda: 0)
        e5.split_connector = _ns(current_request_ids=[], has_splitwise_tasks=lambda: False)
        dc5 = [0]

        def gr5(**kw):
            dc5[0] += 1
            if dc5[0] >= 2:
                e5.running = False
            return [_task()]

        e5.scheduler.get_requests = gr5
        e5._schedule_request_to_worker()
        # num_cache_infos > 0
        e6 = _eng(monkeypatch)
        e6.cfg.scheduler_config.splitwise_role = "mixed"
        nc6 = [0]
        e6.resource_manager.available_batch = lambda: 1
        e6.engine_worker_queue = _ns(
            exist_tasks=lambda: False,
            num_cache_infos=lambda: (1 if nc6[0] == 0 else 0) or (nc6.__setitem__(0, nc6[0] + 1) or 0),
        )
        e6.split_connector = _ns(current_request_ids=[], has_splitwise_tasks=lambda: False)
        nc_sc6 = [0]

        def avail6():
            nc_sc6[0] += 1
            if nc_sc6[0] >= 3:
                e6.running = False
            return 1

        e6.resource_manager.available_batch = avail6
        e6._schedule_request_to_worker()
        # current_request_ids non-empty
        e7 = _eng(monkeypatch)
        e7.cfg.scheduler_config.splitwise_role = "mixed"
        e7.resource_manager.available_batch = lambda: 1
        e7.engine_worker_queue = _ns(exist_tasks=lambda: False, num_cache_infos=lambda: 0)
        cr_ids = [["rid1"]]
        e7.split_connector = _ns(current_request_ids=cr_ids[0], has_splitwise_tasks=lambda: False)
        rc7 = [0]

        def avail7():
            rc7[0] += 1
            if rc7[0] >= 2:
                cr_ids[0].clear()
            if rc7[0] >= 4:
                e7.running = False
            return 1

        e7.resource_manager.available_batch = avail7
        e7._schedule_request_to_worker()
        # non-mixed splitwise → send tasks (L787-789)
        e8 = _eng(monkeypatch)
        e8.cfg.scheduler_config.splitwise_role = "prefill"
        e8.resource_manager.available_batch = lambda: 1
        e8.resource_manager.available_block_num = lambda: 100
        e8.resource_manager.abort_req_ids_set = set()
        e8.resource_manager.allocate_resources_for_new_tasks = lambda t: t
        e8.resource_manager.real_bsz = 1
        e8.split_connector = _ns(current_request_ids=[], has_splitwise_tasks=lambda: False, send_splitwise_tasks=_noop)
        e8.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=_noop, num_cache_infos=lambda: 0)
        cc8 = [0]

        def gr8(**kw):
            cc8[0] += 1
            if cc8[0] == 1:
                return [_task()]
            e8.running = False
            return []

        e8.scheduler.get_requests = gr8
        e8._schedule_request_to_worker()
        # insert_tasks fail → continue (L795)
        e9 = _eng(monkeypatch)
        e9.cfg.scheduler_config.splitwise_role = "mixed"
        e9.resource_manager.available_batch = lambda: 1
        e9.resource_manager.available_block_num = lambda: 100
        e9.resource_manager.abort_req_ids_set = set()
        e9.split_connector = _ns(current_request_ids=[], has_splitwise_tasks=lambda: False)
        e9.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=_noop, num_cache_infos=lambda: 0)
        cc9 = [0]

        def gr9(**kw):
            cc9[0] += 1
            if cc9[0] >= 2:
                e9.running = False
            return [_task()]

        e9.scheduler.get_requests = gr9
        e9.insert_tasks = lambda *a, **kw: False
        e9._schedule_request_to_worker()

    def test_v1_scheduling(self, monkeypatch):
        _ptr(monkeypatch)
        # mixed happy with scheduler_unhandled_request_num
        e = _eng(monkeypatch)
        e.cfg.scheduler_config.splitwise_role = "mixed"
        e.resource_manager.waiting = []
        e.resource_manager.get_real_bsz = _noop
        e.resource_manager.real_bsz = 1
        e.resource_manager.scheduler_unhandled_request_num = 0
        e.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=_noop)
        e.scheduler.get_requests = lambda **kw: [_task()]
        cc = [0]

        def sched():
            cc[0] += 1
            if cc[0] > 1:
                e.running = False
                return [], []
            return [_task()], []

        e.resource_manager.schedule = sched
        e._schedule_request_to_worker_v1()
        # prefill with tasks
        e2 = _eng(monkeypatch)
        e2.cfg.scheduler_config.splitwise_role = "prefill"
        e2.resource_manager.waiting = []
        e2.resource_manager.get_real_bsz = _noop
        e2.resource_manager.real_bsz = 1
        e2.split_connector = _ns(check_decode_allocated=lambda t: (True, None), send_cache_info_to_messager=_noop)
        e2.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=_noop)
        cc2 = [0]

        def sched2():
            cc2[0] += 1
            if cc2[0] == 1:
                return [_task()], []
            e2.running = False
            return [], []

        e2.resource_manager.schedule = sched2
        e2._schedule_request_to_worker_v1()
        # decode preempted
        from fastdeploy.engine.request import Request, RequestType

        e3 = _eng(monkeypatch)
        e3.cfg.scheduler_config.splitwise_role = "decode"
        e3.resource_manager.waiting = []
        e3.resource_manager.get_real_bsz = _noop
        e3.resource_manager.real_bsz = 1
        e3.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=_noop)
        pr = Request.__new__(Request)
        pr.request_id = "rp"
        pr.task_type = RequestType.PREEMPTED
        pr.has_been_preempted_before = True
        pr.trace_carrier = None
        pr.user = "test"
        pr.metrics = _ns(
            scheduler_recv_req_time=time.time(),
            inference_start_time=0,
            decode_inference_start_time=0,
            add_req_to_resource_manager_time=0,
        )
        pr_calls = []
        e3.scheduler.put_results = lambda r: pr_calls.append(r)
        cc3 = [0]

        def sched3():
            cc3[0] += 1
            if cc3[0] == 1:
                return [pr], []
            e3.running = False
            return [], []

        e3.resource_manager.schedule = sched3
        e3._schedule_request_to_worker_v1()
        assert len(pr_calls) >= 1
        # PREFILL task trace spans
        e4 = _eng(monkeypatch)
        e4.cfg.scheduler_config.splitwise_role = "mixed"
        e4.resource_manager.waiting = []
        e4.resource_manager.get_real_bsz = _noop
        e4.resource_manager.real_bsz = 1
        e4.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=_noop)
        pf = Request.__new__(Request)
        pf.request_id = "pf_1"
        pf.task_type = RequestType.PREFILL
        pf.has_been_preempted_before = False
        pf.trace_carrier = {"traceparent": "00-abc"}
        pf.user = "test"
        pf.metrics = _ns(scheduler_recv_req_time=time.time(), inference_start_time=0)
        cc4 = [0]

        def sched4():
            cc4[0] += 1
            if cc4[0] == 1:
                return [pf], []
            e4.running = False
            return [], []

        e4.resource_manager.schedule = sched4
        e4._schedule_request_to_worker_v1()
        # PREFILL preempted (rescheduled)
        e5 = _eng(monkeypatch)
        e5.cfg.scheduler_config.splitwise_role = "mixed"
        e5.resource_manager.waiting = []
        e5.resource_manager.get_real_bsz = _noop
        e5.resource_manager.real_bsz = 1
        e5.engine_worker_queue = _ns(exist_tasks=lambda: False, put_tasks=_noop)
        pf2 = Request.__new__(Request)
        pf2.request_id = "pf_2"
        pf2.task_type = RequestType.PREFILL
        pf2.has_been_preempted_before = True
        pf2.trace_carrier = None
        pf2.user = "test"
        pf2.metrics = _ns(scheduler_recv_req_time=time.time(), inference_start_time=0)
        cc5 = [0]

        def sched5():
            cc5[0] += 1
            if cc5[0] == 1:
                return [pf2], []
            e5.running = False
            return [], []

        e5.resource_manager.schedule = sched5
        e5._schedule_request_to_worker_v1()
        # v1 exist_tasks true → sleep+continue (L984-985)
        e6 = _eng(monkeypatch)
        e6.cfg.scheduler_config.splitwise_role = "mixed"
        e6.resource_manager.waiting = []
        e6.resource_manager.get_real_bsz = _noop
        e6.resource_manager.real_bsz = 1
        et6 = [0]
        e6.engine_worker_queue = _ns(exist_tasks=lambda: (et6.__setitem__(0, et6[0] + 1) or et6[0] <= 1))
        cc6 = [0]

        def sched6():
            cc6[0] += 1
            e6.running = False
            return [], []

        e6.resource_manager.schedule = sched6
        e6._schedule_request_to_worker_v1()

    def test_v1_errors(self, monkeypatch):
        _ptr(monkeypatch)
        # shutdown RuntimeError from schedule()
        e = _eng(monkeypatch)
        e.resource_manager.waiting = []
        cc = [0]

        def sched():
            cc[0] += 1
            if cc[0] > 1:
                raise RuntimeError("cannot schedule new futures after shutdown")
            return [], []

        e.resource_manager.schedule = sched
        e.engine_worker_queue = _ns(exist_tasks=lambda: False)
        e._schedule_request_to_worker_v1()
        # error_tasks
        e2 = _eng(monkeypatch)
        e2.cfg.scheduler_config.splitwise_role = "mixed"
        e2.resource_manager.waiting = []
        sent = []
        e2.send_response_server = _ns(send_response=lambda rid, r: sent.append(rid))
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        e2.engine_worker_queue = _ns(exist_tasks=lambda: False)
        cc2 = [0]

        def sched2():
            cc2[0] += 1
            if cc2[0] == 1:
                return [], [("skip", None), ("real", "actual error")]
            e2.running = False
            return [], []

        e2.resource_manager.schedule = sched2
        e2._schedule_request_to_worker_v1()
        assert "real" in sent and "skip" not in sent

        # RuntimeError from ThreadPoolExecutor.submit (L997-1002) — MUST be last
        class _FailPool:
            def submit(self, fn):
                raise RuntimeError("cannot schedule new futures after shutdown")

            def shutdown(self, wait=False):
                pass

        monkeypatch.setattr("fastdeploy.engine.common_engine.ThreadPoolExecutor", lambda **kw: _FailPool())
        e_tp = _eng(monkeypatch)
        e_tp.cfg.scheduler_config.splitwise_role = "mixed"
        e_tp.resource_manager.waiting = []
        e_tp.resource_manager.get_real_bsz = _noop
        e_tp.resource_manager.real_bsz = 1
        e_tp.engine_worker_queue = _ns(exist_tasks=lambda: False)
        e_tp.resource_manager.schedule = lambda: ([], [])
        e_tp._schedule_request_to_worker_v1()


class TestZmqAndSplitwise:
    def test_zmq_start(self, monkeypatch):
        e = _eng(monkeypatch)
        e.start_zmq_service(api_server_pid=None)
        assert not hasattr(e, "recv_request_server")
        # IPC
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.ZmqIpcServer", lambda **kw: _ns(recv_result_handle=_noop, close=_noop)
        )
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.ZmqTcpServer", lambda **kw: _ns(recv_result_handle=_noop, close=_noop)
        )
        monkeypatch.setattr("fastdeploy.engine.common_engine.time.sleep", _noop)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        e2 = _eng(monkeypatch)
        e2.running = False
        e2.start_zmq_service(api_server_pid=1234)
        assert hasattr(e2, "recv_request_server")
        # adapter
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.InternalAdapter", lambda **kw: _ns())
        e3 = _eng(monkeypatch)
        e3.running = False
        e3.cfg.parallel_config.local_data_parallel_id = 0
        e3.start_zmq_service(api_server_pid=5678)
        assert hasattr(e3, "internal_adapter")

    def test_zmq_requests(self, monkeypatch):
        _ptr(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.main_process_metrics.requests_number.inc", _noop)
        monkeypatch.setattr("fastdeploy.engine.common_engine.main_process_metrics.num_requests_waiting.inc", _noop)
        monkeypatch.setattr("fastdeploy.engine.common_engine.Request.from_dict", lambda d: _task(d["request_id"]))
        monkeypatch.setattr("fastdeploy.engine.common_engine.ControlRequest.is_control_request", lambda d: False)
        # normal request
        e = _eng(monkeypatch)
        e.send_response_server = _ns(send_response=_noop)
        e.guided_decoding_checker = None
        _zmq_recv(e, [(None, {"request_id": "z1", "status": None})])
        # abort
        from fastdeploy.engine.request import RequestStatus

        e2 = _eng(monkeypatch)
        e2.send_response_server = _ns(send_response=_noop)
        e2.resource_manager.abort_req_ids_set = set()
        _zmq_recv(e2, [(None, {"request_id": "a1", "status": RequestStatus.ABORT.value})])
        assert "a1" in e2.resource_manager.abort_req_ids_set
        # paused drops
        e3 = _eng(monkeypatch)
        e3.send_response_server = _ns(send_response=_noop)
        e3.is_paused = True
        e3.guided_decoding_checker = None
        dropped = []
        e3._send_error_response = lambda *a: dropped.append(1)
        _zmq_recv(e3, [(None, {"request_id": "p1", "status": None})])
        assert len(dropped) >= 1
        # v1 abort — req IN resource_manager.requests
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        e4 = _eng(monkeypatch)
        e4.resource_manager.abort_req_ids_set = set()
        e4.resource_manager.requests = {"v1a": _ns(request_id="v1a")}
        e4.resource_manager._prepare_preempt_task = lambda r: _ns(request_id=r.request_id)
        e4.resource_manager.real_bsz = 1
        e4.engine_worker_queue = _ns(put_tasks=_noop)
        _zmq_recv(e4, [(None, {"request_id": "v1a", "status": RequestStatus.ABORT.value})])
        assert "v1a" in e4.resource_manager.abort_req_ids_set
        # v1 abort — req NOT in requests (recycle)
        e4b = _eng(monkeypatch)
        e4b.resource_manager.abort_req_ids_set = set()
        e4b.resource_manager.requests = {}
        e4b.scheduler._recycle = _noop
        _zmq_recv(e4b, [(None, {"request_id": "v1b", "status": RequestStatus.ABORT.value})])
        assert "v1b" not in e4b.resource_manager.abort_req_ids_set  # removed after recycle
        # non-Context-terminated error → reconnect
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        e5 = _eng(monkeypatch)
        e5.api_server_pid = "test"
        rc = [0]

        def _r5(block):
            rc[0] += 1
            if rc[0] == 1:
                return ("socket error", None)
            e5.running = False
            return ("Context was terminated", None)

        e5.recv_request_server = _ns(receive_json_once=_r5)
        monkeypatch.setattr("fastdeploy.engine.common_engine.ZmqIpcServer", lambda **kw: e5.recv_request_server)
        e5._insert_zmq_task_to_scheduler()
        # control request via zmq
        monkeypatch.setattr("fastdeploy.engine.common_engine.ControlRequest.is_control_request", lambda d: True)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.ControlRequest.from_dict",
            lambda d: _ns(request_id="c1", get_method=lambda: "x"),
        )
        e6 = _eng(monkeypatch)
        e6.send_response_server = _ns(send_response=_noop)
        _zmq_recv(e6, [(None, {"request_id": "c1"})])
        # from_dict exception
        monkeypatch.setattr("fastdeploy.engine.common_engine.ControlRequest.is_control_request", lambda d: False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.Request.from_dict", lambda d: (_ for _ in ()).throw(ValueError("bad"))
        )
        e7 = _eng(monkeypatch)
        e7.send_response_server = _ns(send_response=_noop)
        e7.guided_decoding_checker = None
        _zmq_recv(e7, [(None, {"request_id": "bad1", "status": None})])
        # guided_decoding_checker rejects
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.Request.from_dict",
            lambda d: _ns(request_id=d["request_id"], metrics=_ns(scheduler_recv_req_time=0)),
        )
        e8 = _eng(monkeypatch)
        e8.send_response_server = _ns(send_response=_noop)
        e8.guided_decoding_checker = _ns(schema_format=lambda r: (r, "schema err"))
        errs8 = []
        e8._send_error_response = lambda *a: errs8.append(a)
        _zmq_recv(e8, [(None, {"request_id": "g1", "status": None})])
        assert len(errs8) >= 1
        # adapter + decode early return (L1125)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        e9 = _eng(monkeypatch)
        e9.cfg.scheduler_config.splitwise_role = "decode"
        e9._insert_zmq_task_to_scheduler()  # returns immediately
        # pyobj_once path (L1133) — enable_mm triggers receive_pyobj_once
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.ControlRequest.is_control_request", lambda d: False)
        monkeypatch.setattr(
            "fastdeploy.engine.common_engine.Request.from_dict",
            lambda d: _ns(request_id=d["request_id"], metrics=_ns(scheduler_recv_req_time=0)),
        )
        e10 = _eng(monkeypatch)
        e10.cfg.model_config.enable_mm = True
        e10.send_response_server = _ns(send_response=_noop)
        e10.guided_decoding_checker = None
        idx10 = [0]

        def recv10(block):
            idx10[0] += 1
            if idx10[0] == 1:
                return None, {"request_id": "mm1", "status": None}
            e10.running = False
            return "Context was terminated", None

        e10.recv_request_server = _ns(receive_pyobj_once=recv10)
        e10._insert_zmq_task_to_scheduler()
        # adapter reconnect via TcpServer (L1143)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False)
        e11 = _eng(monkeypatch)
        e11.cfg.model_config.enable_mm = False
        e11.cfg.scheduler_config.splitwise_role = "mixed"
        e11.api_server_pid = "test"
        rc11 = [0]

        def recv11(block):
            rc11[0] += 1
            if rc11[0] == 1:
                return ("socket error", None)
            e11.running = False
            return ("Context was terminated", None)

        e11.recv_request_server = _ns(receive_json_once=recv11)
        monkeypatch.setattr("fastdeploy.engine.common_engine.ZmqTcpServer", lambda **kw: e11.recv_request_server)
        e11._insert_zmq_task_to_scheduler()

    def test_send_tokens(self, monkeypatch):
        from fastdeploy.engine.request import CompletionOutput, RequestOutput

        def _ro(rid, tids, finished=False, dt=1):
            co = CompletionOutput.__new__(CompletionOutput)
            co.token_ids, co.decode_type, co.text = tids, dt, ""
            ro = RequestOutput.__new__(RequestOutput)
            ro.request_id, ro.outputs, ro.finished = rid, co, finished
            return ro

        # non-adapter
        e = _eng(monkeypatch)
        e.data_processor = _ns(ids2tokens=lambda t, r: ("x", [1], None), decode_status={"r1": [0, 1]})
        sent = []
        e.send_response_server = _ns(send_response=lambda rid, r: sent.append(rid))
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", False)
        cc = [0]

        def gr():
            cc[0] += 1
            if cc[0] == 1:
                return {"r1": [_ro("r1", [1])]}
            e.running = False
            return {}

        e.scheduler.get_results = gr
        e._zmq_send_generated_tokens()
        assert len(sent) >= 1
        # finished empty
        e2 = _eng(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        s2 = []
        e2.send_response_server = _ns(send_response=lambda rid, r: s2.append(rid))
        cc2 = [0]

        def gr2():
            cc2[0] += 1
            if cc2[0] == 1:
                return {"r2": [_ro("r2", [], finished=True)]}
            e2.running = False
            return {}

        e2.scheduler.get_results = gr2
        e2._zmq_send_generated_tokens()
        assert len(s2) >= 1
        # adapter
        e3 = _eng(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        e3.data_processor = _ns(ids2tokens=lambda t, r: ("t", [42], None), decode_status={"a1": [0, 1]})
        s3 = []
        e3.send_response_server = _ns(send_response=lambda rid, r: s3.append(rid))
        cc3 = [0]

        def gr3():
            cc3[0] += 1
            if cc3[0] == 1:
                return [[_ro("a1", [42], dt=0)]]
            e3.running = False
            return []

        e3.scheduler.get_results = gr3
        e3._zmq_send_generated_tokens()
        assert len(s3) >= 1
        # non-adapter: accumulate warning (empty token_ids, not finished)
        e4 = _eng(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        e4.data_processor = _ns(ids2tokens=lambda t, r: ("", [], None), decode_status={"w1": [0, 0]})
        s4 = []
        e4.send_response_server = _ns(send_response=lambda rid, r: s4.append(rid))
        cc4 = [0]

        def gr4():
            cc4[0] += 1
            if cc4[0] == 1:
                return {"w1": [_ro("w1", [1], dt=0)]}
            e4.running = False
            return {}

        e4.scheduler.get_results = gr4
        e4._zmq_send_generated_tokens()
        # adapter: accumulate warning
        e5 = _eng(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
        e5.data_processor = _ns(ids2tokens=lambda t, r: ("", [], None), decode_status={"w2": [0, 0]})
        s5 = []
        e5.send_response_server = _ns(send_response=lambda rid, r: s5.append(rid))
        cc5 = [0]

        def gr5():
            cc5[0] += 1
            if cc5[0] == 1:
                return [[_ro("w2", [1], dt=0)]]
            e5.running = False
            return []

        e5.scheduler.get_results = gr5
        e5._zmq_send_generated_tokens()
        # adapter: decode_type!=0 (L1492), finished+empty (L1498), non-RequestOutput (L1504)
        e6 = _eng(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", False)
        s6 = []
        e6.send_response_server = _ns(send_response=lambda rid, r: s6.append(r))
        cc6 = [0]

        def gr6():
            cc6[0] += 1
            if cc6[0] == 1:
                return [[_ro("a6", [9], dt=1), _ro("a6f", [], finished=True, dt=1), "raw"]]
            e6.running = False
            return []

        e6.scheduler.get_results = gr6
        e6._zmq_send_generated_tokens()
        assert len(s6) >= 1

    def test_splitwise_decode(self, monkeypatch):
        from fastdeploy.engine.request import CompletionOutput, Request, RequestOutput

        _ptr(monkeypatch)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False)

        def _de(v1=False):
            monkeypatch.setattr("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", v1)
            e = _eng(monkeypatch)
            e.cfg.scheduler_config.splitwise_role = "decode"
            e.cfg.splitwise_version = "v1" if v1 else "v0"
            e.enable_decode_cache_task = False
            e.resource_manager.is_resource_sufficient = lambda n: True
            e.resource_manager.allocate_resources_for_new_tasks = lambda t: t
            e.resource_manager.real_bsz = 1
            e.resource_manager.preallocate_resource_in_d = lambda t: True
            e.resource_manager.pre_recycle_resource = _noop
            e.resource_manager.add_prefilled_request = _noop
            e.split_connector = _ns(send_cache_info_to_prefill=_noop)
            e.token_processor.tokens_counter = {}
            e.engine_worker_queue = _ns(
                disaggregate_queue_empty=lambda: True, get_disaggregated_tasks=lambda: [], put_tasks=_noop
            )
            return e

        def _rr(rid="x"):
            r = Request.__new__(Request)
            r.request_id, r.prompt_token_ids_len = rid, 10
            r.metrics = _ns(decode_recv_req_time=0, decode_preallocate_req_time=0)
            r.error_msg = ""
            return r

        def _rro(rid="y", ec=200):
            co = CompletionOutput.__new__(CompletionOutput)
            co.token_ids = [42]
            ro = RequestOutput.__new__(RequestOutput)
            ro.request_id, ro.outputs, ro.finished = rid, co, False
            ro.error_code, ro.error_msg = ec, "" if ec == 200 else "fail"
            ro.metrics = _ns(decode_recv_first_token_time=0)
            return ro

        def _run(eng, items):
            cc = [0]

            def qe():
                return cc[0] > 0

            def gt():
                cc[0] += 1
                eng.running = False
                return items

            eng.engine_worker_queue.disaggregate_queue_empty = qe
            eng.engine_worker_queue.get_disaggregated_tasks = gt
            eng._decode_process_splitwise_requests()
            time.sleep(0.05)  # let daemon thread finish

        # v0: fetch
        e = _de(v1=False)
        ins = []
        e.insert_tasks = lambda t, **kw: ins.append(t)
        _run(e, [(0, [_rr("rf")])])
        assert len(ins) >= 1
        # v1: fetch outputs
        e2 = _de(v1=True)
        e2.scheduler.has_request = lambda rid: True
        added = []
        e2.resource_manager.add_prefilled_request = lambda r: added.append(r)
        _run(e2, [(0, [_rro("ro")])])
        assert len(added) >= 1
        # v1: alloc fail
        e3 = _de(v1=True)
        e3.resource_manager.preallocate_resource_in_d = lambda t: False
        e3.split_connector.send_cache_info_to_prefill = _noop
        rr3 = _rr("rf2")
        _run(e3, [(0, [rr3])])
        assert rr3.error_msg == "Not enough resources"
        # v1: error code
        e4 = _de(v1=True)
        e4.scheduler.has_request = lambda rid: True
        recycled = []
        e4.resource_manager.pre_recycle_resource = lambda rid: recycled.append(rid)
        _run(e4, [(0, [_rro("re", ec=500)])])
        assert "re" in recycled
        # v1: alloc success (Request + preallocate=True) → L1581-1586
        e5 = _de(v1=True)
        sent5 = []
        e5.split_connector.send_cache_info_to_prefill = lambda t: sent5.append(t)
        _run(e5, [(0, [_rr("rs")])])
        assert len(sent5) >= 1
        # v1: has_request=False → waiting (L1617-1618)
        e6 = _de(v1=True)
        e6.scheduler.has_request = lambda rid: False
        added6 = []
        e6.resource_manager.add_prefilled_request = lambda r: added6.append(r)
        _run(e6, [(0, [_rro("rw")])])
        assert len(added6) == 0
        # v1: enable_decode_cache_task → waiting break (L1602-1603)
        e7 = _de(v1=True)
        e7.enable_decode_cache_task = True
        e7.resource_manager.preallocate_resource_in_d = lambda t: False
        _run(e7, [(0, [_rr("rb")])])
        # v1: error_code + adapter EOS in prefilled (L1635-1641, L1648)
        monkeypatch.setattr("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True)
        e8 = _de(v1=True)
        e8.scheduler.has_request = lambda rid: True
        recycled8 = []
        e8.resource_manager.pre_recycle_resource = lambda rid: recycled8.append(rid)
        e8.token_processor.tokens_counter = {"re8": 1, "eos8": 2}
        ro_err = _rro("re8", ec=500)
        ro_eos = _rro("eos8", ec=200)
        ro_eos.outputs.token_ids = []  # EOS → empty token_ids
        _run(e8, [(0, [ro_err, ro_eos])])
        assert "re8" in recycled8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
