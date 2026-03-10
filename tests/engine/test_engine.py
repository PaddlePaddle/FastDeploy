# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from fastdeploy.engine.engine import LLMEngine
from fastdeploy.utils import EngineError


def _make_cfg(**overrides):
    """Minimal cfg matching LLMEngine expectations."""
    ns = SimpleNamespace
    cfg = ns(
        model_config=ns(
            model="/fake",
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
        ),
        parallel_config=ns(
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
        ),
        scheduler_config=ns(
            max_num_seqs=256,
            max_num_batched_tokens=4096,
            splitwise_role="mixed",
            name="local",
            enable_overlap_schedule=False,
        ),
        cache_config=ns(
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
        ),
        load_config=ns(load_strategy="auto", rsync_config={}, dynamic_load_weight=False, load_choices="auto"),
        speculative_config=ns(model_type="main", to_json_string=lambda: "{}"),
        graph_opt_config=ns(to_json_string=lambda: "{}"),
        structured_outputs_config=ns(
            guided_decoding_backend=None,
            logits_processors=None,
            reasoning_parser="none",
            disable_any_whitespace=False,
        ),
        early_stop_config=ns(to_json_string=lambda: "{}"),
        eplb_config=ns(to_json_string=lambda: "{}"),
        routing_replay_config=ns(to_json_string=lambda: "{}"),
        plas_attention_config=ns(to_json_string=lambda: "{}"),
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
    """Create LLMEngine via object.__new__ — skips __init__ which needs GPU."""
    engine = object.__new__(LLMEngine)
    engine.cfg = _make_cfg(**cfg_overrides)
    engine.running = True
    engine.is_started = False
    engine.do_profile = 0
    engine.engine = SimpleNamespace(scheduler=SimpleNamespace(get_results=lambda: []))
    engine.guided_decoding_checker = None
    engine.ipc_signal_suffix = 6778
    return engine


def _make_request(token_count=10, max_tokens=100, min_tokens=0, stop_seqs_len=None, **overrides):
    """Create a mock request for add_requests testing."""
    vals = {"max_tokens": max_tokens, "min_tokens": min_tokens, "request_id": "x", "stop_seqs_len": stop_seqs_len}
    req = SimpleNamespace(
        prompt_token_ids=list(range(token_count)),
        prompt_token_ids_len=token_count,
        need_prefill_tokens=token_count,
        metrics=SimpleNamespace(scheduler_recv_req_time=0, preprocess_start_time=0, preprocess_end_time=0),
        get=lambda k: vals.get(k),
        set=None,
        sampling_params=None,
        guided_json=None,
        guided_regex=None,
        guided_choice=None,
        structural_tag=None,
        guided_grammar=None,
        guided_json_object=None,
    )
    req.set = lambda k, v: setattr(req, k, v)
    for k, v in overrides.items():
        setattr(req, k, v)
    return req


def _make_tokenizer(**kw):
    """Create a minimal tokenizer for _start_worker_service."""
    base = dict(
        vocab={"<pad>": 0, "hello": 1},
        get_vocab=lambda: {"<think>": 5, "</think>": 6, "<|IMAGE_PLACEHOLDER|>": -1, "\n": 10},
        encode=lambda s, add_special_tokens=False: [10],
        think_truncate_prompt="...",
        tokenize=lambda s: ["..."],
        convert_tokens_to_ids=lambda t: [99],
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestLLMEngine:
    """Tests for LLMEngine — covers testable methods without GPU."""

    def test_has_guided_input(self):
        """None->False; each guided field set->True."""
        e = _make_engine()
        fields = [
            "guided_json",
            "guided_regex",
            "guided_choice",
            "structural_tag",
            "guided_grammar",
            "guided_json_object",
        ]
        assert e._has_guided_input(SimpleNamespace(**{f: None for f in fields})) is False
        for field in fields:
            kw = {f: None for f in fields}
            kw[field] = "value"
            assert e._has_guided_input(SimpleNamespace(**kw)) is True

    def test_setting_environ_variables(self, monkeypatch):
        """Mixed (no disagg), prefill (disagg), and V1 (per_chunk) branches."""
        e = _make_engine()
        result = e._setting_environ_variables()
        assert "OMP_NUM_THREADS=" in result
        assert "NCCL_ALGO=Ring" in result
        assert "FLAGS_use_pd_disaggregation" not in result

        e.cfg.scheduler_config.splitwise_role = "prefill"
        result = e._setting_environ_variables()
        assert "FLAGS_use_pd_disaggregation" in result

        monkeypatch.setattr("fastdeploy.engine.engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True)
        result = e._setting_environ_variables()
        assert "FLAGS_use_pd_disaggregation_per_chunk" in result

    def test_health_and_readiness(self):
        """Worker readiness (single/multi) and health check (ok/stale)."""
        e = _make_engine()
        e.worker_ready_signal = SimpleNamespace(value=np.zeros(1, dtype=np.int32))
        assert e._worker_processes_ready() is False
        e.worker_ready_signal = SimpleNamespace(value=np.ones(1, dtype=np.int32))
        assert e._worker_processes_ready() is True
        e.cfg.worker_num_per_node = 3
        e.worker_ready_signal = SimpleNamespace(value=np.array([1, 1, 0], dtype=np.int32))
        assert e._worker_processes_ready() is False

        e.engine.worker_healthy_live_signal = SimpleNamespace(value=np.array([0.0]))
        assert e.check_health()[0] is True
        e.engine.worker_healthy_live_signal = SimpleNamespace(value=np.array([time.time()]))
        assert e.check_health()[0] is True
        e.engine.worker_healthy_live_signal = SimpleNamespace(value=np.array([time.time() - 60]))
        healthy, msg = e.check_health(time_interval_threashold=30)
        assert healthy is False
        assert "Not Healthy" in msg

    def test_format_and_add_data(self):
        """ID generation, ID preservation, default max_tokens, context extraction."""
        e = _make_engine()
        e.add_requests = lambda t, **kw: None

        prompts = {"prompt": "Hello"}
        req_id = e._format_and_add_data(prompts)
        uuid.UUID(req_id)
        assert prompts["max_tokens"] == 2048

        prompts2 = {"prompt": "Hi", "request_id": "my-id", "max_tokens": 50}
        assert e._format_and_add_data(prompts2) == "my-id"

        prompts3 = {
            "context": [
                {"role": "system", "utterance": "Helper"},
                {"role": "user", "utterance": "Hi"},
                {"role": "assistant", "utterance": "Hey"},
            ]
        }
        e._format_and_add_data(prompts3)
        assert prompts3["system"] == "Helper"
        assert prompts3["prompt"] == ["Hi", "Hey"]

    def test_init_worker_signals(self, monkeypatch):
        """Basic signals + prefix_caching and dp>1 variants."""
        ipc = lambda **kw: SimpleNamespace(
            value=np.zeros(kw.get("array", np.zeros(1)).shape, dtype=kw.get("dtype", np.int32)),
            clear=lambda: None,
        )
        monkeypatch.setattr("fastdeploy.engine.engine.IPCSignal", ipc)

        e = _make_engine()
        e._init_worker_signals()
        assert hasattr(e, "worker_ready_signal")
        assert hasattr(e, "loaded_model_signal")
        assert not hasattr(e, "launched_cache_manager_signal")

        e2 = _make_engine()
        e2.cfg.cache_config.enable_prefix_caching = True
        e2._init_worker_signals()
        assert hasattr(e2, "launched_cache_manager_signal")

        e3 = _make_engine()
        e3.cfg.parallel_config.data_parallel_size = 2
        monkeypatch.setattr("fastdeploy.engine.engine.envs.FD_ENABLE_MULTI_API_SERVER", False)
        e3._init_worker_signals()
        assert hasattr(e3, "launched_expert_service_signal")

        e4 = _make_engine()
        e4.do_profile = 1
        monkeypatch.setattr("fastdeploy.engine.engine.paddle.is_compiled_with_custom_device", lambda x: False)
        e4._init_worker_signals()
        assert hasattr(e4, "get_profile_block_num_signal")

    def test_exit_sub_services(self, monkeypatch):
        """Exit with worker, cache_manager, dp_processes, and zmq cleanup."""
        e = _make_engine()
        e.worker_ready_signal = SimpleNamespace(clear=lambda: None)
        e.loaded_model_signal = SimpleNamespace(clear=lambda: None)
        killed = []
        monkeypatch.setattr("fastdeploy.engine.engine.os.getpgid", lambda pid: pid)
        monkeypatch.setattr("fastdeploy.engine.engine.os.killpg", lambda pgid, sig: killed.append(pgid))
        e.worker_proc = SimpleNamespace(pid=99)
        e.engine.resource_manager = SimpleNamespace(
            cache_manager=SimpleNamespace(
                shm_cache_task_flag_broadcast=SimpleNamespace(clear=lambda: None),
                cache_ready_signal=SimpleNamespace(clear=lambda: None),
            )
        )
        e.cache_manager_processes = [SimpleNamespace(pid=55)]
        joined = []
        e.dp_processed = [SimpleNamespace(pid=77, join=lambda: joined.append(1))]
        e.dp_engine_worker_queue_server = [SimpleNamespace(cleanup=lambda: None)]
        closed = []
        e.zmq_server = SimpleNamespace(close=lambda: closed.append(1))
        e.get_profile_block_num_signal = SimpleNamespace(clear=lambda: None)
        e._exit_sub_services()
        assert e.running is False
        assert 55 in killed and 99 in killed
        assert len(joined) == 1 and len(closed) == 1

    def test_stop_profile(self, monkeypatch):
        """Resets do_profile and cache config."""
        e = _make_engine()
        e.do_profile = 1
        e.get_profile_block_num_signal = SimpleNamespace(value=np.array([100], dtype=np.int32))
        reset_calls = []
        e.engine.resource_manager = SimpleNamespace(reset_cache_config=lambda cfg: None)
        e.cfg.cache_config = SimpleNamespace(
            reset=lambda n: reset_calls.append(n),
            enable_prefix_caching=False,
        )
        e.cfg.scheduler_config.splitwise_role = "mixed"
        e._stop_profile()
        assert e.do_profile == 0
        assert reset_calls == [100]

        e2 = _make_engine()
        e2.do_profile = 1
        e2.get_profile_block_num_signal = SimpleNamespace(value=np.array([100], dtype=np.int32))
        e2.engine.resource_manager = SimpleNamespace(reset_cache_config=lambda cfg: None)
        e2.cfg.cache_config = SimpleNamespace(reset=lambda n: None, enable_prefix_caching=True)
        e2.cfg.scheduler_config.splitwise_role = "mixed"
        monkeypatch.setattr("fastdeploy.engine.engine.current_platform.is_intel_hpu", lambda: False)
        e2.engine.start_cache_service = lambda d, s: [SimpleNamespace(pid=1)]
        e2._stop_profile()
        assert hasattr(e2, "cache_manager_processes")

    def test_from_engine_args(self, monkeypatch):
        """Profile flag depends on num_gpu_blocks_override."""
        monkeypatch.setattr("fastdeploy.engine.engine.EngineService", lambda cfg: SimpleNamespace())
        monkeypatch.setattr("fastdeploy.engine.engine.main_process_metrics.set_cache_config_info", lambda **kw: None)
        monkeypatch.setattr("fastdeploy.engine.engine.tracing.trace_set_thread_info", lambda s: None)

        args = SimpleNamespace(create_engine_config=lambda: _make_cfg())
        assert LLMEngine.from_engine_args(args).do_profile == 1

        cfg2 = _make_cfg()
        cfg2.cache_config.num_gpu_blocks_override = 100
        args2 = SimpleNamespace(create_engine_config=lambda: cfg2)
        assert LLMEngine.from_engine_args(args2).do_profile == 0

    def test_launch_components(self, monkeypatch):
        """Splitwise receiver thread, scheduler start, and DP process creation."""
        e = _make_engine()
        e.cfg.scheduler_config.splitwise_role = "prefill"
        e.cfg.scheduler_config.name = "splitwise"
        started = []
        e.engine.split_connector = SimpleNamespace(start_receiver=lambda: None)
        e.engine.scheduler = SimpleNamespace(start=lambda *a, **kw: started.append(1))
        e.launch_components()
        assert hasattr(e, "splitwise_receive_thread")
        assert len(started) == 1

        e2 = _make_engine()
        e2.cfg.scheduler_config.name = "local"
        e2.cfg.parallel_config.data_parallel_size = 2
        e2.cfg.parallel_config.engine_worker_queue_port = [6778, 6779]
        monkeypatch.setattr("fastdeploy.engine.engine.envs.FD_ENABLE_MULTI_API_SERVER", False)
        monkeypatch.setattr("fastdeploy.engine.engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", False)
        e2.launched_expert_service_signal = SimpleNamespace(value=np.array([0, 1], dtype=np.int32))
        mock_proc = SimpleNamespace(start=lambda: None, pid=111)
        monkeypatch.setattr(
            "fastdeploy.engine.engine.multiprocessing.get_context",
            lambda kind: SimpleNamespace(Process=lambda target, args: mock_proc),
        )
        monkeypatch.setattr("fastdeploy.engine.engine.EngineWorkerQueue", lambda **kw: SimpleNamespace())
        monkeypatch.setattr("fastdeploy.engine.engine.copy.deepcopy", lambda x: x)
        e2.launch_components()
        assert len(e2.dp_processed) == 1
        assert e2.dp_processed[0].pid == 111

    def test_add_requests(self, monkeypatch):
        """Validation errors (overflow, stop seqs, guided) and happy path."""
        monkeypatch.setattr("fastdeploy.engine.engine.Request.from_dict", lambda d: d["_req"])
        e = _make_engine()
        e.engine.data_processor = SimpleNamespace(process_request=lambda r, *a, **kw: r)

        req1 = _make_request(token_count=3000)
        with pytest.raises(EngineError):
            e.add_requests({"prompt": "x", "_req": req1})

        req2 = _make_request(token_count=100, min_tokens=2000)
        with pytest.raises(EngineError):
            e.add_requests({"prompt": "x", "_req": req2})

        monkeypatch.setattr("fastdeploy.engine.engine.envs.FD_MAX_STOP_SEQS_NUM", 10)
        req3 = _make_request(stop_seqs_len=list(range(200)))
        with pytest.raises(EngineError):
            e.add_requests({"prompt": "x", "_req": req3})

        monkeypatch.setattr("fastdeploy.engine.engine.envs.FD_STOP_SEQS_MAX_LEN", 5)
        req4 = _make_request(stop_seqs_len=[20])
        with pytest.raises(EngineError):
            e.add_requests({"prompt": "x", "_req": req4})

        req5 = _make_request(guided_json='{"type":"object"}')
        with pytest.raises(EngineError):
            e.add_requests({"prompt": "x", "_req": req5})

        put_calls = []
        req = _make_request()
        monkeypatch.setattr("fastdeploy.engine.engine.Request.from_dict", lambda d: req)
        monkeypatch.setattr("fastdeploy.engine.engine.asdict", lambda x: {"temperature": 0.0})
        e.engine.scheduler = SimpleNamespace(put_requests=lambda reqs: put_calls.extend(reqs))
        sp = SimpleNamespace(temperature=0.0)
        e.add_requests({"prompt": "hi"}, sampling_params=sp)
        assert len(put_calls) == 1
        assert sp.temperature == 1e-06

    def test_start_worker_service(self, monkeypatch):
        """Builds subprocess command with correct arguments and flags."""
        e = _make_engine()
        e.cfg.cache_config.num_gpu_blocks_override = 200
        e.cfg.parallel_config.enable_expert_parallel = True
        e.cfg.cache_config.enable_prefix_caching = True
        e.cfg.cache_config.kvcache_storage_backend = "rocksdb"
        tok = _make_tokenizer()
        e.data_processor = SimpleNamespace(tokenizer=tok, eos_token_id_len=1, pad_token_id=0)
        e.engine.data_processor = e.data_processor
        e.engine.mm_max_tokens_per_item = None
        captured = []
        monkeypatch.setattr(
            "fastdeploy.engine.engine.subprocess.Popen",
            lambda cmd, **kw: SimpleNamespace(pid=1) if captured.append(cmd) or True else None,
        )
        monkeypatch.setattr("fastdeploy.engine.engine.current_platform.is_iluvatar", lambda: False)
        e._start_worker_service()
        cmd = captured[0]
        assert "--max_model_len 2048" in cmd
        assert "--enable_expert_parallel" in cmd
        assert "--enable_prefix_caching" in cmd
        assert "--num_gpu_blocks_override 200" in cmd
        assert "--kvcache_storage_backend rocksdb" in cmd

        # sp_model vocab variant
        tok2 = _make_tokenizer()
        tok2.sp_model = type("SP", (), {"__len__": lambda self: 5000})()
        e2 = _make_engine()
        e2.data_processor = SimpleNamespace(tokenizer=tok2, eos_token_id_len=1, pad_token_id=0)
        e2.engine.data_processor = e2.data_processor
        e2.engine.mm_max_tokens_per_item = None
        e2._start_worker_service()
        assert "--ori_vocab_size 5000" in captured[-1]

        # nnode > 1 variant
        e3 = _make_engine()
        e3.cfg.nnode = 2
        e3.cfg.ips = ["10.0.0.1", "10.0.0.2"]
        e3.data_processor = SimpleNamespace(tokenizer=tok, eos_token_id_len=1, pad_token_id=0)
        e3.engine.data_processor = e3.data_processor
        e3.engine.mm_max_tokens_per_item = None
        e3._start_worker_service()
        assert "--nnodes 2" in captured[-1]

    def test_generate(self):
        """Stream (intermediate+final), non-stream (final only), and error wrapping."""
        e = _make_engine()
        e.add_requests = lambda t, **kw: None
        e.engine.check_and_free_block_tables = lambda: None
        e.engine.data_processor = SimpleNamespace(
            process_response=lambda r: SimpleNamespace(
                to_dict=lambda: {"outputs": {"text": "hi", "reasoning_content": ""}}
            )
        )
        results_s = [SimpleNamespace(finished=False), SimpleNamespace(finished=True)]
        e._get_generated_tokens = lambda rid: iter(results_s)
        assert len(list(e.generate({"prompt": "x"}, stream=True))) == 2

        results_ns = [SimpleNamespace(finished=True)]
        e._get_generated_tokens = lambda rid: iter(results_ns)
        out = list(e.generate({"prompt": "x"}, stream=False))
        assert len(out) == 1
        assert out[0]["outputs"]["text"] == "hi"

        assert e._get_generated_result() == []

        e.add_requests = lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad"))
        with pytest.raises(EngineError):
            list(e.generate({"prompt": "x"}, stream=False))

    def test_check_worker_initialize_status(self, monkeypatch):
        """Worker status polling — success path and process death."""
        monkeypatch.setattr("fastdeploy.engine.engine.time.sleep", lambda s: None)
        monkeypatch.setattr(
            "fastdeploy.engine.engine.threading.Thread",
            lambda target, daemon: SimpleNamespace(
                start=lambda: target(),
                join=lambda timeout=None: None,
            ),
        )
        monkeypatch.setattr(
            "fastdeploy.engine.engine.tqdm",
            lambda total, desc: SimpleNamespace(
                __enter__=lambda s: SimpleNamespace(n=0, update=lambda x: None, refresh=lambda: None),
                __exit__=lambda s, *a: None,
            ),
        )

        e = _make_engine()
        e.worker_init_status = {}
        e.worker_proc = SimpleNamespace(
            stdout=iter([b"Loading checkpoint shards: 100\n"]),
            poll=lambda: None,
        )
        e.worker_ready_signal = SimpleNamespace(value=np.ones(1, dtype=np.int32))
        assert e.check_worker_initialize_status() is True

        e2 = _make_engine()
        e2.worker_init_status = {}
        e2.worker_proc = SimpleNamespace(
            stdout=iter([]),
            poll=lambda: 1,
        )
        e2.worker_ready_signal = SimpleNamespace(value=np.zeros(1, dtype=np.int32))
        assert e2.check_worker_initialize_status() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
