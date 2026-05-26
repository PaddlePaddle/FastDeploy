# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import os
import tempfile
import time
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from fastdeploy.engine.engine import LLMEngine
from fastdeploy.utils import EngineError


def _make_cfg(**ov):
    ns = SimpleNamespace
    _j = lambda: "{}"
    mc = ns(model="/fake", model_type="ernie", max_model_len=2048, num_hidden_layers=2, quantization="{}")
    mc.runner, mc.convert, mc.override_pooler_config, mc.logprobs_mode = "default", None, None, "none"
    mc.max_logprobs, mc.enable_logprob, mc.lm_head_fp32, mc.moe_gate_fp32 = 0, False, False, False
    mc.enable_entropy, mc.model_impl = False, "default"
    pc = ns(tensor_parallel_size=1, tensor_parallel_rank=0, device_ids="0", data_parallel_size=1)
    pc.expert_parallel_size, pc.chunked_moe_size, pc.engine_worker_queue_port = 1, 0, [6778]
    pc.enable_expert_parallel = pc.enable_chunked_moe = pc.disable_custom_all_reduce = False
    pc.use_internode_ll_two_stage = pc.disable_sequence_parallel_moe = False
    pc.shutdown_comm_group_if_worker_idle = False
    pc.ep_prefill_use_worst_num_tokens = False
    pc.enable_flashinfer_allreduce_fusion = False
    sc = ns(max_num_seqs=256, max_num_batched_tokens=4096, splitwise_role="mixed", name="local")
    sc.enable_overlap_schedule = False
    sc.enable_moe_scores_elementwise_fuse = False
    cc = ns(num_gpu_blocks_override=None, gpu_memory_utilization=0.9, block_size=16, enc_dec_block_num=0)
    cc.enable_prefix_caching = cc.enable_chunked_prefill = False
    cc.kv_cache_ratio, cc.kvcache_storage_backend, cc.num_cpu_blocks, cc.max_encoder_cache = 1.0, None, 0, 0
    cc.cache_transfer_protocol, cc.total_block_num = "tcp", 100
    lc = ns(
        load_strategy="auto",
        rsync_config={},
        dynamic_load_weight=False,
        load_choices="auto",
        model_loader_extra_config={},
    )
    soc = ns(guided_decoding_backend=None, logits_processors=None, reasoning_parser="none")
    soc.disable_any_whitespace = False
    cfg = ns(model_config=mc, parallel_config=pc, scheduler_config=sc, cache_config=cc, load_config=lc)
    cfg.speculative_config = ns(model_type="main", to_json_string=_j)
    cfg.graph_opt_config = cfg.early_stop_config = cfg.eplb_config = ns(to_json_string=_j)
    cfg.routing_replay_config = cfg.plas_attention_config = ns(to_json_string=_j)
    cfg.structured_outputs_config = soc
    cfg.deploy_modality = ns(value="mixed")
    cfg.worker_num_per_node, cfg.master_ip, cfg.host_ip = 1, "127.0.0.1", "127.0.0.1"
    cfg.ips, cfg.nnode, cfg.register_info, cfg.node_rank = None, 1, None, 0
    cfg.print = lambda: None
    for k, v in ov.items():
        setattr(cfg, k, v)
    return cfg


def _make_engine(**ov):
    e = object.__new__(LLMEngine)
    e.cfg = _make_cfg(**ov)
    e.running, e.is_started, e.do_profile = True, False, 0
    e.engine = SimpleNamespace(scheduler=SimpleNamespace(get_results=lambda: []))
    e.guided_decoding_checker, e.ipc_signal_suffix = None, 6778
    return e


def _make_request(token_count=10, max_tokens=100, min_tokens=0, stop_seqs_len=None, **ov):
    vals = {"max_tokens": max_tokens, "min_tokens": min_tokens, "request_id": "x", "stop_seqs_len": stop_seqs_len}
    req = SimpleNamespace(prompt_token_ids=list(range(token_count)), prompt_token_ids_len=token_count)
    req.need_prefill_tokens = token_count
    req.metrics = SimpleNamespace(scheduler_recv_req_time=0, preprocess_start_time=0, preprocess_end_time=0)
    req.get = lambda k: vals.get(k)
    req.set = lambda k, v: setattr(req, k, v)
    req.sampling_params = req.guided_json = req.guided_regex = req.guided_choice = None
    req.structural_tag = req.guided_grammar = req.guided_json_object = None
    for k, v in ov.items():
        setattr(req, k, v)
    return req


def _make_tokenizer(**kw):
    d = dict(vocab={"<pad>": 0, "hello": 1}, think_truncate_prompt="...", tokenize=lambda s: ["..."])
    d["get_vocab"] = lambda: {"<think>": 5, "</think>": 6, "<|IMAGE_PLACEHOLDER|>": -1, "\n": 10}
    d["encode"], d["convert_tokens_to_ids"] = (lambda s, add_special_tokens=False: [10]), (lambda t: [99])
    d.update(kw)
    return SimpleNamespace(**d)


class TestLLMEngineLifecycle(unittest.TestCase):
    def test_start(self):
        ipc = lambda **kw: SimpleNamespace(
            value=np.zeros(kw.get("array", np.zeros(1)).shape, dtype=kw.get("dtype", np.int32)), clear=lambda: None
        )
        tok = _make_tokenizer()
        dp = SimpleNamespace(tokenizer=tok, eos_token_id_len=1, pad_token_id=0)

        def _fake_init_signals(self_arg):
            self_arg.worker_ready_signal = SimpleNamespace(value=np.ones(1, dtype=np.int32), clear=lambda: None)
            self_arg.loaded_model_signal = SimpleNamespace(value=np.array([1], dtype=np.int32), clear=lambda: None)

        def _prepare_started_engine(splitwise_role="mixed", enable_prefix_caching=False):
            e = _make_engine()
            e.cfg.scheduler_config.splitwise_role = splitwise_role
            e.cfg.cache_config.enable_prefix_caching = enable_prefix_caching
            e.engine.start = lambda: None
            e.engine.create_data_processor = lambda: None
            e.engine.data_processor = dp
            e.engine.start_zmq_service = lambda pid: None
            cache_calls = []
            e.engine.start_cache_service = lambda d, s: cache_calls.append((tuple(d), s)) or []
            e.engine.mm_max_tokens_per_item = None
            return e, cache_calls

        with (
            patch("fastdeploy.engine.engine.IPCSignal", ipc),
            patch("fastdeploy.engine.engine.current_platform.is_intel_hpu", lambda: False),
            patch("fastdeploy.engine.engine.time.sleep", lambda s: None),
            patch("fastdeploy.engine.engine.time.time", lambda: 1.0),
            patch("fastdeploy.engine.engine.subprocess.Popen", lambda cmd, **kw: SimpleNamespace(pid=1)),
            patch.object(LLMEngine, "_init_worker_signals", lambda s: _fake_init_signals(s)),
            patch.object(LLMEngine, "launch_components", lambda s: None),
            patch("fastdeploy.engine.engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False),
            patch("fastdeploy.engine.engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True),
        ):
            # splitwise_role != mixed branch: cache service starts before worker launch.
            e_prefill, prefill_cache_calls = _prepare_started_engine(splitwise_role="prefill")
            e_prefill.cfg.cache_config.num_gpu_blocks_override = 50
            e_prefill.cfg.cache_config.num_cpu_blocks = 10
            with patch.object(LLMEngine, "check_worker_initialize_status", lambda s: True):
                result_prefill = e_prefill.start(api_server_pid=999)
            self.assertTrue(result_prefill)
            self.assertEqual(e_prefill.api_server_pid, 999)
            self.assertEqual(len(prefill_cache_calls), 1)

            # splitwise_role == mixed + prefix caching branch.
            e_mixed, mixed_cache_calls = _prepare_started_engine(splitwise_role="mixed", enable_prefix_caching=True)
            with patch.object(LLMEngine, "check_worker_initialize_status", lambda s: True):
                result_mixed = e_mixed.start(api_server_pid=888)
            self.assertTrue(result_mixed)
            self.assertEqual(e_mixed.api_server_pid, 888)
            self.assertEqual(len(mixed_cache_calls), 1)

            # Worker init failure branch after loading path.
            e_fail, _ = _prepare_started_engine(splitwise_role="mixed")
            with patch.object(LLMEngine, "check_worker_initialize_status", lambda s: False):
                self.assertFalse(e_fail.start(api_server_pid=777))

        # Internal adapter branch (covers L185-194).
        with (
            patch("fastdeploy.engine.engine.IPCSignal", ipc),
            patch("fastdeploy.engine.engine.current_platform.is_intel_hpu", lambda: False),
            patch("fastdeploy.engine.engine.time.sleep", lambda s: None),
            patch("fastdeploy.engine.engine.time.time", lambda: 1.0),
            patch("fastdeploy.engine.engine.subprocess.Popen", lambda cmd, **kw: SimpleNamespace(pid=1)),
            patch.object(LLMEngine, "_init_worker_signals", lambda s: _fake_init_signals(s)),
            patch.object(LLMEngine, "launch_components", lambda s: None),
            patch("fastdeploy.engine.engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True),
            patch("fastdeploy.engine.engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True),
            patch("fastdeploy.engine.engine.envs.FD_ZMQ_RECV_REQUEST_SERVER_PORTS", "6000,6001"),
            patch("fastdeploy.engine.engine.envs.FD_ZMQ_SEND_RESPONSE_SERVER_PORTS", "7000,7001"),
            patch.object(LLMEngine, "check_worker_initialize_status", lambda s: True),
        ):
            e_adapter, _ = _prepare_started_engine(splitwise_role="mixed")
            result_adapter = e_adapter.start(api_server_pid=666)
            self.assertTrue(result_adapter)

    def test_from_engine_args(self):
        with (
            patch("fastdeploy.engine.engine.EngineService", lambda cfg: SimpleNamespace()),
            patch("fastdeploy.engine.engine.main_process_metrics.set_cache_config_info", lambda **kw: None),
            patch("fastdeploy.engine.engine.tracing.trace_set_thread_info", lambda s: None),
        ):
            args = SimpleNamespace(create_engine_config=lambda: _make_cfg())
            eng1 = LLMEngine.from_engine_args(args)
            self.assertEqual(eng1.do_profile, 1)
            eng1._finalizer.detach()
            cfg2 = _make_cfg()
            cfg2.cache_config.num_gpu_blocks_override = 100
            eng2 = LLMEngine.from_engine_args(SimpleNamespace(create_engine_config=lambda: cfg2))
            self.assertEqual(eng2.do_profile, 0)
            eng2._finalizer.detach()

    def test_exit_sub_services(self):
        e = _make_engine()
        e.worker_ready_signal = e.loaded_model_signal = SimpleNamespace(clear=lambda: None)
        killed = []
        with (
            patch("fastdeploy.engine.engine.os.getpgid", lambda pid: pid),
            patch("fastdeploy.engine.engine.os.killpg", lambda pgid, sig: killed.append(pgid)),
        ):
            e.worker_proc = SimpleNamespace(pid=99)
            _cm = SimpleNamespace(shm_cache_task_flag_broadcast=SimpleNamespace(clear=lambda: None))
            _cm.cache_ready_signal = SimpleNamespace(clear=lambda: None)
            e.engine.resource_manager = SimpleNamespace(cache_manager=_cm)
            e.cache_manager_processes = [SimpleNamespace(pid=55)]
            joined, closed = [], []
            e.dp_processed = [SimpleNamespace(pid=77, join=lambda: joined.append(1))]
            e.dp_engine_worker_queue_server = [SimpleNamespace(cleanup=lambda: None)]
            e.zmq_server = SimpleNamespace(close=lambda: closed.append(1))
            e.get_profile_block_num_signal = SimpleNamespace(clear=lambda: None)
            e._exit_sub_services()
            self.assertFalse(e.running)
            self.assertIn(55, killed)
            self.assertIn(99, killed)
            self.assertEqual(len(joined), 1)
            self.assertEqual(len(closed), 1)

        # Exception path in cache_manager killpg (L437-438).
        e2 = _make_engine()
        e2.worker_ready_signal = e2.loaded_model_signal = SimpleNamespace(clear=lambda: None)
        with patch("fastdeploy.engine.engine.os.getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError)):
            _cm2 = SimpleNamespace(shm_cache_task_flag_broadcast=SimpleNamespace(clear=lambda: None))
            _cm2.cache_ready_signal = SimpleNamespace(clear=lambda: None)
            e2.engine.resource_manager = SimpleNamespace(cache_manager=_cm2)
            e2.cache_manager_processes = [SimpleNamespace(pid=55)]
            e2.worker_proc = None
            e2._exit_sub_services()
            self.assertFalse(e2.running)

    def test_stop_profile(self):
        e = _make_engine()
        e.do_profile = 1
        e.get_profile_block_num_signal = SimpleNamespace(value=np.array([100], dtype=np.int32))
        reset_calls = []
        e.engine.resource_manager = SimpleNamespace(reset_cache_config=lambda cfg: None)
        e.cfg.cache_config = SimpleNamespace(reset=lambda n: reset_calls.append(n), enable_prefix_caching=False)
        e.cfg.scheduler_config.splitwise_role = "mixed"
        e._stop_profile()
        self.assertEqual(e.do_profile, 0)
        self.assertEqual(reset_calls, [100])
        e2 = _make_engine()
        e2.do_profile = 1
        e2.get_profile_block_num_signal = SimpleNamespace(value=np.array([100], dtype=np.int32))
        e2.engine.resource_manager = SimpleNamespace(reset_cache_config=lambda cfg: None)
        e2.cfg.cache_config = SimpleNamespace(reset=lambda n: None, enable_prefix_caching=True)
        e2.cfg.scheduler_config.splitwise_role = "mixed"
        with patch("fastdeploy.engine.engine.current_platform.is_intel_hpu", lambda: False):
            e2.engine.start_cache_service = lambda d, s: [SimpleNamespace(pid=1)]
            e2._stop_profile()
            self.assertTrue(hasattr(e2, "cache_manager_processes"))


class TestLLMEngineWorker(unittest.TestCase):
    def test_init_worker_signals(self):
        ipc = lambda **kw: SimpleNamespace(
            value=np.zeros(kw.get("array", np.zeros(1)).shape, dtype=kw.get("dtype", np.int32)), clear=lambda: None
        )
        with patch("fastdeploy.engine.engine.IPCSignal", ipc):
            e = _make_engine()
            e._init_worker_signals()
            self.assertTrue(hasattr(e, "worker_ready_signal"))
            self.assertTrue(hasattr(e, "loaded_model_signal"))
            self.assertFalse(hasattr(e, "launched_cache_manager_signal"))
            e2 = _make_engine()
            e2.cfg.cache_config.enable_prefix_caching = True
            e2._init_worker_signals()
            self.assertTrue(hasattr(e2, "launched_cache_manager_signal"))
            e3 = _make_engine()
            e3.cfg.parallel_config.data_parallel_size = 2
            with patch("fastdeploy.engine.engine.envs.FD_ENABLE_MULTI_API_SERVER", False):
                e3._init_worker_signals()
                self.assertTrue(hasattr(e3, "launched_expert_service_signal"))
            e4 = _make_engine()
            e4.do_profile = 1
            with patch("fastdeploy.engine.engine.paddle.is_compiled_with_custom_device", lambda x: False):
                e4._init_worker_signals()
                self.assertTrue(hasattr(e4, "get_profile_block_num_signal"))
            e5 = _make_engine()
            e5.do_profile = 1
            with patch("fastdeploy.engine.engine.paddle.is_compiled_with_custom_device", lambda x: True):
                e5._init_worker_signals()
                self.assertEqual(e5.get_profile_block_num_signal.value.shape[0], e5.cfg.worker_num_per_node)

    def test_start_worker_service(self):
        captured = []
        _popen = lambda cmd, **kw: SimpleNamespace(pid=1) if captured.append(cmd) or True else None
        with (
            patch("fastdeploy.engine.engine.subprocess.Popen", _popen),
            patch("fastdeploy.engine.engine.current_platform.is_iluvatar", lambda: False),
        ):
            e = _make_engine()
            e.cfg.cache_config.num_gpu_blocks_override = 200
            e.cfg.parallel_config.enable_expert_parallel = True
            e.cfg.cache_config.enable_prefix_caching = True
            e.cfg.cache_config.kvcache_storage_backend = "rocksdb"
            tok = _make_tokenizer()
            e.data_processor = SimpleNamespace(tokenizer=tok, eos_token_id_len=1, pad_token_id=0)
            e.engine.data_processor = e.data_processor
            e.engine.mm_max_tokens_per_item = None
            e._start_worker_service()
            cmd = captured[0]
            self.assertIn("--max_model_len 2048", cmd)
            self.assertIn("--enable_expert_parallel", cmd)
            self.assertIn("--enable_prefix_caching", cmd)
            self.assertIn("--num_gpu_blocks_override 200", cmd)
            self.assertIn("--kvcache_storage_backend rocksdb", cmd)

    def test_launch_components(self):
        e = _make_engine()
        e.cfg.scheduler_config.splitwise_role = "prefill"
        e.cfg.scheduler_config.name = "splitwise"
        started = []
        e.engine.split_connector = SimpleNamespace(start_receiver=lambda: None)
        e.engine.scheduler = SimpleNamespace(start=lambda *a, **kw: started.append(1))
        e.launch_components()
        self.assertTrue(hasattr(e, "splitwise_receive_thread"))
        self.assertEqual(len(started), 1)

    def test_check_worker_initialize_status(self):
        _th = lambda target, daemon: SimpleNamespace(start=lambda: target(), join=lambda **kw: None)
        _ctx = SimpleNamespace(n=0, update=lambda x: None, refresh=lambda: None)
        _tq = type("T", (), {"__enter__": lambda s: _ctx, "__exit__": lambda s, *a: None})
        with (
            patch("fastdeploy.engine.engine.time.sleep", lambda s: None),
            patch("fastdeploy.engine.engine.threading.Thread", _th),
            patch("fastdeploy.engine.engine.tqdm", lambda total, desc: _tq()),
        ):
            # Success path with weight + layer loading progress
            e = _make_engine()
            e.worker_init_status = {}
            e.worker_proc = SimpleNamespace(
                stdout=iter([b"Loading checkpoint shards: 100\n", b"Start load layer 1\n"]),
                poll=lambda: None,
            )
            e.worker_ready_signal = SimpleNamespace(value=np.ones(1, dtype=np.int32))
            self.assertTrue(e.check_worker_initialize_status())
            # Failure: poll returns non-None in weight loading
            e2 = _make_engine()
            e2.worker_init_status = {}
            e2.worker_proc = SimpleNamespace(stdout=iter([]), poll=lambda: 1)
            e2.worker_ready_signal = SimpleNamespace(value=np.zeros(1, dtype=np.int32))
            self.assertFalse(e2.check_worker_initialize_status())


class TestLLMEngineRequests(unittest.TestCase):
    def test_add_requests(self):
        e = _make_engine()
        e.engine.data_processor = SimpleNamespace(
            process_request=lambda r, *a, **kw: r,
            process_request_dict=lambda d, *a, **kw: d,
        )
        with patch("fastdeploy.engine.engine.Request.from_dict", lambda d: d["_req"]):
            with self.assertRaises(EngineError):
                e.add_requests({"prompt": "x", "_req": _make_request(token_count=3000)})
            # input_ids_len > max_model_len
            with self.assertRaises(EngineError):
                e.add_requests({"prompt": "x", "_req": _make_request(token_count=2049)})
            with self.assertRaises(EngineError):
                e.add_requests({"prompt": "x", "_req": _make_request(token_count=100, min_tokens=2000)})
            with patch("fastdeploy.engine.engine.envs.FD_MAX_STOP_SEQS_NUM", 10):
                with self.assertRaises(EngineError):
                    e.add_requests({"prompt": "x", "_req": _make_request(stop_seqs_len=list(range(200)))})
                with patch("fastdeploy.engine.engine.envs.FD_STOP_SEQS_MAX_LEN", 5):
                    with self.assertRaises(EngineError):
                        e.add_requests({"prompt": "x", "_req": _make_request(stop_seqs_len=[20])})
            with self.assertRaises(EngineError):
                e.add_requests({"prompt": "x", "_req": _make_request(guided_json='{"type":"object"}')})
            # Guided decoding checker present but returns error (L338).
            e.guided_decoding_checker = SimpleNamespace(schema_format=lambda r: (r, "bad schema"))
            with self.assertRaises(EngineError):
                e.add_requests({"prompt": "x", "_req": _make_request(guided_json='{"type":"object"}')})
            e.guided_decoding_checker = None
        put_calls = []
        with (
            patch("fastdeploy.engine.engine.Request.from_dict", lambda d: _make_request()),
            patch("fastdeploy.engine.engine.asdict", lambda x: {"temperature": 0.0}),
        ):
            e.engine.scheduler = SimpleNamespace(put_requests=lambda reqs: put_calls.extend(reqs))
            e.engine.data_processor.process_request_dict = lambda d, *a, **kw: d
            sp = SimpleNamespace(temperature=0.0)
            e.add_requests({"prompt": "hi"}, sampling_params=sp)
            self.assertEqual(len(put_calls), 1)
            self.assertEqual(sp.temperature, 1e-06)

    def test_format_and_add_data(self):
        e = _make_engine()
        e.add_requests = lambda t, **kw: None
        prompts = {"prompt": "Hello"}
        uuid.UUID(e._format_and_add_data(prompts))
        self.assertEqual(prompts["max_tokens"], 2048)
        self.assertEqual(e._format_and_add_data({"prompt": "Hi", "request_id": "my-id", "max_tokens": 50}), "my-id")
        roles = [("system", "H"), ("user", "Hi"), ("assistant", "Hey")]
        ctx = {"context": [{"role": r, "utterance": u} for r, u in roles]}
        e._format_and_add_data(ctx)
        self.assertEqual(ctx["system"], "H")
        self.assertEqual(ctx["prompt"], ["Hi", "Hey"])

    def test_generate(self):
        e = _make_engine()
        e.add_requests = lambda t, **kw: None
        e.engine.check_and_free_block_tables = lambda: None
        _resp_dict = lambda **kw: {"outputs": {"text": "hi", "reasoning_content": ""}}
        _to_dict = lambda: {"text": "hi", "finished": True}
        e.engine.data_processor = SimpleNamespace(
            process_response_dict=lambda d, **kw: _resp_dict(**kw),
        )
        # stream=True: one non-finished + one finished
        results_s = [
            SimpleNamespace(finished=False, to_dict=_to_dict),
            SimpleNamespace(finished=True, to_dict=_to_dict),
        ]
        e._get_generated_tokens = lambda rid: iter(results_s)
        out_s = list(e.generate({"prompt": "x"}, stream=True))
        self.assertEqual(len(out_s), 2)
        self.assertEqual(out_s[1]["outputs"]["text"], "")

        # stream=False: offline path
        e._get_generated_tokens = lambda rid: iter([SimpleNamespace(finished=True, to_dict=_to_dict)])
        out = list(e.generate({"prompt": "x"}, stream=False))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["outputs"]["text"], "hi")
        self.assertEqual(e._get_generated_result(), [])
        # Error path
        e.add_requests = lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad"))
        with self.assertRaises(EngineError):
            list(e.generate({"prompt": "x"}, stream=False))


class TestLLMEngineUtils(unittest.TestCase):
    def test_has_guided_input(self):
        e = _make_engine()
        fields = "guided_json,guided_regex,guided_choice,structural_tag,guided_grammar,guided_json_object".split(",")
        self.assertFalse(e._has_guided_input(SimpleNamespace(**{f: None for f in fields})))
        for field in fields:
            kw = {f: None for f in fields}
            kw[field] = "value"
            self.assertTrue(e._has_guided_input(SimpleNamespace(**kw)))

    def test_setting_environ_variables(self):
        e = _make_engine()
        result = e._setting_environ_variables()
        self.assertIn("OMP_NUM_THREADS=", result)
        self.assertIn("NCCL_ALGO=Ring", result)
        self.assertNotIn("FLAGS_use_pd_disaggregation", result)
        with patch("fastdeploy.engine.engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False):
            e.cfg.scheduler_config.splitwise_role = "prefill"
            self.assertIn("FLAGS_use_pd_disaggregation", e._setting_environ_variables())
        with patch("fastdeploy.engine.engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True):
            e.cfg.scheduler_config.splitwise_role = "prefill"
            self.assertIn("FLAGS_use_pd_disaggregation_per_chunk", e._setting_environ_variables())

    def test_health_and_readiness(self):
        e = _make_engine()
        e.worker_ready_signal = SimpleNamespace(value=np.zeros(1, dtype=np.int32))
        self.assertFalse(e._worker_processes_ready())
        e.worker_ready_signal = SimpleNamespace(value=np.ones(1, dtype=np.int32))
        self.assertTrue(e._worker_processes_ready())
        e.cfg.worker_num_per_node = 3
        e.worker_ready_signal = SimpleNamespace(value=np.array([1, 1, 0], dtype=np.int32))
        self.assertFalse(e._worker_processes_ready())
        e.engine.worker_healthy_live_signal = SimpleNamespace(value=np.array([0.0]))
        self.assertTrue(e.check_health()[0])
        e.engine.worker_healthy_live_signal = SimpleNamespace(value=np.array([time.time()]))
        self.assertTrue(e.check_health()[0])
        e.engine.worker_healthy_live_signal = SimpleNamespace(value=np.array([time.time() - 60]))
        healthy, msg = e.check_health(time_interval_threashold=30)
        self.assertFalse(healthy)
        self.assertIn("Not Healthy", msg)


class TestLLMEngineStopProfile(unittest.TestCase):
    """测试 LLMEngine._stop_profile 方法"""

    def test_stop_profile_logs_worker_traceback_and_returns_false(self):
        """测试 worker 进程失败时，_stop_profile 打印 traceback 并返回 False"""
        eng = object.__new__(LLMEngine)
        eng.do_profile = 1
        eng.get_profile_block_num_signal = type("Sig", (), {"value": np.array([0])})()
        eng.worker_proc = Mock(poll=lambda: 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            paddle_log_dir = os.path.join(temp_dir, "paddle")
            os.makedirs(paddle_log_dir)
            worker_log = os.path.join(paddle_log_dir, "workerlog.0")
            with open(worker_log, "w", encoding="utf-8") as fp:
                fp.write(
                    "Traceback (most recent call last):\n"
                    "ValueError: The total number of blocks cannot be less than zero.\n"
                )

            with (
                patch("fastdeploy.engine.engine.time.sleep", lambda *_: None),
                patch("fastdeploy.engine.engine.envs.FD_LOG_DIR", temp_dir),
                patch("fastdeploy.engine.engine.console_logger.error") as mock_error,
            ):
                result = eng._stop_profile()

        self.assertFalse(result)
        error_messages = [call.args[0] for call in mock_error.call_args_list]
        self.assertTrue(any("Traceback (most recent call last):" in msg for msg in error_messages))
        self.assertTrue(any("The total number of blocks cannot be less than zero" in msg for msg in error_messages))

    def test_stop_profile_returns_true_on_success(self):
        """测试 _stop_profile 正常完成时返回 True"""
        eng = object.__new__(LLMEngine)
        eng.do_profile = 1
        eng.get_profile_block_num_signal = type("Sig", (), {"value": np.array([100])})()
        eng.worker_proc = Mock(poll=lambda: None)
        eng.ipc_signal_suffix = "_test"
        eng.cfg = SimpleNamespace(
            parallel_config=SimpleNamespace(device_ids="0"),
            scheduler_config=SimpleNamespace(splitwise_role="decode"),
            cache_config=Mock(enable_prefix_caching=False, reset=Mock()),
        )
        eng.engine = SimpleNamespace(
            start_cache_service=lambda *_: None,
            resource_manager=Mock(reset_cache_config=Mock()),
        )
        eng.cache_manager_processes = None

        result = eng._stop_profile()

        self.assertTrue(result)


class TestLLMEngineStartProfile(unittest.TestCase):
    """测试 LLMEngine.start 方法中的错误处理"""

    class _Sig:
        def __init__(self, val):
            self.value = np.array([val])

    def test_start_returns_false_when_profile_worker_dies(self):
        """测试当 profile worker 失败时，start 返回 False"""
        eng = object.__new__(LLMEngine)
        eng.is_started = False
        eng.api_server_pid = None
        eng.do_profile = 1
        port = int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778"))
        eng.cfg = SimpleNamespace(
            parallel_config=SimpleNamespace(engine_worker_queue_port=[port], device_ids="0"),
            scheduler_config=SimpleNamespace(splitwise_role="mixed", max_num_seqs=8),
            cache_config=SimpleNamespace(
                enable_prefix_caching=True,
                block_size=64,
                num_gpu_blocks_override=None,
                total_block_num=0,
                num_cpu_blocks=0,
            ),
            model_config=SimpleNamespace(max_model_len=128),
        )
        eng._init_worker_signals = lambda: setattr(eng, "loaded_model_signal", self._Sig(1))
        eng.launch_components = lambda: None
        eng.worker_proc = None
        eng.engine = SimpleNamespace(
            start=lambda: None,
            create_data_processor=lambda: setattr(eng.engine, "data_processor", object()),
            data_processor=object(),
        )
        eng._start_worker_service = lambda: Mock(stdout=Mock(), poll=lambda: 1)
        eng.check_worker_initialize_status = lambda: False
        eng._stop_profile = lambda: False

        with patch("fastdeploy.engine.engine.time.sleep", lambda *_: None):
            result = eng.start(api_server_pid=None)

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
