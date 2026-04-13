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

import asyncio
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
import paddle
from e2e.utils.serving_utils import clean_ports

if not hasattr(paddle, "compat"):

    class _PaddleCompat:
        @staticmethod
        def enable_torch_proxy(scope=None):
            return None

    paddle.compat = _PaddleCompat()

from fastdeploy.cache_manager.cache_data import CacheStatus
from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.common_engine import (
    EngineService,
    _format_worker_launch_failure_message,
    _read_latest_worker_traceback,
)
from fastdeploy.engine.request import (
    ControlRequest,
    ControlResponse,
    Request,
    RequestOutput,
    RequestStatus,
    RequestType,
)
from fastdeploy.utils import EngineError

MODEL_NAME = os.getenv("MODEL_PATH", "/workspace/wenlei/models") + "/ERNIE-4.5-0.3B-Paddle"

_STUB_PRETRAINED_CONFIG = {
    "architectures": ["StubForCausalLM"],
    "hidden_size": 64,
    "num_attention_heads": 8,
    "num_hidden_layers": 2,
    "vocab_size": 1000,
}


def _fake_model_post_init(self):
    self.is_unified_ckpt = False
    self.runner_type = "generate"
    self.convert_type = "auto"
    self.supported_tasks = []
    if not hasattr(self, "enable_mm"):
        self.enable_mm = False


def _create_engine_config(args):
    with patch(
        "fastdeploy.config.PretrainedConfig.get_config_dict",
        return_value=(_STUB_PRETRAINED_CONFIG, None),
    ):
        with patch("fastdeploy.config.ModelConfig._post_init", _fake_model_post_init):
            return args.create_engine_config()


class TestCommonEngine(unittest.TestCase):
    """Test case for EngineService functionality (lines 1215-1664)"""

    @classmethod
    def setUpClass(cls):
        """Set up EngineService for testing"""
        try:
            # Clean ports before starting the engine
            print("Pre-test port cleanup...")
            clean_ports()

            # Create engine args for testing
            engine_args = EngineArgs(
                model=MODEL_NAME,
                max_model_len=8192,
                tensor_parallel_size=1,
                engine_worker_queue_port=int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")),
                cache_queue_port=int(os.getenv("FD_CACHE_QUEUE_PORT", "6779")),
            )

            # Create and start the engine service
            cls.cfg = _create_engine_config(engine_args)

            with (
                patch(
                    "fastdeploy.engine.common_engine.EngineWorkerQueue",
                    TestCommonEngineAdditionalCoverage._make_full_dummy_q_cls(),
                ),
                patch("fastdeploy.engine.common_engine.EngineCacheQueue"),
            ):
                cls.engine = EngineService(cls.cfg, start_queue=False, use_async_llm=True)

            cls.engine.running = True
            cls.engine.ipc_signal_suffix = cls.cfg.parallel_config.local_engine_worker_queue_port

            cls.engine.worker_ready_signal = TestCommonEngineAdditionalCoverage._Sig(1)
            cls.engine.loaded_model_signal = TestCommonEngineAdditionalCoverage._Sig(1)
            cls.engine.worker_healthy_live_signal = TestCommonEngineAdditionalCoverage._Sig(int(time.time()))
            cls.engine.worker_proc = Mock(pid=12345)

        except Exception as e:
            print(f"Setting up EngineService failed: {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        if hasattr(cls, "engine") and cls.engine is not None:
            try:
                if hasattr(cls.engine, "resource_manager") and hasattr(cls.engine.resource_manager, "cache_manager"):
                    cache_manager = cls.engine.resource_manager.cache_manager
                    if not hasattr(cache_manager, "shm_cache_task_flag_broadcast"):
                        cache_manager.shm_cache_task_flag_broadcast = Mock(clear=Mock())
                    if not hasattr(cache_manager, "cache_ready_signal"):
                        cache_manager.cache_ready_signal = Mock(clear=Mock())
                if getattr(cls.engine, "cache_manager_processes", None) is None:
                    cls.engine.cache_manager_processes = []
                if hasattr(cls.engine, "_finalizer"):
                    cls.engine._finalizer.detach()
                cls.engine.worker_proc = None
                cls.engine._exit_sub_services()
                print("Engine cleanup completed")
            except Exception as e:
                print(f"Error during engine cleanup: {e}")

    def setUp(self):
        """Set up before each test method"""
        print(f"Starting test: {self._testMethodName}")

    def tearDown(self):
        """Clean up after each test method"""
        print(f"Completed test: {self._testMethodName}")

    def test_engine_has_expected_attributes(self):
        """Consolidated lightweight attribute/callable checks."""
        expected_methods = [
            "_exit_sub_services",
            "_start_worker_service",
            "_stop_profile",
            "launch_components",
            "check_worker_initialize_status",
        ]
        for name in expected_methods:
            self.assertTrue(hasattr(self.engine, name))
            self.assertTrue(callable(getattr(self.engine, name)))

        if hasattr(self.engine, "worker_proc"):
            self.assertIsNotNone(self.engine.worker_proc)

        if hasattr(self.engine, "scheduler"):
            self.assertIsNotNone(self.engine.scheduler)

        if hasattr(self.engine, "worker_init_status"):
            self.assertIsInstance(self.engine.worker_init_status, dict)

        self.assertTrue(hasattr(self.engine, "do_profile"))
        self.assertTrue(self.engine.running)

    def test_worker_processes_ready(self):
        """Test _worker_processes_ready method (lines 1292-1299)"""
        # Test with real engine that should have worker_ready_signal
        if hasattr(self.engine, "worker_ready_signal"):
            result = self.engine._worker_processes_ready()
            # Result should be boolean
            self.assertIsInstance(result, bool)
        else:
            self.skipTest("worker_ready_signal not available")

    def test_init_worker_signals(self):
        """Test _init_worker_signals method (lines 1301-1361)"""
        # Since engine is already started, signals should be initialized
        self.assertTrue(hasattr(self.engine, "worker_ready_signal"))
        self.assertTrue(hasattr(self.engine, "loaded_model_signal"))

        # Test that signals have expected properties
        if hasattr(self.engine, "worker_ready_signal"):
            self.assertIsNotNone(self.engine.worker_ready_signal)

        if hasattr(self.engine, "loaded_model_signal"):
            self.assertIsNotNone(self.engine.loaded_model_signal)

    def test_setting_environ_variables(self):
        """Test _setting_environ_variables method (lines 1362-1408)"""
        result = self.engine._setting_environ_variables()

        # Check that result is a string and contains expected variables
        self.assertIsInstance(result, str)
        self.assertIn("ENABLE_FASTDEPLOY_LOAD_MODEL_CONCURRENCY=0", result)
        self.assertIn("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python", result)
        self.assertIn("FLAGS_use_append_attn=1", result)
        self.assertIn("NCCL_ALGO=Ring", result)

    def test_check_health(self):
        """Test check_health method (lines 1533-1544)"""
        if hasattr(self.engine, "worker_healthy_live_signal"):
            is_healthy, message = self.engine.check_health(time_interval_threashold=30)

            # Should return tuple of (bool, str)
            self.assertIsInstance(is_healthy, bool)
            self.assertIsInstance(message, str)
        else:
            self.skipTest("worker_healthy_live_signal not available")

    def test_engine_started_successfully(self):
        """Test that engine started successfully and has expected state"""
        # Verify engine is running
        self.assertTrue(self.engine.running)

        # Verify data processor was created
        if hasattr(self.engine, "data_processor"):
            self.assertIsNotNone(self.engine.data_processor)

        # Verify IPC signal suffix is set
        if hasattr(self.engine, "ipc_signal_suffix"):
            self.assertIsNotNone(self.engine.ipc_signal_suffix)


if __name__ == "__main__":
    unittest.main()


class TestCommonEngineAdditionalCoverage(unittest.TestCase):
    """Additional unit tests focusing on branch coverage for common_engine.py

    These tests heavily mock subprocess/threading/IPC to avoid starting real workers
    and to drive specific code paths that were previously uncovered.
    """

    def setUp(self):
        cache_queue_patcher = patch("fastdeploy.engine.common_engine.EngineCacheQueue")
        cache_queue_patcher.start()
        self.addCleanup(cache_queue_patcher.stop)

    class _Sig:
        def __init__(self, v=0):
            self.value = np.array([v], dtype=np.int32)

        def clear(self):
            pass

    @staticmethod
    @staticmethod
    def _make_full_dummy_q_cls():
        class DummyQ:
            def __init__(self, *a, **k):
                self.available_prefill_instances = type("X", (), {"put": lambda *_: None})()

            def get_server_port(self):
                return 0

            def cleanup(self):
                pass

            def num_tasks(self):
                return 0

            def num_cache_infos(self):
                return 0

            def disaggregate_queue_empty(self):
                return True

            def get_disaggregated_tasks(self):
                return []

        return DummyQ

    @staticmethod
    def _make_dummy_executor(eng):
        class DummyExecutor:
            def __init__(self, max_workers=None):
                pass

            def submit(self, fn):
                try:
                    fn()
                finally:
                    eng.running = False

        return DummyExecutor

    def _make_mixed_engine(self):
        cfg = self._make_cfg(splitwise_role="mixed", num_gpu_blocks_override=4)
        return self._make_engine(cfg)

    def _setup_v1_engine(self, eng):
        eng.running = True
        eng.is_paused = False
        eng._pause_cond = threading.Condition()
        self.addCleanup(lambda: setattr(eng, "running", False))

    @staticmethod
    def _make_v1_decode_rm(eng, schedule_result, with_add_request=False):
        class DummyRM:
            def __init__(self):
                self.abort_req_ids_set = set()
                self.waiting = []
                self.real_bsz = 1
                if with_add_request:
                    self.add_request = Mock()

            def available_batch(self):
                return 1

            def schedule(self):
                eng.running = False
                return schedule_result

            def get_real_bsz(self):
                return self.real_bsz

        return DummyRM()

    @staticmethod
    def _make_v1_prefill_continuous_rm(eng, waiting_async_result=False):
        class DummyRM:
            def __init__(self):
                self.abort_req_ids_set = set()
                self.waiting = []
                self.real_bsz = 1
                self.add_request_in_p = Mock()
                self.pre_recycle_resource = Mock()

            def available_batch(self):
                return 1

            def apply_async_preprocess(self, _task):
                return None

            def preallocate_resource_in_p(self, _task):
                return True

            def waiting_async_process(self, _task):
                return waiting_async_result

            def schedule(self):
                eng.running = False
                return ([], [])

            def get_real_bsz(self):
                return self.real_bsz

        return DummyRM()

    @staticmethod
    def _make_insert_tasks_rm(n=1):
        class DummyRM:
            def __init__(self):
                self.stop_flags = np.array([1] * n, dtype=np.int32)
                self.real_bsz = 1

            def check_and_free_block_tables(self):
                pass

            def allocate_resources_for_new_tasks(self, tasks):
                return tasks

        return DummyRM()

    @staticmethod
    def _make_scheduler_with_output(eng, token_ids, decode_type, finished, fmt="dict", include_raw=False):
        class DummyOutput:
            def __init__(self):
                self.token_ids = token_ids
                self.decode_type = decode_type
                self.tool_calls = None

        output = RequestOutput(
            request_id="rid",
            outputs=DummyOutput(),
            finished=finished,
            metrics=Mock(),
        )

        def get_results():
            eng.running = False
            if fmt == "list":
                return [[output]]
            if include_raw:
                return {"rid": [output, "raw"]}
            return {"rid": [output]}

        eng.scheduler = Mock(get_results=get_results)
        return output

    @staticmethod
    def _make_ctrl_queue(name, payload, payload_wrapped=True):
        class DummyQueue:
            def __init__(self):
                self.name = name

            async def get(self, timeout=None):
                if payload_wrapped:
                    return Mock(payload=payload)
                return payload

        return DummyQueue()

    @staticmethod
    def _make_dummy_recv(eng, payload=None, error=None):
        class DummyRecv:
            def receive_json_once(self, block):
                eng.running = False
                return error, payload

            def receive_pyobj_once(self, block):
                eng.running = False
                return error, payload

            def close(self):
                pass

        return DummyRecv()

    @staticmethod
    def _make_zmq_server_cls():
        class DummyServer:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def recv_result_handle(self):
                return None

        return DummyServer

    @staticmethod
    def _make_zmq_thread_cls(counter=None):
        class DummyThread:
            def __init__(self, target=None, daemon=None):
                self.target = target
                self.daemon = daemon

            def start(self):
                if counter is not None:
                    counter["threads"] += 1

        return DummyThread

    @staticmethod
    def _make_simple_dummy_q_cls():
        class DummyQ:
            def __init__(self, *a, **k):
                pass

        return DummyQ

    @staticmethod
    def _make_mm_stub_module():
        stub_module = types.ModuleType("fastdeploy.model_executor.ops.gpu")
        stub_module.get_mm_split_fuse = lambda *args, **kwargs: (
            np.array([1], dtype="int64"),
            np.array([4], dtype="int64"),
        )
        return stub_module

    class _DummyPbar:
        def __init__(self):
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, delta=0, *args, **kwargs):
            try:
                self.n += int(delta)
            except Exception:
                self.n = 0

        def refresh(self):
            pass

    @staticmethod
    def _detach_finalizer(engine):
        if hasattr(engine, "_finalizer"):
            try:
                engine._finalizer.detach()
            except Exception:
                pass

    def _make_cfg(self, **kwargs):
        # If DP > 1, we must provide enough engine_worker_queue_port for each dp index
        dp = kwargs.get("data_parallel_size", 1)
        nnode = len(kwargs.get("ips", ["127.0.0.1"]))
        engine_worker_queue_port = int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778"))
        cache_queue_port = int(os.getenv("FD_CACHE_QUEUE_PORT", "6779"))
        if dp and dp > 1:
            engine_worker_queue_port = [engine_worker_queue_port + 21 + i for i in range(dp // nnode)]
            cache_queue_port = [cache_queue_port + 21 + i for i in range(dp // nnode)]

        if kwargs.get("num_gpu_blocks_override") is not None and "kv_cache_ratio" not in kwargs:
            kwargs["kv_cache_ratio"] = 1

        args = EngineArgs(
            model=MODEL_NAME,
            max_model_len=128,
            tensor_parallel_size=1,
            # give unique ports to avoid collision with other tests
            engine_worker_queue_port=engine_worker_queue_port,
            cache_queue_port=cache_queue_port,
            enable_prefix_caching=True,
            **kwargs,
        )
        # Keep batch tokens small to satisfy FDConfig checks:
        # max_num_batched_tokens <= max_model_len * max_num_seqs
        if getattr(args, "max_num_batched_tokens", None) is None:
            args.max_num_batched_tokens = 128
        # Always enable chunked prefill in tests to avoid another strict check
        args.enable_chunked_prefill = True

        return _create_engine_config(args)

    def _stub_processor(self):
        class _Tok:
            def __init__(self):
                self.vocab = {"</think>": 42, "\n": 10, "<|IMAGE_PLACEHOLDER|>": 9}

            def get_vocab(self):
                return self.vocab

        class _Proc:
            def __init__(self):
                self.tokenizer = _Tok()
                self.eos_token_id_len = 1
                self.pad_token_id = 0

        return _Proc()

    def _make_engine(self, cfg):
        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", self._make_full_dummy_q_cls()):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        return eng

    def test_start_prefill_branch_cache_manager_and_worker_dead(self):
        """Cover lines 184-185, 194-197, 221, 226-227 in start()."""
        # For prefill + local scheduler the core code now requires a router.
        # Also, with the newer CacheConfig semantics we must ensure that
        # prefill_kvcache_block_num (num_gpu_blocks_override * kv_cache_ratio)
        # is >= max_block_num_per_seq; use 3 blocks so that with the default
        # kv_cache_ratio=0.75 we still satisfy the assertion.
        with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0):
            cfg = self._make_cfg(
                splitwise_role="prefill",
                num_gpu_blocks_override=4,
                router="0.0.0.0:30000",
                kv_cache_ratio=1,
            )

        # Patch EngineWorkerQueue before EngineService ctor to avoid real IPC
        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", self._make_simple_dummy_q_cls()):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        # Patch heavy pieces
        eng.create_data_processor = lambda: setattr(eng, "data_processor", self._stub_processor())
        eng._process_splitwise_task = lambda: None
        eng._schedule_request_to_worker = lambda: None
        eng._schedule_request_to_worker_v1 = lambda: None

        started_cache = {}

        def fake_start_cache(device_ids, suffix):
            started_cache["called"] = True
            # return a list to mimic processes
            return [object()]

        eng.start_cache_service = fake_start_cache

        # Signals: make loaded_model_signal ready immediately; include launched_cache_manager_signal
        def fake_init_signals():
            eng.worker_ready_signal = self._Sig(0)
            eng.loaded_model_signal = self._Sig(1)  # ready -> skip wait loop
            eng.launched_cache_manager_signal = self._Sig(0)

        eng._init_worker_signals = fake_init_signals

        # Worker start stub and initialization status -> False to trigger error path
        eng._start_worker_service = lambda: Mock(stdout=Mock(), poll=lambda: None)
        eng.check_worker_initialize_status = lambda: False

        with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
            # Avoid starting token processor loop
            eng.token_processor.run = lambda: None
            ok = eng.start(async_llm_pid=12345)

        # start() returns False on failure
        self.assertFalse(ok)
        # cache manager started before workers (lines 184-185)
        self.assertTrue(started_cache.get("called", False))
        # avoid atexit finalizer
        self._detach_finalizer(eng)

    def test_start_mixed_branch_cache_after_load_and_zmq(self):
        """Cover lines 215-217 and 231 in start()."""
        cfg = self._make_cfg(splitwise_role="mixed", num_gpu_blocks_override=4)

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", self._make_simple_dummy_q_cls()):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        eng.create_data_processor = lambda: setattr(eng, "data_processor", self._stub_processor())
        eng._process_splitwise_task = lambda: None
        eng._schedule_request_to_worker = lambda: None
        eng._schedule_request_to_worker_v1 = lambda: None

        started_cache = {}

        def fake_start_cache(device_ids, suffix):
            started_cache["called"] = True
            return [object()]

        eng.start_cache_service = fake_start_cache

        def fake_init_signals():
            eng.worker_ready_signal = self._Sig(0)
            eng.loaded_model_signal = self._Sig(1)
            eng.launched_cache_manager_signal = self._Sig(0)

        eng._init_worker_signals = fake_init_signals

        eng._start_worker_service = lambda: Mock(stdout=Mock(), poll=lambda: None)
        eng.check_worker_initialize_status = lambda: True
        eng.do_profile = 0
        eng.cfg.cache_config.enable_prefix_caching = True

        zmq_called = {}
        eng.start_zmq_service = lambda pid: zmq_called.setdefault("pid", pid)

        with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
            eng.token_processor.run = lambda: None
            eng.start(async_llm_pid=8888)

        self.assertTrue(started_cache.get("called", False))  # lines 215-217
        self.assertEqual(zmq_called.get("pid"), 8888)  # line 231
        self._detach_finalizer(eng)

    def test_update_requests_chunk_size_assigns_chunks(self):
        eng = self._make_mixed_engine()
        eng.partial_chunked_tokens = [0, 32, 16, 8]
        eng.cfg.scheduler_config.max_num_batched_tokens = 32
        eng.cfg.cache_config.block_size = 8
        eng.cfg.cache_config.enable_chunked_prefill = True

        requests = [
            Request(request_id="r0", prompt_token_ids=[1] * 24, prompt_token_ids_len=24),
            Request(request_id="r1", prompt_token_ids=[1] * 8, prompt_token_ids_len=8),
        ]

        eng.update_requests_chunk_size(requests)

        for req in requests:
            chunk_info = req.get("prefill_chunk_info")
            self.assertIsInstance(chunk_info, list)
            self.assertGreater(len(chunk_info), 0)
            self.assertEqual(sum(chunk_info), req.prompt_token_ids_len)
        self._detach_finalizer(eng)

    def test_update_mm_requests_chunk_size_with_stub_fuse(self):
        eng = self._make_mixed_engine()
        eng.cfg.cache_config.enable_chunked_prefill = True
        eng.partial_chunked_tokens = [0, 16]
        eng.data_processor = type("DP", (), {"image_patch_id": 9})()

        inputs = {
            "input_ids": np.array([9, 1, 2, 3], dtype="int64"),
            "token_type_ids": np.array([0, 0, 0, 0], dtype="int64"),
            "image_type_ids": np.array([1], dtype="int32"),
            "grid_thw": np.array([[1, 2, 2]], dtype="int64"),
            "images": np.ones((4,), dtype="uint8"),
            "position_ids": np.array([0, 1, 2, 3], dtype="int64"),
        }
        req = Request(request_id="mm0", multimodal_inputs=inputs)

        with patch.dict("sys.modules", {"fastdeploy.model_executor.ops.gpu": self._make_mm_stub_module()}):
            eng.update_mm_requests_chunk_size([req])

        chunk_info = req.get("prefill_chunk_info")
        self.assertIsInstance(chunk_info, list)
        self.assertEqual(len(chunk_info), 1)
        self.assertEqual(chunk_info[0]["input_ids"].tolist(), inputs["input_ids"].tolist())
        self.assertIsNotNone(chunk_info[0]["images"])
        self._detach_finalizer(eng)

    def test_send_error_response_routes(self):
        eng = self._make_mixed_engine()
        eng.send_response_server = Mock()

        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False),
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", False),
        ):
            eng._send_error_response("rid0", "boom", error_code=400)
            eng.send_response_server.send_response.assert_called_with("rid0", [ANY])

        eng.send_response_server.reset_mock()
        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False),
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", True),
        ):
            eng._send_error_response("rid2", "boom", error_code=400)
            eng.send_response_server.send_response.assert_called_with(None, [ANY], worker_pid=None)

        eng.send_response_server.reset_mock()
        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
            eng._send_error_response("rid1", "boom", error_code=500)
            eng.send_response_server.send_response.assert_called_with(None, [ANY])

        self._detach_finalizer(eng)

    def test_decode_token_with_return_text(self):
        eng = self._make_mixed_engine()

        class DummyProcessor:
            def __init__(self):
                self.decode_status = {"rid": (0, 2)}

            def ids2tokens(self, token_ids, req_id):
                return "hi", [101, 102], None

        eng.data_processor = DummyProcessor()

        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True):
            delta, token_ids = eng._decode_token([101, 102], "rid", is_end=True)

        self.assertEqual(delta, "hi")
        self.assertEqual(token_ids, [101, 102])
        self.assertNotIn("rid", eng.data_processor.decode_status)
        self._detach_finalizer(eng)

    def test_decode_token_without_return_text(self):
        eng = self._make_mixed_engine()

        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", False):
            delta, token_ids = eng._decode_token([9, 10], "rid", is_end=False)

        self.assertEqual(delta, "")
        self.assertEqual(token_ids, [9, 10])
        self._detach_finalizer(eng)

    def test_decode_token_return_text_empty_delta(self):
        eng = self._make_mixed_engine()

        class DummyProcessor:
            def __init__(self):
                self.decode_status = {"rid": (0, 1)}

            def ids2tokens(self, token_ids, req_id):
                return "", [7], None

        eng.data_processor = DummyProcessor()

        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True):
            delta, token_ids = eng._decode_token([7], "rid", is_end=True)

        self.assertEqual(delta, "")
        self.assertEqual(token_ids, [])
        self.assertNotIn("rid", eng.data_processor.decode_status)
        self._detach_finalizer(eng)

    def test_clear_data_success_and_failure(self):
        eng = self._make_mixed_engine()
        eng.token_processor = Mock()
        eng.engine_worker_queue = Mock()
        eng.send_response_server = Mock(req_dict={"a": 1})
        eng.recv_request_server = Mock(req_dict={"b": 2})

        self.assertTrue(eng.clear_data())
        self.assertEqual(eng.send_response_server.req_dict, {})
        self.assertEqual(eng.recv_request_server.req_dict, {})

        eng.token_processor.clear_data.side_effect = RuntimeError("boom")
        self.assertFalse(eng.clear_data())
        self._detach_finalizer(eng)

    def test_insert_prefilled_requests_recycles_and_dispatches(self):
        cfg = self._make_cfg(splitwise_role="decode", num_gpu_blocks_override=4, router="0.0.0.0:30000")
        cfg.speculative_config.method = "mtp"
        eng = self._make_engine(cfg)

        class DummyRM:
            def __init__(self):
                self.req_dict = {"r0": 0, "r1": 1, "r2": 2}
                self.tasks_list = [
                    Request(request_id="r0", prompt_token_ids=[0], prompt_token_ids_len=1),
                    Request(request_id="r1", prompt_token_ids=[0], prompt_token_ids_len=1),
                    Request(request_id="r2", prompt_token_ids=[0], prompt_token_ids_len=1),
                ]
                self.stop_flags = np.array([False, False, False])
                self.real_bsz = 1
                self.recycled = []

            def _recycle_block_tables(self, req):
                self.recycled.append(req.request_id)

        eng.resource_manager = DummyRM()
        eng.token_processor = Mock()
        eng.token_processor.tokens_counter = {"r0": 1, "r1": 1}
        eng.scheduler = Mock()
        eng.engine_worker_queue = Mock()

        class DummyOutputs:
            def __init__(self, token_ids, draft_token_ids=None):
                self.token_ids = token_ids
                self.draft_token_ids = draft_token_ids or []
                self.tool_calls = None

        outputs_empty = DummyOutputs([])
        outputs_error = DummyOutputs([1], [9])
        outputs_ok = DummyOutputs([2], [8])
        req_out_empty = RequestOutput(request_id="r0", outputs=outputs_empty, metrics=Mock(), num_cached_tokens=0)
        req_out_error = RequestOutput(
            request_id="r1",
            outputs=outputs_error,
            metrics=Mock(),
            num_cached_tokens=0,
            error_code=500,
            error_msg="bad",
        )
        req_out_ok = RequestOutput(request_id="r2", outputs=outputs_ok, metrics=Mock(), num_cached_tokens=3)

        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
            eng._insert_prefilled_requests([req_out_empty, req_out_error, req_out_ok])

        self.assertIn("r0", eng.resource_manager.recycled)
        self.assertIn("r1", eng.resource_manager.recycled)
        self.assertIn("r2", eng.token_processor.tokens_counter)
        eng.engine_worker_queue.put_tasks.assert_called()
        self._detach_finalizer(eng)

    def test_task_finished_helpers(self):
        eng = self._make_mixed_engine()

        class DummyRM:
            def __init__(self):
                self.stop_flags = np.array([True, False, True])

        eng.resource_manager = DummyRM()

        self.assertTrue(eng.task_is_finished(0))
        self.assertFalse(eng.task_is_finished(1))
        self.assertFalse(eng.all_tasks_finished())
        eng.resource_manager.stop_flags = np.array([True, True])
        self.assertTrue(eng.all_tasks_finished())
        self._detach_finalizer(eng)

    def test_start_worker_queue_service_with_servers(self):
        cfg = self._make_cfg(splitwise_role="mixed", num_gpu_blocks_override=4)

        class DummyQueue:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs

            def get_server_port(self):
                return 12345

            def cleanup(self):
                pass

        class DummyCacheQueue(DummyQueue):
            pass

        eng = self._make_engine(cfg)
        with (
            patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQueue),
            patch("fastdeploy.engine.common_engine.EngineCacheQueue", DummyCacheQueue),
            patch("fastdeploy.engine.common_engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", False),
        ):
            eng.start_worker_queue_service(start_queue=True)

        self.assertEqual(eng.cfg.parallel_config.local_engine_worker_queue_port, 12345)
        self._detach_finalizer(eng)

    def test_init_worker_monitor_signals_creates_ipc(self):
        cfg = self._make_cfg(splitwise_role="mixed", num_gpu_blocks_override=4)

        created = []

        class DummySignal:
            def __init__(self, name, array, dtype, suffix, create):
                self.name = name
                self.array = array
                self.dtype = dtype
                self.suffix = suffix
                self.create = create
                created.append(name)

        with (
            patch("fastdeploy.engine.common_engine.EngineWorkerQueue", self._make_full_dummy_q_cls()),
            patch("fastdeploy.engine.common_engine.IPCSignal", DummySignal),
        ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        self.assertIn("exist_task_signal", created)
        self.assertIn("worker_healthy_live_signal", created)
        self.assertTrue(hasattr(eng, "kv_cache_status_signal"))
        self._detach_finalizer(eng)

    def test_init_worker_signals_with_profile(self):
        eng = self._make_mixed_engine()
        eng.ipc_signal_suffix = 7777
        eng.do_profile = 1

        class DummySignal:
            def __init__(self, *args, **kwargs):
                self.value = np.zeros([1], dtype=np.int32)

            def clear(self):
                pass

        with patch("fastdeploy.engine.common_engine.IPCSignal", DummySignal):
            eng._init_worker_signals()

        self.assertIsNotNone(eng.worker_ready_signal)
        self.assertIsNotNone(eng.loaded_model_signal)
        self.assertTrue(hasattr(eng, "get_profile_block_num_signal"))
        self._detach_finalizer(eng)

    def test_worker_processes_ready_and_health(self):
        eng = self._make_mixed_engine()
        eng.worker_ready_signal = type("Sig", (), {"value": np.array([1], dtype=np.int32)})()
        eng.cfg.worker_num_per_node = 1
        self.assertTrue(eng._worker_processes_ready())

        eng.worker_healthy_live_signal = type("Sig", (), {"value": np.array([time.time() - 100])})()
        is_healthy, message = eng.check_health(time_interval_threashold=1)
        self.assertFalse(is_healthy)
        self.assertIn("Not Healthy", message)
        self._detach_finalizer(eng)

    def test_stop_profile_resets_cache(self):
        cfg = self._make_cfg(splitwise_role="prefill", num_gpu_blocks_override=4, router="0.0.0.0:30000")
        eng = self._make_engine(cfg)
        eng.ipc_signal_suffix = 9999
        eng.do_profile = 1
        eng.get_profile_block_num_signal = type("Sig", (), {"value": np.array([8])})()
        eng.resource_manager = Mock()
        eng.start_cache_service = Mock(return_value=[Mock()])

        eng._stop_profile()

        self.assertEqual(eng.do_profile, 0)
        eng.resource_manager.reset_cache_config.assert_called_once()
        self.assertIsNotNone(eng.cache_manager_processes)
        self._detach_finalizer(eng)

    def test_start_worker_queue_service_with_shm_address(self):
        cfg = self._make_cfg(splitwise_role="mixed", num_gpu_blocks_override=4)

        class DummyQueue:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs

            def get_server_port(self):
                return 22222

            def cleanup(self):
                pass

        class DummyCacheQueue(DummyQueue):
            pass

        eng = self._make_engine(cfg)
        with (
            patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQueue),
            patch("fastdeploy.engine.common_engine.EngineCacheQueue", DummyCacheQueue),
            patch("fastdeploy.engine.common_engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", True),
        ):
            eng.start_worker_queue_service(start_queue=True)

        address = eng.engine_worker_queue.kwargs["address"]
        self.assertTrue(isinstance(address, str))
        self.assertIn("/dev/shm/fd_task_queue_", address)
        self._detach_finalizer(eng)

    def test_start_worker_service_builds_command(self):
        eng = self._make_mixed_engine()
        eng.do_profile = 0
        eng.data_processor = type(
            "DP",
            (),
            {
                "tokenizer": type(
                    "Tok",
                    (),
                    {
                        "vocab": {"</think>": 5, "<|IMAGE_PLACEHOLDER|>": 9, "\n": 10},
                        "get_vocab": lambda self: self.vocab,
                    },
                )(),
                "eos_token_id_len": 1,
                "pad_token_id": 0,
            },
        )()

        with patch("fastdeploy.engine.common_engine.subprocess.Popen") as popen_mock:
            popen_mock.return_value = Mock()
            proc = eng._start_worker_service()

        popen_mock.assert_called_once()
        self.assertIs(proc, popen_mock.return_value)
        self._detach_finalizer(eng)

    def test_exit_sub_services_cleans_up(self):
        eng = self._make_mixed_engine()
        eng.use_async_llm = True
        eng.worker_proc = Mock(pid=1234)
        eng.cache_manager_processes = [Mock(pid=2345)]
        eng.cache_task_queue = Mock(cleanup=Mock())
        eng.resource_manager = Mock(
            cache_manager=Mock(
                shm_cache_task_flag_broadcast=Mock(clear=Mock()),
                cache_ready_signal=Mock(clear=Mock()),
            )
        )
        eng.worker_ready_signal = Mock(clear=Mock())
        eng.loaded_model_signal = Mock(clear=Mock())
        eng.exist_task_signal = Mock(clear=Mock())
        eng.exist_swapped_task_signal = Mock(clear=Mock())
        eng.worker_healthy_live_signal = Mock(clear=Mock())
        eng.cache_ready_signal = Mock(clear=Mock())
        eng.swap_space_ready_signal = Mock(clear=Mock())
        eng.cache_transfer_inited_signal = Mock(clear=Mock())
        eng.exist_prefill_task_signal = Mock(clear=Mock())
        eng.model_weights_status_signal = Mock(clear=Mock())
        eng.prefix_tree_status_signal = Mock(clear=Mock())
        eng.kv_cache_status_signal = Mock(clear=Mock())
        eng.engine_worker_queue_server = Mock(cleanup=Mock())
        eng.send_response_server = Mock(close=Mock())
        eng.recv_request_server = Mock(close=Mock())
        eng.recv_control_cmd_server = Mock(close=Mock())

        with (
            patch("fastdeploy.engine.common_engine.os.getpgid", return_value=1111),
            patch("fastdeploy.engine.common_engine.os.killpg"),
        ):
            eng._exit_sub_services()

        eng.cache_task_queue.cleanup.assert_called_once()
        eng.engine_worker_queue_server.cleanup.assert_called_once()
        eng.send_response_server.close.assert_called_once()

    def test_setting_environ_variables_splitwise_and_mm(self):
        cfg = self._make_cfg(
            splitwise_role="prefill",
            num_gpu_blocks_override=4,
            router="0.0.0.0:30000",
        )
        cfg.model_config.enable_mm = True
        eng = self._make_engine(cfg)

        with patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True):
            result = eng._setting_environ_variables()

        self.assertIn("FLAGS_use_pd_disaggregation_per_chunk=1", result)
        self.assertIn("FLAGS_fmt_write_cache_completed_signal=1", result)
        self.assertIn("FLAGS_max_partition_size=1024", result)
        self._detach_finalizer(eng)

    def test_start_cache_service_forwards_args(self):
        eng = self._make_mixed_engine()
        eng.resource_manager.cache_manager = Mock()
        eng.resource_manager.cache_manager.launch_cache_manager = Mock(return_value=["proc"])

        result = eng.start_cache_service(["0"], 9999)

        eng.resource_manager.cache_manager.launch_cache_manager.assert_called_once()
        self.assertEqual(result, ["proc"])
        self._detach_finalizer(eng)

    def test_control_update_weights_success(self):
        eng = self._make_mixed_engine()
        eng.is_paused = True
        eng._pause_cond = threading.Condition()
        eng._call_worker = Mock(return_value={"ok": True})

        result = eng._control_update_weights(ControlRequest(request_id="ctrl", method="update_weights"))
        self.assertEqual(result, {"ok": True})
        self._detach_finalizer(eng)

    def test_control_update_weights_updates_cfg_version(self):
        eng = self._make_mixed_engine()
        eng.is_paused = True
        eng._pause_cond = threading.Condition()
        eng.cfg.model_config.version = "old-version"
        eng._call_worker = Mock(return_value=[{"version": "new-version"}, {"ok": True}])

        result = eng._control_update_weights(ControlRequest(request_id="ctrl", method="update_weights"))

        self.assertEqual(result, [{"version": "new-version"}, {"ok": True}])
        self.assertEqual(eng.cfg.model_config.version, "new-version")
        self._detach_finalizer(eng)

    def test_control_update_weights_updates_cache_transfer_metadata(self):
        eng = self._make_mixed_engine()
        eng.is_paused = True
        eng._pause_cond = threading.Condition()
        eng.cfg.cache_config.num_cpu_blocks = 1
        eng._call_worker = Mock(return_value=[{"version": "new-version"}])
        eng.cache_task_queue = Mock(put_transfer_task=Mock())
        eng._wait_for_control_responses = AsyncMock(return_value=[{"ok": True}])

        result = eng._control_update_weights(ControlRequest(request_id="ctrl", method="update_weights"))

        self.assertEqual(result, [{"version": "new-version"}])
        payload = eng.cache_task_queue.put_transfer_task.call_args.args[0]
        self.assertEqual(payload[0], CacheStatus.CTRL)
        self.assertEqual(payload[1].method, "update_weights")
        self.assertIn("update_weights", payload[1].request_id)
        eng._wait_for_control_responses.assert_awaited_once_with(
            payload[1].request_id, 60, executors=["cache_transfer"]
        )
        self._detach_finalizer(eng)

    def test_control_pause_and_resume_paths(self):
        eng = self._make_mixed_engine()
        eng.is_paused = False
        eng._pause_cond = threading.Condition()
        eng.engine_worker_queue = Mock(exist_tasks=Mock(return_value=False), put_tasks=Mock())
        eng.resource_manager = Mock(
            preempted_all=Mock(return_value=[Request(request_id="r1", prompt_token_ids=[1], prompt_token_ids_len=1)]),
            get_real_bsz=Mock(),
            wait_worker_inflight_requests_finish=Mock(),
            log_status=Mock(),
            cache_manager=Mock(reset=Mock()),
            real_bsz=1,
        )
        eng.token_processor = Mock(clear_data=Mock())
        eng.scheduler = Mock(get_inflight_requests=Mock(return_value=[]), reset=Mock())
        eng._send_error_response = Mock()

        with patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True):
            eng._control_pause(ControlRequest(request_id="ctrl1", method="pause"))
            self.assertTrue(eng.is_paused)

            eng._control_resume(ControlRequest(request_id="ctrl2", method="resume"))
            self.assertFalse(eng.is_paused)

            status = eng._control_is_paused(ControlRequest(request_id="ctrl3", method="is_paused"))
            self.assertEqual(status, {"is_paused": False})
        self._detach_finalizer(eng)

    def test_run_control_method_unknown_and_success(self):
        eng = self._make_mixed_engine()
        eng.send_response_server = Mock()
        eng._pause_cond = threading.Condition()

        eng.run_control_method(ControlRequest(request_id="bad", method="nope"))
        self.assertTrue(eng.send_response_server.send_response.called)

        eng.send_response_server.reset_mock()
        eng.is_paused = True
        eng.run_control_method(ControlRequest(request_id="good", method="is_paused"))
        eng.send_response_server.send_response.assert_called()
        self._detach_finalizer(eng)

    def test_run_control_method_handler_exception(self):
        eng = self._make_mixed_engine()
        eng.send_response_server = Mock()
        eng._pause_cond = threading.Condition()

        with patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False):
            eng.run_control_method(ControlRequest(request_id="pause", method="pause"))

        eng.send_response_server.send_response.assert_called()
        self._detach_finalizer(eng)

    def test_call_worker_puts_tasks_and_returns(self):
        eng = self._make_mixed_engine()
        eng.engine_worker_queue = Mock()

        class DummyQueue:
            def __init__(self):
                self.name = "q0"

            async def get(self, timeout=None):
                return Mock(payload=ControlResponse(request_id="req", result={"ok": True}, error_code=200))

        eng._ctrl_output_queues = {"ctrl_w2e_rank0_6778": DummyQueue()}
        result = eng._call_worker(ControlRequest(request_id="req", method="noop"), timeout=1)
        self.assertEqual(result, [{"ok": True}])
        eng.engine_worker_queue.put_tasks.assert_called_once()
        self._detach_finalizer(eng)

    def test_control_sleep_defaults_tags_and_dispatches_cache_transfer(self):
        cfg = self._make_cfg(splitwise_role="mixed", num_gpu_blocks_override=4)
        eng = self._make_engine(cfg)
        eng.cfg.cache_config.num_cpu_blocks = 1
        eng.engine_worker_queue = Mock()
        eng.cache_task_queue = Mock()
        eng.resource_manager.cache_manager.reset = Mock()
        eng._control_pause = Mock()
        eng._wait_for_control_responses = AsyncMock(return_value=[{"ok": True}])

        result = eng._control_sleep(ControlRequest(request_id="sleep", method="sleep", args={}))

        self.assertEqual(result, [{"ok": True}])
        eng._control_pause.assert_called_once_with(None)
        eng.resource_manager.cache_manager.reset.assert_called_once()
        eng.engine_worker_queue.put_tasks.assert_called_once()
        eng.cache_task_queue.put_transfer_task.assert_called_once()
        sleep_req = eng.engine_worker_queue.put_tasks.call_args.args[0][0][0]
        self.assertEqual(sleep_req.args["tags"], "weight,kv_cache")
        self._detach_finalizer(eng)

    def test_control_wakeup_resumes_after_wait(self):
        cfg = self._make_cfg(splitwise_role="mixed", num_gpu_blocks_override=4)
        eng = self._make_engine(cfg)
        eng.cfg.cache_config.num_cpu_blocks = 1
        eng.engine_worker_queue = Mock()
        eng.cache_task_queue = Mock()
        eng._control_resume = Mock()
        eng._wait_for_control_responses = AsyncMock(return_value=[{"ok": True}])

        result = eng._control_wakeup(ControlRequest(request_id="wakeup", method="wakeup", args={"tags": "kv_cache"}))

        self.assertEqual(result, [{"ok": True}])
        eng.engine_worker_queue.put_tasks.assert_called_once()
        eng.cache_task_queue.put_transfer_task.assert_called_once()
        eng._control_resume.assert_called_once_with(None)
        self._detach_finalizer(eng)

    def test_control_update_weights_requires_pause(self):
        eng = self._make_mixed_engine()
        eng.is_paused = False
        eng._pause_cond = threading.Condition()

        with self.assertRaises(Exception):
            eng._control_update_weights(ControlRequest(request_id="ctrl", method="update_weights"))
        self._detach_finalizer(eng)

    def test_insert_zmq_task_to_scheduler_normal_request(self):
        eng = self._make_mixed_engine()
        eng.running = True
        eng.is_paused = False
        eng.guided_decoding_checker = None
        eng.resource_manager = Mock(abort_req_ids_set=set(), requests={})
        eng.scheduler = Mock()
        eng.engine_worker_queue = Mock()

        class DummyMetrics:
            def __init__(self):
                self.requests_number = Mock(inc=Mock())
                self.num_requests_waiting = Mock(inc=Mock())

        class DummyRecv:
            def __init__(self):
                self.calls = 0

            def receive_json_once(self, block):
                self.calls += 1
                if self.calls == 1:
                    return None, {"request_id": "ctrl", "method": "is_paused", "args": {}}
                if self.calls == 2:
                    return None, {
                        "request_id": "req1",
                        "prompt_token_ids": [1, 2],
                        "prompt_token_ids_len": 2,
                        "temperature": 1.0,
                    }
                eng.running = False
                return None, None

        eng.recv_request_server = DummyRecv()
        eng.run_control_method = Mock()
        eng.scheduler.put_requests.return_value = [("req1", None)]

        with (
            patch("fastdeploy.engine.common_engine.main_process_metrics", DummyMetrics()),
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", False),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._insert_zmq_task_to_scheduler()

        eng.run_control_method.assert_called_once()
        eng.scheduler.put_requests.assert_called()
        self._detach_finalizer(eng)

    def test_insert_zmq_task_to_scheduler_internal_adapter_decode_returns(self):
        cfg = self._make_cfg(
            splitwise_role="decode",
            num_gpu_blocks_override=4,
            router="0.0.0.0:30000",
        )
        eng = self._make_engine(cfg)
        eng.running = True

        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
            eng._insert_zmq_task_to_scheduler()

        self._detach_finalizer(eng)

    def test_schedule_request_to_worker_sends_tasks(self):
        cfg = self._make_cfg(
            splitwise_role="prefill",
            num_gpu_blocks_override=4,
            router="0.0.0.0:30000",
        )
        eng = self._make_engine(cfg)
        eng.running = True

        eng.exist_prefill_task_signal = self._Sig(0)
        eng.engine_worker_queue = Mock(exist_tasks=Mock(return_value=False), num_cache_infos=Mock(return_value=0))

        class DummyRM:
            def __init__(self):
                self.abort_req_ids_set = set()

            def available_batch(self):
                return 1

            def available_block_num(self):
                return 32

            def check_and_free_block_tables(self):
                pass

        eng.resource_manager = DummyRM()
        eng.split_connector = Mock(current_request_ids=[], has_splitwise_tasks=Mock(return_value=False))
        eng.scheduler = Mock()
        task = Request(request_id="r0", prompt_token_ids=[1], prompt_token_ids_len=1)
        eng.scheduler.get_requests.return_value = [task]

        def insert_tasks(tasks, current_id):
            eng.running = False
            return True

        eng.insert_tasks = Mock(side_effect=insert_tasks)

        with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
            eng._schedule_request_to_worker()

        eng.split_connector.send_splitwise_tasks.assert_called_once()
        eng.insert_tasks.assert_called_once()
        self._detach_finalizer(eng)

    def test_schedule_request_to_worker_waits_for_capacity(self):
        eng = self._make_mixed_engine()
        eng.running = True

        class DummyRM:
            def available_batch(self):
                eng.running = False
                return 0

        eng.resource_manager = DummyRM()
        eng.engine_worker_queue = Mock(exist_tasks=Mock(return_value=False))

        with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
            eng._schedule_request_to_worker()

        self._detach_finalizer(eng)

    def test_schedule_request_to_worker_v1_mixed_single_iteration(self):
        eng = self._make_mixed_engine()
        self._setup_v1_engine(eng)

        task = Request(request_id="v1_r0", prompt_token_ids=[1], prompt_token_ids_len=1)
        task.metrics.scheduler_recv_req_time = time.time()

        eng.scheduler = Mock(get_requests=Mock(return_value=[task]), put_results=Mock())
        eng.engine_worker_queue = Mock(exist_tasks=Mock(return_value=False), put_tasks=Mock())

        eng.resource_manager = self._make_v1_decode_rm(eng, ([], []), with_add_request=True)

        try:
            with (
                patch("fastdeploy.engine.common_engine.ThreadPoolExecutor", self._make_dummy_executor(eng)),
                patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
            ):
                eng._schedule_request_to_worker_v1()
        finally:
            eng.running = False

        eng.resource_manager.add_request.assert_called_once_with(task)
        self._detach_finalizer(eng)

    def test_schedule_request_to_worker_v1_prefill_decode_alloc_error_safe(self):
        cfg = self._make_cfg(
            splitwise_role="prefill",
            num_gpu_blocks_override=4,
            router="0.0.0.0:30000",
            kv_cache_ratio=1,
        )
        eng = self._make_engine(cfg)
        self._setup_v1_engine(eng)

        task = Request(request_id="v1_p0", prompt_token_ids=[2], prompt_token_ids_len=1)
        task.idx = 0
        task.metrics.scheduler_recv_req_time = time.time()

        eng.scheduler = Mock(get_requests=Mock(return_value=[task]), put_results=Mock())
        eng.engine_worker_queue = Mock(
            exist_tasks=Mock(return_value=False),
            get_finished_add_cache_task_req=Mock(return_value=[]),
        )

        eng.resource_manager = self._make_v1_prefill_continuous_rm(eng, waiting_async_result=False)
        eng.split_connector = Mock(
            send_splitwise_tasks=Mock(),
            check_decode_allocated=Mock(return_value=(False, "decode failed")),
            send_cache_info_to_messager=Mock(),
        )

        try:
            with (
                patch("fastdeploy.engine.common_engine.envs.PREFILL_CONTINUOUS_REQUEST_DECODE_RESOURCES", False),
                patch("fastdeploy.engine.common_engine.ThreadPoolExecutor", self._make_dummy_executor(eng)),
                patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
            ):
                eng._schedule_request_to_worker_v1()
        finally:
            eng.running = False

        eng.scheduler.put_results.assert_called_once()
        eng.resource_manager.add_request_in_p.assert_not_called()
        self._detach_finalizer(eng)

    def test_schedule_request_to_worker_v1_decode_preempted_and_errors(self):
        cfg = self._make_cfg(
            splitwise_role="decode",
            num_gpu_blocks_override=4,
            router="0.0.0.0:30000",
        )
        eng = self._make_engine(cfg)
        self._setup_v1_engine(eng)

        task = Request(request_id="v1_d0", prompt_token_ids=[3], prompt_token_ids_len=1)
        task.task_type = RequestType.PREEMPTED
        task.metrics.scheduler_recv_req_time = time.time()

        eng.scheduler = Mock(get_requests=Mock(return_value=[]), put_results=Mock())
        eng.engine_worker_queue = Mock(
            exist_tasks=Mock(return_value=False), put_tasks=Mock(), num_tasks=Mock(return_value=0)
        )
        eng._send_error_response = Mock()

        eng.resource_manager = self._make_v1_decode_rm(eng, ([task], [("rid_x", None), ("rid_y", "bad")]))

        try:
            with (
                patch("fastdeploy.engine.common_engine.ThreadPoolExecutor", self._make_dummy_executor(eng)),
                patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
            ):
                eng._schedule_request_to_worker_v1()
        finally:
            eng.running = False

        eng.scheduler.put_results.assert_called_once()
        eng.engine_worker_queue.put_tasks.assert_called_once()
        eng._send_error_response.assert_called_once_with("rid_y", "bad")
        self._detach_finalizer(eng)

    def test_schedule_request_to_worker_v1_decode_prefill_task_path(self):
        cfg = self._make_cfg(
            splitwise_role="decode",
            num_gpu_blocks_override=4,
            router="0.0.0.0:30000",
        )
        eng = self._make_engine(cfg)
        self._setup_v1_engine(eng)

        task = Request(request_id="v1_d1", prompt_token_ids=[4], prompt_token_ids_len=1)
        task.task_type = RequestType.PREFILL
        task.trace_carrier = {}
        task.metrics.scheduler_recv_req_time = time.time()

        eng.scheduler = Mock(get_requests=Mock(return_value=[]), put_results=Mock())
        eng.engine_worker_queue = Mock(
            exist_tasks=Mock(return_value=False), put_tasks=Mock(), num_tasks=Mock(return_value=0)
        )

        eng.resource_manager = self._make_v1_decode_rm(eng, ([task], []))

        try:
            with (
                patch("fastdeploy.engine.common_engine.ThreadPoolExecutor", self._make_dummy_executor(eng)),
                patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
            ):
                eng._schedule_request_to_worker_v1()
        finally:
            eng.running = False

        eng.engine_worker_queue.put_tasks.assert_called_once()
        self._detach_finalizer(eng)

    def test_schedule_request_to_worker_v1_error_task_none_skips_send(self):
        cfg = self._make_cfg(
            splitwise_role="decode",
            num_gpu_blocks_override=4,
            router="0.0.0.0:30000",
        )
        eng = self._make_engine(cfg)
        self._setup_v1_engine(eng)

        task = Request(request_id="v1_e0", prompt_token_ids=[1], prompt_token_ids_len=1)
        task.task_type = RequestType.PREFILL
        task.trace_carrier = {}
        task.metrics.scheduler_recv_req_time = time.time()

        eng.scheduler = Mock(get_requests=Mock(return_value=[]), put_results=Mock())
        eng.engine_worker_queue = Mock(
            exist_tasks=Mock(return_value=False), put_tasks=Mock(), num_tasks=Mock(return_value=0)
        )
        eng._send_error_response = Mock()

        eng.resource_manager = self._make_v1_decode_rm(eng, ([task], [("rid_none", None)]))

        with (
            patch("fastdeploy.engine.common_engine.ThreadPoolExecutor", self._make_dummy_executor(eng)),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._schedule_request_to_worker_v1()

        eng.engine_worker_queue.put_tasks.assert_called_once()
        eng._send_error_response.assert_not_called()
        self._detach_finalizer(eng)

    def test_schedule_request_to_worker_v1_threadpool_shutdown_breaks(self):
        eng = self._make_mixed_engine()
        self._setup_v1_engine(eng)

        eng.engine_worker_queue = Mock(exist_tasks=Mock(return_value=False))

        eng.resource_manager = self._make_v1_decode_rm(eng, ([], []))

        class DummyExecutor:
            def __init__(self, max_workers=None):
                pass

            def submit(self, fn):
                raise RuntimeError("cannot schedule new futures after shutdown")

        with (
            patch("fastdeploy.engine.common_engine.ThreadPoolExecutor", DummyExecutor),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._schedule_request_to_worker_v1()

        self._detach_finalizer(eng)

    def test_schedule_request_to_worker_v1_prefill_continuous_cache_success(self):
        cfg = self._make_cfg(
            splitwise_role="prefill",
            num_gpu_blocks_override=4,
            router="0.0.0.0:30000",
            kv_cache_ratio=1,
        )
        eng = self._make_engine(cfg)
        self._setup_v1_engine(eng)

        task = Request(request_id="pc_ok", prompt_token_ids=[1], prompt_token_ids_len=1)
        task.idx = 0
        task.metrics.scheduler_recv_req_time = time.time()

        eng.scheduler = Mock(get_requests=Mock(return_value=[task]), put_results=Mock())

        eng.resource_manager = self._make_v1_prefill_continuous_rm(eng, waiting_async_result=False)

        calls = {"n": 0}

        def get_finished_add_cache_task_req():
            if calls["n"] == 0:
                calls["n"] += 1
                return ["pc_ok"]
            return []

        eng.engine_worker_queue = Mock(
            exist_tasks=Mock(return_value=False),
            get_finished_add_cache_task_req=Mock(side_effect=get_finished_add_cache_task_req),
        )

        eng.split_connector = Mock(
            send_splitwise_tasks=Mock(),
            check_decode_allocated=Mock(return_value=(True, "")),
            send_cache_info_to_messager=Mock(),
        )

        with (
            patch("fastdeploy.engine.common_engine.envs.PREFILL_CONTINUOUS_REQUEST_DECODE_RESOURCES", True),
            patch("fastdeploy.engine.common_engine.ThreadPoolExecutor", self._make_dummy_executor(eng)),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._schedule_request_to_worker_v1()

        eng.split_connector.send_splitwise_tasks.assert_called()
        eng.split_connector.send_cache_info_to_messager.assert_called_once()
        eng.resource_manager.add_request_in_p.assert_called_once()
        eng.scheduler.put_results.assert_not_called()
        self._detach_finalizer(eng)

    def test_schedule_request_to_worker_v1_prefill_continuous_wait_async_none(self):
        cfg = self._make_cfg(
            splitwise_role="prefill",
            num_gpu_blocks_override=4,
            router="0.0.0.0:30000",
            kv_cache_ratio=1,
        )
        eng = self._make_engine(cfg)
        self._setup_v1_engine(eng)

        task = Request(request_id="pc_fail", prompt_token_ids=[1], prompt_token_ids_len=1)
        task.idx = 0
        task.error_code = 501
        task.error_message = "prefill bad"
        task.metrics.scheduler_recv_req_time = time.time()

        eng.scheduler = Mock(get_requests=Mock(return_value=[task]), put_results=Mock())

        eng.resource_manager = self._make_v1_prefill_continuous_rm(eng, waiting_async_result=None)

        calls = {"n": 0}

        def get_finished_add_cache_task_req():
            if calls["n"] == 0:
                calls["n"] += 1
                return ["pc_fail"]
            return []

        eng.engine_worker_queue = Mock(
            exist_tasks=Mock(return_value=False),
            get_finished_add_cache_task_req=Mock(side_effect=get_finished_add_cache_task_req),
        )

        eng.split_connector = Mock(
            send_splitwise_tasks=Mock(),
            check_decode_allocated=Mock(return_value=(True, "")),
            send_cache_info_to_messager=Mock(),
        )

        with (
            patch("fastdeploy.engine.common_engine.envs.PREFILL_CONTINUOUS_REQUEST_DECODE_RESOURCES", True),
            patch("fastdeploy.engine.common_engine.ThreadPoolExecutor", self._make_dummy_executor(eng)),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._schedule_request_to_worker_v1()

        eng.scheduler.put_results.assert_called_once()
        eng.resource_manager.pre_recycle_resource.assert_called_once_with("pc_fail")
        eng.resource_manager.add_request_in_p.assert_not_called()
        self._detach_finalizer(eng)

    def test_start_zmq_service_ipc_servers(self):
        eng = self._make_mixed_engine()

        created = {"threads": 0}

        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False),
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", False),
            patch("fastdeploy.engine.common_engine.ZmqIpcServer", self._make_zmq_server_cls()),
            patch("fastdeploy.engine.common_engine.threading.Thread", self._make_zmq_thread_cls(created)),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng.start_zmq_service(api_server_pid=4321)

        self.assertEqual(created["threads"], 3)
        self.assertEqual(eng.recv_request_server.kwargs["name"], 4321)
        self._detach_finalizer(eng)

    def test_start_zmq_service_internal_adapter_tcp(self):
        eng = self._make_mixed_engine()

        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True),
            patch("fastdeploy.engine.common_engine.ZmqTcpServer", self._make_zmq_server_cls()),
            patch("fastdeploy.engine.common_engine.InternalAdapter", Mock()),
            patch("fastdeploy.engine.common_engine.threading.Thread", self._make_zmq_thread_cls()),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng.start_zmq_service(api_server_pid=5555)

        self.assertIsNotNone(eng.internal_adapter)
        self._detach_finalizer(eng)

    def test_start_zmq_service_none(self):
        eng = self._make_mixed_engine()
        eng.start_zmq_service(api_server_pid=None)
        self._detach_finalizer(eng)

    def test_insert_zmq_task_to_scheduler_abort_request(self):
        eng = self._make_mixed_engine()
        eng.running = True
        eng.is_paused = False
        eng.guided_decoding_checker = None

        class DummyRM:
            def __init__(self):
                self.abort_req_ids_set = set()
                self.waiting_abort_req_id_set = set()
                self.real_bsz = 1
                self.requests = {"rid": Mock()}

            def add_abort_req_ids(self, req_id):
                self.waiting_abort_req_id_set.add(req_id)

            def _prepare_preempt_task(self, req):
                return Request(request_id="rid", prompt_token_ids=[1], prompt_token_ids_len=1)

        eng.resource_manager = DummyRM()
        eng.scheduler = Mock(_recycle=Mock())
        eng.engine_worker_queue = Mock()

        eng.recv_request_server = self._make_dummy_recv(
            eng,
            payload={"request_id": "rid", "status": RequestStatus.ABORT.value},
        )

        with (
            patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True),
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", False),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._insert_zmq_task_to_scheduler()

        # Verify abort request was handled correctly - added to waiting_abort_req_id_set
        self.assertIn("rid", eng.resource_manager.waiting_abort_req_id_set)
        self._detach_finalizer(eng)

    def test_insert_zmq_task_to_scheduler_paused_sends_error(self):
        eng = self._make_mixed_engine()
        eng.running = True
        eng.is_paused = True
        eng.guided_decoding_checker = None
        eng.resource_manager = Mock(abort_req_ids_set=set(), requests={})
        eng.scheduler = Mock()
        eng.engine_worker_queue = Mock()
        eng._send_error_response = Mock()

        eng.recv_request_server = self._make_dummy_recv(
            eng,
            payload={
                "request_id": "req1",
                "prompt_token_ids": [1],
                "prompt_token_ids_len": 1,
                "temperature": 1.0,
            },
        )

        with (
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", False),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._insert_zmq_task_to_scheduler()

        eng._send_error_response.assert_called_once()
        self._detach_finalizer(eng)

    def test_insert_zmq_task_to_scheduler_context_terminated(self):
        eng = self._make_mixed_engine()
        eng.running = True

        eng.recv_request_server = self._make_dummy_recv(eng, error=RuntimeError("Context was terminated"))

        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False),
            patch("fastdeploy.engine.common_engine.ZmqIpcServer", self._make_zmq_server_cls()),
            patch.object(eng, "llm_logger") as mock_logger,
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._insert_zmq_task_to_scheduler()

        mock_logger.info.assert_called()
        self._detach_finalizer(eng)

    def test_insert_zmq_task_to_scheduler_error_reinit(self):
        eng = self._make_mixed_engine()
        eng.running = True

        eng.recv_request_server = self._make_dummy_recv(eng, error=RuntimeError("boom"))

        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False),
            patch("fastdeploy.engine.common_engine.ZmqIpcServer", self._make_zmq_server_cls()),
            patch.object(eng, "llm_logger") as mock_logger,
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._insert_zmq_task_to_scheduler()

        mock_logger.error.assert_called()
        self._detach_finalizer(eng)

    def test_decode_process_splitwise_requests_single_cycle(self):
        cfg = self._make_cfg(
            splitwise_role="decode",
            num_gpu_blocks_override=4,
            router="0.0.0.0:30000",
        )
        eng = self._make_engine(cfg)
        eng.running = True
        eng.enable_decode_cache_task = False
        eng.cfg.splitwise_version = "v1"
        eng.scheduler = Mock(has_request=Mock(return_value=True), put_results=Mock())
        eng._insert_prefilled_requests = Mock()

        class DummyRM:
            def is_resource_sufficient(self, prompt_len):
                return True

        eng.resource_manager = DummyRM()
        eng.insert_tasks = Mock()

        task = Request(request_id="r0", prompt_token_ids=[1], prompt_token_ids_len=1)
        output = RequestOutput(
            request_id="r1",
            outputs=Mock(token_ids=[1], decode_type=1, tool_calls=None),
            metrics=Mock(),
            finished=False,
        )

        class DummyQueue:
            def disaggregate_queue_empty(self):
                return False

            def get_disaggregated_tasks(self):
                eng.running = False
                return [
                    (None, [task]),
                    (None, [output]),
                ]

        eng.engine_worker_queue = DummyQueue()

        class DummyThread:
            def __init__(self, target=None, daemon=None):
                self.target = target
                self.daemon = daemon

            def start(self):
                try:
                    self.target()
                finally:
                    eng.running = False

        with (
            patch("fastdeploy.engine.common_engine.threading.Thread", DummyThread),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
            patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False),
        ):
            eng._decode_process_splitwise_requests()

        eng.insert_tasks.assert_called_once()
        eng._insert_prefilled_requests.assert_called_once()
        self._detach_finalizer(eng)

    def test_zmq_send_generated_tokens_single_batch(self):
        eng = self._make_mixed_engine()
        eng.running = True
        eng.send_response_server = Mock()

        self._make_scheduler_with_output(eng, [1, 2], 1, True)

        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False),
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", False),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._zmq_send_generated_tokens()

        eng.send_response_server.send_response.assert_called()
        self._detach_finalizer(eng)

    def test_zmq_send_generated_tokens_non_internal_adapter_empty_and_other(self):
        eng = self._make_mixed_engine()
        eng.running = True
        eng.send_response_server = Mock()
        eng._decode_token = Mock(return_value=("", []))

        self._make_scheduler_with_output(eng, [1], 0, True, include_raw=True)

        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False),
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", False),
        ):
            eng._zmq_send_generated_tokens()

        eng.send_response_server.send_response.assert_called_once()
        self._detach_finalizer(eng)

    def test_zmq_send_generated_tokens_logs_exception(self):
        eng = self._make_mixed_engine()
        eng.running = True
        eng.send_response_server = Mock()

        def get_results():
            eng.running = False
            raise RuntimeError("boom")

        eng.scheduler = Mock(get_results=get_results)

        try:
            with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False):
                eng._zmq_send_generated_tokens()
        finally:
            eng.running = False

        self._detach_finalizer(eng)

    def test_zmq_send_generated_tokens_internal_adapter_decode(self):
        eng = self._make_mixed_engine()
        eng.running = True
        eng.send_response_server = Mock()

        class DummyProcessor:
            def __init__(self):
                self.decode_status = {"rid": (0, 2)}

            def ids2tokens(self, token_ids, req_id):
                return "hi", [1, 2], None

        eng.data_processor = DummyProcessor()

        self._make_scheduler_with_output(eng, [1, 2], 0, True, fmt="list")

        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._zmq_send_generated_tokens()

        eng.send_response_server.send_response.assert_called_once()
        self._detach_finalizer(eng)

    def test_zmq_send_generated_tokens_internal_adapter_decode_type_one(self):
        eng = self._make_mixed_engine()
        eng.running = True
        eng.send_response_server = Mock()

        self._make_scheduler_with_output(eng, [3, 4], 1, True, fmt="list")

        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._zmq_send_generated_tokens()

        eng.send_response_server.send_response.assert_called_once()
        self._detach_finalizer(eng)

    def test_zmq_send_generated_tokens_internal_adapter_warns_on_empty(self):
        eng = self._make_mixed_engine()
        eng.running = True
        eng.send_response_server = Mock()

        self._make_scheduler_with_output(eng, [], 1, False, fmt="list")

        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True),
            patch.object(eng, "llm_logger") as mock_logger,
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._zmq_send_generated_tokens()

        mock_logger.warning.assert_called()
        self._detach_finalizer(eng)

    def test_zmq_send_generated_tokens_empty_results(self):
        eng = self._make_mixed_engine()
        eng.running = True
        eng.scheduler = Mock()

        def get_results():
            eng.running = False
            return []

        eng.scheduler.get_results = get_results

        with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
            eng._zmq_send_generated_tokens()
        self._detach_finalizer(eng)

    def test_zmq_send_generated_tokens_decode_type_zero(self):
        eng = self._make_mixed_engine()
        eng.running = True
        eng.send_response_server = Mock()

        self._make_scheduler_with_output(eng, [1, 2], 0, True)
        eng._decode_token = Mock(return_value=("hi", [1, 2]))

        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False),
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", False),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._zmq_send_generated_tokens()

        eng.send_response_server.send_response.assert_called_once()
        self._detach_finalizer(eng)

    def test_zmq_send_generated_tokens_warns_on_empty(self):
        eng = self._make_mixed_engine()
        eng.running = True
        eng.send_response_server = Mock()

        self._make_scheduler_with_output(eng, [], 1, False)

        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False),
            patch.object(eng, "llm_logger") as mock_logger,
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._zmq_send_generated_tokens()

        mock_logger.warning.assert_called()
        self._detach_finalizer(eng)

    def test_wait_for_control_responses_success(self):
        eng = self._make_mixed_engine()

        eng._ctrl_output_queues = {
            "ctrl_w2e_rank0_6778": self._make_ctrl_queue(
                "q0", Mock(request_id="req", error_code=200, result={"ok": True})
            ),
            "ctrl_w2e_rank1_6778": self._make_ctrl_queue(
                "q1", Mock(request_id="req", error_code=200, result={"ok": True})
            ),
        }

        results = asyncio.run(eng._wait_for_control_responses("req", timeout=1))
        self.assertEqual(results, [{"ok": True}, {"ok": True}])
        self._detach_finalizer(eng)

    def test_wait_for_control_responses_filters_executors(self):
        eng = self._make_mixed_engine()

        eng._ctrl_output_queues = {
            "ctrl_w2e_rank0_6778": self._make_ctrl_queue(
                "worker", Mock(request_id="req", error_code=200, result={"worker": True})
            ),
            "ctrl_c2e_rank0_6779": self._make_ctrl_queue(
                "cache", Mock(request_id="req", error_code=200, result={"cache": True})
            ),
        }

        worker_results = asyncio.run(eng._wait_for_control_responses("req", timeout=1, executors=["worker"]))
        cache_results = asyncio.run(eng._wait_for_control_responses("req", timeout=1, executors=["cache_transfer"]))

        self.assertEqual(worker_results, [{"worker": True}])
        self.assertEqual(cache_results, [{"cache": True}])
        self._detach_finalizer(eng)

    def test_wait_for_control_responses_ignores_mismatch(self):
        eng = self._make_mixed_engine()

        class DummyQueue:
            def __init__(self, name, payloads):
                self.name = name
                self.payloads = list(payloads)

            async def get(self, timeout=None):
                return Mock(payload=self.payloads.pop(0))

        eng._ctrl_output_queues = {
            "ctrl_w2e_rank0_6778": DummyQueue(
                "q0",
                [
                    Mock(request_id="old", error_code=200, result={"ok": False}),
                    Mock(request_id="req", error_code=200, result={"ok": "from-q0"}),
                ],
            ),
            "ctrl_w2e_rank1_6778": self._make_ctrl_queue(
                "q1", Mock(request_id="req", error_code=200, result={"ok": True})
            ),
        }

        results = asyncio.run(eng._wait_for_control_responses("req", timeout=1))
        self.assertEqual(results, [{"ok": "from-q0"}, {"ok": True}])
        self.assertEqual(
            eng._ctrl_response_mailboxes["ctrl_w2e_rank0_6778"]["old"].result,
            {"ok": False},
        )
        self._detach_finalizer(eng)

    def test_wait_for_control_responses_error_paths(self):
        eng = self._make_mixed_engine()

        eng._ctrl_output_queues = {
            "ctrl_w2e_rank0_6778": self._make_ctrl_queue("q0", Exception("boom"), payload_wrapped=False)
        }

        with self.assertRaises(Exception):
            asyncio.run(eng._wait_for_control_responses("req", timeout=1))
        self._detach_finalizer(eng)

    def test_wait_for_control_responses_none_message(self):
        eng = self._make_mixed_engine()

        eng._ctrl_output_queues = {"ctrl_w2e_rank0_6778": self._make_ctrl_queue("q0", None, payload_wrapped=False)}

        with self.assertRaises(Exception):
            asyncio.run(eng._wait_for_control_responses("req", timeout=1))
        self._detach_finalizer(eng)

    def test_wait_for_control_responses_error_code(self):
        eng = self._make_mixed_engine()

        eng._ctrl_output_queues = {
            "ctrl_w2e_rank0_6778": self._make_ctrl_queue(
                "q0", ControlResponse(request_id="req", error_code=500, error_message="bad")
            )
        }

        with self.assertRaises(Exception):
            asyncio.run(eng._wait_for_control_responses("req", timeout=1))
        self._detach_finalizer(eng)

    def test_wait_for_control_responses_timeout(self):
        eng = self._make_mixed_engine()
        eng._ctrl_output_queues = {"ctrl_w2e_rank0_6778": self._make_ctrl_queue("q0", None, payload_wrapped=False)}

        with patch("fastdeploy.engine.common_engine.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with self.assertRaises(Exception):
                asyncio.run(eng._wait_for_control_responses("req", timeout=1))
        self._detach_finalizer(eng)

    def test_wait_for_control_responses_without_matching_queues(self):
        eng = self._make_mixed_engine()
        eng._ctrl_output_queues = {"ctrl_w2e_rank0_6778": self._make_ctrl_queue("q0", None, payload_wrapped=False)}

        result = asyncio.run(eng._wait_for_control_responses("req", timeout=1, executors=["cache_transfer"]))
        self.assertIsNone(result)
        self._detach_finalizer(eng)

    def test_insert_tasks_prefill_error_and_success(self):
        cfg = self._make_cfg(
            splitwise_role="prefill",
            num_gpu_blocks_override=4,
            router="0.0.0.0:30000",
        )
        eng = self._make_engine(cfg)

        eng.resource_manager = self._make_insert_tasks_rm(n=2)
        eng.scheduler = Mock()
        eng.engine_worker_queue = Mock()
        eng.split_connector = Mock()
        eng.split_connector.send_cache_info_to_messager = Mock()
        eng.split_connector.check_decode_allocated = Mock(
            side_effect=[(False, "no"), (True, "")],
        )
        eng.token_processor = Mock(number_of_tasks=0, number_of_input_tokens=0)
        eng.update_requests_chunk_size = Mock()

        tasks = [
            Request(request_id="p0", prompt_token_ids=[1], prompt_token_ids_len=1),
            Request(request_id="p1", prompt_token_ids=[1], prompt_token_ids_len=1),
        ]
        for task in tasks:
            task.metrics.scheduler_recv_req_time = time.time()

        eng.insert_tasks(tasks)

        eng.scheduler.put_results.assert_called_once()
        eng.engine_worker_queue.put_tasks.assert_called_once()
        self._detach_finalizer(eng)

    def test_insert_tasks_decode_disaggregate_sets_flags(self):
        cfg = self._make_cfg(
            splitwise_role="decode",
            num_gpu_blocks_override=4,
            router="0.0.0.0:30000",
        )
        eng = self._make_engine(cfg)

        eng.resource_manager = self._make_insert_tasks_rm()
        eng.engine_worker_queue = Mock()
        eng.split_connector = Mock(send_cache_info_to_prefill=Mock())
        eng.token_processor = Mock(number_of_tasks=0, number_of_input_tokens=0)

        task = Request(request_id="d1", prompt_token_ids=[1], prompt_token_ids_len=1, disaggregate_info={})
        eng.insert_tasks([task])

        eng.split_connector.send_cache_info_to_prefill.assert_called_once()
        self._detach_finalizer(eng)

    def test_insert_tasks_mm_updates_chunk_size(self):
        cfg = self._make_cfg(splitwise_role="mixed", num_gpu_blocks_override=4)
        cfg.model_config.enable_mm = True
        eng = self._make_engine(cfg)

        eng.resource_manager = self._make_insert_tasks_rm()
        eng.engine_worker_queue = Mock()
        eng.token_processor = Mock(number_of_tasks=0, number_of_input_tokens=0)
        eng.update_mm_requests_chunk_size = Mock()

        task = Request(request_id="mm", prompt_token_ids=[1], prompt_token_ids_len=1)
        task.metrics.scheduler_recv_req_time = time.time()
        eng.insert_tasks([task])

        eng.update_mm_requests_chunk_size.assert_called_once()
        self._detach_finalizer(eng)

    def test_insert_tasks_sets_prefill_flag(self):
        eng = self._make_mixed_engine()

        eng.resource_manager = self._make_insert_tasks_rm()
        eng.engine_worker_queue = Mock()
        eng.token_processor = Mock(number_of_tasks=0, number_of_input_tokens=0)
        eng.update_requests_chunk_size = Mock()

        task = Request(
            request_id="prefill",
            prompt_token_ids=[1],
            prompt_token_ids_len=1,
            disaggregate_info={},
        )
        task.metrics.scheduler_recv_req_time = time.time()
        eng.insert_tasks([task])

        eng.update_requests_chunk_size.assert_not_called()
        self._detach_finalizer(eng)

    def test_update_requests_chunk_size_empty_inputs(self):
        eng = self._make_mixed_engine()
        eng.cfg.cache_config.enable_chunked_prefill = True
        eng.update_requests_chunk_size([])
        self._detach_finalizer(eng)

    def test_update_mm_requests_chunk_size_handles_none_images(self):
        eng = self._make_mixed_engine()
        eng.cfg.cache_config.enable_chunked_prefill = True
        eng.partial_chunked_tokens = [0, 16]
        eng.data_processor = type("DP", (), {"image_patch_id": 9})()

        inputs = {
            "input_ids": np.array([9, 1, 2, 3], dtype="int64"),
            "token_type_ids": np.array([0, 0, 0, 0], dtype="int64"),
            "image_type_ids": np.array([1], dtype="int32"),
            "grid_thw": np.array([[2, 1, 1]], dtype="int64"),
            "images": None,
            "position_ids": np.array([0, 1, 2, 3], dtype="int64"),
        }
        req = Request(request_id="mm1", multimodal_inputs=inputs)

        with patch.dict("sys.modules", {"fastdeploy.model_executor.ops.gpu": self._make_mm_stub_module()}):
            eng.update_mm_requests_chunk_size([req])

        chunk_info = req.get("prefill_chunk_info")
        self.assertEqual(len(chunk_info), 1)
        self.assertIsNone(chunk_info[0]["images"])
        self._detach_finalizer(eng)

    def test_update_mm_requests_chunk_size_expands_grid(self):
        eng = self._make_mixed_engine()
        eng.cfg.cache_config.enable_chunked_prefill = True
        eng.partial_chunked_tokens = [0, 16]
        eng.data_processor = type("DP", (), {"image_patch_id": 9})()

        inputs = {
            "input_ids": np.array([9, 1, 2, 3], dtype="int64"),
            "token_type_ids": np.array([0, 0, 0, 0], dtype="int64"),
            "image_type_ids": np.array([1, 1], dtype="int32"),
            "grid_thw": np.array([[2, 1, 1]], dtype="int64"),
            "images": np.ones((2,), dtype="uint8"),
            "position_ids": np.array([0, 1, 2, 3], dtype="int64"),
        }
        req = Request(request_id="mm3", multimodal_inputs=inputs)

        with patch.dict("sys.modules", {"fastdeploy.model_executor.ops.gpu": self._make_mm_stub_module()}):
            eng.update_mm_requests_chunk_size([req])

        self.assertTrue(req.get("prefill_chunk_info"))
        self._detach_finalizer(eng)

    def test_update_mm_requests_chunk_size_skips_when_disabled(self):
        eng = self._make_mixed_engine()
        eng.cfg.cache_config.enable_chunked_prefill = False
        req = Request(request_id="mm2", multimodal_inputs={"images": None})

        eng.update_mm_requests_chunk_size([req])
        self._detach_finalizer(eng)

    def test_insert_tasks_single_request_with_trace_carrier(self):
        eng = self._make_mixed_engine()

        eng.resource_manager = self._make_insert_tasks_rm()
        eng.engine_worker_queue = Mock()
        eng.token_processor = Mock(number_of_tasks=0, number_of_input_tokens=0)
        eng.update_requests_chunk_size = Mock()

        task = Request(
            request_id="trace",
            prompt_token_ids=[1],
            prompt_token_ids_len=1,
            trace_carrier={"trace_id": "1"},
        )
        task.metrics.scheduler_recv_req_time = time.time()
        eng.insert_tasks(task)

        eng.update_requests_chunk_size.assert_called_once()
        self._detach_finalizer(eng)

    def test_exit_sub_services_cleanup_paths(self):
        """Cover lines 1312-1340, 1350-1354 in _exit_sub_services."""
        cfg = self._make_cfg(splitwise_role="mixed")

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", self._make_simple_dummy_q_cls()):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        # attach stubs used by cleanup
        eng.worker_ready_signal = self._Sig(0)
        eng.loaded_model_signal = self._Sig(0)
        eng.exist_task_signal = self._Sig(0)
        eng.exist_swapped_task_signal = self._Sig(0)
        eng.worker_healthy_live_signal = self._Sig(0)
        eng.cache_ready_signal = self._Sig(0)
        eng.swap_space_ready_signal = self._Sig(0)
        eng.exist_prefill_task_signal = self._Sig(0)
        eng.model_weights_status_signal = self._Sig(0)
        eng.prefix_tree_status_signal = self._Sig(0)
        eng.kv_cache_status_signal = self._Sig(0)
        eng.send_response_server = Mock()
        eng.recv_request_server = Mock()
        eng.recv_control_cmd_server = Mock()

        # ensure cache manager control flags exist before first call
        eng.resource_manager.cache_manager.shm_cache_task_flag_broadcast = Mock(clear=lambda: None)
        eng.resource_manager.cache_manager.cache_ready_signal = Mock(clear=lambda: None)
        eng.cache_manager_processes = []

        # worker_proc kill raises -> cover 1312-1313
        eng.worker_proc = MagicMock(pid=1001)
        with patch("fastdeploy.engine.common_engine.os.getpgid", side_effect=RuntimeError("boom")):
            eng._exit_sub_services()

        # Prepare cache manager processes to hit both normal and exception branch
        class DummyCacheMgr:
            def __init__(self, pid, raise_on_kill=False):
                self.pid = pid
                self.raise_on_kill = raise_on_kill

        eng.cache_manager_processes = [DummyCacheMgr(2001, False), DummyCacheMgr(2002, True)]
        eng.resource_manager.cache_manager.shm_cache_task_flag_broadcast = Mock(clear=lambda: None)
        eng.resource_manager.cache_manager.cache_ready_signal = Mock(clear=lambda: None)

        def fake_getpgid(pid):
            return pid

        def fake_killpg(pid, sig):
            if pid == 2002:
                raise RuntimeError("kill fail")

        # cache_task_queue with cleanup
        eng.cache_task_queue = Mock()
        eng.cache_task_queue.cleanup = Mock()

        eng.dp_processed = [Mock(pid=3001, join=lambda: None)]
        eng.dp_engine_worker_queue_server = [Mock(cleanup=lambda: None)]

        with (
            patch("fastdeploy.engine.common_engine.os.getpgid", side_effect=fake_getpgid),
            patch("fastdeploy.engine.common_engine.os.killpg", side_effect=fake_killpg),
        ):
            eng._exit_sub_services()

        # Now cover manager.shutdown warning path (no cleanup attribute)
        class DummyMgr:
            def __init__(self):
                self.manager = Mock(shutdown=Mock(side_effect=RuntimeError("shutdown fail")))

        eng.cache_task_queue = DummyMgr()
        eng._exit_sub_services()
        self._detach_finalizer(eng)

    def test_start_worker_service_cmd_build(self):
        """Cover 1517, 1526, 1568, 1592, 1595 by building the worker command with mocks."""
        with patch("fastdeploy.config.get_host_ip", return_value="127.0.0.1"):
            cfg = self._make_cfg(
                splitwise_role="mixed", num_gpu_blocks_override=4, ips=["127.0.0.1", "127.0.0.2"], data_parallel_size=2
            )
        # Make model multi-modal so env var branch already covered above; here not required
        cfg.structured_outputs_config.logits_processors = ["A", "B"]

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", self._make_simple_dummy_q_cls()):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        eng.data_processor = self._stub_processor()

        captured = {"cmd": None}

        class DummyProc:
            def __init__(self):
                self.stdout = None

            def poll(self):
                return None

        def fake_popen(cmd, stdout, shell, preexec_fn):
            captured["cmd"] = cmd
            return DummyProc()

        with patch("fastdeploy.engine.common_engine.subprocess.Popen", side_effect=fake_popen):
            with patch("fastdeploy.engine.common_engine.llm_logger"):
                p = eng._start_worker_service()

        self.assertIsNotNone(p)
        self.assertIsInstance(captured["cmd"], str)
        # logits processors added (1568)
        self.assertIn("--logits-processors A B", captured["cmd"])  # type: ignore
        # num_gpu_blocks_override added (1592)
        self.assertIn("--num_gpu_blocks_override 4", captured["cmd"])  # type: ignore
        # ips/nnodes added when nnode > 1 (1595)
        self.assertIn("--nnodes 2", captured["cmd"])  # type: ignore
        self._detach_finalizer(eng)

    def test_check_health_unhealthy(self):
        """Cover line 1628: unhealthy worker."""
        cfg = self._make_cfg(splitwise_role="mixed")

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", self._make_simple_dummy_q_cls()):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        # set worker live time far past threshold
        eng.worker_healthy_live_signal = self._Sig(int(time.time()) - 1000)
        ok, msg = eng.check_health(time_interval_threashold=1)
        self.assertFalse(ok)
        self.assertIn("Not Healthy".lower(), msg.lower())
        self._detach_finalizer(eng)

    def test_launch_components_expert_parallel(self):
        """Cover 1635-1638, 1660-1676, 1684-1703 in launch_components()."""
        # For prefill + local scheduler the core code now requires a router
        # and ENABLE_V1_KVCACHE_SCHEDULER=0 when using the default IPC protocol.
        with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0):
            cfg = self._make_cfg(
                splitwise_role="prefill",
                # enable expert parallel and dp > 1 to go into the branch
                data_parallel_size=2,
                enable_expert_parallel=True,
                router="0.0.0.0:30000",
            )

        # Provide EngineWorkerQueue stub for ctor
        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", self._make_full_dummy_q_cls()):
            eng = EngineService(cfg, start_queue=True, use_async_llm=True)

        # Init signals to create launched_expert_service_signal
        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_MULTI_API_SERVER", False):
            eng.ipc_signal_suffix = cfg.parallel_config.engine_worker_queue_port[0]
            eng._init_worker_signals()

            # Don't create real queues/processes
            with (
                patch("fastdeploy.engine.common_engine.EngineWorkerQueue") as FakeQ,
                patch("fastdeploy.engine.common_engine.multiprocessing.Process") as FakeP,
            ):
                # Fake queue instances with cleanup
                FakeQ.return_value = Mock(cleanup=lambda: None)

                # When starting process, immediately mark the signal as 1 to break waiting loop
                def start_side_effect(*args, **kwargs):
                    # set value for dp id 1
                    eng.launched_expert_service_signal.value[1] = 1

                proc_instance = Mock(start=start_side_effect)
                FakeP.return_value = proc_instance

                # Avoid scheduler doing real work
                eng.scheduler.start = lambda *a, **k: None
                with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
                    eng.launch_components()

                # Verify expert service branch executed
                self.assertTrue(hasattr(eng, "dp_processed"))
                self.assertGreaterEqual(len(eng.dp_processed), 1)
        self._detach_finalizer(eng)

    def test_check_worker_initialize_status_progress(self):
        """Cover 1710-1762 by simulating stdout and ready signals."""
        cfg = self._make_cfg(splitwise_role="mixed")

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", self._make_full_dummy_q_cls()):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        # Fake worker process stdout content that matches regexes
        lines = [
            b"Loading checkpoint shards: 1\n",
            b"Start load layer 5\n",
        ]

        class DummyProc:
            def __init__(self, it):
                self._it = iter(it)

            @property
            def stdout(self):
                return self._it

            def poll(self):
                return None

        eng.worker_proc = DummyProc(lines)
        eng.worker_init_status = {}
        eng.cfg.model_config.num_hidden_layers = 8

        # worker_ready_signal makes _worker_processes_ready() return True
        eng.worker_ready_signal = self._Sig(1)

        # Replace tqdm and sleep for fast execution
        with patch("fastdeploy.engine.common_engine.tqdm", lambda *a, **k: self._DummyPbar()):
            with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
                ok = eng.check_worker_initialize_status()
        self.assertTrue(ok)
        self._detach_finalizer(eng)

    def test_worker_processes_ready_false(self):
        """Cover line 1382 returning False."""
        cfg = self._make_cfg()

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", self._make_full_dummy_q_cls()):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        eng.worker_ready_signal = self._Sig(0)
        self.assertFalse(eng._worker_processes_ready())
        self._detach_finalizer(eng)

    def test_init_worker_signals_profile_iluvatar(self):
        """Cover line 1434 by forcing iluvatar custom device and do_profile=True."""
        # do_profile=True when num_gpu_blocks_override is None
        cfg = self._make_cfg(num_gpu_blocks_override=None)

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", self._make_full_dummy_q_cls()):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        eng.ipc_signal_suffix = cfg.parallel_config.engine_worker_queue_port[0]
        with patch("fastdeploy.engine.common_engine.paddle.is_compiled_with_custom_device", return_value=True):
            eng._init_worker_signals()
        # signal should exist
        self.assertTrue(hasattr(eng, "get_profile_block_num_signal"))
        self._detach_finalizer(eng)

    def test_launch_components_dp_mode(self):
        """Cover 1648-1652 branch for DP scheduler mode."""
        # When ENABLE_V1_KVCACHE_SCHEDULER=1 the IPC cache-transfer protocol
        # is no longer supported; force it to 0 here to avoid the
        # NotImplementedError raised in EngineArgs.__post_init__ so we can
        # still exercise the DP branch of launch_components.
        with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0):
            cfg = self._make_cfg(
                splitwise_role="prefill",
                data_parallel_size=2,
                scheduler_name="dp",
            )

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", self._make_full_dummy_q_cls()):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        # Patch scheduler.start so it doesn't do heavy work
        eng.scheduler.start = Mock()
        eng.launch_components()
        eng.scheduler.start.assert_called()
        self._detach_finalizer(eng)

    def test_insert_tasks_raises_when_no_resources(self):
        """Cover insert_tasks resource exhaustion error branch."""
        cfg = self._make_cfg(splitwise_role="mixed")

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", self._make_full_dummy_q_cls()):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)

        eng.resource_manager.stop_flags = np.zeros_like(eng.resource_manager.stop_flags)

        token_ids = paddle.to_tensor([1, 2, 3], dtype="int64")
        request = Request(
            request_id="req1",
            prompt_token_ids=token_ids.numpy().tolist(),
            prompt_token_ids_len=3,
        )
        with self.assertRaises(EngineError) as ctx:
            eng.insert_tasks([request])
        self.assertIn("request id", str(ctx.exception))
        self._detach_finalizer(eng)

    def test_get_scheduler_unhandled_request_num(self):
        """Cover _get_scheduler_unhandled_request_num normal/fallback paths."""
        eng = EngineService.__new__(EngineService)
        eng.llm_logger = Mock()

        # Scheduler does not provide API -> fallback 0
        eng.scheduler = object()
        self.assertEqual(eng._get_scheduler_unhandled_request_num(), 0)

        # Positive value -> return int value
        eng.scheduler = type("SchedOK", (), {"get_unhandled_request_num": lambda self: "3"})()
        self.assertEqual(eng._get_scheduler_unhandled_request_num(), 3)

        # Negative value -> clamp to 0
        eng.scheduler = type("SchedNeg", (), {"get_unhandled_request_num": lambda self: -5})()
        self.assertEqual(eng._get_scheduler_unhandled_request_num(), 0)

        # Exception -> debug log + fallback 0
        eng.scheduler = type(
            "SchedErr", (), {"get_unhandled_request_num": lambda self: (_ for _ in ()).throw(RuntimeError("boom"))}
        )()
        self.assertEqual(eng._get_scheduler_unhandled_request_num(), 0)
        eng.llm_logger.debug.assert_called()

    def test_insert_zmq_task_trace_carrier_handling(self):
        """Cover lines 1164-1167: trace_carrier handling in _insert_zmq_task_to_scheduler."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                self.available_prefill_instances = type("X", (), {"put": lambda *_: None})()

            def get_server_port(self):
                return 0

            def cleanup(self):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        eng.running = True

        # Mock data with trace_carrier to trigger lines 1164-1167
        test_request_id = "test_req_123"
        trace_carrier_data = {"trace_id": "abc123", "span_id": "def456"}
        mock_data_with_trace = {
            "request_id": test_request_id,
            "trace_carrier": trace_carrier_data,
            "status": None,
            "user": "test_user",
        }

        class DummyRecv:
            def __init__(self, data):
                self.data = data
                self.call_count = 0

            def receive_json_once(self, block):
                self.call_count += 1
                if self.call_count == 1:
                    return None, self.data
                else:
                    eng.running = False
                    return None, None

            def receive_pyobj_once(self, block):
                return self.receive_json_once(block)

            def close(self):
                pass

        eng.recv_request_server = DummyRecv(mock_data_with_trace)

        # Mock tracing.trace_set_proc_propagate_context to verify it's called
        with patch("fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context") as mock_trace_set:
            with patch.object(eng, "llm_logger"):
                with patch("fastdeploy.engine.common_engine.Request") as MockRequest:
                    mock_request = Mock()
                    mock_request.metrics.scheduler_recv_req_time = 0
                    MockRequest.from_dict.return_value = mock_request

                    with (
                        patch("fastdeploy.engine.common_engine.trace_print"),
                        patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", False),
                    ):
                        eng._insert_zmq_task_to_scheduler()

                        # Verify trace_set_proc_propagate_context was called with correct args (lines 1165-1167)
                        mock_trace_set.assert_called_once()
                        call_args = mock_trace_set.call_args
                        # request_id should be "test" (first part after split on "_") and trace_carrier
                        self.assertEqual(call_args[0][0], "test")
                        self.assertEqual(call_args[0][1], trace_carrier_data)

        # Reset and test without trace_carrier - should not call trace_set_proc_propagate_context
        eng.running = True
        mock_data_without_trace = {
            "request_id": "test_req_456",
            "status": None,
            "user": "test_user",
        }
        eng.recv_request_server = DummyRecv(mock_data_without_trace)

        with patch("fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context") as mock_trace_set:
            with patch.object(eng, "llm_logger"):
                with patch("fastdeploy.engine.common_engine.Request") as MockRequest:
                    mock_request = Mock()
                    mock_request.metrics.scheduler_recv_req_time = 0
                    MockRequest.from_dict.return_value = mock_request

                    with (
                        patch("fastdeploy.engine.common_engine.trace_print"),
                        patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", False),
                    ):
                        eng._insert_zmq_task_to_scheduler()

                        # Verify trace_set_proc_propagate_context was NOT called when no trace_carrier
                        mock_trace_set.assert_not_called()

        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_start_zmq_service_internal_adapter(self):
        """Cover lines 1107, 1110: start_zmq_service with FD_ENABLE_INTERNAL_ADAPTER=1."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        # Mock the necessary components
        eng.api_server_pid = 12345

        mock_tcp_server = Mock()
        mock_tcp_server.recv_result_handle = Mock()

        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", 1),
            patch("fastdeploy.engine.common_engine.envs.FD_ZMQ_RECV_REQUEST_SERVER_PORT", "6666"),
            patch("fastdeploy.engine.common_engine.envs.FD_ZMQ_SEND_RESPONSE_SERVER_PORT", "6667"),
            patch("fastdeploy.engine.common_engine.ZmqTcpServer", return_value=mock_tcp_server),
            patch("fastdeploy.engine.common_engine.InternalAdapter"),
            patch("fastdeploy.engine.common_engine.threading.Thread") as mock_thread,
            patch("fastdeploy.engine.common_engine.time.sleep"),
        ):
            eng.start_zmq_service(12345)

            # Verify thread was created for recv_result_handle (lines 1107-1110)
            self.assertTrue(mock_thread.called)
            # Check that thread was started
            for call in mock_thread.call_args_list:
                if "target" in call[1]:
                    thread_instance = mock_thread.return_value
                    thread_instance.start.assert_called()

        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_start_zmq_service_batch_mode(self):
        """Cover line 1115: start_zmq_service with ZMQ_SEND_BATCH_DATA=1."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        eng.api_server_pid = 12345

        mock_ipc_server = Mock()

        with (
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", 1),
            patch("fastdeploy.engine.common_engine.ZmqIpcServer", return_value=mock_ipc_server) as mock_server,
            patch("fastdeploy.engine.common_engine.time.sleep"),
        ):
            eng.start_zmq_service(12345)

            # Verify ZmqIpcServer was called with PUSH mode (line 1115)
            import zmq

            calls = mock_server.call_args_list
            push_mode_found = False
            for call in calls:
                # call[0] is positional args, call[1] is keyword args
                # The actual code uses: ZmqIpcServer(name=api_server_pid, mode=zmq.PUSH)
                # So mode is passed as a keyword argument
                if call[1].get("mode") == zmq.PUSH:
                    push_mode_found = True
                    break
            self.assertTrue(push_mode_found, "PUSH mode should be used when ZMQ_SEND_BATCH_DATA=1")

        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_insert_zmq_abort_request_paused(self):
        """Cover abort request handling: abort bypasses is_paused check and routes to add_abort_req_ids (v1)."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        eng.running = True
        eng.is_paused = True  # Engine is paused, but abort requests bypass this check

        abort_data = {
            "request_id": "abort_test_req",
            "status": 5,  # RequestStatus.ABORT.value
        }

        class DummyRecv:
            def __init__(self):
                self.call_count = 0

            def receive_json_once(self, block):
                self.call_count += 1
                if self.call_count == 1:
                    return None, abort_data
                else:
                    eng.running = False
                    return None, None

            def receive_pyobj_once(self, block):
                return self.receive_json_once(block)

            def close(self):
                pass

        eng.recv_request_server = DummyRecv()

        # Setup resource_manager with abort_req_ids_set
        eng.resource_manager.abort_req_ids_set = set()
        eng.resource_manager.add_abort_req_ids = Mock()

        with (
            patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1),
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", False),
            patch.object(eng, "llm_logger") as mock_logger,
            patch("fastdeploy.engine.common_engine.RequestStatus") as mock_status,
        ):
            mock_status.ABORT.value = 5
            eng._insert_zmq_task_to_scheduler()

            # Verify abort request was logged
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            abort_logged = any("abort" in call.lower() for call in info_calls)
            self.assertTrue(abort_logged, "Should log 'Receive abort request'")

            # Verify add_abort_req_ids was called (v1 scheduler path)
            eng.resource_manager.add_abort_req_ids.assert_called_once_with("abort_test_req")

        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_insert_zmq_abort_request_in_requests(self):
        """Cover abort request handling: when ENABLE_V1_KVCACHE_SCHEDULER=1, add_abort_req_ids is called."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        eng.running = True
        eng.is_paused = False

        abort_data = {
            "request_id": "abort_in_requests",
            "status": 5,  # RequestStatus.ABORT.value
        }

        class DummyRecv:
            def __init__(self):
                self.call_count = 0

            def receive_json_once(self, block):
                self.call_count += 1
                if self.call_count == 1:
                    return None, abort_data
                else:
                    eng.running = False
                    return None, None

            def receive_pyobj_once(self, block):
                return self.receive_json_once(block)

            def close(self):
                pass

        eng.recv_request_server = DummyRecv()
        eng.resource_manager.abort_req_ids_set = set()

        # Mock add_abort_req_ids on resource_manager
        eng.resource_manager.add_abort_req_ids = Mock()

        with (
            patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1),
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", False),
            patch.object(eng, "llm_logger"),
            patch("fastdeploy.engine.common_engine.RequestStatus") as mock_status,
        ):
            mock_status.ABORT.value = 5
            eng._insert_zmq_task_to_scheduler()

            # Verify add_abort_req_ids was called with the correct req_id (v1 scheduler path)
            eng.resource_manager.add_abort_req_ids.assert_called_once_with("abort_in_requests")

        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_run_control_method_with_batch_data(self):
        """Cover lines 1283, 1284, 1290, 1291, 1297, 1298: run_control_method with ZMQ_SEND_BATCH_DATA=1."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        # Mock send_response_server
        eng.send_response_server = Mock()
        eng.send_response_server.send_response = Mock()

        control_req = Mock()
        control_req.get_method.return_value = "is_paused"  # Use existing method
        control_req.request_id = "control_test_123"

        with (
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", 1),
            patch.object(eng, "llm_logger"),
            patch.object(eng, "_control_is_paused") as mock_handler,
        ):
            mock_handler.return_value = {"is_paused": False}
            eng.run_control_method(control_req)

            # Verify send_response was called with 2D array (line 1291)
            eng.send_response_server.send_response.assert_called_once()
            call_args = eng.send_response_server.send_response.call_args
            data = call_args[0][1]
            # Should be [[response]] format for batch mode
            self.assertIsInstance(data, list)
            self.assertIsInstance(data[0], list)

        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_run_control_method_unknown_with_batch_data(self):
        """Cover lines 1283-1284: unknown control method with ZMQ_SEND_BATCH_DATA=1."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        eng.send_response_server = Mock()
        eng.send_response_server.send_response = Mock()

        control_req = Mock()
        control_req.get_method.return_value = "unknown_method"
        control_req.request_id = "control_unknown"

        with (
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", 1),
            patch.object(eng, "llm_logger"),
        ):
            eng.run_control_method(control_req)

            # Verify send_response was called with error response (lines 1283-1284)
            eng.send_response_server.send_response.assert_called_once()
            call_args = eng.send_response_server.send_response.call_args
            data = call_args[0][1]
            # Should be [[error_response]] format
            self.assertIsInstance(data, list)
            self.assertIsInstance(data[0], list)

        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_send_error_response_with_batch_data(self):
        """Cover lines 1467, 1468: _send_error_response with ZMQ_SEND_BATCH_DATA=1."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        eng.send_response_server = Mock()
        eng.send_response_server.send_response = Mock()

        with (
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", 1),
            patch.object(eng, "llm_logger"),
        ):
            eng._send_error_response("test_req_id", "Test error message", 500)

            # Verify send_response was called with 2D array format (lines 1467-1468)
            eng.send_response_server.send_response.assert_called_once()
            call_args = eng.send_response_server.send_response.call_args
            data = call_args[0][1]
            # Should be [[error_result]] format
            self.assertIsInstance(data, list)
            self.assertIsInstance(data[0], list)

        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_zmq_send_generated_tokens_batch_mode(self):
        """Cover lines 1530, 1557-1563: _zmq_send_generated_tokens with ZMQ_SEND_BATCH_DATA=1."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)

        # Initialize request_worker_map for batch mode routing
        import threading as _threading

        eng.request_worker_map = {}
        eng.request_worker_map_lock = _threading.Lock()

        # Setup scheduler to return results
        mock_output = Mock()
        mock_output.outputs = Mock()
        mock_output.outputs.token_ids = [1, 2, 3]
        mock_output.outputs.decode_type = 1  # Not decode_type 0
        mock_output.finished = False
        mock_output.request_id = "test_req"

        eng.scheduler = Mock()
        eng.scheduler.get_results.return_value = {"test_req": [mock_output]}

        eng.send_response_server = Mock()
        eng.send_response_server.send_response = Mock()

        # Make the loop run only once
        call_count = [0]

        def get_results_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"test_req": [mock_output]}
            else:
                eng.running = False
                return {}

        eng.scheduler.get_results.side_effect = get_results_side_effect

        with (
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", 1),
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", 0),
            patch.object(eng, "llm_logger"),
        ):
            eng.running = True
            eng._zmq_send_generated_tokens()

            # Verify send_response was called with batch_data (lines 1557-1563)
            eng.send_response_server.send_response.assert_called_once()
            call_args = eng.send_response_server.send_response.call_args
            # First arg should be None, second should be batch_data (list of lists)
            self.assertIsNone(call_args[0][0])
            batch_data = call_args[0][1]
            self.assertIsInstance(batch_data, list)

        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_run_control_method_exception_with_batch_data(self):
        """Cover lines 1297-1298: run_control_method exception handling with ZMQ_SEND_BATCH_DATA=1."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        eng.send_response_server = Mock()
        eng.send_response_server.send_response = Mock()

        control_req = Mock()
        control_req.get_method.return_value = "is_paused"  # Use existing method
        control_req.request_id = "control_exception"

        with (
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", 1),
            patch.object(eng, "llm_logger"),
            patch.object(eng, "_control_is_paused", side_effect=RuntimeError("Test exception")),
        ):
            eng.run_control_method(control_req)

            # Verify send_response was called with error response (lines 1297-1298)
            eng.send_response_server.send_response.assert_called_once()
            call_args = eng.send_response_server.send_response.call_args
            data = call_args[0][1]
            # Should be [[error_response]] format
            self.assertIsInstance(data, list)
            self.assertIsInstance(data[0], list)

        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # New tests targeting uncovered violation lines
    # -----------------------------------------------------------------------

    def test_insert_zmq_task_control_request_with_worker_pid(self):
        """Lines 1183-1189: control request when ZMQ_SEND_BATCH_DATA=True maps worker_pid and calls run_control_method."""
        eng = self._make_mixed_engine()
        eng.running = True
        eng.is_paused = False
        eng.guided_decoding_checker = None
        eng.resource_manager = Mock(abort_req_ids_set=set(), requests={})
        eng.scheduler = Mock()
        eng.engine_worker_queue = Mock()
        eng.run_control_method = Mock()

        import threading as _threading

        eng.request_worker_map = {}
        eng.request_worker_map_lock = _threading.Lock()

        ctrl_data = {
            "request_id": "ctrl-batch",
            "method": "is_paused",
            "args": {},
            "zmq_worker_pid": 9999,
        }

        class DummyRecv:
            def __init__(self):
                self.calls = 0

            def receive_json_once(self, block):
                self.calls += 1
                if self.calls == 1:
                    return None, ctrl_data
                eng.running = False
                return None, None

            def receive_pyobj_once(self, block):
                return self.receive_json_once(block)

            def close(self):
                pass

        eng.recv_request_server = DummyRecv()

        with (
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", True),
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._insert_zmq_task_to_scheduler()

        # worker_pid should be stored in request_worker_map for the control request
        self.assertIn("ctrl-batch", eng.request_worker_map)
        self.assertEqual(eng.request_worker_map["ctrl-batch"], 9999)
        eng.run_control_method.assert_called_once()
        self._detach_finalizer(eng)

    def test_insert_zmq_task_control_request_exception_with_worker_pid(self):
        """Lines 1188-1189: exception during control request processing is caught and logged."""
        eng = self._make_mixed_engine()
        eng.running = True
        eng.is_paused = False
        eng.guided_decoding_checker = None
        eng.resource_manager = Mock(abort_req_ids_set=set(), requests={})
        eng.scheduler = Mock()
        eng.engine_worker_queue = Mock()
        eng.run_control_method = Mock(side_effect=RuntimeError("ctrl boom"))

        import threading as _threading

        eng.request_worker_map = {}
        eng.request_worker_map_lock = _threading.Lock()

        ctrl_data = {
            "request_id": "ctrl-err",
            "method": "is_paused",
            "args": {},
            "zmq_worker_pid": 1111,
        }

        class DummyRecv:
            def __init__(self):
                self.calls = 0

            def receive_json_once(self, block):
                self.calls += 1
                if self.calls == 1:
                    return None, ctrl_data
                eng.running = False
                return None, None

            def receive_pyobj_once(self, block):
                return self.receive_json_once(block)

            def close(self):
                pass

        eng.recv_request_server = DummyRecv()

        with (
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", True),
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False),
            patch.object(eng, "llm_logger") as mock_logger,
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._insert_zmq_task_to_scheduler()

        mock_logger.error.assert_called()
        self._detach_finalizer(eng)

    def test_insert_zmq_task_normal_request_with_worker_pid(self):
        """Lines 1204-1207: normal request stores worker_pid in request_worker_map; abort request handled."""
        eng = self._make_mixed_engine()
        eng.running = True
        eng.is_paused = False
        eng.guided_decoding_checker = None
        eng.resource_manager = Mock(abort_req_ids_set=set(), requests={})
        eng.scheduler = Mock()
        eng.engine_worker_queue = Mock()

        import threading as _threading

        eng.request_worker_map = {}
        eng.request_worker_map_lock = _threading.Lock()

        normal_data = {
            "request_id": "normal-batch",
            "prompt_token_ids": [1, 2],
            "prompt_token_ids_len": 2,
            "temperature": 1.0,
            "zmq_worker_pid": 7777,
        }

        class DummyRecv:
            def __init__(self):
                self.calls = 0

            def receive_json_once(self, block):
                self.calls += 1
                if self.calls == 1:
                    return None, normal_data
                eng.running = False
                return None, None

            def receive_pyobj_once(self, block):
                return self.receive_json_once(block)

            def close(self):
                pass

        eng.recv_request_server = DummyRecv()
        eng.scheduler.put_requests.return_value = [("normal-batch", None)]

        class DummyMetrics:
            def __init__(self):
                self.requests_number = Mock(inc=Mock())
                self.num_requests_waiting = Mock(inc=Mock())

        with (
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", True),
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False),
            patch("fastdeploy.engine.common_engine.main_process_metrics", DummyMetrics()),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._insert_zmq_task_to_scheduler()

        # worker_pid for normal request should be stored
        self.assertIn("normal-batch", eng.request_worker_map)
        self.assertEqual(eng.request_worker_map["normal-batch"], 7777)
        self._detach_finalizer(eng)

    def test_insert_zmq_task_abort_request_with_worker_pid(self):
        """Lines 1206-1207: abort request with worker_pid stores mapping then continues."""
        eng = self._make_mixed_engine()
        eng.running = True
        eng.is_paused = False
        eng.guided_decoding_checker = None

        import threading as _threading

        eng.request_worker_map = {}
        eng.request_worker_map_lock = _threading.Lock()

        eng.resource_manager = Mock(abort_req_ids_set=set(), requests={})
        eng.resource_manager.add_abort_req_ids = Mock()
        eng.scheduler = Mock()
        eng.engine_worker_queue = Mock()

        abort_data = {
            "request_id": "abort-worker",
            "status": RequestStatus.ABORT.value,
            "zmq_worker_pid": 4444,
        }

        class DummyRecv:
            def __init__(self):
                self.calls = 0

            def receive_json_once(self, block):
                self.calls += 1
                if self.calls == 1:
                    return None, abort_data
                eng.running = False
                return None, None

            def receive_pyobj_once(self, block):
                return self.receive_json_once(block)

            def close(self):
                pass

        eng.recv_request_server = DummyRecv()

        with (
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", True),
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False),
            patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._insert_zmq_task_to_scheduler()

        # worker_pid stored for abort request
        self.assertIn("abort-worker", eng.request_worker_map)
        self.assertEqual(eng.request_worker_map["abort-worker"], 4444)
        eng.resource_manager.add_abort_req_ids.assert_called_once_with("abort-worker")
        self._detach_finalizer(eng)

    def test_run_control_method_logging_with_request_worker_map(self):
        """Lines 1299-1300: run_control_method logs start when ZMQ_SEND_BATCH_DATA=True with request_worker_map."""
        eng = self._make_mixed_engine()
        eng.send_response_server = Mock()
        eng._pause_cond = threading.Condition()

        import threading as _threading

        eng.request_worker_map = {"ctrl-log": 5555}
        eng.request_worker_map_lock = _threading.Lock()

        ctrl_req = ControlRequest(request_id="ctrl-log", method="is_paused")
        eng.is_paused = False

        with (
            patch("fastdeploy.engine.common_engine.envs.ZMQ_SEND_BATCH_DATA", True),
            patch.object(eng, "llm_logger") as mock_logger,
        ):
            eng.run_control_method(ctrl_req)

        # Lines 1299-1300: try block start + info logging
        info_msgs = [str(c) for c in mock_logger.info.call_args_list]
        self.assertTrue(any("Start to run control method" in m for m in info_msgs))
        # worker_pid should be popped from the map
        self.assertNotIn("ctrl-log", eng.request_worker_map)
        self._detach_finalizer(eng)

    def test_decode_token_return_text_non_empty_delta_is_end_deletes_status(self):
        """Lines 1510-1511: _decode_token with non-empty delta and is_end=True deletes decode_status entry."""
        eng = self._make_mixed_engine()

        class DummyProcessor:
            def __init__(self):
                self.decode_status = {"tok-req": (1, 3)}

            def ids2tokens(self, token_ids, req_id):
                return "hello", [10, 20, 30], None

        eng.data_processor = DummyProcessor()

        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True):
            delta, ids = eng._decode_token([10, 20, 30], "tok-req", is_end=True)

        self.assertEqual(delta, "hello")
        # decode_status key should be deleted (line 1511)
        self.assertNotIn("tok-req", eng.data_processor.decode_status)
        self._detach_finalizer(eng)

    def test_decode_process_splitwise_requests_empty_queue_returns_early(self):
        """Lines 1613-1614: _fetch_requests returns early when disaggregate_queue_empty() is True."""
        cfg = self._make_cfg(
            splitwise_role="decode",
            num_gpu_blocks_override=4,
            router="0.0.0.0:30000",
        )
        eng = self._make_engine(cfg)
        eng.running = True
        eng.enable_decode_cache_task = False
        eng.cfg.splitwise_version = "v1"
        eng.scheduler = Mock(has_request=Mock(return_value=True), put_results=Mock())
        eng._insert_prefilled_requests = Mock()
        eng.insert_tasks = Mock()

        class DummyRM:
            def is_resource_sufficient(self, prompt_len):
                return True

        eng.resource_manager = DummyRM()

        empty_queue_call_count = [0]

        class DummyQueueAlwaysEmpty:
            def disaggregate_queue_empty(self):
                empty_queue_call_count[0] += 1
                # Return empty on first call then stop the engine
                eng.running = False
                return True

            def get_disaggregated_tasks(self):
                return []

        eng.engine_worker_queue = DummyQueueAlwaysEmpty()

        class DummyThread:
            def __init__(self, target=None, daemon=None):
                self.target = target

            def start(self):
                try:
                    self.target()
                finally:
                    eng.running = False

        with (
            patch("fastdeploy.engine.common_engine.threading.Thread", DummyThread),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
            patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False),
        ):
            eng._decode_process_splitwise_requests()

        # Queue was seen as empty so get_disaggregated_tasks should not be called
        self.assertEqual(empty_queue_call_count[0], 1)
        eng.insert_tasks.assert_not_called()
        self._detach_finalizer(eng)

    def test_register_to_router_inner_function_runs(self):
        """_register inner function body executes (timeout and sleep_seconds set)."""
        from fastdeploy.engine.register_manager import RegisterManager

        eng = self._make_mixed_engine()
        eng.cfg.router_config.router = "http://fake-router"
        eng.cfg.router_config.api_server_host = "127.0.0.1"
        eng.cfg.router_config.api_server_port = 19999
        eng.cfg.register_info = {"name": "test-server"}

        reg_mgr = RegisterManager(
            cfg=eng.cfg,
            engine_worker_queue=MagicMock(),
            get_is_paused=lambda: False,
        )

        captured_target = [None]

        class _CapturingThread:
            def __init__(self, target=None, daemon=None):
                captured_target[0] = target
                self.target = target
                self.daemon = daemon

            def start(self):
                pass  # don't auto-start

        with patch("fastdeploy.engine.register_manager.threading.Thread", _CapturingThread):
            reg_mgr._register_to_router()

        # Verify the inner _register function was captured
        self.assertIsNotNone(captured_target[0])

        # Now invoke the inner _register function directly.
        # Mock out check_service_health to return False so it doesn't hang,
        # and time.sleep to raise StopIteration to break the while True loop.
        call_count = [0]

        def _fake_sleep(s):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise StopIteration("stop")

        with (
            patch("fastdeploy.engine.register_manager.check_service_health", return_value=False),
            patch("fastdeploy.engine.register_manager.time.sleep", _fake_sleep),
        ):
            try:
                captured_target[0]()
            except StopIteration:
                pass

        # At least one sleep call was made, confirming the inner function executed
        self.assertGreaterEqual(call_count[0], 1)
        self._detach_finalizer(eng)

    # ── _control_abort_requests / _wait_abort_complete ───────────────

    def _make_abort_engine(self, splitwise_role="mixed"):
        """Create an engine wired up for abort tests."""
        extra = {}
        if splitwise_role != "mixed":
            extra["router"] = "0.0.0.0:9000"
        cfg = self._make_cfg(splitwise_role=splitwise_role, num_gpu_blocks_override=4, **extra)
        eng = self._make_engine(cfg)
        eng.llm_logger = MagicMock()

        # data_processor with eos token
        eng.data_processor = MagicMock()
        eng.data_processor.eos_token_ids = [2]

        # resource_manager with requests dict and abort sets
        eng.resource_manager = MagicMock()
        eng.resource_manager.requests = {}
        eng.resource_manager.waiting_abort_req_id_set = set()
        eng.resource_manager.to_be_aborted_req_id_set = set()
        eng.resource_manager.get_reqs_in_aborting = lambda: (
            eng.resource_manager.waiting_abort_req_id_set | eng.resource_manager.to_be_aborted_req_id_set
        )

        # scheduler with requests dict and put_results
        eng.scheduler = MagicMock()
        eng.scheduler.requests = {}
        eng.scheduler.put_results = MagicMock()

        return eng

    def _make_fake_request(self, output_token_ids=None):
        """Create a fake request object for abort tests."""
        req = MagicMock()
        req.output_token_ids = output_token_ids or [10, 20, 30]
        req.metrics = MagicMock()
        req.metrics.arrival_time = 1000.0
        req.metrics.inference_start_time = 1000.1
        req.metrics.engine_recv_first_token_time = 1000.2
        return req

    def test_control_abort_requests_not_v1_raises(self):
        """abort_requests raises when ENABLE_V1_KVCACHE_SCHEDULER is off."""
        eng = self._make_abort_engine()
        control_req = ControlRequest("ctrl-1", "abort_requests", {"abort_all": True, "req_ids": []})
        with patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0):
            with self.assertRaises(Exception) as ctx:
                eng._control_abort_requests(control_req)
            self.assertIn("only supported", str(ctx.exception))
        self._detach_finalizer(eng)

    def test_control_abort_requests_abort_all(self):
        """abort_all=True aborts all requests in resource_manager + scheduler."""
        eng = self._make_abort_engine()
        eng.resource_manager.requests = {"req-1_0": self._make_fake_request([10, 20])}
        eng.scheduler.requests = {"req-2_0": MagicMock(raw=self._make_fake_request([30]))}

        control_req = ControlRequest("ctrl-1", "abort_requests", {"abort_all": True, "req_ids": []})

        def clear_abort_sets(req_id):
            # Simulate immediate abort completion
            eng.resource_manager.waiting_abort_req_id_set.discard(req_id)

        eng.resource_manager.add_abort_req_ids = MagicMock(side_effect=clear_abort_sets)

        with patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1):
            result = eng._control_abort_requests(control_req)

        self.assertEqual(len(result["aborted"]), 2)
        self.assertEqual(result["not_found"], [])
        ids = {a["request_id"] for a in result["aborted"]}
        self.assertEqual(ids, {"req-1_0", "req-2_0"})
        # put_results should have been called (not prefill)
        eng.scheduler.put_results.assert_called_once()
        self._detach_finalizer(eng)

    def test_control_abort_requests_by_req_ids_with_suffix_match(self):
        """req_ids match both exact and _0 suffix."""
        eng = self._make_abort_engine()
        eng.resource_manager.requests = {
            "req-A_0": self._make_fake_request([1, 2, 3]),
            "req-B": self._make_fake_request([4, 5]),
        }

        control_req = ControlRequest(
            "ctrl-1",
            "abort_requests",
            {
                "abort_all": False,
                "req_ids": ["req-A", "req-B", "req-C"],
            },
        )

        def clear_abort_sets(req_id):
            eng.resource_manager.waiting_abort_req_id_set.discard(req_id)

        eng.resource_manager.add_abort_req_ids = MagicMock(side_effect=clear_abort_sets)

        with patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1):
            result = eng._control_abort_requests(control_req)

        aborted_ids = {a["request_id"] for a in result["aborted"]}
        self.assertIn("req-A_0", aborted_ids)  # matched via _0 suffix
        self.assertIn("req-B", aborted_ids)  # exact match
        self.assertEqual(result["not_found"], ["req-C"])
        self._detach_finalizer(eng)

    def test_control_abort_requests_no_match(self):
        """No requests found returns empty aborted and all in not_found."""
        eng = self._make_abort_engine()
        control_req = ControlRequest(
            "ctrl-1",
            "abort_requests",
            {
                "abort_all": False,
                "req_ids": ["nonexistent"],
            },
        )

        with patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1):
            result = eng._control_abort_requests(control_req)

        self.assertEqual(result["aborted"], [])
        self.assertEqual(result["not_found"], ["nonexistent"])
        self._detach_finalizer(eng)

    def test_control_abort_requests_prefill_skips_wait_and_put(self):
        """Prefill role skips _wait_abort_complete and put_results."""
        eng = self._make_abort_engine(splitwise_role="prefill")
        eng.resource_manager.requests = {"req-1_0": self._make_fake_request()}

        control_req = ControlRequest("ctrl-1", "abort_requests", {"abort_all": True, "req_ids": []})
        eng.resource_manager.add_abort_req_ids = MagicMock()

        with patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1):
            result = eng._control_abort_requests(control_req)

        self.assertEqual(len(result["aborted"]), 1)
        eng.scheduler.put_results.assert_not_called()
        self._detach_finalizer(eng)

    def test_control_abort_requests_output_token_count(self):
        """output_token_count reflects partial_token_ids length."""
        eng = self._make_abort_engine()
        eng.resource_manager.requests = {"req-1_0": self._make_fake_request([10, 20, 30, 40, 50])}

        control_req = ControlRequest("ctrl-1", "abort_requests", {"abort_all": True, "req_ids": []})

        def clear_abort_sets(req_id):
            eng.resource_manager.waiting_abort_req_id_set.discard(req_id)

        eng.resource_manager.add_abort_req_ids = MagicMock(side_effect=clear_abort_sets)

        with patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1):
            result = eng._control_abort_requests(control_req)

        self.assertEqual(result["aborted"][0]["output_token_count"], 5)
        self._detach_finalizer(eng)

    def test_wait_abort_complete_immediate(self):
        """_wait_abort_complete returns immediately when all requests already cleaned."""
        eng = self._make_abort_engine()
        # Empty abort sets → remaining is empty → returns immediately
        eng._wait_abort_complete(["req-1_0"])
        self._detach_finalizer(eng)

    def test_wait_abort_complete_progress(self):
        """_wait_abort_complete exits when background thread cleans up."""
        eng = self._make_abort_engine()
        eng.resource_manager.waiting_abort_req_id_set = {"req-1_0"}

        call_count = [0]

        def fake_sleep(s):
            call_count[0] += 1
            # Simulate background thread cleaning up after first sleep
            eng.resource_manager.waiting_abort_req_id_set.discard("req-1_0")

        with patch("fastdeploy.engine.common_engine.time.sleep", fake_sleep):
            eng._wait_abort_complete(["req-1_0"])

        self.assertGreaterEqual(call_count[0], 1)
        self._detach_finalizer(eng)

    def test_wait_abort_complete_force_cleanup_stuck_in_to_be_aborted(self):
        """Stall timeout triggers force cleanup for requests in to_be_aborted_req_id_set."""
        eng = self._make_abort_engine()
        eng.resource_manager.to_be_aborted_req_id_set = {"req-1_0"}

        def mock_recycle(req_id):
            eng.resource_manager.to_be_aborted_req_id_set.discard(req_id)

        eng.resource_manager.recycle_abort_task = MagicMock(side_effect=mock_recycle)

        # Make time.time() advance past stall_timeout
        time_values = [100.0, 100.0, 102.0, 102.0, 102.0]
        time_idx = [0]

        def fake_time():
            idx = min(time_idx[0], len(time_values) - 1)
            time_idx[0] += 1
            return time_values[idx]

        with (
            patch("fastdeploy.engine.common_engine.time.time", fake_time),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda s: None),
        ):
            eng._wait_abort_complete(["req-1_0"], stall_timeout=1)

        eng.resource_manager.recycle_abort_task.assert_called_with("req-1_0")
        self._detach_finalizer(eng)


class TestWorkerTracebackFunctions(unittest.TestCase):
    """测试 _read_latest_worker_traceback 和 _format_worker_launch_failure_message 函数"""

    def test_read_latest_worker_traceback_finds_traceback(self):
        """测试能够正确读取 workerlog 文件中的 traceback"""
        with tempfile.TemporaryDirectory() as temp_dir:
            worker_log = os.path.join(temp_dir, "workerlog.0")
            with open(worker_log, "w", encoding="utf-8") as fp:
                fp.write(
                    "Some normal log output\n"
                    "Traceback (most recent call last):\n"
                    '  File "worker_process.py", line 1, in <module>\n'
                    "    run_worker_proc()\n"
                    "ValueError: The total number of blocks cannot be less than zero.\n"
                )

            result = _read_latest_worker_traceback(temp_dir)
            self.assertIsNotNone(result)
            self.assertIn("Traceback (most recent call last):", result)
            self.assertIn("ValueError:", result)

    def test_read_latest_worker_traceback_returns_none_when_no_traceback(self):
        """测试当没有 traceback 时返回 None"""
        with tempfile.TemporaryDirectory() as temp_dir:
            worker_log = os.path.join(temp_dir, "workerlog.0")
            with open(worker_log, "w", encoding="utf-8") as fp:
                fp.write("Normal log output without any errors\n")

            result = _read_latest_worker_traceback(temp_dir)
            self.assertIsNone(result)

    def test_read_latest_worker_traceback_returns_none_when_no_files(self):
        """测试当没有 workerlog 文件时返回 None"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = _read_latest_worker_traceback(temp_dir)
            self.assertIsNone(result)

    def test_read_latest_worker_traceback_returns_none_for_nonexistent_dir(self):
        """测试当目录不存在时返回 None"""
        result = _read_latest_worker_traceback("/nonexistent/path")
        self.assertIsNone(result)

    def test_read_latest_worker_traceback_picks_latest_file(self):
        """测试当有多个 workerlog 文件时选择最新的"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建较旧的文件
            old_log = os.path.join(temp_dir, "workerlog.0")
            with open(old_log, "w", encoding="utf-8") as fp:
                fp.write("Traceback (most recent call last):\nOldError: old error\n")

            # 短暂等待以确保时间戳不同
            time.sleep(0.01)

            # 创建较新的文件
            new_log = os.path.join(temp_dir, "workerlog.1")
            with open(new_log, "w", encoding="utf-8") as fp:
                fp.write("Traceback (most recent call last):\nNewError: new error\n")

            result = _read_latest_worker_traceback(temp_dir)
            self.assertIsNotNone(result)
            self.assertIn("NewError", result)

    def test_format_worker_launch_failure_message_with_traceback(self):
        """测试带有 traceback 的错误消息格式化"""
        with tempfile.TemporaryDirectory() as temp_dir:
            worker_log = os.path.join(temp_dir, "workerlog.0")
            with open(worker_log, "w", encoding="utf-8") as fp:
                fp.write("Traceback (most recent call last):\n" "ValueError: Test error message\n")

            result = _format_worker_launch_failure_message(temp_dir)
            self.assertIn("Failed to launch worker processes", result)
            self.assertIn("workerlog.*", result)
            self.assertIn("Traceback (most recent call last):", result)
            self.assertIn("ValueError: Test error message", result)

    def test_format_worker_launch_failure_message_without_traceback(self):
        """测试没有 traceback 时的错误消息格式化"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = _format_worker_launch_failure_message(temp_dir)
            self.assertIn("Failed to launch worker processes", result)
            self.assertIn("workerlog.*", result)
            self.assertNotIn("Traceback", result)
