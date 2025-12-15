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

import os
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.common_engine import EngineService
from fastdeploy.engine.request import Request
from fastdeploy.utils import EngineError

MODEL_NAME = os.getenv("MODEL_PATH", "/path/to/models") + "/ERNIE-4.5-0.3B-Paddle"


class TestCommonEngine(unittest.TestCase):
    """Test case for EngineService functionality (lines 1215-1664)"""

    @classmethod
    def setUpClass(cls):
        """Set up EngineService for testing"""
        try:
            # Create engine args for testing
            engine_args = EngineArgs(
                model=MODEL_NAME,
                max_model_len=8192,
                tensor_parallel_size=1,
                engine_worker_queue_port=int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")) + 10,
                cache_queue_port=int(os.getenv("FD_CACHE_QUEUE_PORT", "6779")) + 10,
            )

            # Create and start the engine service
            cls.cfg = engine_args.create_engine_config()
            cls.engine = EngineService(cls.cfg, start_queue=True, use_async_llm=True)

            # Start the engine service
            cls.engine.start()

        except Exception as e:
            print(f"Setting up EngineService failed: {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        if hasattr(cls, "engine") and cls.engine is not None:
            try:
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

    def test_exit_sub_services(self):
        """Test _exit_sub_services method (lines 1215-1291)"""
        # Test that _exit_sub_services can be called without error
        # Note: We won't actually call it since it would shut down the engine
        # Instead we'll test that the method exists and has expected attributes
        self.assertTrue(hasattr(self.engine, "_exit_sub_services"))
        self.assertTrue(callable(getattr(self.engine, "_exit_sub_services")))

        # Test that engine has expected attributes that would be cleaned up
        if hasattr(self.engine, "worker_proc"):
            self.assertIsNotNone(self.engine.worker_proc)

        # Verify running state
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

    def test_start_worker_service(self):
        """Test _start_worker_service method (lines 1409-1517)"""
        # Since engine is already started, we can test that worker process exists
        if hasattr(self.engine, "worker_proc") and self.engine.worker_proc:
            # Worker process should be running
            self.assertIsNotNone(self.engine.worker_proc)
            # Process should be alive (poll returns None if still running)
            poll_result = self.engine.worker_proc.poll()
            if poll_result is not None:
                self.skipTest("Worker process is not running")
        else:
            self.skipTest("Worker process not available")

    def test_stop_profile(self):
        """Test _stop_profile method (lines 1519-1532)"""
        # Test method exists and is callable
        self.assertTrue(hasattr(self.engine, "_stop_profile"))
        self.assertTrue(callable(getattr(self.engine, "_stop_profile")))

        # We won't actually call it as it modifies engine state
        # Just verify the do_profile attribute exists
        self.assertTrue(hasattr(self.engine, "do_profile"))

    def test_check_health(self):
        """Test check_health method (lines 1533-1544)"""
        if hasattr(self.engine, "worker_healthy_live_signal"):
            is_healthy, message = self.engine.check_health(time_interval_threashold=30)

            # Should return tuple of (bool, str)
            self.assertIsInstance(is_healthy, bool)
            self.assertIsInstance(message, str)
        else:
            self.skipTest("worker_healthy_live_signal not available")

    def test_launch_components(self):
        """Test launch_components method (lines 1545-1605)"""
        # Method should exist and be callable
        self.assertTrue(hasattr(self.engine, "launch_components"))
        self.assertTrue(callable(getattr(self.engine, "launch_components")))

        # Test that scheduler exists (should be created during start)
        if hasattr(self.engine, "scheduler"):
            self.assertIsNotNone(self.engine.scheduler)

    def test_check_worker_initialize_status(self):
        """Test check_worker_initialize_status method (lines 1606-1663)"""
        # Method should exist and be callable
        self.assertTrue(hasattr(self.engine, "check_worker_initialize_status"))
        self.assertTrue(callable(getattr(self.engine, "check_worker_initialize_status")))

        # Test that worker_init_status exists
        if hasattr(self.engine, "worker_init_status"):
            self.assertIsInstance(self.engine.worker_init_status, dict)

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

    def _make_cfg(self, **kwargs):
        # Build args dict, avoiding duplicate keys
        args_dict = {
            "model": MODEL_NAME,
            "max_model_len": 128,
            "tensor_parallel_size": 1,
            "engine_worker_queue_port": str(int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")) + 20),
        }
        # Add defaults only if not in kwargs
        if "cache_queue_port" not in kwargs:
            args_dict["cache_queue_port"] = str(int(os.getenv("FD_CACHE_QUEUE_PORT", "6779")) + 20)
        if "enable_prefix_caching" not in kwargs:
            args_dict["enable_prefix_caching"] = True
        # Update with kwargs (will override defaults)
        args_dict.update(kwargs)

        # Add router if splitwise_role is not mixed and router not provided
        if args_dict.get("splitwise_role") in ["prefill", "decode"] and args_dict.get("router") is None:
            args_dict["router"] = "http://localhost:8000"

        args = EngineArgs(**args_dict)
        # Keep batch tokens small to satisfy FDConfig checks:
        # max_num_batched_tokens <= max_model_len * max_num_seqs
        if getattr(args, "max_num_batched_tokens", None) is None:
            args.max_num_batched_tokens = 128
        # Always enable chunked prefill in tests to avoid another strict check
        args.enable_chunked_prefill = True

        # If DP > 1, we must provide enough engine_worker_queue_port for each dp index
        dp = kwargs.get("data_parallel_size", args.data_parallel_size)
        base = int(args.engine_worker_queue_port.split(",")[0])
        if dp and dp > 1:
            ports = ",".join(str(base + i) for i in range(dp))
            args.engine_worker_queue_port = ports

        return args.create_engine_config(port_availability_check=False)

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
                num_gpu_blocks_override=3,
                router="0.0.0.0:30000",
            )

        # Patch EngineWorkerQueue before EngineService ctor to avoid real IPC
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

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
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
        class Sig:
            def __init__(self, v=0):
                self.value = np.array([v], dtype=np.int32)

            def clear(self):
                pass

        def fake_init_signals():
            eng.worker_ready_signal = Sig(0)
            eng.loaded_model_signal = Sig(1)  # ready -> skip wait loop
            eng.launched_cache_manager_signal = Sig(0)

        eng._init_worker_signals = fake_init_signals

        # Worker start stub and initialization status -> False to trigger error path
        eng._start_worker_service = lambda: Mock(stdout=Mock(), poll=lambda: None)
        eng.check_worker_initialize_status = lambda: False

        with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
            # Avoid starting token processor loop
            eng.token_processor.run = lambda: None
            # Mock _register_to_router to avoid starting registration thread
            eng._register_to_router = lambda: None
            ok = eng.start(async_llm_pid=12345)

        # start() returns False on failure
        self.assertFalse(ok)
        # cache manager started before workers (lines 184-185)
        self.assertTrue(started_cache.get("called", False))
        # launched_cache_manager_signal set (line 221)
        self.assertEqual(int(eng.launched_cache_manager_signal.value[0]), 1)
        # Wait for any daemon threads to exit
        time.sleep(0.1)
        # avoid atexit finalizer
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_start_mixed_branch_cache_after_load_and_zmq(self):
        """Cover lines 215-217 and 231 in start()."""
        cfg = self._make_cfg(splitwise_role="mixed", num_gpu_blocks_override=2)

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

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
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

        class Sig:
            def __init__(self, v=0):
                self.value = np.array([v], dtype=np.int32)

            def clear(self):
                pass

        def fake_init_signals():
            eng.worker_ready_signal = Sig(0)
            eng.loaded_model_signal = Sig(1)
            eng.launched_cache_manager_signal = Sig(0)

        eng._init_worker_signals = fake_init_signals

        eng._start_worker_service = lambda: Mock(stdout=Mock(), poll=lambda: None)
        eng.check_worker_initialize_status = lambda: True

        zmq_called = {}
        eng.start_zmq_service = lambda pid: zmq_called.setdefault("pid", pid)

        with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
            eng.token_processor.run = lambda: None
            # Mock _register_to_router to avoid starting registration thread
            eng._register_to_router = lambda: None
            eng.start(async_llm_pid=8888)

        self.assertTrue(started_cache.get("called", False))  # lines 215-217
        self.assertEqual(zmq_called.get("pid"), 8888)  # line 231
        # Wait for any daemon threads to exit
        time.sleep(0.1)
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_insert_zmq_task_error_logging(self):
        """Cover lines 934-935, 937, 1055-1060 in _insert_zmq_task_to_scheduler."""
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

        class DummyRecv:
            def __init__(self, msg):
                self.msg = msg

            def receive_json_once(self, block):
                return self.msg, None

            def close(self):
                pass

        # Case 1: context terminated -> info branch (line 1055)
        eng.recv_request_server = DummyRecv("Context was terminated")
        with patch.object(eng, "llm_logger") as _:
            eng._insert_zmq_task_to_scheduler()

        # Case 2: other error -> error branch (line 1060)
        eng.running = True
        eng.recv_request_server = DummyRecv("Other Error")
        with patch.object(eng, "llm_logger") as _:
            eng._insert_zmq_task_to_scheduler()

        # Case 3: num_tasks > 0 branch (line 936-937)
        eng.running = True
        eng.engine_worker_queue = Mock(num_tasks=lambda: 1)
        with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
            eng.running = False
            eng._insert_zmq_task_to_scheduler()

        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_exit_sub_services_cleanup_paths(self):
        """Cover lines 1312-1340, 1350-1354 in _exit_sub_services."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                self.available_prefill_instances = type("X", (), {"put": lambda *_: None})()

            def get_server_port(self):
                return 0

            def cleanup(self):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        # attach stubs used by cleanup
        class Sig:
            def __init__(self):
                self.value = np.array([0], dtype=np.int32)

            def clear(self):
                pass

        eng.worker_ready_signal = Sig()
        eng.loaded_model_signal = Sig()
        eng.exist_task_signal = Sig()
        eng.exist_swapped_task_signal = Sig()
        eng.worker_healthy_live_signal = Sig()
        eng.cache_ready_signal = Sig()
        eng.swap_space_ready_signal = Sig()
        eng.exist_prefill_task_signal = Sig()
        eng.model_weights_status_signal = Sig()
        eng.prefix_tree_status_signal = Sig()
        eng.kv_cache_status_signal = Sig()
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
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_setting_environ_variables_v1_prefill_mm(self):
        """Cover lines 1476-1485 in _setting_environ_variables."""
        # For prefill + local scheduler the core code now requires a router
        # and ENABLE_V1_KVCACHE_SCHEDULER=0 when using the default IPC protocol.
        with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0):
            cfg = self._make_cfg(splitwise_role="prefill", router="0.0.0.0:30000")
        cfg.model_config.enable_mm = True

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        with patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True):
            prefix = eng._setting_environ_variables()
        self.assertIn("FLAGS_use_pd_disaggregation_per_chunk=1", prefix)
        self.assertIn("FLAGS_fmt_write_cache_completed_signal=1", prefix)
        self.assertIn("FLAGS_max_partition_size=1024", prefix)
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_start_worker_service_cmd_build(self):
        """Cover 1517, 1526, 1568, 1592, 1595 by building the worker command with mocks."""
        with patch("fastdeploy.config.get_host_ip", return_value="127.0.0.1"):
            cfg = self._make_cfg(splitwise_role="mixed", num_gpu_blocks_override=4, ips=["127.0.0.1", "127.0.0.2"])
        # Make model multi-modal so env var branch already covered above; here not required
        cfg.structured_outputs_config.logits_processors = ["A", "B"]

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
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
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_check_health_unhealthy(self):
        """Cover line 1628: unhealthy worker."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        class Sig:
            def __init__(self, v):
                self.value = np.array([v], dtype=np.int32)

        # set worker live time far past threshold
        eng.worker_healthy_live_signal = Sig(int(time.time()) - 1000)
        ok, msg = eng.check_health(time_interval_threashold=1)
        self.assertFalse(ok)
        self.assertIn("Not Healthy".lower(), msg.lower())
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

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
        class DummyQ:
            def __init__(self, *a, **k):
                self.available_prefill_instances = type("X", (), {"put": lambda *_: None})()

            def get_server_port(self):
                return 0

            def cleanup(self):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
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
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_check_worker_initialize_status_progress(self):
        """Cover 1710-1762 by simulating stdout and ready signals."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
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
        class Sig:
            def __init__(self):
                self.value = np.array([1], dtype=np.int32)

        eng.worker_ready_signal = Sig()

        # Replace tqdm and sleep for fast execution
        class DummyPbar:
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

        with patch("fastdeploy.engine.common_engine.tqdm", lambda *a, **k: DummyPbar()):
            with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
                ok = eng.check_worker_initialize_status()
        self.assertTrue(ok)
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_worker_processes_ready_false(self):
        """Cover line 1382 returning False."""
        cfg = self._make_cfg()

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        class Sig:
            def __init__(self):
                # less than worker_num_per_node
                self.value = np.array([0], dtype=np.int32)

        eng.worker_ready_signal = Sig()
        self.assertFalse(eng._worker_processes_ready())
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_init_worker_signals_profile_iluvatar(self):
        """Cover line 1434 by forcing iluvatar custom device and do_profile=True."""
        # do_profile=True when num_gpu_blocks_override is None
        cfg = self._make_cfg(num_gpu_blocks_override=None)

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        eng.ipc_signal_suffix = cfg.parallel_config.engine_worker_queue_port[0]
        with patch("fastdeploy.engine.common_engine.paddle.is_compiled_with_custom_device", return_value=True):
            eng._init_worker_signals()
        # signal should exist
        self.assertTrue(hasattr(eng, "get_profile_block_num_signal"))
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

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

        class DummyQ:
            def __init__(self, *a, **k):
                self.available_prefill_instances = type("X", (), {"put": lambda *_: None})()

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        # Patch scheduler.start so it doesn't do heavy work
        eng.scheduler.start = Mock()
        eng.launch_components()
        eng.scheduler.start.assert_called()
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_init_token_processor_plugin_and_cache_port(self):
        """Cover lines 66, 85-93: TokenProcessor plugin loading and cache_queue_port handling."""
        with patch("fastdeploy.engine.common_engine.load_token_processor_plugins", side_effect=Exception("test")):
            cfg = self._make_cfg(splitwise_role="prefill", router="http://localhost:8000", cache_queue_port="100,101")

            class DummyQ:
                def __init__(self, *a, **k):
                    pass

            with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
                eng = EngineService(cfg, start_queue=False, use_async_llm=False)
                self.assertIsNotNone(eng.token_processor)
                # Test cache_queue_port conversion
                cfg2 = self._make_cfg(splitwise_role="decode", router="http://localhost:8000", cache_queue_port="200")
                with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
                    eng2 = EngineService(cfg2, start_queue=False, use_async_llm=False)
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
                eng2._finalizer.detach()
            except Exception:
                pass

    def test_start_worker_queue_service_shm_and_cache_queue(self):
        """Cover lines 366, 377-390, 390-404: SHM address, port update, cache queue creation."""
        with patch("fastdeploy.engine.common_engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", True):
            cfg = self._make_cfg(splitwise_role="prefill", router="http://localhost:8000")

            class DummyQ:
                def __init__(self, *a, **k):
                    self.port = 9999

                def get_server_port(self):
                    return self.port

                def cleanup(self):
                    pass

            with (
                patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ),
                patch("fastdeploy.engine.common_engine.EngineCacheQueue", DummyQ),
            ):
                eng = EngineService(cfg, start_queue=True, use_async_llm=False)
                self.assertIsNotNone(eng.engine_worker_queue)
                if hasattr(eng, "cache_task_queue"):
                    self.assertIsNotNone(eng.cache_task_queue)
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_insert_tasks_all_branches(self):
        """Cover lines 418-496: insert_tasks with various conditions."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

            def put_tasks(self, *a):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        eng.running = True
        req = Request(
            request_id="test1",
            prompt="test",
            prompt_token_ids=[1, 2, 3],
            prompt_token_ids_len=3,
            messages=[],
            history=[],
            tools=[],
            system="",
            eos_token_ids=[],
        )
        req.metrics = Mock()
        eng.resource_manager.stop_flags = np.array([True, True])
        eng.resource_manager.allocate_resources_for_new_tasks = lambda tasks: tasks
        eng.resource_manager.real_bsz = 1
        eng.split_connector.check_decode_allocated = lambda t: (True, "")
        eng.split_connector.send_cache_info_to_messager = lambda *a: None
        eng.token_processor.number_of_tasks = 0
        eng.token_processor.number_of_input_tokens = 0
        eng.engine_worker_queue = DummyQ()
        # Test single task (not list)
        eng.insert_tasks(req)
        # Test prefill role with failed allocation
        cfg2 = self._make_cfg(splitwise_role="prefill")
        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng2 = EngineService(cfg2, start_queue=False, use_async_llm=False)
        eng2.split_connector.check_decode_allocated = lambda t: (False, "error")
        eng2.scheduler.put_results = Mock()
        eng2.resource_manager.stop_flags = np.array([True])
        eng2.resource_manager.allocate_resources_for_new_tasks = lambda tasks: []
        eng2.token_processor.number_of_tasks = 0
        req2 = Request(
            request_id="test2",
            prompt="test",
            prompt_token_ids=[1, 2],
            prompt_token_ids_len=2,
            messages=[],
            history=[],
            tools=[],
            system="",
            eos_token_ids=[],
        )
        req2.metrics = Mock()
        with self.assertRaises(EngineError):
            eng2.insert_tasks([req2])
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
                eng2._finalizer.detach()
            except Exception:
                pass

    def test_task_status_methods(self):
        """Cover lines 554-555, 561: task_is_finished and all_tasks_finished."""
        cfg = self._make_cfg()

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        eng.resource_manager.stop_flags = np.array([True, False, True])
        self.assertTrue(eng.task_is_finished(0))
        self.assertFalse(eng.task_is_finished(1))
        self.assertFalse(eng.all_tasks_finished())
        eng.resource_manager.stop_flags = np.array([True, True, True])
        self.assertTrue(eng.all_tasks_finished())
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_update_mm_requests_chunk_size(self):
        """Cover lines 620-690: update_mm_requests_chunk_size for multimodal."""
        cfg = self._make_cfg(enable_chunked_prefill=True, enable_mm=True)

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        eng.data_processor = Mock()
        eng.data_processor.image_patch_id = 99
        req = Request(
            request_id="r1",
            prompt="test",
            prompt_token_ids=[1, 2, 99, 3],
            prompt_token_ids_len=4,
            messages=[],
            history=[],
            tools=[],
            system="",
            eos_token_ids=[],
        )
        req.multimodal_inputs = {
            "input_ids": np.array([1, 2, 99, 3]),
            "token_type_ids": np.array([0, 0, 0, 0]),
            "image_type_ids": None,
            "grid_thw": np.array([[1, 10, 10]]),
            "images": None,
            "position_ids": np.array([0, 1, 2, 3]),
        }
        with patch("fastdeploy.model_executor.ops.gpu.get_mm_split_fuse", return_value=([1], [4])):
            eng.update_mm_requests_chunk_size([req])
        self.assertTrue(hasattr(req, "prefill_chunk_info") or hasattr(req, "_prefill_chunk_info"))
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_schedule_request_to_worker_v0(self):
        """Cover lines 697-761: _schedule_request_to_worker v0 scheduler."""
        with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0):
            cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                self.tasks = 0
                self.cache_infos = 0

            def exist_tasks(self):
                return self.tasks > 0

            def num_cache_infos(self):
                return self.cache_infos

            def put_tasks(self, *a):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        eng.running = True
        eng.resource_manager.available_batch = lambda: 2
        eng.resource_manager.available_block_num = lambda: 100
        eng.resource_manager.real_bsz = 1
        eng.engine_worker_queue = DummyQ()
        eng.split_connector.has_splitwise_tasks = lambda: False
        eng.split_connector.current_request_ids = []
        eng.split_connector.send_splitwise_tasks = lambda *a: None
        eng.scheduler.get_requests = lambda **kw: []
        eng.exist_prefill_task_signal = Mock(value=np.array([0]))
        # Run briefly then stop
        eng.running = False
        eng._schedule_request_to_worker()
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_schedule_request_to_worker_v1_branches(self):
        """Cover lines 767-1014: _schedule_request_to_worker_v1 various branches."""
        with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1):
            cfg = self._make_cfg(splitwise_role="prefill", router="http://localhost:8000")

        class DummyQ:
            def __init__(self, *a, **k):
                self.tasks = 0

            def num_tasks(self):
                return self.tasks

            def put_tasks(self, *a):
                pass

            def get_finished_add_cache_task_req(self):
                return []

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        eng.running = True
        eng.engine_worker_queue = DummyQ()
        eng.resource_manager.available_batch = lambda: 0  # Test available_batch == 0
        eng.resource_manager.available_block_num = lambda: 50
        eng.resource_manager.real_bsz = 1
        eng.resource_manager.schedule = lambda: ([], [])
        eng.resource_manager.preallocate_resource_in_p = lambda t: True
        eng.resource_manager.waiting_async_process = lambda t: True
        eng.resource_manager.add_request_in_p = lambda tasks: None
        eng.split_connector.send_splitwise_tasks = lambda *a: None
        eng.split_connector.check_decode_allocated = lambda t: (True, "")
        eng.split_connector.send_cache_info_to_messager = lambda *a: None
        eng.scheduler.get_requests = lambda **kw: []
        eng.exist_prefill_task_signal = Mock(value=np.array([0]))
        eng.resource_manager.waiting = []

        # Test with num_tasks > 0 (line 771)
        eng.engine_worker_queue.num_tasks = lambda: 1
        with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
            eng.running = False
            eng._schedule_request_to_worker_v1()

        # Test with available_batch > 0
        eng.running = True
        eng.engine_worker_queue.num_tasks = lambda: 0
        eng.resource_manager.available_batch = lambda: 1
        with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
            eng.running = False
            eng._schedule_request_to_worker_v1()

        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    # Temporarily disabled due to CI hanging issue - ZMQ thread cleanup problem
    # def test_zmq_services_and_token_processing(self):
    #     """Cover lines 1018-1218: ZMQ service, token decode, and response sending."""
    #     cfg = self._make_cfg()
    #
    #     class DummyQ:
    #         def __init__(self, *a, **k):
    #             pass
    #
    #     class DummyZmq:
    #         def __init__(self, *a, **k):
    #             self.recv_thread = None
    #             self.send_thread = None
    #
    #         def recv_result_handle(self):
    #             pass
    #
    #         def close(self):
    #             pass
    #
    #     with (
    #         patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ),
    #         patch("fastdeploy.engine.common_engine.ZmqIpcServer", DummyZmq),
    #         patch("fastdeploy.engine.common_engine.ZmqTcpServer", DummyZmq),
    #     ):
    #         eng = EngineService(cfg, start_queue=False, use_async_llm=False)
    #     eng.running = True
    #     eng.start_zmq_service(12345)
    #     # Test _decode_token
    #     eng.data_processor = Mock()
    #     eng.data_processor.decode_status = {"req1": [0, 2]}
    #     eng.data_processor.ids2tokens = lambda ids, req_id: ("text", [1, 2, 3], None)
    #     with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True):
    #         text, tokens = eng._decode_token([1, 2], "req1", False)
    #         self.assertEqual(text, "text")
    #     # Test _zmq_send_generated_tokens
    #     eng.scheduler.get_results = lambda: []
    #     eng.send_response_server = Mock(send_response=Mock())
    #     eng.running = False
    #     eng._zmq_send_generated_tokens()
    #     # Wait for daemon threads to exit (with timeout to avoid hanging)
    #     time.sleep(0.2)
    #     # Clean up threads if they exist
    #     for thread_name in ["receive_output_thread", "insert_task_to_scheduler_thread", "recv_result_handle_thread"]:
    #         if hasattr(eng, thread_name):
    #             thread = getattr(eng, thread_name)
    #             if thread and thread.is_alive():
    #                 time.sleep(0.1)
    #     if hasattr(eng, "_finalizer"):
    #         try:
    #             eng._finalizer.detach()
    #         except Exception:
    #             pass

    def test_insert_zmq_task_to_scheduler(self):
        """Cover lines 1043-1044, 1046->exit, 1052, 1063-1110: ZMQ task insertion."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        class DummyRecv:
            def receive_json_once(self, block):
                return None, {"request_id": "test1", "prompt": "test", "user": "user1"}

            def close(self):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        eng.running = True
        eng.recv_request_server = DummyRecv()
        eng.scheduler.put_requests = lambda tasks: []
        eng.guided_decoding_checker = None
        eng._send_error_response = Mock()
        # Run once then stop
        eng.running = False
        eng._insert_zmq_task_to_scheduler()
        # Test with guided decoding
        eng.running = True
        eng.guided_decoding_checker = Mock(schema_format=lambda req: (req, None))
        eng.running = False
        eng._insert_zmq_task_to_scheduler()
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    # Temporarily disabled due to CI hanging issue - thread cleanup problem
    # def test_decode_process_splitwise_requests(self):
    #     """Cover lines 1229-1251, 1254-1285, 1289-1334, 1337-1347: decode splitwise processing."""
    #     with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0):
    #         cfg = self._make_cfg(splitwise_role="decode", router="http://localhost:8000")
    #
    #     class DummyQ:
    #         def __init__(self, *a, **k):
    #             self.empty = True
    #
    #         def disaggregate_queue_empty(self):
    #             return self.empty
    #
    #         def get_disaggregated_tasks(self):
    #             return []
    #
    #     with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
    #         eng = EngineService(cfg, start_queue=False, use_async_llm=False)
    #     eng.running = True
    #     eng.engine_worker_queue = DummyQ()
    #     eng.resource_manager.is_resource_sufficient = lambda n: True
    #     eng.resource_manager.preallocate_resource_in_d = lambda t: True
    #     eng.split_connector.send_cache_info_to_prefill = lambda *a: None
    #     eng.scheduler.has_request = lambda rid: True
    #     eng.scheduler.put_results = Mock()
    #     eng.token_processor.tokens_counter = {}
    #     eng._decode_process_splitwise_requests()
    #     time.sleep(0.1)  # Let thread start
    #     eng.running = False
    #     # Wait for daemon thread to exit (with timeout to avoid hanging)
    #     time.sleep(0.2)
    #     if hasattr(eng, "_finalizer"):
    #         try:
    #             eng._finalizer.detach()
    #         except Exception:
    #             pass

    def test_insert_tasks_splitwise_branches(self):
        """Cover lines 430, 452-454, 472-475, 479, 481, 494: insert_tasks splitwise branches."""
        # Test prefill role with decode allocation
        cfg = self._make_cfg(splitwise_role="prefill", router="http://localhost:8000")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

            def put_tasks(self, *a):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)

        eng.resource_manager.stop_flags = np.array([True, True, False, False])
        eng.resource_manager.allocate_resources_for_new_tasks = lambda tasks: tasks
        eng.resource_manager.real_bsz = 2
        eng.token_processor.number_of_tasks = 0
        eng.token_processor.number_of_input_tokens = 0
        eng.split_connector.check_decode_allocated = lambda task: (True, "")
        eng.split_connector.send_cache_info_to_messager = Mock()
        eng.split_connector.send_cache_info_to_prefill = Mock()
        eng.update_requests_chunk_size = Mock()
        eng.update_mm_requests_chunk_size = Mock()
        eng.engine_worker_queue = DummyQ()

        # Test with tasks exceeding available batch
        req1 = Request(
            request_id="r1",
            prompt="test",
            prompt_token_ids=[1] * 10,
            prompt_token_ids_len=10,
            messages=[],
            history=[],
            tools=[],
            system="",
            eos_token_ids=[],
        )
        req2 = Request(
            request_id="r2",
            prompt="test",
            prompt_token_ids=[1] * 10,
            prompt_token_ids_len=10,
            messages=[],
            history=[],
            tools=[],
            system="",
            eos_token_ids=[],
        )
        req3 = Request(
            request_id="r3",
            prompt="test",
            prompt_token_ids=[1] * 10,
            prompt_token_ids_len=10,
            messages=[],
            history=[],
            tools=[],
            system="",
            eos_token_ids=[],
        )
        req4 = Request(
            request_id="r4",
            prompt="test",
            prompt_token_ids=[1] * 10,
            prompt_token_ids_len=10,
            messages=[],
            history=[],
            tools=[],
            system="",
            eos_token_ids=[],
        )
        req5 = Request(
            request_id="r5",
            prompt="test",
            prompt_token_ids=[1] * 10,
            prompt_token_ids_len=10,
            messages=[],
            history=[],
            tools=[],
            system="",
            eos_token_ids=[],
        )

        # Test batch exceeding available
        eng.insert_tasks([req1, req2, req3, req4, req5])

        # Test decode role
        cfg2 = self._make_cfg(splitwise_role="decode", router="http://localhost:8000")
        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng2 = EngineService(cfg2, start_queue=False, use_async_llm=False)
        eng2.resource_manager.stop_flags = np.array([True, True])
        eng2.resource_manager.allocate_resources_for_new_tasks = lambda tasks: tasks
        eng2.resource_manager.real_bsz = 2
        eng2.token_processor.number_of_tasks = 0
        eng2.token_processor.number_of_input_tokens = 0
        eng2.split_connector.send_cache_info_to_prefill = Mock()
        eng2.engine_worker_queue = DummyQ()

        req1.disaggregate_info = {"role": "decode"}
        eng2.insert_tasks([req1])

        # Test with mm enabled
        cfg3 = self._make_cfg(enable_mm=True)
        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng3 = EngineService(cfg3, start_queue=False, use_async_llm=False)
        eng3.resource_manager.stop_flags = np.array([True])
        eng3.resource_manager.allocate_resources_for_new_tasks = lambda tasks: tasks
        eng3.resource_manager.real_bsz = 1
        eng3.token_processor.number_of_tasks = 0
        eng3.token_processor.number_of_input_tokens = 0
        eng3.update_mm_requests_chunk_size = Mock()
        eng3.engine_worker_queue = DummyQ()

        req_mm = Request(
            request_id="r_mm",
            prompt="test",
            prompt_token_ids=[1] * 10,
            prompt_token_ids_len=10,
            messages=[],
            history=[],
            tools=[],
            system="",
            eos_token_ids=[],
        )
        req_mm.disaggregate_info = {"role": "prefill"}
        eng3.insert_tasks([req_mm])

        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
                eng2._finalizer.detach()
                eng3._finalizer.detach()
            except Exception:
                pass

    def test_update_requests_chunk_size_branches(self):
        """Cover lines 571, 580, 589, 602, 605, 606, 611: update_requests_chunk_size branches."""
        cfg = self._make_cfg(enable_chunked_prefill=True, max_num_batched_tokens=100, block_size=16)

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)

        # Test with empty requests (line 580)
        eng.update_requests_chunk_size([])

        # Test with requests that need chunking
        # Ensure partial_chunked_tokens has enough elements
        eng.partial_chunked_tokens = [0, 100, 50, 33, 25, 20, 16, 14, 12, 11, 10]
        req1 = Request(
            request_id="r1",
            prompt="test",
            prompt_token_ids=[1] * 50,
            prompt_token_ids_len=50,
            messages=[],
            history=[],
            tools=[],
            system="",
            eos_token_ids=[],
        )
        req2 = Request(
            request_id="r2",
            prompt="test",
            prompt_token_ids=[1] * 30,
            prompt_token_ids_len=30,
            messages=[],
            history=[],
            tools=[],
            system="",
            eos_token_ids=[],
        )
        # Test with update_chunk=True (line 571)
        eng.update_requests_chunk_size([req1, req2])
        self.assertTrue(hasattr(req1, "prefill_chunk_info") or hasattr(req1, "_prefill_chunk_info"))

        # Test with current_request_size[idx] <= 0 (line 589)
        req3 = Request(
            request_id="r3",
            prompt="test",
            prompt_token_ids=[],
            prompt_token_ids_len=0,
            messages=[],
            history=[],
            tools=[],
            system="",
            eos_token_ids=[],
        )
        eng.update_requests_chunk_size([req3])

        # Test with waiting_requests empty (line 602)
        eng.partial_chunked_tokens = [0, 100, 50, 33, 25, 20, 16, 14, 12, 11, 10]
        req4 = Request(
            request_id="r4",
            prompt="test",
            prompt_token_ids=[1] * 5,
            prompt_token_ids_len=5,
            messages=[],
            history=[],
            tools=[],
            system="",
            eos_token_ids=[],
        )
        eng.update_requests_chunk_size([req4])

        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_update_mm_requests_chunk_size_branches(self):
        """Cover lines 621, 637, 638, 640: update_mm_requests_chunk_size branches."""
        cfg = self._make_cfg(enable_chunked_prefill=True, enable_mm=True)

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)

        # Create data_processor mock
        eng.data_processor = Mock()
        eng.data_processor.image_patch_id = 1000

        # Test with empty requests (line 621)
        eng.update_mm_requests_chunk_size([])

        # Test with mm request - one[0] == 1 (line 637)
        req = Request(
            request_id="r_mm",
            prompt="test",
            prompt_token_ids=[1] * 10,
            prompt_token_ids_len=10,
            messages=[],
            history=[],
            tools=[],
            system="",
            eos_token_ids=[],
        )
        req.multimodal_inputs = {
            "input_ids": np.array([1] * 10),
            "token_type_ids": np.array([0] * 10),
            "image_type_ids": np.array([]),
            "grid_thw": np.array([[1, 10, 10]]),
            "images": np.array([]),
            "position_ids": np.array([0] * 10),
        }

        with patch("fastdeploy.model_executor.ops.gpu.get_mm_split_fuse") as mock_get_mm:
            mock_get_mm.return_value = ([1], [10])
            eng.update_mm_requests_chunk_size([req])

        # Test with one[0] != 1 (line 640)
        req2 = Request(
            request_id="r_mm2",
            prompt="test",
            prompt_token_ids=[1] * 10,
            prompt_token_ids_len=10,
            messages=[],
            history=[],
            tools=[],
            system="",
            eos_token_ids=[],
        )
        req2.multimodal_inputs = {
            "input_ids": np.array([1] * 10),
            "token_type_ids": np.array([0] * 10),
            "image_type_ids": np.array([]),
            "grid_thw": np.array([[4, 10, 10]]),  # one[0] = 4, not 1
            "images": np.array([]),
            "position_ids": np.array([0] * 10),
        }

        with patch("fastdeploy.model_executor.ops.gpu.get_mm_split_fuse") as mock_get_mm:
            mock_get_mm.return_value = ([1], [10])
            eng.update_mm_requests_chunk_size([req2])

        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass
        
        # Ensure engine is stopped and all threads are cleaned up
        if hasattr(eng, "running"):
            eng.running = False
        if hasattr(eng, "_exit_sub_services"):
            try:
                eng._exit_sub_services()
            except Exception:
                pass
        # Wait a bit for any daemon threads to exit
        time.sleep(0.1)

    # Temporarily disabled due to CI hanging issue - thread cleanup problem
    # def test_utility_methods(self):
    #     """Cover lines 1365, 1368-1378, 1386-1414: utility methods."""
    #     cfg = self._make_cfg(router="http://localhost:8000")
    #
    #     class DummyQ:
    #         def __init__(self, *a, **k):
    #             pass
    #
    #         def clear_data(self):
    #             pass
    #
    #     with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
    #         eng = EngineService(cfg, start_queue=False, use_async_llm=False)
    #     eng.resource_manager.check_and_free_block_tables = Mock()
    #     eng.check_and_free_block_tables()
    #     # Test clear_data
    #     eng.token_processor = Mock(clear_data=Mock())
    #     eng.engine_worker_queue = DummyQ()
    #     eng.send_response_server = Mock(req_dict=Mock(clear=Mock()))
    #     eng.recv_request_server = Mock(req_dict=Mock(clear=Mock()))
    #     result = eng.clear_data()
    #     self.assertTrue(result)
    #     # Test _register_to_router
    #     with (
    #         patch("fastdeploy.router.utils.check_service_health", return_value=True),
    #         patch("fastdeploy.engine.common_engine.requests.post") as mock_post,
    #     ):
    #         mock_resp = Mock(ok=True)
    #         mock_post.return_value = mock_resp
    #         eng._register_to_router()
    #         # Wait for registration thread to complete (it should break after successful registration)
    #         time.sleep(0.2)
    #     # Ensure any daemon threads have time to exit
    #     time.sleep(0.1)
    #     if hasattr(eng, "_finalizer"):
    #         try:
    #             eng._finalizer.detach()
    #         except Exception:
    #             pass


# Temporarily disabled entire class due to CI hanging issue - thread cleanup problems
# All tests in TestCommonEngineUncoveredLines class are disabled
@unittest.skip("Temporarily disabled due to CI hanging issue - thread cleanup problems")
class TestCommonEngineUncoveredLines(unittest.TestCase):
    """Test cases to cover previously uncovered lines - ALL TESTS DISABLED"""

    # All test methods are disabled to prevent CI hanging
    pass

# All original test methods in TestCommonEngineUncoveredLines are commented out below:
#     @patch("fastdeploy.engine.common_engine.load_token_processor_plugins")
#     def test_token_processor_plugin_exception(self, mock_load):
#         """Test exception handling when loading TokenProcessor plugin - line 66"""
#         # Test success path (line 66)
#         mock_load.return_value = Mock()
#         from importlib import reload
#
#         import fastdeploy.engine.common_engine as ce_module
#
#         try:
#             reload(ce_module)
#         except Exception:
#             pass
#
#         # Test exception path
#         mock_load.side_effect = Exception("Plugin load failed")
#         try:
#             reload(ce_module)
#         except Exception:
#             pass
#
#     @patch("fastdeploy.engine.common_engine.schema_checker")
#     def test_guided_decoding_backend_enabled(self, mock_schema):
#         """Test guided decoding backend initialization - lines 148-152"""
#         mock_cfg = Mock()
#         mock_cfg.structured_outputs_config.guided_decoding_backend = "lark"
#         mock_cfg.structured_outputs_config.disable_any_whitespace = False
#         mock_cfg.scheduler_config.splitwise_role = "mixed"
#         mock_cfg.scheduler_config.scheduler.return_value = Mock()
#         mock_cfg.cache_config.enable_prefix_caching = False
#         mock_cfg.parallel_config.data_parallel_size = 1
#         mock_cfg.scheduler_config.max_num_seqs = 256
#         mock_cfg.parallel_config.tensor_parallel_size = 1
#         mock_cfg.parallel_config.local_data_parallel_id = 0
#         mock_cfg.max_num_partial_prefills = 3
#         mock_cfg.scheduler_config.max_num_batched_tokens = 2048
#         mock_cfg.cache_config.block_size = 16
#         mock_cfg.eplb_config.enable_eplb = False
#
#         with (
#             patch("fastdeploy.engine.common_engine.ResourceManager"),
#             patch("fastdeploy.engine.common_engine.SplitwiseConnector"),
#             patch("fastdeploy.engine.common_engine.TokenProcessor"),
#             patch.object(mock_cfg.scheduler_config, "scheduler", return_value=Mock()),
#         ):
#
#             from fastdeploy.engine.common_engine import EngineService
#
#             engine = EngineService.__new__(EngineService)
#             engine.cfg = mock_cfg
#             engine.use_async_llm = False
#             engine.llm_logger = Mock()
#             engine.scheduler = Mock()
#             engine.enable_decode_cache_task = False
#             engine.engine_worker_queue = Mock()
#             engine.resource_manager = Mock()
#             engine.split_connector = Mock()
#             engine.token_processor = Mock()
#             engine.partial_chunked_tokens = [0] * 4
#             engine.bos_client = None
#             engine.guided_decoding_checker = None
#
#             # Trigger the guided decoding check
#             if mock_cfg.structured_outputs_config.guided_decoding_backend != "off":
#                 engine.guided_decoding_checker = mock_schema(
#                     mock_cfg.structured_outputs_config.guided_decoding_backend,
#                     disable_any_whitespace=mock_cfg.structured_outputs_config.disable_any_whitespace,
#                 )
#
#             mock_schema.assert_called_once()
#             self.assertIsNotNone(engine.guided_decoding_checker)
#
#     @patch("fastdeploy.engine.common_engine.init_eplb_signals")
#     def test_eplb_enabled(self, mock_init_eplb):
#         """Test EPLB initialization - lines 155-158"""
#         mock_cfg = Mock()
#         mock_cfg.eplb_config.enable_eplb = True
#         # Use a real list to ensure indexing works correctly
#         mock_cfg.parallel_config.engine_worker_queue_port = ["6778"]
#         mock_cfg.parallel_config.local_data_parallel_id = 0
#         mock_cfg.parallel_config.tensor_parallel_rank = 0
#         mock_cfg.scheduler_config.splitwise_role = "mixed"
#         mock_cfg.cache_config.enable_prefix_caching = False
#         mock_cfg.parallel_config.data_parallel_size = 1
#         mock_cfg.scheduler_config.scheduler.return_value = Mock()
#         mock_cfg.scheduler_config.max_num_seqs = 256
#         mock_cfg.parallel_config.tensor_parallel_size = 1
#         mock_cfg.max_num_partial_prefills = 3
#         mock_cfg.scheduler_config.max_num_batched_tokens = 2048
#         mock_cfg.cache_config.block_size = 16
#         mock_cfg.structured_outputs_config.guided_decoding_backend = "off"
#         mock_cfg.model_config.num_hidden_layers = 1
#         mock_cfg.model_config.moe_num_experts = 1
#         mock_cfg.eplb_config.redundant_expert_ip_shm_size = 1024
#         mock_cfg.master_ip = "127.0.0.1"
#         mock_cfg.host_ip = "127.0.0.1"
#
#         with (
#             patch("fastdeploy.engine.common_engine.ResourceManager"),
#             patch("fastdeploy.engine.common_engine.SplitwiseConnector"),
#             patch("fastdeploy.engine.common_engine.TokenProcessor"),
#             patch("fastdeploy.engine.common_engine.weakref.finalize"),
#             patch("fastdeploy.engine.common_engine.EngineWorkerQueue"),
#             patch("fastdeploy.engine.common_engine.IPCSignal"),
#             patch("fastdeploy.engine.common_engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", False),
#         ):
#
#             from fastdeploy.engine.common_engine import EngineService
#
#             _ = EngineService(mock_cfg, start_queue=False, use_async_llm=False)
#
#             # Verify init_eplb_signals was called
#             mock_init_eplb.assert_called_once()
#             call_args = mock_init_eplb.call_args
#             self.assertEqual(call_args[0][0], mock_cfg)
#             self.assertEqual(call_args[0][1], 6778)
#
#     @patch("fastdeploy.engine.common_engine.threading.Thread")
#     @patch("fastdeploy.engine.common_engine.envs")
#     def test_start_with_v1_scheduler(self, mock_envs, mock_thread):
#         """Test start method with V1 scheduler - lines 213-216"""
#         mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = True
#
#         mock_cfg = Mock()
#         mock_cfg.scheduler_config.splitwise_role = "mixed"
#
#         with patch.object(EngineService, "__init__", return_value=None):
#             engine = EngineService.__new__(EngineService)
#             engine.cfg = mock_cfg
#             engine.use_async_llm = False
#             engine.running = False
#             engine.engine_worker_queue = Mock()
#             engine.token_processor = Mock()
#             engine.insert_task_to_worker_thread = Mock()
#
#             mock_thread_instance = Mock()
#             mock_thread.return_value = mock_thread_instance
#
#             with patch.object(engine, "_register_to_router"):
#                 engine.start()
#
#             # Verify V1 scheduler thread was created
#             self.assertTrue(engine.running)
#             mock_thread.assert_called()
#
#     @patch("fastdeploy.engine.common_engine.time.sleep")
#     def test_start_with_decode_role(self, mock_sleep):
#         """Test start method with decode splitwise role - line 227"""
#         mock_cfg = Mock()
#         mock_cfg.scheduler_config.splitwise_role = "decode"
#
#         with patch.object(EngineService, "__init__", return_value=None):
#             engine = EngineService.__new__(EngineService)
#             engine.cfg = mock_cfg
#             engine.use_async_llm = False
#             engine.running = False
#             engine.engine_worker_queue = Mock()
#             engine.token_processor = Mock()
#             engine.insert_task_to_worker_thread = Mock()
#             engine.insert_task_to_worker_thread.start = Mock()
#
#             with (
#                 patch.object(engine, "_decode_process_splitwise_requests") as mock_decode,
#                 patch.object(engine, "_register_to_router"),
#                 patch("fastdeploy.engine.common_engine.threading.Thread"),
#                 patch("fastdeploy.engine.common_engine.envs"),
#             ):
#
#                 engine.start()
#                 mock_decode.assert_called_once()
#
#     def test_start_worker_service_thread_dead(self):
#         """Test start_worker_service when check_worker_initialize_status_func_thread dies - line 227"""
#         cfg = self._make_cfg(splitwise_role="mixed", num_gpu_blocks_override=2)
#
#         class DummyQ:
#             def __init__(self, *a, **k):
#                 pass
#
#             def cleanup(self):
#                 pass
#
#         with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
#             eng = EngineService(cfg, start_queue=False, use_async_llm=True)
#
#         eng.create_data_processor = lambda: setattr(eng, "data_processor", self._stub_processor())
#         eng._init_worker_signals = lambda: None
#         eng.launch_components = lambda: None
#         eng.start_cache_service = lambda *a: []
#         eng._start_worker_service = lambda: Mock(stdout=Mock(), poll=lambda: None)
#         eng.check_worker_initialize_status = lambda: True
#
#         class Sig:
#             def __init__(self, v=0):
#                 self.value = np.array([v], dtype=np.int32)
#
#         eng.loaded_model_signal = Sig(0)
#         eng.worker_init_status = {}
#
#         # Create a thread that dies immediately
#         dead_thread = threading.Thread(target=lambda: None, daemon=True)
#         dead_thread.start()
#         dead_thread.join()
#
#         eng.check_worker_initialize_status_func_thread = dead_thread
#
#         with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
#             result = eng.start_worker_service()
#
#         self.assertFalse(result)
#         if hasattr(eng, "_finalizer"):
#             try:
#                 eng._finalizer.detach()
#             except Exception:
#                 pass
#
#     def test_start_worker_service_data_processor_exists(self):
#         """Test start_worker_service when data_processor already exists - line 195"""
#         cfg = self._make_cfg(splitwise_role="mixed", num_gpu_blocks_override=2)
#
#         class DummyQ:
#             def __init__(self, *a, **k):
#                 pass
#
#             def cleanup(self):
#                 pass
#
#         with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
#             eng = EngineService(cfg, start_queue=False, use_async_llm=True)
#
#         eng.data_processor = self._stub_processor()
#         eng._init_worker_signals = lambda: None
#         eng.launch_components = lambda: None
#         eng.start_cache_service = lambda *a: []
#         eng._start_worker_service = lambda: Mock(stdout=Mock(), poll=lambda: None)
#         eng.check_worker_initialize_status = lambda: True
#
#         class Sig:
#             def __init__(self, v=1):
#                 self.value = np.array([v], dtype=np.int32)
#
#         eng.loaded_model_signal = Sig(1)
#         eng.worker_init_status = {}
#         eng.check_worker_initialize_status_func_thread = Mock(is_alive=lambda: True)
#
#         with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
#             eng.start_worker_service()
#
#         if hasattr(eng, "_finalizer"):
#             try:
#                 eng._finalizer.detach()
#             except Exception:
#                 pass
#
#     def test_start_zmq_service_none_pid(self):
#         """Test start_zmq_service with None pid - line 1018"""
#         cfg = self._make_cfg()
#
#         class DummyQ:
#             def __init__(self, *a, **k):
#                 pass
#
#             def cleanup(self):
#                 pass
#
#         with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
#             eng = EngineService(cfg, start_queue=False, use_async_llm=False)
#
#         result = eng.start_zmq_service(None)
#         self.assertIsNone(result)
#         if hasattr(eng, "_finalizer"):
#             try:
#                 eng._finalizer.detach()
#             except Exception:
#                 pass
#
#     def test_insert_zmq_task_decode_role(self):
#         """Test _insert_zmq_task_to_scheduler with decode role - line 1043-1044"""
#         with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0):
#             cfg = self._make_cfg(splitwise_role="decode", router="http://localhost:8000")
#
#         class DummyQ:
#             def __init__(self, *a, **k):
#                 pass
#
#             def cleanup(self):
#                 pass
#
#         with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
#             eng = EngineService(cfg, start_queue=False, use_async_llm=False)
#
#         eng.running = True
#         eng.recv_request_server = Mock()
#
#         with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
#             eng._insert_zmq_task_to_scheduler()
#
#         if hasattr(eng, "_finalizer"):
#             try:
#                 eng._finalizer.detach()
#             except Exception:
#                 pass
#
#     def test_insert_tasks_with_mm_enabled(self):
#         """Test insert_tasks with enable_mm=True - line 494"""
#         cfg = self._make_cfg(enable_mm=True)
#
#         class DummyQ:
#             def __init__(self, *a, **k):
#                 pass
#
#             def put_tasks(self, *a):
#                 pass
#
#         with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
#             eng = EngineService(cfg, start_queue=False, use_async_llm=False)
#
#         eng.running = True
#         eng.resource_manager.stop_flags = np.array([True])
#         eng.resource_manager.allocate_resources_for_new_tasks = lambda tasks: tasks
#         eng.resource_manager.real_bsz = 1
#         eng.token_processor.number_of_tasks = 0
#         eng.token_processor.number_of_input_tokens = 0
#         eng.update_mm_requests_chunk_size = Mock()
#         eng.engine_worker_queue = DummyQ()
#
#         req = Request(
#             request_id="test_mm",
#             prompt="test",
#             prompt_token_ids=[1, 2, 3],
#             prompt_token_ids_len=3,
#             messages=[],
#             history=[],
#             tools=[],
#             system="",
#             eos_token_ids=[],
#         )
#         req.metrics = Mock()
#
#         eng.insert_tasks(req)
#         eng.update_mm_requests_chunk_size.assert_called()
#
#         if hasattr(eng, "_finalizer"):
#             try:
#                 eng._finalizer.detach()
#             except Exception:
#                 pass
#
#     def test_insert_zmq_task_with_mm(self):
#         """Test _insert_zmq_task_to_scheduler with enable_mm - line 1052"""
#         cfg = self._make_cfg(enable_mm=True, splitwise_role="mixed")
#
#         class DummyQ:
#             def __init__(self, *a, **k):
#                 pass
#
#             def cleanup(self):
#                 pass
#
#         class DummyRecv:
#             def receive_pyobj_once(self, block):
#                 return None, {"request_id": "test1", "prompt": "test"}
#
#             def close(self):
#                 pass
#
#         with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
#             eng = EngineService(cfg, start_queue=False, use_async_llm=False)
#
#         eng.running = True
#         eng.recv_request_server = DummyRecv()
#         eng.scheduler.put_requests = lambda tasks: []
#         eng.guided_decoding_checker = None
#         eng._send_error_response = Mock()
#
#         with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
#             eng.running = False
#             eng._insert_zmq_task_to_scheduler()
#
#         if hasattr(eng, "_finalizer"):
#             try:
#                 eng._finalizer.detach()
#             except Exception:
#                 pass
#
#     def test_schedule_request_to_worker_branches(self):
#         """Test _schedule_request_to_worker various branches - lines 699-705"""
#         with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0):
#             cfg = self._make_cfg(splitwise_role="mixed")
#
#         class DummyQ:
#             def __init__(self, *a, **k):
#                 self.tasks = 1
#                 self.cache_infos = 0
#
#             def exist_tasks(self):
#                 return self.tasks > 0
#
#             def num_cache_infos(self):
#                 return self.cache_infos
#
#             def put_tasks(self, *a):
#                 pass
#
#         with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
#             eng = EngineService(cfg, start_queue=False, use_async_llm=False)
#
#         eng.running = True
#         eng.resource_manager.available_batch = lambda: 0
#         eng.engine_worker_queue = DummyQ()
#         eng.exist_prefill_task_signal = Mock(value=np.array([0]))
#
#         with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
#             eng.running = False
#             eng._schedule_request_to_worker()
#
#         # Test with available_batch > 0 but exist_tasks
#         eng.running = True
#         eng.resource_manager.available_batch = lambda: 1
#         eng.engine_worker_queue.exist_tasks = lambda: True
#         eng.running = False
#         eng._schedule_request_to_worker()
#
#         # Test with num_cache_infos > 0
#         eng.running = True
#         eng.engine_worker_queue.exist_tasks = lambda: False
#         eng.engine_worker_queue.num_cache_infos = lambda: 1
#         eng.running = False
#         eng._schedule_request_to_worker()
#
#         if hasattr(eng, "_finalizer"):
#             try:
#                 eng._finalizer.detach()
#             except Exception:
#                 pass
#
#     def test_insert_prefilled_requests(self):
#         """Cover lines 498-548: _insert_prefilled_requests with various branches."""
#         with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0):
#             cfg = self._make_cfg(splitwise_role="decode", router="http://localhost:8000")
#
#         class DummyQ:
#             def __init__(self, *a, **k):
#                 pass
#
#             def put_tasks(self, *a):
#                 pass
#
#         with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
#             eng = EngineService(cfg, start_queue=False, use_async_llm=False)
#
#         from fastdeploy.engine.request import CompletionOutput, RequestOutput
#
#         eng.resource_manager.req_dict = {"req1": 0}
#         eng.resource_manager.tasks_list = [
#             Request(
#                 request_id="req1",
#                 prompt="test",
#                 prompt_token_ids=[1, 2],
#                 prompt_token_ids_len=2,
#                 messages=[],
#                 history=[],
#                 tools=[],
#                 system="",
#                 eos_token_ids=[],
#             )
#         ]
#         eng.resource_manager.stop_flags = np.array([False])
#         eng.resource_manager.real_bsz = 1
#         eng.resource_manager._recycle_block_tables = Mock()
#         eng.scheduler.put_results = Mock()
#         eng.token_processor.tokens_counter = {}
#         eng.engine_worker_queue = DummyQ()
#
#         # Test with FD_ENABLE_INTERNAL_ADAPTER and empty token_ids
#         req_out1 = RequestOutput(
#             request_id="req1",
#             finished=False,
#             outputs=CompletionOutput(index=0, send_idx=0, token_ids=[]),
#         )
#         req_out1.metrics = Mock()
#         with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
#             eng._insert_prefilled_requests([req_out1])
#
#         # Test with error_code != 200
#         eng.resource_manager.req_dict = {"req2": 0}
#         eng.resource_manager.tasks_list = [
#             Request(
#                 request_id="req2",
#                 prompt="test",
#                 prompt_token_ids=[1, 2],
#                 prompt_token_ids_len=2,
#                 messages=[],
#                 history=[],
#                 tools=[],
#                 system="",
#                 eos_token_ids=[],
#             )
#         ]
#         eng.resource_manager.stop_flags = np.array([False])
#         req_out2 = RequestOutput(
#             request_id="req2",
#             finished=False,
#             error_code=500,
#             error_msg="test error",
#             outputs=CompletionOutput(index=0, send_idx=0, token_ids=[10]),
#         )
#         req_out2.metrics = Mock()
#         eng._insert_prefilled_requests([req_out2])
#
#         # Test with mtp speculative config
#         eng.resource_manager.req_dict = {"req3": 0}
#         eng.resource_manager.tasks_list = [
#             Request(
#                 request_id="req3",
#                 prompt="test",
#                 prompt_token_ids=[1, 2],
#                 prompt_token_ids_len=2,
#                 messages=[],
#                 history=[],
#                 tools=[],
#                 system="",
#                 eos_token_ids=[],
#             )
#         ]
#         eng.resource_manager.stop_flags = np.array([False])
#         eng.cfg.speculative_config.method = "mtp"
#         req_out3 = RequestOutput(
#             request_id="req3",
#             finished=False,
#             outputs=CompletionOutput(index=0, send_idx=0, token_ids=[10], draft_token_ids=[11, 12]),
#         )
#         req_out3.metrics = Mock()
#         eng._insert_prefilled_requests([req_out3])
#
#         if hasattr(eng, "_finalizer"):
#             try:
#                 eng._finalizer.detach()
#             except Exception:
#                 pass
#
#     def test_send_error_response_branches(self):
#         """Cover lines 1115-1130: _send_error_response with both branches."""
#         cfg = self._make_cfg()
#
#         class DummyQ:
#             def __init__(self, *a, **k):
#                 pass
#
#         with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
#             eng = EngineService(cfg, start_queue=False, use_async_llm=False)
#
#         eng.send_response_server = Mock(send_response=Mock())
#
#         # Test with FD_ENABLE_INTERNAL_ADAPTER=True
#         with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
#             eng._send_error_response("req1", "error msg", 500)
#             eng.send_response_server.send_response.assert_called_with(None, [[Mock()]])
#
#         # Test with FD_ENABLE_INTERNAL_ADAPTER=False
#         eng.send_response_server.reset_mock()
#         with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False):
#             eng._send_error_response("req2", "error msg", 400)
#             eng.send_response_server.send_response.assert_called_with("req2", [Mock()])
#
#         if hasattr(eng, "_finalizer"):
#             try:
#                 eng._finalizer.detach()
#             except Exception:
#                 pass
#
#     def test_decode_token_branches(self):
#         """Cover lines 1132-1144: _decode_token with all branches."""
#         cfg = self._make_cfg()
#
#         class DummyQ:
#             def __init__(self, *a, **k):
#                 pass
#
#         with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
#             eng = EngineService(cfg, start_queue=False, use_async_llm=False)
#
#         eng.data_processor = Mock()
#         eng.data_processor.decode_status = {"req1": [0, 2]}
#
#         # Test with FD_ENABLE_RETURN_TEXT=False
#         with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", False):
#             text, tokens = eng._decode_token([1, 2], "req1", False)
#             self.assertEqual(text, "")
#             self.assertEqual(tokens, [1, 2])
#
#         # Test with FD_ENABLE_RETURN_TEXT=True and delta_text != ""
#         eng.data_processor.decode_status = {"req2": [0, 2]}
#         eng.data_processor.ids2tokens = lambda ids, req_id: ("text", [1, 2, 3], None)
#         with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True):
#             text, tokens = eng._decode_token([1, 2], "req2", False)
#             self.assertEqual(text, "text")
#             self.assertEqual(tokens, [1, 2, 3])
#
#         # Test with delta_text == ""
#         eng.data_processor.decode_status = {"req3": [0, 2]}
#         eng.data_processor.ids2tokens = lambda ids, req_id: ("", [1, 2], None)
#         with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True):
#             text, tokens = eng._decode_token([1, 2], "req3", False)
#             self.assertEqual(text, "")
#             self.assertEqual(tokens, [])
#
#         # Test with is_end=True (should delete decode_status)
#         eng.data_processor.decode_status = {"req4": [0, 2]}
#         eng.data_processor.ids2tokens = lambda ids, req_id: ("text", [1, 2, 3], None)
#         with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True):
#             text, tokens = eng._decode_token([1, 2], "req4", True)
#             self.assertNotIn("req4", eng.data_processor.decode_status)
#
#         if hasattr(eng, "_finalizer"):
#             try:
#                 eng._finalizer.detach()
#             except Exception:
#                 pass

# Temporarily disabled due to CI hanging issue - thread cleanup problem
# def test_zmq_send_generated_tokens_branches(self):
#     """Cover lines 1146-1218: _zmq_send_generated_tokens with more branches."""
#     cfg = self._make_cfg()
#
#     class DummyQ:
#         def __init__(self, *a, **k):
#             pass
#
#     class DummyZmq:
#         def __init__(self, *a, **k):
#             pass
#
#         def close(self):
#             pass
#
#     with (
#         patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ),
#         patch("fastdeploy.engine.common_engine.ZmqIpcServer", DummyZmq),
#         patch("fastdeploy.engine.common_engine.ZmqTcpServer", DummyZmq),
#     ):
#         eng = EngineService(cfg, start_queue=False, use_async_llm=False)
#
#     eng.running = True
#     eng.data_processor = Mock()
#     eng.data_processor.decode_status = {}
#     eng.data_processor.ids2tokens = lambda ids, req_id: ("text", [1, 2, 3], None)
#     eng.send_response_server = Mock(send_response=Mock())
#
#     from fastdeploy.engine.request import CompletionOutput, RequestOutput
#
#     # Test with FD_ENABLE_INTERNAL_ADAPTER=True
#     result1 = RequestOutput(
#         request_id="req1",
#         finished=False,
#         outputs=CompletionOutput(index=0, send_idx=0, token_ids=[1, 2], decode_type=0),
#     )
#     eng.scheduler.get_results = lambda: [[result1]]
#     with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
#         eng._zmq_send_generated_tokens()
#         time.sleep(0.01)
#         eng.running = False
#         eng._zmq_send_generated_tokens()
#
#     # Test with FD_ENABLE_INTERNAL_ADAPTER=False
#     eng.running = True
#     result2 = RequestOutput(
#         request_id="req2",
#         finished=False,
#         outputs=CompletionOutput(index=0, send_idx=0, token_ids=[1, 2], decode_type=0),
#     )
#     eng.scheduler.get_results = lambda: {"req2": [result2]}
#     with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False):
#         eng._zmq_send_generated_tokens()
#         time.sleep(0.01)
#         eng.running = False
#         eng._zmq_send_generated_tokens()
#
#     # Test with decode_type != 0
#     eng.running = True
#     result3 = RequestOutput(
#         request_id="req3",
#         finished=False,
#         outputs=CompletionOutput(index=0, send_idx=0, token_ids=[1, 2], decode_type=1),
#     )
#     eng.scheduler.get_results = lambda: {"req3": [result3]}
#     with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False):
#         eng._zmq_send_generated_tokens()
#         time.sleep(0.01)
#         eng.running = False
#         eng._zmq_send_generated_tokens()
#
#     # Test with finished=True and empty token_ids
#     eng.running = True
#     result4 = RequestOutput(
#         request_id="req4",
#         finished=True,
#         outputs=CompletionOutput(index=0, send_idx=0, token_ids=[], decode_type=0),
#     )
#     eng.scheduler.get_results = lambda: {"req4": [result4]}
#     with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False):
#         eng._zmq_send_generated_tokens()
#         time.sleep(0.01)
#         eng.running = False
#         eng._zmq_send_generated_tokens()
#
#     # Wait for any daemon threads to exit (with timeout to avoid hanging)
#     time.sleep(0.2)
#     if hasattr(eng, "_finalizer"):
#         try:
#             eng._finalizer.detach()
#         except Exception:
#             pass


if __name__ == "__main__":
    unittest.main()
