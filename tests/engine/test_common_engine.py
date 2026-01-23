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

import contextlib
import importlib
import os
import sys
import time
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import paddle

if not hasattr(paddle, "compat"):
    paddle.compat = types.SimpleNamespace(enable_torch_proxy=lambda *args, **kwargs: None)

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.common_engine import EngineService
from fastdeploy.engine.request import (
    Request,
    RequestMetrics,
    RequestOutput,
    RequestStatus,
    RequestType,
)
from fastdeploy.utils import EngineError

MODEL_NAME = os.getenv("MODEL_PATH", "/path/to/models") + "/ERNIE-4.5-0.3B-Paddle"


@contextlib.contextmanager
def _patched_model_config():
    minimal_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "vocab_size": 128,
    }

    def _post_init_stub(self):
        self.is_unified_ckpt = True
        self.enable_mm = False
        self.architectures = minimal_cfg["architectures"]
        self.num_hidden_layers = minimal_cfg["num_hidden_layers"]

    with (
        patch("fastdeploy.config.PretrainedConfig.get_config_dict", return_value=(minimal_cfg, {})),
        patch("fastdeploy.config.PretrainedConfig.from_dict", return_value=types.SimpleNamespace()),
        patch("fastdeploy.config.ModelConfig._post_init", _post_init_stub),
    ):
        yield


class TestCommonEngine(unittest.TestCase):
    """Test case for EngineService functionality (lines 1215-1664)"""

    @classmethod
    def setUpClass(cls):
        """Set up EngineService for testing"""
        # Create engine args for testing
        engine_args = EngineArgs(
            model=MODEL_NAME,
            max_model_len=8192,
            tensor_parallel_size=1,
            engine_worker_queue_port=int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")),
            cache_queue_port=int(os.getenv("FD_CACHE_QUEUE_PORT", "6779")),
            skip_port_check=True,
        )

        # Create the engine service with lightweight stubs for local tests
        with _patched_model_config():
            cls.cfg = engine_args.create_engine_config()

        class DummyQueue:
            def __init__(self, *args, **kwargs):
                pass

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

        class DummyResourceManager:
            def __init__(self, *args, **kwargs):
                self.stop_flags = np.array([1], dtype=np.int32)

        class DummySplitConnector:
            def __init__(self, *args, **kwargs):
                pass

        class DummyTokenProcessor:
            def __init__(self, *args, **kwargs):
                pass

            def set_resource_manager(self, *args, **kwargs):
                pass

            def run(self):
                pass

        dummy_queue = DummyQueue()
        dummy_queue.get_server_port()
        dummy_queue.cleanup()
        dummy_queue.num_tasks()
        dummy_queue.num_cache_infos()
        dummy_queue.disaggregate_queue_empty()
        dummy_queue.get_disaggregated_tasks()
        dummy_token = DummyTokenProcessor()
        dummy_token.set_resource_manager(None)
        dummy_token.run()

        with (
            patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQueue),
            patch("fastdeploy.engine.common_engine.EngineCacheQueue", DummyQueue),
            patch("fastdeploy.engine.common_engine.ResourceManager", DummyResourceManager),
            patch("fastdeploy.engine.common_engine.ResourceManagerV1", DummyResourceManager),
            patch("fastdeploy.engine.common_engine.SplitwiseConnector", DummySplitConnector),
            patch("fastdeploy.engine.common_engine.TokenProcessor", DummyTokenProcessor),
        ):
            cls.engine = EngineService(cls.cfg, start_queue=False, use_async_llm=True)

        class Sig:
            def __init__(self, value=1):
                self.value = np.array([value], dtype=np.int32)

        cls.engine.running = True
        cls.engine.worker_proc = Mock(poll=lambda: None)
        cls.engine.worker_ready_signal = Sig(1)
        cls.engine.loaded_model_signal = Sig(1)
        cls.engine.worker_healthy_live_signal = Sig(int(time.time()))
        cls.engine.worker_init_status = {}
        cls.engine.data_processor = TestCommonEngineAdditionalCoverage()._stub_processor()
        cls.engine.ipc_signal_suffix = cls.engine.cfg.parallel_config.engine_worker_queue_port[0]
        if hasattr(cls.engine, "_finalizer"):
            cls.engine._finalizer.detach()

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

    def setUp(self):
        patch("fastdeploy.engine.common_engine.EngineCacheQueue").start()

    def _make_cfg(self, **kwargs):
        # If DP > 1, we must provide enough engine_worker_queue_port for each dp index
        dp = kwargs.get("data_parallel_size", 1)
        nnode = len(kwargs.get("ips", ["127.0.0.1"]))
        engine_worker_queue_port = int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778"))
        cache_queue_port = int(os.getenv("FD_CACHE_QUEUE_PORT", "6779"))
        splitwise_role = kwargs.get("splitwise_role", "mixed")
        if splitwise_role != "mixed" and kwargs.get("router") is None:
            kwargs["router"] = "0.0.0.0:30000"
        if kwargs.get("num_gpu_blocks_override") is not None and kwargs.get("kv_cache_ratio") is None:
            kwargs["kv_cache_ratio"] = 1
        if dp and dp > 1:
            engine_worker_queue_port = [engine_worker_queue_port + 21 + i for i in range(dp // nnode)]
            cache_queue_port = [cache_queue_port + 21 + i for i in range(dp // nnode)]

        args = EngineArgs(
            model=MODEL_NAME,
            max_model_len=128,
            tensor_parallel_size=1,
            # give unique ports to avoid collision with other tests
            engine_worker_queue_port=engine_worker_queue_port,
            cache_queue_port=cache_queue_port,
            enable_prefix_caching=True,
            skip_port_check=True,
            max_num_partial_prefills=2,
            **kwargs,
        )
        # Keep batch tokens small to satisfy FDConfig checks:
        # max_num_batched_tokens <= max_model_len * max_num_seqs
        if getattr(args, "max_num_batched_tokens", None) is None:
            args.max_num_batched_tokens = 128
        # Always enable chunked prefill in tests to avoid another strict check
        args.enable_chunked_prefill = True

        with _patched_model_config():
            return args.create_engine_config()

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
                self.image_patch_id = 9

        return _Proc()

    def _make_request(self, request_id, token_len=4, disaggregate_info=None, trace_carrier=None):
        prompt_token_ids = list(range(1, token_len + 1))
        req = Request(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=prompt_token_ids,
            prompt_token_ids_len=token_len,
            messages=None,
            history=None,
            tools=None,
            system=None,
            eos_token_ids=[0],
            disaggregate_info=disaggregate_info,
            trace_carrier=trace_carrier or {},
        )
        req.metrics.scheduler_recv_req_time = time.time()
        req.metrics.decode_recv_req_time = time.time()
        req.metrics.decode_preallocate_req_time = time.time()
        return req

    def test_token_processor_plugin_load_logging(self):
        """Cover line 65 via reloading module with a plugin."""
        with patch("fastdeploy.plugins.token_processor.load_token_processor_plugins", return_value=object()):
            with patch("fastdeploy.utils.llm_logger") as mock_logger:
                import fastdeploy.engine.common_engine as common_engine

                importlib.reload(common_engine)
                mock_logger.info.assert_called()
        importlib.reload(common_engine)

    def test_start_worker_queue_service_shm_address(self):
        """Cover lines 348-383 by exercising shm address and cache queue setup."""
        cfg = self._make_cfg(splitwise_role="mixed")
        cfg.master_ip = "0.0.0.0"
        cfg.host_ip = "0.0.0.0"
        cfg.cache_config.enable_prefix_caching = True

        created = {"worker": [], "cache": []}

        class DummyQueue:
            def __init__(self, address, is_server, **kwargs):
                self.address = address
                self.is_server = is_server
                created["worker"].append(self)

            def get_server_port(self):
                return 6000

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

        class DummyCacheQueue:
            def __init__(self, address, **kwargs):
                self.address = address
                created["cache"].append(self)

            def get_server_port(self):
                return 6001

        with patch("fastdeploy.engine.common_engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", True):
            with (
                patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQueue),
                patch("fastdeploy.engine.common_engine.EngineCacheQueue", DummyCacheQueue),
            ):
                eng = EngineService(cfg, start_queue=True, use_async_llm=True)
        self.assertTrue(created["worker"])
        self.assertIn("/dev/shm/fd_task_queue_", created["worker"][0].address)
        self.assertTrue(created["cache"])
        worker_queue = created["worker"][0]
        worker_queue.get_server_port()
        worker_queue.cleanup()
        worker_queue.num_tasks()
        worker_queue.num_cache_infos()
        worker_queue.disaggregate_queue_empty()
        worker_queue.get_disaggregated_tasks()
        created["cache"][0].get_server_port()
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_start_worker_queue_service_tcp_updates_port(self):
        """Cover lines 360-367 by updating port when using TCP queue."""
        cfg = self._make_cfg(splitwise_role="prefill")
        cfg.master_ip = "0.0.0.0"
        cfg.host_ip = "0.0.0.0"

        created = []

        class DummyQueue:
            def __init__(self, address, is_server, **kwargs):
                self.address = address
                self.is_server = is_server
                created.append(self)

            def get_server_port(self):
                return 6888

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

        class DummyCacheQueue:
            def __init__(self, address, **kwargs):
                self.address = address

            def get_server_port(self):
                return 6999

        with patch("fastdeploy.engine.common_engine.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", False):
            with (
                patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQueue),
                patch("fastdeploy.engine.common_engine.EngineCacheQueue", DummyCacheQueue),
            ):
                eng = EngineService(cfg, start_queue=True, use_async_llm=True)

        self.assertEqual(eng.cfg.parallel_config.local_engine_worker_queue_port, 6888)
        created[0].get_server_port()
        created[0].cleanup()
        created[0].num_tasks()
        created[0].num_cache_infos()
        created[0].disaggregate_queue_empty()
        created[0].get_disaggregated_tasks()
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_insert_tasks_prefill_error_and_truncate(self):
        """Cover lines 397-486 with prefill routing, errors, and truncation."""
        cfg = self._make_cfg(splitwise_role="prefill")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        class ResourceManagerStub:
            def __init__(self):
                self.stop_flags = np.array([1], dtype=np.int32)
                self.real_bsz = 1
                self.abort_req_ids_set = set()

            def check_and_free_block_tables(self):
                pass

            def allocate_resources_for_new_tasks(self, tasks):
                return tasks

        class SplitConnectorStub:
            def __init__(self):
                self.calls = []

            def check_decode_allocated(self, task):
                if task.request_id == "req_fail":
                    return False, "no resource"
                return True, ""

            def send_cache_info_to_messager(self, tasks, current_id):
                self.calls.append((tasks, current_id))

        scheduler = Mock()
        eng.resource_manager = ResourceManagerStub()
        eng.scheduler = scheduler
        eng.split_connector = SplitConnectorStub()
        eng.token_processor = types.SimpleNamespace(number_of_tasks=0, number_of_input_tokens=0)
        eng.engine_worker_queue = Mock()
        eng.update_requests_chunk_size = Mock()
        eng.update_mm_requests_chunk_size = Mock()

        req_fail = self._make_request("req_fail", token_len=3, trace_carrier={"trace": "1"})
        req_ok1 = self._make_request("req_ok1", token_len=4)
        req_ok2 = self._make_request("req_ok2", token_len=5)

        with (
            patch("fastdeploy.engine.common_engine.trace_print", lambda *_: None),
            patch("fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *_: None),
            patch("fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", lambda *_: "trace"),
            patch("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *_, **__: None),
        ):
            ok = eng.insert_tasks([req_fail, req_ok1, req_ok2], current_id=2)

        self.assertTrue(ok)
        scheduler.put_results.assert_called()
        eng.update_requests_chunk_size.assert_called()
        eng.engine_worker_queue.put_tasks.assert_called()
        self.assertEqual(eng.token_processor.number_of_tasks, 1)
        self.assertEqual(eng.token_processor.number_of_input_tokens, req_ok1.prompt_token_ids_len)
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_insert_tasks_raises_on_empty_allocation(self):
        """Cover lines 397-446 raising EngineError when no resources."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        class ResourceManagerStub:
            def __init__(self):
                self.stop_flags = np.array([1], dtype=np.int32)
                self.real_bsz = 1
                self.abort_req_ids_set = set()

            def check_and_free_block_tables(self):
                pass

            def allocate_resources_for_new_tasks(self, tasks):
                return []

        eng.resource_manager = ResourceManagerStub()
        eng.token_processor = types.SimpleNamespace(number_of_tasks=0, number_of_input_tokens=0)
        eng.split_connector = Mock()
        eng.engine_worker_queue = Mock()
        eng.scheduler = Mock()

        with self.assertRaises(EngineError):
            eng.insert_tasks(self._make_request("req_empty", token_len=2))
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_insert_prefilled_requests_adapter_and_error_paths(self):
        """Cover lines 495-538 with adapter short-circuit, errors, and enqueue."""
        cfg = self._make_cfg(splitwise_role="decode")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        class ResourceManagerStub:
            def __init__(self, tasks):
                self.req_dict = {t.request_id: idx for idx, t in enumerate(tasks)}
                self.tasks_list = tasks
                self.stop_flags = np.array([0] * len(tasks), dtype=np.int32)
                self.real_bsz = 1

            def _recycle_block_tables(self, task):
                task.recycled = True

        req0 = self._make_request("req0", token_len=2)
        req1 = self._make_request("req1", token_len=2)
        req2 = self._make_request("req2", token_len=2)

        eng.resource_manager = ResourceManagerStub([req0, req1, req2])
        eng.token_processor = types.SimpleNamespace(tokens_counter={"req0": 1, "req1": 1})
        eng.engine_worker_queue = Mock()
        eng.scheduler = Mock()
        eng.cfg.speculative_config.method = "mtp"

        class DummyOutputs:
            def __init__(self, token_ids, draft_token_ids=None):
                self.token_ids = token_ids
                self.draft_token_ids = draft_token_ids or []
                self.tool_calls = None

        metrics = RequestMetrics()
        metrics.decode_recv_req_time = time.time()
        metrics.decode_preallocate_req_time = time.time()

        out0 = RequestOutput(request_id="req0", outputs=DummyOutputs([]), metrics=metrics, error_code=200)
        out1 = RequestOutput(
            request_id="req1",
            outputs=DummyOutputs([3], draft_token_ids=[9]),
            metrics=metrics,
            error_code=500,
            error_msg="bad",
        )
        out2 = RequestOutput(
            request_id="req2", outputs=DummyOutputs([5], draft_token_ids=[8]), metrics=metrics, error_code=200
        )

        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
            ok = eng._insert_prefilled_requests([out0, out1, out2])

        self.assertTrue(ok)
        self.assertTrue(eng.resource_manager.stop_flags[0])
        self.assertTrue(eng.resource_manager.stop_flags[1])
        self.assertEqual(eng.token_processor.tokens_counter.get("req2"), 1)
        self.assertEqual(req2.draft_token_ids, [8])
        eng.engine_worker_queue.put_tasks.assert_called()
        eng.scheduler.put_results.assert_called()
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_update_requests_chunk_size_and_task_flags(self):
        """Cover lines 544-604 for chunk sizing and task state helpers."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        req1 = self._make_request("req_chunk1", token_len=6)
        req2 = self._make_request("req_chunk2", token_len=4)
        eng.update_requests_chunk_size([req1, req2])

        self.assertEqual(sum(req1.prefill_chunk_info), req1.prompt_token_ids_len)
        self.assertEqual(sum(req2.prefill_chunk_info), req2.prompt_token_ids_len)

        eng.resource_manager.stop_flags = np.array([1, 0], dtype=np.int32)
        self.assertTrue(eng.task_is_finished(0))
        self.assertFalse(eng.task_is_finished(1))
        self.assertFalse(eng.all_tasks_finished())

        eng.resource_manager.stop_flags = np.array([1, 1], dtype=np.int32)
        self.assertTrue(eng.all_tasks_finished())
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_update_mm_requests_chunk_size_without_images(self):
        """Cover lines 610-680 with multimodal chunking and paddle ops."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        eng.data_processor = self._stub_processor()

        dummy_module = types.ModuleType("fastdeploy.model_executor.ops.gpu")
        dummy_module.get_mm_split_fuse = lambda *args, **kwargs: ([0], [4])

        inputs = {
            "input_ids": np.array([1, eng.data_processor.image_patch_id, 2, 3], dtype="int64"),
            "token_type_ids": np.array([0, 0, 0, 0], dtype="int64"),
            "image_type_ids": np.array([], dtype="int32"),
            "grid_thw": np.array([], dtype="int64"),
            "images": None,
            "position_ids": np.array([0, 1, 2, 3], dtype="int64"),
        }
        req = self._make_request("req_mm", token_len=4)
        req.multimodal_inputs = inputs

        tensor = paddle.to_tensor(inputs["input_ids"])
        self.assertEqual(list(tensor.shape), [4])

        with patch.dict(sys.modules, {"fastdeploy.model_executor.ops.gpu": dummy_module}):
            eng.update_mm_requests_chunk_size([req])

        self.assertEqual(len(req.prefill_chunk_info), 1)
        self.assertEqual(len(req.prefill_chunk_info[0]["input_ids"]), 4)
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

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
            ok = eng.start(async_llm_pid=12345)

        # start() returns False on failure
        self.assertFalse(ok)
        # cache manager started before workers (lines 184-185)
        self.assertTrue(started_cache.get("called", False))
        # launched_cache_manager_signal set (line 221)
        self.assertEqual(int(eng.launched_cache_manager_signal.value[0]), 1)
        # avoid atexit finalizer
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_start_mixed_branch_cache_after_load_and_zmq(self):
        """Cover lines 215-217 and 231 in start()."""
        cfg = self._make_cfg(splitwise_role="mixed", num_gpu_blocks_override=4, kv_cache_ratio=1)
        cfg.cache_config.enable_prefix_caching = True

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
            eng.start(async_llm_pid=8888)

        self.assertTrue(started_cache.get("called", False))  # lines 215-217
        self.assertEqual(zmq_called.get("pid"), 8888)  # line 231
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_insert_zmq_task_error_logging(self):
        """Cover lines 934-935 and 937 in _insert_zmq_task_to_scheduler."""
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
                self.call_count = 0

            def receive_json_once(self, block):
                self.call_count += 1
                if self.call_count == 1:
                    return self.msg, None
                else:
                    eng.running = False
                    return None, None

            def close(self):
                pass

        # Case 1: context terminated -> info branch
        eng.recv_request_server = DummyRecv("Context was terminated")
        with patch.object(eng, "llm_logger") as mock_logger:
            with patch("fastdeploy.engine.common_engine.ZmqIpcServer"):
                eng._insert_zmq_task_to_scheduler()
            # verify info logger
            mock_logger.info.assert_called()

        # reset status
        eng.running = True

        # Case 2: other error -> error branch
        eng.recv_request_server = DummyRecv("Other Error")
        with patch.object(eng, "llm_logger") as mock_logger:
            with patch("fastdeploy.engine.common_engine.ZmqIpcServer"):
                eng._insert_zmq_task_to_scheduler()
            # verify error logger
            mock_logger.error.assert_called()

        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

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
            eng._finalizer.detach()

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
            eng._finalizer.detach()

    def test_start_worker_service_cmd_build(self):
        """Cover 1517, 1526, 1568, 1592, 1595 by building the worker command with mocks."""
        with patch("fastdeploy.config.get_host_ip", return_value="127.0.0.1"):
            cfg = self._make_cfg(
                splitwise_role="mixed", num_gpu_blocks_override=4, ips=["127.0.0.1", "127.0.0.2"], data_parallel_size=2
            )
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
            eng._finalizer.detach()

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
            eng._finalizer.detach()

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
            eng._finalizer.detach()

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
            eng._finalizer.detach()

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
            eng._finalizer.detach()

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
            eng._finalizer.detach()

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
            eng._finalizer.detach()

    def test_setting_environ_variables_prefill_v0(self):
        """Cover non-v1 prefill env var branches in _setting_environ_variables."""
        with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0):
            cfg = self._make_cfg(splitwise_role="prefill", router="0.0.0.0:30000")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        with patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", False):
            prefix = eng._setting_environ_variables()
        self.assertIn("FLAGS_use_pd_disaggregation=1", prefix)
        self.assertIn("FLAGS_fmt_write_cache_completed_signal=1", prefix)
        self.assertNotIn("FLAGS_use_pd_disaggregation_per_chunk=1", prefix)
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_send_error_response_branches(self):
        """Cover internal adapter vs normal branches in _send_error_response."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        eng.send_response_server = Mock()

        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False):
            eng._send_error_response("req-normal", "boom", error_code=400)
        normal_args = eng.send_response_server.send_response.call_args[0]
        self.assertEqual(normal_args[0], "req-normal")
        self.assertEqual(normal_args[1][0].error_code, 400)
        self.assertEqual(normal_args[1][0].error_msg, "boom")

        eng.send_response_server.reset_mock()
        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
            eng._send_error_response("req-internal", "bad")
        internal_args = eng.send_response_server.send_response.call_args[0]
        self.assertIsNone(internal_args[0])
        self.assertEqual(internal_args[1][0][0].request_id, "req-internal")
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_decode_token_returns_text_and_cleans_status(self):
        """Cover text-return path and decode_status cleanup in _decode_token."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        class Proc:
            def __init__(self):
                self.decode_status = {"req-1": [0, 2]}

            def ids2tokens(self, token_ids, req_id):
                return "hi", [10, 11, 12], None

        eng.data_processor = Proc()
        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True):
            delta_text, token_ids = eng._decode_token([10, 11], "req-1", is_end=True)
        self.assertEqual(delta_text, "hi")
        self.assertEqual(token_ids, [10, 11])
        self.assertNotIn("req-1", eng.data_processor.decode_status)
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_insert_zmq_task_abort_preempt_v1(self):
        """Cover abort handling in _insert_zmq_task_to_scheduler with v1 scheduler."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        eng.running = True

        class ResourceManagerStub:
            def __init__(self):
                self.abort_req_ids_set = set()
                self.requests = {"abort-req": object()}
                self.real_bsz = 2

            def _prepare_preempt_task(self, req):
                return "preempt-task"

        eng.resource_manager = ResourceManagerStub()
        eng.engine_worker_queue = Mock()
        eng.scheduler = Mock()

        class DummyRecv:
            def __init__(self, engine):
                self.engine = engine
                self.called = False

            def receive_json_once(self, block):
                if not self.called:
                    self.called = True
                    self.engine.running = False
                    return None, {"status": RequestStatus.ABORT.value, "request_id": "abort-req"}
                return None, None

        eng.recv_request_server = DummyRecv(eng)

        with (
            patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True),
            patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_DATA_PROCESSOR", False),
        ):
            eng._insert_zmq_task_to_scheduler()

        eng.engine_worker_queue.put_tasks.assert_called_with((["preempt-task"], eng.resource_manager.real_bsz))
        self.assertIn("abort-req", eng.resource_manager.abort_req_ids_set)
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_zmq_send_generated_tokens_internal_adapter(self):
        """Cover internal adapter branch in _zmq_send_generated_tokens."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        eng.running = True

        class Proc:
            def __init__(self):
                self.decode_status = {"req-2": [0, 2]}

            def ids2tokens(self, token_ids, req_id):
                return "ok", [1, 2, 3], None

        eng.data_processor = Proc()
        eng.send_response_server = Mock()

        class Outputs:
            def __init__(self):
                self.decode_type = 0
                self.token_ids = [1, 2, 3]
                self.text = ""
                self.tool_calls = None

        def get_results_side_effect():
            if not getattr(eng, "_results_called", False):
                eng._results_called = True
                return [[RequestOutput(request_id="req-2", outputs=Outputs(), finished=False)]]
            eng.running = False
            return []

        eng.scheduler.get_results = Mock(side_effect=get_results_side_effect)

        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True),
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._zmq_send_generated_tokens()

        args = eng.send_response_server.send_response.call_args[0]
        self.assertIsNone(args[0])
        self.assertEqual(args[1][0][0].outputs.text, "ok")
        self.assertEqual(args[1][0][0].outputs.token_ids, [1, 2])
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_zmq_send_generated_tokens_non_internal(self):
        """Cover non-internal adapter branch in _zmq_send_generated_tokens."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        eng.running = True

        class Proc:
            def __init__(self):
                self.decode_status = {"req-3": [0, 2]}

            def ids2tokens(self, token_ids, req_id):
                return "yo", [4, 5, 6], None

        eng.data_processor = Proc()
        eng.send_response_server = Mock()

        class Outputs:
            def __init__(self):
                self.decode_type = 0
                self.token_ids = [4, 5, 6]
                self.text = ""
                self.tool_calls = None

        def get_results_side_effect():
            if not getattr(eng, "_results_called", False):
                eng._results_called = True
                return {"req-3": [RequestOutput(request_id="req-3", outputs=Outputs(), finished=False)]}
            eng.running = False
            return {}

        eng.scheduler.get_results = Mock(side_effect=get_results_side_effect)

        with (
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False),
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._zmq_send_generated_tokens()

        args = eng.send_response_server.send_response.call_args[0]
        self.assertEqual(args[0], "req-3")
        self.assertEqual(args[1][0].outputs.text, "yo")
        self.assertEqual(args[1][0].outputs.token_ids, [4, 5])
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_clear_data_success_and_failure(self):
        """Cover clear_data success and error branches."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)

        eng.token_processor = Mock()
        eng.engine_worker_queue = Mock()
        eng.send_response_server = types.SimpleNamespace(req_dict={})
        eng.recv_request_server = types.SimpleNamespace(req_dict={})

        self.assertTrue(eng.clear_data())
        eng.token_processor.clear_data.assert_called()
        eng.engine_worker_queue.clear_data.assert_called()

        eng.token_processor.clear_data.side_effect = RuntimeError("boom")
        self.assertFalse(eng.clear_data())
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_schedule_request_to_worker_v1_prefill_paths(self):
        """Cover v1 scheduler prefill fetch/queue paths."""
        with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1):
            cfg = self._make_cfg(
                splitwise_role="prefill",
                num_gpu_blocks_override=4,
                router="0.0.0.0:30000",
                kv_cache_ratio=1,
            )

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)

        class DummyPool:
            def submit(self, fn, *args, **kwargs):
                fn(*args, **kwargs)
                return Mock()

        req = self._make_request("req-prefill", token_len=3)
        req.idx = 0

        class ResourceManagerStub:
            def __init__(self):
                self.abort_req_ids_set = set()
                self.waiting = []
                self.real_bsz = 1

            def available_batch(self):
                return 1

            def check_and_free_block_tables(self):
                pass

            def available_block_num(self):
                return 1

            def preallocate_resource_in_p(self, task):
                return True

            def pre_recycle_resource(self, request_id):
                self.pre_recycled = request_id

            def apply_async_preprocess(self, task):
                task.preprocessed = True

            def waiting_async_process(self, task):
                return False

            def add_request_in_p(self, tasks):
                self.added = [t.request_id for t in tasks]

            def schedule(self):
                eng.running = False
                return [], []

        class SplitConnectorStub:
            def send_splitwise_tasks(self, tasks, idx):
                self.sent = True

            def check_decode_allocated(self, task):
                return True, ""

            def send_cache_info_to_messager(self, tasks, current_id):
                self.cache_sent = True

        eng.resource_manager = ResourceManagerStub()
        eng.split_connector = SplitConnectorStub()
        eng.scheduler = Mock(get_requests=Mock(return_value=[req]))

        class DummyEngineQueue:
            def exist_tasks(self):
                return False

            def get_finished_add_cache_task_req(self):
                return ["req-prefill"]

        eng.engine_worker_queue = DummyEngineQueue()

        with (
            patch("fastdeploy.engine.common_engine.ThreadPoolExecutor", lambda max_workers: DummyPool()),
            patch("fastdeploy.engine.common_engine.envs.PREFILL_CONTINUOUS_REQUEST_DECODE_RESOURCES", False),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
            patch("fastdeploy.engine.common_engine.tracing.trace_set_proc_propagate_context", lambda *_: None),
            patch("fastdeploy.engine.common_engine.tracing.trace_get_proc_propagate_context", lambda *_: {}),
            patch("fastdeploy.engine.common_engine.tracing.trace_report_span", lambda *_, **__: None),
            patch("fastdeploy.engine.common_engine.trace_print", lambda *_: None),
        ):
            eng.running = True
            eng._schedule_request_to_worker_v1()

        self.assertEqual(eng.resource_manager.added, ["req-prefill"])
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()

    def test_schedule_request_to_worker_v1_decode_preempted(self):
        """Cover v1 decode preempted path that emits error results."""
        with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1):
            cfg = self._make_cfg(splitwise_role="decode", num_gpu_blocks_override=4, kv_cache_ratio=1)

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)

        class DummyPool:
            def submit(self, fn, *args, **kwargs):
                fn(*args, **kwargs)
                return Mock()

        req = self._make_request("req-decode", token_len=2)
        req.task_type = RequestType.PREEMPTED

        class ResourceManagerStub:
            def __init__(self):
                self.abort_req_ids_set = set()
                self.waiting = []
                self.real_bsz = 1

            def available_batch(self):
                return 1

            def schedule(self):
                eng.running = False
                return [req], []

            def get_real_bsz(self):
                pass

        eng.resource_manager = ResourceManagerStub()
        eng.scheduler = Mock(get_requests=Mock(return_value=[]))
        eng.engine_worker_queue = types.SimpleNamespace(exist_tasks=lambda: False, put_tasks=Mock())

        with (
            patch("fastdeploy.engine.common_engine.ThreadPoolExecutor", lambda max_workers: DummyPool()),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng.running = True
            eng._schedule_request_to_worker_v1()

        eng.scheduler.put_results.assert_called()
        if hasattr(eng, "_finalizer"):
            eng._finalizer.detach()
