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

import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import paddle

# Ensure paddle exposes compat.enable_torch_proxy for fastdeploy import compatibility.
if not hasattr(paddle, "compat"):

    class _DummyCompat:
        @staticmethod
        def enable_torch_proxy(scope=None):
            return None

    paddle.compat = _DummyCompat()

# Add the root directory to Python path so we can import fastdeploy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import fastdeploy.cache_manager.cache_transfer_manager as cache_transfer_manager
from fastdeploy.cache_manager.cache_tasks import ReadStorageTask, WriteStorageTask
from fastdeploy.cache_manager.cache_transfer_manager import CacheTransferManager


# ==========================
# 测试用 Args
# ==========================
class Args:
    rank = 0
    local_data_parallel_id = 0
    mp_num = 1
    device_id = 0
    speculative_config = {}
    model_id = "test_model"
    ipc_suffix = "test_ipc_suffix"
    cache_queue_port = 9999
    pod_ip = "127.0.0.1"
    engine_worker_queue_port = 9998
    num_cpu_blocks = 1
    num_gpu_blocks = 1
    num_layers = 1
    key_cache_shape = "1,1,1,1"
    value_cache_shape = ""
    create_cache_tensor = False
    cache_dtype = "bfloat16"
    default_dtype = "bfloat16"
    kvcache_storage_backend = None
    write_policy = "write_through"
    model_path = "test_model"


# ==========================
# 测试类
# ==========================
class TestCacheTransferManager(unittest.TestCase):
    def setUp(self):
        # --------------------------
        # mock logger
        # --------------------------
        cache_transfer_manager.logger = MagicMock()

        # --------------------------
        # mock current_platform
        # --------------------------
        class DummyPlatform:
            @staticmethod
            def is_iluvatar():
                return False

            @staticmethod
            def is_xpu():
                # 测试环境下不使用 XPU，返回 False
                return False

            @staticmethod
            def is_cuda():
                # 测试环境下不使用 CUDA，返回 False
                return False

        cache_transfer_manager.current_platform = DummyPlatform()

        # --------------------------
        # mock EngineCacheQueue
        # --------------------------
        patcher1 = patch("fastdeploy.cache_manager.cache_transfer_manager.EngineCacheQueue", new=MagicMock())
        patcher1.start()
        self.addCleanup(patcher1.stop)

        # --------------------------
        # mock IPCSignal
        # --------------------------
        patcher2 = patch("fastdeploy.cache_manager.cache_transfer_manager.IPCSignal", new=MagicMock())
        patcher2.start()
        self.addCleanup(patcher2.stop)

        # --------------------------
        # mock _init_cpu_cache 和 _init_gpu_cache
        # --------------------------
        self._orig_init_cpu_cache = CacheTransferManager._init_cpu_cache
        self._orig_init_gpu_cache = CacheTransferManager._init_gpu_cache
        patcher3 = patch.object(CacheTransferManager, "_init_cpu_cache", lambda self, args: None)
        patcher4 = patch.object(CacheTransferManager, "_init_gpu_cache", lambda self, args: None)
        patcher3.start()
        patcher4.start()
        self.addCleanup(patcher3.stop)
        self.addCleanup(patcher4.stop)

        # --------------------------
        # 创建 manager
        # --------------------------
        self.manager = CacheTransferManager(Args())

        # --------------------------
        # mock worker_healthy_live_signal
        # --------------------------
        class DummySignal:
            def __init__(self):
                self.value = [0]

        self.manager.worker_healthy_live_signal = DummySignal()

        # --------------------------
        # mock swap thread pools
        # --------------------------
        self.manager.swap_to_cpu_thread_pool = MagicMock()
        self.manager.swap_to_gpu_thread_pool = MagicMock()

        # --------------------------
        # mock cache_task_queue
        # --------------------------
        self.manager.cache_task_queue = MagicMock()
        self.manager.cache_task_queue.empty.return_value = False
        self.manager.cache_task_queue.get_transfer_task.return_value = (([0], 0, 0, MagicMock(value=0), 0), True)
        self.manager.cache_task_queue.barrier1 = MagicMock()
        self.manager.cache_task_queue.barrier2 = MagicMock()
        self.manager.cache_task_queue.barrier3 = MagicMock()

        # --------------------------
        # 避免 sleep 阻塞测试
        # --------------------------
        self.sleep_patch = patch("time.sleep", lambda x: None)
        self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    # ==========================
    # check_work_status 测试
    # ==========================
    def test_check_work_status_no_signal(self):
        healthy, msg = self.manager.check_work_status()
        self.assertTrue(healthy)

    def test_check_work_status_healthy(self):
        self.manager.worker_healthy_live_signal.value[0] = int(time.time())
        healthy, msg = self.manager.check_work_status()
        self.assertTrue(healthy)

    def test_check_work_status_unhealthy(self):
        self.manager.worker_healthy_live_signal.value[0] = int(time.time()) - 1000
        healthy, msg = self.manager.check_work_status(time_interval_threashold=10)
        self.assertFalse(healthy)

    def test_parse_args_defaults(self):
        with patch.object(sys, "argv", ["prog"]):
            args = cache_transfer_manager.parse_args()
        self.assertEqual(args.cache_dtype, "bfloat16")

    def test_init_storage_invalid_backend_logs_error(self):
        class LocalArgs(Args):
            kvcache_storage_backend = "unknown"

        with (
            patch.object(CacheTransferManager, "_init_cpu_cache", lambda self, args: None),
            patch.object(CacheTransferManager, "_init_gpu_cache", lambda self, args: None),
            patch("fastdeploy.cache_manager.cache_transfer_manager.console_logger") as mock_console,
        ):
            with self.assertRaises(NotImplementedError):
                CacheTransferManager(LocalArgs())

        self.assertTrue(cache_transfer_manager.logger.error.called)
        mock_console.error.assert_called_once()

    def test_init_storage_block_wise_fp8_invalid_backend(self):
        class LocalArgs(Args):
            cache_dtype = "block_wise_fp8"
            kvcache_storage_backend = "file"

        with (
            patch.object(CacheTransferManager, "_init_cpu_cache", lambda self, args: None),
            patch.object(CacheTransferManager, "_init_gpu_cache", lambda self, args: None),
        ):
            with self.assertRaises(ValueError):
                CacheTransferManager(LocalArgs())

    def test_init_storage_sets_key_prefix_from_version(self):
        args = Args()
        with tempfile.TemporaryDirectory() as tmpdir:
            version_path = os.path.join(tmpdir, "version.yaml")
            with open(version_path, "w", encoding="utf-8") as handle:
                handle.write("version: RL-STEP03-20250101-uuid\n")
            args.model_path = tmpdir
            args.kvcache_storage_backend = None
            self.manager._init_storage(args)

        self.assertEqual(self.manager.key_prefix, "RL-STEP03")

    # ==========================
    # do_data_transfer 异常处理测试
    # ==========================
    def test_do_data_transfer_broken_pipe(self):
        # mock get_transfer_task 抛出 BrokenPipeError
        self.manager.cache_task_queue.get_transfer_task.side_effect = BrokenPipeError("mock broken pipe")

        # mock check_work_status 返回 False，触发 break
        self.manager.check_work_status = MagicMock(return_value=(False, "Not Healthy"))
        self.manager.cache_task_broadcast_signal = type("DummySignal", (), {"value": [0]})()

        with (
            patch("fastdeploy.cache_manager.cache_transfer_manager.envs.FD_CACHE_PROC_ERROR_COUNT", 0),
            patch("time.sleep", lambda *_: None),
        ):
            self.manager.do_data_transfer()

        self.assertGreaterEqual(self.manager.check_work_status.call_count, 1)
        self.assertGreaterEqual(cache_transfer_manager.logger.critical.call_count, 1)

    def test_do_data_transfer_barriers_swap2gpu_then_idle(self):
        class DummySignal:
            def __init__(self, value):
                self.value = [value]

        self.manager.rank = 0
        self.manager.n_ranks = 2
        self.manager.cache_task_broadcast_signal = DummySignal(0)
        self.manager.cache_task_queue.empty.side_effect = [False, True]
        self.manager.cache_task_queue.get_transfer_task.side_effect = [
            ((cache_transfer_manager.CacheStatus.SWAP2GPU, 1, [0], [1], [2]), True)
        ]
        self.manager.cache_task_queue.barrier1 = MagicMock()
        self.manager.cache_task_queue.barrier2 = MagicMock()
        self.manager.cache_task_queue.barrier3 = MagicMock()
        self.manager.cache_task_queue.barrier2.reset.side_effect = SystemExit
        self.manager.swap_to_gpu_thread_pool.submit = MagicMock()

        with self.assertRaises(SystemExit):
            self.manager.do_data_transfer()

        self.manager.swap_to_gpu_thread_pool.submit.assert_called_once()
        self.manager.cache_task_queue.barrier3.reset.assert_called_once()
        self.manager.cache_task_queue.barrier2.wait.assert_called_once()

    def test_do_data_transfer_broken_pipe_check_status_exception(self):
        class DummySignal:
            def __init__(self, value):
                self.value = [value]

        self.manager.rank = 0
        self.manager.n_ranks = 1
        self.manager.cache_task_broadcast_signal = DummySignal(1)
        self.manager.cache_task_queue.get_transfer_task.side_effect = [
            BrokenPipeError("boom"),
            SystemExit,
        ]
        self.manager.check_work_status = MagicMock(side_effect=RuntimeError("status failed"))

        with (
            patch("fastdeploy.cache_manager.cache_transfer_manager.envs.FD_CACHE_PROC_ERROR_COUNT", 0),
            patch("time.sleep", MagicMock()),
        ):
            with self.assertRaises(SystemExit):
                self.manager.do_data_transfer()

        self.assertGreaterEqual(cache_transfer_manager.logger.error.call_count, 1)

    def test_do_data_transfer_generic_exception_logs(self):
        class DummySignal:
            def __init__(self, value):
                self.value = [value]

        self.manager.rank = 0
        self.manager.n_ranks = 1
        self.manager.cache_task_broadcast_signal = DummySignal(1)
        self.manager.cache_task_queue.get_transfer_task.side_effect = [
            RuntimeError("oops"),
            SystemExit,
        ]

        with patch.object(cache_transfer_manager.logger, "info") as mock_info:
            with self.assertRaises(SystemExit):
                self.manager.do_data_transfer()

        mock_info.assert_called()

    # ==========================
    # 工具函数与存储相关测试
    # ==========================
    def test_get_cache_bytes_and_invalid(self):
        self.assertEqual(self.manager._get_cache_item_bytes("bfloat16"), 2)
        self.assertEqual(self.manager._get_cache_item_bytes("float32"), 4)
        with self.assertRaises(ValueError):
            self.manager._get_cache_item_bytes("int32")

    def test_run_read_storage_swaps_valid_blocks(self):
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "mooncake"
        self.manager.storage_key_read_buffer = 1000
        self.manager.storage_value_read_buffer = 2000
        self.manager.cache_buffer_stride_bytes = 10
        self.manager.key_cache_shape = [2, 1, 1, 1]
        self.manager.value_cache_shape = [2, 1, 1, 1]
        self.manager.device = 0
        self.manager.gpu_cache_k_tensors = [paddle.zeros([1])]
        self.manager.gpu_cache_v_tensors = [paddle.zeros([1])]
        self.manager.storage_backend.batch_get.return_value = [1, 1, 1, 0]

        with patch("fastdeploy.cache_manager.cache_transfer_manager.swap_cache_layout"):
            valid_ids = self.manager._run_read_storage(
                "test_task",
                [1, 2, 3, 4],
                0,
                ["k1", "k2"],
                ["v1", "v2"],
                None,
                None,
                [5, 6],
                [0, 1],
                30.0,
            )

        self.assertEqual(valid_ids, [5])

    def test_run_read_storage_mooncake_error_raises(self):
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "mooncake"
        self.manager.storage_key_read_buffer = 100
        self.manager.storage_value_read_buffer = 200
        self.manager.cache_buffer_stride_bytes = 10
        self.manager.key_cache_shape = [1, 1, 1, 1]
        self.manager.value_cache_shape = [1, 1, 1, 1]
        self.manager.device = 0
        self.manager.gpu_cache_k_tensors = [paddle.zeros([1])]
        self.manager.gpu_cache_v_tensors = [paddle.zeros([1])]
        self.manager.storage_backend.batch_get.side_effect = RuntimeError("read error")

        with self.assertRaises(RuntimeError):
            self.manager._run_read_storage("task", [1], 0, ["k1"], ["v1"], None, None, [0], [0], 0.1)

    def test_read_storage_task_reports_result(self):
        self.manager.cache_task_queue.swap_storage_to_gpu_barrier = MagicMock()
        self.manager.cache_task_queue.put_transfer_done_signal = MagicMock()
        self.manager._run_read_storage = MagicMock(return_value=[3])
        self.manager.block_size = 1

        # Mock storage backend to return 2 matches
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "attention_store"
        self.manager.storage_backend.query.return_value = 2

        task = ReadStorageTask(
            task_id="7",
            keys=["a", "b", "c"],
            token_ids=[1, 2, 3, 4, 5, 6],  # 3 keys * 2 tokens per key
            gpu_block_ids=[3, 4, 5],
            start_read_block_idx=0,
            timeout=0.2,
        )

        self.manager.read_storage_task(task)

        self.manager._run_read_storage.assert_called_once_with(
            "7",
            [1, 2],
            0,
            ["prefix_a_0_key", "prefix_b_0_key"],
            ["prefix_a_0_value", "prefix_b_0_value"],
            None,
            None,
            [3, 4],
            [0, 1],
            0.2,
        )
        self.manager.cache_task_queue.put_transfer_done_signal.assert_called_once_with(
            (cache_transfer_manager.CacheStatus.STORAGE2GPU, "7", ["a", "b", "c"], [3])
        )

    def test_read_storage_task_exception_logs(self):
        self.manager.cache_task_queue.swap_storage_to_gpu_barrier = MagicMock()
        self.manager.cache_task_queue.swap_storage_to_gpu_barrier.wait.side_effect = RuntimeError("barrier error")
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "attention_store"
        self.manager.storage_backend.query.return_value = 0

        task = ReadStorageTask(
            task_id="read_error",
            keys=["k1"],
            token_ids=[1, 2],
            gpu_block_ids=[0],
            start_read_block_idx=0,
            timeout=0.1,
        )

        self.manager.read_storage_task(task)

        self.assertGreaterEqual(cache_transfer_manager.logger.error.call_count, 1)

    def test_write_back_storage_task_skips_cached_keys(self):
        self.manager.cache_task_queue.swap_to_storage_barrier = MagicMock()
        self.manager.cache_task_queue.put_transfer_done_signal = MagicMock()
        self.manager._run_write_back_storage = MagicMock()

        # Mock storage backend to return all keys exist (2 matches for 2 keys)
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "attention_store"
        self.manager.storage_backend.query.return_value = 2

        task = WriteStorageTask(
            task_id="5", keys=["k1", "k2"], token_ids=[1, 2, 3, 4], gpu_block_ids=[0, 1], timeout=0.3
        )

        self.manager.write_back_storage_task(task)

        self.manager._run_write_back_storage.assert_not_called()
        self.manager.cache_task_queue.put_transfer_done_signal.assert_called_once_with(
            (cache_transfer_manager.CacheStatus.GPU2STORAGE, "5", ["k1", "k2"], [])
        )

    def test_read_storage_task_no_matches(self):
        self.manager.cache_task_queue.swap_storage_to_gpu_barrier = MagicMock()
        self.manager.cache_task_queue.put_transfer_done_signal = MagicMock()

        # Mock storage backend to return 0 matches
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "attention_store"
        self.manager.storage_backend.query.return_value = 0

        task = ReadStorageTask(
            task_id="3", keys=["a"], token_ids=[1, 2], gpu_block_ids=[2], start_read_block_idx=0, timeout=0.1
        )

        self.manager.read_storage_task(task)

        self.manager.cache_task_queue.put_transfer_done_signal.assert_called_once_with(
            (cache_transfer_manager.CacheStatus.STORAGE2GPU, "3", ["a"], [])
        )

    def test_init_cpu_cache_no_blocks_sets_ready(self):
        class DummySignal:
            def __init__(self):
                self.value = [0]

        args = Args()
        args.num_cpu_blocks = 0
        self.manager.swap_space_ready_signal = DummySignal()

        self._orig_init_cpu_cache(self.manager, args)

        self.assertEqual(self.manager.swap_space_ready_signal.value[0], 1)

    def test_init_cpu_cache_allocates_block_wise_fp8(self):
        class DummySignal:
            def __init__(self):
                self.value = [0]

        args = Args()
        args.num_cpu_blocks = 2
        args.cache_dtype = "block_wise_fp8"
        args.value_cache_shape = "2,1,1,1"
        self.manager.swap_space_ready_signal = DummySignal()
        self.manager.value_cache_shape = [2, 1, 1, 1]

        with (
            patch(
                "fastdeploy.cache_manager.cache_transfer_manager.cuda_host_alloc",
                side_effect=[10, 11, 12, 13],
            ),
            patch("fastdeploy.cache_manager.cache_transfer_manager.paddle.set_device"),
        ):
            self._orig_init_cpu_cache(self.manager, args)

        self.assertEqual(self.manager.swap_space_ready_signal.value[0], 1)

    def test_init_gpu_cache_block_wise_fp8_create_tensor(self):
        class DummySignal:
            def __init__(self):
                self.value = [0]

        class LocalArgs(Args):
            cache_dtype = "block_wise_fp8"
            kvcache_storage_backend = "mooncake"
            create_cache_tensor = True
            num_layers = 1
            key_cache_shape = "2,1,1,1"
            value_cache_shape = "2,1,1,1"

        with (
            patch.object(CacheTransferManager, "_init_cpu_cache", lambda self, args: None),
            patch.object(CacheTransferManager, "_init_gpu_cache", lambda self, args: None),
            patch("fastdeploy.cache_manager.cache_transfer_manager.MooncakeStore"),
            patch.object(CacheTransferManager, "_init_storage_buffer"),
        ):
            manager = CacheTransferManager(LocalArgs())

        manager.cache_ready_signal = DummySignal()
        manager.gpu_cache_kvs = {}
        manager.gpu_cache_k_tensors = []
        manager.gpu_cache_v_tensors = []
        manager.gpu_cache_scales_k_tensors = []
        manager.gpu_cache_scales_v_tensors = []

        with (
            patch("fastdeploy.cache_manager.cache_transfer_manager.set_device"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.set_data_ipc") as mock_set_ipc,
            patch("fastdeploy.cache_manager.cache_transfer_manager.memory_allocated", return_value=0),
        ):
            self._orig_init_gpu_cache(manager, LocalArgs())

        self.assertEqual(mock_set_ipc.call_count, 4)
        self.assertIn("key_caches_0_rank0.device0", manager.gpu_cache_kvs)

    def test_init_gpu_cache_attach_block_wise_fp8(self):
        class DummySignal:
            def __init__(self):
                self.value = [1]

        class LocalArgs(Args):
            cache_dtype = "block_wise_fp8"
            kvcache_storage_backend = "mooncake"
            create_cache_tensor = False
            num_layers = 1
            key_cache_shape = "2,1,1,1"
            value_cache_shape = "2,1,1,1"

        with (
            patch.object(CacheTransferManager, "_init_cpu_cache", lambda self, args: None),
            patch.object(CacheTransferManager, "_init_gpu_cache", lambda self, args: None),
            patch("fastdeploy.cache_manager.cache_transfer_manager.MooncakeStore"),
            patch.object(CacheTransferManager, "_init_storage_buffer"),
        ):
            manager = CacheTransferManager(LocalArgs())

        manager.cache_ready_signal = DummySignal()
        manager.gpu_cache_kvs = {}
        manager.gpu_cache_k_tensors = []
        manager.gpu_cache_v_tensors = []
        manager.gpu_cache_scales_k_tensors = []
        manager.gpu_cache_scales_v_tensors = []

        def fake_share(tensor, name, shape, _):
            return paddle.zeros(shape=shape, dtype=tensor.dtype)

        with (
            patch("fastdeploy.cache_manager.cache_transfer_manager.set_device"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.share_external_data_", side_effect=fake_share),
            patch("fastdeploy.cache_manager.cache_transfer_manager.memory_allocated", return_value=0),
        ):
            self._orig_init_gpu_cache(manager, LocalArgs())

        self.assertIn("key_cache_scales_0_rank0.device0", manager.gpu_cache_kvs)

    def test_init_gpu_cache_waits_for_ready_signal(self):
        class LocalArgs(Args):
            create_cache_tensor = False
            cache_dtype = "bfloat16"
            num_layers = 1
            key_cache_shape = "1,1,1,1"
            value_cache_shape = "1,1,1,1"

        with (
            patch.object(CacheTransferManager, "_init_cpu_cache", lambda self, args: None),
            patch.object(CacheTransferManager, "_init_gpu_cache", lambda self, args: None),
        ):
            manager = CacheTransferManager(LocalArgs())

        class DummySignal:
            def __init__(self, value):
                self.value = [value]

        manager.cache_ready_signal = DummySignal(0)
        manager.gpu_cache_kvs = {}
        manager.gpu_cache_k_tensors = []
        manager.gpu_cache_v_tensors = []

        def fake_share(tensor, name, shape, _):
            return paddle.zeros(shape=shape, dtype=tensor.dtype)

        def fake_sleep(_):
            manager.cache_ready_signal.value[0] = 1

        with (
            patch("fastdeploy.cache_manager.cache_transfer_manager.time.sleep", side_effect=fake_sleep),
            patch("fastdeploy.cache_manager.cache_transfer_manager.set_device"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.share_external_data_", side_effect=fake_share),
            patch("fastdeploy.cache_manager.cache_transfer_manager.memory_allocated", return_value=0),
        ):
            self._orig_init_gpu_cache(manager, LocalArgs())

        self.assertIn("key_caches_0_rank0.device0", manager.gpu_cache_kvs)

    def test_init_storage_buffer_registers_buffers(self):
        class DummyStorage:
            def __init__(self):
                self.registered = []

            def register_buffer(self, ptr, size):
                self.registered.append((ptr, size))

        args = Args()
        args.max_model_len = 4
        args.num_layers = 1
        self.manager.storage_backend = DummyStorage()
        self.manager.cache_dtype = "bfloat16"
        self.manager.key_cache_shape = [2, 1, 2, 2]
        self.manager.num_extra_layers = 0

        with patch("fastdeploy.cache_manager.cache_transfer_manager.cuda_host_alloc", side_effect=[1000, 2000]):
            self.manager._init_storage_buffer(args)

        self.assertEqual(self.manager.storage_key_read_buffer, 1000)

    def test_init_storage_buffer_registers_scale_buffers(self):
        class DummyStorage:
            def __init__(self):
                self.registered = []

            def register_buffer(self, ptr, size):
                self.registered.append((ptr, size))

        args = Args()
        args.max_model_len = 4
        self.manager.storage_backend = DummyStorage()
        self.manager.has_cache_scale = True
        self.manager.scale_item_bytes = 2
        self.manager.cache_item_bytes = 2
        self.manager.head_num = 1
        self.manager.block_size = 2
        self.manager.head_dim = 2
        self.manager.num_layers = 1
        self.manager.num_extra_layers = 0

        with patch("fastdeploy.cache_manager.cache_transfer_manager.cuda_host_alloc", side_effect=[10, 20, 30, 40]):
            self.manager._init_storage_buffer(args)

        self.assertEqual(len(self.manager.storage_backend.registered), 4)
        self.assertEqual(self.manager.storage_key_scale_read_buffer, 30)

    def test_init_with_mooncake_storage_backend(self):
        class LocalArgs(Args):
            kvcache_storage_backend = "mooncake"

        with (
            patch("fastdeploy.cache_manager.cache_transfer_manager.MooncakeStore") as mock_store,
            patch.object(CacheTransferManager, "_init_storage_buffer") as mock_init_buffer,
        ):
            manager = CacheTransferManager(LocalArgs())

        mock_store.assert_called_once_with(tp_rank=manager.rank)
        self.assertIsInstance(mock_init_buffer.call_args[0][0], LocalArgs)

    def test_init_with_file_storage_backend(self):
        class LocalArgs(Args):
            kvcache_storage_backend = "file"

        with (
            patch("fastdeploy.cache_manager.cache_transfer_manager.FileStore") as mock_store,
            patch.object(CacheTransferManager, "_init_storage_buffer") as mock_init_buffer,
        ):
            manager = CacheTransferManager(LocalArgs())

        mock_store.assert_called_once_with(namespace=manager.model_id, tp_rank=manager.rank, tp_size=manager.n_ranks)
        self.assertIsInstance(mock_init_buffer.call_args[0][0], LocalArgs)

    def test_init_with_attention_store_backend(self):
        class LocalArgs(Args):
            kvcache_storage_backend = "attention_store"
            key_cache_shape = "1,1,1,1"
            value_cache_shape = "1,1,1,1"

        with (
            patch("fastdeploy.cache_manager.cache_transfer_manager.AttentionStore") as mock_store,
            patch.object(CacheTransferManager, "_init_cpu_cache"),
            patch.object(CacheTransferManager, "_init_gpu_cache"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value.start = MagicMock()
            manager = CacheTransferManager(LocalArgs())

        mock_store.assert_called_once_with(
            namespace=manager.model_id,
            shard_id=manager.rank,
            shard_num=manager.n_ranks,
            layer_num=manager.num_layers + manager.num_extra_layers,
            block_token_size=manager.block_size,
            bytes_per_shard_layer_per_block=manager.head_num
            * manager.block_size
            * manager.head_dim
            * manager.cache_item_bytes,
            device_id=manager.device,
            dp_id=manager.local_data_parallel_id,
        )

    def test_invalid_write_policy_raises(self):
        class LocalArgs(Args):
            write_policy = "invalid"

        with self.assertRaises(ValueError):
            CacheTransferManager(LocalArgs())

    def test_write_back_storage_task_nonzero_rank_no_signal(self):
        self.manager.cache_task_queue.swap_to_storage_barrier = MagicMock()
        self.manager.cache_task_queue.put_transfer_done_signal = MagicMock()
        self.manager._run_write_back_storage = MagicMock()
        self.manager.rank = 1

        # Mock storage backend to return 0 matches (no keys exist)
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "attention_store"
        self.manager.storage_backend.query.return_value = 0

        task = WriteStorageTask(task_id="9", keys=["k1"], token_ids=[1, 2], gpu_block_ids=[0], timeout=0.1)

        self.manager.write_back_storage_task(task)

        self.manager._run_write_back_storage.assert_called_once_with(
            "9",
            [1, 2],
            0,
            ["prefix_k1_1_key"],
            ["prefix_k1_1_value"],
            None,
            None,
            [0],
            [0],
            0.1,
        )
        self.manager.cache_task_queue.put_transfer_done_signal.assert_not_called()

    def test_get_key_prefix_from_version(self):
        with patch("fastdeploy.cache_manager.cache_transfer_manager.yaml.safe_load") as mock_load:
            mock_load.return_value = {"version": "RL-STEP02-20250101-uuid"}
            with patch("builtins.open", unittest.mock.mock_open(read_data="version: RL-STEP02-20250101-uuid")):
                key_prefix = cache_transfer_manager.get_key_prefix_from_version("fake_path/version.yaml")

        self.assertEqual(key_prefix, "RL-STEP02")

    def test_run_write_back_storage_sets_backend(self):
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "mooncake"
        self.manager.storage_key_write_buffer = 3000
        self.manager.storage_value_write_buffer = 4000
        self.manager.cache_buffer_stride_bytes = 8
        self.manager.key_cache_shape = [2, 1, 1, 1]
        self.manager.gpu_cache_k_tensors = [paddle.zeros([1])]
        self.manager.gpu_cache_v_tensors = [paddle.zeros([1])]
        self.manager.device = 0

        with patch("fastdeploy.cache_manager.cache_transfer_manager.swap_cache_layout"):
            self.manager._run_write_back_storage("test_task", [1, 2], 0, ["k1"], ["v1"], None, None, [2], [0], 30.0)

        self.manager.storage_backend.batch_set.assert_called_once_with(
            keys=["k1", "v1"],
            target_locations=[3000, 4000],
            target_sizes=[self.manager.cache_buffer_stride_bytes] * 2,
        )

    def test_run_write_back_storage_error_returns_zero(self):
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "mooncake"
        self.manager.storage_key_write_buffer = 3000
        self.manager.storage_value_write_buffer = 4000
        self.manager.cache_buffer_stride_bytes = 8
        self.manager.key_cache_shape = [2, 1, 1, 1]
        self.manager.gpu_cache_k_tensors = [paddle.zeros([1])]
        self.manager.gpu_cache_v_tensors = [paddle.zeros([1])]
        self.manager.device = 0
        self.manager.storage_backend.batch_set.side_effect = RuntimeError("write error")

        with patch("fastdeploy.cache_manager.cache_transfer_manager.swap_cache_layout"):
            result = self.manager._run_write_back_storage("task", [1, 2], 0, ["k1"], ["v1"], None, None, [0], [0], 0.1)

        self.assertEqual(result, 0)

    def test_run_read_storage_with_scales(self):
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "mooncake"
        self.manager.storage_key_read_buffer = 100
        self.manager.storage_value_read_buffer = 200
        self.manager.storage_key_scale_read_buffer = 300
        self.manager.storage_value_scale_read_buffer = 400
        self.manager.cache_buffer_stride_bytes = 10
        self.manager.scale_buffer_stride_bytes = 5
        self.manager.key_cache_shape = [2, 1, 1, 1]
        self.manager.value_cache_shape = [2, 1, 1, 1]
        self.manager.cache_scale_shape = [2, 1, 1]
        self.manager.device = 0
        self.manager.gpu_cache_k_tensors = [paddle.zeros([1])]
        self.manager.gpu_cache_v_tensors = [paddle.zeros([1])]
        self.manager.gpu_cache_scales_k_tensors = [paddle.zeros([1])]
        self.manager.gpu_cache_scales_v_tensors = [paddle.zeros([1])]
        self.manager.storage_backend.batch_get.return_value = [1, 1, 1, 1, 1, 0, 1, 0]

        with patch("fastdeploy.cache_manager.cache_transfer_manager.swap_cache_layout") as mock_swap:
            valid_ids = self.manager._run_read_storage(
                "task",
                [1, 2],
                0,
                ["k1", "k2"],
                ["v1", "v2"],
                ["ks1", "ks2"],
                ["vs1", "vs2"],
                [9, 10],
                [0, 1],
                0.1,
            )

        self.assertEqual(valid_ids, [9])
        self.assertEqual(mock_swap.call_count, 4)

    def test_run_read_storage_attention_store(self):
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "attention_store"
        self.manager.num_layers = 1
        self.manager.num_extra_layers = 0
        self.manager.device = 0
        self.manager.rank = 0
        self.manager.gpu_cache_kvs = {
            "key_caches_0_rank0.device0": paddle.zeros([1]),
            "value_caches_0_rank0.device0": paddle.zeros([1]),
        }
        self.manager.storage_backend.read.return_value = 1

        valid_ids = self.manager._run_read_storage(
            "task_read",
            [1, 2],
            0,
            ["k1"],
            ["v1"],
            None,
            None,
            [9],
            [0],
            0.1,
        )

        self.assertEqual(valid_ids, [9])
        self.manager.storage_backend.read.assert_called_once()

    def test_read_storage_task_flush_error_logs(self):
        self.manager.cache_task_queue.swap_storage_to_gpu_barrier = MagicMock()
        self.manager.cache_task_queue.put_transfer_done_signal = MagicMock()
        self.manager._run_read_storage = MagicMock(return_value=[1])
        self.manager.block_size = 1
        self.manager.rank = 0
        self.manager.has_cache_scale = True

        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "attention_store"
        self.manager.storage_backend.query.return_value = 1
        self.manager.storage_backend.flush_token_index.side_effect = RuntimeError("flush failed")

        task = ReadStorageTask(
            task_id="flush_fail",
            keys=["k1"],
            token_ids=[1, 2],
            gpu_block_ids=[0],
            start_read_block_idx=0,
            timeout=0.1,
        )

        self.manager.read_storage_task(task)

        self.manager.storage_backend.flush_token_index.assert_called_once()

    def test_run_write_back_storage_attention_store(self):
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "attention_store"
        self.manager.num_layers = 1
        self.manager.num_extra_layers = 0
        self.manager.device = 0
        self.manager.rank = 0
        self.manager.gpu_cache_kvs = {
            "key_caches_0_rank0.device0": paddle.zeros([1]),
            "value_caches_0_rank0.device0": paddle.zeros([1]),
        }
        self.manager.storage_backend.write.return_value = 1

        write_count = self.manager._run_write_back_storage(
            "task_write",
            [1, 2],
            0,
            ["k1"],
            ["v1"],
            None,
            None,
            [9],
            [0],
            0.1,
        )

        self.assertEqual(write_count, 1)
        self.manager.storage_backend.write.assert_called_once()

    def test_run_write_back_storage_with_scales(self):
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "mooncake"
        self.manager.storage_key_write_buffer = 3000
        self.manager.storage_value_write_buffer = 4000
        self.manager.storage_key_scale_write_buffer = 5000
        self.manager.storage_value_scale_write_buffer = 6000
        self.manager.cache_buffer_stride_bytes = 8
        self.manager.scale_buffer_stride_bytes = 4
        self.manager.key_cache_shape = [2, 1, 1, 1]
        self.manager.cache_scale_shape = [2, 1, 1]
        self.manager.gpu_cache_k_tensors = [paddle.zeros([1])]
        self.manager.gpu_cache_v_tensors = [paddle.zeros([1])]
        self.manager.gpu_cache_scales_k_tensors = [paddle.zeros([1])]
        self.manager.gpu_cache_scales_v_tensors = [paddle.zeros([1])]
        self.manager.device = 0

        with patch("fastdeploy.cache_manager.cache_transfer_manager.swap_cache_layout") as mock_swap:
            self.manager._run_write_back_storage(
                "task",
                [1, 2],
                0,
                ["k1"],
                ["v1"],
                ["ks1"],
                ["vs1"],
                [0],
                [0],
                0.1,
            )

        self.assertEqual(mock_swap.call_count, 4)
        self.manager.storage_backend.batch_set.assert_called_once()

    def test_write_back_storage_task_flush_error_logs(self):
        self.manager.cache_task_queue.swap_to_storage_barrier = MagicMock()
        self.manager.cache_task_queue.put_transfer_done_signal = MagicMock()
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "attention_store"
        self.manager.storage_backend.query.return_value = 0
        self.manager.storage_backend.flush_token_index.side_effect = RuntimeError("flush failed")
        self.manager._run_write_back_storage = MagicMock(return_value=0)
        self.manager.rank = 0

        task = WriteStorageTask(
            task_id="flush_fail",
            keys=["k1"],
            token_ids=[1, 2],
            gpu_block_ids=[0],
            timeout=0.1,
        )

        self.manager.write_back_storage_task(task)

        self.manager.storage_backend.flush_token_index.assert_called_once()

    def test_read_storage_task_handles_run_error(self):
        self.manager.cache_task_queue.swap_storage_to_gpu_barrier = MagicMock()
        self.manager.cache_task_queue.put_transfer_done_signal = MagicMock()
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "mooncake"
        self.manager.storage_backend.query.return_value = 1
        self.manager._run_read_storage = MagicMock(side_effect=RuntimeError("read failed"))

        task = ReadStorageTask(
            task_id="read_fail",
            keys=["k1"],
            token_ids=[1, 2],
            gpu_block_ids=[3],
            start_read_block_idx=0,
            timeout=0.1,
        )

        self.manager.read_storage_task(task)

        self.manager.cache_task_queue.put_transfer_done_signal.assert_called_once_with(
            (cache_transfer_manager.CacheStatus.STORAGE2GPU, "read_fail", ["k1"], [])
        )

    def test_write_back_storage_task_handles_run_error(self):
        self.manager.cache_task_queue.swap_to_storage_barrier = MagicMock()
        self.manager.cache_task_queue.put_transfer_done_signal = MagicMock()
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "mooncake"
        self.manager.storage_backend.query.return_value = 0
        self.manager._run_write_back_storage = MagicMock(side_effect=RuntimeError("write failed"))
        self.manager.rank = 0

        task = WriteStorageTask(
            task_id="write_fail",
            keys=["k1"],
            token_ids=[1, 2],
            gpu_block_ids=[0],
            timeout=0.1,
        )

        self.manager.write_back_storage_task(task)

        self.manager.cache_task_queue.put_transfer_done_signal.assert_called_once_with(
            (cache_transfer_manager.CacheStatus.GPU2STORAGE, "write_fail", ["k1"], [])
        )

    def test_write_back_storage_task_exception_logs(self):
        self.manager.cache_task_queue.swap_to_storage_barrier = MagicMock()
        self.manager.cache_task_queue.swap_to_storage_barrier.wait.side_effect = RuntimeError("barrier error")
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = "attention_store"
        self.manager.storage_backend.query.return_value = 0

        task = WriteStorageTask(
            task_id="write_error",
            keys=["k1"],
            token_ids=[1, 2],
            gpu_block_ids=[0],
            timeout=0.1,
        )

        self.manager.write_back_storage_task(task)

        self.assertGreaterEqual(cache_transfer_manager.logger.error.call_count, 1)

    def test_write_back_storage_task_with_scale_keys_and_tuple_backend(self):
        self.manager.cache_task_queue.swap_to_storage_barrier = MagicMock()
        self.manager.cache_task_queue.put_transfer_done_signal = MagicMock()
        self.manager.storage_backend = MagicMock()
        self.manager.storage_backend_type = ("mooncake", "file")
        self.manager.has_cache_scale = True
        self.manager.storage_backend.query.return_value = 0
        self.manager._run_write_back_storage = MagicMock(return_value=1)
        self.manager.rank = 0

        task = WriteStorageTask(
            task_id="tuple_backend",
            keys=["k1"],
            token_ids=[1, 2],
            gpu_block_ids=[0],
            timeout=0.1,
        )

        self.manager.write_back_storage_task(task)

        self.manager._run_write_back_storage.assert_called_once()
        args = self.manager._run_write_back_storage.call_args[0]
        self.assertIn("key_scale", args[5][0])
        self.assertIn("value_scale", args[6][0])

    # ==========================
    # transfer_data 分支测试
    # ==========================
    def test_transfer_data_block_wise_fp8_swap_paths(self):
        self.manager.cache_dtype = "block_wise_fp8"
        tensor = paddle.zeros([1], dtype="float32")
        self.assertEqual(tensor.dtype, paddle.float32)
        self.manager.gpu_cache_k_tensors = [tensor]
        self.manager.gpu_cache_v_tensors = [tensor]
        self.manager.gpu_cache_scales_k_tensors = [tensor]
        self.manager.gpu_cache_scales_v_tensors = [tensor]
        self.manager.k_dst_ptrs = [1]
        self.manager.v_dst_ptrs = [2]
        self.manager.k_scales_ptrs = [3]
        self.manager.v_scales_ptrs = [4]
        self.manager.num_cpu_blocks = 1
        self.manager.device = 0

        with patch("fastdeploy.cache_manager.cache_transfer_manager.swap_cache_all_layers"):
            result_cpu = self.manager._transfer_data(
                [0], [0], [0], cache_transfer_manager.CacheStatus.SWAP2CPU, transfer_task_id=11
            )
            result_gpu = self.manager._transfer_data(
                [0], [0], [0], cache_transfer_manager.CacheStatus.SWAP2GPU, transfer_task_id=12
            )

        self.assertEqual(result_cpu[0], cache_transfer_manager.CacheStatus.SWAP2CPU)
        self.assertEqual(result_gpu[0], cache_transfer_manager.CacheStatus.SWAP2GPU)

    def test_transfer_data_unknown_event_logs_warning(self):
        dummy_event = MagicMock()
        dummy_event.value = 999
        with patch.object(cache_transfer_manager.logger, "warning") as mock_warning:
            result = self.manager._transfer_data([1], [2], [3], dummy_event, transfer_task_id=9)
        self.assertEqual(result[0], dummy_event)
        mock_warning.assert_called_once()

    def test_transfer_data_length_mismatch_logs_error(self):
        self.manager.cache_dtype = "bfloat16"
        event_type = cache_transfer_manager.CacheStatus.SWAP2CPU
        with patch.object(cache_transfer_manager.logger, "error") as mock_error:
            with self.assertRaises(AssertionError):
                self.manager._transfer_data([0], [0], [0, 1], event_type, transfer_task_id=5)

        mock_error.assert_called_once()

    def test_do_swap_tasks_signal_and_return(self):
        self.manager.cache_task_queue.swap_to_cpu_barrier1 = MagicMock()
        self.manager.cache_task_queue.swap_to_cpu_barrier2 = MagicMock()
        self.manager.cache_task_queue.swap_to_gpu_barrier1 = MagicMock()
        self.manager.cache_task_queue.swap_to_gpu_barrier2 = MagicMock()
        self.manager.cache_task_queue.put_transfer_done_signal = MagicMock()
        self.manager.rank = 0
        self.manager._transfer_data = MagicMock(return_value=("event", 1, [0], [1], [2]))

        self.manager._do_swap_to_cpu_task([0], [1], [2], cache_transfer_manager.CacheStatus.SWAP2CPU, 1)
        self.manager._do_swap_to_gpu_task([0], [1], [2], cache_transfer_manager.CacheStatus.SWAP2GPU, 2)

        self.assertEqual(self.manager.cache_task_queue.put_transfer_done_signal.call_count, 2)

    def test_do_data_transfer_swap_to_cpu(self):
        class DummySignal:
            def __init__(self, value):
                self.value = [value]

        self.manager.rank = 0
        self.manager.n_ranks = 1
        self.manager.cache_task_broadcast_signal = DummySignal(1)
        self.manager.cache_task_queue.empty.return_value = False
        self.manager.check_work_status = MagicMock(return_value=(False, "Not Healthy"))

        data = (cache_transfer_manager.CacheStatus.SWAP2CPU, 7, [0], [1], [2])
        self.manager.cache_task_queue.get_transfer_task.side_effect = [
            (data, False),
            BrokenPipeError("done"),
        ]

        with patch("fastdeploy.cache_manager.cache_transfer_manager.envs.FD_CACHE_PROC_ERROR_COUNT", 0):
            self.manager.do_data_transfer()

        self.manager.swap_to_cpu_thread_pool.submit.assert_called_once()

    def test_do_data_transfer_paused_waits_for_inflight(self):
        class DummySignal:
            def __init__(self, value):
                self.value = [value]

        self.manager.rank = 0
        self.manager.n_ranks = 1
        self.manager.is_paused = True
        self.manager.inflight = 1
        self.manager.cache_task_is_paused_signal = DummySignal(0)
        self.manager.cache_task_inflight_signal = DummySignal(1)

        call_count = {"count": 0}

        def fake_sleep(_):
            call_count["count"] += 1
            if call_count["count"] == 1:
                self.manager.inflight = 0
                return None
            raise SystemExit

        with patch("fastdeploy.cache_manager.cache_transfer_manager.time.sleep", side_effect=fake_sleep):
            with self.assertRaises(SystemExit):
                self.manager.do_data_transfer()

        self.assertEqual(self.manager.cache_task_inflight_signal.value[0], 0)

    def test_do_data_transfer_storage2gpu_submits(self):
        class DummySignal:
            def __init__(self, value):
                self.value = [value]

        self.manager.rank = 0
        self.manager.n_ranks = 1
        self.manager.cache_task_is_paused_signal = DummySignal(0)
        self.manager.cache_task_inflight_signal = DummySignal(1)
        self.manager.cache_task_broadcast_signal = DummySignal(1)
        self.manager.cache_task_queue.empty.return_value = False
        self.manager.read_storage_thread_pool = MagicMock()
        self.manager.submit_task = MagicMock()

        task = ReadStorageTask(
            task_id="storage_read",
            keys=["k1"],
            token_ids=[1, 2],
            gpu_block_ids=[0],
            start_read_block_idx=0,
            timeout=0.1,
        )

        self.manager.cache_task_queue.get_transfer_task.side_effect = [
            ((cache_transfer_manager.CacheStatus.STORAGE2GPU, task), True),
            SystemExit,
        ]

        with self.assertRaises(SystemExit):
            self.manager.do_data_transfer()

        self.manager.submit_task.assert_called_once_with(
            self.manager.read_storage_thread_pool, self.manager.read_storage_task, task
        )

    def test_do_data_transfer_no_broadcast_barrier2(self):
        class DummySignal:
            def __init__(self, value):
                self.value = [value]

        self.manager.rank = 0
        self.manager.n_ranks = 2
        self.manager.cache_task_is_paused_signal = DummySignal(0)
        self.manager.cache_task_inflight_signal = DummySignal(1)
        self.manager.cache_task_broadcast_signal = DummySignal(0)
        self.manager.cache_task_queue.empty.return_value = True
        self.manager.cache_task_queue.barrier0 = MagicMock()
        self.manager.cache_task_queue.barrier1 = MagicMock()
        self.manager.cache_task_queue.barrier2 = MagicMock()
        self.manager.cache_task_queue.barrier2.reset.side_effect = SystemExit

        with self.assertRaises(SystemExit):
            self.manager.do_data_transfer()

        self.manager.cache_task_queue.barrier2.wait.assert_called_once()

    def test_check_cache_status_clearing_updates_flags(self):
        class DummySignal:
            def __init__(self, value):
                self.value = [value]

        args = Args()
        args.splitwise_role = "mixed"
        args.create_cache_tensor = False
        self.manager.kv_cache_status_signal = DummySignal(cache_transfer_manager.KVCacheStatus.CLEARING)
        self.manager.cache_ready_signal = DummySignal(0)
        self.manager.swap_space_ready_signal = DummySignal(0)
        self.manager.gpu_cache_kvs = {"k": paddle.zeros([1])}
        self.manager.gpu_cache_k_tensors = [paddle.zeros([1])]
        self.manager.gpu_cache_v_tensors = [paddle.zeros([1])]
        self.manager.num_cpu_blocks = 0

        def maybe_stop_cleared(*_):
            if self.manager.kv_cache_status_signal.value[0] == cache_transfer_manager.KVCacheStatus.CLEARED:
                raise StopIteration

        with (
            patch("fastdeploy.cache_manager.cache_transfer_manager.unset_data_ipc") as mock_unset,
            patch.object(self.manager, "pause"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.set_device"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.envs.FD_ENABLE_SWAP_SPACE_CLEARING", False),
            patch("time.sleep", side_effect=maybe_stop_cleared),
        ):
            with self.assertRaises(StopIteration):
                self.manager.check_cache_status(args)

        self.assertEqual(self.manager.kv_cache_status_signal.value[0], cache_transfer_manager.KVCacheStatus.CLEARED)
        mock_unset.assert_called_once()

    def test_check_cache_status_clearing_with_create_tensor_clears_gpu_cache(self):
        class DummySignal:
            def __init__(self, value):
                self.value = [value]

        args = Args()
        args.splitwise_role = "mixed"
        args.create_cache_tensor = True
        self.manager.kv_cache_status_signal = DummySignal(cache_transfer_manager.KVCacheStatus.CLEARING)
        self.manager.cache_ready_signal = DummySignal(0)
        self.manager.swap_space_ready_signal = DummySignal(0)
        self.manager.gpu_cache_kvs = {"k": paddle.zeros([1])}
        self.manager.gpu_cache_k_tensors = [paddle.zeros([1])]
        self.manager.gpu_cache_v_tensors = [paddle.zeros([1])]
        self.manager.num_cpu_blocks = 0

        def maybe_stop_cleared_with_tensor(*_):
            if self.manager.kv_cache_status_signal.value[0] == cache_transfer_manager.KVCacheStatus.CLEARED:
                raise StopIteration

        with (
            patch("fastdeploy.cache_manager.cache_transfer_manager.unset_data_ipc"),
            patch.object(self.manager, "pause"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.set_device"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.envs.FD_ENABLE_SWAP_SPACE_CLEARING", False),
            patch("paddle.device.cuda.empty_cache") as mock_empty,
            patch("paddle.set_device"),
            patch.object(self.manager, "_log_memory"),
            patch("time.sleep", side_effect=maybe_stop_cleared_with_tensor),
        ):
            with self.assertRaises(StopIteration):
                self.manager.check_cache_status(args)

        self.assertEqual(self.manager.kv_cache_status_signal.value[0], cache_transfer_manager.KVCacheStatus.CLEARED)
        mock_empty.assert_called_once()

    def test_check_cache_status_clearing_waits_for_swap_space(self):
        class DummySignal:
            def __init__(self, value):
                self.value = [value]

        args = Args()
        args.splitwise_role = "mixed"
        args.create_cache_tensor = False
        self.manager.kv_cache_status_signal = DummySignal(cache_transfer_manager.KVCacheStatus.CLEARING)
        self.manager.cache_ready_signal = DummySignal(0)
        self.manager.swap_space_ready_signal = DummySignal([1])
        self.manager.num_cpu_blocks = 1
        self.manager.k_dst_ptrs = [11]
        self.manager.v_dst_ptrs = [22]
        self.manager.cpu_cache_kvs = {"k": 11, "v": 22}

        def fake_sleep(_):
            if self.manager.swap_space_ready_signal.value[0] == 1:
                self.manager.swap_space_ready_signal.value[0] = 0
                return None
            if self.manager.kv_cache_status_signal.value[0] == cache_transfer_manager.KVCacheStatus.CLEARED:
                raise StopIteration
            return None

        with (
            patch("fastdeploy.cache_manager.cache_transfer_manager.unset_data_ipc"),
            patch.object(self.manager, "pause"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.cuda_host_free"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.paddle.set_device"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.envs.FD_ENABLE_SWAP_SPACE_CLEARING", True),
            patch("fastdeploy.cache_manager.cache_transfer_manager.time.sleep", side_effect=fake_sleep),
        ):
            with self.assertRaises(StopIteration):
                self.manager.check_cache_status(args)

        self.assertEqual(self.manager.swap_space_ready_signal.value[0], 0)

    def test_check_cache_status_clearing_with_swap_space_clear(self):
        class DummySignal:
            def __init__(self, value):
                self.value = value

        args = Args()
        args.splitwise_role = "mixed"
        args.create_cache_tensor = True
        args.mp_num = 2
        self.manager.kv_cache_status_signal = DummySignal([cache_transfer_manager.KVCacheStatus.CLEARING])
        self.manager.cache_ready_signal = DummySignal([1, 1])
        self.manager.swap_space_ready_signal = DummySignal([0, 0])
        self.manager.num_cpu_blocks = 1
        self.manager.cache_dtype = "block_wise_fp8"
        self.manager.k_dst_ptrs = [101]
        self.manager.v_dst_ptrs = [202]
        self.manager.k_scales_ptrs = [303]
        self.manager.v_scales_ptrs = [404]
        self.manager.cpu_cache_kvs = {"k": 101, "v": 202}
        self.manager.gpu_cache_kvs = {"k": paddle.zeros([1])}
        self.manager.gpu_cache_k_tensors = [paddle.zeros([1])]
        self.manager.gpu_cache_v_tensors = [paddle.zeros([1])]
        self.manager.gpu_cache_scales_k_tensors = [paddle.zeros([1])]
        self.manager.gpu_cache_scales_v_tensors = [paddle.zeros([1])]

        sleep_calls = {"count": 0}

        def fake_sleep(_):
            sleep_calls["count"] += 1
            if sleep_calls["count"] == 1:
                self.manager.cache_ready_signal.value[0] = 0
            elif sleep_calls["count"] == 2:
                self.manager.cache_ready_signal.value[1] = 0
            elif sleep_calls["count"] >= 3:
                raise StopIteration

        with (
            patch("fastdeploy.cache_manager.cache_transfer_manager.unset_data_ipc"),
            patch.object(self.manager, "pause"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.cuda_host_free"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.paddle.set_device"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.envs.FD_ENABLE_SWAP_SPACE_CLEARING", True),
            patch("paddle.device.cuda.empty_cache"),
            patch.object(self.manager, "_log_memory"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.time.sleep", side_effect=fake_sleep),
        ):
            with self.assertRaises(StopIteration):
                self.manager.check_cache_status(args)

        self.assertEqual(self.manager.kv_cache_status_signal.value[0], cache_transfer_manager.KVCacheStatus.CLEARED)

    def test_check_cache_status_updating_sets_normal(self):
        class DummySignal:
            def __init__(self, value):
                self.value = [value]

        args = Args()
        args.splitwise_role = "mixed"
        args.mp_num = 1
        self.manager.kv_cache_status_signal = DummySignal(cache_transfer_manager.KVCacheStatus.UPDATING)
        self.manager.cache_ready_signal = DummySignal(1)
        self.manager.swap_space_ready_signal = DummySignal(1)
        self.manager.num_cpu_blocks = 0

        def maybe_stop_normal(*_):
            if self.manager.kv_cache_status_signal.value[0] == cache_transfer_manager.KVCacheStatus.NORMAL:
                raise StopIteration

        with (
            patch.object(self.manager, "_init_cpu_cache"),
            patch.object(self.manager, "_init_gpu_cache"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.unset_data_ipc"),
            patch.object(self.manager, "resume"),
            patch("fastdeploy.cache_manager.cache_transfer_manager.envs.FD_ENABLE_SWAP_SPACE_CLEARING", False),
            patch("time.sleep", side_effect=maybe_stop_normal),
        ):
            with self.assertRaises(StopIteration):
                self.manager.check_cache_status(args)

        self.assertEqual(self.manager.kv_cache_status_signal.value[0], cache_transfer_manager.KVCacheStatus.NORMAL)

    def test_check_cache_status_updating_restores_with_version(self):
        class DummySignal:
            def __init__(self, value):
                self.value = value

        args = Args()
        args.splitwise_role = "mixed"
        args.mp_num = 2
        self.manager.kv_cache_status_signal = DummySignal([cache_transfer_manager.KVCacheStatus.UPDATING])
        self.manager.cache_ready_signal = DummySignal([0, 1])
        self.manager.swap_space_ready_signal = DummySignal([0, 1])
        self.manager.num_cpu_blocks = 1
        self.manager.storage_backend_type = "mooncake"

        with tempfile.TemporaryDirectory() as tmpdir:
            version_path = os.path.join(tmpdir, "version.yaml")
            with open(version_path, "w", encoding="utf-8") as handle:
                handle.write("version: RL-STEP99-20250101-uuid\n")
            args.model_path = tmpdir

            sleep_calls = {"count": 0}

            def fake_sleep(_):
                sleep_calls["count"] += 1
                if sleep_calls["count"] == 1:
                    self.manager.swap_space_ready_signal.value = [1, 1]
                elif sleep_calls["count"] == 2:
                    self.manager.cache_ready_signal.value = [1, 1]
                elif sleep_calls["count"] >= 3:
                    raise StopIteration

            with (
                patch("fastdeploy.cache_manager.cache_transfer_manager.unset_data_ipc"),
                patch.object(self.manager, "_init_cpu_cache"),
                patch.object(self.manager, "_init_gpu_cache"),
                patch.object(self.manager, "resume"),
                patch("fastdeploy.cache_manager.cache_transfer_manager.envs.FD_ENABLE_SWAP_SPACE_CLEARING", True),
                patch.object(self.manager, "_log_memory"),
                patch("fastdeploy.cache_manager.cache_transfer_manager.time.sleep", side_effect=fake_sleep),
            ):
                with self.assertRaises(StopIteration):
                    self.manager.check_cache_status(args)

        self.assertEqual(self.manager.kv_cache_status_signal.value[0], cache_transfer_manager.KVCacheStatus.NORMAL)

    def test_log_memory_records_gpu_stats(self):
        with patch.object(cache_transfer_manager.logger, "warning") as mock_warning:
            with (
                patch("paddle.device.cuda.max_memory_allocated", return_value=1024**3),
                patch("paddle.device.cuda.max_memory_reserved", return_value=2 * 1024**3),
                patch("paddle.device.cuda.memory_allocated", return_value=512 * 1024**2),
                patch("paddle.device.cuda.memory_reserved", return_value=768 * 1024**2),
            ):
                self.manager._log_memory("test")

        mock_warning.assert_called_once()

    def test_pause_and_resume_wait_for_signals(self):
        class DummySignal:
            def __init__(self, value):
                self.value = value

        self.manager.rank = 0
        self.manager.n_ranks = 2
        self.manager.cache_task_queue.pause_barrier = MagicMock()
        self.manager.cache_task_queue.resume_barrier = MagicMock()
        self.manager.cache_task_inflight_signal = DummySignal([1, 0])

        def pause_sleep(_):
            self.manager.cache_task_inflight_signal.value = [0, 0]

        with patch("fastdeploy.cache_manager.cache_transfer_manager.time.sleep", side_effect=pause_sleep):
            self.manager.pause()

        self.assertTrue(self.manager.is_paused)

        def resume_sleep(_):
            self.manager.cache_task_inflight_signal.value = [1, 1]

        with patch("fastdeploy.cache_manager.cache_transfer_manager.time.sleep", side_effect=resume_sleep):
            self.manager.resume()

        self.assertFalse(self.manager.is_paused)

    def test_submit_task_decrements_inflight(self):
        class DummyPool:
            def submit(self, fn, *args):
                try:
                    fn(*args)
                except RuntimeError:
                    return None

        def raise_task():
            raise RuntimeError("boom")

        self.manager.inflight = 1
        self.manager.submit_task(DummyPool(), raise_task)
        self.assertEqual(self.manager.inflight, 0)

    def test_main_invokes_manager(self):
        cache_transfer_manager.args = Args()
        with patch("fastdeploy.cache_manager.cache_transfer_manager.CacheTransferManager") as mock_manager:
            mock_instance = mock_manager.return_value
            mock_instance.do_data_transfer.side_effect = SystemExit
            with self.assertRaises(SystemExit):
                cache_transfer_manager.main()

        mock_manager.assert_called_once_with(cache_transfer_manager.args)


if __name__ == "__main__":
    unittest.main()
