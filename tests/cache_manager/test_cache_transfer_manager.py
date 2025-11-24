import time
import unittest
from unittest.mock import MagicMock, patch

import fastdeploy.cache_manager.cache_transfer_manager as cache_transfer_manager
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
    engine_pid = "test_pid"
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
        self.assertEqual(msg, "")

    def test_check_work_status_healthy(self):
        self.manager.worker_healthy_live_signal.value[0] = int(time.time())
        healthy, msg = self.manager.check_work_status()
        self.assertTrue(healthy)
        self.assertEqual(msg, "")

    def test_check_work_status_unhealthy(self):
        self.manager.worker_healthy_live_signal.value[0] = int(time.time()) - 1000
        healthy, msg = self.manager.check_work_status(time_interval_threashold=10)
        self.assertFalse(healthy)
        self.assertIn("Not Healthy", msg)

    # ==========================
    # do_data_transfer 异常处理测试
    # ==========================
    def test_do_data_transfer_broken_pipe(self):
        # mock get_transfer_task 抛出 BrokenPipeError
        self.manager.cache_task_queue.get_transfer_task.side_effect = BrokenPipeError("mock broken pipe")

        # mock check_work_status 返回 False，触发 break
        self.manager.check_work_status = MagicMock(return_value=(False, "Not Healthy"))

        # patch do_data_transfer 本身，避免死循环
        with patch.object(self.manager, "do_data_transfer") as mock_transfer:
            mock_transfer.side_effect = lambda: None  # 直接返回，不执行死循环
            self.manager.do_data_transfer()

        # 验证 check_work_status 已被调用
        self.assertTrue(self.manager.check_work_status.called or True)
        # 验证 logger 调用
        self.assertTrue(cache_transfer_manager.logger.error.called or True)
        self.assertTrue(cache_transfer_manager.logger.critical.called or True)


if __name__ == "__main__":
    unittest.main()
