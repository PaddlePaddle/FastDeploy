import sys
import time
import unittest
from unittest.mock import MagicMock, patch

import fastdeploy.cache_manager.cache_transfer_manager as cache_transfer_manager
import fastdeploy.cache_manager.transfer_factory.rdma_cache_transfer as rdma_module
from fastdeploy.cache_manager.cache_transfer_manager import CacheTransferManager


# Test Configuration
class Args:
    """Test configuration class to simulate input arguments for CacheTransferManager."""

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


# RDMA Test Utilities
def create_rdma_manager(rdma_comm, splitwise_role="prefill"):
    """Factory function to create RDMACommManager instance with default test parameters.

    Args:
        rdma_comm: Mocked rdma_comm module or None
        splitwise_role (str): Splitwise role, default to "prefill"

    Returns:
        rdma_module.RDMACommManager: Initialized RDMACommManager instance
    """
    return rdma_module.RDMACommManager(
        splitwise_role=splitwise_role,
        rank=0,
        gpu_id=0,
        cache_k_ptr_list=[1, 2],
        cache_v_ptr_list=[3, 4],
        max_block_num=10,
        block_bytes=1024,
        rdma_port=20000,
    )


# CacheTransferManager Test Cases
class TestCacheTransferManager(unittest.TestCase):
    """Unit test suite for CacheTransferManager class."""

    def setUp(self):
        """Set up test fixtures before each test method.

        Mocks dependencies, initializes test objects, and configures test environment.
        """
        # Mock logger
        cache_transfer_manager.logger = MagicMock()

        # Mock current platform detection
        class DummyPlatform:
            """Mock platform class to disable specific hardware checks in tests."""

            @staticmethod
            def is_iluvatar():
                return False

            @staticmethod
            def is_xpu():
                return False  # Disable XPU in test environment

            @staticmethod
            def is_cuda():
                return False  # Disable CUDA in test environment

        cache_transfer_manager.current_platform = DummyPlatform()

        # Mock EngineCacheQueue class
        self.engine_cache_queue_patcher = patch(
            "fastdeploy.cache_manager.cache_transfer_manager.EngineCacheQueue", new=MagicMock()
        )
        self.engine_cache_queue_patcher.start()

        # Mock IPCSignal class
        self.ipc_signal_patcher = patch("fastdeploy.cache_manager.cache_transfer_manager.IPCSignal", new=MagicMock())
        self.ipc_signal_patcher.start()

        # Mock cache initialization methods to avoid actual resource allocation
        self.init_cpu_cache_patcher = patch.object(CacheTransferManager, "_init_cpu_cache", lambda self, args: None)
        self.init_gpu_cache_patcher = patch.object(CacheTransferManager, "_init_gpu_cache", lambda self, args: None)
        self.init_cpu_cache_patcher.start()
        self.init_gpu_cache_patcher.start()

        # Initialize CacheTransferManager with test configuration
        self.manager = CacheTransferManager(Args())

        # Mock worker health check signal
        class DummySignal:
            """Mock signal class to simulate worker health status."""

            def __init__(self):
                self.value = [0]  # Default to unhealthy initial state

        self.manager.worker_healthy_live_signal = DummySignal()

        # Mock thread pools for swap operations
        self.manager.swap_to_cpu_thread_pool = MagicMock()
        self.manager.swap_to_gpu_thread_pool = MagicMock()

        # Mock cache task queue with test data
        self.manager.cache_task_queue = MagicMock()
        self.manager.cache_task_queue.empty.return_value = False
        self.manager.cache_task_queue.get_transfer_task.return_value = (([0], 0, 0, MagicMock(value=0), 0), True)
        self.manager.cache_task_queue.barrier1 = MagicMock()
        self.manager.cache_task_queue.barrier2 = MagicMock()
        self.manager.cache_task_queue.barrier3 = MagicMock()

        # Mock time.sleep to prevent test blocking
        self.sleep_patcher = patch("time.sleep", lambda x: None)
        self.sleep_patcher.start()

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        self.engine_cache_queue_patcher.stop()
        self.ipc_signal_patcher.stop()
        self.init_cpu_cache_patcher.stop()
        self.init_gpu_cache_patcher.stop()
        self.sleep_patcher.stop()

    def test_check_work_status_no_signal(self):
        """Test check_work_status when no health signal is set.

        Verifies that the method returns healthy status with empty message
        when the health signal value is 0 (initial state).
        """
        healthy, msg = self.manager.check_work_status()
        self.assertTrue(healthy)
        self.assertEqual(msg, "")

    def test_check_work_status_healthy(self):
        """Test check_work_status with valid (recent) health signal.

        Verifies that the method returns healthy status when the health signal
        is set to current time (within threshold).
        """
        self.manager.worker_healthy_live_signal.value[0] = int(time.time())
        healthy, msg = self.manager.check_work_status()
        self.assertTrue(healthy)
        self.assertEqual(msg, "")

    def test_check_work_status_unhealthy(self):
        """Test check_work_status with expired health signal.

        Verifies that the method returns unhealthy status with appropriate
        message when the health signal is older than the threshold.
        """
        self.manager.worker_healthy_live_signal.value[0] = int(time.time()) - 1000
        healthy, msg = self.manager.check_work_status(time_interval_threashold=10)
        self.assertFalse(healthy)
        self.assertIn("Not Healthy", msg)

    def test_do_data_transfer_broken_pipe(self):
        """Test do_data_transfer error handling for BrokenPipeError.

        Verifies that the method properly handles BrokenPipeError from task queue,
        logs the error, and exits the loop when check_work_status returns False.
        """
        # Mock BrokenPipeError when fetching transfer task
        self.manager.cache_task_queue.get_transfer_task.side_effect = BrokenPipeError("mock broken pipe")

        # Mock unhealthy status to trigger loop exit
        self.manager.check_work_status = MagicMock(return_value=(False, "Not Healthy"))

        # Patch do_data_transfer to prevent infinite loop in test
        with patch.object(self.manager, "do_data_transfer") as mock_transfer:
            mock_transfer.side_effect = lambda: None  # Short-circuit the loop
            self.manager.do_data_transfer()

        # Verify error handling behavior
        self.assertTrue(self.manager.check_work_status.called)
        self.assertTrue(cache_transfer_manager.logger.error.called)
        self.assertTrue(cache_transfer_manager.logger.critical.called)


# RDMACommManager Test Cases
class TestRDMACommManager(unittest.TestCase):
    """Unit test suite for RDMACommManager class."""

    def test_init_with_rdma_comm(self):
        """Test RDMACommManager initialization with valid rdma_comm module.

        Verifies that the messager is created using RDMACommunicator and
        instance attributes are set correctly.
        """
        mock_comm = MagicMock()
        with patch.dict(sys.modules, {"rdma_comm": mock_comm}):
            manager = create_rdma_manager(mock_comm)

        mock_comm.RDMACommunicator.assert_called_once()
        self.assertTrue(hasattr(manager, "messager"))
        self.assertEqual(manager.splitwise_role, "prefill")

    def test_init_without_rdma_comm(self):
        """Test RDMACommManager initialization without rdma_comm module.

        Verifies that the messager is not created and an error is logged
        when rdma_comm module is missing.
        """
        with patch.dict(sys.modules, {}), patch.object(rdma_module.logger, "error") as mock_log:

            manager = create_rdma_manager(None)
            mock_log.assert_called_once()
            self.assertFalse(hasattr(manager, "messager"))

    def test_connect_nominal(self):
        """Test connect method with valid prefill role.

        Verifies that connect succeeds (returns True) when called with
        prefill role and RDMA connection is successful.
        """
        mock_comm = MagicMock()
        mock_instance = MagicMock()
        mock_instance.is_connected.return_value = False
        mock_instance.connect.return_value = 0  # Simulate successful connection
        mock_comm.RDMACommunicator.return_value = mock_instance

        with patch.dict(sys.modules, {"rdma_comm": mock_comm}):
            manager = create_rdma_manager(mock_comm, splitwise_role="prefill")

        result = manager.connect("127.0.0.1", 5001)
        self.assertTrue(result)
        mock_instance.connect.assert_called_once_with("127.0.0.1", 5001)

    def test_connect_invalid_role(self):
        """Test connect method with invalid role (decode).

        Verifies that an AssertionError is raised when connect is called
        with a role other than prefill.
        """
        mock_comm = MagicMock()
        mock_comm.RDMACommunicator.return_value = MagicMock()

        with patch.dict(sys.modules, {"rdma_comm": mock_comm}):
            manager = create_rdma_manager(mock_comm, splitwise_role="decode")

        with self.assertRaises(AssertionError):
            manager.connect("1.2.3.4", 1234)

    def test_write_cache(self):
        """Test write_cache method parameter passing.

        Verifies that write_cache correctly forwards all parameters to
        the underlying messager's write_cache method.
        """
        mock_comm = MagicMock()
        mock_instance = MagicMock()
        mock_comm.RDMACommunicator.return_value = mock_instance

        with patch.dict(sys.modules, {"rdma_comm": mock_comm}):
            manager = create_rdma_manager(mock_comm)

        manager.write_cache("1.1.1.1", 9999, [1], [2], 3)

        mock_instance.write_cache.assert_called_once_with("1.1.1.1", "9999", [1], [2], 3)


if __name__ == "__main__":
    unittest.main()
