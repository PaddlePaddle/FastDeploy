import unittest
from unittest.mock import Mock

from fastdeploy.worker.worker_base import WorkerBase


class TestWorkerBase(unittest.TestCase):
    """Test cases for WorkerBase abstract class."""

    def setUp(self):
        # Create mock FDConfig
        self.mock_fd_config = Mock()
        self.mock_fd_config.model_config = Mock()
        self.mock_fd_config.load_config = Mock()
        self.mock_fd_config.parallel_config = Mock()
        self.mock_fd_config.device_config = Mock()
        self.mock_fd_config.cache_config = Mock()
        self.mock_fd_config.scheduler_config = Mock()

        # Create a concrete subclass for testing
        class TestWorker(WorkerBase):
            def init_device(self):
                self.device_initialized = True

            def initialize_cache(self, num_gpu_blocks):
                self.cache_initialized = num_gpu_blocks

            def get_model(self):
                return Mock()

            def load_model(self):
                self.model_loaded = True

            def execute_model(self, model_forward_batch=None):
                return Mock()

            def graph_optimize_and_warm_up_model(self):
                self.optimized = True

            def check_health(self):
                return True

        self.worker = TestWorker(fd_config=self.mock_fd_config, local_rank=0, rank=0)

    def test_exist_prefill(self):
        """Test exist_prefill default implementation."""
        self.assertTrue(self.worker.exist_prefill())

    def test_abstract_methods(self):
        """Verify all abstract methods are callable."""
        self.worker.init_device()
        self.assertTrue(hasattr(self.worker, "device_initialized"))

        self.worker.initialize_cache(10)
        self.assertEqual(self.worker.cache_initialized, 10)

        model = self.worker.get_model()
        self.assertIsNotNone(model)

        self.worker.load_model()
        self.assertTrue(hasattr(self.worker, "model_loaded"))

        output = self.worker.execute_model()
        self.assertIsNotNone(output)

        self.worker.graph_optimize_and_warm_up_model()
        self.assertTrue(hasattr(self.worker, "optimized"))

        self.assertTrue(self.worker.check_health())

    def test_constructor_initialization(self):
        """Test that constructor properly initializes all config attributes."""
        self.assertEqual(self.worker.fd_config, self.mock_fd_config)
        self.assertEqual(self.worker.model_config, self.mock_fd_config.model_config)
        self.assertEqual(self.worker.load_config, self.mock_fd_config.load_config)
        self.assertEqual(self.worker.parallel_config, self.mock_fd_config.parallel_config)
        self.assertEqual(self.worker.device_config, self.mock_fd_config.device_config)
        self.assertEqual(self.worker.cache_config, self.mock_fd_config.cache_config)
        self.assertEqual(self.worker.scheduler_config, self.mock_fd_config.scheduler_config)
        self.assertEqual(self.worker.local_rank, 0)
        self.assertEqual(self.worker.rank, 0)


if __name__ == "__main__":
    unittest.main()
