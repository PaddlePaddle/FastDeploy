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

import logging
import unittest
from unittest import mock


class TestInterceptPaddleLoggers(unittest.TestCase):
    """Test cases for intercept_paddle_loggers context manager from tools.logger_patch"""

    def test_intercept_paddle_loggers_with_paddle_prefix(self):
        """Test intercept_paddle_loggers configures paddle loggers correctly (line 28-30)"""
        from tools.logger_patch import intercept_paddle_loggers

        # Create a logger with existing handlers before interception
        test_logger_name = "paddle.test.logger"
        test_logger = logging.getLogger(test_logger_name)

        # Add some handlers to the logger
        handler1 = logging.StreamHandler()
        handler2 = logging.StreamHandler()
        test_logger.addHandler(handler1)
        test_logger.addHandler(handler2)
        self.assertEqual(len(test_logger.handlers), 2)

        # Use the context manager to intercept paddle loggers
        with intercept_paddle_loggers():
            # Get logger inside context - should be configured by interceptor
            intercepted_logger = logging.getLogger(test_logger_name)

            # Verify the logger was reconfigured by the interceptor
            self.assertEqual(len(intercepted_logger.handlers), 1)
            self.assertIsInstance(intercepted_logger.handlers[0], logging.StreamHandler)
            self.assertEqual(intercepted_logger.level, logging.INFO)
            self.assertFalse(intercepted_logger.propagate)

        # Clean up
        test_logger.handlers = []

    def test_intercept_paddle_loggers_restores_original(self):
        """Test intercept_paddle_loggers restores original getLogger after exit (line 46)"""
        from tools.logger_patch import intercept_paddle_loggers

        # Store original getLogger before context
        original_getLogger = logging.getLogger

        # Use the context manager
        with intercept_paddle_loggers():
            # Inside context, getLogger should be patched
            self.assertNotEqual(logging.getLogger, original_getLogger)

        # After exit, getLogger should be restored
        self.assertEqual(logging.getLogger, original_getLogger)

    def test_intercept_paddle_loggers_non_paddle_logger_unchanged(self):
        """Test non-paddle loggers are not affected by intercept_paddle_loggers"""
        from tools.logger_patch import intercept_paddle_loggers

        # Create a non-paddle logger
        test_logger_name = "other.test.logger"
        test_logger = logging.getLogger(test_logger_name)

        # Add a handler
        original_handler = logging.StreamHandler()
        test_logger.addHandler(original_handler)
        original_handler_count = len(test_logger.handlers)

        # Use the context manager
        with intercept_paddle_loggers():
            # Get the same logger
            result_logger = logging.getLogger(test_logger_name)
            # Non-paddle loggers should not be modified
            self.assertEqual(len(result_logger.handlers), original_handler_count)
            self.assertEqual(result_logger.handlers[0], original_handler)

        # Clean up
        test_logger.handlers = []

    def test_intercept_paddle_loggers_exception_safety(self):
        """Test intercept_paddle_loggers restores getLogger even if exception occurs"""
        from tools.logger_patch import intercept_paddle_loggers

        original_getLogger = logging.getLogger

        try:
            with intercept_paddle_loggers():
                # Raise an exception inside context
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # After exception, getLogger should still be restored
        self.assertEqual(logging.getLogger, original_getLogger)


class TestGetWorker(unittest.TestCase):
    """Test cases for get_worker function - covering different platform branches"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_fd_config = mock.MagicMock()
        self.mock_fd_config.model_config.enable_logprob = False

    @mock.patch("fastdeploy.worker.worker_process.current_platform")
    @mock.patch("fastdeploy.worker.worker_process.configure_third_party_loggers")
    def test_get_worker_dcu_platform(self, mock_configure_loggers, mock_platform):
        """Test get_worker for DCU platform (line 111-115)"""
        mock_platform.is_dcu.return_value = True
        mock_platform.is_cuda.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_gcu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False

        from fastdeploy.worker.worker_process import get_worker

        # Mock the import inside get_worker function
        mock_dcu_module = mock.MagicMock()
        mock_worker_instance = mock.MagicMock()
        mock_dcu_module.DcuWorker.return_value = mock_worker_instance

        with mock.patch.dict("sys.modules", {"fastdeploy.worker.dcu_worker": mock_dcu_module}):
            result = get_worker(self.mock_fd_config, local_rank=0, rank=0)

            # Verify configure_third_party_loggers was called (line 114)
            mock_configure_loggers.assert_called_once()
            mock_dcu_module.DcuWorker.assert_called_once_with(fd_config=self.mock_fd_config, local_rank=0, rank=0)
            self.assertEqual(result, mock_worker_instance)

    @mock.patch("fastdeploy.worker.worker_process.current_platform")
    @mock.patch("fastdeploy.worker.worker_process.configure_third_party_loggers")
    def test_get_worker_xpu_platform(self, mock_configure_loggers, mock_platform):
        """Test get_worker for XPU platform (line 121-125)"""
        mock_platform.is_dcu.return_value = False
        mock_platform.is_cuda.return_value = False
        mock_platform.is_xpu.return_value = True
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_gcu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False

        from fastdeploy.worker.worker_process import get_worker

        # Mock the import inside get_worker function
        mock_xpu_module = mock.MagicMock()
        mock_worker_instance = mock.MagicMock()
        mock_xpu_module.XpuWorker.return_value = mock_worker_instance

        with mock.patch.dict("sys.modules", {"fastdeploy.worker.xpu_worker": mock_xpu_module}):
            result = get_worker(self.mock_fd_config, local_rank=0, rank=0)

            # Verify configure_third_party_loggers was called (line 119)
            mock_configure_loggers.assert_called_once()
            mock_xpu_module.XpuWorker.assert_called_once_with(fd_config=self.mock_fd_config, local_rank=0, rank=0)
            self.assertEqual(result, mock_worker_instance)

    @mock.patch("fastdeploy.worker.worker_process.current_platform")
    @mock.patch("fastdeploy.worker.worker_process.configure_third_party_loggers")
    def test_get_worker_iluvatar_platform(self, mock_configure_loggers, mock_platform):
        """Test get_worker for Iluvatar platform (line 126-130)"""
        mock_platform.is_dcu.return_value = False
        mock_platform.is_cuda.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_iluvatar.return_value = True
        mock_platform.is_gcu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False

        from fastdeploy.worker.worker_process import get_worker

        # Mock the import inside get_worker function
        mock_iluvatar_module = mock.MagicMock()
        mock_worker_instance = mock.MagicMock()
        mock_iluvatar_module.IluvatarWorker.return_value = mock_worker_instance

        with mock.patch.dict("sys.modules", {"fastdeploy.worker.iluvatar_worker": mock_iluvatar_module}):
            result = get_worker(self.mock_fd_config, local_rank=0, rank=0)

            # Verify configure_third_party_loggers was called (line 129)
            mock_configure_loggers.assert_called_once()
            mock_iluvatar_module.IluvatarWorker.assert_called_once_with(
                fd_config=self.mock_fd_config, local_rank=0, rank=0
            )
            self.assertEqual(result, mock_worker_instance)

    @mock.patch("fastdeploy.worker.worker_process.current_platform")
    @mock.patch("fastdeploy.worker.worker_process.configure_third_party_loggers")
    def test_get_worker_gcu_platform(self, mock_configure_loggers, mock_platform):
        """Test get_worker for GCU platform (line 131-135)"""
        mock_platform.is_dcu.return_value = False
        mock_platform.is_cuda.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_gcu.return_value = True
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False

        from fastdeploy.worker.worker_process import get_worker

        # Mock the import inside get_worker function
        mock_gcu_module = mock.MagicMock()
        mock_worker_instance = mock.MagicMock()
        mock_gcu_module.GcuWorker.return_value = mock_worker_instance

        with mock.patch.dict("sys.modules", {"fastdeploy.worker.gcu_worker": mock_gcu_module}):
            result = get_worker(self.mock_fd_config, local_rank=0, rank=0)

            # Verify configure_third_party_loggers was called (line 134)
            mock_configure_loggers.assert_called_once()
            mock_gcu_module.GcuWorker.assert_called_once_with(fd_config=self.mock_fd_config, local_rank=0, rank=0)
            self.assertEqual(result, mock_worker_instance)

    @mock.patch("fastdeploy.worker.worker_process.current_platform")
    @mock.patch("fastdeploy.worker.worker_process.configure_third_party_loggers")
    def test_get_worker_maca_platform(self, mock_configure_loggers, mock_platform):
        """Test get_worker for MACA (Metax) platform (line 136-140)"""
        mock_platform.is_dcu.return_value = False
        mock_platform.is_cuda.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_gcu.return_value = False
        mock_platform.is_maca.return_value = True
        mock_platform.is_intel_hpu.return_value = False

        from fastdeploy.worker.worker_process import get_worker

        # Mock the import inside get_worker function
        mock_metax_module = mock.MagicMock()
        mock_worker_instance = mock.MagicMock()
        mock_metax_module.MetaxWorker.return_value = mock_worker_instance

        with mock.patch.dict("sys.modules", {"fastdeploy.worker.metax_worker": mock_metax_module}):
            result = get_worker(self.mock_fd_config, local_rank=0, rank=0)

            # Verify configure_third_party_loggers was called (line 139)
            mock_configure_loggers.assert_called_once()
            mock_metax_module.MetaxWorker.assert_called_once_with(fd_config=self.mock_fd_config, local_rank=0, rank=0)
            self.assertEqual(result, mock_worker_instance)

    @mock.patch("fastdeploy.worker.worker_process.current_platform")
    @mock.patch("fastdeploy.worker.worker_process.configure_third_party_loggers")
    def test_get_worker_intel_hpu_platform(self, mock_configure_loggers, mock_platform):
        """Test get_worker for Intel HPU platform (line 141-145)"""
        mock_platform.is_dcu.return_value = False
        mock_platform.is_cuda.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_gcu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = True

        from fastdeploy.worker.worker_process import get_worker

        # Mock the import inside get_worker function
        mock_hpu_module = mock.MagicMock()
        mock_worker_instance = mock.MagicMock()
        mock_hpu_module.HpuWorker.return_value = mock_worker_instance

        with mock.patch.dict("sys.modules", {"fastdeploy.worker.hpu_worker": mock_hpu_module}):
            result = get_worker(self.mock_fd_config, local_rank=0, rank=0)

            # Verify configure_third_party_loggers was called (line 144)
            mock_configure_loggers.assert_called_once()
            mock_hpu_module.HpuWorker.assert_called_once_with(fd_config=self.mock_fd_config, local_rank=0, rank=0)
            self.assertEqual(result, mock_worker_instance)

    @mock.patch("fastdeploy.worker.worker_process.current_platform")
    def test_get_worker_logprob_not_supported(self, mock_platform):
        """Test get_worker raises error when logprob is enabled on unsupported platform"""
        mock_platform.is_cuda.return_value = False
        mock_platform.is_xpu.return_value = False

        from fastdeploy.worker.worker_process import get_worker

        mock_fd_config = mock.MagicMock()
        mock_fd_config.model_config.enable_logprob = True

        with self.assertRaises(NotImplementedError) as context:
            get_worker(mock_fd_config, local_rank=0, rank=0)

        self.assertIn("Only CUDA and XPU platforms support logprob", str(context.exception))


class TestConfigureThirdPartyLoggers(unittest.TestCase):
    """Test cases for configure_third_party_loggers function from setup_logging"""

    def test_configure_third_party_loggers_paddleformers(self):
        """Test configure_third_party_loggers configures paddleformers loggers"""
        from fastdeploy.logger.setup_logging import configure_third_party_loggers

        # Get paddleformers loggers before configuration
        pf_logger = logging.getLogger("paddleformers")
        original_handlers = list(pf_logger.handlers)

        # Clear existing handlers for clean test
        pf_logger.handlers = []
        pf_logger.addHandler(logging.StreamHandler())

        # Call the function
        configure_third_party_loggers()

        # Verify the logger was configured
        self.assertEqual(len(pf_logger.handlers), 1)
        self.assertIsInstance(pf_logger.handlers[0], logging.StreamHandler)
        self.assertEqual(pf_logger.level, logging.INFO)

        # Restore original handlers
        pf_logger.handlers = original_handlers

    def test_configure_third_party_loggers_paddle(self):
        """Test configure_third_party_loggers configures paddle loggers"""
        from fastdeploy.logger.setup_logging import configure_third_party_loggers

        # Create a test paddle logger
        pd_logger = logging.getLogger("paddle.test.module")
        original_handlers = list(pd_logger.handlers)

        # Clear and add a handler to test removal
        pd_logger.handlers = []
        pd_logger.addHandler(logging.StreamHandler())

        # Call the function
        configure_third_party_loggers()

        # Verify the logger was configured
        self.assertEqual(len(pd_logger.handlers), 1)
        self.assertIsInstance(pd_logger.handlers[0], logging.StreamHandler)
        self.assertEqual(pd_logger.level, logging.INFO)
        self.assertFalse(pd_logger.propagate)

        # Restore original handlers
        pd_logger.handlers = original_handlers


if __name__ == "__main__":
    unittest.main()
