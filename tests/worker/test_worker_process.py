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
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

import pytest

from fastdeploy.config import FDConfig
from fastdeploy.engine.request import ControlRequest
from fastdeploy.worker.worker_process import PaddleDisWorkerProc


class TestInterceptPaddleLoggers(unittest.TestCase):
    """Test cases for intercept_paddle_loggers context manager from tools.logger_patch"""

    def test_intercept_paddle_loggers_with_paddle_prefix(self):
        """Test intercept_paddle_loggers configures paddle loggers correctly"""
        from fastdeploy.logger.logger import intercept_paddle_loggers

        # Create a logger with existing handlers before interception
        test_logger_name = "paddle.test.logger"
        test_logger = logging.getLogger(test_logger_name)

        # Add some handlers to the logger
        handler1 = logging.StreamHandler()
        handler2 = logging.StreamHandler()
        test_logger.addHandler(handler1)
        test_logger.addHandler(handler2)
        self.assertEqual(len(test_logger.handlers), 2)

        # Use context manager to intercept paddle loggers
        with intercept_paddle_loggers():
            # Get logger inside context - should be configured by interceptor
            intercepted_logger = logging.getLogger(test_logger_name)

            # Verify the logger was reconfigured by interceptor
            self.assertEqual(len(intercepted_logger.handlers), 1)
            self.assertIsInstance(intercepted_logger.handlers[0], logging.StreamHandler)
            self.assertEqual(intercepted_logger.level, logging.INFO)
            self.assertFalse(intercepted_logger.propagate)

        # Clean up
        test_logger.handlers = []

    def test_intercept_paddle_loggers_restores_original(self):
        """Test intercept_paddle_loggers restores original getLogger after exit"""
        from fastdeploy.logger.logger import intercept_paddle_loggers

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
        from fastdeploy.logger.logger import intercept_paddle_loggers

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
        from fastdeploy.logger.logger import intercept_paddle_loggers

        original_getLogger = logging.getLogger

        try:
            with intercept_paddle_loggers():
                # Raise an exception inside context
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # After exception, getLogger should still be restored
        self.assertEqual(logging.getLogger, original_getLogger)


class TestWorkerProcessControlMethod(unittest.TestCase):
    """Test cases for PaddleDisWorkerProc control method handling - Coverage for lines 761-786"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_fd_config = Mock(spec=FDConfig)
        self.mock_fd_config.parallel_config = Mock()
        self.mock_fd_config.parallel_config.use_ep = False
        self.mock_fd_config.parallel_config.tensor_parallel_size = 1
        self.mock_fd_config.load_config = Mock()
        self.mock_fd_config.load_config.dynamic_load_weight = False

        self.process = PaddleDisWorkerProc.__new__(PaddleDisWorkerProc)
        self.process.fd_config = self.mock_fd_config
        self.process.parallel_config = self.mock_fd_config.parallel_config
        self.process.local_rank = 0
        self.process.eplb_config = types.SimpleNamespace(enable_eplb=False)

        # Mock worker - use spec to avoid auto-creating Mock methods
        self.process.worker = Mock(spec=[])  # Empty spec = no methods defined

        # Create async mock for queue
        self.mock_queue = Mock()
        self.mock_queue.put = AsyncMock()
        self.process._ctrl_output = self.mock_queue

    def test_run_control_method_unknown_handler(self):
        """Test run_control_method with unknown control method"""
        # Create a request with unknown method
        request = ControlRequest(request_id="test_id", method="unknown_method", args={})

        self.process.run_control_method(request)

        # Verify put was called with error response
        self.mock_queue.put.assert_called_once()
        call_args = self.mock_queue.put.call_args[0][0]
        self.assertEqual(call_args.request_id, "test_id")
        self.assertEqual(call_args.error_code, 400)

    def test_run_control_method_non_callable_handler(self):
        """Test run_control_method with non-callable handler"""
        # Add a non-callable attribute to worker
        self.process.worker.some_method = "not_callable"

        request = ControlRequest(request_id="test_id", method="some_method", args={})

        self.process.run_control_method(request)

        # Verify put was called with error response
        self.mock_queue.put.assert_called_once()
        call_args = self.mock_queue.put.call_args[0][0]
        self.assertEqual(call_args.error_code, 400)

    def test_run_control_method_success(self):
        """Test run_control_method with successful execution"""
        # Add a callable method to worker
        mock_result = {"result": "success"}
        self.process.worker.test_method = Mock(return_value=mock_result)

        request = ControlRequest(request_id="test_id", method="test_method", args={"param": "value"})

        self.process.run_control_method(request)

        # Verify handler was called with args
        self.process.worker.test_method.assert_called_once_with(param="value")

        # Verify put was called with success response
        self.mock_queue.put.assert_called_once()
        call_args = self.mock_queue.put.call_args[0][0]
        self.assertEqual(call_args.request_id, "test_id")
        self.assertEqual(call_args.error_code, 200)

    def test_run_control_method_exception(self):
        """Test run_control_method with exception in handler"""

        # Add a method that raises exception
        def failing_method(**kwargs):
            raise ValueError("Test error")

        self.process.worker.test_method = failing_method

        request = ControlRequest(request_id="test_id", method="test_method", args={})

        with patch("fastdeploy.worker.worker_process.traceback") as mock_traceback:
            mock_traceback.format_exc.return_value = "Traceback..."

            self.process.run_control_method(request)

            # Verify put was called with error response
            self.mock_queue.put.assert_called_once()
            call_args = self.mock_queue.put.call_args[0][0]
            self.assertEqual(call_args.request_id, "test_id")
            self.assertEqual(call_args.error_code, 500)

    def test_run_control_directly_when_not_use_ep(self):
        """Test running control request directly when use_ep is disabled"""
        self.process.parallel_config.use_ep = False

        # Add a callable method to worker
        self.process.worker.test_method = Mock(return_value={"result": "ok"})

        control_req = ControlRequest(request_id="test_id", method="test_method", args={})

        self.process.run_control_method(control_req)

        # Verify handler was called
        self.process.worker.test_method.assert_called_once()

        # Verify put was called
        self.mock_queue.put.assert_called_once()

    @pytest.mark.skip("This case might hang in ci environment, to be fixed in the future")
    def test_event_loop_caches_ep_control_requests_before_collective_run(self):
        self.process.parallel_config.use_ep = True
        self.process.parallel_config.ep_group = Mock(world_size=1)
        self.process.cached_control_reqs = []
        self.process._run_eplb = Mock()
        self.process._tp_barrier_wait = Mock()
        self.process.run_control_method = Mock()
        self.process.worker_healthy_live_signal = Mock(value=[0])
        self.process.max_chips_per_node = 8
        self.process.nnode = 1
        self.process.ranks = 1
        self.process.task_queue = Mock()
        self.process.task_queue.exist_tasks.return_value = False
        self.process.task_queue.read_finish_flag = types.SimpleNamespace(get=Mock(return_value=1))
        control_req = ControlRequest(request_id="ep-ctrl", method="pause", args={})
        self.process.task_queue.get_tasks.return_value = ([([control_req], 1)], False)
        self.process.exist_task_signal = types.SimpleNamespace(value=[1])
        self.process.worker = types.SimpleNamespace(
            preprocess_new_task=Mock(),
            model_runner=types.SimpleNamespace(),
            execute_model=Mock(),
            exist_prefill=Mock(return_value=False),
        )
        with (
            patch("fastdeploy.utils.all_gather_values", side_effect=SystemExit),
            patch("fastdeploy.worker.worker_process.all_gather_values", side_effect=SystemExit),
        ):
            with self.assertRaises(SystemExit):
                self.process.event_loop_normal()

        self.assertEqual(self.process.cached_control_reqs, [control_req])
        self.process.run_control_method.assert_not_called()

    def test_event_loop_skips_execute_model_when_runner_is_sleeping(self):
        self.process.parallel_config.use_ep = False
        self.process.parallel_config.tensor_parallel_size = 2
        self.process.fd_config.load_config.dynamic_load_weight = True
        self.process.cached_control_reqs = []
        self.process._run_eplb = Mock()
        self.process._tp_barrier_wait = Mock(side_effect=SystemExit)
        self.process.worker_healthy_live_signal = Mock(value=[0])
        self.process.max_chips_per_node = 8
        self.process.nnode = 1
        self.process.ranks = 1
        self.process.local_rank = 0
        self.process.task_queue = Mock()
        self.process.task_queue.exist_tasks.return_value = False
        self.process.task_queue.read_finish_flag = types.SimpleNamespace(get=Mock(return_value=0))
        self.process.exist_task_signal = types.SimpleNamespace(value=[0])
        self.process.worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(is_sleeping=True),
            execute_model=Mock(),
            exist_prefill=Mock(return_value=False),
        )

        with patch("fastdeploy.worker.worker_process.envs.FD_ENABLE_V1_UPDATE_WEIGHTS", "1"):
            with self.assertRaises(SystemExit):
                self.process.event_loop_normal()

        self.process.worker.execute_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
