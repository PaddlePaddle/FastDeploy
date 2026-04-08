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
import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.engine.request import RequestStatus
from fastdeploy.entrypoints.engine_client import EngineClient


class TestEngineClientAbort(unittest.TestCase):
    """Test cases for EngineClient.abort method"""

    def setUp(self):
        """Set up test fixtures"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Create a mock FDConfig
        self.mock_fd_config = MagicMock()
        self.mock_fd_config.parallel_config.tensor_parallel_size = 1
        self.mock_fd_config.model_config.enable_mm = False
        self.mock_fd_config.model_config.max_model_len = 2048
        self.mock_fd_config.model_config.enable_logprob = True
        self.mock_fd_config.cache_config.enable_prefix_caching = False
        self.mock_fd_config.scheduler_config.splitwise_role = "mixed"
        self.mock_fd_config.limit_mm_per_prompt = 5
        self.mock_fd_config.eplb_config.enable_eplb = False
        self.mock_fd_config.structured_outputs_config.reasoning_parser = None
        self.mock_fd_config.mm_processor_kwargs = {}
        self.mock_fd_config.tool_parser = None
        self.mock_fd_config.cache_config.max_processor_cache = 0

        # Create EngineClient instance
        with patch("fastdeploy.entrypoints.engine_client.InputPreprocessor"):
            with patch("fastdeploy.entrypoints.engine_client.IPCSignal"):
                with patch("fastdeploy.entrypoints.engine_client.StatefulSemaphore"):
                    with patch("fastdeploy.entrypoints.engine_client.DealerConnectionManager"):
                        with patch("fastdeploy.entrypoints.engine_client.FileLock"):
                            self.engine_client = EngineClient(
                                pid=12345, port=8000, fd_config=self.mock_fd_config, workers=1
                            )

    def tearDown(self):
        """Clean up test fixtures"""
        self.loop.close()

    @patch("fastdeploy.entrypoints.engine_client.envs.FD_ENABLE_REQUEST_DISCONNECT_STOP_INFERENCE", True)
    @patch.object(EngineClient, "_send_task")
    def test_abort_single_request(self, mock_send_task):
        """Test aborting a single request"""
        request_id = "test_request"

        # Run the abort method
        self.loop.run_until_complete(self.engine_client.abort(request_id, n=1))

        # Verify _send_task was called with correct data
        expected_data = {
            "request_id": "test_request_0",
            "status": RequestStatus.ABORT.value,
        }
        mock_send_task.assert_called_once_with(expected_data)

    @patch("fastdeploy.entrypoints.engine_client.envs.FD_ENABLE_REQUEST_DISCONNECT_STOP_INFERENCE", True)
    @patch.object(EngineClient, "_send_task")
    def test_abort_multiple_requests(self, mock_send_task):
        """Test aborting multiple requests"""
        request_id = "test_request"
        n = 3

        # Run the abort method
        self.loop.run_until_complete(self.engine_client.abort(request_id, n=n))

        # Verify _send_task was called correct number of times
        self.assertEqual(mock_send_task.call_count, n)

        # Verify each call had correct request_id
        expected_calls = [
            ({"request_id": "test_request_0", "status": RequestStatus.ABORT.value},),
            ({"request_id": "test_request_1", "status": RequestStatus.ABORT.value},),
            ({"request_id": "test_request_2", "status": RequestStatus.ABORT.value},),
        ]

        actual_calls = [call.args for call in mock_send_task.call_args_list]
        self.assertEqual(actual_calls, expected_calls)

    @patch("fastdeploy.entrypoints.engine_client.envs.FD_ENABLE_REQUEST_DISCONNECT_STOP_INFERENCE", True)
    @patch.object(EngineClient, "_send_task")
    def test_abort_with_existing_suffix(self, mock_send_task):
        """Test aborting request that already has _number suffix"""
        request_id = "test_request_123_2"
        n = 2

        # Run the abort method
        self.loop.run_until_complete(self.engine_client.abort(request_id, n=n))

        # Verify _send_task was called correct number of times
        self.assertEqual(mock_send_task.call_count, n)

        # Verify each call had correct request_id (should use prefix before existing suffix)
        expected_calls = [
            ({"request_id": "test_request_123_0", "status": RequestStatus.ABORT.value},),
            ({"request_id": "test_request_123_1", "status": RequestStatus.ABORT.value},),
        ]

        actual_calls = [call.args for call in mock_send_task.call_args_list]
        self.assertEqual(actual_calls, expected_calls)

    @patch("fastdeploy.entrypoints.engine_client.envs.FD_ENABLE_REQUEST_DISCONNECT_STOP_INFERENCE", True)
    @patch.object(EngineClient, "_send_task")
    def test_abort_with_no_suffix(self, mock_send_task):
        """Test aborting request without _number suffix"""
        request_id = "test_request_without_suffix"
        n = 2

        # Run the abort method
        self.loop.run_until_complete(self.engine_client.abort(request_id, n=n))

        # Verify _send_task was called correct number of times
        self.assertEqual(mock_send_task.call_count, n)

        # Verify each call had correct request_id (should use full request_id as prefix)
        expected_calls = [
            ({"request_id": "test_request_without_suffix_0", "status": RequestStatus.ABORT.value},),
            ({"request_id": "test_request_without_suffix_1", "status": RequestStatus.ABORT.value},),
        ]

        actual_calls = [call.args for call in mock_send_task.call_args_list]
        self.assertEqual(actual_calls, expected_calls)

    @patch("fastdeploy.entrypoints.engine_client.envs.FD_ENABLE_REQUEST_DISCONNECT_STOP_INFERENCE", True)
    @patch.object(EngineClient, "_send_task")
    def test_abort_with_zero_n(self, mock_send_task):
        """Test aborting with n=0 should not send any requests"""
        request_id = "test_request_123"

        # Run the abort method
        self.loop.run_until_complete(self.engine_client.abort(request_id, n=0))

        # Verify _send_task was not called
        mock_send_task.assert_not_called()

    @patch("fastdeploy.entrypoints.engine_client.envs.FD_ENABLE_REQUEST_DISCONNECT_STOP_INFERENCE", True)
    @patch.object(EngineClient, "_send_task")
    def test_abort_with_negative_n(self, mock_send_task):
        """Test aborting with negative n should not send any requests"""
        request_id = "test_request_123"

        # Run the abort method
        self.loop.run_until_complete(self.engine_client.abort(request_id, n=-1))

        # Verify _send_task was not called
        mock_send_task.assert_not_called()

    @patch("fastdeploy.entrypoints.engine_client.envs.FD_ENABLE_REQUEST_DISCONNECT_STOP_INFERENCE", False)
    @patch.object(EngineClient, "_send_task")
    def test_abort_when_feature_disabled(self, mock_send_task):
        """Test abort when FD_ENABLE_REQUEST_DISCONNECT_STOP_INFERENCE is False"""
        request_id = "test_request_123"

        # Run the abort method
        self.loop.run_until_complete(self.engine_client.abort(request_id, n=1))

        # Verify _send_task was not called
        mock_send_task.assert_not_called()

    @patch("fastdeploy.entrypoints.engine_client.envs.FD_ENABLE_REQUEST_DISCONNECT_STOP_INFERENCE", True)
    @patch.object(EngineClient, "_send_task")
    def test_abort_request_id_regex_parsing(self, mock_send_task):
        """Test that request_id regex parsing works correctly for various formats"""
        test_cases = [
            ("simple_request", "simple_request"),
            ("request_with_underscores", "request_with_underscores"),
            ("request_123", "request"),
            ("request_123_456", "request_123"),
            ("request_0", "request"),
            ("complex_name_123_456_789", "complex_name_123_456"),
        ]

        for input_request_id, expected_prefix in test_cases:
            with self.subTest(input_request_id=input_request_id):
                mock_send_task.reset_mock()

                # Run the abort method
                self.loop.run_until_complete(self.engine_client.abort(input_request_id, n=1))

                # Verify _send_task was called with correct prefix
                expected_data = {
                    "request_id": f"{expected_prefix}_0",
                    "status": RequestStatus.ABORT.value,
                }
                mock_send_task.assert_called_once_with(expected_data)

    @patch("fastdeploy.entrypoints.engine_client.envs.FD_ENABLE_REQUEST_DISCONNECT_STOP_INFERENCE", True)
    @patch("fastdeploy.entrypoints.engine_client.api_server_logger")
    @patch.object(EngineClient, "_send_task")
    def test_abort_logging(self, mock_send_task, mock_logger):
        """Test that abort method logs correctly"""
        request_id = "test_request"
        n = 2

        # Run the abort method
        self.loop.run_until_complete(self.engine_client.abort(request_id, n=n))

        # Verify info log was called twice
        self.assertEqual(mock_logger.info.call_count, 2)

        # Verify the first log message (abort start)
        first_call = mock_logger.info.call_args_list[0]
        self.assertEqual(first_call[0][0], "abort request_id:test_request")

        # Verify the second log message (abort completion with request IDs)
        second_call = mock_logger.info.call_args_list[1]
        expected_log_message = "Aborted request(s) %s."
        self.assertEqual(second_call[0][0], expected_log_message)
        self.assertEqual(second_call[0][1], "test_request_0,test_request_1")

    @patch("fastdeploy.entrypoints.engine_client.envs.FD_ENABLE_REQUEST_DISCONNECT_STOP_INFERENCE", True)
    @patch("fastdeploy.entrypoints.engine_client.api_server_logger")
    @patch.object(EngineClient, "_send_task")
    def test_abort_warning_logging_for_invalid_format(self, mock_send_task, mock_logger):
        """Test that abort method logs warning for invalid request_id format"""
        request_id = "invalid_format_no_suffix"  # This should actually not trigger warning
        n = 1

        # Run the abort method
        self.loop.run_until_complete(self.engine_client.abort(request_id, n=n))

        # Verify warning log was called (this case might not actually trigger warning)
        # The warning is only triggered when regex doesn't match, but our test case has valid format
        # Let's test with a case that should trigger warning
        mock_logger.reset_mock()

        # This should trigger warning because it doesn't end with _number
        request_id_no_suffix = "just_a_string"
        self.loop.run_until_complete(self.engine_client.abort(request_id_no_suffix, n=1))

        # Should have logged warning about format error
        mock_logger.warning.assert_called()


if __name__ == "__main__":
    unittest.main()
