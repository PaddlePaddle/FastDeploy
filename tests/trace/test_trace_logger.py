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
from unittest.mock import patch

import pytest

from fastdeploy.trace.constants import LoggingEventName, StageName
from fastdeploy.trace.trace_logger import print as trace_print


class TestTraceLogging(unittest.TestCase):
    """Test cases for trace_logger.print function"""

    @pytest.fixture(autouse=True)
    def inject_caplog(self, caplog):
        """Inject pytest caplog fixture into unittest"""
        self._caplog = caplog

    @patch("fastdeploy.trace.trace_logger.get_trace_info_for_request")
    def test_trace_print_with_unknown_trace_id(self, mock_get_trace_info):
        """Test trace_print when get_trace_info_for_request returns None (line 40)"""
        mock_get_trace_info.return_value = None
        request_id = "test123"
        user = "test_user"
        event = LoggingEventName.PREPROCESSING_START

        with self._caplog.at_level(logging.INFO):
            trace_print(event, request_id, user)

        self.assertEqual(len(self._caplog.records), 1)
        record = self._caplog.records[0]
        self.assertIn(f"[request_id={request_id}]", record.message)
        self.assertIn(f"[user_id={user}]", record.message)
        self.assertIn(f"[event={event.value}]", record.message)
        self.assertIn(f"[stage={StageName.PREPROCESSING.value}]", record.message)
        self.assertIn("[trace_id=unknown]", record.message)

    @patch("fastdeploy.trace.trace_logger.get_trace_info_for_request")
    def test_trace_print_with_valid_trace_id(self, mock_get_trace_info):
        """Test trace_print when get_trace_info_for_request returns valid trace info"""
        mock_get_trace_info.return_value = {"trace_id": "abc-123-xyz"}
        request_id = "test456"
        user = "test_user2"
        event = LoggingEventName.INFERENCE_START

        with self._caplog.at_level(logging.INFO):
            trace_print(event, request_id, user)

        self.assertEqual(len(self._caplog.records), 1)
        record = self._caplog.records[0]
        self.assertIn(f"[request_id={request_id}]", record.message)
        self.assertIn(f"[user_id={user}]", record.message)
        self.assertIn(f"[event={event.value}]", record.message)
        self.assertIn(f"[stage={StageName.PREFILL.value}]", record.message)
        self.assertIn("[trace_id=abc-123-xyz]", record.message)

    @patch("fastdeploy.trace.trace_logger.get_trace_info_for_request")
    def test_trace_print_different_events(self, mock_get_trace_info):
        """Test trace_print with different event types and stage mapping"""
        mock_get_trace_info.return_value = None
        test_cases = [
            (LoggingEventName.PREPROCESSING_START, StageName.PREPROCESSING),
            (LoggingEventName.REQUEST_SCHEDULE_START, StageName.SCHEDULE),
            (LoggingEventName.INFERENCE_START, StageName.PREFILL),
            (LoggingEventName.DECODE_START, StageName.DECODE),
            (LoggingEventName.POSTPROCESSING_START, StageName.POSTPROCESSING),
        ]

        for event, expected_stage in test_cases:
            self._caplog.clear()
            with self._caplog.at_level(logging.INFO):
                trace_print(event, "req_123", "user_1")

            self.assertEqual(len(self._caplog.records), 1)
            record = self._caplog.records[0]
            self.assertIn(f"[event={event.value}]", record.message)
            self.assertIn(f"[stage={expected_stage.value}]", record.message)

    @patch("fastdeploy.trace.trace_logger.get_trace_info_for_request")
    def test_trace_print_exception_handling(self, mock_get_trace_info):
        """Test trace_print handles exceptions gracefully (line 47-48)"""
        mock_get_trace_info.side_effect = Exception("Unexpected error")
        request_id = "test789"
        user = "test_user"
        event = LoggingEventName.FIRST_TOKEN_GENERATED

        # Should not raise exception
        with self._caplog.at_level(logging.INFO):
            trace_print(event, request_id, user)

        # No records should be logged due to exception handling
        self.assertEqual(len(self._caplog.records), 0)

    @patch("fastdeploy.trace.trace_logger.trace_logger")
    @patch("fastdeploy.trace.trace_logger.get_trace_info_for_request")
    def test_trace_print_logger_called_with_correct_attributes(self, mock_get_trace_info, mock_trace_logger):
        """Test that trace_logger.info is called with correct attributes structure"""
        mock_get_trace_info.return_value = {"trace_id": "test-trace-123"}
        request_id = "req_abc"
        user = "user_xyz"
        event = LoggingEventName.POSTPROCESSING_END

        trace_print(event, request_id, user)

        # Verify trace_logger.info was called
        mock_trace_logger.info.assert_called_once()
        call_args = mock_trace_logger.info.call_args

        # Check positional arguments
        self.assertEqual(call_args[0][0], "")  # First positional arg is empty string

        # Check keyword arguments
        self.assertIn("extra", call_args[1])
        self.assertIn("attributes", call_args[1]["extra"])
        attributes = call_args[1]["extra"]["attributes"]
        self.assertEqual(attributes["request_id"], request_id)
        self.assertEqual(attributes["user_id"], user)
        self.assertEqual(attributes["event"], event.value)
        self.assertEqual(attributes["stage"], StageName.POSTPROCESSING.value)
        self.assertEqual(attributes["trace_id"], "test-trace-123")

        # Verify stacklevel is set to 2
        self.assertEqual(call_args[1].get("stacklevel"), 2)
