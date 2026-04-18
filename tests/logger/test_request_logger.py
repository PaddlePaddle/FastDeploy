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

import unittest
from unittest.mock import patch

from fastdeploy.logger.request_logger import (
    RequestLogLevel,
    _should_log,
    _truncate,
    log_request,
    log_request_error,
)


class TestRequestLogLevel(unittest.TestCase):
    """Test RequestLogLevel enum"""

    def test_level_values(self):
        """Test level values"""
        self.assertEqual(int(RequestLogLevel.LIFECYCLE), 0)
        self.assertEqual(int(RequestLogLevel.STAGES), 1)
        self.assertEqual(int(RequestLogLevel.CONTENT), 2)
        self.assertEqual(int(RequestLogLevel.FULL), 3)


class TestShouldLog(unittest.TestCase):
    """Test _should_log function"""

    def test_disabled_returns_false(self):
        """FD_LOG_REQUESTS=0 should return False"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 0
            mock_envs.FD_LOG_REQUESTS_LEVEL = 3
            self.assertFalse(_should_log(RequestLogLevel.LIFECYCLE))

    def test_level_within_threshold(self):
        """Level within threshold should return True"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 1
            mock_envs.FD_LOG_REQUESTS_LEVEL = 2
            self.assertTrue(_should_log(RequestLogLevel.LIFECYCLE))
            self.assertTrue(_should_log(RequestLogLevel.STAGES))
            self.assertTrue(_should_log(RequestLogLevel.CONTENT))

    def test_level_above_threshold(self):
        """Level above threshold should return False"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 1
            mock_envs.FD_LOG_REQUESTS_LEVEL = 1
            self.assertFalse(_should_log(RequestLogLevel.CONTENT))
            self.assertFalse(_should_log(RequestLogLevel.FULL))


class TestTruncate(unittest.TestCase):
    """Test _truncate function"""

    def test_short_text_unchanged(self):
        """Short text should remain unchanged"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_MAX_LEN = 100
            result = _truncate("short text")
            self.assertEqual(result, "short text")

    def test_long_text_truncated(self):
        """Long text should be truncated"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_MAX_LEN = 10
            result = _truncate("this is a very long text")
            self.assertEqual(result, "this is a ")
            self.assertEqual(len(result), 10)

    def test_non_string_converted(self):
        """Non-string should be converted"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_MAX_LEN = 100
            result = _truncate(12345)
            self.assertEqual(result, "12345")


class TestLogRequest(unittest.TestCase):
    """Test log_request function"""

    @patch("fastdeploy.logger._request_logger")
    def test_log_when_enabled(self, mock_logger):
        """Should log when enabled"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 1
            mock_envs.FD_LOG_REQUESTS_LEVEL = 0
            mock_envs.FD_LOG_MAX_LEN = 2048

            log_request(RequestLogLevel.LIFECYCLE, message="test {value}", value="hello")
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertEqual(call_args, "test hello")

    @patch("fastdeploy.logger._request_logger")
    def test_no_log_when_disabled(self, mock_logger):
        """Should not log when disabled"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 0
            mock_envs.FD_LOG_REQUESTS_LEVEL = 3

            log_request(RequestLogLevel.LIFECYCLE, message="test {value}", value="hello")
            mock_logger.info.assert_not_called()

    @patch("fastdeploy.logger._request_logger")
    def test_no_log_when_level_too_high(self, mock_logger):
        """Should not log when level is too high"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 1
            mock_envs.FD_LOG_REQUESTS_LEVEL = 0

            log_request(RequestLogLevel.CONTENT, message="test {value}", value="hello")
            mock_logger.info.assert_not_called()

    @patch("fastdeploy.logger._request_logger")
    def test_content_level_truncates_content(self, mock_logger):
        """CONTENT level should truncate content"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 1
            mock_envs.FD_LOG_REQUESTS_LEVEL = 3
            mock_envs.FD_LOG_MAX_LEN = 5

            log_request(RequestLogLevel.CONTENT, message="content: {data}", data="very long data")
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertEqual(call_args, "content: very ")

    @patch("fastdeploy.logger._request_logger")
    def test_lifecycle_level_no_truncation(self, mock_logger):
        """LIFECYCLE level should not truncate content"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 1
            mock_envs.FD_LOG_REQUESTS_LEVEL = 3
            mock_envs.FD_LOG_MAX_LEN = 5

            log_request(RequestLogLevel.LIFECYCLE, message="content: {data}", data="very long data")
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertEqual(call_args, "content: very long data")


class TestLogRequestError(unittest.TestCase):
    """Test log_request_error function"""

    @patch("fastdeploy.logger._request_logger")
    def test_error_with_fields(self, mock_logger):
        """Error log with fields should format message"""
        log_request_error(message="request {request_id} failed: {error}", request_id="req-123", error="timeout")
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        self.assertEqual(call_args, "request req-123 failed: timeout")

    @patch("fastdeploy.logger._request_logger")
    def test_error_without_fields(self, mock_logger):
        """Error log without fields should not call format"""
        log_request_error(message="simple error message")
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        self.assertEqual(call_args, "simple error message")


if __name__ == "__main__":
    unittest.main()
