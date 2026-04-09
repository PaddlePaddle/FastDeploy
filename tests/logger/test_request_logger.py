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
)


class TestRequestLogLevel(unittest.TestCase):
    """测试 RequestLogLevel 枚举"""

    def test_level_values(self):
        """测试级别值"""
        self.assertEqual(int(RequestLogLevel.L0), 0)
        self.assertEqual(int(RequestLogLevel.L1), 1)
        self.assertEqual(int(RequestLogLevel.L2), 2)
        self.assertEqual(int(RequestLogLevel.L3), 3)


class TestShouldLog(unittest.TestCase):
    """测试 _should_log 函数"""

    def test_disabled_returns_false(self):
        """FD_LOG_REQUESTS=0 应该返回 False"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 0
            mock_envs.FD_LOG_REQUESTS_LEVEL = 3
            self.assertFalse(_should_log(0))

    def test_level_within_threshold(self):
        """级别在阈值内应该返回 True"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 1
            mock_envs.FD_LOG_REQUESTS_LEVEL = 2
            self.assertTrue(_should_log(0))
            self.assertTrue(_should_log(1))
            self.assertTrue(_should_log(2))

    def test_level_above_threshold(self):
        """级别超过阈值应该返回 False"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 1
            mock_envs.FD_LOG_REQUESTS_LEVEL = 1
            self.assertFalse(_should_log(2))
            self.assertFalse(_should_log(3))


class TestTruncate(unittest.TestCase):
    """测试 _truncate 函数"""

    def test_short_text_unchanged(self):
        """短文本应该保持不变"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_MAX_LEN = 100
            result = _truncate("short text")
            self.assertEqual(result, "short text")

    def test_long_text_truncated(self):
        """长文本应该被截断"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_MAX_LEN = 10
            result = _truncate("this is a very long text")
            self.assertEqual(result, "this is a ")
            self.assertEqual(len(result), 10)

    def test_non_string_converted(self):
        """非字符串应该被转换"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_MAX_LEN = 100
            result = _truncate(12345)
            self.assertEqual(result, "12345")


class TestLogRequest(unittest.TestCase):
    """测试 log_request 函数"""

    @patch("fastdeploy.logger.request_logger._request_logger")
    def test_log_when_enabled(self, mock_logger):
        """启用时应该记录日志"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 1
            mock_envs.FD_LOG_REQUESTS_LEVEL = 0
            mock_envs.FD_LOG_MAX_LEN = 2048

            log_request(level=0, message="test {value}", value="hello")
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertEqual(call_args, "test hello")

    @patch("fastdeploy.logger.request_logger._request_logger")
    def test_no_log_when_disabled(self, mock_logger):
        """禁用时不应该记录日志"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 0
            mock_envs.FD_LOG_REQUESTS_LEVEL = 3

            log_request(level=0, message="test {value}", value="hello")
            mock_logger.info.assert_not_called()

    @patch("fastdeploy.logger.request_logger._request_logger")
    def test_no_log_when_level_too_high(self, mock_logger):
        """级别过高时不应该记录日志"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 1
            mock_envs.FD_LOG_REQUESTS_LEVEL = 0

            log_request(level=2, message="test {value}", value="hello")
            mock_logger.info.assert_not_called()

    @patch("fastdeploy.logger.request_logger._request_logger")
    def test_l2_level_truncates_content(self, mock_logger):
        """L2 级别应该截断内容"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 1
            mock_envs.FD_LOG_REQUESTS_LEVEL = 3
            mock_envs.FD_LOG_MAX_LEN = 5

            log_request(level=2, message="content: {data}", data="very long data")
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertEqual(call_args, "content: very ")

    @patch("fastdeploy.logger.request_logger._request_logger")
    def test_l0_level_no_truncation(self, mock_logger):
        """L0 级别不应该截断内容"""
        with patch("fastdeploy.logger.request_logger.envs") as mock_envs:
            mock_envs.FD_LOG_REQUESTS = 1
            mock_envs.FD_LOG_REQUESTS_LEVEL = 3
            mock_envs.FD_LOG_MAX_LEN = 5

            log_request(level=0, message="content: {data}", data="very long data")
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertEqual(call_args, "content: very long data")


if __name__ == "__main__":
    unittest.main()
