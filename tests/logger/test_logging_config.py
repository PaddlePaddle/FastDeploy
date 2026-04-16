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

from fastdeploy.logger.config import resolve_log_level, resolve_request_logging_defaults


class TestResolveLogLevel(unittest.TestCase):
    """测试 resolve_log_level 函数"""

    def test_explicit_info_level(self):
        """显式设置 INFO 级别"""
        result = resolve_log_level(raw_level="INFO")
        self.assertEqual(result, "INFO")

    def test_explicit_debug_level(self):
        """显式设置 DEBUG 级别"""
        result = resolve_log_level(raw_level="DEBUG")
        self.assertEqual(result, "DEBUG")

    def test_case_insensitive(self):
        """级别名称应该大小写不敏感"""
        self.assertEqual(resolve_log_level(raw_level="info"), "INFO")
        self.assertEqual(resolve_log_level(raw_level="debug"), "DEBUG")

    def test_invalid_level_raises(self):
        """无效级别应该抛出 ValueError"""
        with self.assertRaises(ValueError) as ctx:
            resolve_log_level(raw_level="INVALID")
        self.assertIn("Unsupported FD_LOG_LEVEL", str(ctx.exception))

    def test_debug_enabled_fallback(self):
        """FD_DEBUG=1 应该返回 DEBUG"""
        result = resolve_log_level(raw_level=None, debug_enabled=1)
        self.assertEqual(result, "DEBUG")

    def test_debug_disabled_fallback(self):
        """FD_DEBUG=0 应该返回 INFO"""
        result = resolve_log_level(raw_level=None, debug_enabled=0)
        self.assertEqual(result, "INFO")

    def test_env_fd_log_level_priority(self):
        """FD_LOG_LEVEL 环境变量优先级高于 FD_DEBUG"""
        with patch.dict("os.environ", {"FD_LOG_LEVEL": "INFO", "FD_DEBUG": "1"}):
            result = resolve_log_level()
            self.assertEqual(result, "INFO")

    def test_env_fd_debug_fallback(self):
        """无 FD_LOG_LEVEL 时使用 FD_DEBUG"""
        with patch.dict("os.environ", {"FD_DEBUG": "1"}, clear=True):
            result = resolve_log_level()
            self.assertEqual(result, "DEBUG")


class TestResolveRequestLoggingDefaults(unittest.TestCase):
    """测试 resolve_request_logging_defaults 函数"""

    def test_default_values(self):
        """默认值测试"""
        with patch.dict("os.environ", {}, clear=True):
            result = resolve_request_logging_defaults()
            self.assertEqual(result["enabled"], 1)
            self.assertEqual(result["level"], 2)
            self.assertEqual(result["max_len"], 2048)

    def test_custom_values(self):
        """自定义值测试"""
        with patch.dict(
            "os.environ", {"FD_LOG_REQUESTS": "0", "FD_LOG_REQUESTS_LEVEL": "2", "FD_LOG_MAX_LEN": "1024"}
        ):
            result = resolve_request_logging_defaults()
            self.assertEqual(result["enabled"], 0)
            self.assertEqual(result["level"], 2)
            self.assertEqual(result["max_len"], 1024)


if __name__ == "__main__":
    unittest.main()
