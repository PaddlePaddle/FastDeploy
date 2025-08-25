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

from fastdeploy.logger.formatters import ColoredFormatter


class TestColoredFormatter(unittest.TestCase):
    """测试 ColoredFormatter 类"""

    def setUp(self):
        """测试前准备"""
        self.formatter = ColoredFormatter("%(levelname)s - %(message)s")

    def test_color_codes_definition(self):
        """测试颜色代码定义"""
        expected_colors = {
            logging.WARNING: 33,  # 黄色
            logging.ERROR: 31,  # 红色
            logging.CRITICAL: 31,  # 红色
        }
        self.assertEqual(self.formatter.COLOR_CODES, expected_colors)

    def test_format_warning_message(self):
        """测试 WARNING 级别日志格式化（黄色）"""
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0, msg="This is a warning", args=(), exc_info=None
        )

        formatted_message = self.formatter.format(record)
        expected = "\033[33mWARNING - This is a warning\033[0m"
        self.assertEqual(formatted_message, expected)

    def test_format_error_message(self):
        """测试 ERROR 级别日志格式化（红色）"""
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0, msg="This is an error", args=(), exc_info=None
        )

        formatted_message = self.formatter.format(record)
        expected = "\033[31mERROR - This is an error\033[0m"
        self.assertEqual(formatted_message, expected)

    def test_format_critical_message(self):
        """测试 CRITICAL 级别日志格式化（红色）"""
        record = logging.LogRecord(
            name="test", level=logging.CRITICAL, pathname="", lineno=0, msg="This is critical", args=(), exc_info=None
        )

        formatted_message = self.formatter.format(record)
        expected = "\033[31mCRITICAL - This is critical\033[0m"
        self.assertEqual(formatted_message, expected)

    def test_format_info_message(self):
        """测试 INFO 级别日志格式化（无颜色）"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="This is info", args=(), exc_info=None
        )

        formatted_message = self.formatter.format(record)
        expected = "INFO - This is info"
        self.assertEqual(formatted_message, expected)

    def test_format_debug_message(self):
        """测试 DEBUG 级别日志格式化（无颜色）"""
        record = logging.LogRecord(
            name="test", level=logging.DEBUG, pathname="", lineno=0, msg="This is debug", args=(), exc_info=None
        )

        formatted_message = self.formatter.format(record)
        expected = "DEBUG - This is debug"
        self.assertEqual(formatted_message, expected)

    def test_format_custom_level(self):
        """测试自定义级别日志格式化（无颜色）"""
        # 创建自定义级别
        custom_level = 25  # 介于 INFO(20) 和 WARNING(30) 之间
        record = logging.LogRecord(
            name="test", level=custom_level, pathname="", lineno=0, msg="This is custom level", args=(), exc_info=None
        )
        record.levelname = "CUSTOM"

        formatted_message = self.formatter.format(record)
        expected = "CUSTOM - This is custom level"
        self.assertEqual(formatted_message, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
