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

from fastdeploy.logger.formatters import ColoredFormatter, CustomFormatter


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

    def test_format_with_otel_span_id(self):
        """测试带otelSpanID的日志格式化"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="This has span", args=(), exc_info=None
        )
        record.otelSpanID = "span123"

        formatted_message = self.formatter.format(record)
        expected = "INFO - [otel_span_id=span123] This has span"
        self.assertEqual(formatted_message, expected)

    def test_format_with_otel_trace_id(self):
        """测试带otelTraceID的日志格式化"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="This has trace", args=(), exc_info=None
        )
        record.otelTraceID = "trace456"

        formatted_message = self.formatter.format(record)
        expected = "INFO - [otel_trace_id=trace456] This has trace"
        self.assertEqual(formatted_message, expected)


class TestCustomFormatter(unittest.TestCase):
    """测试 CustomFormatter 类"""

    def setUp(self):
        """测试前准备"""
        self.formatter = CustomFormatter("%(levelname)s - %(message)s")

    def test_format_with_attributes(self):
        """测试带attributes的日志格式化"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="This has attrs", args=(), exc_info=None
        )
        record.attributes = {"key1": "value1", "key2": "value2"}

        formatted_message = self.formatter.format(record)
        self.assertIn("[key1=value1]", formatted_message)
        self.assertIn("[key2=value2]", formatted_message)
        self.assertIn("This has attrs", formatted_message)

    def test_format_with_camel_case_attributes(self):
        """测试带驼峰命名attributes的转换"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="This has camelCase", args=(), exc_info=None
        )
        record.attributes = {"camelCaseKey": "value"}

        formatted_message = self.formatter.format(record)
        self.assertIn("[camel_case_key=value]", formatted_message)

    def test_format_with_empty_attributes(self):
        """测试空attributes的处理"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Empty attrs", args=(), exc_info=None
        )
        record.attributes = {}

        formatted_message = self.formatter.format(record)
        # 检查是否包含线程信息和时间戳
        self.assertIn("[thread=", formatted_message)
        self.assertIn("[thread_name=", formatted_message)
        self.assertIn("[timestamp=", formatted_message)
        self.assertTrue(formatted_message.endswith("Empty attrs"))

    def test_format_with_thread_info(self):
        """测试线程信息的添加"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Thread test", args=(), exc_info=None
        )
        record.thread = 123
        record.threadName = "TestThread"

        formatted_message = self.formatter.format(record)
        self.assertIn("[thread=123]", formatted_message)
        self.assertIn("[thread_name=TestThread]", formatted_message)
        self.assertIn("[timestamp=", formatted_message)  # 检查时间戳

    def test_format_attributes_method(self):
        """测试_format_attributes方法"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Test attributes", args=(), exc_info=None
        )
        record.attributes = {"key1": "value1", "key2": "value2"}

        # 直接调用_format_attributes方法
        formatted_attrs = self.formatter._format_attributes(record)
        self.assertEqual(formatted_attrs, "[key1=value1] [key2=value2]")

    def test_format_attributes_method_empty(self):
        """测试_format_attributes方法处理空attributes"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Test empty", args=(), exc_info=None
        )
        record.attributes = {}

        formatted_attrs = self.formatter._format_attributes(record)
        self.assertEqual(formatted_attrs, "")

    def test_format_attributes_method_none(self):
        """测试_format_attributes方法处理无attributes"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Test none", args=(), exc_info=None
        )

        formatted_attrs = self.formatter._format_attributes(record)
        self.assertEqual(formatted_attrs, "")

    def test_format_attributes_method_invalid_type(self):
        """测试_format_attributes方法处理非字典attributes"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Test invalid", args=(), exc_info=None
        )
        record.attributes = "invalid"

        formatted_attrs = self.formatter._format_attributes(record)
        self.assertEqual(formatted_attrs, "")

    def test_camel_to_snake_method(self):
        """测试_camel_to_snake方法"""
        # 测试驼峰转下划线
        self.assertEqual(self.formatter._camel_to_snake("camelCase"), "camel_case")
        self.assertEqual(self.formatter._camel_to_snake("CamelCase"), "camel_case")
        self.assertEqual(self.formatter._camel_to_snake("camelCaseKey"), "camel_case_key")
        self.assertEqual(self.formatter._camel_to_snake("already_snake"), "already_snake")

    def test_format_with_empty_string_attributes(self):
        """测试带空字符串attributes的处理"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Empty string attrs", args=(), exc_info=None
        )
        record.attributes = {"key1": "", "key2": "value2"}

        formatted_message = self.formatter.format(record)
        # 空字符串的key1应该被过滤掉
        self.assertNotIn("[key1=]", formatted_message)
        self.assertIn("[key2=value2]", formatted_message)

    def test_format_with_both_otel_and_attributes(self):
        """测试同时有otel字段和attributes的情况"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Both otel and attrs", args=(), exc_info=None
        )
        record.attributes = {"key1": "value1"}
        record.otelSpanID = "span123"
        record.otelTraceID = "trace456"

        formatted_message = self.formatter.format(record)
        self.assertIn("[key1=value1]", formatted_message)
        self.assertIn("[otel_span_id=span123]", formatted_message)
        self.assertIn("[otel_trace_id=trace456]", formatted_message)

    def test_format_exception_handling(self):
        """测试异常处理机制"""
        # 创建一个会引发异常的record
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Exception test", args=(), exc_info=None
        )
        # 添加一个会引发异常的属性
        record.thread = "invalid_thread"  # 这会引发异常，因为thread应该是整数

        # 即使有异常，format方法也应该正常返回
        formatted_message = self.formatter.format(record)
        self.assertIsInstance(formatted_message, str)
        self.assertIn("Exception test", formatted_message)

    def test_format_with_none_otel_fields(self):
        """测试None值的otel字段处理"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="None otel", args=(), exc_info=None
        )
        record.otelSpanID = None
        record.otelTraceID = None

        formatted_message = self.formatter.format(record)
        # None值的otel字段不应该被添加
        self.assertNotIn("otel_span_id", formatted_message)
        self.assertNotIn("otel_trace_id", formatted_message)


class TestColoredFormatterExceptionHandling(unittest.TestCase):
    """测试ColoredFormatter的异常处理"""

    def setUp(self):
        """测试前准备"""
        self.formatter = ColoredFormatter("%(levelname)s - %(message)s")

    def test_format_exception_handling(self):
        """测试ColoredFormatter的异常处理机制"""
        # 创建一个会引发异常的record
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Exception test", args=(), exc_info=None
        )
        # 添加一个会引发异常的属性
        record.otelSpanID = object()  # 非字符串类型，可能会引发异常

        # 即使有异常，format方法也应该正常返回
        formatted_message = self.formatter.format(record)
        self.assertIsInstance(formatted_message, str)
        self.assertIn("Exception test", formatted_message)

    def test_format_with_none_otel_fields(self):
        """测试None值的otel字段处理"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="None otel", args=(), exc_info=None
        )
        record.otelSpanID = None
        record.otelTraceID = None

        formatted_message = self.formatter.format(record)
        # None值的otel字段不应该被添加
        self.assertNotIn("otel_span_id", formatted_message)
        self.assertNotIn("otel_trace_id", formatted_message)

    def test_format_with_invalid_otel_fields(self):
        """测试无效的otel字段处理"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Invalid otel", args=(), exc_info=None
        )
        # 设置无效的属性，确保异常被捕获
        record.otelSpanID = 123  # 整数类型，不是字符串
        record.otelTraceID = 456  # 整数类型，不是字符串

        formatted_message = self.formatter.format(record)
        self.assertIsInstance(formatted_message, str)
        self.assertIn("Invalid otel", formatted_message)

    def test_colored_formatter_exception_handling_with_forced_error(self):
        """测试ColoredFormatter异常处理 - 强制抛出异常"""
        # 创建测试record并添加会引发异常的特殊属性
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Forced error test", args=(), exc_info=None
        )

        # 添加会引发AttributeError的属性
        class BadOtelSpanID:
            def __str__(self):
                raise AttributeError("Forced attribute error")

        record.otelSpanID = BadOtelSpanID()

        # 调用format方法，应该捕获异常并继续执行
        formatted_message = self.formatter.format(record)
        self.assertIsInstance(formatted_message, str)
        self.assertIn("Forced error test", formatted_message)

    def test_custom_colored_formatter_exception_handling_with_forced_error(self):
        """测试CustomFormatter异常处理 - 强制抛出异常"""
        custom_formatter = CustomFormatter("%(levelname)s - %(message)s")

        # 创建测试record并添加会引发异常的特殊属性
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Forced error test", args=(), exc_info=None
        )

        # 添加会引发TypeError的attributes
        class BadAttributes:
            def items(self):
                raise TypeError("Forced type error")

        record.attributes = BadAttributes()

        # 调用format方法，应该捕获异常并继续执行
        formatted_message = custom_formatter.format(record)
        self.assertIsInstance(formatted_message, str)
        self.assertIn("Forced error test", formatted_message)

    def test_colored_formatter_otel_processing_exception(self):
        """测试ColoredFormatter中otel处理异常"""
        # 创建测试record并添加会引发异常的特殊属性
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Otel exception test", args=(), exc_info=None
        )

        # 添加会引发Exception的otelSpanID
        class BadOtelSpanID:
            def __str__(self):
                raise Exception("Forced otel processing error")

        record.otelSpanID = BadOtelSpanID()

        # 调用format方法，应该捕获异常并继续执行
        formatted_message = self.formatter.format(record)
        self.assertIsInstance(formatted_message, str)
        self.assertIn("Otel exception test", formatted_message)

    def test_custom_colored_formatter_thread_processing_exception(self):
        """测试CustomFormatter中线程处理异常"""
        custom_formatter = CustomFormatter("%(levelname)s - %(message)s")

        # 创建测试record并添加会引发异常的特殊属性
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Thread exception test", args=(), exc_info=None
        )

        # 添加会引发Exception的thread属性
        class BadThread:
            def __int__(self):
                raise Exception("Forced thread processing error")

        record.thread = BadThread()

        # 添加会引发AttributeError的属性
        class BadOtelSpanID:
            def __str__(self):
                raise AttributeError("Forced attribute error")

        record.otelSpanID = BadOtelSpanID()

        # 调用format方法，应该捕获异常并继续执行
        formatted_message = custom_formatter.format(record)
        self.assertIsInstance(formatted_message, str)
        self.assertIn("Thread exception test", formatted_message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
