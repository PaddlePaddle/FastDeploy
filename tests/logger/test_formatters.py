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
    """Test ColoredFormatter class"""

    def setUp(self):
        """Test preparation"""
        self.formatter = ColoredFormatter("%(levelname)s - %(message)s")

    def test_color_codes_definition(self):
        """Test color code definition"""
        expected_colors = {
            logging.WARNING: 33,  # yellow
            logging.ERROR: 31,  # red
            logging.CRITICAL: 31,  # red
        }
        self.assertEqual(self.formatter.COLOR_CODES, expected_colors)

    def test_format_warning_message(self):
        """Test WARNING level log formatting (yellow)"""
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0, msg="This is a warning", args=(), exc_info=None
        )

        formatted_message = self.formatter.format(record)
        expected = "\033[33mWARNING - This is a warning\033[0m"
        self.assertEqual(formatted_message, expected)

    def test_format_error_message(self):
        """Test ERROR level log formatting (red)"""
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0, msg="This is an error", args=(), exc_info=None
        )

        formatted_message = self.formatter.format(record)
        expected = "\033[31mERROR - This is an error\033[0m"
        self.assertEqual(formatted_message, expected)

    def test_format_critical_message(self):
        """Test CRITICAL level log formatting (red)"""
        record = logging.LogRecord(
            name="test", level=logging.CRITICAL, pathname="", lineno=0, msg="This is critical", args=(), exc_info=None
        )

        formatted_message = self.formatter.format(record)
        expected = "\033[31mCRITICAL - This is critical\033[0m"
        self.assertEqual(formatted_message, expected)

    def test_format_info_message(self):
        """Test INFO level log formatting (no color)"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="This is info", args=(), exc_info=None
        )

        formatted_message = self.formatter.format(record)
        expected = "INFO - This is info"
        self.assertEqual(formatted_message, expected)

    def test_format_debug_message(self):
        """Test DEBUG level log formatting (no color)"""
        record = logging.LogRecord(
            name="test", level=logging.DEBUG, pathname="", lineno=0, msg="This is debug", args=(), exc_info=None
        )

        formatted_message = self.formatter.format(record)
        expected = "DEBUG - This is debug"
        self.assertEqual(formatted_message, expected)

    def test_format_custom_level(self):
        """Test custom level log formatting (no color)"""
        # Create custom level
        custom_level = 25  # Between INFO(20) and WARNING(30)
        record = logging.LogRecord(
            name="test", level=custom_level, pathname="", lineno=0, msg="This is custom level", args=(), exc_info=None
        )
        record.levelname = "CUSTOM"

        formatted_message = self.formatter.format(record)
        expected = "CUSTOM - This is custom level"
        self.assertEqual(formatted_message, expected)

    def test_format_with_otel_span_id(self):
        """Test log formatting with otelSpanID"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="This has span", args=(), exc_info=None
        )
        record.otelSpanID = "span123"

        formatted_message = self.formatter.format(record)
        expected = "INFO - [otel_span_id=span123] This has span"
        self.assertEqual(formatted_message, expected)

    def test_format_with_otel_trace_id(self):
        """Test log formatting with otelTraceID"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="This has trace", args=(), exc_info=None
        )
        record.otelTraceID = "trace456"

        formatted_message = self.formatter.format(record)
        expected = "INFO - [otel_trace_id=trace456] This has trace"
        self.assertEqual(formatted_message, expected)


class TestCustomFormatter(unittest.TestCase):
    """Test CustomFormatter class"""

    def setUp(self):
        """Test preparation"""
        self.formatter = CustomFormatter("%(levelname)s - %(message)s")

    def test_format_with_attributes(self):
        """Test log formatting with attributes"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="This has attrs", args=(), exc_info=None
        )
        record.attributes = {"key1": "value1", "key2": "value2"}

        formatted_message = self.formatter.format(record)
        self.assertIn("[key1=value1]", formatted_message)
        self.assertIn("[key2=value2]", formatted_message)
        self.assertIn("This has attrs", formatted_message)

    def test_format_with_camel_case_attributes(self):
        """Test conversion of camelCase attributes"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="This has camelCase", args=(), exc_info=None
        )
        record.attributes = {"camelCaseKey": "value"}

        formatted_message = self.formatter.format(record)
        self.assertIn("[camel_case_key=value]", formatted_message)

    def test_format_with_empty_attributes(self):
        """Test handling of empty attributes"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Empty attrs", args=(), exc_info=None
        )
        record.attributes = {}

        formatted_message = self.formatter.format(record)
        # Check if thread info and timestamp are included
        self.assertIn("[thread=", formatted_message)
        self.assertIn("[thread_name=", formatted_message)
        self.assertIn("[timestamp=", formatted_message)
        self.assertTrue(formatted_message.endswith("Empty attrs"))

    def test_format_with_thread_info(self):
        """Test addition of thread information"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Thread test", args=(), exc_info=None
        )
        record.thread = 123
        record.threadName = "TestThread"

        formatted_message = self.formatter.format(record)
        self.assertIn("[thread=123]", formatted_message)
        self.assertIn("[thread_name=TestThread]", formatted_message)
        self.assertIn("[timestamp=", formatted_message)  # Check timestamp

    def test_format_attributes_method(self):
        """Test _format_attributes method"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Test attributes", args=(), exc_info=None
        )
        record.attributes = {"key1": "value1", "key2": "value2"}

        # Directly call _format_attributes method
        formatted_attrs = self.formatter._format_attributes(record)
        self.assertEqual(formatted_attrs, "[key1=value1] [key2=value2]")

    def test_format_attributes_method_empty(self):
        """Test _format_attributes method handling empty attributes"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Test empty", args=(), exc_info=None
        )
        record.attributes = {}

        formatted_attrs = self.formatter._format_attributes(record)
        self.assertEqual(formatted_attrs, "")

    def test_format_attributes_method_none(self):
        """Test _format_attributes method handling no attributes"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Test none", args=(), exc_info=None
        )

        formatted_attrs = self.formatter._format_attributes(record)
        self.assertEqual(formatted_attrs, "")

    def test_format_attributes_method_invalid_type(self):
        """Test _format_attributes method handling non-dict attributes"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Test invalid", args=(), exc_info=None
        )
        record.attributes = "invalid"

        formatted_attrs = self.formatter._format_attributes(record)
        self.assertEqual(formatted_attrs, "")

    def test_camel_to_snake_method(self):
        """Test _camel_to_snake method"""
        # Test camelCase to snake_case conversion
        self.assertEqual(self.formatter._camel_to_snake("camelCase"), "camel_case")
        self.assertEqual(self.formatter._camel_to_snake("CamelCase"), "camel_case")
        self.assertEqual(self.formatter._camel_to_snake("camelCaseKey"), "camel_case_key")
        self.assertEqual(self.formatter._camel_to_snake("already_snake"), "already_snake")

    def test_format_with_empty_string_attributes(self):
        """Test handling of empty string attributes"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Empty string attrs", args=(), exc_info=None
        )
        record.attributes = {"key1": "", "key2": "value2"}

        formatted_message = self.formatter.format(record)
        # Empty string key1 should be filtered out
        self.assertNotIn("[key1=]", formatted_message)
        self.assertIn("[key2=value2]", formatted_message)

    def test_format_with_both_otel_and_attributes(self):
        """Test case with both otel fields and attributes"""
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
        """Test exception handling mechanism"""
        # Create a record that will cause an exception
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Exception test", args=(), exc_info=None
        )
        # Add an attribute that will cause an exception
        record.thread = "invalid_thread"  # This will cause an exception because thread should be an integer

        # Even with exceptions, the format method should return normally
        formatted_message = self.formatter.format(record)
        self.assertIsInstance(formatted_message, str)
        self.assertIn("Exception test", formatted_message)

    def test_format_with_none_otel_fields(self):
        """Test handling of None value otel fields"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="None otel", args=(), exc_info=None
        )
        record.otelSpanID = None
        record.otelTraceID = None

        formatted_message = self.formatter.format(record)
        # None value otel fields should not be added
        self.assertNotIn("otel_span_id", formatted_message)
        self.assertNotIn("otel_trace_id", formatted_message)


class TestColoredFormatterExceptionHandling(unittest.TestCase):
    """Test ColoredFormatter exception handling"""

    def setUp(self):
        """Test preparation"""
        self.formatter = ColoredFormatter("%(levelname)s - %(message)s")

    def test_format_exception_handling(self):
        """Test ColoredFormatter exception handling mechanism"""
        # Create a record that will cause an exception
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Exception test", args=(), exc_info=None
        )
        # Add an attribute that will cause an exception
        record.otelSpanID = object()  # Non-string type, may cause an exception

        # Even with exceptions, the format method should return normally
        formatted_message = self.formatter.format(record)
        self.assertIsInstance(formatted_message, str)
        self.assertIn("Exception test", formatted_message)

    def test_format_with_none_otel_fields(self):
        """Test handling of None value otel fields"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="None otel", args=(), exc_info=None
        )
        record.otelSpanID = None
        record.otelTraceID = None

        formatted_message = self.formatter.format(record)
        # None value otel fields should not be added
        self.assertNotIn("otel_span_id", formatted_message)
        self.assertNotIn("otel_trace_id", formatted_message)

    def test_format_with_invalid_otel_fields(self):
        """Test handling of invalid otel fields"""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Invalid otel", args=(), exc_info=None
        )
        # Set invalid attributes to ensure exceptions are caught
        record.otelSpanID = 123  # Integer type, not string
        record.otelTraceID = 456  # Integer type, not string

        formatted_message = self.formatter.format(record)
        self.assertIsInstance(formatted_message, str)
        self.assertIn("Invalid otel", formatted_message)

    def test_colored_formatter_exception_handling_with_forced_error(self):
        """Test ColoredFormatter exception handling - forced exception"""
        # Create test record and add special attributes that will cause exceptions
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Forced error test", args=(), exc_info=None
        )

        # Add attribute that will cause AttributeError
        class BadOtelSpanID:
            def __str__(self):
                raise AttributeError("Forced attribute error")

        record.otelSpanID = BadOtelSpanID()

        # Call format method, should catch exception and continue execution
        formatted_message = self.formatter.format(record)
        self.assertIsInstance(formatted_message, str)
        self.assertIn("Forced error test", formatted_message)

    def test_custom_colored_formatter_exception_handling_with_forced_error(self):
        """Test CustomFormatter exception handling - forced exception"""
        custom_formatter = CustomFormatter("%(levelname)s - %(message)s")

        # Create test record and add special attributes that will cause exceptions
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Forced error test", args=(), exc_info=None
        )

        # Add attributes that will cause TypeError
        class BadAttributes:
            def items(self):
                raise TypeError("Forced type error")

        record.attributes = BadAttributes()

        # Call format method, should catch exception and continue execution
        formatted_message = custom_formatter.format(record)
        self.assertIsInstance(formatted_message, str)
        self.assertIn("Forced error test", formatted_message)

    def test_colored_formatter_otel_processing_exception(self):
        """Test otel processing exception in ColoredFormatter"""
        # Create test record and add special attributes that will cause exceptions
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Otel exception test", args=(), exc_info=None
        )

        # Add otelSpanID that will cause Exception
        class BadOtelSpanID:
            def __str__(self):
                raise Exception("Forced otel processing error")

        record.otelSpanID = BadOtelSpanID()

        # Call format method, should catch exception and continue execution
        formatted_message = self.formatter.format(record)
        self.assertIsInstance(formatted_message, str)
        self.assertIn("Otel exception test", formatted_message)

    def test_custom_colored_formatter_thread_processing_exception(self):
        """Test thread processing exception in CustomFormatter"""
        custom_formatter = CustomFormatter("%(levelname)s - %(message)s")

        # Create test record and add special attributes that will cause exceptions
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Thread exception test", args=(), exc_info=None
        )

        # Add thread attribute that will cause Exception
        class BadThread:
            def __int__(self):
                raise Exception("Forced thread processing error")

        record.thread = BadThread()

        # Add attribute that will cause AttributeError
        class BadOtelSpanID:
            def __str__(self):
                raise AttributeError("Forced attribute error")

        record.otelSpanID = BadOtelSpanID()

        # Call format method, should catch exception and continue execution
        formatted_message = custom_formatter.format(record)
        self.assertIsInstance(formatted_message, str)
        self.assertIn("Thread exception test", formatted_message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
