"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

"""
Custom log formatter module
This module defines the ColoredFormatter class for outputting colored log information to the console,
helping developers quickly identify different levels of logs in the terminal.
"""

import logging
import re
import time


class ColoredFormatter(logging.Formatter):
    """
    Custom log formatter for console output with colored logs.
    Supported colors:
        - WARNING: Yellow
        - ERROR: Red
        - CRITICAL: Red
        - Other levels: Default terminal color
    """

    COLOR_CODES = {
        logging.WARNING: 33,  # Yellow
        logging.ERROR: 31,  # Red
        logging.CRITICAL: 31,  # Red
    }

    def format(self, record):
        """
        Format log record and add ANSI color prefix and suffix based on log level.
        Newly supports attributes expansion and otelSpanID/otelTraceID fields.
        Args:
            record (LogRecord): Log record object.
        Returns:
            str: Colored log message string.
        """

        try:
            # Add OpenTelemetry-related fields.
            if hasattr(record, "otelSpanID") and record.otelSpanID is not None:
                record.msg = f"[otel_span_id={record.otelSpanID}] {record.msg}"
            if hasattr(record, "otelTraceID") and record.otelTraceID is not None:
                record.msg = f"[otel_trace_id={record.otelTraceID}] {record.msg}"
        except:
            pass

        color_code = self.COLOR_CODES.get(record.levelno, 0)
        prefix = f"\033[{color_code}m"
        suffix = "\033[0m"
        message = super().format(record)
        if color_code:
            message = f"{prefix}{message}{suffix}"
        return message


class CustomFormatter(logging.Formatter):
    """
    Custom log formatter for console output.
    Supports field expansion and adds thread, timestamp and other information.
    """

    def _format_attributes(self, record):
        """
        Expand attributes in record to [attr=value] format
        """
        if hasattr(record, "attributes"):
            if isinstance(record.attributes, dict):
                return " ".join(f"[{k}={v}]" for k, v in record.attributes.items())
        return ""

    def _camel_to_snake(self, name: str) -> str:
        """Convert camel case to snake case"""
        s1 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
        return s1.lower()

    def format(self, record):
        """
        Format log record, with new support for attributes expansion and otelSpanID/otelTraceID fields.
        Supports field expansion and adds thread, timestamp and other information.
        Args:
            record (LogRecord): Log record object.
        Returns:
            str: Log message string.
        """

        try:
            log_fields = {
                "thread": record.thread,
                "thread_name": record.threadName,
                "timestamp": int(time.time() * 1000),
            }

            if hasattr(record, "attributes") and isinstance(record.attributes, dict):
                for k, v in record.attributes.items():
                    log_fields[self._camel_to_snake(k)] = v

            # filter out null values.
            log_fields = {k: v for k, v in log_fields.items() if not (isinstance(v, str) and v == "")}

            log_str = " ".join(f"[{k}={v}]" for k, v in log_fields.items())
            if log_str:
                record.msg = f"{log_str} {record.msg}"

            # Add OpenTelemetry-related fields.
            if hasattr(record, "otelSpanID") and record.otelSpanID is not None:
                record.msg = f"[otel_span_id={record.otelSpanID}] {record.msg}"
            if hasattr(record, "otelTraceID") and record.otelTraceID is not None:
                record.msg = f"[otel_trace_id={record.otelTraceID}] {record.msg}"

        except:
            pass

        return super().format(record)
