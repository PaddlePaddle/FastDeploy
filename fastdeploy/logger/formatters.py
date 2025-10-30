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
自定义日志格式化器模块
该模块定义了 ColoredFormatter 类，用于在控制台输出带颜色的日志信息，
便于开发者在终端中快速识别不同级别的日志。
"""

import logging
import re
import time


class ColoredFormatter(logging.Formatter):
    """
    自定义日志格式器，用于控制台输出带颜色的日志。
    支持的颜色：
        - WARNING: 黄色
        - ERROR: 红色
        - CRITICAL: 红色
        - 其他等级: 默认终端颜色
    """

    COLOR_CODES = {
        logging.WARNING: 33,  # 黄色
        logging.ERROR: 31,  # 红色
        logging.CRITICAL: 31,  # 红色
    }

    def format(self, record):
        """
        格式化日志记录，并根据日志等级添加 ANSI 颜色前缀和后缀。
        新增支持attributes展开和otelSpanID/otelTraceID字段。
        Args:
            record (LogRecord): 日志记录对象。
        Returns:
            str: 带有颜色的日志消息字符串。
        """

        try:
            # 添加otel相关字段
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
    自定义日志格式器，用于控制台输出日志。
    支持字段展开，并添加线程、时间戳等信息。
    """

    def _format_attributes(self, record):
        """
        将record中的attributes展开为[attr=value]格式
        """
        if hasattr(record, "attributes"):
            if isinstance(record.attributes, dict):
                return " ".join(f"[{k}={v}]" for k, v in record.attributes.items())
        return ""

    def _camel_to_snake(self, name: str) -> str:
        """驼峰转下划线"""
        s1 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
        return s1.lower()

    def format(self, record):
        """
        格式化日志记录，新增支持attributes展开和otelSpanID/otelTraceID字段。
        支持字段展开，并添加线程、时间戳等信息。
        Args:
            record (LogRecord): 日志记录对象。
        Returns:
            str: 日志消息字符串。
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

            # 过滤空值
            log_fields = {k: v for k, v in log_fields.items() if not (isinstance(v, str) and v == "")}

            log_str = " ".join(f"[{k}={v}]" for k, v in log_fields.items())
            if log_str:
                record.msg = f"{log_str} {record.msg}"

            # 添加otel相关字段
            if hasattr(record, "otelSpanID") and record.otelSpanID is not None:
                record.msg = f"[otel_span_id={record.otelSpanID}] {record.msg}"
            if hasattr(record, "otelTraceID") and record.otelTraceID is not None:
                record.msg = f"[otel_trace_id={record.otelTraceID}] {record.msg}"

        except:
            pass

        return super().format(record)
