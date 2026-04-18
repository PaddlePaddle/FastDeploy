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
FastDeploy Logging Module

Unified logging management module providing:
- Formatters: ColoredFormatter, CustomFormatter
- Handlers: DailyRotatingFileHandler, IntervalRotatingFileHandler, LazyFileHandler
- Logger: FastDeployLogger
- Request Logger: log_request, log_request_error, RequestLogLevel
- Setup: setup_logging
- Pre-defined logger instances (lazy-loaded)

Usage:
    from fastdeploy.logger import get_logger, llm_logger, console_logger
    from fastdeploy.logger import log_request, log_request_error, RequestLogLevel
"""

# Formatters
from fastdeploy.logger.formatters import ColoredFormatter, CustomFormatter

# Handlers
from fastdeploy.logger.handlers import (
    DailyRotatingFileHandler,
    IntervalRotatingFileHandler,
    LazyFileHandler,
)

# Logger
from fastdeploy.logger.logger import FastDeployLogger

# Request logger
from fastdeploy.logger.request_logger import (
    RequestLogLevel,
    log_request,
    log_request_error,
)

# Setup
from fastdeploy.logger.setup_logging import setup_logging


def get_logger(name, file_name=None, without_formater=False, print_to_console=False, channel=None):
    """Convenience function to get a logger instance"""
    return FastDeployLogger().get_logger(name, file_name, without_formater, print_to_console, channel=channel)


# Pre-defined logger instance configs (name, file_name, without_formater, print_to_console, channel)
_LOGGER_CONFIGS = {
    "llm_logger": ("fastdeploy", None, False, False, "main"),
    "data_processor_logger": ("data_processor", None, False, False, "main"),
    "scheduler_logger": ("scheduler", None, False, False, "main"),
    "api_server_logger": ("api_server", None, False, False, "main"),
    "console_logger": (None, None, False, False, "console"),
    "spec_logger": ("speculate", "speculate.log", False, False, None),
    "zmq_client_logger": ("zmq_client", "comm.log", False, False, None),
    "router_logger": ("router", "comm.log", False, False, None),
    "fmq_logger": ("fmq", "comm.log", False, False, None),
    "obj_logger": ("obj", "obj.log", False, False, None),
    "register_manager_logger": ("register_manager", "register_manager.log", False, False, None),
    "_request_logger": ("request", None, False, False, "request"),
}

_logger_cache = {}


def __getattr__(name):
    """Lazy-load pre-defined logger instances"""
    if name in _LOGGER_CONFIGS:
        if name not in _logger_cache:
            cfg = _LOGGER_CONFIGS[name]
            _logger_cache[name] = get_logger(cfg[0], cfg[1], cfg[2], cfg[3], cfg[4])
        return _logger_cache[name]
    if name == "trace_logger":
        if name not in _logger_cache:
            _logger_cache[name] = FastDeployLogger().get_trace_logger("trace", "trace.log")
        return _logger_cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Formatters
    "ColoredFormatter",
    "CustomFormatter",
    # Handlers
    "DailyRotatingFileHandler",
    "IntervalRotatingFileHandler",
    "LazyFileHandler",
    # Logger
    "FastDeployLogger",
    "get_logger",
    # Request logger
    "RequestLogLevel",
    "log_request",
    "log_request_error",
    # Setup
    "setup_logging",
    # Pre-defined logger instances (lazy-loaded)
    "llm_logger",
    "data_processor_logger",
    "scheduler_logger",
    "api_server_logger",
    "console_logger",
    "spec_logger",
    "zmq_client_logger",
    "trace_logger",
    "router_logger",
    "fmq_logger",
    "obj_logger",
    "register_manager_logger",
]
