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
Logging module: Used for initializing and getting FastDeploy loggers.
This module provides the get_logger method to uniformly manage logging behavior for each submodule.
"""

import logging
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from fastdeploy import envs
from fastdeploy.logger.config import resolve_log_level
from fastdeploy.logger.formatters import ColoredFormatter, CustomFormatter
from fastdeploy.logger.handlers import DailyRotatingFileHandler, LazyFileHandler
from fastdeploy.logger.setup_logging import setup_logging

# Standard log format
_LOG_FORMAT = "%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s"


class FastDeployLogger:
    _instance = None
    _initialized = False
    _lock = threading.RLock()

    # Channel to file mapping
    _channel_files = {
        "main": "fastdeploy.log",
        "request": "request.log",
        "console": "console.log",
    }

    # Cache for channel loggers that have been configured
    _configured_channels = set()

    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def _initialize(self):
        """Explicitly initialize the logging system"""
        with self._lock:
            if not self._initialized:
                setup_logging()
                self._initialized = True

    def get_logger(self, name, file_name=None, without_formater=False, print_to_console=False, channel=None):
        """
        Get logger (compatible with the original interface)

        Args:
            name: Logger name
            file_name: Log file name (for compatibility)
            without_formater: Whether to not use a formatter
            print_to_console: Whether to print to console
            channel: Log channel (main, request, console)
        """
        # If channel is specified, use channel-based logging
        if channel is not None:
            if not self._initialized:
                self._initialize()
            return self._get_channel_logger(name, channel)

        # If only one parameter is provided, use the new unified naming convention
        if file_name is None and not without_formater and not print_to_console:
            # Lazy initialization
            if not self._initialized:
                self._initialize()
            return self._get_unified_logger(name)

        # Compatible with the original interface
        return self._get_legacy_logger(name, file_name, without_formater, print_to_console)

    def _get_unified_logger(self, name):
        """
        New unified way to get logger
        """
        return self._get_channel_logger(name, "main")

    def _get_channel_logger(self, name, channel):
        """
        Get logger through channel with manual handler setup.

        Uses manual addHandler instead of dictConfig for better performance.
        Handlers are attached to the channel root logger (fastdeploy.{channel}),
        and child loggers propagate to it.

        Args:
            name: logger name
            channel: log channel (main, request, console)
        """
        # Get or create the channel root logger (all handlers go here)
        channel_root_name = f"fastdeploy.{channel}"
        channel_logger = logging.getLogger(channel_root_name)

        # Configure the channel root logger once
        if channel not in self._configured_channels:
            self._configured_channels.add(channel)

            log_dir = envs.FD_LOG_DIR
            os.makedirs(log_dir, exist_ok=True)

            # Resolve log level (priority: FD_LOG_LEVEL > FD_DEBUG)
            log_level = resolve_log_level()
            channel_logger.setLevel(logging.DEBUG if log_level == "DEBUG" else logging.INFO)

            # Create formatters
            file_formatter = logging.Formatter(_LOG_FORMAT)
            console_formatter = ColoredFormatter(_LOG_FORMAT)

            # Clear existing handlers
            for handler in channel_logger.handlers[:]:
                channel_logger.removeHandler(handler)

            # Create file handler for this channel
            file_name = self._channel_files.get(channel, f"{channel}.log")
            log_file = os.path.join(log_dir, file_name)
            backup_count = int(envs.FD_LOG_BACKUP_COUNT)

            file_handler = LazyFileHandler(log_file, backupCount=backup_count)
            file_handler.setFormatter(file_formatter)
            channel_logger.addHandler(file_handler)

            # Error file handler (all channels write errors to error.log)
            error_log_file = os.path.join(log_dir, "error.log")
            error_file_handler = LazyFileHandler(
                filename=error_log_file, backupCount=backup_count, level=logging.ERROR
            )
            error_file_handler.setFormatter(file_formatter)
            channel_logger.addHandler(error_file_handler)

            # Stderr handler for ERROR level (all channels output errors to stderr)
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setLevel(logging.ERROR)
            stderr_handler.setFormatter(console_formatter)
            channel_logger.addHandler(stderr_handler)

            # Console stdout handler for console channel only
            if channel == "console":
                stdout_handler = logging.StreamHandler(sys.stdout)
                stdout_handler.setLevel(logging.DEBUG if log_level == "DEBUG" else logging.INFO)
                stdout_handler.setFormatter(console_formatter)
                # Filter to exclude ERROR and above (they go to stderr)
                stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
                channel_logger.addHandler(stdout_handler)

            channel_logger.propagate = False

        # Determine the actual logger name and return the appropriate logger
        if name is None or name == "fastdeploy":
            return channel_logger
        elif name == "__main__":
            import __main__

            if hasattr(__main__, "__file__"):
                base_name = Path(__main__.__file__).stem
                logger_name = f"{channel_root_name}.{base_name}"
            else:
                return channel_logger
        elif name.startswith("fastdeploy."):
            logger_name = name
        else:
            logger_name = f"{channel_root_name}.{name}"

        # Child loggers propagate to channel_logger (which has handlers)
        return logging.getLogger(logger_name)

    def get_trace_logger(self, name, file_name, without_formater=False, print_to_console=False):
        """
        Log retrieval method compatible with the original interface
        """

        log_dir = envs.FD_LOG_DIR
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        is_debug = int(envs.FD_DEBUG)
        # logger = logging.getLogger(name)
        # Use namespace for isolation to avoid logger overwrite and confusion issues, for compatibility with original interface
        legacy_name = f"legacy.{name}"
        logger = logging.getLogger(legacy_name)

        # Set log level
        if is_debug:
            logger.setLevel(level=logging.DEBUG)
        else:
            logger.setLevel(level=logging.INFO)

        # Set formatter
        formatter = CustomFormatter(
            "%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s"
        )

        # Clear existing handlers (maintain original logic)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create main log file handler
        LOG_FILE = f"{log_dir}/{file_name}"
        backup_count = int(envs.FD_LOG_BACKUP_COUNT)
        # handler = LazyFileHandler(filename=LOG_FILE, backupCount=backup_count, level=hanlder_level)
        handler = DailyRotatingFileHandler(LOG_FILE, backupCount=backup_count)

        # Create ERROR log file handler (new feature)
        if not file_name.endswith(".log"):
            file_name = f"{file_name}.log" if "." not in file_name else file_name.split(".")[0] + ".log"
        ERROR_LOG_FILE = os.path.join(log_dir, file_name.replace(".log", "_error.log"))
        error_handler = LazyFileHandler(
            filename=ERROR_LOG_FILE, backupCount=backup_count, level=logging.ERROR, formatter=None
        )

        if not without_formater:
            handler.setFormatter(formatter)
            error_handler.setFormatter(formatter)

        # Add file handlers
        logger.addHandler(handler)
        logger.addHandler(error_handler)

        # Console handler
        if print_to_console:
            console_handler = logging.StreamHandler()
            if not without_formater:
                console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            console_handler.propagate = False

        # Set propagate (maintain original logic)
        # logger.propagate = False

        return logger

    def _get_legacy_logger(self, name, file_name, without_formater=False, print_to_console=False):
        """
        Legacy-compatible way to get logger
        """

        log_dir = envs.FD_LOG_DIR
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        is_debug = envs.FD_DEBUG
        # logger = logging.getLogger(name)
        # Use namespace for isolation to avoid logger overwrite and confusion issues, for compatibility with original interface
        legacy_name = f"legacy.{name}"
        logger = logging.getLogger(legacy_name)

        # Set log level
        if is_debug:
            logger.setLevel(level=logging.DEBUG)
        else:
            logger.setLevel(level=logging.INFO)

        # Set formatter - use standard format for both file and console (no color)
        formatter = logging.Formatter(
            "%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s"
        )

        # Clear existing handlers (maintain original logic)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create main log file handler
        LOG_FILE = f"{log_dir}/{file_name}"
        backup_count = int(envs.FD_LOG_BACKUP_COUNT)
        # handler = LazyFileHandler(filename=LOG_FILE, backupCount=backup_count, level=hanlder_level)
        handler = LazyFileHandler(LOG_FILE, backupCount=backup_count)

        # Create ERROR log file handler (new feature)
        if not file_name.endswith(".log"):
            file_name = f"{file_name}.log" if "." not in file_name else file_name.split(".")[0] + ".log"
        ERROR_LOG_FILE = os.path.join(log_dir, file_name.replace(".log", "_error.log"))
        error_handler = LazyFileHandler(
            filename=ERROR_LOG_FILE, backupCount=backup_count, level=logging.ERROR, formatter=None
        )

        if not without_formater:
            handler.setFormatter(formatter)
            error_handler.setFormatter(formatter)

        # Add file handlers
        logger.addHandler(handler)
        logger.addHandler(error_handler)

        # Console handler
        if print_to_console:
            console_handler = logging.StreamHandler()
            if not without_formater:
                console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            console_handler.propagate = False

        # Set propagate (maintain original logic)
        # logger.propagate = False

        return logger


@contextmanager
def intercept_paddle_loggers():
    """Intercept and configure paddle loggers during import."""
    _original = logging.getLogger

    def _patched(name=None):
        logger = _original(name)
        if name and str(name).startswith("paddle"):
            formatter = logging.Formatter(
                "%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s"
            )
            logger.setLevel(logging.DEBUG if envs.FD_DEBUG else logging.INFO)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
            logger.propagate = False
        return logger

    logging.getLogger = _patched
    try:
        yield
    finally:
        logging.getLogger = _original
