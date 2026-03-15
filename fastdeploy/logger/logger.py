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
import threading
from contextlib import contextmanager
from pathlib import Path

from fastdeploy import envs
from fastdeploy.logger.formatters import CustomFormatter
from fastdeploy.logger.handlers import DailyRotatingFileHandler, LazyFileHandler
from fastdeploy.logger.setup_logging import setup_logging


class FastDeployLogger:
    _instance = None
    _initialized = False
    _lock = threading.RLock()

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

    def get_logger(self, name, file_name=None, without_formater=False, print_to_console=False):
        """
        Get logger (compatible with the original interface)

        Args:
            name: Logger name
            file_name: Log file name (for compatibility)
            without_formater: Whether to not use a formatter
            print_to_console: Whether to print to console
        """
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
        if name is None:
            return logging.getLogger("fastdeploy")

        # Handle __main__ special case
        if name == "__main__":
            import __main__

            # Get the __file__ attribute of the main module
            if hasattr(__main__, "__file__"):
                # Get the main module file name
                base_name = Path(__main__.__file__).stem
                # Create logger with prefix
                return logging.getLogger(f"fastdeploy.main.{base_name}")
            return logging.getLogger("fastdeploy.main")

        # If already in fastdeploy namespace, use directly
        if name.startswith("fastdeploy.") or name == "fastdeploy":
            return logging.getLogger(name)
        else:
            # Add fastdeploy prefix for other cases
            return logging.getLogger(f"fastdeploy.{name}")

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
