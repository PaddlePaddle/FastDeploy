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
Logging utilities for initializing and retrieving FastDeploy loggers.
Provides get_logger to centralize logging behavior across submodules.
"""

import logging
import os
import threading
from pathlib import Path

from fastdeploy import envs
from fastdeploy.logger.formatters import ColoredFormatter, CustomFormatter
from fastdeploy.logger.handlers import DailyRotatingFileHandler, LazyFileHandler
from fastdeploy.logger.setup_logging import setup_logging


class FastDeployLogger:
    _instance = None
    _initialized = False
    _lock = threading.RLock()

    def __new__(cls):
        """Implement the singleton pattern for the logger helper."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def _initialize(self):
        """Explicitly initialize the logging system once."""
        with self._lock:
            if not self._initialized:
                setup_logging()
                self._initialized = True

    def get_logger(self, name, file_name=None, without_formater=False, print_to_console=False):
        """
        Retrieve a logger (compatible with the legacy interface).

        Args:
            name: Logger name.
            file_name: Log file name (kept for compatibility).
            without_formater: Whether to skip applying a formatter.
            print_to_console: Whether to emit logs to stdout as well.
        """
        # Use the new unified naming scheme when only a name is provided
        if file_name is None and not without_formater and not print_to_console:
            # Lazy initialization
            if not self._initialized:
                self._initialize()
            return self._get_unified_logger(name)

        # Fall back to the legacy interface for compatibility
        return self._get_legacy_logger(name, file_name, without_formater, print_to_console)

    def _get_unified_logger(self, name):
        """
        Retrieve a logger using the unified naming approach.
        """
        if name is None:
            return logging.getLogger("fastdeploy")

        # Handle the __main__ special case
        if name == "__main__":
            import __main__

            # Extract the __file__ attribute of the main module
            if hasattr(__main__, "__file__"):
                # Use the main module filename
                base_name = Path(__main__.__file__).stem
                # Build a logger name with the FastDeploy prefix
                return logging.getLogger(f"fastdeploy.main.{base_name}")
            return logging.getLogger("fastdeploy.main")

        # If already in the fastdeploy namespace, return directly
        if name.startswith("fastdeploy.") or name == "fastdeploy":
            return logging.getLogger(name)
        else:
            # Otherwise, prepend the fastdeploy prefix
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
        # Use namespace for isolation to avoid logger overwrite and confusion issues for compatibility with original interface
        legacy_name = f"legacy.{name}"
        logger = logging.getLogger(legacy_name)

        # Set log level
        if is_debug:
            logger.setLevel(level=logging.DEBUG)
        else:
            logger.setLevel(level=logging.INFO)

        # Set formatter
        formatter = CustomFormatter(
            "[%(asctime)s] [%(levelname)-8s] (%(filename)s:%(funcName)s:%(lineno)d) %(message)s"
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
        Legacy logger retrieval interface kept for backward compatibility.
        """

        log_dir = envs.FD_LOG_DIR
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        is_debug = envs.FD_DEBUG
        # logger = logging.getLogger(name)
        # Use a namespace to avoid clobbering loggers and keep the legacy API intact
        legacy_name = f"legacy.{name}"
        logger = logging.getLogger(legacy_name)

        # Configure log level
        if is_debug:
            logger.setLevel(level=logging.DEBUG)
        else:
            logger.setLevel(level=logging.INFO)

        # Configure formatter
        formatter = ColoredFormatter(
            "%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s"
        )

        # Clear existing handlers (keep the original logic)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create the primary log file handler
        LOG_FILE = f"{log_dir}/{file_name}"
        backup_count = int(envs.FD_LOG_BACKUP_COUNT)
        # handler = LazyFileHandler(filename=LOG_FILE, backupCount=backup_count, level=hanlder_level)
        handler = DailyRotatingFileHandler(LOG_FILE, backupCount=backup_count)

        # Create an ERROR log file handler (additional feature)
        if not file_name.endswith(".log"):
            file_name = f"{file_name}.log" if "." not in file_name else file_name.split(".")[0] + ".log"
        ERROR_LOG_FILE = os.path.join(log_dir, file_name.replace(".log", "_error.log"))
        error_handler = LazyFileHandler(
            filename=ERROR_LOG_FILE, backupCount=backup_count, level=logging.ERROR, formatter=None
        )

        if not without_formater:
            handler.setFormatter(formatter)
            error_handler.setFormatter(formatter)

        # Attach file handlers
        logger.addHandler(handler)
        logger.addHandler(error_handler)

        # Console handler
        if print_to_console:
            console_handler = logging.StreamHandler()
            if not without_formater:
                console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            console_handler.propagate = False

        # Configure propagate (maintain original logic)
        # logger.propagate = False

        return logger
