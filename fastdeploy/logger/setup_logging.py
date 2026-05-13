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
Setup logging system

This module handles logging initialization:
- Basic setup: log directory creation
- Optional: load external JSON config file via dictConfig

Channel-based logger configuration is handled by FastDeployLogger._get_channel_logger()
using manual addHandler for better performance.

Log channels:
- main: Main logs -> fastdeploy.log
- request: Request logs -> request.log
- console: Console logs -> fastdeploy.log + terminal (stdout/stderr)
"""

import json
import logging
import logging.config
import os
from pathlib import Path

from fastdeploy import envs


class MaxLevelFilter(logging.Filter):
    """Filter log records below specified level.

    Used to route INFO/DEBUG to stdout, ERROR/CRITICAL to stderr.
    """

    def __init__(self, level):
        super().__init__()
        self.level = logging._nameToLevel.get(level, level) if isinstance(level, str) else level

    def filter(self, record):
        return record.levelno < self.level


def _build_default_config(log_dir, log_level, backup_count):
    """Build default logging configuration for dictConfig"""
    _FORMAT = "%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s"

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "below_error": {
                "()": MaxLevelFilter,
                "level": "ERROR",
            }
        },
        "formatters": {
            "standard": {
                "class": "logging.Formatter",
                "format": _FORMAT,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "colored": {
                "class": "fastdeploy.logger.formatters.ColoredFormatter",
                "format": _FORMAT,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            # Console stdout for INFO/DEBUG (below ERROR level)
            "console_stdout": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "filters": ["below_error"],
                "formatter": "colored",
                "stream": "ext://sys.stdout",
            },
            # Console stderr for ERROR/CRITICAL
            "console_stderr": {
                "class": "logging.StreamHandler",
                "level": "ERROR",
                "formatter": "colored",
                "stream": "ext://sys.stderr",
            },
            # Main log file
            "main_file": {
                "class": "fastdeploy.logger.handlers.LazyFileHandler",
                "level": log_level,
                "formatter": "standard",
                "filename": os.path.join(log_dir, "fastdeploy.log"),
                "backupCount": backup_count,
            },
            # Request log file
            "request_file": {
                "class": "fastdeploy.logger.handlers.LazyFileHandler",
                "level": log_level,
                "formatter": "standard",
                "filename": os.path.join(log_dir, "request.log"),
                "backupCount": backup_count,
            },
            # Error log file
            "error_file": {
                "class": "fastdeploy.logger.handlers.LazyFileHandler",
                "level": "ERROR",
                "formatter": "standard",
                "filename": os.path.join(log_dir, "error.log"),
                "backupCount": backup_count,
            },
        },
        "loggers": {
            # Default logger
            "fastdeploy": {
                "level": "DEBUG",
                "handlers": ["main_file", "error_file", "console_stderr"],
                "propagate": False,
            },
            # Main channel
            "fastdeploy.main": {
                "level": "DEBUG",
                "handlers": ["main_file", "error_file", "console_stderr"],
                "propagate": False,
            },
            # Request channel - only output to request.log and error.log
            "fastdeploy.request": {
                "level": "DEBUG",
                "handlers": ["request_file", "error_file", "console_stderr"],
                "propagate": False,
            },
            # Console channel - terminal output + merged into fastdeploy.log
            "fastdeploy.console": {
                "level": "DEBUG",
                "handlers": ["main_file", "console_stdout", "error_file", "console_stderr"],
                "propagate": False,
            },
        },
    }


def setup_logging(log_dir=None, config_file=None):
    """
    Setup FastDeploy logging configuration.

    This function:
    1. Ensures the log directory exists
    2. Optionally loads external JSON config file via dictConfig

    Note: Channel-based loggers (get_logger with channel param) use manual addHandler
    for better performance, independent of dictConfig.

    Args:
        log_dir: Log file storage directory, uses environment variable if not provided
        config_file: Optional JSON config file path for dictConfig
    """
    # Avoid duplicate configuration
    if getattr(setup_logging, "_configured", False):
        return

    # Use log directory from environment variable, or use provided parameter or default value
    if log_dir is None:
        log_dir = getattr(envs, "FD_LOG_DIR", "log")

    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Store log_dir for later use
    setup_logging._log_dir = log_dir

    # If config_file is provided, use dictConfig to load it
    if config_file is not None:
        is_debug = int(getattr(envs, "FD_DEBUG", 0))
        log_level = "DEBUG" if is_debug else "INFO"
        backup_count = int(getattr(envs, "FD_LOG_BACKUP_COUNT", 7))

        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Merge environment variable config into user config
            if "handlers" in config:
                for handler_config in config["handlers"].values():
                    if "backupCount" not in handler_config and "DailyRotating" in handler_config.get("class", ""):
                        handler_config["backupCount"] = backup_count
                    if handler_config.get("level") == "INFO" and log_level == "DEBUG":
                        handler_config["level"] = "DEBUG"
        else:
            # Config file not found, use default config
            config = _build_default_config(log_dir, log_level, backup_count)

        # Apply logging configuration via dictConfig
        logging.config.dictConfig(config)

    # Mark as configured
    setup_logging._configured = True
