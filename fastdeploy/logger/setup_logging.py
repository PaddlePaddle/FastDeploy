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
Configure the FastDeploy logging system.
"""

import json
import logging
import logging.config
import os
from pathlib import Path

from fastdeploy import envs


def setup_logging(log_dir=None, config_file=None):
    """
    Configure FastDeploy logging.

    Args:
        log_dir: Directory for log files; falls back to environment variables if not provided.
        config_file: Path to a JSON config file; uses the default configuration when absent.
    """

    # Avoid configuring logging multiple times
    if getattr(setup_logging, "_configured", False):
        return logging.getLogger("fastdeploy")

    # Resolve the log directory from env vars, input argument, or default value
    if log_dir is None:
        log_dir = getattr(envs, "FD_LOG_DIR", "logs")

    # Ensure the log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Read logging level and backup count from environment variables
    is_debug = int(getattr(envs, "FD_DEBUG", 0))
    FASTDEPLOY_LOGGING_LEVEL = "DEBUG" if is_debug else "INFO"
    backup_count = int(getattr(envs, "FD_LOG_BACKUP_COUNT", 7))

    # Define the log output format
    _FORMAT = "%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s"

    # Default configuration
    default_config = {
        "version": 1,
        "disable_existing_loggers": False,
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
            "console": {
                "class": "logging.StreamHandler",
                "level": FASTDEPLOY_LOGGING_LEVEL,
                "formatter": "colored",
                "stream": "ext://sys.stdout",
            },
            # Default error log: keep the latest hour at log/error.log
            "error_file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": "ERROR",
                "formatter": "standard",
                "filename": os.path.join(log_dir, "error.log"),
                "when": "H",
                "interval": 1,
                "backupCount": 1,
            },
            # Full logs: keep the latest hour at log/default.log
            "default_file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": FASTDEPLOY_LOGGING_LEVEL,
                "formatter": "standard",
                "filename": os.path.join(log_dir, "default.log"),
                "when": "H",
                "interval": 1,
                "backupCount": 1,
            },
            # Error log archive: keep the last 7 days with hourly rotation, e.g., FastDeploy/log/2025-08-14/error_2025-08-14-18.log
            "error_archive": {
                "class": "fastdeploy.logger.handlers.IntervalRotatingFileHandler",
                "level": "ERROR",
                "formatter": "standard",
                "filename": os.path.join(log_dir, "error.log"),
                "backupDays": 7,
                "interval": 1,
                "encoding": "utf-8",
            },
            # Full log archive: keep the last 7 days with hourly rotation, e.g., FastDeploy/log/2025-08-14/default_2025-08-14-18.log
            "default_archive": {
                "class": "fastdeploy.logger.handlers.IntervalRotatingFileHandler",
                "level": FASTDEPLOY_LOGGING_LEVEL,
                "formatter": "standard",
                "filename": os.path.join(log_dir, "default.log"),
                "backupDays": 7,
                "interval": 1,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            # Default logger shared globally
            "fastdeploy": {
                "level": "DEBUG",
                "handlers": ["error_file", "default_file", "error_archive", "default_archive"],
                "propagate": False,
            }
        },
    }

    # Load configuration from file if provided
    if config_file and os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Merge environment-driven settings into user config; environment variables take precedence
        if "handlers" in config:
            for handler_name, handler_config in config["handlers"].items():
                if "backupCount" not in handler_config and "DailyRotating" in handler_config.get("class", ""):
                    handler_config["backupCount"] = backup_count
                if handler_config.get("level") == "INFO" and is_debug:
                    handler_config["level"] = "DEBUG"
    else:
        config = default_config

    # Apply logging configuration
    logging.config.dictConfig(config)

    # Prevent repeated configuration
    setup_logging._configured = True

    # Return the fastdeploy logger
    return logging.getLogger("fastdeploy")
