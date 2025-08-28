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
配置日志系统
"""

import json
import logging
import logging.config
import os
from pathlib import Path

from fastdeploy import envs


def setup_logging(log_dir=None, config_file=None):
    """
    设置FastDeploy的日志配置

    Args:
        log_dir: 日志文件存储目录，如果不提供则使用环境变量
        config_file: JSON配置文件路径，如果不提供则使用默认配置
    """

    # 避免重复配置
    if getattr(setup_logging, "_configured", False):
        return logging.getLogger("fastdeploy")

    # 使用环境变量中的日志目录，如果没有则使用传入的参数或默认值
    if log_dir is None:
        log_dir = getattr(envs, "FD_LOG_DIR", "logs")

    # 确保日志目录存在
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 从环境变量获取日志级别和备份数量
    is_debug = int(getattr(envs, "FD_DEBUG", 0))
    FASTDEPLOY_LOGGING_LEVEL = "DEBUG" if is_debug else "INFO"
    backup_count = int(getattr(envs, "FD_LOG_BACKUP_COUNT", 7))

    # 定义日志输出格式
    _FORMAT = "%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s"

    # 默认配置
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
            # 默认错误日志，保留最新1个小时的日志，位置在log/error.log
            "error_file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": "ERROR",
                "formatter": "standard",
                "filename": os.path.join(log_dir, "error.log"),
                "when": "H",
                "interval": 1,
                "backupCount": 1,
            },
            # 全量日志，保留最新1小时的日志，位置在log/default.log
            "default_file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": FASTDEPLOY_LOGGING_LEVEL,
                "formatter": "standard",
                "filename": os.path.join(log_dir, "default.log"),
                "when": "H",
                "interval": 1,
                "backupCount": 1,
            },
            # 错误日志归档，保留7天内的日志，每隔1小时一个文件，形式如：FastDeploy/log/2025-08-14/error_2025-08-14-18.log
            "error_archive": {
                "class": "fastdeploy.logger.handlers.IntervalRotatingFileHandler",
                "level": "ERROR",
                "formatter": "standard",
                "filename": os.path.join(log_dir, "error.log"),
                "backupDays": 7,
                "interval": 1,
                "encoding": "utf-8",
            },
            # 全量日志归档，保留7天内的日志，每隔1小时一个文件，形式如：FastDeploy/log/2025-08-14/default_2025-08-14-18.log
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
            # 默认日志记录器,全局共享
            "fastdeploy": {
                "level": "DEBUG",
                "handlers": ["error_file", "default_file", "error_archive", "default_archive"],
                "propagate": False,
            }
        },
    }

    # 如果提供了配置文件，则加载配置文件
    if config_file and os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 合并环境变量配置到用户配置中,环境变量的优先级高于自定义的优先级
        if "handlers" in config:
            for handler_name, handler_config in config["handlers"].items():
                if "backupCount" not in handler_config and "DailyRotating" in handler_config.get("class", ""):
                    handler_config["backupCount"] = backup_count
                if handler_config.get("level") == "INFO" and is_debug:
                    handler_config["level"] = "DEBUG"
    else:
        config = default_config

    # 应用日志配置
    logging.config.dictConfig(config)

    # 避免重复加载
    setup_logging._configured = True

    # 返回fastdeploy的logger
    return logging.getLogger("fastdeploy")
