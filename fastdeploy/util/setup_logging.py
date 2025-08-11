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
                "class": "fastdeploy.util.formatters.ColoredFormatter",
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
            "error_file": {
                "class": "fastdeploy.util.handlers.DailyRotatingFileHandler",
                "level": "ERROR",
                "formatter": "standard",
                "filename": os.path.join(log_dir, "error.log"),
                "backupCount": backup_count,
                "encoding": "utf-8",
            },
            "error_file2": {
                "class": "fastdeploy.util.handlers.DailyFolderTimedRotatingFileHandler",
                "level": "ERROR",
                "formatter": "standard",
                "filename": os.path.join(log_dir, "error.log"),
                "when": "H",
                "interval": 1,
                "backupCount": 48,
                "encoding": "utf-8",
            },
            "default_file": {
                "class": "fastdeploy.util.handlers.DailyRotatingFileHandler",
                "level": FASTDEPLOY_LOGGING_LEVEL,
                "formatter": "standard",
                "filename": os.path.join(log_dir, "default.log"),
                "backupCount": backup_count,
                "encoding": "utf-8",
            },
            "default_file2": {
                "class": "fastdeploy.util.handlers.DailyFolderTimedRotatingFileHandler",
                "level": FASTDEPLOY_LOGGING_LEVEL,
                "formatter": "standard",
                "filename": os.path.join(log_dir, "default.log"),
                "when": "H",
                "interval": 1,
                "backupCount": 48,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "fastdeploy": {
                "level": "DEBUG",
                "handlers": ["console", "error_file", "default_file", "error_file2", "default_file2"],
                "propagate": False,
            }
        },
        "root": {"level": "WARNING", "handlers": ["console"]},
    }

    # 如果提供了配置文件，则加载配置文件
    if config_file and os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 合并环境变量配置到用户配置中
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
