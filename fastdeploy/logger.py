"""
日志模块：用于初始化和获取 FastDeploy 日志记录器。
本模块提供 get_logger 方法，统一管理各子模块的日志记录行为。
"""

import logging
import os

from fastdeploy import envs
from fastdeploy.util.formatters import ColoredFormatter
from fastdeploy.util.handlers import DailyRotatingFileHandler
from fastdeploy.util.setup_logging import setup_logging

# 初始化一次日志系统
setup_logging()


def get_logger(name, file_name=None, without_formater=False, print_to_console=False):
    """
    获取日志记录器（兼容原有接口）

    Args:
        name: 日志器名称
        file_name: 日志文件名（保持兼容性）
        without_formater: 是否不使用格式化器
        print_to_console: 是否打印到控制台
    """
    # 如果只有一个参数，使用新的统一命名方式
    if file_name is None and not without_formater and not print_to_console:
        return _get_unified_logger(name)

    # 兼容原有接口
    return _get_legacy_logger(name, file_name, without_formater, print_to_console)


def _get_unified_logger(name):
    """
    新的统一日志获取方式
    """
    if name is None:
        return logging.getLogger("fastdeploy")

    # 处理 __main__ 特殊情况
    if name == "__main__":
        return logging.getLogger("fastdeploy.main")

    # 如果已经是fastdeploy命名空间，直接使用
    if name.startswith("fastdeploy.") or name == "fastdeploy":
        return logging.getLogger(name)
    else:
        # 其他情况添加fastdeploy前缀
        return logging.getLogger(f"fastdeploy.{name}")


def _get_legacy_logger(name, file_name, without_formater=False, print_to_console=False):
    """
    兼容原有接口的日志获取方式
    """

    log_dir = envs.FD_LOG_DIR
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    is_debug = int(envs.FD_DEBUG)
    # logger = logging.getLogger(name)
    legacy_name = f"legacy.{name}"
    logger = logging.getLogger(legacy_name)

    # 设置日志级别
    if is_debug:
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(level=logging.INFO)

    # 清除现有的handlers（保持原有逻辑）
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建主日志文件handler
    LOG_FILE = f"{log_dir}/{file_name}"
    backup_count = int(envs.FD_LOG_BACKUP_COUNT)
    handler = DailyRotatingFileHandler(LOG_FILE, backupCount=backup_count)

    # 创建ERROR日志文件handler（新增功能）
    ERROR_LOG_FILE = f"{log_dir}/error_{file_name}"
    error_handler = DailyRotatingFileHandler(ERROR_LOG_FILE, backupCount=backup_count)
    error_handler.setLevel(logging.ERROR)

    # 设置格式化器
    formatter = ColoredFormatter("%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s")

    if not without_formater:
        handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)

    # 添加文件handlers
    logger.addHandler(handler)
    logger.addHandler(error_handler)

    # 控制台handler（如果需要）
    if print_to_console:
        console_handler = logging.StreamHandler()
        if not without_formater:
            console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        console_handler.propagate = False

    # 设置propagate（保持原有逻辑）
    handler.propagate = False
    error_handler.propagate = False
    logger.propagate = False

    return logger
