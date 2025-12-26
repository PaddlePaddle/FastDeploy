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
日志模块：用于初始化和获取 FastDeploy 日志记录器。
本模块提供 get_logger 方法，统一管理各子模块的日志记录行为。
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
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def _initialize(self):
        """显式初始化日志系统"""
        with self._lock:
            if not self._initialized:
                setup_logging()
                self._initialized = True

    def get_logger(self, name, file_name=None, without_formater=False, print_to_console=False):
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
            # 延迟初始化
            if not self._initialized:
                self._initialize()
            return self._get_unified_logger(name)

        # 兼容原有接口
        return self._get_legacy_logger(name, file_name, without_formater, print_to_console)

    def _get_unified_logger(self, name):
        """
        新的统一日志获取方式
        """
        if name is None:
            return logging.getLogger("fastdeploy")

        # 处理 __main__ 特殊情况
        if name == "__main__":
            import __main__

            # 获取主模块的 __file__ 属性
            if hasattr(__main__, "__file__"):
                # 获取主模块的文件名
                base_name = Path(__main__.__file__).stem
                # 创建带前缀的日志器
                return logging.getLogger(f"fastdeploy.main.{base_name}")
            return logging.getLogger("fastdeploy.main")

        # 如果已经是fastdeploy命名空间，直接使用
        if name.startswith("fastdeploy.") or name == "fastdeploy":
            return logging.getLogger(name)
        else:
            # 其他情况添加fastdeploy前缀
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
        兼容原有接口的日志获取方式
        """

        log_dir = envs.FD_LOG_DIR
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        is_debug = envs.FD_DEBUG
        # logger = logging.getLogger(name)
        # 为了兼容原有接口，使用命名空间进行隔离，避免logger覆盖、混乱等问题
        legacy_name = f"legacy.{name}"
        logger = logging.getLogger(legacy_name)

        # 设置日志级别
        if is_debug:
            logger.setLevel(level=logging.DEBUG)
        else:
            logger.setLevel(level=logging.INFO)

        # 设置格式化器
        formatter = ColoredFormatter(
            "%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s"
        )

        # 清除现有的handlers（保持原有逻辑）
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # 创建主日志文件handler
        LOG_FILE = f"{log_dir}/{file_name}"
        backup_count = int(envs.FD_LOG_BACKUP_COUNT)
        # handler = LazyFileHandler(filename=LOG_FILE, backupCount=backup_count, level=hanlder_level)
        handler = DailyRotatingFileHandler(LOG_FILE, backupCount=backup_count)

        # 创建ERROR日志文件handler（新增功能）
        if not file_name.endswith(".log"):
            file_name = f"{file_name}.log" if "." not in file_name else file_name.split(".")[0] + ".log"
        ERROR_LOG_FILE = os.path.join(log_dir, file_name.replace(".log", "_error.log"))
        error_handler = LazyFileHandler(
            filename=ERROR_LOG_FILE, backupCount=backup_count, level=logging.ERROR, formatter=None
        )

        if not without_formater:
            handler.setFormatter(formatter)
            error_handler.setFormatter(formatter)

        # 添加文件handlers
        logger.addHandler(handler)
        logger.addHandler(error_handler)

        # 控制台handler
        if print_to_console:
            console_handler = logging.StreamHandler()
            if not without_formater:
                console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            console_handler.propagate = False

        # 设置propagate（保持原有逻辑）
        # logger.propagate = False

        return logger
