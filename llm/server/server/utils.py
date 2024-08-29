# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import codecs
import logging
import os
import pickle
import re
import time
from datetime import datetime
from enum import Enum
from logging.handlers import BaseRotatingHandler
from pathlib import Path
import subprocess


class DailyRotatingFileHandler(BaseRotatingHandler):
    """
    - 可以支持多进程
    - 只支持自然日分割
    - 暂不支持UTC
    """

    def __init__(
        self,
        filename,
        backupCount=0,
        encoding="utf-8",
        delay=False,
        utc=False,
        **kwargs
    ):
        self.backup_count = backupCount
        self.utc = utc
        self.suffix = "%Y-%m-%d"
        self.base_log_path = Path(filename)
        self.base_filename = self.base_log_path.name
        self.current_filename = self._compute_fn()
        self.current_log_path = self.base_log_path.with_name(self.current_filename)
        BaseRotatingHandler.__init__(self, filename, "a", encoding, delay)

    def shouldRollover(self, record):
        """
        判断是否该滚动日志，如果当前时间对应的日志文件名与当前打开的日志文件名不一致，则需要滚动日志
        """
        if self.current_filename != self._compute_fn():
            return True
        return False

    def doRollover(self):
        """
        滚动日志
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        self.current_filename = self._compute_fn()
        self.current_log_path = self.base_log_path.with_name(self.current_filename)

        if not self.delay:
            self.stream = self._open()

        self.delete_expired_files()

    def _compute_fn(self):
        """
        计算当前时间对应的日志文件名
        """
        return self.base_filename + "." + time.strftime(self.suffix, time.localtime())

    def _open(self):
        """
        打开新的日志文件，同时更新base_filename指向的软链，修改软链不会对日志记录产生任何影响
        """
        if self.encoding is None:
            stream = open(str(self.current_log_path), self.mode)
        else:
            stream = codecs.open(str(self.current_log_path), self.mode, self.encoding)

        # 删除旧的软链
        if self.base_log_path.exists():
            try:
                if (
                    not self.base_log_path.is_symlink()
                    or os.readlink(self.base_log_path) != self.current_filename
                ):
                    os.remove(self.base_log_path)
            except OSError:
                pass

        try:
            os.symlink(self.current_filename, str(self.base_log_path))
        except OSError:
            pass
        return stream

    def delete_expired_files(self):
        """
        删除过期的日志
        """
        if self.backup_count <= 0:
            return

        file_names = os.listdir(str(self.base_log_path.parent))
        result = []
        prefix = self.base_filename + "."
        plen = len(prefix)
        for file_name in file_names:
            if file_name[:plen] == prefix:
                suffix = file_name[plen:]
                if re.match(r"^\d{4}-\d{2}-\d{2}(\.\w+)?$", suffix):
                    result.append(file_name)
        if len(result) < self.backup_count:
            result = []
        else:
            result.sort()
            result = result[: len(result) - self.backup_count]

        for file_name in result:
            os.remove(str(self.base_log_path.with_name(file_name)))


def get_logger(name, file_name, without_formater=False):
    """
    获取logger
    """
    log_dir = os.getenv("FD_LOG_DIR", default="log")
    is_debug = int(os.getenv("FD_DEBUG", default=0))
    logger = logging.getLogger(name)
    if is_debug:
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(level=logging.INFO)

    LOG_FILE = "{0}/{1}".format(log_dir, file_name)
    backup_count = int(os.getenv("FD_LOG_BACKUP_COUNT", 7))
    handler = DailyRotatingFileHandler(LOG_FILE, backupCount=backup_count)

    formatter = logging.Formatter(
        "%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s"
    )
    if not without_formater:
        handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler.propagate = False
    return logger

# 实例化单例logger
model_server_logger = get_logger("model_server", "infer_server.log")
http_server_logger = get_logger("http_server", "http_server.log")
data_processor_logger = get_logger("data_processor", "data_processor.log")
monitor_logger = get_logger("monitor_logger", "monitor_logger.log", True)
error_logger = get_logger("error_logger", "error_logger.log", True)


def str_to_datetime(date_string):
    """datetime字符串转datetime对象"""
    if "." in date_string:
        return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S.%f")
    else:
        return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")


def datetime_diff(datetime_start, datetime_end):
    """
    计算两个日期时间之间的差值（以秒为单位）。

    Args:
        datetime_start (Union[str, datetime.datetime]): 开始时间，可以是字符串或datetime.datetime对象。
        datetime_end (Union[str, datetime.datetime]): 结束时间，可以是字符串或datetime.datetime对象。

    Returns:
        float: 日期时间差值，以秒为单位。
    """
    if isinstance(datetime_start, str):
        datetime_start = str_to_datetime(datetime_start)
    if isinstance(datetime_end, str):
        datetime_end = str_to_datetime(datetime_end)
    if datetime_end > datetime_start:
        cost = datetime_end - datetime_start
    else:
        cost = datetime_start - datetime_end
    return cost.total_seconds()
