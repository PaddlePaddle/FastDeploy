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

import codecs
import logging
import os
import re
import time
from logging.handlers import BaseRotatingHandler
from pathlib import Path

"""自定义日志处理器模块：
该模块包含FastDeploy项目中使用的自定义日志处理器实现，
用于处理和控制日志输出格式、级别和目标等。
"""


class IntervalRotatingFileHandler(BaseRotatingHandler):
    """
    按天创建文件夹(YYYY-MM-DD)，每n小时创建日志文件(prefix_YYYY-MM-DD-HH.log)
    自动清理过期数据，清理频率与interval同步，支持多进程环境
    """

    def __init__(
        self,
        filename,
        backupDays=7,
        interval=1,
        encoding="utf-8",
        delay=False,
        utc=False,
        **kwargs,
    ):
        """
        初始化日志处理器

        Args:
            filename (str): 日志文件基础路径
            backupDays (int): 保留天数，默认7天
            interval (int): 日志分割间隔小时数，必须能被24整除，默认1小时
            encoding (str): 文件编码，默认utf-8
            delay (bool): 是否延迟打开文件，默认False
            utc (bool): 是否使用UTC时间，默认False
        """
        if 24 % interval != 0:
            raise ValueError("interval必须能被24整除")

        self.backup_days = backupDays
        self.interval = interval
        self.utc = utc
        self.base_path = Path(filename)
        self.current_day = self._get_current_day()
        self.current_hour = self._get_current_hour()
        self.current_dir = self._get_day_dir()
        self.current_filename = self._get_hourly_filename()
        self.current_filepath = self.current_dir / self.current_filename
        self.last_clean_time = 0  # 初始化为0确保第一次会执行清理
        self.seconds_per_hour = 3600
        # 确保目录存在
        self.current_dir.mkdir(parents=True, exist_ok=True)

        BaseRotatingHandler.__init__(self, str(self.current_filepath), "a", encoding, delay)

    def _get_current_time(self):
        """获取当前时间"""
        return time.gmtime() if self.utc else time.localtime()

    def _get_current_day(self):
        """获取当前日期字符串(YYYY-MM-DD)"""
        return time.strftime("%Y-%m-%d", self._get_current_time())

    def _get_current_hour(self):
        """获取当前小时数(0-23)"""
        current_hour = self._get_current_time().tm_hour
        return current_hour - (current_hour % self.interval)

    def _get_day_dir(self):
        """获取当天目录路径"""
        return self.base_path.parent / self.current_day

    def _get_hourly_filename(self):
        """获取按小时分割的文件名"""
        prefix = self.base_path.stem
        hour_str = f"{self.current_hour:02d}"
        return f"{prefix}_{self.current_day}-{hour_str}.log"

    def shouldRollover(self, record):
        """检查是否需要滚动日志"""
        now_day = self._get_current_day()
        now_hour = self._get_current_hour()

        # 检查日期或小时是否变化
        if now_day != self.current_day or now_hour != self.current_hour:
            return True

        # 检查是否需要执行清理（每个interval小时执行一次）
        current_time = time.time()
        if current_time - self.last_clean_time > self.interval * self.seconds_per_hour:
            return True

        return False

    def doRollover(self):
        """执行日志滚动和清理"""
        if self.stream:
            self.stream.close()
            self.stream = None

        # 更新当前日期和小时
        self.current_day = self._get_current_day()
        self.current_hour = self._get_current_hour()
        self.current_dir = self._get_day_dir()
        self.current_filename = self._get_hourly_filename()
        self.current_filepath = self.current_dir / self.current_filename

        # 创建新目录（如果不存在）
        self.current_dir.mkdir(parents=True, exist_ok=True)

        # 打开新日志文件
        if not self.delay:
            self.stream = self._open()

        # 执行清理（每个interval小时执行一次）
        current_time = time.time()
        if current_time - self.last_clean_time > self.interval * self.seconds_per_hour:
            self._clean_expired_data()
            self.last_clean_time = current_time

    def _open(self):
        """打开日志文件并创建符号链接"""
        if self.encoding is None:
            stream = open(str(self.current_filepath), self.mode)
        else:
            stream = codecs.open(str(self.current_filepath), self.mode, self.encoding)

        # 创建符号链接（支持多进程）
        self._create_symlink()
        return stream

    def _create_symlink(self):
        """创建指向当前日志文件的符号链接"""
        symlink_path = self.base_path.parent / f"current_{self.base_path.stem}.log"

        try:
            if symlink_path.exists():
                if symlink_path.is_symlink():
                    os.remove(str(symlink_path))
                else:
                    # 不是符号链接则重命名避免冲突
                    backup_path = symlink_path.with_name(f"{symlink_path.stem}_backup.log")
                    os.rename(str(symlink_path), str(backup_path))

            # 创建相对路径符号链接
            rel_path = self.current_filepath.relative_to(self.base_path.parent)
            os.symlink(str(rel_path), str(symlink_path))
        except OSError:
            # 多进程环境下可能发生竞争，忽略错误
            pass

    def _clean_expired_data(self):
        """清理过期数据"""
        if self.backup_days <= 0:
            return

        cutoff_time = time.time() - (self.backup_days * 24 * self.seconds_per_hour)
        day_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        file_pattern = re.compile(r"^.+_\d{4}-\d{2}-\d{2}-\d{2}\.log$")

        # 清理过期日目录
        for dir_name in os.listdir(str(self.base_path.parent)):
            dir_path = self.base_path.parent / dir_name
            if not dir_path.is_dir():
                continue

            if day_pattern.match(dir_name):
                try:
                    dir_mtime = os.path.getmtime(str(dir_path))
                    if dir_mtime < cutoff_time:
                        # 删除整个过期目录
                        for file in dir_path.glob("*"):
                            try:
                                file.unlink()
                            except OSError:
                                pass
                        dir_path.rmdir()
                except OSError:
                    pass

        # 额外检查当前目录下的过期文件
        for file_name in os.listdir(str(self.base_path.parent)):
            file_path = self.base_path.parent / file_name
            if file_path.is_file() and file_pattern.match(file_name):
                try:
                    file_mtime = os.path.getmtime(str(file_path))
                    if file_mtime < cutoff_time:
                        file_path.unlink()
                except OSError:
                    pass


class LazyFileHandler(logging.Handler):
    """
    延迟创建日志文件的处理器，仅在首次写入日志时创建实际的文件处理器
    """

    def __init__(self, filename, backupCount, level=logging.NOTSET, formatter=None):
        super().__init__(level=level)
        self.filename = filename
        self.backupCount = backupCount
        self.formatter = formatter
        self._real_handler = None

    def create_real_handler(self):
        """创建实际的文件处理器"""
        handler = DailyRotatingFileHandler(self.filename, backupCount=self.backupCount)
        handler.setLevel(self.level)
        if self.formatter:
            handler.setFormatter(self.formatter)
        return handler

    def emit(self, record):
        # 检查日志级别
        if record.levelno < self.level:
            return

        self.acquire()
        try:
            if self._real_handler is None:
                self._real_handler = self.create_real_handler()
        finally:
            self.release()
        # 将日志记录传递给实际处理器
        self._real_handler.emit(record)

    def close(self):
        # 关闭实际处理器（如果存在）
        if self._real_handler is not None:
            self._real_handler.close()
        super().close()


class DailyRotatingFileHandler(BaseRotatingHandler):
    """
    like `logging.TimedRotatingFileHandler`, but this class support multi-process
    """

    def __init__(
        self,
        filename,
        backupCount=0,
        encoding="utf-8",
        delay=False,
        utc=False,
        **kwargs,
    ):
        """
            初始化 RotatingFileHandler 对象。

        Args:
            filename (str): 日志文件的路径，可以是相对路径或绝对路径。
            backupCount (int, optional, default=0): 保存的备份文件数量，默认为 0，表示不保存备份文件。
            encoding (str, optional, default='utf-8'): 编码格式，默认为 'utf-8'。
            delay (bool, optional, default=False): 是否延迟写入，默认为 False，表示立即写入。
            utc (bool, optional, default=False): 是否使用 UTC 时区，默认为 False，表示不使用 UTC 时区。
            kwargs (dict, optional): 其他参数将被传递给 BaseRotatingHandler 类的 init 方法。

        Raises:
            TypeError: 如果 filename 不是 str 类型。
            ValueError: 如果 backupCount 小于等于 0。
        """
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
        check scroll through the log
        """
        if self.current_filename != self._compute_fn():
            return True
        return False

    def doRollover(self):
        """
        scroll log
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
        Calculate the log file name corresponding current time
        """
        return self.base_filename + "." + time.strftime(self.suffix, time.localtime())

    def _open(self):
        """
        open new log file
        """
        if self.encoding is None:
            stream = open(str(self.current_log_path), self.mode)
        else:
            stream = codecs.open(str(self.current_log_path), self.mode, self.encoding)

        if self.base_log_path.exists():
            try:
                if not self.base_log_path.is_symlink() or os.readlink(self.base_log_path) != self.current_filename:
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
        delete expired log files
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
