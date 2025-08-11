import codecs
import os
import re
import time
from datetime import datetime
from logging.handlers import BaseRotatingHandler, TimedRotatingFileHandler
from pathlib import Path

"""自定义日志处理器模块：
该模块包含FastDeploy项目中使用的自定义日志处理器实现，
用于处理和控制日志输出格式、级别和目标等。
"""


class DailyFolderTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    自定义处理器：每天一个目录，每小时一个文件
    文件结构：
        logs/
        └── 2025-08-05/
            ├── fastdeploy_error_10.log
            └── fastdeploy_debug_10.log
    """

    def __init__(self, filename, when="H", interval=1, backupCount=48, encoding=None, utc=False, **kwargs):
        # 支持从dictConfig中通过filename传入 base_log_dir/base_filename
        base_log_dir, base_name = os.path.split(filename)
        base_filename = os.path.splitext(base_name)[0]

        self.base_log_dir = base_log_dir
        self.base_filename = base_filename
        self.current_day = datetime.now().strftime("%Y-%m-%d")
        self._update_baseFilename()

        super().__init__(
            filename=self.baseFilename,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
            utc=utc,
        )

    def _update_baseFilename(self):
        dated_dir = os.path.join(self.base_log_dir, self.current_day)
        os.makedirs(dated_dir, exist_ok=True)
        self.baseFilename = os.path.abspath(
            os.path.join(dated_dir, f"{self.base_filename}_{datetime.now().strftime('%H')}.log")
        )

    def shouldRollover(self, record):
        new_day = datetime.now().strftime("%Y-%m-%d")
        if new_day != self.current_day:
            self.current_day = new_day
            return 1
        return super().shouldRollover(record)

    def doRollover(self):
        self.stream.close()
        self._update_baseFilename()
        self.stream = self._open()


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
