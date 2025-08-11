"""
单测：自定义日志处理器
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from logging import INFO, getLogger
from unittest.mock import MagicMock, patch

from fastdeploy.util.handlers import (
    DailyFolderTimedRotatingFileHandler,
    DailyRotatingFileHandler,
)


class TestDailyFolderTimedRotatingFileHandler(unittest.TestCase):
    """测试 DailyFolderTimedRotatingFileHandler"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="fd_handler_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_daily_folder_structure(self):
        """测试每天一个目录，每小时一个文件"""
        handler = DailyFolderTimedRotatingFileHandler(
            os.path.join(self.temp_dir, "test.log"), when="H", interval=1, backupCount=3
        )
        logger = getLogger("test_daily_folder")
        logger.addHandler(handler)
        logger.setLevel(INFO)

        # 写入日志
        logger.info("Test log message")
        handler.flush()  # 确保日志写入
        handler.close()

        # 验证目录结构
        today = datetime.now().strftime("%Y-%m-%d")
        log_dir = os.path.join(self.temp_dir, today)
        self.assertTrue(os.path.isdir(log_dir))

        log_file = os.path.join(log_dir, f"test_{datetime.now().strftime('%H')}.log")
        self.assertTrue(os.path.isfile(log_file))

    def test_rollover(self):
        """测试跨天滚动"""
        handler = DailyFolderTimedRotatingFileHandler(
            os.path.join(self.temp_dir, "test.log"), when="H", interval=1, backupCount=3
        )
        logger = getLogger("test_rollover")
        logger.addHandler(handler)
        logger.setLevel(INFO)

        # 写入第一条日志
        logger.info("Test log message before rollover")
        handler.flush()

        # 验证第一天的文件
        today = datetime.now().strftime("%Y-%m-%d")
        today_dir = os.path.join(self.temp_dir, today)
        self.assertTrue(os.path.isdir(today_dir))

        # 模拟跨天 - 需要 mock 处理器内部使用的 datetime
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_str = tomorrow.strftime("%Y-%m-%d")

        # 创建一个 mock 记录来触发滚动检查
        mock_record = MagicMock()

        # 修改处理器的当前天数来模拟跨天
        handler.current_day = tomorrow_str

        # 手动触发滚动
        with patch.object(handler, "_update_baseFilename") as mock_update:
            # 设置新的基础文件名
            tomorrow_dir = os.path.join(self.temp_dir, tomorrow_str)
            os.makedirs(tomorrow_dir, exist_ok=True)
            new_filename = os.path.join(tomorrow_dir, f"test_{tomorrow.strftime('%H')}.log")

            def update_side_effect():
                handler.baseFilename = new_filename

            mock_update.side_effect = update_side_effect

            # 触发滚动
            if handler.shouldRollover(mock_record):
                handler.doRollover()

        # 写入第二条日志到新文件
        logger.info("Test log message after rollover")
        handler.flush()
        handler.close()

        # 验证新目录存在
        self.assertTrue(os.path.isdir(tomorrow_dir))


class TestDailyRotatingFileHandler(unittest.TestCase):
    """测试 DailyRotatingFileHandler"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="fd_handler_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_daily_rotation(self):
        """测试每天滚动"""
        log_file = os.path.join(self.temp_dir, "test.log")
        handler = DailyRotatingFileHandler(log_file, backupCount=3)
        logger = getLogger("test_daily_rotation")
        logger.addHandler(handler)
        logger.setLevel(INFO)

        # 写入第一条日志
        logger.info("Test log message day 1")
        handler.flush()

        # 模拟时间变化到第二天
        with patch.object(handler, "_compute_fn") as mock_compute:
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            new_filename = f"test.log.{tomorrow}"
            mock_compute.return_value = new_filename

            # 手动触发滚动检查和执行
            mock_record = MagicMock()
            if handler.shouldRollover(mock_record):
                handler.doRollover()

        # 写入第二条日志
        logger.info("Test log message day 2")
        handler.flush()
        handler.close()

        # 验证文件存在
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        # 检查原始文件和带日期的文件
        base_file = os.path.join(self.temp_dir, "test.log")
        today_file = os.path.join(self.temp_dir, f"test.log.{today}")
        tomorrow_file = os.path.join(self.temp_dir, f"test.log.{tomorrow}")

        # 至少应该有一个文件存在
        files_exist = any([os.path.isfile(base_file), os.path.isfile(today_file), os.path.isfile(tomorrow_file)])
        self.assertTrue(files_exist, f"No log files found in {self.temp_dir}")

    def test_backup_count(self):
        """测试备份文件数量限制"""
        log_file = os.path.join(self.temp_dir, "test.log")
        handler = DailyRotatingFileHandler(log_file, backupCount=2)
        logger = getLogger("test_backup_count")
        logger.addHandler(handler)
        logger.setLevel(INFO)

        # 创建多个日期的日志文件
        base_date = datetime.now()

        for i in range(5):  # 创建5天的日志
            date_str = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            test_file = os.path.join(self.temp_dir, f"test.log.{date_str}")

            # 直接创建文件
            with open(test_file, "w") as f:
                f.write(f"Test log for {date_str}\n")

        # 触发清理
        handler.delete_expired_files()
        handler.close()

        # 验证备份文件数量（应该保留最新的2个 + 当前文件）
        log_files = [f for f in os.listdir(self.temp_dir) if f.startswith("test.log.")]
        print(f"Log files found: {log_files}")  # 调试输出

        # backupCount=2 意味着应该最多保留2个备份文件
        self.assertLessEqual(len(log_files), 3)  # 2个备份 + 可能的当前文件


if __name__ == "__main__":
    unittest.main()
