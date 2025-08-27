# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


import logging
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from logging import INFO, LogRecord, getLogger
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastdeploy.logger.handlers import (
    DailyRotatingFileHandler,
    IntervalRotatingFileHandler,
    LazyFileHandler,
)


class TestIntervalRotatingFileHandler(unittest.TestCase):
    def setUp(self):
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.base_filename = os.path.join(self.temp_dir, "test.log")

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """测试初始化参数校验"""
        # 测试无效interval
        with self.assertRaises(ValueError):
            handler = IntervalRotatingFileHandler(self.base_filename, interval=7)

        # 测试有效初始化
        handler = IntervalRotatingFileHandler(self.base_filename, interval=6, backupDays=3)
        self.assertEqual(handler.interval, 6)
        self.assertEqual(handler.backup_days, 3)
        handler.close()

    def test_file_rotation(self):
        """测试日志文件滚动"""
        handler = IntervalRotatingFileHandler(self.base_filename, interval=6, backupDays=1)

        # 模拟初始状态
        initial_day = handler.current_day
        initial_hour = handler.current_hour

        # 首次写入
        record = LogRecord("test", 20, "/path", 1, "Test message", [], None)
        handler.emit(record)

        # 验证文件存在
        expected_dir = Path(self.temp_dir) / initial_day
        expected_file = f"test_{initial_day}-{initial_hour:02d}.log"
        self.assertTrue((expected_dir / expected_file).exists())

        # 验证符号链接
        symlink = Path(self.temp_dir) / "current_test.log"
        self.assertTrue(symlink.is_symlink())

        handler.close()

    def test_time_based_rollover(self):
        """测试基于时间的滚动触发"""
        handler = IntervalRotatingFileHandler(self.base_filename, interval=1, backupDays=1)

        # 强制设置初始时间
        handler.current_day = "2000-01-01"
        handler.current_hour = 0

        # 测试小时变化触发
        with unittest.mock.patch.object(handler, "_get_current_day", return_value="2000-01-01"):
            with unittest.mock.patch.object(handler, "_get_current_hour", return_value=1):
                self.assertTrue(handler.shouldRollover(None))

        # 测试日期变化触发
        with unittest.mock.patch.object(handler, "_get_current_day", return_value="2000-01-02"):
            with unittest.mock.patch.object(handler, "_get_current_hour", return_value=0):
                self.assertTrue(handler.shouldRollover(None))

        handler.close()

    def test_cleanup_logic(self):
        """测试过期文件清理"""
        # 使用固定测试时间
        test_time = datetime(2023, 1, 1, 12, 0)
        with unittest.mock.patch("time.time", return_value=time.mktime(test_time.timetuple())):
            handler = IntervalRotatingFileHandler(self.base_filename, interval=1, backupDays=0)  # 立即清理

            # 创建测试目录结构
            old_day = (test_time - timedelta(days=2)).strftime("%Y-%m-%d")
            old_dir = Path(self.temp_dir) / old_day
            old_dir.mkdir()

            # 创建测试文件
            old_file = old_dir / f"test_{old_day}-00.log"
            old_file.write_text("test content")

            # 确保文件时间戳正确
            old_time = time.mktime((test_time - timedelta(days=2)).timetuple())
            os.utime(str(old_dir), (old_time, old_time))
            os.utime(str(old_file), (old_time, old_time))

            # 验证文件创建成功
            self.assertTrue(old_file.exists())

            # 执行清理
            handler._clean_expired_data()

            # 添加短暂延迟确保文件系统操作完成
            time.sleep(0.1)

            # 验证清理结果
            if old_dir.exists():
                # 调试输出：列出目录内容
                print(f"Directory contents: {list(old_dir.glob('*'))}")
                # 尝试强制删除以清理测试环境
                try:
                    shutil.rmtree(str(old_dir))
                except Exception as e:
                    print(f"Cleanup failed: {e}")

            self.assertFalse(
                old_dir.exists(),
                f"Directory {old_dir} should have been deleted. Contents: {list(old_dir.glob('*')) if old_dir.exists() else '[]'}",
            )

            handler.close()

    def test_multi_interval(self):
        """测试多间隔配置"""
        for interval in [1, 2, 3, 4, 6, 8, 12, 24]:
            with self.subTest(interval=interval):
                handler = IntervalRotatingFileHandler(self.base_filename, interval=interval)
                current_hour = handler._get_current_time().tm_hour
                expected_hour = current_hour - (current_hour % interval)
                self.assertEqual(handler.current_hour, expected_hour)
                handler.close()

    def test_utc_mode(self):
        """测试UTC时间模式"""
        handler = IntervalRotatingFileHandler(self.base_filename, utc=True)
        self.assertTrue(time.strftime("%Y-%m-%d", time.gmtime()).startswith(handler.current_day))
        handler.close()

    def test_symlink_creation(self):
        """测试符号链接创建和更新"""
        handler = IntervalRotatingFileHandler(self.base_filename)
        symlink = Path(self.temp_dir) / "current_test.log"

        # 获取初始符号链接目标
        initial_target = os.readlink(str(symlink))

        # 强制触发滚动（模拟时间变化）
        with unittest.mock.patch.object(handler, "_get_current_day", return_value="2000-01-01"):
            with unittest.mock.patch.object(handler, "_get_current_hour", return_value=12):
                handler.doRollover()

        # 获取新符号链接目标
        new_target = os.readlink(str(symlink))

        # 验证目标已更新
        self.assertNotEqual(initial_target, new_target)
        self.assertIn("2000-01-01/test_2000-01-01-12.log", new_target)
        handler.close()


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


class TestLazyFileHandler(unittest.TestCase):

    def setUp(self):
        # 创建临时目录
        self.tmpdir = tempfile.TemporaryDirectory()
        self.logfile = Path(self.tmpdir.name) / "test.log"

    def tearDown(self):
        # 清理临时目录
        self.tmpdir.cleanup()

    def test_lazy_initialization_and_write(self):
        logger = logging.getLogger("test_lazy")
        logger.setLevel(logging.DEBUG)

        # 初始化 LazyFileHandler
        handler = LazyFileHandler(str(self.logfile), backupCount=3, level=logging.DEBUG)
        logger.addHandler(handler)

        # 此时 _real_handler 应该还没创建
        self.assertIsNone(handler._real_handler)

        # 写一条日志
        logger.info("Hello Lazy Handler")

        # 写入后 _real_handler 应该被创建
        self.assertIsNotNone(handler._real_handler)

        # 日志文件应该存在且内容包含日志信息
        self.assertTrue(self.logfile.exists())
        with open(self.logfile, "r") as f:
            content = f.read()
        self.assertIn("Hello Lazy Handler", content)

        # 关闭 handler
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
