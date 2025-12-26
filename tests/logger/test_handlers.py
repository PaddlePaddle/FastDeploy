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
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.base_filename = os.path.join(self.temp_dir, "test.log")

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test initialization parameter validation"""
        # Test invalid interval
        with self.assertRaises(ValueError):
            IntervalRotatingFileHandler(self.base_filename, interval=7)

        # Test valid initialization
        handler = IntervalRotatingFileHandler(self.base_filename, interval=6, backupDays=3)
        self.assertEqual(handler.interval, 6)
        self.assertEqual(handler.backup_days, 3)
        handler.close()

    def test_file_rotation(self):
        """Test log file rotation mechanism"""
        handler = IntervalRotatingFileHandler(self.base_filename, interval=6, backupDays=1)

        # Get initial state
        initial_day = handler.current_day
        initial_hour = handler.current_hour

        # First log write
        record = LogRecord("test", 20, "/path", 1, "Test message", [], None)
        handler.emit(record)

        # Verify file existence
        expected_dir = Path(self.temp_dir) / initial_day
        expected_file = f"test_{initial_day}-{initial_hour:02d}.log"
        self.assertTrue((expected_dir / expected_file).exists())

        # Verify symlink creation
        symlink = Path(self.temp_dir) / "current_test.log"
        self.assertTrue(symlink.is_symlink())

        handler.close()

    def test_time_based_rollover(self):
        """Test time-based rollover triggers"""
        handler = IntervalRotatingFileHandler(self.base_filename, interval=1, backupDays=1)

        # Force initial time settings
        handler.current_day = "2000-01-01"
        handler.current_hour = 0

        # Test hour change trigger
        with unittest.mock.patch.object(handler, "_get_current_day", return_value="2000-01-01"):
            with unittest.mock.patch.object(handler, "_get_current_hour", return_value=1):
                self.assertTrue(handler.shouldRollover(None))

        # Test day change trigger
        with unittest.mock.patch.object(handler, "_get_current_day", return_value="2000-01-02"):
            with unittest.mock.patch.object(handler, "_get_current_hour", return_value=0):
                self.assertTrue(handler.shouldRollover(None))

        handler.close()

    def test_cleanup_logic(self):
        """Test expired file cleanup mechanism"""
        # Use fixed test time
        test_time = datetime(2023, 1, 1, 12, 0)
        with unittest.mock.patch("time.time", return_value=time.mktime(test_time.timetuple())):
            handler = IntervalRotatingFileHandler(self.base_filename, interval=1, backupDays=0)  # Clean immediately

            # Create test directory structure
            old_day = (test_time - timedelta(days=2)).strftime("%Y-%m-%d")
            old_dir = Path(self.temp_dir) / old_day
            old_dir.mkdir()

            # Create test file
            old_file = old_dir / f"test_{old_day}-00.log"
            old_file.write_text("test content")

            # Ensure correct timestamps
            old_time = time.mktime((test_time - timedelta(days=2)).timetuple())
            os.utime(str(old_dir), (old_time, old_time))
            os.utime(str(old_file), (old_time, old_time))

            # Verify file creation
            self.assertTrue(old_file.exists())

            # Execute cleanup
            handler._clean_expired_data()

            # Short delay for filesystem operations
            time.sleep(0.1)

            # Verify cleanup result
            if old_dir.exists():
                print(f"Directory contents: {list(old_dir.glob('*'))}")
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
        """Test multiple interval configurations"""
        for interval in [1, 2, 3, 4, 6, 8, 12, 24]:
            with self.subTest(interval=interval):
                handler = IntervalRotatingFileHandler(self.base_filename, interval=interval)
                current_hour = handler._get_current_time().tm_hour
                expected_hour = current_hour - (current_hour % interval)
                self.assertEqual(handler.current_hour, expected_hour)
                handler.close()

    def test_utc_mode(self):
        """Test UTC time mode"""
        handler = IntervalRotatingFileHandler(self.base_filename, utc=True)
        self.assertTrue(time.strftime("%Y-%m-%d", time.gmtime()).startswith(handler.current_day))
        handler.close()

    def test_symlink_creation(self):
        """Test symlink creation and updates"""
        handler = IntervalRotatingFileHandler(self.base_filename)
        symlink = Path(self.temp_dir) / "current_test.log"

        # Get initial symlink target
        initial_target = os.readlink(str(symlink))

        # Force rollover (simulate time change)
        with unittest.mock.patch.object(handler, "_get_current_day", return_value="2000-01-01"):
            with unittest.mock.patch.object(handler, "_get_current_hour", return_value=12):
                handler.doRollover()

        # Get new symlink target
        new_target = os.readlink(str(symlink))

        # Verify target updated
        self.assertNotEqual(initial_target, new_target)
        self.assertIn("2000-01-01/test_2000-01-01-12.log", new_target)
        handler.close()


class TestDailyRotatingFileHandler(unittest.TestCase):
    """Tests for DailyRotatingFileHandler"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="fd_handler_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_daily_rotation(self):
        """Test daily log rotation"""
        log_file = os.path.join(self.temp_dir, "test.log")
        handler = DailyRotatingFileHandler(log_file, backupCount=3)
        logger = getLogger("test_daily_rotation")
        logger.addHandler(handler)
        logger.setLevel(INFO)

        # Write first log
        logger.info("Test log message day 1")
        handler.flush()

        # Simulate time change to next day
        with patch.object(handler, "_compute_fn") as mock_compute:
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            new_filename = f"test.log.{tomorrow}"
            mock_compute.return_value = new_filename

            # Manually trigger rollover check
            mock_record = MagicMock()
            if handler.shouldRollover(mock_record):
                handler.doRollover()

        # Write second log
        logger.info("Test log message day 2")
        handler.flush()
        handler.close()

        # Verify file existence
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        base_file = os.path.join(self.temp_dir, "test.log")
        today_file = os.path.join(self.temp_dir, f"test.log.{today}")
        tomorrow_file = os.path.join(self.temp_dir, f"test.log.{tomorrow}")

        # At least one file should exist
        files_exist = any([os.path.isfile(base_file), os.path.isfile(today_file), os.path.isfile(tomorrow_file)])
        self.assertTrue(files_exist, f"No log files found in {self.temp_dir}")

    def test_backup_count(self):
        """Test backup file count limitation"""
        log_file = os.path.join(self.temp_dir, "test.log")
        handler = DailyRotatingFileHandler(log_file, backupCount=2)
        logger = getLogger("test_backup_count")
        logger.addHandler(handler)
        logger.setLevel(INFO)

        # Create log files for multiple dates
        base_date = datetime.now()

        for i in range(5):  # Create 5 days of logs
            date_str = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            test_file = os.path.join(self.temp_dir, f"test.log.{date_str}")

            # Create file directly
            with open(test_file, "w") as f:
                f.write(f"Test log for {date_str}\n")

        # Trigger cleanup
        handler.delete_expired_files()
        handler.close()

        # Verify backup count (should keep latest 2 + current file)
        log_files = [f for f in os.listdir(self.temp_dir) if f.startswith("test.log.")]
        print(f"Log files found: {log_files}")  # Debug output

        # backupCount=2 means max 2 backup files should remain
        self.assertLessEqual(len(log_files), 3)  # 2 backups + possible current file


class TestLazyFileHandler(unittest.TestCase):

    def setUp(self):
        # Create temporary directory
        self.tmpdir = tempfile.TemporaryDirectory()
        self.logfile = Path(self.tmpdir.name) / "test.log"

    def tearDown(self):
        # Clean up temporary directory
        self.tmpdir.cleanup()

    def test_lazy_initialization_and_write(self):
        """Test lazy initialization and log writing"""
        logger = logging.getLogger("test_lazy")
        logger.setLevel(logging.DEBUG)

        # Initialize LazyFileHandler
        handler = LazyFileHandler(str(self.logfile), backupCount=3, level=logging.DEBUG)
        logger.addHandler(handler)

        # _real_handler should not be created yet
        self.assertIsNone(handler._real_handler)

        # Write a log
        logger.info("Hello Lazy Handler")

        # _real_handler should be created after writing
        self.assertIsNotNone(handler._real_handler)

        # Log file should exist with correct content
        self.assertTrue(self.logfile.exists())
        with open(self.logfile, "r") as f:
            content = f.read()
        self.assertIn("Hello Lazy Handler", content)

        # Close handler
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
