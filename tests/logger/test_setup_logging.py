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

import json
import logging
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastdeploy.logger.setup_logging import MaxLevelFilter, setup_logging


class TestSetupLogging(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="logger_setup_test_")
        if hasattr(setup_logging, "_configured"):
            delattr(setup_logging, "_configured")
        self.patches = [
            patch("fastdeploy.envs.FD_LOG_DIR", self.temp_dir),
            patch("fastdeploy.envs.FD_DEBUG", 0),
            patch("fastdeploy.envs.FD_LOG_BACKUP_COUNT", "3"),
        ]
        [p.start() for p in self.patches]

    def tearDown(self):
        [p.stop() for p in self.patches]
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if hasattr(setup_logging, "_configured"):
            delattr(setup_logging, "_configured")

    def test_log_dir_created(self):
        """Log directory should be created"""
        nested = os.path.join(self.temp_dir, "a", "b", "c")
        setup_logging(log_dir=nested)
        self.assertTrue(Path(nested).is_dir())

    def test_configure_once(self):
        """Ensure idempotent setup - only configures once"""
        setup_logging()
        self.assertTrue(setup_logging._configured)
        # Second call should not raise
        setup_logging()
        self.assertTrue(setup_logging._configured)

    def test_envs_priority_used_for_log_dir(self):
        """When log_dir=None, should use envs.FD_LOG_DIR"""
        with patch("fastdeploy.envs.FD_LOG_DIR", self.temp_dir):
            setup_logging()
            self.assertTrue(os.path.exists(self.temp_dir))

    def test_log_dir_stored(self):
        """setup_logging should store log_dir for later use"""
        setup_logging(log_dir=self.temp_dir)
        self.assertEqual(setup_logging._log_dir, self.temp_dir)

    def test_no_config_file_no_dictconfig(self):
        """When config_file is not provided, dictConfig should not be called"""
        with patch("logging.config.dictConfig") as mock_dict:
            setup_logging()
            self.assertFalse(mock_dict.called)

    def test_config_file_with_dictconfig(self):
        """When config_file is provided, dictConfig should be called"""
        custom_cfg = {
            "version": 1,
            "handlers": {},
            "loggers": {},
        }
        cfg_path = Path(self.temp_dir) / "cfg.json"
        cfg_path.write_text(json.dumps(custom_cfg))

        with patch("logging.config.dictConfig") as mock_dict:
            setup_logging(config_file=str(cfg_path))
            self.assertTrue(mock_dict.called)

    def test_config_file_not_exists_uses_default(self):
        """When config_file doesn't exist, use default config"""
        fake_cfg = os.path.join(self.temp_dir, "no_such_cfg.json")

        with patch("logging.config.dictConfig") as mock_dict:
            setup_logging(config_file=fake_cfg)
            self.assertTrue(mock_dict.called)
            # Should use default config
            config_used = mock_dict.call_args[0][0]
            self.assertIn("handlers", config_used)
            self.assertIn("loggers", config_used)

    def test_backup_count_merging(self):
        """backupCount should be merged into handler config"""
        custom_cfg = {
            "version": 1,
            "handlers": {"daily": {"class": "logging.handlers.DailyRotatingFileHandler", "formatter": "plain"}},
            "loggers": {"fastdeploy": {"handlers": ["daily"], "level": "INFO"}},
        }
        cfg_path = Path(self.temp_dir) / "cfg.json"
        cfg_path.write_text(json.dumps(custom_cfg))

        with patch("logging.config.dictConfig") as mock_dict:
            setup_logging(config_file=str(cfg_path))
            config_used = mock_dict.call_args[0][0]
            self.assertEqual(config_used["handlers"]["daily"]["backupCount"], 3)

    def test_debug_level_affects_handlers(self):
        """FD_DEBUG=1 should force DEBUG level in handlers"""
        custom_cfg = {
            "version": 1,
            "handlers": {"test": {"class": "logging.StreamHandler", "level": "INFO"}},
            "loggers": {},
        }
        cfg_path = Path(self.temp_dir) / "cfg.json"
        cfg_path.write_text(json.dumps(custom_cfg))

        with patch("fastdeploy.envs.FD_DEBUG", 1):
            with patch("logging.config.dictConfig") as mock_dict:
                setup_logging(config_file=str(cfg_path))
                config_used = mock_dict.call_args[0][0]
                self.assertEqual(config_used["handlers"]["test"]["level"], "DEBUG")

    def test_default_config_has_channels(self):
        """Default config should have channel loggers configured"""
        fake_cfg = os.path.join(self.temp_dir, "no_such_cfg.json")

        with patch("logging.config.dictConfig") as mock_dict:
            setup_logging(config_file=fake_cfg)
            config_used = mock_dict.call_args[0][0]
            # Check channel loggers exist
            self.assertIn("fastdeploy.main", config_used["loggers"])
            self.assertIn("fastdeploy.request", config_used["loggers"])
            self.assertIn("fastdeploy.console", config_used["loggers"])

    def test_default_config_has_handlers(self):
        """Default config should have file handlers configured"""
        fake_cfg = os.path.join(self.temp_dir, "no_such_cfg.json")

        with patch("logging.config.dictConfig") as mock_dict:
            setup_logging(config_file=fake_cfg)
            config_used = mock_dict.call_args[0][0]
            # Check handlers exist
            self.assertIn("main_file", config_used["handlers"])
            self.assertIn("request_file", config_used["handlers"])
            self.assertIn("error_file", config_used["handlers"])
            self.assertIn("console_stderr", config_used["handlers"])

    def test_default_config_stderr_handler(self):
        """Default config console_stderr should output to stderr"""
        fake_cfg = os.path.join(self.temp_dir, "no_such_cfg.json")

        with patch("logging.config.dictConfig") as mock_dict:
            setup_logging(config_file=fake_cfg)
            config_used = mock_dict.call_args[0][0]
            self.assertEqual(config_used["handlers"]["console_stderr"]["stream"], "ext://sys.stderr")
            self.assertEqual(config_used["handlers"]["console_stderr"]["level"], "ERROR")

    def test_default_config_stdout_filters_below_error(self):
        """Default config console_stdout should filter below ERROR level"""
        fake_cfg = os.path.join(self.temp_dir, "no_such_cfg.json")

        with patch("logging.config.dictConfig") as mock_dict:
            setup_logging(config_file=fake_cfg)
            config_used = mock_dict.call_args[0][0]
            self.assertIn("console_stdout", config_used["handlers"])
            self.assertIn("below_error", config_used["handlers"]["console_stdout"]["filters"])
            self.assertEqual(config_used["handlers"]["console_stdout"]["stream"], "ext://sys.stdout")


class TestMaxLevelFilter(unittest.TestCase):
    def test_filter_allows_below_level(self):
        """MaxLevelFilter should allow logs below specified level"""
        filter = MaxLevelFilter("ERROR")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        self.assertTrue(filter.filter(record))

    def test_filter_blocks_at_level(self):
        """MaxLevelFilter should block logs at specified level"""
        filter = MaxLevelFilter("ERROR")
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        self.assertFalse(filter.filter(record))

    def test_filter_blocks_above_level(self):
        """MaxLevelFilter should block logs above specified level"""
        filter = MaxLevelFilter("ERROR")
        record = logging.LogRecord(
            name="test", level=logging.CRITICAL, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        self.assertFalse(filter.filter(record))

    def test_filter_with_numeric_level(self):
        """MaxLevelFilter should support numeric level"""
        filter = MaxLevelFilter(logging.WARNING)
        info_record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        warning_record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        self.assertTrue(filter.filter(info_record))
        self.assertFalse(filter.filter(warning_record))


class TestChannelLoggers(unittest.TestCase):
    """Test channel logger configuration via get_logger"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="logger_channel_test_")
        if hasattr(setup_logging, "_configured"):
            delattr(setup_logging, "_configured")
        # Clear channel configuration cache
        from fastdeploy.logger.logger import FastDeployLogger

        FastDeployLogger._configured_channels = set()

        self.patches = [
            patch("fastdeploy.envs.FD_LOG_DIR", self.temp_dir),
            patch("fastdeploy.envs.FD_DEBUG", 0),
            patch("fastdeploy.envs.FD_LOG_BACKUP_COUNT", "3"),
            patch("fastdeploy.envs.FD_LOG_LEVEL", None),
        ]
        [p.start() for p in self.patches]

    def tearDown(self):
        [p.stop() for p in self.patches]
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if hasattr(setup_logging, "_configured"):
            delattr(setup_logging, "_configured")
        # Clear channel configuration cache
        from fastdeploy.logger.logger import FastDeployLogger

        FastDeployLogger._configured_channels = set()

    def test_main_channel_has_handlers(self):
        """main channel root logger should have handlers"""
        from fastdeploy.logger import get_logger

        get_logger("test", channel="main")
        main_channel = logging.getLogger("fastdeploy.main")
        self.assertTrue(len(main_channel.handlers) > 0)

    def test_request_channel_has_handlers(self):
        """request channel root logger should have handlers"""
        from fastdeploy.logger import get_logger

        get_logger("test", channel="request")
        request_channel = logging.getLogger("fastdeploy.request")
        self.assertTrue(len(request_channel.handlers) > 0)

    def test_console_channel_has_stdout_handler(self):
        """console channel should have stdout handler"""
        from fastdeploy.logger import get_logger

        get_logger("test", channel="console")
        console_channel = logging.getLogger("fastdeploy.console")
        handler_types = [type(h).__name__ for h in console_channel.handlers]
        self.assertIn("StreamHandler", handler_types)

    def test_child_logger_propagates_to_channel(self):
        """Child loggers should propagate to channel root logger"""
        from fastdeploy.logger import get_logger

        logger = get_logger("child_test", channel="main")
        # Child logger should have no direct handlers (propagates to parent)
        self.assertEqual(len(logger.handlers), 0)
        self.assertEqual(logger.name, "fastdeploy.main.child_test")

    def test_channel_file_mapping(self):
        """Each channel should write to correct log file"""
        from fastdeploy.logger.logger import FastDeployLogger

        expected_files = {
            "main": "fastdeploy.log",
            "request": "request.log",
            "console": "console.log",
        }
        self.assertEqual(FastDeployLogger._channel_files, expected_files)

    def test_multiple_loggers_same_channel(self):
        """Multiple loggers on same channel should share channel root handlers"""
        from fastdeploy.logger import get_logger

        logger1 = get_logger("test1", channel="main")
        logger2 = get_logger("test2", channel="main")

        main_channel = logging.getLogger("fastdeploy.main")
        # Both child loggers should have no handlers
        self.assertEqual(len(logger1.handlers), 0)
        self.assertEqual(len(logger2.handlers), 0)
        # Channel root should have handlers
        self.assertTrue(len(main_channel.handlers) > 0)


if __name__ == "__main__":
    unittest.main()
