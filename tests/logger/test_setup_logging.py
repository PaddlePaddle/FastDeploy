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
        nested = os.path.join(self.temp_dir, "a", "b", "c")
        setup_logging(log_dir=nested)
        self.assertTrue(Path(nested).is_dir())

    def test_default_config_fallback(self):
        """Pass a non-existent config_file to trigger default_config"""
        fake_cfg = os.path.join(self.temp_dir, "no_such_cfg.json")
        setup_logging(config_file=fake_cfg)
        logger = logging.getLogger("fastdeploy")
        self.assertTrue(logger.handlers)
        handler_classes = [h.__class__.__name__ for h in logger.handlers]
        self.assertIn("TimedRotatingFileHandler", handler_classes)

    def test_debug_level_affects_handlers(self):
        """FD_DEBUG=1 should force DEBUG level"""
        with patch("fastdeploy.envs.FD_DEBUG", 1):
            with patch("logging.config.dictConfig") as mock_cfg:
                setup_logging()
                called_config = mock_cfg.call_args[0][0]
                for handler in called_config["handlers"].values():
                    self.assertIn("formatter", handler)
                self.assertEqual(called_config["handlers"]["console_stdout"]["level"], "DEBUG")

    @patch("logging.config.dictConfig")
    def test_custom_config_with_dailyrotating_and_debug(self, mock_dict):
        custom_cfg = {
            "version": 1,
            "handlers": {
                "daily": {
                    "class": "logging.handlers.DailyRotatingFileHandler",
                    "level": "INFO",
                    "formatter": "plain",
                }
            },
            "loggers": {"fastdeploy": {"handlers": ["daily"], "level": "INFO"}},
        }
        cfg_path = Path(self.temp_dir) / "cfg.json"
        cfg_path.write_text(json.dumps(custom_cfg))

        with patch("fastdeploy.envs.FD_DEBUG", 1):
            setup_logging(config_file=str(cfg_path))

        config_used = mock_dict.call_args[0][0]
        self.assertIn("daily", config_used["handlers"])
        self.assertEqual(config_used["handlers"]["daily"]["level"], "DEBUG")
        self.assertIn("backupCount", config_used["handlers"]["daily"])

    def test_configure_once(self):
        """Ensure idempotent setup"""
        l1 = setup_logging()
        l2 = setup_logging()
        self.assertIs(l1, l2)

    def test_envs_priority_used_for_log_dir(self):
        """When log_dir=None, should use envs.FD_LOG_DIR"""
        with patch("fastdeploy.envs.FD_LOG_DIR", self.temp_dir):
            setup_logging()
            self.assertTrue(os.path.exists(self.temp_dir))

    @patch("logging.StreamHandler.emit")
    def test_console_colored(self, mock_emit):
        setup_logging()
        logger = logging.getLogger("fastdeploy")
        logger.error("color test")
        self.assertTrue(mock_emit.called)

    @patch("logging.config.dictConfig")
    def test_backup_count_merging(self, mock_dict):
        custom_cfg = {
            "version": 1,
            "handlers": {"daily": {"class": "logging.handlers.DailyRotatingFileHandler", "formatter": "plain"}},
            "loggers": {"fastdeploy": {"handlers": ["daily"], "level": "INFO"}},
        }
        cfg_path = Path(self.temp_dir) / "cfg.json"
        cfg_path.write_text(json.dumps(custom_cfg))

        setup_logging(config_file=str(cfg_path))

        config_used = mock_dict.call_args[0][0]
        self.assertEqual(config_used["handlers"]["daily"]["backupCount"], 3)

    @patch("logging.config.dictConfig")
    def test_error_logs_use_stderr_handler(self, mock_dict):
        """ERROR级别日志应该使用stderr输出"""
        setup_logging()
        config_used = mock_dict.call_args[0][0]
        self.assertIn("console_stderr", config_used["handlers"])
        self.assertEqual(config_used["handlers"]["console_stderr"]["stream"], "ext://sys.stderr")
        self.assertEqual(config_used["handlers"]["console_stderr"]["level"], "ERROR")

    @patch("logging.config.dictConfig")
    def test_console_stdout_filters_below_error(self, mock_dict):
        """console_stdout应该只输出低于ERROR级别的日志"""
        setup_logging()
        config_used = mock_dict.call_args[0][0]
        self.assertIn("console_stdout", config_used["handlers"])
        self.assertIn("below_error", config_used["handlers"]["console_stdout"]["filters"])
        self.assertEqual(config_used["handlers"]["console_stdout"]["stream"], "ext://sys.stdout")


class TestMaxLevelFilter(unittest.TestCase):
    def test_filter_allows_below_level(self):
        """MaxLevelFilter应该允许低于指定级别的日志通过"""
        filter = MaxLevelFilter("ERROR")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        self.assertTrue(filter.filter(record))

    def test_filter_blocks_at_level(self):
        """MaxLevelFilter应该阻止等于指定级别的日志"""
        filter = MaxLevelFilter("ERROR")
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        self.assertFalse(filter.filter(record))

    def test_filter_blocks_above_level(self):
        """MaxLevelFilter应该阻止高于指定级别的日志"""
        filter = MaxLevelFilter("ERROR")
        record = logging.LogRecord(
            name="test", level=logging.CRITICAL, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        self.assertFalse(filter.filter(record))

    def test_filter_with_numeric_level(self):
        """MaxLevelFilter应该支持数字级别"""
        filter = MaxLevelFilter(logging.WARNING)
        info_record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        warning_record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        self.assertTrue(filter.filter(info_record))
        self.assertFalse(filter.filter(warning_record))


if __name__ == "__main__":
    unittest.main()
