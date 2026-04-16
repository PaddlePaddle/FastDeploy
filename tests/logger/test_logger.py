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
import unittest
from unittest.mock import patch

from fastdeploy.logger.handlers import LazyFileHandler
from fastdeploy.logger.logger import FastDeployLogger


class LoggerTests(unittest.TestCase):
    """Modified test class, testing internal methods through instances"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="fd_unittest_")
        self.env_patchers = [
            patch("fastdeploy.envs.FD_LOG_DIR", self.tmp_dir),
            patch("fastdeploy.envs.FD_DEBUG", 0),
            patch("fastdeploy.envs.FD_LOG_BACKUP_COUNT", 1),
        ]
        for p in self.env_patchers:
            p.start()

        # Create test instance
        self.logger = FastDeployLogger()

    def tearDown(self):
        for p in self.env_patchers:
            p.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_unified_logger(self):
        """Test _get_unified_logger through instance"""
        test_cases = [(None, "fastdeploy"), ("module", "fastdeploy.module"), ("fastdeploy.utils", "fastdeploy.utils")]

        for name, expected in test_cases:
            with self.subTest(name=name):
                result = self.logger._get_unified_logger(name)
                self.assertEqual(result.name, expected)

    def test_main_module_handling(self):
        """Test __main__ special handling"""
        with patch("__main__.__file__", "/path/to/test_script.py"):
            result = self.logger._get_unified_logger("__main__")
            self.assertEqual(result.name, "fastdeploy.main.test_script")

    def test_legacy_logger_creation(self):
        """Test _get_legacy_logger through instance"""
        legacy_logger = self.logger._get_legacy_logger(
            "test", "test.log", without_formater=False, print_to_console=True
        )

        # Verify basic properties
        self.assertTrue(legacy_logger.name.startswith("legacy."))
        self.assertEqual(legacy_logger.level, logging.INFO)

        # Verify handlers
        self.assertEqual(len(legacy_logger.handlers), 3)  # file + error + console

    def test_logger_propagate(self):
        """Test log propagation settings"""
        legacy_logger = self.logger._get_legacy_logger("test", "test.log")
        self.assertTrue(legacy_logger.propagate)

    def test_get_trace_logger_basic(self):
        """Test basic functionality of get_trace_logger"""
        logger = self.logger.get_trace_logger("test_trace", "trace_test.log")

        # Verify basic properties
        self.assertTrue(logger.name.startswith("legacy."))
        self.assertEqual(logger.level, logging.INFO)

        # Verify handler count
        self.assertEqual(len(logger.handlers), 2)  # main log and error log

    def test_get_trace_logger_with_console(self):
        """Test trace logger with console output"""
        logger = self.logger.get_trace_logger("test_trace_console", "trace_console_test.log", print_to_console=True)

        # Verify handler count
        self.assertEqual(len(logger.handlers), 3)  # main log + error log + console

    def test_get_trace_logger_without_formatter(self):
        """Test trace logger without formatting"""
        logger = self.logger.get_trace_logger("test_trace_no_fmt", "trace_no_fmt_test.log", without_formater=True)

        # Verify handlers have no formatter
        for handler in logger.handlers:
            self.assertIsNone(handler.formatter)

    def test_get_trace_logger_debug_mode(self):
        """Test trace logger in debug mode"""
        with patch("fastdeploy.envs.FD_DEBUG", "1"):
            logger = self.logger.get_trace_logger("test_trace_debug", "trace_debug_test.log")
            self.assertEqual(logger.level, logging.DEBUG)

    def test_get_trace_logger_directory_creation(self):
        """Test line 105: log directory creation functionality"""
        import os
        from unittest.mock import patch

        # Test creation of non-existent directory
        with tempfile.TemporaryDirectory() as temp_dir:
            test_log_dir = os.path.join(temp_dir, "test_logs")
            with patch("fastdeploy.envs.FD_LOG_DIR", test_log_dir):
                # Ensure directory does not exist
                self.assertFalse(os.path.exists(test_log_dir))

                # Call get_trace_logger, should create directory
                self.logger.get_trace_logger("test_dir_creation", "test.log")

                # Verify directory is created
                self.assertTrue(os.path.exists(test_log_dir))
                self.assertTrue(os.path.isdir(test_log_dir))

    def test_get_trace_logger_handler_cleanup(self):
        """Test line 126: handler cleanup functionality"""
        # First create a logger and add some handlers
        test_logger = logging.getLogger("legacy.test_cleanup")
        initial_handler_count = len(test_logger.handlers)

        # Add some test handlers
        test_handler1 = logging.StreamHandler()
        test_handler2 = logging.StreamHandler()
        test_logger.addHandler(test_handler1)
        test_logger.addHandler(test_handler2)

        # Verify handlers are added
        self.assertEqual(len(test_logger.handlers), initial_handler_count + 2)

        # Call get_trace_logger, should clean up existing handlers
        logger = self.logger.get_trace_logger("test_cleanup", "cleanup_test.log")

        # Verify new logger's handler count (should be 2: main log and error log)
        self.assertEqual(len(logger.handlers), 2)

    def test_log_file_name_handling_error(self):
        """Test log file name handling logic"""
        test_cases = [
            ("test", "test_error.log"),
        ]

        for input_name, expected_name in test_cases:
            with self.subTest(input_name=input_name):
                # Create logger and get actual processed file name
                logger = self.logger.get_trace_logger("test_file_name", input_name)

                # Get file name from handler
                for handler in logger.handlers:
                    if isinstance(handler, LazyFileHandler):
                        actual_name = os.path.basename(handler.filename)
                        self.assertTrue(actual_name.endswith(expected_name))


class LoggerExtraTests(unittest.TestCase):
    def setUp(self):
        self.logger = FastDeployLogger()

    def tearDown(self):
        if hasattr(FastDeployLogger, "_instance"):
            FastDeployLogger._instance = None
        if hasattr(FastDeployLogger, "_initialized"):
            FastDeployLogger._initialized = False

    def test_singleton_behavior(self):
        """Ensure multiple instances are same"""
        a = FastDeployLogger()
        b = FastDeployLogger()
        self.assertIs(a, b)

    def test_initialize_only_once(self):
        """Ensure _initialize won't re-run if already initialized"""
        self.logger._initialized = True
        with patch("fastdeploy.logger.logger.setup_logging") as mock_setup:
            self.logger._initialize()
            mock_setup.assert_not_called()

    def test_get_logger_unified_path(self):
        """Directly test get_logger unified path"""
        with patch("fastdeploy.logger.logger.setup_logging") as mock_setup:
            log = self.logger.get_logger("utils")
            self.assertTrue(log.name.startswith("fastdeploy."))
            mock_setup.assert_called_once()

    def test_get_logger_legacy_path(self):
        """Test legacy get_logger path"""
        with patch("fastdeploy.logger.logger.FastDeployLogger._get_legacy_logger") as mock_legacy:
            self.logger.get_logger("x", "y.log", False, False)
            mock_legacy.assert_called_once()

    def test_get_legacy_logger_debug_mode(self):
        """Ensure debug level is set when FD_DEBUG=1"""
        with patch("fastdeploy.envs.FD_DEBUG", 1):
            logger = self.logger._get_legacy_logger("debug_case", "d.log")
            self.assertEqual(logger.level, logging.DEBUG)

    def test_get_legacy_logger_without_formatter(self):
        """Test legacy logger without formatter"""
        logger = self.logger._get_legacy_logger("nofmt", "n.log", without_formater=True)
        for h in logger.handlers:
            self.assertIsNone(h.formatter)


if __name__ == "__main__":
    unittest.main(verbosity=2)
