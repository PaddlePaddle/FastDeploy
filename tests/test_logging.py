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

"""
Unit tests for FastDeploy logging infrastructure.

Tests cover:
- get_logger() returns loggers with correct naming
- FD_LOG_LEVEL env var controls log level
- FastDeployLogger singleton behavior
- Console handler presence in unified logger
- Legacy get_logger(name, file) backward compatibility
"""

import logging
import os
import unittest
from unittest.mock import patch

try:
    import paddle  # noqa: F401

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

SKIP_MSG = "PaddlePaddle is not installed"


@unittest.skipUnless(HAS_PADDLE, SKIP_MSG)
class TestGetLogger(unittest.TestCase):
    """Tests for the fastdeploy.logger.get_logger convenience function."""

    def test_get_logger_returns_logger_instance(self):
        from fastdeploy.logger import get_logger

        logger = get_logger("test_module")
        self.assertIsInstance(logger, logging.Logger)

    def test_get_logger_prefixes_with_fastdeploy(self):
        from fastdeploy.logger import get_logger

        logger = get_logger("test_module")
        self.assertTrue(
            logger.name.startswith("fastdeploy"),
            f"Expected logger name to start with 'fastdeploy', got '{logger.name}'",
        )

    def test_get_logger_none_returns_root_fastdeploy(self):
        from fastdeploy.logger import get_logger

        logger = get_logger(None)
        self.assertEqual(logger.name, "fastdeploy")

    def test_get_logger_already_namespaced(self):
        from fastdeploy.logger import get_logger

        logger = get_logger("fastdeploy.engine")
        self.assertEqual(logger.name, "fastdeploy.engine")

    def test_get_logger_adds_prefix_for_plain_name(self):
        from fastdeploy.logger import get_logger

        logger = get_logger("scheduler")
        self.assertEqual(logger.name, "fastdeploy.scheduler")


@unittest.skipUnless(HAS_PADDLE, SKIP_MSG)
class TestFDLogLevel(unittest.TestCase):
    """Tests for FD_LOG_LEVEL environment variable."""

    def test_fd_log_level_default_is_info(self):
        with patch.dict(os.environ, {}, clear=False):
            # Remove FD_LOG_LEVEL and FD_DEBUG if set
            os.environ.pop("FD_LOG_LEVEL", None)
            os.environ.pop("FD_DEBUG", None)
            # envs uses lazy lambdas, so reading the attribute re-evaluates os.getenv
            from fastdeploy import envs

            level = envs.FD_LOG_LEVEL
            self.assertEqual(level, "INFO")

    def test_fd_log_level_debug_when_fd_debug_set(self):
        with patch.dict(os.environ, {"FD_DEBUG": "1"}, clear=False):
            os.environ.pop("FD_LOG_LEVEL", None)
            from fastdeploy import envs

            level = envs.FD_LOG_LEVEL
            self.assertEqual(level, "DEBUG")

    def test_fd_log_level_explicit_overrides_fd_debug(self):
        with patch.dict(os.environ, {"FD_DEBUG": "1", "FD_LOG_LEVEL": "WARNING"}, clear=False):
            from fastdeploy import envs

            level = envs.FD_LOG_LEVEL
            self.assertEqual(level, "WARNING")

    def test_fd_log_level_accepts_error(self):
        with patch.dict(os.environ, {"FD_LOG_LEVEL": "ERROR"}, clear=False):
            from fastdeploy import envs

            level = envs.FD_LOG_LEVEL
            self.assertEqual(level, "ERROR")


@unittest.skipUnless(HAS_PADDLE, SKIP_MSG)
class TestFastDeployLoggerSingleton(unittest.TestCase):
    """Tests for FastDeployLogger singleton pattern."""

    def test_singleton_returns_same_instance(self):
        from fastdeploy.logger import FastDeployLogger

        instance1 = FastDeployLogger()
        instance2 = FastDeployLogger()
        self.assertIs(instance1, instance2)


@unittest.skipUnless(HAS_PADDLE, SKIP_MSG)
class TestConsoleHandler(unittest.TestCase):
    """Tests for console handler presence in unified logger setup."""

    def setUp(self):
        """Reset setup_logging state for clean test isolation."""
        from fastdeploy.logger.setup_logging import setup_logging

        setup_logging._configured = False

    def test_unified_logger_has_console_handler(self):
        from fastdeploy.logger.setup_logging import setup_logging

        fd_logger = setup_logging()

        handler_classes = [type(h).__name__ for h in fd_logger.handlers]
        self.assertTrue(
            any("StreamHandler" in cls for cls in handler_classes),
            f"Expected a StreamHandler (console) among handlers, got: {handler_classes}",
        )

    def tearDown(self):
        """Reset setup_logging state after test."""
        from fastdeploy.logger.setup_logging import setup_logging

        setup_logging._configured = False


@unittest.skipUnless(HAS_PADDLE, SKIP_MSG)
class TestLegacyGetLogger(unittest.TestCase):
    """Tests for backward compatibility with legacy get_logger(name, file)."""

    def test_legacy_get_logger_still_works(self):
        from fastdeploy.utils import get_logger

        logger = get_logger("test_legacy", "test_legacy.log")
        self.assertIsInstance(logger, logging.Logger)
        # Legacy loggers use "legacy." prefix namespace
        self.assertTrue(
            logger.name.startswith("legacy."),
            f"Expected legacy logger to have 'legacy.' prefix, got '{logger.name}'",
        )

    def test_legacy_prebuilt_loggers_accessible(self):
        from fastdeploy.utils import llm_logger, scheduler_logger

        self.assertIsInstance(llm_logger, logging.Logger)
        self.assertIsInstance(scheduler_logger, logging.Logger)


if __name__ == "__main__":
    unittest.main()
