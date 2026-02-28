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
import unittest


class TestInterceptPaddleLoggers(unittest.TestCase):
    """Test cases for intercept_paddle_loggers context manager from tools.logger_patch"""

    def test_intercept_paddle_loggers_with_paddle_prefix(self):
        """Test intercept_paddle_loggers configures paddle loggers correctly (line 28-30)"""
        from fastdeploy.logger.logger import intercept_paddle_loggers

        # Create a logger with existing handlers before interception
        test_logger_name = "paddle.test.logger"
        test_logger = logging.getLogger(test_logger_name)

        # Add some handlers to the logger
        handler1 = logging.StreamHandler()
        handler2 = logging.StreamHandler()
        test_logger.addHandler(handler1)
        test_logger.addHandler(handler2)
        self.assertEqual(len(test_logger.handlers), 2)

        # Use the context manager to intercept paddle loggers
        with intercept_paddle_loggers():
            # Get logger inside context - should be configured by interceptor
            intercepted_logger = logging.getLogger(test_logger_name)

            # Verify the logger was reconfigured by the interceptor
            self.assertEqual(len(intercepted_logger.handlers), 1)
            self.assertIsInstance(intercepted_logger.handlers[0], logging.StreamHandler)
            self.assertEqual(intercepted_logger.level, logging.INFO)
            self.assertFalse(intercepted_logger.propagate)

        # Clean up
        test_logger.handlers = []

    def test_intercept_paddle_loggers_restores_original(self):
        """Test intercept_paddle_loggers restores original getLogger after exit (line 46)"""
        from fastdeploy.logger.logger import intercept_paddle_loggers

        # Store original getLogger before context
        original_getLogger = logging.getLogger

        # Use the context manager
        with intercept_paddle_loggers():
            # Inside context, getLogger should be patched
            self.assertNotEqual(logging.getLogger, original_getLogger)

        # After exit, getLogger should be restored
        self.assertEqual(logging.getLogger, original_getLogger)

    def test_intercept_paddle_loggers_non_paddle_logger_unchanged(self):
        """Test non-paddle loggers are not affected by intercept_paddle_loggers"""
        from fastdeploy.logger.logger import intercept_paddle_loggers

        # Create a non-paddle logger
        test_logger_name = "other.test.logger"
        test_logger = logging.getLogger(test_logger_name)

        # Add a handler
        original_handler = logging.StreamHandler()
        test_logger.addHandler(original_handler)
        original_handler_count = len(test_logger.handlers)

        # Use the context manager
        with intercept_paddle_loggers():
            # Get the same logger
            result_logger = logging.getLogger(test_logger_name)
            # Non-paddle loggers should not be modified
            self.assertEqual(len(result_logger.handlers), original_handler_count)
            self.assertEqual(result_logger.handlers[0], original_handler)

        # Clean up
        test_logger.handlers = []

    def test_intercept_paddle_loggers_exception_safety(self):
        """Test intercept_paddle_loggers restores getLogger even if exception occurs"""
        from fastdeploy.logger.logger import intercept_paddle_loggers

        original_getLogger = logging.getLogger

        try:
            with intercept_paddle_loggers():
                # Raise an exception inside context
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # After exception, getLogger should still be restored
        self.assertEqual(logging.getLogger, original_getLogger)


if __name__ == "__main__":
    unittest.main()
