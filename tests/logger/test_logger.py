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
    """修改后的测试类，通过实例测试内部方法"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="fd_unittest_")
        self.env_patchers = [
            patch("fastdeploy.envs.FD_LOG_DIR", self.tmp_dir),
            patch("fastdeploy.envs.FD_DEBUG", 0),
            patch("fastdeploy.envs.FD_LOG_BACKUP_COUNT", "1"),
        ]
        for p in self.env_patchers:
            p.start()

        # 创建测试用实例
        self.logger = FastDeployLogger()

    def tearDown(self):
        for p in self.env_patchers:
            p.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_unified_logger(self):
        """通过实例测试_get_unified_logger"""
        test_cases = [(None, "fastdeploy"), ("module", "fastdeploy.module"), ("fastdeploy.utils", "fastdeploy.utils")]

        for name, expected in test_cases:
            with self.subTest(name=name):
                result = self.logger._get_unified_logger(name)
                self.assertEqual(result.name, expected)

    def test_main_module_handling(self):
        """测试__main__特殊处理"""
        with patch("__main__.__file__", "/path/to/test_script.py"):
            result = self.logger._get_unified_logger("__main__")
            self.assertEqual(result.name, "fastdeploy.main.test_script")

    def test_legacy_logger_creation(self):
        """通过实例测试_get_legacy_logger"""
        legacy_logger = self.logger._get_legacy_logger(
            "test", "test.log", without_formater=False, print_to_console=True
        )

        # 验证基础属性
        self.assertTrue(legacy_logger.name.startswith("legacy."))
        self.assertEqual(legacy_logger.level, logging.INFO)

        # 验证handler
        self.assertEqual(len(legacy_logger.handlers), 3)  # 文件+错误+控制台

    def test_logger_propagate(self):
        """测试日志传播设置"""
        legacy_logger = self.logger._get_legacy_logger("test", "test.log")
        self.assertTrue(legacy_logger.propagate)

    def test_get_trace_logger_basic(self):
        """测试get_trace_logger基础功能"""
        logger = self.logger.get_trace_logger("test_trace", "trace_test.log")

        # 验证基础属性
        self.assertTrue(logger.name.startswith("legacy."))
        self.assertEqual(logger.level, logging.INFO)

        # 验证handler数量
        self.assertEqual(len(logger.handlers), 2)  # 主日志和错误日志

    def test_get_trace_logger_with_console(self):
        """测试带控制台输出的trace logger"""
        logger = self.logger.get_trace_logger("test_trace_console", "trace_console_test.log", print_to_console=True)

        # 验证handler数量
        self.assertEqual(len(logger.handlers), 3)  # 主日志+错误日志+控制台

    def test_get_trace_logger_without_formatter(self):
        """测试不带格式化的trace logger"""
        logger = self.logger.get_trace_logger("test_trace_no_fmt", "trace_no_fmt_test.log", without_formater=True)

        # 验证handler是否没有格式化器
        for handler in logger.handlers:
            self.assertIsNone(handler.formatter)

    def test_get_trace_logger_debug_mode(self):
        """测试debug模式下的trace logger"""
        with patch("fastdeploy.envs.FD_DEBUG", "1"):
            logger = self.logger.get_trace_logger("test_trace_debug", "trace_debug_test.log")
            self.assertEqual(logger.level, logging.DEBUG)

    def test_get_trace_logger_directory_creation(self):
        """测试第105行：日志目录创建功能"""
        import os
        from unittest.mock import patch

        # 测试不存在目录的创建
        with tempfile.TemporaryDirectory() as temp_dir:
            test_log_dir = os.path.join(temp_dir, "test_logs")
            with patch("fastdeploy.envs.FD_LOG_DIR", test_log_dir):
                # 确保目录不存在
                self.assertFalse(os.path.exists(test_log_dir))

                # 调用get_trace_logger，应该创建目录
                self.logger.get_trace_logger("test_dir_creation", "test.log")

                # 验证目录已创建
                self.assertTrue(os.path.exists(test_log_dir))
                self.assertTrue(os.path.isdir(test_log_dir))

    def test_get_trace_logger_handler_cleanup(self):
        """测试第126行：handler清理功能"""
        # 先创建一个logger并添加一些handler
        test_logger = logging.getLogger("legacy.test_cleanup")
        initial_handler_count = len(test_logger.handlers)

        # 添加一些测试handler
        test_handler1 = logging.StreamHandler()
        test_handler2 = logging.StreamHandler()
        test_logger.addHandler(test_handler1)
        test_logger.addHandler(test_handler2)

        # 验证handler已添加
        self.assertEqual(len(test_logger.handlers), initial_handler_count + 2)

        # 调用get_trace_logger，应该清理现有handler
        logger = self.logger.get_trace_logger("test_cleanup", "cleanup_test.log")

        # 验证新logger的handler数量（应该是2个：主日志和错误日志）
        self.assertEqual(len(logger.handlers), 2)

    def test_log_file_name_handling_error(self):
        """测试日志文件名处理逻辑"""
        test_cases = [
            ("test", "test_error.log"),
        ]

        for input_name, expected_name in test_cases:
            with self.subTest(input_name=input_name):
                # 创建logger并获取实际处理的文件名
                logger = self.logger.get_trace_logger("test_file_name", input_name)

                # 获取handler中的文件名
                for handler in logger.handlers:
                    if isinstance(handler, LazyFileHandler):
                        actual_name = os.path.basename(handler.filename)
                        self.assertTrue(actual_name.endswith(expected_name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
