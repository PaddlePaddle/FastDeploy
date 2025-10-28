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
import shutil
import tempfile
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
