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

from fastdeploy.logger.setup_logging import setup_logging


class TestSetupLogging(unittest.TestCase):

    # -------------------------------------------------
    # 夹具：每个测试独占临时目录
    # -------------------------------------------------
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="logger_setup_test_")
        # 统一 patch 环境变量
        self.patches = [
            patch("fastdeploy.envs.FD_LOG_DIR", self.temp_dir),
            patch("fastdeploy.envs.FD_DEBUG", "0"),
            patch("fastdeploy.envs.FD_LOG_BACKUP_COUNT", "3"),
        ]
        [p.start() for p in self.patches]

    def tearDown(self):
        [p.stop() for p in self.patches]
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # 清理单例标记，避免影响其他测试
        if hasattr(setup_logging, "_configured"):
            delattr(setup_logging, "_configured")

    # -------------------------------------------------
    # 基础：目录自动创建
    # -------------------------------------------------
    def test_log_dir_created(self):
        nested = os.path.join(self.temp_dir, "a", "b", "c")
        setup_logging(log_dir=nested)
        self.assertTrue(Path(nested).is_dir())

    # -------------------------------------------------
    # 默认配置文件：文件 handler 不带颜色
    # -------------------------------------------------
    def test_default_config_file_no_ansi(self):
        setup_logging()
        logger = logging.getLogger("fastdeploy")
        logger.error("test ansi")

        default_file = Path(self.temp_dir) / "default.log"
        self.assertTrue(default_file.exists())
        with default_file.open() as f:
            content = f.read()
        # 文件中不应出现 ANSI 转义
        self.assertNotIn("\033[", content)

    # -------------------------------------------------
    # 调试级别开关
    # -------------------------------------------------
    def test_debug_level(self):
        with patch("fastdeploy.envs.FD_DEBUG", "1"):
            setup_logging()
            logger = logging.getLogger("fastdeploy")
            self.assertEqual(logger.level, logging.DEBUG)
            # debug 消息应该能落到文件
            logger.debug("debug msg")
            default_file = Path(self.temp_dir) / "default.log"
            self.assertIn("debug msg", default_file.read_text())

    # -------------------------------------------------
    # 自定义 JSON 配置文件加载
    # -------------------------------------------------
    def test_custom_config_file(self):
        custom_cfg = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"plain": {"format": "%(message)s"}},
            "handlers": {
                "custom": {
                    "class": "logging.FileHandler",
                    "filename": os.path.join(self.temp_dir, "custom.log"),
                    "formatter": "plain",
                }
            },
            "loggers": {"fastdeploy": {"handlers": ["custom"], "level": "INFO"}},
        }
        cfg_path = Path(self.temp_dir) / "cfg.json"
        cfg_path.write_text(json.dumps(custom_cfg))

        setup_logging(config_file=str(cfg_path))
        logger = logging.getLogger("fastdeploy")
        logger.info("from custom cfg")

        custom_file = Path(self.temp_dir) / "custom.log"
        self.assertEqual(custom_file.read_text().strip(), "from custom cfg")

    # -------------------------------------------------
    # 重复调用 setup_logging 不会重复配置
    # -------------------------------------------------
    def test_configure_once(self):
        logger1 = setup_logging()
        logger2 = setup_logging()
        self.assertIs(logger1, logger2)

    # -------------------------------------------------
    # 控制台 handler 使用 ColoredFormatter
    # -------------------------------------------------
    @patch("logging.StreamHandler.emit")
    def test_console_colored(self, mock_emit):
        setup_logging()
        logger = logging.getLogger("fastdeploy")
        logger.error("color test")
        # 只要 ColoredFormatter 被实例化即可，简单断言 emit 被调用
        self.assertTrue(mock_emit.called)


if __name__ == "__main__":
    unittest.main()
