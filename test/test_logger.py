"""
unittest 版本
python -m unittest tests.test_logger -v
"""

import logging
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fastdeploy.logger import _get_legacy_logger, _get_unified_logger, get_logger


class LoggerTests(unittest.TestCase):
    """logger 模块单元测试"""

    # -------------------------------------------------
    # 夹具：每个测试独占临时日志目录
    # -------------------------------------------------
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="fd_unittest_")
        # 统一 patch 环境变量
        self.patchers = [
            patch("fastdeploy.envs.FD_LOG_DIR", self.tmp_dir),
            patch("fastdeploy.envs.FD_DEBUG", "0"),
            patch("fastdeploy.envs.FD_LOG_BACKUP_COUNT", "1"),
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # -------------------------------------------------
    # 测试 _get_unified_logger 命名空间
    # -------------------------------------------------
    def test_unified_namespace(self):
        cases = [
            (None, "fastdeploy"),
            ("__main__", "fastdeploy.main"),
            ("fastdeploy.xxx", "fastdeploy.xxx"),
            ("foo", "fastdeploy.foo"),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(_get_unified_logger(inp).name, expected)

    # -------------------------------------------------
    # 测试 legacy logger 控制台输出（patch stderr）
    # -------------------------------------------------
    @patch("sys.stderr", new_callable=lambda: open(os.devnull, "w"))
    def test_legacy_console(self, mock_stderr):
        # 这里简单验证 handler 数量即可；真正颜色在终端可见
        logger = _get_legacy_logger("console_test", "console_test.log", without_formater=False, print_to_console=True)
        # 至少有一个 StreamHandler
        handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        self.assertTrue(len(handlers) >= 1)

    # -------------------------------------------------
    # 测试 legacy logger 关闭格式化器
    # -------------------------------------------------
    def test_legacy_without_formatter(self):
        logger = _get_legacy_logger("no_fmt", "no_fmt.log", without_formater=True)
        logger.info("no fmt")
        with open(os.path.join(self.tmp_dir, "no_fmt.log")) as f:
            line = f.read().strip()
        self.assertEqual(line, "no fmt")

    # -------------------------------------------------
    # 测试 DEBUG 级别开关
    # -------------------------------------------------
    def test_legacy_debug_level(self):
        with patch("fastdeploy.envs.FD_DEBUG", "1"):
            logger = _get_legacy_logger("debug", "debug.log")
            self.assertEqual(logger.level, logging.DEBUG)
            logger.debug("debug msg")
            with open(os.path.join(self.tmp_dir, "debug.log")) as f:
                self.assertIn("debug msg", f.read())

    # -------------------------------------------------
    # 测试 get_logger 分支选择
    # -------------------------------------------------
    def test_get_logger_branch(self):
        # 只给 name -> unified
        unified_logger = get_logger("foo")
        self.assertEqual(unified_logger.name, "fastdeploy.foo")

        # 给了 file_name -> legacy
        legacy_logger = get_logger("foo", "foo.log")
        self.assertTrue(legacy_logger.name.startswith("legacy."))


if __name__ == "__main__":
    unittest.main()
