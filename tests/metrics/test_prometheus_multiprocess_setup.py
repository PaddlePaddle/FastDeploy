import os
import shutil
from unittest.mock import patch

from fastdeploy.metrics.prometheus_multiprocess_setup import (
    setup_multiprocess_prometheus,
)


class TestSetupMultiprocessPrometheus:

    def test_setup_with_existing_dir(self, tmp_path):
        """测试当目录已存在时清理并重新创建"""
        # 确保环境变量不存在
        if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
            del os.environ["PROMETHEUS_MULTIPROC_DIR"]

        test_dir = "/tmp/prom_main_test-uuid"
        # 使用 tmp_path 创建临时目录
        os.makedirs(test_dir, exist_ok=True)

        with (
            patch("uuid.uuid4", return_value="test-uuid"),
            patch("fastdeploy.utils.console_logger.info") as mock_logger,
        ):

            result = setup_multiprocess_prometheus()

            assert result == test_dir
            assert os.path.exists(test_dir)
            mock_logger.assert_called_once_with("PROMETHEUS_MULTIPROC_DIR is set to be /tmp/prom_main_test-uuid")

    def test_when_env_var_already_set(self):
        """测试当环境变量已设置时的情况"""
        test_dir = "/tmp/existing_dir"
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = test_dir

        with patch("fastdeploy.utils.console_logger.warning") as mock_logger:
            result = setup_multiprocess_prometheus()

            assert result == test_dir
            mock_logger.assert_called_once_with(
                "Found PROMETHEUS_MULTIPROC_DIR:/tmp/existing_dir was set by user. "
                "you will find inaccurate metrics. Unset the variable "
                "will properly handle cleanup."
            )

    def test_cleanup_failure_handling(self):
        """测试清理目录失败时的处理"""
        if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
            del os.environ["PROMETHEUS_MULTIPROC_DIR"]

        with (
            patch("os.path.exists", return_value=True),
            patch("uuid.uuid4", return_value="test-uuid"),
            patch("fastdeploy.utils.console_logger.info") as mock_logger,
        ):

            # 模拟 rmtree 但确保 ignore_errors=True 生效
            original_rmtree = shutil.rmtree

            def mock_rmtree(path, ignore_errors=False):
                if ignore_errors:
                    return  # 忽略错误
                original_rmtree(path)

            with patch("shutil.rmtree", side_effect=mock_rmtree):
                result = setup_multiprocess_prometheus()

                assert result == "/tmp/prom_main_test-uuid"
                mock_logger.assert_called_once_with("PROMETHEUS_MULTIPROC_DIR is set to be /tmp/prom_main_test-uuid")
