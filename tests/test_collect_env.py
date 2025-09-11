import unittest
from unittest.mock import MagicMock, patch

import fastdeploy.collect_env as collect_env


class TestCollectEnv(unittest.TestCase):
    def setUp(self):
        self.run_lambda = MagicMock()
        self.run_lambda.return_value = (0, "test output", "")

    def test_run(self):
        result = collect_env.run("echo test")
        self.assertIsInstance(result, tuple)

    def test_run_and_read_all(self):
        result = collect_env.run_and_read_all(self.run_lambda, "test command")
        self.assertEqual(result, "test output")

    def test_run_and_parse_first_match(self):
        self.run_lambda.return_value = (0, "version 1.0", "")
        result = collect_env.run_and_parse_first_match(self.run_lambda, "test command", r"version (.*)")
        self.assertEqual(result, "1.0")

    def test_run_and_return_first_line(self):
        self.run_lambda.return_value = (0, "line1\nline2", "")
        result = collect_env.run_and_return_first_line(self.run_lambda, "test command")
        self.assertEqual(result, "line1")

    def test_get_conda_packages(self):
        with patch("fastdeploy.collect_env.run_and_read_all") as mock_read:
            mock_read.return_value = "package1\npackage2"
            result = collect_env.get_conda_packages(self.run_lambda)
            self.assertIsNotNone(result)

    def test_get_gcc_version(self):
        with patch("fastdeploy.collect_env.run_and_parse_first_match") as mock_parse:
            mock_parse.return_value = "1.0"
            result = collect_env.get_gcc_version(self.run_lambda)
            self.assertEqual(result, "1.0")

    def test_get_clang_version(self):
        with patch("fastdeploy.collect_env.run_and_parse_first_match") as mock_parse:
            mock_parse.return_value = "1.0"
            result = collect_env.get_clang_version(self.run_lambda)
            self.assertEqual(result, "1.0")

    def test_get_cmake_version(self):
        with patch("fastdeploy.collect_env.run_and_parse_first_match") as mock_parse:
            mock_parse.return_value = "1.0"
            result = collect_env.get_cmake_version(self.run_lambda)
            self.assertEqual(result, "1.0")

    def test_get_nvidia_driver_version(self):
        with patch("fastdeploy.collect_env.run_and_parse_first_match") as mock_parse:
            mock_parse.return_value = "1.0"
            result = collect_env.get_nvidia_driver_version(self.run_lambda)
            self.assertEqual(result, "1.0")

    def test_get_gpu_info(self):
        with patch("fastdeploy.collect_env.TORCH_AVAILABLE", False):
            result = collect_env.get_gpu_info(self.run_lambda)
            self.assertIsNotNone(result)

    def test_get_running_cuda_version(self):
        with patch("fastdeploy.collect_env.run_and_parse_first_match") as mock_parse:
            mock_parse.return_value = "1.0"
            result = collect_env.get_running_cuda_version(self.run_lambda)
            self.assertEqual(result, "1.0")

    def test_get_cudnn_version(self):
        with (
            patch("fastdeploy.collect_env.run") as mock_run,
            patch("fastdeploy.collect_env.get_platform", return_value="linux"),
        ):
            mock_run.return_value = (0, "/usr/local/cuda/lib64/libcudnn.so.8.4.1", "")
            result = collect_env.get_cudnn_version(self.run_lambda)
            self.assertEqual(result, None)

    def test_get_nvidia_smi(self):
        result = collect_env.get_nvidia_smi()
        self.assertIsNotNone(result)

    def test_get_fastdeploy_version(self):
        with patch("fastdeploy.collect_env.os.environ.get", return_value="1.0"):
            result = collect_env.get_fastdeploy_version()
            self.assertEqual(result, "1.0")

    def test_summarize_fastdeploy_build_flags(self):
        result = collect_env.summarize_fastdeploy_build_flags()
        self.assertIsNotNone(result)

    def test_get_gpu_topo(self):
        result = collect_env.get_gpu_topo(self.run_lambda)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
