import io
import os
import sys
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

    def test_run_nvidia(self):
        result = collect_env.run("nvidia-smi topo -m")
        self.assertIsInstance(result, tuple)

    def test_run_and_read_all(self):
        result = collect_env.run_and_read_all(self.run_lambda, "test command")
        self.assertEqual(result, "test output")
        self.run_lambda.return_value = (1, "version 1.0", "")
        result = collect_env.run_and_read_all(self.run_lambda, "test command")
        self.assertEqual(result, None)

    def test_run_and_parse_first_match(self):
        self.run_lambda.return_value = (0, "version 1.0", "")
        result = collect_env.run_and_parse_first_match(self.run_lambda, "test command", r"version (.*)")
        self.assertEqual(result, "1.0")
        self.run_lambda.return_value = (1, "version 1.0", "")
        result = collect_env.run_and_parse_first_match(self.run_lambda, "test command", r"version (.*)")
        self.assertEqual(result, None)
        self.run_lambda.return_value = (0, "version 1.0", "")
        result = collect_env.run_and_parse_first_match(self.run_lambda, "test command", r"sadsad")
        self.assertEqual(result, None)

    def test_run_and_return_first_line(self):
        self.run_lambda.return_value = (0, "line1\nline2", "")
        result = collect_env.run_and_return_first_line(self.run_lambda, "test command")
        self.assertEqual(result, "line1")
        self.run_lambda.return_value = (1, "line1\nline2", "")
        result = collect_env.run_and_return_first_line(self.run_lambda, "test command")
        self.assertEqual(result, None)

    def test_get_conda_packages(self):
        with patch("fastdeploy.collect_env.run_and_read_all") as mock_read:
            mock_read.return_value = "package1\npackage2"
            result = collect_env.get_conda_packages(self.run_lambda)
            self.assertIsNotNone(result)
        with patch("fastdeploy.collect_env.run_and_read_all") as mock_read:
            mock_read.return_value = None
            result = collect_env.get_conda_packages(self.run_lambda)
            self.assertIsNone(result)

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
        with (
            patch("fastdeploy.collect_env.run_and_parse_first_match", return_value="1.0"),
            patch("fastdeploy.collect_env.get_platform", return_value="darwin"),
        ):
            result = collect_env.get_nvidia_driver_version(self.run_lambda)
            self.assertEqual(result, "1.0")

    def test_get_gpu_info(self):
        with patch("fastdeploy.collect_env.TORCH_AVAILABLE", False):
            result = collect_env.get_gpu_info(self.run_lambda)
            self.assertIsNotNone(result)
        with (
            patch("fastdeploy.collect_env.get_platform", return_value="darwin"),
            patch("fastdeploy.collect_env.TORCH_AVAILABLE", True),
            patch("fastdeploy.collect_env.torch", create=True),
        ):
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
        with (
            patch("fastdeploy.collect_env.run") as mock_run,
            patch("fastdeploy.collect_env.get_platform", return_value="win32"),
        ):
            mock_run.return_value = (0, "/usr/local/cuda/lib64/libcudnn.so.8.4.1", "")
            result = collect_env.get_cudnn_version(self.run_lambda)
            self.assertEqual(result, None)
        with (
            patch("fastdeploy.collect_env.run") as mock_run,
            patch("fastdeploy.collect_env.get_platform", return_value="darwin"),
        ):
            mock_run.return_value = (0, "/usr/local/cuda/lib64/libcudnn.so.8.4.1", "")
            result = collect_env.get_cudnn_version(self.run_lambda)
            self.assertEqual(result, None)
        with (
            patch("fastdeploy.collect_env.run") as mock_run,
            patch("fastdeploy.collect_env.get_platform", return_value="darwin"),
        ):
            mock_run.return_value = (2, "/usr/local/cuda/lib64/libcudnn.so.8.4.1\n/usr/xxx", "")
            self.run_lambda.return_value = (2, "version 1.0", "")
            result = collect_env.get_cudnn_version(self.run_lambda)
            self.assertEqual(result, None)

        with (
            patch("os.path.realpath", side_effect=lambda x: f"/real_path/to/{x.split('/')[-1]}"),
            patch("os.path.isfile", return_value=True),
        ):
            self.run_lambda.return_value = (
                0,
                "/usr/local/cuda/lib/libcudnn.so.8\n/usr/local/cuda/lib/libcudnn.so.8.2.1",
                "",
            )
            cudnn_version = collect_env.get_cudnn_version(self.run_lambda)
            # 验证返回结果是预期的多行字符串
            expected_output = (
                "Probably one of the following:\n/real_path/to/libcudnn.so.8\n/real_path/to/libcudnn.so.8.2.1"
            )
            self.assertEqual(cudnn_version, expected_output)

    def test_get_nvidia_smi(self):
        result = collect_env.get_nvidia_smi()
        self.assertIsNotNone(result)
        with patch("fastdeploy.collect_env.get_platform", return_value="win32"):
            result = collect_env.get_nvidia_smi()
            self.assertIsNotNone(result)

    def test_get_fastdeploy_version(self):
        with patch("fastdeploy.collect_env.os.environ.get", return_value="1.0"):
            result = collect_env.get_fastdeploy_version()
            self.assertEqual(result, "1.0")
        with patch("fastdeploy.collect_env.os.environ.get", return_value=None):
            result = collect_env.get_fastdeploy_version()
            self.assertIsNotNone(result)
        with patch("pkg_resources.get_distribution", side_effect=Exception("Package not found")):
            with patch("fastdeploy.collect_env.os.environ.get", return_value=None):
                with patch("subprocess.run", return_value=None):
                    result = collect_env.get_fastdeploy_version()
                    self.assertIsNotNone(result)

    def test_summarize_fastdeploy_build_flags(self):
        result = collect_env.summarize_fastdeploy_build_flags()
        self.assertIsNotNone(result)

    def test_get_gpu_topo(self):
        result = collect_env.get_gpu_topo(self.run_lambda)
        self.assertIsNotNone(result)

    def test_get_cpu_info(self):
        self.run_lambda.return_value = (0, "Architecture: x86_64\nModel name: Intel(R) Xeon(R) CPU", "")
        with patch("fastdeploy.collect_env.get_platform", return_value="linux"):
            cpu_info = collect_env.get_cpu_info(self.run_lambda)
            self.assertIn("x86_64", cpu_info)
            self.assertIn("Intel(R) Xeon(R)", cpu_info)
        with patch("fastdeploy.collect_env.get_platform", return_value="win32"):
            cpu_info = collect_env.get_cpu_info(self.run_lambda)
            self.assertIn("x86_64", cpu_info)
            self.assertIn("Intel(R) Xeon(R)", cpu_info)
        with patch("fastdeploy.collect_env.get_platform", return_value="darwin"):
            cpu_info = collect_env.get_cpu_info(self.run_lambda)
            self.assertIn("x86_64", cpu_info)
            self.assertIn("Intel(R) Xeon(R)", cpu_info)
        self.run_lambda.return_value = (1, "Architecture: x86_64\nModel name: Intel(R) Xeon(R) CPU", "err")
        with patch("fastdeploy.collect_env.get_platform", return_value="darwin"):
            cpu_info = collect_env.get_cpu_info(self.run_lambda)
            self.assertIn("err", cpu_info)

    def test_get_platform(self):
        with patch("sys.platform", "linux"):
            self.assertEqual(collect_env.get_platform(), "linux")
        with patch("sys.platform", "win32"):
            self.assertEqual(collect_env.get_platform(), "win32")
        with patch("sys.platform", "cygwin"):
            self.assertEqual(collect_env.get_platform(), "cygwin")
        with patch("sys.platform", "darwin"):
            self.assertEqual(collect_env.get_platform(), "darwin")

    def test_get_os_linux_lsb_success(self):
        """测试 Linux 环境下，lsb_release 命令成功。"""
        with patch("sys.platform", "linux"):
            # 模拟 get_lsb_version 成功返回
            with patch("fastdeploy.collect_env.get_lsb_version", return_value="Ubuntu 20.04 LTS"):
                # 模拟 platform.machine 成功返回
                with patch("platform.machine", return_value="x86_64"):
                    result = collect_env.get_os(self.run_lambda)
                    self.assertEqual(result, "Ubuntu 20.04 LTS (x86_64)")

    def test_get_os_linux_lsb_fail_check_release_success(self):
        """测试 Linux 环境下，lsb_release 失败，但 /etc/*-release 成功。"""
        with patch("sys.platform", "linux"):
            # 模拟 get_lsb_version 失败
            with patch("fastdeploy.collect_env.get_lsb_version", return_value=None):
                # 模拟 check_release_file 成功返回
                with patch("fastdeploy.collect_env.check_release_file", return_value="CentOS Linux 8"):
                    with patch("platform.machine", return_value="x86_64"):
                        result = collect_env.get_os(self.run_lambda)
                        self.assertEqual(result, "CentOS Linux 8 (x86_64)")

    def test_get_os_linux_all_fail(self):
        """测试 Linux 环境下，所有方法都失败。"""
        with patch("sys.platform", "linux"):
            with patch("fastdeploy.collect_env.get_lsb_version", return_value=None):
                with patch("fastdeploy.collect_env.check_release_file", return_value=None):
                    with patch("platform.machine", return_value="x86_64"):
                        result = collect_env.get_os(self.run_lambda)
                        self.assertEqual(result, "linux (x86_64)")

    def test_get_os_windows_success(self):
        """测试 Windows 环境下，命令成功。"""
        with patch("sys.platform", "win32"):
            # 模拟 get_windows_version 成功返回
            with patch("fastdeploy.collect_env.get_windows_version", return_value="Microsoft Windows 10"):
                result = collect_env.get_os(self.run_lambda)
                self.assertEqual(result, "Microsoft Windows 10")

    def test_get_os_windows_fail(self):
        """测试 Windows 环境下，命令失败。"""
        with patch("sys.platform", "win32"):
            # 模拟 get_windows_version 失败返回 None
            with patch("fastdeploy.collect_env.get_windows_version", return_value=None):
                result = collect_env.get_os(self.run_lambda)
                self.assertIsNone(result)

    def test_get_os_macos_success(self):
        """测试 macOS 环境下，命令成功。"""
        with patch("sys.platform", "darwin"):
            # 模拟 get_mac_version 成功返回
            with patch("fastdeploy.collect_env.get_mac_version", return_value="12.3.1"):
                with patch("platform.machine", return_value="arm64"):
                    result = collect_env.get_os(self.run_lambda)
                    self.assertEqual(result, "macOS 12.3.1 (arm64)")

    def test_get_os_macos_fail(self):
        """测试 macOS 环境下，命令失败。"""
        with patch("sys.platform", "darwin"):
            # 模拟 get_mac_version 失败返回 None
            with patch("fastdeploy.collect_env.get_mac_version", return_value=None):
                result = collect_env.get_os(self.run_lambda)
                self.assertIsNone(result)

    def test_get_os_unknown_platform(self):
        """测试未知平台的情况。"""
        with patch("sys.platform", "solaris"):
            result = collect_env.get_os(self.run_lambda)
            self.assertEqual(result, "solaris")

    def test_get_python_platform(self):
        """测试 get_python_platform 函数返回正确值。"""
        with patch("platform.platform", return_value="Linux-5.15.0-76-generic-x86_64-with-glibc2.35"):
            result = collect_env.get_python_platform()
            self.assertEqual(result, "Linux-5.15.0-76-generic-x86_64-with-glibc2.35")

    def test_get_libc_version_linux_success(self):
        """测试在 Linux 环境下成功获取 libc 版本。"""
        with patch("fastdeploy.collect_env.get_platform", return_value="linux"):
            with patch("platform.libc_ver", return_value=("glibc", "2.35")):
                result = collect_env.get_libc_version()
                self.assertEqual(result, "glibc-2.35")

    def test_get_libc_version_non_linux(self):
        """测试在非 Linux 环境下返回 'N/A'。"""
        with patch("fastdeploy.collect_env.get_platform", return_value="win32"):
            # 确保 platform.libc_ver() 不被调用，或者即使调用也不会影响结果
            with patch("platform.libc_ver") as mock_libc_ver:
                result = collect_env.get_libc_version()
                self.assertEqual(result, "N/A")
                mock_libc_ver.assert_not_called()

    def test_get_pip_packages_no_pip_or_uv(self):
        """Test that a RuntimeError is raised when neither pip nor uv are available."""
        with patch.dict("os.environ", {}, clear=True), patch("importlib.util.find_spec", return_value=None):

            with self.assertRaises(RuntimeError) as cm:
                collect_env.get_pip_packages(self.run_lambda)
            self.assertIn("Could not collect pip list output", str(cm.exception))

    def test_get_pip_packages_success(self):
        """Test that the pip module is available and a list of packages is returned."""
        with (
            patch("fastdeploy.collect_env.run_and_read_all") as mock_run_and_read_all,
            patch("importlib.util.find_spec", return_value=True),
        ):

            mock_run_and_read_all.return_value = "torch==2.0.0\nregex==2023.1.1\nnumpy==1.25.0"

            pip_version, packages = collect_env.get_pip_packages(self.run_lambda)

            self.assertEqual(pip_version, "pip3" if sys.version[0] == "3" else "pip")
            self.assertIn("torch==2.0.0", packages)
            self.assertIn("numpy==1.25.0", packages)
            self.assertNotIn("regex", packages)

    def test_get_pip_packages_uv_available(self):
        """Test that uv is used when pip is not available but the UV environment variable is set."""
        with (
            patch.dict("os.environ", {"UV": "1"}),
            patch("fastdeploy.collect_env.run_and_read_all") as mock_run_and_read_all,
            patch("importlib.util.find_spec", return_value=False),
        ):

            mock_run_and_read_all.return_value = "torch==2.0.0\nregex==2023.1.1\nnumpy==1.25.0"

            pip_version, packages = collect_env.get_pip_packages(self.run_lambda)

            self.assertIsNotNone(packages)

    def test_get_pip_packages_command_fail(self):
        """Test that an empty string is returned when the pip command fails."""
        with (
            patch("fastdeploy.collect_env.run_and_read_all", return_value="\n"),
            patch("importlib.util.find_spec", return_value=True),
        ):

            pip_version, packages = collect_env.get_pip_packages(self.run_lambda)
            self.assertEqual(packages, "")

    def test_get_pip_packages_custom_patterns(self):
        """Test that the function correctly filters packages based on custom patterns."""
        with (
            patch("fastdeploy.collect_env.run_and_read_all") as mock_run_and_read_all,
            patch("importlib.util.find_spec", return_value=True),
        ):

            mock_run_and_read_all.return_value = "torch==2.0.0\nmy-custom-lib==1.0.0\nrequests==2.28.1"

            custom_patterns = {"my-custom-lib", "requests"}
            pip_version, packages = collect_env.get_pip_packages(self.run_lambda, patterns=custom_patterns)

            self.assertIn("my-custom-lib==1.0.0", packages)
            self.assertIn("requests==2.28.1", packages)
            self.assertNotIn("torch", packages)

    def test_xnnpack_available_with_torch(self):
        with patch("fastdeploy.collect_env.TORCH_AVAILABLE", True), patch("fastdeploy.collect_env.torch", create=True):
            result = collect_env.is_xnnpack_available()
            self.assertIsNotNone(result)

    def test_xnnpack_not_available_without_torch(self):
        """测试 torch 不可用时，返回 'N/A'。"""
        with patch("fastdeploy.collect_env.TORCH_AVAILABLE", False):
            result = collect_env.is_xnnpack_available()
            self.assertEqual(result, "N/A")

    def test_get_env_vars_with_relevant_vars(self):
        """测试正确收集相关环境变量。"""
        # 准备一个包含各种类型环境变量的字典
        mock_env = {
            "TORCH_DEBUG": "1",
            "CUDA_ARCHS": "7.5",
            "SOME_OTHER_VAR": "value",
            "MY_API_KEY": "secret_key",
            "FASTDEPLOY_MODEL_DIR": "/path/to/model",  # 假设这个在 environment_variables 中
        }

        # 模拟 environment_variables 列表
        with patch.dict(os.environ, mock_env, clear=True):
            with patch(
                "fastdeploy.collect_env.environment_variables",
                ["FASTDEPLOY_MODEL_DIR"],
            ):
                env_vars_string = collect_env.get_env_vars()
                self.assertIn("TORCH_DEBUG=1", env_vars_string)
                self.assertIn("CUDA_ARCHS=7.5", env_vars_string)
                self.assertIn("FASTDEPLOY_MODEL_DIR=/path/to/model", env_vars_string)
                self.assertNotIn("SOME_OTHER_VAR", env_vars_string)
                self.assertNotIn("MY_API_KEY", env_vars_string)

    def test_get_cuda_config_with_both_vars_set(self):
        with patch("fastdeploy.collect_env.TORCH_AVAILABLE", True), patch("fastdeploy.collect_env.torch", create=True):
            mock_env = {
                "CUDA_MODULE_LOADING": "xxx",
            }
            with patch.dict(os.environ, mock_env, clear=True):
                result = collect_env.get_cuda_module_loading_config()
                self.assertEqual(result, "xxx")

    def test_get_cuda_config_with_no_vars_set(self):
        """测试两个环境变量都未设置。"""
        with patch("fastdeploy.collect_env.TORCH_AVAILABLE", False):
            result = collect_env.get_cuda_module_loading_config()
            self.assertEqual(result, "N/A")

    def test_get_env_info_full(self):

        # 使用嵌套的 with patch 语句来模拟所有依赖函数的返回值
        with (
            patch("fastdeploy.collect_env.get_platform", return_value="linux"),
            patch("fastdeploy.collect_env.get_os", return_value="Ubuntu 20.04 (x86_64)"),
            patch("fastdeploy.collect_env.get_python_platform", return_value="Python 3.8.10"),
            patch("fastdeploy.collect_env.get_cuda_module_loading_config", return_value="CUDA_DEVICE_VISIBLE=0"),
            patch("fastdeploy.collect_env.get_libc_version", return_value="glibc-2.31"),
            patch("fastdeploy.collect_env.get_gcc_version", return_value="9.3.0"),
            patch("fastdeploy.collect_env.get_clang_version", return_value=None),
            patch("fastdeploy.collect_env.get_cmake_version", return_value="3.18.4"),
            patch("fastdeploy.collect_env.get_nvidia_driver_version", return_value="470.82.00"),
            patch("fastdeploy.collect_env.get_cudnn_version", return_value="8.2.1"),
            patch("fastdeploy.collect_env.get_running_cuda_version", return_value="11.4"),
            patch("fastdeploy.collect_env.get_gpu_info", return_value="GeForce RTX 3080"),
            patch("fastdeploy.collect_env.get_cpu_info", return_value="Intel(R) Core(TM) i9-10900K"),
            patch("fastdeploy.collect_env.is_xnnpack_available", return_value="True"),
            patch("fastdeploy.collect_env.get_fastdeploy_version", return_value="1.0.0"),
            patch("fastdeploy.collect_env.get_conda_packages", return_value="numpy==1.22.0\ntorch==1.11.0"),
            patch("fastdeploy.collect_env.get_pip_packages", return_value=("pip3", "requests==2.27.1\nscipy==1.8.0")),
            patch("fastdeploy.collect_env.get_env_vars", return_value="CUDA_VISIBLE_DEVICES=0"),
            patch("fastdeploy.collect_env.get_gpu_topo", return_value="GPU-Direct disabled"),
            patch("fastdeploy.collect_env.torch", create=True, __version__="1.0.0"),
        ):

            info_string = collect_env.get_env_info()

            self.assertIsNotNone(info_string)

    def test_get_env_info_all_na(self):

        with (
            patch("fastdeploy.collect_env.get_python_platform", return_value="N/A"),
            patch("fastdeploy.collect_env.get_cuda_module_loading_config", return_value="N/A"),
            patch("fastdeploy.collect_env.get_libc_version", return_value="N/A"),
            patch("fastdeploy.collect_env.get_gcc_version", return_value="N/A"),
            patch("fastdeploy.collect_env.get_clang_version", return_value="N/A"),
            patch("fastdeploy.collect_env.get_cmake_version", return_value="N/A"),
            patch("fastdeploy.collect_env.get_nvidia_driver_version", return_value="N/A"),
            patch("fastdeploy.collect_env.get_cudnn_version", return_value="N/A"),
            patch("fastdeploy.collect_env.get_running_cuda_version", return_value="N/A"),
            patch("fastdeploy.collect_env.get_gpu_info", return_value="N/A"),
            patch("fastdeploy.collect_env.get_cpu_info", return_value="N/A"),
            patch("fastdeploy.collect_env.is_xnnpack_available", return_value="N/A"),
            patch("fastdeploy.collect_env.get_fastdeploy_version", return_value="N/A"),
            patch("fastdeploy.collect_env.get_conda_packages", return_value=None),
            patch("fastdeploy.collect_env.get_pip_packages", return_value=("N/A", "N/A")),
            patch("fastdeploy.collect_env.get_env_vars", return_value=""),
            patch("fastdeploy.collect_env.get_gpu_topo", return_value="N/A"),
            patch("fastdeploy.collect_env.TORCH_AVAILABLE", return_value=False),
            patch("fastdeploy.collect_env.PADDLE_AVAILABLE", return_value=False),
            patch("fastdeploy.collect_env.torch", create=True, __version__="1.0.0"),
        ):

            info_string = collect_env.get_env_info()

            self.assertIsNotNone(info_string)

    def test_main_with_collect(self):
        captured_output = io.StringIO()
        with (
            patch("sys.stdout", new=captured_output),
            patch("fastdeploy.collect_env.torch", create=True, __version__="1.0.0"),
            patch("fastdeploy.collect_env.hasattr", return_value=False),
        ):
            collect_env.main()
        output = captured_output.getvalue()
        expected_message = "Collecting environment information"
        self.assertIn(expected_message, output)


if __name__ == "__main__":
    unittest.main()
