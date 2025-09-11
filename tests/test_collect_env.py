import unittest
from unittest.mock import patch

from fastdeploy.collect_env import (
    SystemEnv,
    get_clang_version,
    get_cmake_version,
    get_fastdeploy_version,
    get_gcc_version,
    get_nvidia_driver_version,
    main,
    run,
)


class TestCollectEnv(unittest.TestCase):
    def test_run_success(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.communicate.return_value = (b"output", b"error")
            mock_popen.return_value.returncode = 0
            rc, out, err = run("echo hello")
            self.assertEqual(rc, 0)
            self.assertEqual(out, "output")
            self.assertEqual(err, "error")

    def test_run_command_not_found(self):
        rc, out, err = run("nonexistent_command")
        self.assertEqual(rc, 127)
        self.assertEqual(out, "")

    @patch("fastdeploy.collect_env.run_and_parse_first_match")
    def test_get_gcc_version(self, mock_run):
        mock_run.return_value = "9.4.0"
        version = get_gcc_version(lambda x: (0, "gcc (Ubuntu 9.4.0-1ubuntu1~20.04) 9.4.0", ""))
        self.assertEqual(version, "9.4.0")

    @patch("fastdeploy.collect_env.run_and_parse_first_match")
    def test_get_clang_version(self, mock_run):
        mock_run.return_value = "10.0.0"
        version = get_clang_version(lambda x: (0, "clang version 10.0.0-4ubuntu1", ""))
        self.assertEqual(version, "10.0.0")

    @patch("fastdeploy.collect_env.run_and_parse_first_match")
    def test_get_cmake_version(self, mock_run):
        mock_run.return_value = "3.16.3"
        version = get_cmake_version(lambda x: (0, "cmake version 3.16.3", ""))
        self.assertEqual(version, "3.16.3")

    @patch("fastdeploy.collect_env.run_and_parse_first_match")
    def test_get_nvidia_driver_version(self, mock_run):
        mock_run.return_value = "470.182.03"
        version = get_nvidia_driver_version(lambda x: (0, "Driver Version: 470.182.03", ""))
        self.assertEqual(version, "470.182.03")

    @patch("fastdeploy.collect_env.subprocess.run")
    def test_get_fastdeploy_version(self, mock_run):
        mock_run.return_value.stdout = "Version: 1.0.0\n"
        version = get_fastdeploy_version()
        self.assertTrue(version != "unknown")

    def test_system_env_namedtuple(self):
        env = SystemEnv(
            torch_version="2.0.0",
            is_debug_build=False,
            cuda_compiled_version="11.7",
            paddle_version="2.4.0",
            cuda_compiled_version_paddle="11.7",
            gcc_version="9.4.0",
            clang_version="10.0.0",
            cmake_version="3.16.3",
            os="Linux",
            libc_version="2.31",
            python_version="3.8.10",
            python_platform="linux",
            is_cuda_available=True,
            cuda_runtime_version="11.7",
            cuda_module_loading="LAZY",
            nvidia_driver_version="470.182.03",
            nvidia_gpu_models="NVIDIA GeForce RTX 3090",
            cudnn_version="8.5.0",
            pip_version="22.0.2",
            pip_packages="torch==2.0.0",
            conda_packages="numpy",
            is_xnnpack_available=True,
            cpu_info="Intel(R) Xeon(R) CPU",
            fastdeploy_version="1.0.0",
            fastdeploy_build_flags="CUDA Archs: []",
            gpu_topo=None,
            env_vars={},
        )
        self.assertEqual(env.torch_version, "2.0.0")
        self.assertEqual(env.paddle_version, "2.4.0")

    @patch("fastdeploy.collect_env.get_pretty_env_info")
    def test_main(self, mock_pretty_env):
        # Just call main function to increase coverage
        main()
        self.assertTrue(mock_pretty_env.called)


if __name__ == "__main__":
    unittest.main()
