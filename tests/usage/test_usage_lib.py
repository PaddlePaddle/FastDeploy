"""
Unit tests for usage_lib.py
"""

import json
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, mock_open, patch

from requests.exceptions import RequestException

from fastdeploy.usage.usage_lib import (
    _GLOBAL_RUNTIME_DATA,
    UsageMessage,
    cuda_device_count,
    cuda_get_device_properties,
    cuda_is_initialized,
    detect_cloud_provider,
    get_cuda_version,
    get_current_timestamp_ns,
    get_xpu_model,
    is_usage_stats_enabled,
    report_usage_stats,
    set_runtime_usage_data,
    simple_convert,
    xpu_device_count,
)


class TestCudaDeviceProperties(unittest.TestCase):
    """Test cuda_get_device_properties function"""

    @patch("fastdeploy.usage.usage_lib.paddle.device.cuda.get_device_properties")
    def test_cuda_initialized(self, mock_props):
        """Test when CUDA is initialized"""
        mock_obj = MagicMock()
        mock_obj.major = 8
        mock_obj.minor = 6
        mock_obj.name = "A100"
        mock_obj.total_memory = 40 * 1024**3
        mock_obj.multi_processor_count = 108
        mock_props.return_value = mock_obj

        # Test getting all properties
        result = cuda_get_device_properties(
            0, ["major", "minor", "name", "total_memory", "multi_processor_count"], True
        )
        self.assertEqual(result, (8, 6, "A100", 40 * 1024**3, 108))

        # Test getting partial properties
        result = cuda_get_device_properties(0, ["name", "total_memory"], True)
        self.assertEqual(result, ("A100", 40 * 1024**3))


class TestGetXpuModel(unittest.TestCase):
    """Test get_xpu_model function"""

    @patch("fastdeploy.usage.usage_lib.subprocess.run")
    def test_success_with_valid_model(self, mock_run):
        """Test successful command execution with valid model"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "|   0 P900                    On  |"
        mock_run.return_value = mock_result

        result = get_xpu_model()
        self.assertEqual(result, "P900")
        mock_run.assert_called_once_with(["xpu-smi"], capture_output=True, text=True, timeout=5)

    @patch("fastdeploy.usage.usage_lib.subprocess.run")
    def test_command_failure(self, mock_run):
        """Test when command fails"""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        result = get_xpu_model()
        self.assertIsNone(result)

    @patch("fastdeploy.usage.usage_lib.subprocess.run")
    def test_no_matching_pattern(self, mock_run):
        """Test when output doesn't match pattern"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Invalid output format"
        mock_run.return_value = mock_result

        result = get_xpu_model()
        self.assertEqual(result, "P800")

    @patch("fastdeploy.usage.usage_lib.subprocess.run")
    def test_exception_handling(self, mock_run):
        """Test exception handling"""
        mock_run.side_effect = Exception("Command failed")
        result = get_xpu_model()
        self.assertEqual(result, "P800")


class TestGetCudaVersion(unittest.TestCase):
    """Test get_cuda_version function"""

    @patch("fastdeploy.usage.usage_lib.os.popen")
    def test_success(self, mock_popen):
        """Test successful version extraction"""
        mock_popen.return_value.read.return_value = """
        nvcc: NVIDIA (R) Cuda compiler driver
        Cuda compilation tools, release 12.1, V12.1.105
        """
        result = get_cuda_version()
        self.assertEqual(result, "12.1")

    @patch("fastdeploy.usage.usage_lib.os.popen")
    def test_no_match(self, mock_popen):
        """Test when version can't be extracted"""
        mock_popen.return_value.read.return_value = "Invalid output"
        result = get_cuda_version()
        self.assertIsNone(result)

    @patch("fastdeploy.usage.usage_lib.os.popen")
    def test_command_failure(self, mock_popen):
        """Test when command fails"""
        mock_popen.side_effect = Exception("Command failed")
        result = get_cuda_version()
        self.assertIsNone(result)


# Enhanced tests for cuda_device_count and xpu_device_count functions
class TestDeviceCountFunctions(unittest.TestCase):
    """Enhanced tests for device count functions"""

    @patch("fastdeploy.usage.usage_lib.paddle.device.is_compiled_with_cuda")
    @patch("fastdeploy.usage.usage_lib.paddle.device.cuda.device_count")
    def test_cuda_device_count_with_cuda(self, mock_device_count, mock_is_compiled):
        """Test cuda_device_count when CUDA is compiled and available"""
        mock_is_compiled.return_value = True
        mock_device_count.return_value = 4
        result = cuda_device_count()
        self.assertEqual(result, 4)

    @patch("fastdeploy.usage.usage_lib.paddle.device.is_compiled_with_cuda")
    def test_cuda_device_count_without_cuda(self, mock_is_compiled):
        """Test cuda_device_count when CUDA is not compiled"""
        mock_is_compiled.return_value = False
        result = cuda_device_count()
        self.assertEqual(result, 0)

    @patch("fastdeploy.usage.usage_lib.paddle.device.is_compiled_with_xpu")
    @patch("fastdeploy.usage.usage_lib.paddle.device.xpu.device_count")
    def test_xpu_device_count_with_xpu(self, mock_device_count, mock_is_compiled):
        """Test xpu_device_count when XPU is compiled and available"""
        mock_is_compiled.return_value = True
        mock_device_count.return_value = 2
        result = xpu_device_count()
        self.assertEqual(result, 2)

    @patch("fastdeploy.usage.usage_lib.paddle.device.is_compiled_with_xpu")
    def test_xpu_device_count_without_xpu(self, mock_is_compiled):
        """Test xpu_device_count when XPU is not compiled"""
        mock_is_compiled.return_value = False
        result = xpu_device_count()
        self.assertEqual(result, 0)


# Enhanced tests for TestUsageMessage class
class TestUsageMessage(unittest.TestCase):
    """Test UsageMessage class with enhanced coverage"""

    def setUp(self):
        self.usage_message = UsageMessage()

    def tearDown(self):
        # Clean up any global data that might have been modified
        _GLOBAL_RUNTIME_DATA.clear()

    def test_initialization(self):
        """Test UsageMessage initialization"""
        self.assertIsNotNone(self.usage_message.uuid)
        self.assertIsNone(self.usage_message.provider)
        self.assertIsNone(self.usage_message.cpu_num)
        self.assertIsNone(self.usage_message.cpu_type)

    @patch("fastdeploy.usage.usage_lib.Thread")
    @patch("fastdeploy.usage.usage_lib.is_usage_stats_enabled")
    def test_report_usage_disabled(self, mock_is_enabled, mock_thread):
        """Test report_usage when stats are disabled"""
        mock_is_enabled.return_value = False

        # Mock FDConfig
        mock_fd_config = MagicMock()
        mock_fd_config.model_config.quantization = None
        mock_fd_config.model_config.num_hidden_layers = 12
        mock_fd_config.cache_config.block_size = 16
        mock_fd_config.cache_config.gpu_memory_utilization = 0.8
        mock_fd_config.cache_config.enable_prefix_caching = True
        mock_fd_config.parallel_config.disable_custom_all_reduce = False
        mock_fd_config.parallel_config.tensor_parallel_size = 1
        mock_fd_config.parallel_config.data_parallel_size = 1
        mock_fd_config.parallel_config.enable_expert_parallel = False

        report_usage_stats(mock_fd_config)

        # Thread should not be started when stats are disabled
        mock_thread.assert_not_called()

    @patch("fastdeploy.usage.usage_lib.requests.post")
    def test_send_to_server_success(self, mock_post):
        """Test successful server communication"""
        mock_post.return_value.status_code = 200

        data = {"test": "data"}
        self.usage_message._send_to_server(data)

        mock_post.assert_called_once()

    @patch("fastdeploy.usage.usage_lib.requests.post")
    def test_send_to_server_failure(self, mock_post):
        """Test server communication failure"""
        mock_post.side_effect = RequestException("Network unreachable")

        data = {"test": "data"}
        # Should not raise exception, just log debug message
        self.usage_message._send_to_server(data)


class TestUsageLibFunctions(unittest.TestCase):
    """Test individual functions in usage_lib.py"""

    def setUp(self):
        # Clear global data before each test
        _GLOBAL_RUNTIME_DATA.clear()

    def tearDown(self):
        # Clear global data after each test
        _GLOBAL_RUNTIME_DATA.clear()

    def test_set_runtime_usage_data(self):
        """Test setting runtime usage data"""
        set_runtime_usage_data("test_key", "test_value")
        self.assertEqual(_GLOBAL_RUNTIME_DATA["test_key"], "test_value")

        set_runtime_usage_data("int_key", 123)
        self.assertEqual(_GLOBAL_RUNTIME_DATA["int_key"], 123)

    def test_is_usage_stats_enabled(self):
        """Test usage stats enable/disable logic"""
        # Test when DO_NOT_TRACK is not set
        self.assertTrue(is_usage_stats_enabled())

    def test_get_current_timestamp_ns(self):
        """Test timestamp generation"""
        before = time.time_ns()
        timestamp = get_current_timestamp_ns()
        after = time.time_ns()

        self.assertIsInstance(timestamp, int)
        self.assertGreaterEqual(timestamp, before)
        self.assertLessEqual(timestamp, after)

    @patch("fastdeploy.usage.usage_lib.paddle")
    def test_cuda_is_initialized(self, mock_paddle):
        """Test CUDA initialization check"""
        # Test when CUDA is not compiled
        mock_paddle.is_compiled_with_cuda.return_value = False
        self.assertFalse(cuda_is_initialized())

        # Test when CUDA is compiled but no devices
        mock_paddle.is_compiled_with_cuda.return_value = True
        mock_paddle.device.cuda.device_count.return_value = 0
        self.assertFalse(cuda_is_initialized())

        # Test when CUDA is compiled and has devices
        mock_paddle.device.cuda.device_count.return_value = 2
        self.assertTrue(cuda_is_initialized())

    @patch("fastdeploy.usage.usage_lib.paddle")
    def test_cuda_device_count(self, mock_paddle):
        """Test CUDA device count"""
        # Test when not compiled with CUDA
        mock_paddle.device.is_compiled_with_cuda.return_value = False
        self.assertEqual(cuda_device_count(), 0)

        # Test when compiled with CUDA
        mock_paddle.device.is_compiled_with_cuda.return_value = True
        mock_paddle.device.cuda.device_count.return_value = 4
        self.assertEqual(cuda_device_count(), 4)

    @patch("fastdeploy.usage.usage_lib.paddle")
    def test_xpu_device_count(self, mock_paddle):
        """Test XPU device count"""
        # Test when not compiled with XPU
        mock_paddle.device.is_compiled_with_xpu.return_value = False
        self.assertEqual(xpu_device_count(), 0)

        # Test when compiled with XPU
        mock_paddle.device.is_compiled_with_xpu.return_value = True
        mock_paddle.device.xpu.device_count.return_value = 2
        self.assertEqual(xpu_device_count(), 2)

    @patch("fastdeploy.usage.usage_lib.os")
    @patch("fastdeploy.usage.usage_lib.Path")
    def test_detect_cloud_provider(self, mock_path, mock_os):
        """Test cloud provider detection"""
        # Test PDC detection
        mock_os.environ.get.return_value = "test_job"
        self.assertEqual(detect_cloud_provider(), "PDC")

        # Test unknown provider
        mock_os.environ.get.return_value = None
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.is_file.return_value = False

        self.assertEqual(detect_cloud_provider(), "Unknown")

    def test_simple_convert(self):
        """Test object conversion for serialization"""
        # Test basic types
        self.assertEqual(simple_convert("test"), "test")
        self.assertEqual(simple_convert(123), 123)
        self.assertEqual(simple_convert(True), True)

        # Test list
        self.assertEqual(simple_convert([1, "test"]), [1, "test"])

        # Test dict
        self.assertEqual(simple_convert({"key": "value"}), {"key": "value"})

        # Test object with to_dict method
        class TestObj:
            def to_dict(self):
                return {"converted": True}

        obj = TestObj()
        self.assertEqual(simple_convert(obj), {"converted": True})


class TestFileWriting(unittest.TestCase):
    """Test file writing functionality"""

    @patch("fastdeploy.usage.usage_lib.os.makedirs")
    @patch("fastdeploy.usage.usage_lib.Path.touch")
    @patch("fastdeploy.usage.usage_lib.open", new_callable=mock_open)
    def test_write_to_file(self, mock_file, mock_touch, mock_makedirs):
        """Test writing usage data to file"""
        usage_message = UsageMessage()
        data = {"uuid": "test-uuid", "timestamp": 1234567890}

        usage_message._write_to_file(data)

        # Verify file operations
        mock_makedirs.assert_called_once()
        mock_touch.assert_called_once()

        # Verify JSON was written
        all_writes = [call.args[0] for call in mock_file().write.call_args_list]
        full_content = "".join(all_writes)
        self.assertEqual(json.loads(full_content), data)


class TestReportUsageWorker(unittest.TestCase):
    """Test _report_usage_worker method"""

    def setUp(self):
        self.usage_message = UsageMessage()
        self.mock_fd_config = MagicMock()
        self.mock_extra_kvs = {"test_param": "test_value"}

    @patch("fastdeploy.usage.usage_lib.UsageMessage._report_usage_once")
    @patch("fastdeploy.usage.usage_lib.UsageMessage._report_continuous_usage")
    def test_report_usage_worker_calls_methods(self, mock_continuous, mock_once):
        """Test that _report_usage_worker calls required methods"""
        self.usage_message._report_usage_worker(self.mock_fd_config, self.mock_extra_kvs)

        # Verify that both methods are called with correct arguments
        mock_once.assert_called_once_with(self.mock_fd_config, self.mock_extra_kvs)
        mock_continuous.assert_called_once()


class TestReportUsageOnce(unittest.TestCase):
    """Test _report_usage_once method"""

    def setUp(self):
        self.usage_message = UsageMessage()
        self.mock_fd_config = MagicMock()

        # Setup mock FDConfig
        self.mock_fd_config.model_config.architectures = ["TestModel"]
        self.mock_fd_config.model_config.quantization = None

    @patch("fastdeploy.usage.usage_lib.current_platform")
    @patch("fastdeploy.usage.usage_lib.cuda_device_count")
    @patch("fastdeploy.usage.usage_lib.cuda_get_device_properties")
    @patch("fastdeploy.usage.usage_lib.xpu_device_count")
    @patch("fastdeploy.usage.usage_lib.get_xpu_model")
    @patch("fastdeploy.usage.usage_lib.get_cuda_version")
    @patch("fastdeploy.usage.usage_lib.detect_cloud_provider")
    @patch("fastdeploy.usage.usage_lib.platform.machine")
    @patch("fastdeploy.usage.usage_lib.platform.platform")
    @patch("fastdeploy.usage.usage_lib.psutil.virtual_memory")
    @patch("fastdeploy.usage.usage_lib.cpuinfo.get_cpu_info")
    @patch("fastdeploy.usage.usage_lib.get_current_timestamp_ns")
    @patch("fastdeploy.usage.usage_lib.simple_convert")
    @patch("fastdeploy.usage.usage_lib.UsageMessage._write_to_file")
    @patch("fastdeploy.usage.usage_lib.UsageMessage._send_to_server")
    def test_report_usage_once_cuda_platform(
        self,
        mock_send,
        mock_write,
        mock_convert,
        mock_timestamp,
        mock_cpuinfo,
        mock_virtual_memory,
        mock_platform,
        mock_machine,
        mock_detector,
        mock_cuda_version,
        mock_xpu_model,
        mock_xpu_count,
        mock_cuda_props,
        mock_cuda_count,
        mock_current_platform,
    ):
        """Test _report_usage_once method for CUDA platform"""
        # Mock platform
        mock_current_platform.is_cuda_alike.return_value = True
        mock_current_platform.is_xpu.return_value = False
        mock_current_platform.is_cuda.return_value = True

        # Mock device properties
        mock_cuda_count.return_value = 2
        mock_cuda_props.return_value = ("TestGPU", 1024 * 1024 * 1024)  # 1GB

        # Mock system info
        mock_detector.return_value = "AWS"
        mock_machine.return_value = "x86_64"
        mock_platform.return_value = "Linux-5.15.0"

        vm_mock = MagicMock()
        vm_mock.total = 1024 * 1024 * 1024 * 16  # 16GB
        mock_virtual_memory.return_value = vm_mock

        # Mock CPU info
        mock_cpuinfo.return_value = {
            "count": 8,
            "brand_raw": "Intel Xeon",
            "family": "6",
            "model": "85",
            "stepping": "7",
        }

        # Mock other values
        mock_timestamp.return_value = 1234567890000000000
        mock_cuda_version.return_value = "12.1"
        mock_convert.return_value = {"config": "test"}

        fake_envs = SimpleNamespace(
            ENABLE_V1_KVCACHE_SCHEDULER="test_source",
            FD_DISABLE_CHUNKED_PREFILL=False,
            FD_USE_HF_TOKENIZER=False,
            FD_PLUGINS="",
            FD_USAGE_SOURCE="",
        )

        # Mock imports
        with patch.dict("sys.modules", {"fastdeploy": MagicMock()}):
            mock_fastdeploy = sys.modules["fastdeploy"]
            mock_fastdeploy.__version__ = "1.0.0"
            with patch("fastdeploy.usage.usage_lib.envs", fake_envs):
                self.usage_message._report_usage_once(self.mock_fd_config, {})

        # Verify platform detection was called
        mock_current_platform.is_cuda_alike.assert_called()
        mock_current_platform.is_xpu.assert_called()
        mock_current_platform.is_cuda.assert_called()

        # Verify device properties were collected
        mock_cuda_count.assert_called()
        mock_cuda_props.assert_called()
        mock_cuda_version.assert_called()

        # Verify system info was collected
        mock_detector.assert_called()
        mock_machine.assert_called()
        mock_platform.assert_called()

        # Verify file operations were called
        mock_write.assert_called_once()
        mock_send.assert_called_once()

    @patch("fastdeploy.usage.usage_lib.current_platform")
    @patch("fastdeploy.usage.usage_lib.paddle.device.xpu")
    @patch("fastdeploy.usage.usage_lib.xpu_device_count")
    @patch("fastdeploy.usage.usage_lib.get_xpu_model")
    @patch("fastdeploy.usage.usage_lib.UsageMessage._write_to_file")
    @patch("fastdeploy.usage.usage_lib.UsageMessage._send_to_server")
    def test_report_usage_once_xpu_platform(
        self, mock_send, mock_write, mock_xpu_model, mock_xpu_count, mock_xpu, mock_current_platform
    ):
        """Test _report_usage_once method for XPU platform"""
        # Mock platform
        mock_current_platform.is_cuda_alike.return_value = False
        mock_current_platform.is_xpu.return_value = True

        # Mock XPU properties
        mock_xpu_count.return_value = 1
        mock_xpu_model.return_value = "P900"
        mock_xpu.memory_total.return_value = 1024 * 1024 * 1024  # 1GB

        fake_envs = SimpleNamespace(
            ENABLE_V1_KVCACHE_SCHEDULER="test_source",
            FD_DISABLE_CHUNKED_PREFILL=False,
            FD_USE_HF_TOKENIZER=False,
            FD_PLUGINS="",
            FD_USAGE_SOURCE="",
        )

        # Mock other necessary methods
        with patch.multiple(
            "fastdeploy.usage.usage_lib",
            detect_cloud_provider=MagicMock(return_value="Unknown"),
            platform=MagicMock(),
            psutil=MagicMock(),
            cpuinfo=MagicMock(),
            get_current_timestamp_ns=MagicMock(return_value=1234567890000000000),
            envs=fake_envs,
            simple_convert=MagicMock(return_value={}),
        ):

            with patch.dict("sys.modules", {"fastdeploy": MagicMock()}):
                mock_fastdeploy = sys.modules["fastdeploy"]
                mock_fastdeploy.__version__ = "1.0.0"

                self.usage_message._report_usage_once(self.mock_fd_config, {})

        # Verify XPU properties were collected
        mock_xpu_count.assert_called()
        mock_xpu_model.assert_called()
        mock_xpu.memory_total.assert_called()

        # Verify file operations were called
        mock_write.assert_called_once()
        mock_send.assert_called_once()


if __name__ == "__main__":
    unittest.main()
