"""
Unit tests for usage_lib.py
"""

import json
import time
import unittest
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


if __name__ == "__main__":
    unittest.main()
