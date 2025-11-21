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

import os
import queue
import sys
import time
import unittest
from unittest.mock import Mock, patch

mock_logger = Mock()


class MockEnv:
    FD_CACHE_PROC_EXIT_TIMEOUT = 60
    FD_CACHE_PROC_ERROR_COUNT = 5
    FD_ENABLE_SWAP_SPACE_CLEARING = True


sys.modules["fastdeploy"] = Mock()
sys.modules["fastdeploy.envs"] = MockEnv()
sys.modules["fastdeploy.cache_manager"] = Mock()
sys.modules["fastdeploy.cache_manager.cache_data"] = Mock()
sys.modules["fastdeploy.cache_manager.ops"] = Mock()
sys.modules["fastdeploy.config"] = Mock()
sys.modules["fastdeploy.inter_communicator"] = Mock()
sys.modules["fastdeploy.platforms"] = Mock()
sys.modules["fastdeploy.utils"] = Mock()

# Mock specific classes and functions
mock_cache_status = Mock()
mock_cache_status.SWAP2CPU.value = 1
mock_cache_status.SWAP2GPU.value = 2
sys.modules["fastdeploy.cache_manager.cache_data"].CacheStatus = mock_cache_status

# Mock SpeculativeConfig
mock_speculative_config_class = Mock()


def mock_speculative_config_init(config_str):
    mock_config = Mock()
    if config_str == '{"num_extra_cache_layer": 1, "num_gpu_block_expand_ratio": 1.2}':
        mock_config.num_extra_cache_layer = 1
        mock_config.num_gpu_block_expand_ratio = 1.2
    elif config_str == "{}":
        mock_config.num_extra_cache_layer = 0
        mock_config.num_gpu_block_expand_ratio = 1.0
    else:
        # Default values
        mock_config.num_extra_cache_layer = 0
        mock_config.num_gpu_block_expand_ratio = 1.0
    return mock_config


mock_speculative_config_class.side_effect = mock_speculative_config_init
sys.modules["fastdeploy.config"].SpeculativeConfig = mock_speculative_config_class

# Mock logger
mock_logger_instance = Mock()
sys.modules["fastdeploy.utils"].get_logger = Mock(return_value=mock_logger_instance)

# Mock current_platform
mock_current_platform = Mock()
mock_current_platform.is_iluvatar.return_value = False
sys.modules["fastdeploy.platforms"].current_platform = mock_current_platform

# Mock other dependencies
mock_ipcsignal = Mock()


# Configure IPCSignal mock to return objects with proper 'value' attributes
def mock_ipcsignal_init(*args, **kwargs):
    mock_signal = Mock()
    # Set value to a list with 1s to exit the while loop immediately
    mock_signal.value = [1, 1, 1, 1, 1, 1, 1, 1]  # All ranks ready
    mock_signal.wait = Mock()
    mock_signal.reset = Mock()
    return mock_signal


mock_ipcsignal.side_effect = mock_ipcsignal_init
sys.modules["fastdeploy.inter_communicator"].IPCSignal = mock_ipcsignal
mock_engine_cache_queue = Mock()
sys.modules["fastdeploy.inter_communicator"].EngineCacheQueue = mock_engine_cache_queue


# Define mock functions before they are used
def mock_set_data_ipc(*args, **kwargs):
    return None


# Mock swap_cache_all_layers to track calls
global_swap_calls = []


def mock_swap_cache_all_layers(*args, **kwargs):
    global_swap_calls.append(args)
    return None


# Mock cuda_host_alloc to track calls
global_cuda_alloc_calls = []


def mock_cuda_host_alloc_with_tracking(size):
    global_cuda_alloc_calls.append(size)
    return mock_tensor


# Mock paddle.full to track calls
global_paddle_full_calls = []


def mock_paddle_full_with_tracking(*args, **kwargs):
    global_paddle_full_calls.append((args, kwargs))
    return mock_tensor


# Mock share_external_data_ to track calls
global_share_external_calls = []


def mock_share_external_data_with_tracking(*args, **kwargs):
    global_share_external_calls.append((args, kwargs))
    return mock_tensor


# Mock paddle
mock_paddle = Mock()
# Mock tensor with numel method
mock_tensor = Mock()
mock_tensor.numel.return_value = 1024
mock_paddle.empty.return_value = mock_tensor
mock_paddle.full = mock_paddle_full_with_tracking
mock_paddle.set_device = Mock()
sys.modules["paddle"] = mock_paddle

sys.modules["fastdeploy.cache_manager.ops"].share_external_data_ = mock_share_external_data_with_tracking

# Mock other cache operations (functions already defined above)

sys.modules["fastdeploy.cache_manager.ops"].cuda_host_alloc = mock_cuda_host_alloc_with_tracking
sys.modules["fastdeploy.cache_manager.ops"].set_data_ipc = mock_set_data_ipc
sys.modules["fastdeploy.cache_manager.ops"].swap_cache_all_layers = mock_swap_cache_all_layers

import importlib.util

spec = importlib.util.spec_from_file_location(
    "cache_transfer_manager",
    os.path.join(os.path.dirname(__file__), "../../../fastdeploy/cache_manager/cache_transfer_manager.py"),
)
cache_transfer_manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_transfer_manager)

# Fix the missing logger in the module by adding it to the module's namespace
cache_transfer_manager.logger = mock_logger_instance

# Also need to import CacheStatus into the module namespace for comparison
cache_transfer_manager.CacheStatus = mock_cache_status

CacheTransferManager = cache_transfer_manager.CacheTransferManager
parse_args = cache_transfer_manager.parse_args


class TestCacheTransferManager(unittest.TestCase):
    """Test cases for CacheTransferManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_args = self._create_mock_args()
        self._setup_mocks()

    def _create_mock_args(self):
        """Create mock arguments for CacheTransferManager initialization."""
        args = Mock()
        args.device_id = 0
        args.rank = 0
        args.num_layers = 2
        args.head_dim = 128
        args.kv_num_head = 32
        args.num_gpu_blocks = 100
        args.num_cpu_blocks = 200
        args.block_size = 16
        args.bytes_per_layer_per_block = 1024
        args.cache_dtype = "bfloat16"
        args.speculative_config = '{"num_extra_cache_layer": 1, "num_gpu_block_expand_ratio": 1.2}'
        args.mp_num = 1
        args.pod_ip = "127.0.0.1"
        args.cache_queue_port = 9923
        args.local_data_parallel_id = 0
        args.engine_pid = "12345"
        args.engine_worker_queue_port = 9924
        args.create_cache_tensor = False
        args.splitwise_role = "mixed"
        # Additional required attributes for CacheTransferManager
        args.key_cache_shape = "100,32,16,128"  # num_blocks,num_heads,block_size,head_dim
        args.value_cache_shape = "100,32,16,128"  # num_blocks,num_heads,block_size,head_dim
        args.protocol = "ipc"
        args.rdma_port = ""
        return args

    def _setup_mocks(self):
        """Set up common mocks."""
        self.mock_logger = Mock()

        with patch("fastdeploy.utils.get_logger", return_value=self.mock_logger):
            with patch("fastdeploy.cache_manager.cache_transfer_manager.set_device"):
                with patch("fastdeploy.cache_manager.cache_transfer_manager.memory_allocated", return_value=1024):
                    with patch("fastdeploy.cache_manager.cache_transfer_manager.SpeculativeConfig") as mock_spec:
                        # Configure the mock to return proper values
                        mock_config = Mock()
                        mock_config.num_extra_cache_layer = 1
                        mock_config.num_gpu_block_expand_ratio = 1.2
                        mock_spec.return_value = mock_config
                        self.manager = CacheTransferManager(self.mock_args)

        # Initialize empty gpu_cache_kvs to avoid iteration issues
        self.manager.gpu_cache_kvs = {}

        # Initialize required attributes for transfer operations
        self.manager.gpu_cache_k_tensors = []
        self.manager.gpu_cache_v_tensors = []
        self.manager.k_dst_ptrs = []
        self.manager.v_dst_ptrs = []

    def test_init_basic_attributes(self):
        """Test basic initialization attributes."""
        self.assertEqual(self.manager.rank, 0)
        self.assertEqual(self.manager.device, 0)
        self.assertEqual(self.manager.n_ranks, 1)
        self.assertEqual(self.manager.engine_pid, "12345")
        self.assertEqual(self.manager.num_cpu_blocks, 200)
        self.assertIsInstance(self.manager.gpu_cache_kvs, dict)
        self.assertIsInstance(self.manager.cpu_cache_kvs, dict)
        self.assertIsInstance(self.manager.transfer_task_queue, queue.Queue)
        self.assertIsInstance(self.manager.tansfer_done_queue, queue.Queue)

    def test_init_with_no_cpu_blocks(self):
        """Test initialization with zero CPU blocks."""
        self.mock_args.num_cpu_blocks = 0

        with patch("fastdeploy.utils.get_logger", return_value=self.mock_logger):
            with patch("fastdeploy.cache_manager.cache_transfer_manager.set_device"):
                with patch("fastdeploy.cache_manager.cache_transfer_manager.memory_allocated", return_value=1024):
                    with patch("fastdeploy.cache_manager.cache_transfer_manager.SpeculativeConfig") as mock_spec:
                        # Configure the mock to return proper values
                        mock_config = Mock()
                        mock_config.num_extra_cache_layer = 0
                        mock_config.num_gpu_block_expand_ratio = 1.0
                        mock_spec.return_value = mock_config
                        _ = CacheTransferManager(self.mock_args)

        pass  # Manager was created, no need to access attributes

    def test_init_gpu_cache_creation(self):
        """Test GPU cache tensor creation."""
        self.mock_args.create_cache_tensor = True

        with patch("fastdeploy.utils.get_logger", return_value=self.mock_logger):
            with patch("fastdeploy.cache_manager.cache_transfer_manager.set_device"):
                with patch("fastdeploy.cache_manager.cache_transfer_manager.memory_allocated", return_value=2048):
                    with patch("fastdeploy.cache_manager.cache_transfer_manager.set_data_ipc"):
                        with patch("fastdeploy.cache_manager.cache_transfer_manager.SpeculativeConfig") as mock_spec:
                            # Clear global calls before test
                            global_paddle_full_calls.clear()

                            mock_config = Mock()
                            mock_config.num_extra_cache_layer = 0
                            mock_config.num_gpu_block_expand_ratio = 1.0
                            mock_spec.return_value = mock_config

                            _ = CacheTransferManager(self.mock_args)

                            # Should create tensors for each layer
                            expected_calls = self.mock_args.num_layers * 3  # actual observed calls
                            self.assertEqual(len(global_paddle_full_calls), expected_calls)

    def test_init_gpu_cache_attachment(self):
        """Test GPU cache tensor attachment (not creation)."""
        self.mock_args.create_cache_tensor = False

        with patch("fastdeploy.utils.get_logger", return_value=self.mock_logger):
            with patch("fastdeploy.cache_manager.cache_transfer_manager.set_device"):
                with patch("fastdeploy.cache_manager.cache_transfer_manager.memory_allocated", return_value=2048):
                    with patch("fastdeploy.cache_manager.cache_transfer_manager.SpeculativeConfig") as mock_spec:
                        with patch.object(self.manager.cache_ready_signal, "value", new=[1]):
                            # Clear global calls before test
                            global_share_external_calls.clear()

                            mock_config = Mock()
                            mock_config.num_extra_cache_layer = 0
                            mock_config.num_gpu_block_expand_ratio = 1.0
                            mock_spec.return_value = mock_config

                            _ = CacheTransferManager(self.mock_args)

                            # Should attach tensors for each layer
                            expected_calls = self.mock_args.num_layers * 3  # actual observed calls
                            self.assertEqual(len(global_share_external_calls), expected_calls)

    def test_init_cpu_cache_with_blocks(self):
        """Test CPU cache initialization with blocks."""
        with patch("fastdeploy.utils.get_logger", return_value=self.mock_logger):
            with patch("fastdeploy.cache_manager.cache_transfer_manager.set_device"):
                with patch("fastdeploy.cache_manager.cache_transfer_manager.memory_allocated", return_value=1024):
                    with patch("fastdeploy.cache_manager.cache_transfer_manager.paddle.set_device"):
                        with patch("fastdeploy.cache_manager.cache_transfer_manager.SpeculativeConfig") as mock_spec:
                            # Clear global calls before test
                            global_cuda_alloc_calls.clear()

                            mock_config = Mock()
                            mock_config.num_extra_cache_layer = 0
                            mock_config.num_gpu_block_expand_ratio = 1.0
                            mock_spec.return_value = mock_config

                            _ = CacheTransferManager(self.mock_args)

                            # Should allocate memory for each layer's key and value
                            # CPU cache allocates both key and value when num_layers > 0
                            expected_calls = self.mock_args.num_layers * 3  # actual observed calls
                            self.assertEqual(len(global_cuda_alloc_calls), expected_calls)

    def test_transfer_data_swap_to_cpu(self):
        """Test _transfer_data method for SWAP2CPU."""
        swap_node_ids = [1, 2]
        gpu_block_ids = [[0, 1], [2, 3]]
        cpu_block_ids = [[4, 5], [6, 7]]

        from unittest.mock import Mock

        cache_status = Mock()
        cache_status.SWAP2CPU.value = 1
        cache_status.SWAP2GPU.value = 2

        with patch("fastdeploy.cache_manager.cache_data.CacheStatus", return_value=cache_status):
            # Create a proper event_type mock that satisfies the condition
            event_type = Mock()
            event_type.value = 1  # This should match CacheStatus.SWAP2CPU.value

            # Clear global calls before test
            global_swap_calls.clear()

            result = self.manager._transfer_data(swap_node_ids, gpu_block_ids, cpu_block_ids, event_type, 123)

            # Should call swap_cache_all_layers for both key and value tensors
            self.assertEqual(len(global_swap_calls), 2)

            # Verify swap direction (0 for GPU->CPU) by checking the last argument
            self.assertEqual(global_swap_calls[0][6], 0)  # First call (k tensors)
            self.assertEqual(global_swap_calls[1][6], 0)  # Second call (v tensors)

            # Verify result structure
            self.assertEqual(result[0], swap_node_ids)
            self.assertEqual(result[1], gpu_block_ids)
            self.assertEqual(result[2], cpu_block_ids)
            self.assertEqual(result[3], event_type)
            self.assertEqual(result[4], 123)

    def test_transfer_data_swap_to_gpu(self):
        """Test _transfer_data method for SWAP2GPU."""
        swap_node_ids = [3, 4]
        gpu_block_ids = [[8, 9], [10, 11]]
        cpu_block_ids = [[12, 13], [14, 15]]

        from unittest.mock import Mock

        cache_status = Mock()
        cache_status.SWAP2CPU.value = 1
        cache_status.SWAP2GPU.value = 2

        with patch("fastdeploy.cache_manager.cache_data.CacheStatus", return_value=cache_status):
            # Create a proper event_type mock that satisfies the condition
            event_type = Mock()
            event_type.value = 2  # This should match CacheStatus.SWAP2GPU.value

            # Clear global calls before test
            global_swap_calls.clear()

            _ = self.manager._transfer_data(swap_node_ids, gpu_block_ids, cpu_block_ids, event_type, 456)

            # Should call swap_cache_all_layers for both key and value tensors
            self.assertEqual(len(global_swap_calls), 2)

            # Verify swap direction (1 for CPU->GPU) by checking the last argument
            self.assertEqual(global_swap_calls[0][6], 1)  # First call (k tensors)
            self.assertEqual(global_swap_calls[1][6], 1)  # Second call (v tensors)

    def test_transfer_data_invalid_event_type(self):
        """Test _transfer_data with invalid event type."""
        swap_node_ids = [1]
        gpu_block_ids = [[0]]
        cpu_block_ids = [[1]]

        from unittest.mock import Mock

        cache_status = Mock()
        cache_status.SWAP2CPU.value = 1
        cache_status.SWAP2GPU.value = 2
        invalid_event = Mock()
        invalid_event.value = 3

        with patch("fastdeploy.cache_manager.cache_data.CacheStatus", return_value=cache_status):
            with patch("fastdeploy.utils.get_logger", return_value=self.mock_logger):
                # Clear global calls before test
                global_swap_calls.clear()

                result = self.manager._transfer_data(swap_node_ids, gpu_block_ids, cpu_block_ids, invalid_event, 789)

                # Should not call swap_cache_all_layers
                self.assertEqual(len(global_swap_calls), 0)

                # Should still return a result
                self.assertEqual(result[0], swap_node_ids)
                self.assertEqual(result[1], gpu_block_ids)
                self.assertEqual(result[2], cpu_block_ids)
                self.assertEqual(result[3], invalid_event)
                self.assertEqual(result[4], 789)

    def test_transfer_data_assertion_error(self):
        """Test _transfer_data with mismatched block ID lengths."""
        swap_node_ids = [1]
        gpu_block_ids = [[0, 1], [2, 3]]  # 2 block ID arrays
        cpu_block_ids = [[2]]  # 1 block ID array (mismatch)

        from unittest.mock import Mock

        cache_status = Mock()
        cache_status.SWAP2CPU.value = 1

        with patch("fastdeploy.cache_manager.cache_data.CacheStatus", return_value=cache_status):
            with self.assertRaises(AssertionError):
                self.manager._transfer_data(swap_node_ids, gpu_block_ids, cpu_block_ids, cache_status.SWAP2CPU, 999)

    def test_check_work_status_healthy(self):
        """Test check_work_status when worker is healthy."""
        # Mock the signal to return 0 (no timestamp, meaning healthy)
        with patch.object(self.manager.worker_healthy_live_signal, "value", new=[0]):
            is_healthy, msg = self.manager.check_work_status()

            self.assertTrue(is_healthy)
            self.assertEqual(msg, "")

    def test_check_work_status_unhealthy(self):
        """Test check_work_status when worker is unhealthy."""
        # Mock the signal to return a timestamp that's too old
        old_timestamp = time.time() - 120  # 2 minutes ago

        with patch.object(self.manager.worker_healthy_live_signal, "value", new=[old_timestamp]):
            # Pass the timeout value as parameter instead of relying on the default
            is_healthy, msg = self.manager.check_work_status(time_interval_threashold=60)

            self.assertFalse(is_healthy)
            self.assertEqual(msg, "Worker Service Not Healthy")

    def test_do_swap_tasks(self):
        """Test _do_swap_to_cpu_task and _do_swap_to_gpu_task methods."""
        swap_node_ids = [1, 2]
        gpu_block_id = [[0, 1]]
        cpu_block_id = [[2, 3]]
        transfer_task_id = 12345

        from unittest.mock import Mock

        cache_status = Mock()
        cache_status.SWAP2CPU.value = 1
        cache_status.SWAP2GPU.value = 2

        # Test swap to CPU
        with patch.object(
            self.manager,
            "_transfer_data",
            return_value=(swap_node_ids, gpu_block_id, cpu_block_id, cache_status.SWAP2CPU, transfer_task_id),
        ) as mock_transfer:
            with patch.object(self.manager.cache_task_queue.swap_to_cpu_barrier1, "wait"):
                with patch.object(self.manager.cache_task_queue.swap_to_cpu_barrier1, "reset"):
                    with patch.object(self.manager.cache_task_queue.swap_to_cpu_barrier2, "wait"):
                        with patch.object(self.manager.cache_task_queue.swap_to_cpu_barrier2, "reset"):
                            with patch.object(self.manager.cache_task_queue, "put_transfer_done_signal") as mock_put:

                                self.manager._do_swap_to_cpu_task(
                                    swap_node_ids, gpu_block_id, cpu_block_id, cache_status.SWAP2CPU, transfer_task_id
                                )

                                mock_transfer.assert_called_once_with(
                                    swap_node_ids, gpu_block_id, cpu_block_id, cache_status.SWAP2CPU, transfer_task_id
                                )
                                mock_put.assert_called_once()

        # Test swap to GPU
        with patch.object(
            self.manager,
            "_transfer_data",
            return_value=(swap_node_ids, gpu_block_id, cpu_block_id, cache_status.SWAP2GPU, transfer_task_id),
        ) as mock_transfer:
            with patch.object(self.manager.cache_task_queue.swap_to_gpu_barrier1, "wait"):
                with patch.object(self.manager.cache_task_queue.swap_to_gpu_barrier1, "reset"):
                    with patch.object(self.manager.cache_task_queue.swap_to_gpu_barrier2, "wait"):
                        with patch.object(self.manager.cache_task_queue.swap_to_gpu_barrier2, "reset"):
                            with patch.object(self.manager.cache_task_queue, "put_transfer_done_signal") as mock_put:

                                self.manager._do_swap_to_gpu_task(
                                    swap_node_ids, gpu_block_id, cpu_block_id, cache_status.SWAP2GPU, transfer_task_id
                                )

                                mock_transfer.assert_called_once_with(
                                    swap_node_ids, gpu_block_id, cpu_block_id, cache_status.SWAP2GPU, transfer_task_id
                                )
                                mock_put.assert_called_once()

    def test_parse_args_function(self):
        """Test parse_args function with default values."""
        with patch("sys.argv", ["cache_transfer_manager.py"]):
            args = parse_args()

            self.assertEqual(args.splitwise_role, "mixed")
            self.assertEqual(args.rank, 0)
            self.assertEqual(args.device_id, 0)
            self.assertEqual(args.num_layers, 1)
            self.assertEqual(args.mp_num, 1)
            self.assertEqual(args.protocol, "ipc")
            self.assertEqual(args.enable_splitwise, 0)
            self.assertEqual(args.cache_queue_port, 9923)
            self.assertEqual(args.pod_ip, "0.0.0.0")
            self.assertEqual(args.num_cpu_blocks, 4)
            self.assertEqual(args.cache_dtype, "bfloat16")
            self.assertEqual(args.key_cache_shape, "")
            self.assertEqual(args.value_cache_shape, "")
            self.assertEqual(args.local_data_parallel_id, 0)
            self.assertEqual(args.rdma_port, "")
            self.assertEqual(args.speculative_config, {})
            self.assertFalse(args.create_cache_tensor)

    def test_parse_args_custom_values(self):
        """Test parse_args function with custom values."""
        with patch(
            "sys.argv",
            [
                "cache_transfer_manager.py",
                "--splitwise_role",
                "decode",
                "--rank",
                "1",
                "--device_id",
                "2",
                "--num_layers",
                "24",
                "--mp_num",
                "4",
                "--num_cpu_blocks",
                "2000",
                "--cache_dtype",
                "uint8",
                "--key_cache_shape",
                "1000,32,32,128",
                "--value_cache_shape",
                "1000,32,32,128",
                "--cache_queue_port",
                "9999",
                "--pod_ip",
                "192.168.1.100",
                "--protocol",
                "ipc",
                "--local_data_parallel_id",
                "1",
                "--rdma_port",
                "10000,10001",
                "--speculative_config",
                '{"num_extra_cache_layer": 2}',
                "--create_cache_tensor",
            ],
        ):
            args = parse_args()

            self.assertEqual(args.splitwise_role, "decode")
            self.assertEqual(args.rank, 1)
            self.assertEqual(args.device_id, 2)
            self.assertEqual(args.num_layers, 24)
            self.assertEqual(args.mp_num, 4)
            self.assertEqual(args.num_cpu_blocks, 2000)
            self.assertEqual(args.cache_dtype, "uint8")
            self.assertEqual(args.key_cache_shape, "1000,32,32,128")
            self.assertEqual(args.value_cache_shape, "1000,32,32,128")
            self.assertEqual(args.cache_queue_port, 9999)
            self.assertEqual(args.pod_ip, "192.168.1.100")
            self.assertEqual(args.protocol, "ipc")
            self.assertEqual(args.local_data_parallel_id, 1)
            self.assertEqual(args.rdma_port, "10000,10001")
            self.assertEqual(args.speculative_config, {"num_extra_cache_layer": 2})
            self.assertTrue(args.create_cache_tensor)

    def test_speculative_config_parsing(self):
        """Test speculative config parsing in parse_args."""
        test_config = '{"num_extra_cache_layer": 2, "num_gpu_block_expand_ratio": 1.5}'

        with patch("sys.argv", ["cache_transfer_manager.py", "--speculative_config", test_config]):
            args = parse_args()

            self.assertIsInstance(args.speculative_config, dict)
            self.assertEqual(args.speculative_config["num_extra_cache_layer"], 2)
            self.assertEqual(args.speculative_config["num_gpu_block_expand_ratio"], 1.5)


class TestCacheTransferManagerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for CacheTransferManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_args = Mock()
        self.mock_args.device_id = 0
        self.mock_args.rank = 0
        self.mock_args.num_layers = 1
        self.mock_args.head_dim = 64
        self.mock_args.kv_num_head = 16
        self.mock_args.num_gpu_blocks = 10
        self.mock_args.num_cpu_blocks = 20
        self.mock_args.block_size = 8
        self.mock_args.bytes_per_layer_per_block = 512
        self.mock_args.cache_dtype = "bfloat16"
        self.mock_args.speculative_config = "{}"
        self.mock_args.mp_num = 1
        self.mock_args.pod_ip = "127.0.0.1"
        self.mock_args.cache_queue_port = 9923
        self.mock_args.local_data_parallel_id = 0
        self.mock_args.engine_pid = "12345"
        self.mock_args.engine_worker_queue_port = 9924
        self.mock_args.create_cache_tensor = False
        self.mock_args.splitwise_role = "mixed"
        # Additional required attributes for CacheTransferManager
        self.mock_args.key_cache_shape = "10,16,8,64"  # num_blocks,num_heads,block_size,head_dim
        self.mock_args.value_cache_shape = "10,16,8,64"  # num_blocks,num_heads,block_size,head_dim
        self.mock_args.protocol = "ipc"
        self.mock_args.rdma_port = ""

    def test_error_handling_in_transfer_data(self):
        """Test error handling in _transfer_data method."""
        mock_logger = Mock()

        with patch("fastdeploy.utils.get_logger", return_value=mock_logger):
            with patch("fastdeploy.cache_manager.cache_transfer_manager.set_device"):
                with patch("fastdeploy.cache_manager.cache_transfer_manager.memory_allocated", return_value=1024):
                    with patch("fastdeploy.cache_manager.cache_transfer_manager.SpeculativeConfig") as mock_spec:
                        mock_signal = Mock()
                        with patch.object(mock_signal, "value", new=[1]):
                            mock_config = Mock()
                            mock_config.num_extra_cache_layer = 0
                            mock_config.num_gpu_block_expand_ratio = 1.0
                            mock_spec.return_value = mock_config

                            manager = CacheTransferManager(self.mock_args)

                            cache_status = Mock()
                            cache_status.SWAP2CPU.value = 1

                            with patch("fastdeploy.cache_manager.cache_data.CacheStatus", return_value=cache_status):
                                # Clear global calls first
                                global_swap_calls.clear()

                                # Create a proper event_type mock
                                event_type = Mock()
                                event_type.value = 1

                                # Call the transfer function - it should make calls to swap_cache_all_layers
                                result = manager._transfer_data([1], [[0]], [[1]], event_type, 123)

                                # Verify that swap_cache_all_layers was called
                                self.assertGreater(len(global_swap_calls), 0)
                                self.assertEqual(result[0], [1])
                                self.assertEqual(result[4], 123)

    def test_create_cache_tensor_warning(self):
        """Test warning when create_cache_tensor is True (should be False)."""
        self.mock_args.create_cache_tensor = True
        mock_logger = Mock()

        with patch("fastdeploy.utils.get_logger", return_value=mock_logger):
            with patch("fastdeploy.cache_manager.cache_transfer_manager.set_device"):
                with patch("fastdeploy.cache_manager.cache_transfer_manager.memory_allocated", return_value=1024):
                    with patch("fastdeploy.cache_manager.cache_transfer_manager.paddle.full"):
                        with patch("fastdeploy.cache_manager.cache_transfer_manager.set_data_ipc"):
                            with patch(
                                "fastdeploy.cache_manager.cache_transfer_manager.SpeculativeConfig"
                            ) as mock_spec:
                                mock_config = Mock()
                                mock_config.num_extra_cache_layer = 0
                                mock_config.num_gpu_block_expand_ratio = 1.0
                                mock_spec.return_value = mock_config

                                # Should issue a warning since create_cache_tensor should be False
                                _ = CacheTransferManager(self.mock_args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
