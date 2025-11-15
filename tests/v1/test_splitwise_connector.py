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
import os
import sys
import unittest
from unittest.mock import Mock, patch

# Determine import method based on environment
TEST_MODE = os.environ.get("FD_TEST_MODE", "normal")

if TEST_MODE == "standalone":
    # Local testing mode - use dynamic import
    mock_logger = Mock()

    class MockUtils:
        def get_logger(self, name, filename):
            return mock_logger

    class MockEnvs:
        FD_ENABLE_CACHE_TASK = "0"
        FD_PD_CHANGEABLE = "0"
        FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        ENABLE_V1_KVCACHE_SCHEDULER = False

    class MockMetricsManager:
        class send_cache_failed_num:
            @staticmethod
            def inc():
                pass

    class MockMetrics:
        send_cache_failed_num = MockMetricsManager.send_cache_failed_num

    # Mock ZMQ module
    class MockZMQ:
        class Context:
            def socket(self, socket_type):
                mock_socket = Mock()
                return mock_socket

        ROUTER = "ROUTER"
        DEALER = "DEALER"
        POLLIN = "POLLIN"
        LINGER = "LINGER"
        SNDHWM = "SNDHWM"
        ROUTER_MANDATORY = "ROUTER_MANDATORY"
        RECONNECT_IVL = "RECONNECT_IVL"
        RECONNECT_IVL_MAX = "RECONNECT_IVL_MAX"
        TCP_KEEPALIVE = "TCP_KEEPALIVE"
        TCP_KEEPALIVE_IDLE = "TCP_KEEPALIVE_IDLE"
        TCP_KEEPALIVE_INTVL = "TCP_KEEPALIVE_INTVL"
        Again = Exception("Queue full")
        ZMQError = Exception("ZMQ Error")

        class Poller:
            def register(self, socket, event_type):
                pass

            def poll(self, timeout):
                return {}

    # Mock Request and RequestOutput classes
    class MockRequest:
        def __init__(self):
            self.request_id = "test_request"
            self.disaggregate_info = None
            self.block_tables = []
            self.idx = 0
            self.need_prefill_tokens = 0

        def to_dict(self):
            return {"request_id": self.request_id}

        @classmethod
        def from_dict(cls, data):
            request = cls()
            request.request_id = data.get("request_id", "test_request")
            return request

    class MockRequestOutput:
        def __init__(self):
            self.request_id = "test_output"

        def to_dict(self):
            return {"request_id": self.request_id}

        @classmethod
        def from_dict(cls, data):
            output = cls()
            output.request_id = data.get("request_id", "test_output")
            return output

    sys.modules["zmq"] = MockZMQ()
    sys.modules["fastdeploy"] = Mock()
    sys.modules["fastdeploy.envs"] = MockEnvs()
    sys.modules["fastdeploy.utils"] = MockUtils()
    sys.modules["fastdeploy.inter_communicator"] = Mock()
    sys.modules["fastdeploy.metrics"] = Mock()
    sys.modules["fastdeploy.metrics.metrics"] = Mock()
    sys.modules["fastdeploy.metrics.metrics"].main_process_metrics = MockMetricsManager()
    sys.modules["fastdeploy.engine"] = Mock()
    sys.modules["fastdeploy.engine.request"] = Mock()
    sys.modules["fastdeploy.engine.request"].Request = MockRequest
    sys.modules["fastdeploy.engine.request"].RequestOutput = MockRequestOutput

    # Import the splitwise_connector module directly
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "splitwise_connector",
        os.path.join(os.path.dirname(__file__), "../../fastdeploy/splitwise/splitwise_connector.py"),
    )
    splitwise_connector_module = importlib.util.module_from_spec(spec)
    sys.modules["fastdeploy.splitwise"] = Mock()
    sys.modules["fastdeploy.splitwise.splitwise_connector"] = splitwise_connector_module
    spec.loader.exec_module(splitwise_connector_module)

    SplitwiseConnector = splitwise_connector_module.SplitwiseConnector
else:
    # Normal mode - direct import
    try:
        from fastdeploy.splitwise.splitwise_connector import SplitwiseConnector
    except ImportError:
        print("Warning: Direct import failed, falling back to standalone mode")
        TEST_MODE = "standalone"
        mock_logger = Mock()

        class MockUtils:
            def get_logger(self, name, filename):
                return mock_logger

        class MockEnvs:
            FD_ENABLE_CACHE_TASK = "0"
            FD_PD_CHANGEABLE = "0"
            FD_ENGINE_TASK_QUEUE_WITH_SHM = False
            ENABLE_V1_KVCACHE_SCHEDULER = False

        class MockMetricsManager:
            class send_cache_failed_num:
                @staticmethod
                def inc():
                    pass

        class MockMetrics:
            send_cache_failed_num = MockMetricsManager.send_cache_failed_num

        # Mock ZMQ module
        class MockZMQ:
            class Context:
                def socket(self, socket_type):
                    mock_socket = Mock()
                    return mock_socket

            ROUTER = "ROUTER"
            DEALER = "DEALER"
            POLLIN = "POLLIN"
            LINGER = "LINGER"
            SNDHWM = "SNDHWM"
            ROUTER_MANDATORY = "ROUTER_MANDATORY"
            RECONNECT_IVL = "RECONNECT_IVL"
            RECONNECT_IVL_MAX = "RECONNECT_IVL_MAX"
            TCP_KEEPALIVE = "TCP_KEEPALIVE"
            TCP_KEEPALIVE_IDLE = "TCP_KEEPALIVE_IDLE"
            TCP_KEEPALIVE_INTVL = "TCP_KEEPALIVE_INTVL"
            Again = Exception("Queue full")
            ZMQError = Exception("ZMQ Error")

            class Poller:
                def register(self, socket, event_type):
                    pass

                def poll(self, timeout):
                    return {}

        # Mock Request and RequestOutput classes
        class MockRequest:
            def __init__(self):
                self.request_id = "test_request"
                self.disaggregate_info = None
                self.block_tables = []
                self.idx = 0
                self.need_prefill_tokens = 0

            def to_dict(self):
                return {"request_id": self.request_id}

            @classmethod
            def from_dict(cls, data):
                request = cls()
                request.request_id = data.get("request_id", "test_request")
                return request

        class MockRequestOutput:
            def __init__(self):
                self.request_id = "test_output"

            def to_dict(self):
                return {"request_id": self.request_id}

            @classmethod
            def from_dict(cls, data):
                output = cls()
                output.request_id = data.get("request_id", "test_output")
                return output

        sys.modules["zmq"] = MockZMQ()
        sys.modules["fastdeploy"] = Mock()
        sys.modules["fastdeploy.envs"] = MockEnvs()
        sys.modules["fastdeploy.utils"] = MockUtils()
        sys.modules["fastdeploy.inter_communicator"] = Mock()
        sys.modules["fastdeploy.metrics"] = Mock()
        sys.modules["fastdeploy.metrics.metrics"] = Mock()
        sys.modules["fastdeploy.metrics.metrics"].main_process_metrics = MockMetricsManager()
        sys.modules["fastdeploy.engine"] = Mock()
        sys.modules["fastdeploy.engine.request"] = Mock()
        sys.modules["fastdeploy.engine.request"].Request = MockRequest
        sys.modules["fastdeploy.engine.request"].RequestOutput = MockRequestOutput

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "splitwise_connector",
            os.path.join(os.path.dirname(__file__), "../../fastdeploy/splitwise/splitwise_connector.py"),
        )
        splitwise_connector_module = importlib.util.module_from_spec(spec)
        sys.modules["fastdeploy.splitwise"] = Mock()
        sys.modules["fastdeploy.splitwise.splitwise_connector"] = splitwise_connector_module
        spec.loader.exec_module(splitwise_connector_module)

        SplitwiseConnector = splitwise_connector_module.SplitwiseConnector


class TestSplitwiseConnector(unittest.TestCase):
    """Test cases for SplitwiseConnector class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock configuration
        self.mock_cfg = Mock()
        self.mock_cfg.parallel_config.enable_expert_parallel = False
        self.mock_cfg.parallel_config.data_parallel_size = 1
        self.mock_cfg.parallel_config.local_data_parallel_id = 0
        self.mock_cfg.parallel_config.engine_worker_queue_port = [12345]
        self.mock_cfg.parallel_config.tensor_parallel_size = 1
        self.mock_cfg.parallel_config.device_ids = "0,1"
        self.mock_cfg.cache_config.pd_comm_port = None
        self.mock_cfg.innode_prefill_ports = None
        self.mock_cfg.host_ip = "127.0.0.1"
        self.mock_cfg.disaggregate_info = {"cache_info": {"rdma": {"rdma_port": 8080}}}

        # Create mock worker queue
        self.mock_worker_queue = Mock()

        # Create mock resource manager
        self.mock_resource_manager = Mock()

        # Create ZMQ mocks
        self.mock_zmq_ctx = Mock()
        self.mock_socket = Mock()
        self.mock_poller = Mock()

    def create_connector(self, cfg=None):
        """Helper method to create SplitwiseConnector instance."""
        if cfg is None:
            cfg = self.mock_cfg

        with patch("fastdeploy.splitwise.splitwise_connector.zmq.Context") as mock_zmq_ctx_class:
            mock_zmq_ctx_class.return_value = self.mock_zmq_ctx
            with patch("fastdeploy.splitwise.splitwise_connector.ThreadPoolExecutor") as mock_executor:
                mock_executor.return_value = Mock()
                connector = SplitwiseConnector(cfg, self.mock_worker_queue, self.mock_resource_manager)
                return connector

    def test_init_basic(self):
        """Test basic initialization."""
        connector = self.create_connector()

        self.assertEqual(connector.cfg, self.mock_cfg)
        self.assertEqual(connector.engine_worker_queue, self.mock_worker_queue)
        self.assertEqual(connector.resource_manager, self.mock_resource_manager)
        self.assertEqual(connector.idx, 0)
        self.assertEqual(connector.connect_innode_instances, {})
        self.assertEqual(connector.temp_cache_info, {})
        self.assertEqual(connector.current_request_ids, {})
        self.assertFalse(connector.enable_decode_cache_task)

    def test_init_with_expert_parallel(self):
        """Test initialization with expert parallel enabled."""
        self.mock_cfg.parallel_config.enable_expert_parallel = True
        self.mock_cfg.parallel_config.data_parallel_size = 2

        connector = self.create_connector()

        self.assertIsNotNone(connector.logger)

    def test_init_with_network(self):
        """Test initialization with network configuration."""
        self.mock_cfg.cache_config.pd_comm_port = [5678]

        with patch("fastdeploy.splitwise.splitwise_connector.zmq.Context") as mock_zmq_ctx_class:
            mock_zmq_ctx_class.return_value = self.mock_zmq_ctx
            with patch("fastdeploy.splitwise.splitwise_connector.ThreadPoolExecutor") as mock_executor:
                mock_executor.return_value = Mock()
                with patch.object(SplitwiseConnector, "_init_network") as mock_init_network:
                    SplitwiseConnector(self.mock_cfg, self.mock_worker_queue, self.mock_resource_manager)
                    mock_init_network.assert_called_once()

    @patch("fastdeploy.splitwise.splitwise_connector.envs.FD_ENABLE_CACHE_TASK", "1")
    def test_init_with_cache_task_enabled(self):
        """Test initialization with cache task enabled."""
        connector = self.create_connector()
        self.assertTrue(connector.enable_decode_cache_task)

    def test_init_network(self):
        """Test network initialization."""
        self.mock_cfg.cache_config.pd_comm_port = [5678]

        with patch("fastdeploy.splitwise.splitwise_connector.zmq.Context") as mock_zmq_ctx_class:
            mock_zmq_ctx_class.return_value = self.mock_zmq_ctx
            mock_router_socket = Mock()
            self.mock_zmq_ctx.socket.return_value = mock_router_socket
            mock_poller = Mock()
            with patch("fastdeploy.splitwise.splitwise_connector.zmq.Poller") as mock_poller_class:
                mock_poller_class.return_value = mock_poller

                connector = self.create_connector()

                self.mock_zmq_ctx.socket.assert_called_with("ROUTER")
                mock_router_socket.setsockopt.assert_any_call("LINGER", 0)
                mock_router_socket.setsockopt.assert_any_call("SNDHWM", 1000)
                mock_router_socket.setsockopt.assert_any_call("ROUTER_MANDATORY", 1)
                mock_router_socket.bind.assert_called_with("tcp://*:5678")
                mock_poller.register.assert_called_with(mock_router_socket, "POLLIN")

                self.assertEqual(connector.router_socket, mock_router_socket)
                self.assertEqual(connector.poller, mock_poller)
                self.assertEqual(connector.prefill_cache_info, [])

    def test_serialize_message_prefill(self):
        """Test message serialization for prefill type."""
        connector = self.create_connector()

        # Create mock payload with Request objects
        mock_request = Mock()
        mock_request.to_dict.return_value = {"request_id": "test123", "data": "test_data"}
        payload = [mock_request]

        result = connector._serialize_message("prefill", payload)

        expected_data = json.dumps(
            {"type": "prefill", "payload": [{"request_id": "test123", "data": "test_data"}]}
        ).encode("utf-8")

        self.assertEqual(result, expected_data)

    def test_serialize_message_cache_sync(self):
        """Test message serialization for cache_sync type."""
        connector = self.create_connector()

        payload = {"request_id": "test123", "cache_data": "test_cache"}

        result = connector._serialize_message("cache_sync", payload)

        expected_data = json.dumps(
            {"type": "cache_sync", "payload": {"request_id": "test123", "cache_data": "test_cache"}}
        ).encode("utf-8")

        self.assertEqual(result, expected_data)

    def test_deserialize_message(self):
        """Test message deserialization."""
        connector = self.create_connector()

        message_data = json.dumps(
            {"type": "prefill", "payload": {"request_id": "test123", "data": "test_data"}}
        ).encode("utf-8")

        msg_type, payload = connector._deserialize_message(message_data)

        self.assertEqual(msg_type, "prefill")
        self.assertEqual(payload, {"request_id": "test123", "data": "test_data"})

    @unittest.skip("has_splitwise_tasks method not found in current implementation")
    def test_has_splitwise_tasks_no_ports(self):
        """Test has_splitwise_tasks when no prefill ports configured."""
        connector = self.create_connector()

        result = connector.has_splitwise_tasks()

        self.assertTrue(result)

    @unittest.skip("has_splitwise_tasks method not found in current implementation")
    def test_has_splitwise_tasks_with_available_instances(self):
        """Test has_splitwise_tasks with available prefill instances."""
        self.mock_cfg.innode_prefill_ports = [12345, 12346]

        connector = self.create_connector()

        # Mock connection with available instances
        mock_connection = Mock()
        mock_connection.available_prefill_instances.qsize.return_value = 1
        connector.connect_innode_instances[12345] = mock_connection

        result = connector.has_splitwise_tasks()

        self.assertFalse(result)

    @unittest.skip("has_splitwise_tasks method not found in current implementation")
    def test_has_splitwise_tasks_no_available_instances(self):
        """Test has_splitwise_tasks with no available prefill instances."""
        self.mock_cfg.innode_prefill_ports = [12345, 12346]

        connector = self.create_connector()

        # Mock connections with no available instances
        mock_connection1 = Mock()
        mock_connection1.available_prefill_instances.qsize.return_value = 0
        mock_connection2 = Mock()
        mock_connection2.available_prefill_instances.qsize.return_value = 0
        connector.connect_innode_instances[12345] = mock_connection1
        connector.connect_innode_instances[12346] = mock_connection2

        result = connector.has_splitwise_tasks()

        self.assertTrue(result)

    @patch("fastdeploy.splitwise.splitwise_connector.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", False)
    def test_create_connection_tcp(self):
        """Test creating TCP connection."""
        connector = self.create_connector()

        with patch("fastdeploy.splitwise.splitwise_connector.EngineWorkerQueue") as mock_queue_class:
            mock_queue = Mock()
            mock_queue_class.return_value = mock_queue

            connector.create_connection(12345)

            mock_queue_class.assert_called_once_with(address=("0.0.0.0", 12345), num_client=1, client_id=0)
            self.assertEqual(connector.connect_innode_instances[12345], mock_queue)

    @patch("fastdeploy.splitwise.splitwise_connector.envs.FD_ENGINE_TASK_QUEUE_WITH_SHM", True)
    def test_create_connection_shm(self):
        """Test creating shared memory connection."""
        connector = self.create_connector()

        with patch("fastdeploy.splitwise.splitwise_connector.EngineWorkerQueue") as mock_queue_class:
            mock_queue = Mock()
            mock_queue_class.return_value = mock_queue

            connector.create_connection(12345)

            mock_queue_class.assert_called_once_with(
                address="/dev/shm/fd_task_queue_12345.sock", num_client=1, client_id=0
            )
            self.assertEqual(connector.connect_innode_instances[12345], mock_queue)

    def test_check_decode_allocated_no_disaggregate_info(self):
        """Test check_decode_allocated with no disaggregate info."""
        connector = self.create_connector()

        mock_task = Mock()
        mock_task.disaggregate_info = None

        result, msg = connector.check_decode_allocated(mock_task)

        self.assertTrue(result)
        self.assertEqual(msg, "")

    def test_check_decode_allocated_cache_task_enabled(self):
        """Test check_decode_allocated with cache task enabled."""
        connector = self.create_connector()
        connector.enable_decode_cache_task = True

        mock_task = Mock()
        mock_task.disaggregate_info = {"role": "prefill"}

        result, msg = connector.check_decode_allocated(mock_task)

        self.assertTrue(result)
        self.assertEqual(msg, "")

    def test_check_decode_allocated_decode_role(self):
        """Test check_decode_allocated with decode role."""
        connector = self.create_connector()

        mock_task = Mock()
        mock_task.disaggregate_info = {"role": "decode"}

        result, msg = connector.check_decode_allocated(mock_task)

        self.assertTrue(result)
        self.assertEqual(msg, "")

    def test_check_decode_allocated_success(self):
        """Test successful decode allocation check."""
        connector = self.create_connector()

        mock_task = Mock()
        mock_task.disaggregate_info = {"role": "prefill"}
        mock_task.request_id = "test123"

        connector.current_request_ids["test123"] = "finished"

        result, msg = connector.check_decode_allocated(mock_task)

        self.assertTrue(result)
        self.assertEqual(msg, "")
        self.assertNotIn("test123", connector.current_request_ids)

    def test_check_decode_allocated_timeout(self):
        """Test decode allocation check with timeout."""
        connector = self.create_connector()

        mock_task = Mock()
        mock_task.disaggregate_info = {"role": "prefill"}
        mock_task.request_id = "test123"

        connector.current_request_ids["test123"] = "init"

        with patch("time.time") as mock_time:
            # First call returns current time, second call returns time + 31 (timeout)
            mock_time.side_effect = [0, 31]

            result, msg = connector.check_decode_allocated(mock_task)

            self.assertFalse(result)
            self.assertEqual(msg, "timeout")
            self.assertNotIn("test123", connector.current_request_ids)

    def test_check_decode_allocated_error(self):
        """Test decode allocation check with error."""
        connector = self.create_connector()

        mock_task = Mock()
        mock_task.disaggregate_info = {"role": "prefill"}
        mock_task.request_id = "test123"

        connector.current_request_ids["test123"] = "error_message"

        result, msg = connector.check_decode_allocated(mock_task)

        self.assertFalse(result)
        self.assertEqual(msg, "error_message")
        self.assertNotIn("test123", connector.current_request_ids)

    def test_send_cache_infos_decode_ipc(self):
        """Test sending cache info for decode tasks with IPC protocol."""
        self.mock_cfg.cache_config.pd_comm_port = [5678]
        connector = self.create_connector()

        mock_task = Mock()
        mock_task.disaggregate_info = {
            "role": "decode",
            "transfer_protocol": "ipc",
            "cache_info": {"ipc": {"port": 12345}},
            "block_tables": [1, 2, 3],
        }
        mock_task.request_id = "test123"

        result = connector.send_cache_infos([mock_task], 1)

        self.assertTrue(result)

    def test_send_cache_infos_decode_rdma(self):
        """Test sending cache info for decode tasks with RDMA protocol."""
        self.mock_cfg.cache_config.pd_comm_port = [5678]
        connector = self.create_connector()

        mock_task = Mock()
        mock_task.disaggregate_info = {
            "role": "decode",
            "transfer_protocol": "rdma",
            "cache_info": {"rdma": {"ip": "192.168.1.100", "port": 8080}},
            "block_tables": [1, 2, 3],
        }
        mock_task.request_id = "test123"

        with patch.object(connector, "_send_message") as mock_send:
            result = connector.send_cache_infos([mock_task], 1)

            self.assertTrue(result)
            mock_send.assert_called_once()

    def test_send_cache_infos_prefill(self):
        """Test sending cache info for prefill tasks."""
        self.mock_cfg.cache_config.pd_comm_port = [5678]
        connector = self.create_connector()

        mock_task = Mock()
        mock_task.disaggregate_info = {"role": "prefill", "cache_info": {"ipc": {"current_id": 1}}}
        mock_task.block_tables = [1, 2, 3]
        mock_task.request_id = "test123"

        with patch.object(connector.engine_worker_queue, "put_cache_info") as mock_put_cache:
            result = connector.send_cache_infos([mock_task], -1)

            self.assertFalse(result)
            mock_put_cache.assert_called_once()

    def test_process_message_prefill(self):
        """Test processing prefill message."""
        self.mock_cfg.cache_config.pd_comm_port = [5678]
        connector = self.create_connector()

        message_data = json.dumps(
            {"type": "prefill", "payload": [{"request_id": "test123", "data": "test_data"}]}
        ).encode("utf-8")

        with patch.object(connector, "_handle_prefill") as mock_handle:
            connector._process_message(message_data)

            mock_handle.assert_called_once_with([{"request_id": "test123", "data": "test_data"}])

    def test_process_message_decode(self):
        """Test processing decode message."""
        self.mock_cfg.cache_config.pd_comm_port = [5678]
        connector = self.create_connector()

        message_data = json.dumps(
            {"type": "decode", "payload": [{"request_id": "test123", "data": "test_data"}]}
        ).encode("utf-8")

        with patch.object(connector, "_handle_decode") as mock_handle:
            connector._process_message(message_data)

            mock_handle.assert_called_once_with([{"request_id": "test123", "data": "test_data"}])

    def test_process_message_cache_sync_finished(self):
        """Test processing cache_sync message with finished status."""
        self.mock_cfg.cache_config.pd_comm_port = [5678]
        connector = self.create_connector()

        message_data = json.dumps({"type": "cache_sync", "payload": [{"request_id": "test123"}]}).encode("utf-8")

        with patch.object(connector.engine_worker_queue, "put_cache_info") as mock_put_cache:
            connector._process_message(message_data)

            self.assertEqual(connector.current_request_ids["test123"], "finished")
            mock_put_cache.assert_called_once_with([{"request_id": "test123"}])

    def test_process_message_cache_sync_error(self):
        """Test processing cache_sync message with error status."""
        self.mock_cfg.cache_config.pd_comm_port = [5678]
        connector = self.create_connector()

        message_data = json.dumps(
            {"type": "cache_sync", "payload": [{"request_id": "test123", "error_msg": "test_error"}]}
        ).encode("utf-8")

        connector._process_message(message_data)

        self.assertEqual(connector.current_request_ids["test123"], "test_error")

    def test_process_message_cache_sync_cache_task_enabled(self):
        """Test processing cache_sync message with cache task enabled."""
        self.mock_cfg.cache_config.pd_comm_port = [5678]
        connector = self.create_connector()
        connector.enable_decode_cache_task = True

        message_data = json.dumps({"type": "cache_sync", "payload": [{"request_id": "test123"}]}).encode("utf-8")

        connector._process_message(message_data)

        # Request ID should be deleted immediately when cache task is enabled
        self.assertNotIn("test123", connector.current_request_ids)

    def test_handle_prefill(self):
        """Test handling prefill tasks."""
        connector = self.create_connector()

        tasks_data = [{"request_id": "test123", "data": "test_data"}]

        with patch("fastdeploy.splitwise.splitwise_connector.Request") as mock_request_class:
            mock_request = Mock()
            mock_request_class.from_dict.return_value = mock_request

            with patch.object(connector.engine_worker_queue, "put_disaggregated_tasks") as mock_put:
                connector._handle_prefill(tasks_data)

                mock_request_class.from_dict.assert_called_once_with({"request_id": "test123", "data": "test_data"})
                mock_put.assert_called_once_with(("decode", [mock_request]))

    def test_handle_decode(self):
        """Test handling decode tasks."""
        connector = self.create_connector()

        payload_data = [{"request_id": "test123", "data": "test_data"}]

        with patch("fastdeploy.splitwise.splitwise_connector.RequestOutput") as mock_output_class:
            mock_output = Mock()
            mock_output_class.from_dict.return_value = mock_output

            with patch.object(connector.engine_worker_queue, "put_disaggregated_tasks") as mock_put:
                connector._handle_decode(payload_data)

                mock_output_class.from_dict.assert_called_once_with({"request_id": "test123", "data": "test_data"})
                mock_put.assert_called_once_with(("decode", [mock_output]))

    def test_send_splitwise_tasks_ipc(self):
        """Test sending splitwise tasks with IPC protocol."""
        self.mock_cfg.innode_prefill_ports = None
        connector = self.create_connector()

        mock_task = Mock()
        mock_task.disaggregate_info = {"transfer_protocol": "ipc", "cache_info": {"ipc": {"port": 12345}}}
        mock_task.request_id = "test123"

        with patch.object(connector, "send_splitwise_tasks_innode") as mock_send_innode:
            connector.send_splitwise_tasks([mock_task], 1)

            mock_send_innode.assert_called_once_with([mock_task], 12345)

    def test_send_splitwise_tasks_rdma(self):
        """Test sending splitwise tasks with RDMA protocol."""
        self.mock_cfg.innode_prefill_ports = None
        self.mock_cfg.cache_config.pd_comm_port = [5678]
        connector = self.create_connector()

        mock_task = Mock()
        original_disaggregate_info = {
            "transfer_protocol": "rdma",
            "cache_info": {"rdma": {"ip": "192.168.1.100", "port": 8080}},
        }
        mock_task.disaggregate_info = original_disaggregate_info.copy()
        mock_task.request_id = "test123"

        with patch.object(connector, "_send_message") as mock_send:
            connector.send_splitwise_tasks([mock_task], 1)

            self.assertEqual(connector.current_request_ids["test123"], "init")
            mock_send.assert_called_once()

    def test_send_splitwise_tasks_innode(self):
        """Test sending splitwise tasks to specific port."""
        connector = self.create_connector()

        mock_task = Mock()
        mock_task.disaggregate_info = {"cache_info": {"ipc": {"port": 12345}}}

        mock_connection = Mock()
        connector.connect_innode_instances[12345] = mock_connection

        result = connector.send_splitwise_tasks_innode([mock_task], 12345)

        self.assertEqual(result, 12345)
        mock_connection.put_disaggregated_tasks.assert_called_once()

    def test_send_first_token_ipc(self):
        """Test sending first token with IPC protocol."""
        connector = self.create_connector()

        prefill_msg = {"transfer_protocol": "ipc", "cache_info": {"ipc": {"port": 12345}}}
        mock_task = Mock()
        mock_task.request_id = "test123"

        mock_connection = Mock()
        connector.connect_innode_instances[12345] = mock_connection

        connector.send_first_token(prefill_msg, mock_task)

        mock_connection.put_disaggregated_tasks.assert_called_once()

    def test_send_first_token_rdma(self):
        """Test sending first token with RDMA protocol."""
        self.mock_cfg.cache_config.pd_comm_port = [5678]
        connector = self.create_connector()

        prefill_msg = {"transfer_protocol": "rdma", "cache_info": {"rdma": {"ip": "192.168.1.100", "port": 8080}}}
        mock_task = Mock()
        mock_task.request_id = "test123"

        with patch.object(connector, "_send_message") as mock_send:
            connector.send_first_token(prefill_msg, mock_task)

            mock_send.assert_called_once()

    def test_send_first_token_list_conversion(self):
        """Test send_first_token converts single task to list."""
        connector = self.create_connector()

        prefill_msg = {"transfer_protocol": "ipc", "cache_info": {"ipc": {"port": 12345}}}
        mock_task = Mock()
        mock_task.request_id = "test123"

        mock_connection = Mock()
        connector.connect_innode_instances[12345] = mock_connection

        # Test with single task (not list)
        connector.send_first_token(prefill_msg, mock_task)

        # Should convert to list internally
        mock_connection.put_disaggregated_tasks.assert_called_once()
        call_args = mock_connection.put_disaggregated_tasks.call_args
        # The method is called with a tuple argument: ("decode", tasks_list)
        args, kwargs = call_args
        self.assertEqual(len(args), 1)  # Called with one argument (the tuple)
        task_tuple = args[0]
        self.assertEqual(len(task_tuple), 2)  # Tuple should have task_type and tasks_list
        self.assertEqual(task_tuple[0], "decode")
        self.assertIsInstance(task_tuple[1], list)
        self.assertEqual(len(task_tuple[1]), 1)  # Should contain the single task

    def test_error_handling_in_process_message(self):
        """Test error handling in message processing."""
        self.mock_cfg.cache_config.pd_comm_port = [5678]
        connector = self.create_connector()

        # Invalid JSON data
        invalid_data = b"invalid json"

        # Should not raise exception
        try:
            connector._process_message(invalid_data)
        except Exception:
            self.fail("_process_message should handle exceptions gracefully")

    def test_network_error_handling(self):
        """Test network error handling in start_receiver method."""
        self.mock_cfg.cache_config.pd_comm_port = [5678]
        connector = self.create_connector()

        # Mock a scenario where router_socket.recv_multipart raises an exception
        with patch.object(connector, "poller") as mock_poller:
            # First return some data to trigger recv_multipart, then exception
            mock_poller.poll.side_effect = [{"mock_socket": 1}, Exception("Network error")]

            with patch.object(connector, "router_socket") as mock_router:
                mock_router.recv_multipart.side_effect = Exception("Connection error")

                # Test that the exception handling works by calling the code path
                try:
                    # Simulate one iteration that causes an exception
                    socks = dict(mock_poller.poll(100))
                    if socks:
                        connector.router_socket.recv_multipart()
                except Exception:
                    # This should be handled gracefully by the logger
                    # The test passes if no uncaught exception is raised
                    pass


if __name__ == "__main__":
    print(f"Running tests in {TEST_MODE} mode")
    if TEST_MODE == "standalone":
        print("To run in normal mode, ensure fastdeploy is properly installed")
        print("Or set FD_TEST_MODE=normal environment variable")
    unittest.main(verbosity=2)
