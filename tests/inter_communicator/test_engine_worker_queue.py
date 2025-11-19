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

import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

try:
    import paddle

    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    paddle = None

# Handle import gracefully
try:
    from fastdeploy.inter_communicator.engine_worker_queue import EngineWorkerQueue

    ENGINE_WORKER_QUEUE_AVAILABLE = True
except ImportError as e:
    ENGINE_WORKER_QUEUE_AVAILABLE = False
    print(f"Warning: Could not import EngineWorkerQueue: {e}")


class MockTask:
    """Mock task object for testing."""

    def __init__(self, task_id="test_task", multimodal_inputs=None):
        self.task_id = task_id
        self.multimodal_inputs = multimodal_inputs or {}


class TestEngineWorkerQueue(unittest.TestCase):
    """Test cases for EngineWorkerQueue class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        if not ENGINE_WORKER_QUEUE_AVAILABLE:
            self.skipTest("EngineWorkerQueue not available")

        self.test_address = ("127.0.0.1", 0)  # Use port 0 for automatic port assignment
        self.test_authkey = b"test_auth_key"
        self.test_num_client = 2
        self.test_client_id = 0
        self.test_local_data_parallel_size = 1
        self.test_local_data_parallel_id = 0

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up any queue managers that might be running
        pass

    def test_server_initialization(self):
        """Test server-side initialization."""
        queue = EngineWorkerQueue(
            address=self.test_address,
            authkey=self.test_authkey,
            is_server=True,
            num_client=self.test_num_client,
            local_data_parallel_size=self.test_local_data_parallel_size,
        )

        # Verify server-specific attributes
        self.assertTrue(queue.is_server)
        self.assertEqual(queue.num_client, self.test_num_client)
        self.assertEqual(queue.local_data_parallel_size, self.test_local_data_parallel_size)
        self.assertIsNotNone(queue.manager)
        self.assertIsNotNone(queue.address)

        # Verify initialization of shared resources
        self.assertEqual(len(queue.tasks_init), self.test_local_data_parallel_size)
        self.assertEqual(len(queue.client_read_flag_init), self.test_local_data_parallel_size)
        self.assertEqual(len(queue.lock_init), self.test_local_data_parallel_size)

        # Cleanup
        queue.cleanup()

    def test_client_initialization(self):
        """Test client-side initialization."""
        # First create a server
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=self.test_num_client
        )

        # Get the actual server address
        server_address = server_queue.address

        # Create a client
        client_queue = EngineWorkerQueue(
            address=server_address,
            authkey=self.test_authkey,
            is_server=False,
            num_client=self.test_num_client,
            client_id=self.test_client_id,
        )

        # Verify client-specific attributes
        self.assertFalse(client_queue.is_server)
        self.assertEqual(client_queue.client_id, self.test_client_id)
        self.assertEqual(client_queue.num_client, self.test_num_client)
        self.assertIsNotNone(client_queue.manager)

        # Verify proxy objects are initialized
        self.assertIsNotNone(client_queue.tasks)
        self.assertIsNotNone(client_queue.client_read_flag)
        self.assertIsNotNone(client_queue.lock)

        # Cleanup
        client_queue.cleanup()
        server_queue.cleanup()

    def test_invalid_client_id(self):
        """Test client initialization with invalid client_id."""
        # Create a server first
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=self.test_num_client
        )

        server_address = server_queue.address

        # Test invalid client_id (negative)
        with self.assertRaises(AssertionError):
            EngineWorkerQueue(
                address=server_address,
                authkey=self.test_authkey,
                is_server=False,
                num_client=self.test_num_client,
                client_id=-1,
            )

        # Test invalid client_id (too large)
        with self.assertRaises(AssertionError):
            EngineWorkerQueue(
                address=server_address,
                authkey=self.test_authkey,
                is_server=False,
                num_client=self.test_num_client,
                client_id=self.test_num_client,
            )

        server_queue.cleanup()

    def test_get_server_port(self):
        """Test get_server_port method."""
        server_queue = EngineWorkerQueue(address=self.test_address, authkey=self.test_authkey, is_server=True)

        port = server_queue.get_server_port()
        self.assertIsInstance(port, int)
        self.assertGreater(port, 0)

        # Test calling from client
        client_queue = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=1, client_id=0
        )

        with self.assertRaises(RuntimeError):
            client_queue.get_server_port()

        client_queue.cleanup()
        server_queue.cleanup()

    def test_connect_with_retry_success(self):
        """Test successful connection with retry."""
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=1
        )

        # This should succeed immediately
        client_queue = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=1, client_id=0
        )

        self.assertIsNotNone(client_queue.manager)

        client_queue.cleanup()
        server_queue.cleanup()

    @patch("time.sleep")
    def test_connect_with_retry_failure(self, mock_sleep):
        """Test connection retry failure."""
        # Try to connect to a non-existent server
        with self.assertRaises(ConnectionError):
            EngineWorkerQueue(
                address=("127.0.0.1", 9999),  # Non-existent port
                authkey=self.test_authkey,
                is_server=False,
                num_client=1,
                client_id=0,
            )

        # Verify sleep was called
        mock_sleep.assert_called()

    def test_to_tensor(self):
        """Test tensor conversion static method."""
        if not PADDLE_AVAILABLE:
            self.skipTest("PaddlePaddle not available")

        # Create mock tasks with numpy arrays
        image_array = np.random.rand(3, 224, 224).astype(np.float32)
        patch_idx_array = np.array([0, 1, 2])

        task1 = MockTask(
            task_id="task1",
            multimodal_inputs={"images": image_array, "patch_idx": patch_idx_array, "text": "hello world"},
        )

        task2 = MockTask(task_id="task2", multimodal_inputs=None)

        tasks = ([task1, task2], 2)

        # Mock environment variables
        with patch("fastdeploy.inter_communicator.engine_worker_queue.envs") as mock_envs:
            mock_envs.FD_ENABLE_MAX_PREFILL = False
            mock_envs.FD_ENABLE_E2W_TENSOR_CONVERT = True

            EngineWorkerQueue.to_tensor(tasks)

        # Verify conversion
        batch_tasks, _ = tasks
        self.assertIsInstance(batch_tasks[0].multimodal_inputs["images"], paddle.Tensor)
        self.assertIsInstance(batch_tasks[0].multimodal_inputs["patch_idx"], paddle.Tensor)
        self.assertEqual(batch_tasks[0].multimodal_inputs["text"], "hello world")

        # Task 2 should be unchanged
        self.assertIsNone(batch_tasks[1].multimodal_inputs)

    def test_to_tensor_disabled(self):
        """Test tensor conversion when disabled."""
        image_array = np.random.rand(3, 224, 224).astype(np.float32)
        task = MockTask(task_id="task1", multimodal_inputs={"images": image_array})

        tasks = ([task], 1)

        # Mock environment variables to disable conversion
        with patch("fastdeploy.inter_communicator.engine_worker_queue.envs") as mock_envs:
            mock_envs.FD_ENABLE_MAX_PREFILL = False
            mock_envs.FD_ENABLE_E2W_TENSOR_CONVERT = False

            EngineWorkerQueue.to_tensor(tasks)

        # Verify no conversion occurred
        batch_tasks, _ = tasks
        self.assertIsInstance(batch_tasks[0].multimodal_inputs["images"], np.ndarray)

    def test_to_numpy(self):
        """Test numpy conversion static method."""
        if not PADDLE_AVAILABLE:
            self.skipTest("PaddlePaddle not available")

        # Create mock tasks with paddle tensors
        image_tensor = paddle.randn([3, 224, 224])

        task = MockTask(task_id="task1", multimodal_inputs={"images": image_tensor})

        tasks_list = [([task], 1)]

        # Mock environment variable
        with patch("fastdeploy.inter_communicator.engine_worker_queue.envs") as mock_envs:
            mock_envs.FD_ENABLE_MAX_PREFILL = True

            EngineWorkerQueue.to_numpy(tasks_list)

        # Verify conversion
        batch_tasks, _ = tasks_list[0]
        self.assertIsInstance(batch_tasks[0].multimodal_inputs["images"], np.ndarray)

    def test_put_and_get_tasks(self):
        """Test putting and getting tasks."""
        # Create server and client
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=1
        )

        client_queue = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=1, client_id=0
        )

        # Create test tasks
        test_tasks = ["task1", "task2", "task3"]

        # Put tasks
        server_queue.put_tasks(test_tasks)

        # Get tasks
        retrieved_tasks, all_read = client_queue.get_tasks()

        self.assertEqual(retrieved_tasks, [test_tasks])
        self.assertTrue(all_read)

        # Test num_tasks
        self.assertEqual(server_queue.num_tasks(), 0)
        self.assertEqual(client_queue.num_tasks(), 0)

        client_queue.cleanup()
        server_queue.cleanup()

    def test_put_and_get_cache_info(self):
        """Test putting and getting cache info."""
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=1
        )

        client_queue = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=1, client_id=0
        )

        # Create test cache info
        cache_info = ["cache1", "cache2", "cache3"]

        # Put cache info
        server_queue.put_cache_info(cache_info)

        # Get cache info
        retrieved_cache = client_queue.get_cache_info()

        self.assertEqual(retrieved_cache, cache_info)

        # Test num_cache_infos
        self.assertEqual(server_queue.num_cache_infos(), 0)
        self.assertEqual(client_queue.num_cache_infos(), 0)

        client_queue.cleanup()
        server_queue.cleanup()

    def test_put_and_get_connect_rdma_task(self):
        """Test putting and getting RDMA connect tasks."""
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=1
        )

        client_queue = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=1, client_id=0
        )

        # Create test RDMA task
        rdma_task = {"task_id": "rdma_task_1", "data": "test_data"}

        # Put RDMA task
        server_queue.put_connect_rdma_task(rdma_task)

        # Get RDMA task
        retrieved_task, all_read = client_queue.get_connect_rdma_task()

        self.assertEqual(retrieved_task, rdma_task)
        self.assertTrue(all_read)

        client_queue.cleanup()
        server_queue.cleanup()

    def test_put_and_get_connect_rdma_task_response(self):
        """Test putting and getting RDMA connect task responses."""
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=1
        )

        client_queue = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=1, client_id=0
        )

        # Create test RDMA task response
        rdma_response = {"success": True, "task_id": "rdma_task_1"}

        # Put RDMA task response
        all_put = client_queue.put_connect_rdma_task_response(rdma_response)
        self.assertTrue(all_put)

        # Get RDMA task response
        retrieved_response = server_queue.get_connect_rdma_task_response()

        self.assertEqual(retrieved_response, rdma_response)

        client_queue.cleanup()
        server_queue.cleanup()

    def test_put_and_get_finished_req(self):
        """Test putting and getting finished requests."""
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=1
        )

        client_queue = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=1, client_id=0
        )

        # Create test finished request
        finished_req = ["req1", {"status": "completed"}]

        # Put finished request
        all_put = client_queue.put_finished_req(finished_req)
        self.assertTrue(all_put)

        # Get finished request
        retrieved_req = server_queue.get_finished_req()

        self.assertEqual(retrieved_req, [finished_req])

        client_queue.cleanup()
        server_queue.cleanup()

    def test_put_and_get_finished_add_cache_task_req(self):
        """Test putting and getting finished add cache task requests."""
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=1
        )

        client_queue = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=1, client_id=0
        )

        # Create test finished add cache task request
        add_cache_req = "cache_req_1"

        # Put finished add cache task request
        all_put = client_queue.put_finished_add_cache_task_req(add_cache_req)
        self.assertTrue(all_put)

        # Get finished add cache task request
        retrieved_req = server_queue.get_finished_add_cache_task_req()

        self.assertEqual(retrieved_req, add_cache_req)

        client_queue.cleanup()
        server_queue.cleanup()

    def test_disaggregate_queue_operations(self):
        """Test disaggregated queue operations."""
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=1
        )

        client_queue = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=1, client_id=0
        )

        # Test empty queue
        self.assertTrue(client_queue.disaggregate_queue_empty())

        # Put disaggregated tasks
        test_items = ["item1", "item2", "item3"]
        for item in test_items:
            client_queue.put_disaggregated_tasks(item)

        # Check queue is not empty
        self.assertFalse(client_queue.disaggregate_queue_empty())

        # Get disaggregated tasks
        retrieved_items = client_queue.get_disaggregated_tasks()
        self.assertEqual(retrieved_items, test_items)

        # Check queue is empty again
        self.assertTrue(client_queue.disaggregate_queue_empty())

        client_queue.cleanup()
        server_queue.cleanup()

    def test_get_prefill_instances(self):
        """Test getting prefill instances."""
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=1
        )

        client_queue = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=1, client_id=0
        )

        # Test empty queue
        result = client_queue.get_prefill_instances()
        self.assertEqual(result, 0)

        # Put available instances
        client_queue.available_prefill_instances.put(2)

        # Get instances
        result = client_queue.get_prefill_instances()
        self.assertEqual(result, 2)

        client_queue.cleanup()
        server_queue.cleanup()

    def test_clear_data(self):
        """Test clearing data from the queue."""
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=1
        )

        client_queue = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=1, client_id=0
        )

        # Put some tasks
        test_tasks = ["task1", "task2"]
        server_queue.put_tasks(test_tasks)

        # Verify tasks are there
        self.assertEqual(client_queue.num_tasks(), 1)

        # Clear data
        server_queue.clear_data()

        # Verify tasks are cleared
        self.assertEqual(client_queue.num_tasks(), 0)

        client_queue.cleanup()
        server_queue.cleanup()

    def test_cleanup(self):
        """Test cleanup method."""
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=1
        )

        # Verify manager exists
        self.assertIsNotNone(server_queue.manager)

        # Cleanup
        server_queue.cleanup()

        # Note: After shutdown, the manager might still exist but won't be functional
        # This is expected behavior for multiprocessing managers

    def test_multi_client_scenario(self):
        """Test scenario with multiple clients."""
        num_clients = 2

        # Create server
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=num_clients
        )

        # Create multiple clients
        clients = []
        for i in range(num_clients):
            client = EngineWorkerQueue(
                address=server_queue.address,
                authkey=self.test_authkey,
                is_server=False,
                num_client=num_clients,
                client_id=i,
            )
            clients.append(client)

        # Verify all clients are connected
        self.assertEqual(server_queue.connected_client_counter.get(), num_clients)

        # Test task distribution
        test_tasks = ["shared_task"]
        server_queue.put_tasks(test_tasks)

        # All clients should be able to get the tasks
        for client in clients:
            tasks, all_read = client.get_tasks()
            self.assertEqual(tasks, [test_tasks])

        # Cleanup
        for client in clients:
            client.cleanup()
        server_queue.cleanup()

    def test_thread_safety(self):
        """Test thread safety of queue operations."""
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=2
        )

        client1 = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=2, client_id=0
        )

        client2 = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=2, client_id=1
        )

        results = []

        def worker_put_tasks():
            for i in range(10):
                test_tasks = [f"task_{i}"]
                server_queue.put_tasks(test_tasks)
                time.sleep(0.01)

        def worker_get_tasks(client, client_id):
            for i in range(10):
                tasks, all_read = client.get_tasks()
                if tasks:
                    results.append((client_id, tasks))
                time.sleep(0.01)

        # Start threads
        put_thread = threading.Thread(target=worker_put_tasks)
        get_thread1 = threading.Thread(target=worker_get_tasks, args=(client1, 0))
        get_thread2 = threading.Thread(target=worker_get_tasks, args=(client2, 1))

        put_thread.start()
        get_thread1.start()
        get_thread2.start()

        # Wait for completion
        put_thread.join()
        get_thread1.join()
        get_thread2.join()

        # Verify some tasks were retrieved
        self.assertGreater(len(results), 0)

        # Cleanup
        client1.cleanup()
        client2.cleanup()
        server_queue.cleanup()

    def test_barrier_operations(self):
        """Test barrier synchronization operations."""
        num_clients = 2

        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=num_clients
        )

        clients = []
        for i in range(num_clients):
            client = EngineWorkerQueue(
                address=server_queue.address,
                authkey=self.test_authkey,
                is_server=False,
                num_client=num_clients,
                client_id=i,
            )
            clients.append(client)

        # Test that barriers are properly initialized
        self.assertIsNotNone(clients[0].finish_request_barrier)
        self.assertIsNotNone(clients[0].connect_task_barrier)
        self.assertIsNotNone(clients[0].connect_task_response_barrier)
        self.assertIsNotNone(clients[0].finish_add_cache_task_barrier)
        self.assertIsNotNone(clients[0].begin_send_cache_barrier)
        self.assertIsNotNone(clients[0].finish_send_cache_barrier)
        self.assertIsNotNone(clients[0].cache_info_barrier)
        self.assertIsNotNone(clients[0].worker_process_tp_barrier)

        # Cleanup
        for client in clients:
            client.cleanup()
        server_queue.cleanup()

    def test_data_parallel_operations(self):
        """Test operations with multiple data parallel groups."""
        data_parallel_size = 2
        num_clients = 2

        server_queue = EngineWorkerQueue(
            address=self.test_address,
            authkey=self.test_authkey,
            is_server=True,
            num_client=num_clients,
            local_data_parallel_size=data_parallel_size,
        )

        # Create clients for each data parallel group
        clients = []
        for dp_id in range(data_parallel_size):
            for client_id in range(num_clients):
                client = EngineWorkerQueue(
                    address=server_queue.address,
                    authkey=self.test_authkey,
                    is_server=False,
                    num_client=num_clients,
                    client_id=client_id,
                    local_data_parallel_size=data_parallel_size,
                    local_data_parallel_id=dp_id,
                )
                clients.append(client)

        # Test operations in different data parallel groups
        test_tasks_dp0 = ["dp0_task"]

        # Put tasks for different data parallel groups
        server_queue.put_cache_info(test_tasks_dp0)  # Uses dp_id=0 by default
        self.assertEqual(len(clients[0].get_cache_info()), 1)

        # Cleanup
        for client in clients:
            client.cleanup()
        server_queue.cleanup()

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=1
        )

        client_queue = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=1, client_id=0
        )

        # Test empty operations
        empty_tasks, all_read = client_queue.get_tasks()
        self.assertEqual(empty_tasks, [])
        self.assertFalse(all_read)

        # Test empty cache info
        empty_cache = client_queue.get_cache_info()
        self.assertEqual(empty_cache, [])

        # Test empty RDMA task
        empty_rdma, all_read = client_queue.get_connect_rdma_task()
        self.assertIsNone(empty_rdma)
        self.assertFalse(all_read)

        # Test empty RDMA response
        empty_response = server_queue.get_connect_rdma_task_response()
        self.assertIsNone(empty_response)

        # Test empty finished requests
        empty_finished = server_queue.get_finished_req()
        self.assertEqual(empty_finished, [])

        # Test empty finished add cache task
        empty_add_cache = server_queue.get_finished_add_cache_task_req()
        self.assertEqual(empty_add_cache, [])

        # Test empty disaggregated tasks
        empty_disaggregated = client_queue.get_disaggregated_tasks()
        self.assertIsNone(empty_disaggregated)

        client_queue.cleanup()
        server_queue.cleanup()

    def test_tensor_conversion_edge_cases(self):
        """Test tensor conversion edge cases."""
        if not PADDLE_AVAILABLE:
            self.skipTest("PaddlePaddle not available")

        # Test to_tensor with multimodal inputs containing various data types
        task_with_mixed_data = MockTask(
            task_id="mixed_task",
            multimodal_inputs={
                "images": np.random.rand(3, 224, 224).astype(np.float32),
                "patch_idx": np.array([0, 1, 2], dtype=np.int32),
                "token_type_ids": np.array([0, 1, 1, 0], dtype=np.int64),
                "position_ids": np.array([0, 1, 2, 3], dtype=np.int32),
                "attention_mask_offset": np.array([1, 1, 1, 1], dtype=np.int32),
                "text": "sample text",
                "none_value": None,
                "empty_array": np.array([]),
            },
        )

        tasks = ([task_with_mixed_data], 1)

        with patch("fastdeploy.inter_communicator.engine_worker_queue.envs") as mock_envs:
            mock_envs.FD_ENABLE_MAX_PREFILL = False
            mock_envs.FD_ENABLE_E2W_TENSOR_CONVERT = True

            EngineWorkerQueue.to_tensor(tasks)

        batch_tasks, _ = tasks
        task = batch_tasks[0]

        # Verify tensor conversions
        self.assertIsInstance(task.multimodal_inputs["images"], paddle.Tensor)
        self.assertIsInstance(task.multimodal_inputs["patch_idx"], paddle.Tensor)
        self.assertIsInstance(task.multimodal_inputs["token_type_ids"], paddle.Tensor)
        self.assertIsInstance(task.multimodal_inputs["position_ids"], paddle.Tensor)
        self.assertIsInstance(task.multimodal_inputs["attention_mask_offset"], paddle.Tensor)

        # Verify non-tensor data is unchanged
        self.assertEqual(task.multimodal_inputs["text"], "sample text")
        self.assertIsNone(task.multimodal_inputs["none_value"])
        self.assertEqual(task.multimodal_inputs["empty_array"].size, 0)

    def test_lock_timeout_scenarios(self):
        """Test scenarios that might cause lock timeouts."""
        server_queue = EngineWorkerQueue(
            address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=2
        )

        client1 = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=2, client_id=0
        )

        client2 = EngineWorkerQueue(
            address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=2, client_id=1
        )

        # Test multiple rapid put operations
        for i in range(5):
            server_queue.put_tasks([f"rapid_task_{i}"])

        # Test multiple rapid get operations
        for i in range(5):
            tasks, _ = client1.get_tasks()
            if tasks:
                self.assertTrue(len(tasks) > 0)

        # Test cache info operations
        for i in range(3):
            server_queue.put_cache_info([f"cache_info_{i}"])
            cache_info = client2.get_cache_info()
            # Cache info might be empty if already consumed
            self.assertIsInstance(cache_info, list)

        client1.cleanup()
        client2.cleanup()
        server_queue.cleanup()

    def test_memory_cleanup(self):
        """Test memory cleanup and resource management."""
        # Test that multiple queue instances can be created and cleaned up
        for i in range(3):
            server_queue = EngineWorkerQueue(
                address=self.test_address, authkey=self.test_authkey, is_server=True, num_client=1
            )

            client_queue = EngineWorkerQueue(
                address=server_queue.address, authkey=self.test_authkey, is_server=False, num_client=1, client_id=0
            )

            # Perform some operations
            server_queue.put_tasks([f"cleanup_test_{i}"])
            tasks, _ = client_queue.get_tasks()

            # Cleanup
            client_queue.cleanup()
            server_queue.cleanup()

        # If we reach here without exceptions, cleanup is working properly
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
