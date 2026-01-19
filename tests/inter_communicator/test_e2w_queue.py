"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import paddle

# Import fastdeploy modules - these should be available in CI environment
from fastdeploy import envs
from fastdeploy.engine.request import Request
from fastdeploy.inter_communicator.engine_worker_queue import EngineWorkerQueue
from fastdeploy.utils import to_numpy, to_tensor


class TestEngineWorkerQueue(unittest.TestCase):

    def test_get_server_port_error_on_client(self):
        """Test get_server_port raises RuntimeError when called on client instance (line 482)"""
        # Create client instance (is_server=False)
        client_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=False,
            num_client=1,
            client_id=0
        )

        # Should raise RuntimeError for client instance
        with self.assertRaises(RuntimeError) as cm:
            client_queue.get_server_port()

        self.assertIn("Only the server instance can provide the port", str(cm.exception))

    @patch('fastdeploy.inter_communicator.engine_worker_queue.IPCSignal')
    def test_exist_tasks_multinode_path(self, mock_ipc_signal):
        """Test exist_tasks method for multi-node deployment (line 501)"""
        # Create client instance with non-localhost address (multi-node)
        client_queue = EngineWorkerQueue(
            address=("192.168.1.100", 5000),
            is_server=False,
            num_client=1,
            client_id=0
        )

        # Mock the inter-process signal
        mock_signal = MagicMock()
        mock_signal.get.return_value = 1
        client_queue.exist_tasks_inter_signal = mock_signal

        # Test exist_tasks returns True when signal is 1
        result = client_queue.exist_tasks()
        self.assertTrue(result)
        mock_signal.get.assert_called_once()

        # Test exist_tasks returns False when signal is 0
        mock_signal.get.return_value = 0
        result = client_queue.exist_tasks()
        self.assertFalse(result)

    @patch('fastdeploy.inter_communicator.engine_worker_queue.IPCSignal')
    def test_set_exist_tasks_multinode_path(self, mock_ipc_signal):
        """Test set_exist_tasks method for multi-node deployment (line 518)"""
        # Create client instance with non-localhost address (multi-node)
        client_queue = EngineWorkerQueue(
            address=("192.168.1.100", 5000),
            is_server=False,
            num_client=1,
            client_id=0
        )

        # Mock the inter-process signal
        mock_signal = MagicMock()
        client_queue.exist_tasks_inter_signal = mock_signal

        # Test setting flag to True
        client_queue.set_exist_tasks(True)
        mock_signal.set.assert_called_with(1)

        # Test setting flag to False
        client_queue.set_exist_tasks(False)
        mock_signal.set.assert_called_with(0)

    @patch('multiprocessing.managers.BaseManager.connect')
    @patch('time.sleep')
    def test_connect_with_retry_success(self, mock_sleep, mock_connect):
        """Test _connect_with_retry method successful connection (lines 535-537)"""
        # Create client instance
        client_queue = EngineWorkerQueue(
            address=("127.0.0.1", 5000),
            is_server=False,
            num_client=1,
            client_id=0
        )

        # Mock successful connection
        mock_connect.return_value = None

        # Should not raise exception
        client_queue._connect_with_retry(max_retries=3, interval=0.1)

        # Verify connect was called once
        mock_connect.assert_called_once()

    @patch('multiprocessing.managers.BaseManager.connect')
    @patch('time.sleep')
    def test_connect_with_retry_failure(self, mock_sleep, mock_connect):
        """Test _connect_with_retry method with connection failures (lines 535-537)"""
        # Create client instance
        client_queue = EngineWorkerQueue(
            address=("127.0.0.1", 5000),
            is_server=False,
            num_client=1,
            client_id=0
        )

        # Mock connection failure
        mock_connect.side_effect = ConnectionRefusedError("Connection refused")

        # Should raise ConnectionError after retries
        with self.assertRaises(ConnectionError) as cm:
            client_queue._connect_with_retry(max_retries=2, interval=0.1)

        self.assertIn("TaskQueue cannot connect", str(cm.exception))

        # Verify connect was called max_retries times
        self.assertEqual(mock_connect.call_count, 2)
        # Verify sleep was called (max_retries - 1) times
        self.assertEqual(mock_sleep.call_count, 1)

    def test_num_tasks(self):
        """Test num_tasks method (lines 592-595)"""
        # Create server instance to test
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=1
        )

        # Initially should be 0
        self.assertEqual(server_queue.num_tasks(), 0)

        # Add some tasks directly to test
        test_tasks = [{"task_id": 1, "data": "test"}]
        server_queue.tasks[0].extend(test_tasks)

        # Should return correct count
        self.assertEqual(server_queue.num_tasks(), 1)

    def test_num_cache_infos(self):
        """Test num_cache_infos method (lines 707-710)"""
        # Create server instance to test
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=1
        )

        # Initially should be 0
        self.assertEqual(server_queue.num_cache_infos(), 0)

        # Add some cache infos directly to test
        test_cache_infos = [{"cache_id": 1, "data": "test"}]
        server_queue.cache_infos[0].extend(test_cache_infos)

        # Should return correct count
        self.assertEqual(server_queue.num_cache_infos(), 1)

    def test_put_connect_rdma_task(self):
        """Test put_connect_rdma_task method (lines 598-607)"""
        # Create server instance
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=2
        )

        # Set all clients as read
        server_queue.client_get_connect_task_flag[0][:] = [1, 1]

        # Put RDMA task
        test_task = {"rdma_task": "connect", "params": {"host": "192.168.1.100"}}
        server_queue.put_connect_rdma_task(test_task)

        # Verify task was added and flags reset
        self.assertEqual(server_queue.connect_rdma_tasks[0][0], test_task)
        self.assertEqual(server_queue.client_get_connect_task_flag[0], [0, 0])

    def test_get_connect_rdma_task(self):
        """Test get_connect_rdma_task method (lines 610-619)"""
        # Create server instance
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=2
        )

        # Test empty queue
        result, all_read = server_queue.get_connect_rdma_task()
        self.assertIsNone(result)
        self.assertFalse(all_read)

        # Add task and set client as read
        test_task = {"rdma_task": "connect"}
        server_queue.connect_rdma_tasks[0].append(test_task)
        server_queue.client_get_connect_task_flag[0][0] = 1  # client 0 read
        server_queue.client_get_connect_task_flag[0][1] = 1  # client 1 read

        # Get task
        result, all_read = server_queue.get_connect_rdma_task()
        self.assertEqual(result, test_task)
        self.assertTrue(all_read)
        self.assertEqual(len(server_queue.connect_rdma_tasks[0]), 0)

    def test_put_connect_rdma_task_response(self):
        """Test put_connect_rdma_task_response method (lines 622-633)"""
        # Create server instance
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=2
        )

        # Set flag to allow putting
        server_queue.can_put_next_connect_task_response_flag[0].set(1)

        # Put response
        test_response = {"success": True, "connection_id": 123}
        result = server_queue.put_connect_rdma_task_response(test_response)

        # Verify response was added and return value
        self.assertIn(test_response, server_queue.connect_rdma_task_responses[0])
        self.assertFalse(result)  # Not all clients put yet

    def test_get_connect_rdma_task_response(self):
        """Test get_connect_rdma_task_response method (lines 636-653)"""
        # Create server instance
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=2
        )

        # Test empty responses
        result = server_queue.get_connect_rdma_task_response()
        self.assertIsNone(result)

        # Add responses and set all clients as put
        responses = [
            {"success": True, "id": 1},
            {"success": False, "id": 2}
        ]
        for resp in responses:
            server_queue.connect_rdma_task_responses[0].append(resp)
        server_queue.client_get_connect_task_response_flag[0][:] = [1, 1]

        # Get response
        result = server_queue.get_connect_rdma_task_response()
        self.assertIsNotNone(result)
        self.assertFalse(result["success"])  # Combined result should be False
        self.assertEqual(len(server_queue.connect_rdma_task_responses[0]), 0)

    def test_put_cache_info(self):
        """Test put_cache_info method (lines 660-673)"""
        # Create server instance
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=2
        )

        # Set all clients as read
        server_queue.client_read_info_flag[0][:] = [1, 1]

        # Put cache info
        test_cache_info = [{"cache_key": "key1", "data": "value1"}]
        server_queue.put_cache_info(test_cache_info)

        # Verify cache info was added and flags reset
        self.assertEqual(server_queue.cache_infos[0], test_cache_info)
        self.assertEqual(server_queue.client_read_info_flag[0], [0, 0])

    def test_get_cache_info(self):
        """Test get_cache_info method (lines 675-698, including line 690-692 branch)"""
        # Create server instance
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=2
        )

        # Test when client already read (should return empty)
        server_queue.client_read_info_flag[0][0] = 1  # client 0 already read
        result = server_queue.get_cache_info()
        self.assertEqual(result, [])

        # Reset and add cache info
        server_queue.client_read_info_flag[0][0] = 0  # reset read flag
        test_cache_info = [{"cache_key": "key1", "data": "value1"}]
        server_queue.cache_infos[0].extend(test_cache_info)

        # Get cache info for first time
        result = server_queue.get_cache_info()
        self.assertEqual(result, test_cache_info)
        self.assertEqual(server_queue.client_read_info_flag[0][0], 1)  # marked as read

        # Set all clients as read to trigger cache clearing (lines 690-692)
        server_queue.client_read_info_flag[0][:] = [1, 1]
        server_queue.cache_infos[0].extend(test_cache_info)  # add again

        # Get cache info - should trigger clearing
        result = server_queue.get_cache_info()
        self.assertEqual(result, test_cache_info)
        self.assertEqual(len(server_queue.cache_infos[0]), 0)  # should be cleared

    def test_put_finished_req(self):
        """Test put_finished_req method (lines 712-730, including 721-723, 727-729)"""
        # Create server instance
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=2
        )

        # Set flag to allow putting (line 721-723)
        server_queue.can_put_next_send_cache_finished_flag[0].set(1)

        # Put finished request
        send_cache_result = [{"req_id": "req1", "status": "completed"}]
        result = server_queue.put_finished_req(send_cache_result)

        # Verify request was added
        self.assertIn(send_cache_result[0], server_queue.finished_send_cache_list[0])
        self.assertFalse(result)  # Not all clients put yet

        # Test when all clients put (should set flag to 0, lines 727-729)
        server_queue.client_get_finish_send_cache_flag[0][:] = [1, 1]  # all clients put
        server_queue.finished_send_cache_list[0].clear()  # reset
        server_queue.can_put_next_send_cache_finished_flag[0].set(1)  # reset flag

        result = server_queue.put_finished_req(send_cache_result)
        self.assertTrue(result)  # All clients put
        self.assertEqual(server_queue.can_put_next_send_cache_finished_flag[0].get(), 0)

    def test_get_finished_req(self):
        """Test get_finished_req method (lines 732-759)"""
        # Create server instance
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=2
        )

        # Test empty list
        result = server_queue.get_finished_req()
        self.assertEqual(result, [])

        # Add finished requests and set all clients as put
        finished_reqs = [
            [{"req_id": "req1", "status": "completed"}],
            [{"req_id": "req2", "status": "error", "error": "timeout"}]
        ]
        for req in finished_reqs:
            server_queue.finished_send_cache_list[0].append(req)
        server_queue.client_get_finish_send_cache_flag[0][:] = [1, 1]  # all clients put

        # Get finished request
        result = server_queue.get_finished_req()
        self.assertEqual(len(result), 1)
        # Should pick the one with error (line 751-752)
        self.assertIn("error", result[0][1])
        self.assertEqual(len(server_queue.finished_send_cache_list[0]), 0)  # should be cleared

    def test_put_finished_add_cache_task_req(self):
        """Test put_finished_add_cache_task_req method (lines 768-779)"""
        # Create server instance
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=2
        )

        # Set flag to allow putting
        server_queue.can_put_next_add_task_finished_flag[0].set(1)

        # Put finished request
        req_ids = {"req_id": "req1", "status": "completed"}
        result = server_queue.put_finished_add_cache_task_req(req_ids)

        # Verify request was added
        self.assertIn(req_ids, server_queue.finished_add_cache_task_list[0])
        self.assertFalse(result)  # Not all clients put yet

    def test_get_finished_add_cache_task_req(self):
        """Test get_finished_add_cache_task_req method (lines 788-805)"""
        # Create server instance
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=2
        )

        # Test empty list
        result = server_queue.get_finished_add_cache_task_req()
        self.assertEqual(result, [])

        # Add finished requests and set all clients as put
        finished_reqs = [
            {"req_id": "req1", "status": "completed"},
            {"req_id": "req2", "status": "completed"}
        ]
        for req in finished_reqs:
            server_queue.finished_add_cache_task_list[0].append(req)
        server_queue.client_get_finished_add_cache_task_flag[0][:] = [1, 1]  # all clients put

        # Get finished request
        result = server_queue.get_finished_add_cache_task_req()
        self.assertEqual(len(result), 1)
        self.assertEqual(result, finished_reqs[0])  # Should return first item
        self.assertEqual(len(server_queue.finished_add_cache_task_list[0]), 0)  # should be cleared

    def test_disaggregate_queue_empty(self):
        """Test disaggregate_queue_empty method (line 811)"""
        # Create server instance to test
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=1
        )

        # Initially should be empty
        self.assertTrue(server_queue.disaggregate_queue_empty())

        # Add item to queue
        test_item = {"task": "test"}
        server_queue.disaggregate_requests[0].put(test_item)

        # Should not be empty
        self.assertFalse(server_queue.disaggregate_queue_empty())

    def test_put_disaggregated_tasks(self):
        """Test put_disaggregated_tasks method (lines 817-819)"""
        # Create server instance to test
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=1
        )

        # Put disaggregated task
        test_item = {"task_id": 1, "data": "test"}
        server_queue.put_disaggregated_tasks(test_item)

        # Verify item was added
        self.assertFalse(server_queue.disaggregate_requests[0].empty())
        retrieved_item = server_queue.disaggregate_requests[0].get()
        self.assertEqual(retrieved_item, test_item)

    def test_get_disaggregated_tasks(self):
        """Test get_disaggregated_tasks method (lines 825-832)"""
        # Create server instance to test
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=1
        )

        # Test empty queue
        result = server_queue.get_disaggregated_tasks()
        self.assertIsNone(result)

        # Add items to queue
        test_items = [{"task_id": 1}, {"task_id": 2}]
        for item in test_items:
            server_queue.disaggregate_requests[0].put(item)

        # Get disaggregated tasks
        result = server_queue.get_disaggregated_tasks()
        self.assertEqual(result, test_items)
        self.assertTrue(server_queue.disaggregate_requests[0].empty())

    def test_clear_data(self):
        """Test clear_data method (lines 835-839)"""
        # Create server instance to test
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=2
        )

        # Add some test data
        test_tasks = [{"task_id": 1}]
        server_queue.tasks[0].extend(test_tasks)
        server_queue.client_read_flag[0][:] = [0, 0]  # Not read by any client

        # Clear data
        server_queue.clear_data()

        # Verify data was cleared and flags reset
        self.assertEqual(len(server_queue.tasks[0]), 0)
        self.assertEqual(server_queue.client_read_flag[0], [1, 1])

    def test_cleanup_server(self):
        """Test cleanup method for server instance (line 845+)"""
        # Create server instance
        server_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=True,
            num_client=1
        )

        # Mock manager shutdown
        server_queue.manager.shutdown = Mock()

        # Call cleanup
        server_queue.cleanup()

        # Verify shutdown was called
        server_queue.manager.shutdown.assert_called_once()

    def test_cleanup_client(self):
        """Test cleanup method for client instance (no-op)"""
        # Create client instance
        client_queue = EngineWorkerQueue(
            address=("127.0.0.1", 0),
            is_server=False,
            num_client=1,
            client_id=0
        )

        # Call cleanup (should not raise exception)
        client_queue.cleanup()

        # No assertions needed, just verify no exception


if __name__ == "__main__":
    unittest.main()