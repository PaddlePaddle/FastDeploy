"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack

from fastdeploy.entrypoints.openai.utils import DealerConnectionManager


class TestDealerConnectionManager(unittest.TestCase):
    """Test case for DealerConnectionManager"""

    def setUp(self):
        """Set up test environment"""
        self.pid = 12345
        self.max_connections = 5
        self.manager = DealerConnectionManager(self.pid, self.max_connections)

    def test_init(self):
        """Test DealerConnectionManager initialization"""
        self.assertEqual(self.manager.pid, self.pid)
        self.assertEqual(self.manager.max_connections, self.max_connections)
        self.assertEqual(self.manager.connections, [])
        self.assertEqual(self.manager.connection_load, [])
        self.assertEqual(self.manager.connection_heap, [])
        self.assertEqual(self.manager.request_map, {})
        self.assertEqual(self.manager.request_num, {})
        self.assertFalse(self.manager.running)
        self.assertEqual(self.manager.connection_tasks, [])

    def test_init_min_connections(self):
        """Test that minimum connections is enforced"""
        manager = DealerConnectionManager(self.pid, 5)
        self.assertEqual(manager.max_connections, 10)  # Should be at least 10

    @patch('fastdeploy.entrypoints.openai.utils.aiozmq.create_zmq_stream')
    @patch('fastdeploy.entrypoints.openai.utils.asyncio.create_task')
    async def test_initialize(self, mock_create_task, mock_create_stream):
        """Test initialization of connections"""
        mock_dealer = AsyncMock()
        mock_create_stream.return_value = mock_dealer
        mock_task = AsyncMock()
        mock_create_task.return_value = mock_task

        await self.manager.initialize()

        self.assertTrue(self.manager.running)
        self.assertEqual(len(self.manager.connections), self.max_connections)
        self.assertEqual(len(self.manager.connection_load), self.max_connections)
        self.assertEqual(len(self.manager.connection_heap), self.max_connections)
        self.assertEqual(len(self.manager.connection_tasks), self.max_connections)
        self.assertEqual(mock_create_stream.call_count, self.max_connections)

    @patch('fastdeploy.entrypoints.openai.utils.aiozmq.create_zmq_stream')
    @patch('fastdeploy.entrypoints.openai.utils.asyncio.create_task')
    async def test_add_connection_success(self, mock_create_task, mock_create_stream):
        """Test successful connection addition"""
        mock_dealer = AsyncMock()
        mock_create_stream.return_value = mock_dealer
        mock_task = AsyncMock()
        mock_create_task.return_value = mock_task

        result = await self.manager._add_connection(0)

        self.assertTrue(result)
        self.assertEqual(len(self.manager.connections), 1)
        self.assertEqual(len(self.manager.connection_load), 1)
        self.assertEqual(len(self.manager.connection_heap), 1)
        self.assertEqual(len(self.manager.connection_tasks), 1)
        mock_create_stream.assert_called_once_with(
            unittest.mock.ANY,  # zmq.DEALER
            connect=f"ipc:///dev/shm/router_{self.pid}.ipc"
        )

    @patch('fastdeploy.entrypoints.openai.utils.aiozmq.create_zmq_stream')
    async def test_add_connection_failure(self, mock_create_stream):
        """Test connection addition failure"""
        mock_create_stream.side_effect = Exception("Connection failed")

        result = await self.manager._add_connection(0)

        self.assertFalse(result)
        self.assertEqual(len(self.manager.connections), 0)

    async def test_listen_connection(self):
        """Test listening for messages on a connection"""
        mock_dealer = AsyncMock()
        
        # Mock message data
        response_data = [{"request_id": "test-123", "finished": True}]
        packed_data = msgpack.packb(response_data)
        mock_dealer.read.return_value = [packed_data]
        
        # Set up request mapping
        self.manager.running = True
        self.manager.request_map["test"] = asyncio.Queue()
        self.manager.request_num["test"] = 1
        self.manager.connection_load = [0]
        self.manager.connection_heap = [(0, 0)]

        # Start listener and let it process one message
        listener_task = asyncio.create_task(self.manager._listen_connection(mock_dealer, 0))
        
        # Give the listener a moment to process
        await asyncio.sleep(0.01)
        
        # Stop the listener
        self.manager.running = False
        listener_task.cancel()

        # Verify the message was processed
        self.assertFalse(self.manager.request_map["test"].empty())

    def test_update_load(self):
        """Test connection load update"""
        self.manager.connection_load = [1, 2, 3]
        self.manager.connection_heap = [(1, 0), (2, 1), (3, 2)]

        with patch('random.random', return_value=0.005):  # Force debug logging
            self.manager._update_load(1, 2)

        self.assertEqual(self.manager.connection_load[1], 4)

    def test_get_least_loaded_connection_empty(self):
        """Test getting connection when no connections available"""
        result = self.manager._get_least_loaded_connection()
        self.assertIsNone(result)

    def test_get_least_loaded_connection(self):
        """Test getting least loaded connection"""
        mock_dealer = MagicMock()
        self.manager.connections = [mock_dealer]
        self.manager.connection_load = [0]
        self.manager.connection_heap = [(0, 0)]

        result = self.manager._get_least_loaded_connection()

        self.assertEqual(result, mock_dealer)
        self.assertEqual(self.manager.connection_load[0], 1)

    async def test_get_connection(self):
        """Test getting a connection for a request"""
        mock_dealer = MagicMock()
        self.manager.connections = [mock_dealer]
        self.manager.connection_load = [0]
        self.manager.connection_heap = [(0, 0)]

        dealer, queue = await self.manager.get_connection("test-123", 2)

        self.assertEqual(dealer, mock_dealer)
        self.assertIsInstance(queue, asyncio.Queue)
        self.assertIn("test-123", self.manager.request_map)
        self.assertEqual(self.manager.request_num["test-123"], 2)

    async def test_get_connection_no_available(self):
        """Test getting connection when none available"""
        with self.assertRaises(RuntimeError) as context:
            await self.manager.get_connection("test-123")
        
        self.assertIn("No available connections", str(context.exception))

    async def test_cleanup_request(self):
        """Test cleaning up a request"""
        self.manager.request_map["test-123"] = asyncio.Queue()
        self.manager.request_num["test-123"] = 1

        await self.manager.cleanup_request("test-123")

        self.assertNotIn("test-123", self.manager.request_map)
        self.assertNotIn("test-123", self.manager.request_num)

    async def test_cleanup_request_not_exists(self):
        """Test cleaning up a non-existent request"""
        # Should not raise any exceptions
        await self.manager.cleanup_request("non-existent")

    async def test_close(self):
        """Test closing the manager"""
        # Set up some connections and tasks
        mock_dealer1 = MagicMock()
        mock_dealer2 = MagicMock()
        mock_task1 = MagicMock()
        mock_task2 = MagicMock()
        
        self.manager.running = True
        self.manager.connections = [mock_dealer1, mock_dealer2]
        self.manager.connection_load = [0, 1]
        self.manager.connection_tasks = [mock_task1, mock_task2]
        self.manager.request_map = {"test": asyncio.Queue()}

        await self.manager.close()

        self.assertFalse(self.manager.running)
        mock_task1.cancel.assert_called_once()
        mock_task2.cancel.assert_called_once()
        mock_dealer1.close.assert_called_once()
        mock_dealer2.close.assert_called_once()
        self.assertEqual(len(self.manager.connections), 0)
        self.assertEqual(len(self.manager.connection_load), 0)
        self.assertEqual(len(self.manager.request_map), 0)

    async def test_close_with_close_errors(self):
        """Test closing with connection close errors"""
        mock_dealer = MagicMock()
        mock_dealer.close.side_effect = Exception("Close error")
        
        self.manager.connections = [mock_dealer]
        self.manager.connection_load = [0]

        # Should not raise an exception
        await self.manager.close()
        
        self.assertEqual(len(self.manager.connections), 0)


# Test utility functions that use asyncio
class AsyncTestCase(unittest.TestCase):
    """Base class for async tests"""
    
    def run_async(self, coro):
        """Helper to run async functions in tests"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


class TestDealerConnectionManagerAsync(AsyncTestCase):
    """Async test case for DealerConnectionManager"""

    def setUp(self):
        """Set up test environment"""
        self.manager = DealerConnectionManager(12345, 5)

    def test_initialize_async(self):
        """Test async initialization"""
        async def test():
            with patch('fastdeploy.entrypoints.openai.utils.aiozmq.create_zmq_stream') as mock_create:
                with patch('fastdeploy.entrypoints.openai.utils.asyncio.create_task') as mock_task:
                    mock_create.return_value = AsyncMock()
                    mock_task.return_value = AsyncMock()
                    await self.manager.initialize()
                    self.assertTrue(self.manager.running)

        self.run_async(test())

    def test_get_connection_async(self):
        """Test async get connection"""
        async def test():
            mock_dealer = MagicMock()
            self.manager.connections = [mock_dealer]
            self.manager.connection_load = [0]
            self.manager.connection_heap = [(0, 0)]

            dealer, queue = await self.manager.get_connection("test-123")
            self.assertEqual(dealer, mock_dealer)
            self.assertIsInstance(queue, asyncio.Queue)

        self.run_async(test())


if __name__ == "__main__":
    unittest.main()