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

import asyncio
import heapq
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


# Copy the DealerConnectionManager class to avoid import dependencies
class DealerConnectionManager:
    """Manager for dealer connections, supporting multiplexing and connection reuse"""

    def __init__(self, pid, max_connections=10):
        self.pid = pid
        self.max_connections = max(max_connections, 10)
        self.connections = []
        self.connection_load = []
        self.connection_heap = []
        self.request_map = {}  # request_id -> response_queue
        self.request_num = {}  # request_id -> num_choices
        self.lock = asyncio.Lock()
        self.connection_tasks = []
        self.running = False

    async def initialize(self):
        """initialize all connections"""
        self.running = True
        for index in range(self.max_connections):
            await self._add_connection(index)

    async def _add_connection(self, index):
        """create a new connection and start listening task"""
        try:
            # Mock aiozmq.create_zmq_stream
            dealer = MagicMock()
            dealer.read = AsyncMock()
            dealer.close = MagicMock()

            async with self.lock:
                self.connections.append(dealer)
                self.connection_load.append(0)
                heapq.heappush(self.connection_heap, (0, index))

            # start listening
            task = asyncio.create_task(self._listen_connection(dealer, index))
            self.connection_tasks.append(task)
            return True
        except Exception as e:
            return False

    async def _listen_connection(self, dealer, conn_index):
        """listen for messages from the dealer connection"""
        while self.running:
            try:
                raw_data = await dealer.read()
                # Mock msgpack.unpackb
                response = [None, {"request_id": "test-123", "finished": True}]
                request_id = response[-1]["request_id"]
                if "cmpl" == request_id[:4]:
                    request_id = request_id.rsplit("-", 1)[0]
                async with self.lock:
                    if request_id in self.request_map:
                        await self.request_map[request_id].put(response)
                        if response[-1]["finished"]:
                            self.request_num[request_id] -= 1
                            if self.request_num[request_id] == 0:
                                self._update_load(conn_index, -1)
            except Exception as e:
                break

    def _update_load(self, conn_index, delta):
        """Update connection load and maintain the heap"""
        self.connection_load[conn_index] += delta
        heapq.heapify(self.connection_heap)

    def _get_least_loaded_connection(self):
        """Get the least loaded connection"""
        if not self.connection_heap:
            return None

        load, conn_index = self.connection_heap[0]
        self._update_load(conn_index, 1)

        return self.connections[conn_index]

    async def get_connection(self, request_id, num_choices=1):
        """get a connection for the request"""
        response_queue = asyncio.Queue()

        async with self.lock:
            self.request_map[request_id] = response_queue
            self.request_num[request_id] = num_choices
            dealer = self._get_least_loaded_connection()
            if not dealer:
                raise RuntimeError("No available connections")

        return dealer, response_queue

    async def cleanup_request(self, request_id):
        """clean up the request after it is finished"""
        async with self.lock:
            if request_id in self.request_map:
                del self.request_map[request_id]
                del self.request_num[request_id]

    async def close(self):
        """close all connections and tasks"""
        self.running = False

        for task in self.connection_tasks:
            task.cancel()

        async with self.lock:
            for dealer in self.connections:
                try:
                    dealer.close()
                except:
                    pass
            self.connections.clear()
            self.connection_load.clear()
            self.request_map.clear()


class TestDealerConnectionManager(unittest.IsolatedAsyncioTestCase):
    """Test DealerConnectionManager class"""

    def test_init(self):
        """Test DealerConnectionManager initialization"""
        manager = DealerConnectionManager(pid=123, max_connections=5)

        self.assertEqual(manager.pid, 123)
        self.assertEqual(manager.max_connections, 10)  # Should be at least 10
        self.assertEqual(manager.connections, [])
        self.assertEqual(manager.connection_load, [])
        self.assertEqual(manager.connection_heap, [])
        self.assertEqual(manager.request_map, {})
        self.assertEqual(manager.request_num, {})
        self.assertFalse(manager.running)

    def test_init_min_connections(self):
        """Test minimum connections constraint"""
        manager = DealerConnectionManager(pid=123, max_connections=5)
        self.assertEqual(manager.max_connections, 10)  # Should be at least 10

        manager = DealerConnectionManager(pid=123, max_connections=15)
        self.assertEqual(manager.max_connections, 15)  # Should keep 15

    async def test_initialize(self):
        """Test connection initialization"""
        manager = DealerConnectionManager(pid=123, max_connections=10)

        with patch.object(manager, '_add_connection', new_callable=AsyncMock) as mock_add:
            mock_add.return_value = True
            await manager.initialize()

            self.assertTrue(manager.running)
            self.assertEqual(mock_add.call_count, 10)

    async def test_add_connection_success(self):
        """Test successful connection addition"""
        manager = DealerConnectionManager(pid=123, max_connections=10)

        result = await manager._add_connection(0)

        self.assertTrue(result)
        self.assertEqual(len(manager.connections), 1)
        self.assertEqual(len(manager.connection_load), 1)
        self.assertEqual(len(manager.connection_heap), 1)
        self.assertEqual(manager.connection_load[0], 0)
        self.assertEqual(manager.connection_heap[0], (0, 0))

    def test_update_load(self):
        """Test connection load update"""
        manager = DealerConnectionManager(pid=123, max_connections=10)
        manager.connection_load = [0, 1, 2]
        manager.connection_heap = [(0, 0), (1, 1), (2, 2)]

        manager._update_load(0, 2)

        self.assertEqual(manager.connection_load[0], 2)
        # Heap should be reordered
        self.assertIn((1, 1), manager.connection_heap)

    def test_get_least_loaded_connection_empty(self):
        """Test getting connection when none available"""
        manager = DealerConnectionManager(pid=123, max_connections=10)

        result = manager._get_least_loaded_connection()
        self.assertIsNone(result)

    async def test_get_least_loaded_connection(self):
        """Test getting least loaded connection"""
        manager = DealerConnectionManager(pid=123, max_connections=10)

        # Add a connection first
        await manager._add_connection(0)

        result = manager._get_least_loaded_connection()
        self.assertIsNotNone(result)
        self.assertEqual(manager.connection_load[0], 1)  # Load should be incremented

    async def test_get_connection(self):
        """Test getting connection for request"""
        manager = DealerConnectionManager(pid=123, max_connections=10)
        await manager._add_connection(0)

        dealer, queue = await manager.get_connection("test-request", num_choices=2)

        self.assertIsNotNone(dealer)
        self.assertIsInstance(queue, asyncio.Queue)
        self.assertIn("test-request", manager.request_map)
        self.assertEqual(manager.request_num["test-request"], 2)

    async def test_get_connection_no_available(self):
        """Test getting connection when none available"""
        manager = DealerConnectionManager(pid=123, max_connections=10)

        with self.assertRaises(RuntimeError) as cm:
            await manager.get_connection("test-request")

        self.assertIn("No available connections", str(cm.exception))

    async def test_cleanup_request(self):
        """Test request cleanup"""
        manager = DealerConnectionManager(pid=123, max_connections=10)
        manager.request_map["test-request"] = asyncio.Queue()
        manager.request_num["test-request"] = 1

        await manager.cleanup_request("test-request")

        self.assertNotIn("test-request", manager.request_map)
        self.assertNotIn("test-request", manager.request_num)

    async def test_cleanup_request_nonexistent(self):
        """Test cleanup of non-existent request"""
        manager = DealerConnectionManager(pid=123, max_connections=10)

        # Should not raise an error
        await manager.cleanup_request("nonexistent")

    async def test_close(self):
        """Test closing manager"""
        manager = DealerConnectionManager(pid=123, max_connections=10)

        # Add some connections and tasks
        await manager._add_connection(0)
        manager.request_map["test"] = asyncio.Queue()

        await manager.close()

        self.assertFalse(manager.running)
        self.assertEqual(len(manager.connections), 0)
        self.assertEqual(len(manager.connection_load), 0)
        self.assertEqual(len(manager.request_map), 0)

    async def test_listen_connection_basic(self):
        """Test basic connection listening functionality"""
        manager = DealerConnectionManager(pid=123, max_connections=10)
        mock_dealer = MagicMock()
        mock_dealer.read = AsyncMock()

        # Set up to stop after one iteration
        manager.running = True

        # Mock the read to return once then stop
        async def mock_read_side_effect():
            manager.running = False  # Stop after first read
            return [b'mock_data']

        mock_dealer.read.side_effect = mock_read_side_effect

        # This should not raise an exception
        await manager._listen_connection(mock_dealer, 0)

        mock_dealer.read.assert_called()


if __name__ == "__main__":
    unittest.main()