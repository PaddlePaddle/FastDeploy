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

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import msgpack

from fastdeploy.entrypoints.openai.utils import DealerConnectionManager


class TestDealerConnectionManager(unittest.TestCase):
    """Test cases for DealerConnectionManager"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.manager = DealerConnectionManager(pid=1, max_connections=5)

    def tearDown(self):
        self.loop.run_until_complete(self.manager.close())
        self.loop.close()

    @patch("aiozmq.create_zmq_stream")
    async def test_initialization(self, mock_create):
        """Test manager initialization creates connections"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        # Test initialization
        await self.manager.initialize()

        # Verify connections were created
        self.assertEqual(len(self.manager.connections), 10)
        self.assertEqual(len(self.manager.connection_load), 10)
        self.assertEqual(len(self.manager.connection_tasks), 10)

        # Verify connection tasks are running
        for task in self.manager.connection_tasks:
            self.assertFalse(task.done())

    @patch("aiozmq.create_zmq_stream")
    async def test_get_connection(self, mock_create):
        """Test getting a connection with load balancing"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream
        await self.manager.initialize()

        # Get a connection
        dealer, queue = await self.manager.get_connection("req1")

        # Verify least loaded connection is returned
        self.assertEqual(self.manager.connection_load[0], 1)
        self.assertIsNotNone(dealer)
        self.assertIsNotNone(queue)
        self.assertIn("req1", self.manager.request_map)

    @patch("aiozmq.create_zmq_stream")
    async def test_connection_listening(self, mock_create):
        """Test connection listener handles responses"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream
        await self.manager.initialize()

        # Setup test response
        test_response = {"request_id": "req1", "finished": True}
        mock_stream.read.return_value = [b"", msgpack.packb(test_response)]

        # Simulate response
        dealer, queue = await self.manager.get_connection("req1")
        response = await queue.get()

        # Verify response handling
        self.assertEqual(response[-1]["request_id"], "req1")
        self.assertEqual(self.manager.connection_load[0], 0)  # Should be decremented after finish

    @patch("aiozmq.create_zmq_stream")
    async def test_request_cleanup(self, mock_create):
        """Test request cleanup removes request tracking"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream
        await self.manager.initialize()

        await self.manager.get_connection("req1")
        self.assertIn("req1", self.manager.request_map)

        await self.manager.cleanup_request("req1")
        self.assertNotIn("req1", self.manager.request_map)

    @patch("aiozmq.create_zmq_stream")
    async def test_multiple_requests(self, mock_create):
        """Test load balancing with multiple requests"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream
        await self.manager.initialize()

        # Get multiple connections
        connections = []
        for i in range(1, 6):
            dealer, queue = await self.manager.get_connection(f"req{i}")
            connections.append((dealer, queue))

        # Verify load is distributed
        load_counts = [0] * 5
        for i in range(5):
            load_counts[i] = self.manager.connection_load[i]

        self.assertEqual(sum(load_counts), 5)
        self.assertTrue(all(1 <= load <= 2 for load in load_counts))

    @patch("aiozmq.create_zmq_stream")
    async def test_connection_failure(self, mock_create):
        """Test connection failure handling"""
        mock_create.side_effect = Exception("Connection failed")

        with self.assertLogs(level="ERROR") as log:
            await self.manager._add_connection(0)
            self.assertTrue(any("Failed to create dealer" in msg for msg in log.output))

        self.assertEqual(len(self.manager.connections), 0)

    @patch("aiozmq.create_zmq_stream")
    async def test_close_manager(self, mock_create):
        """Test manager shutdown"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream
        await self.manager.initialize()

        # Verify connections exist
        self.assertEqual(len(self.manager.connections), 5)

        # Close manager
        await self.manager.close()

        # Verify cleanup
        self.assertEqual(len(self.manager.connections), 0)
        self.assertEqual(len(self.manager.request_map), 0)
        for task in self.manager.connection_tasks:
            self.assertTrue(task.cancelled())


if __name__ == "__main__":
    unittest.main()
