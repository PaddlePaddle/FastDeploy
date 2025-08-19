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
from unittest import mock

import msgpack

from fastdeploy.entrypoints.openai.utils import DealerConnectionManager


class TestDealerConnectionManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patchers = [mock.patch("aiozmq.create_zmq_stream"), mock.patch("fastdeploy.utils.api_server_logger")]
        for p in self.patchers:
            p.start()
            self.addCleanup(p.stop)

        self.mock_create_stream = self.patchers[0].start()
        self.mock_logger = self.patchers[1].start()

    async def test_initialize(self):
        """Test initialization of connections"""
        manager = DealerConnectionManager(pid=1, max_connections=5)

        # Mock the stream creation
        mock_stream = mock.AsyncMock()
        self.mock_create_stream.return_value = mock_stream

        await manager.initialize()

        # Verify connections were created
        self.assertEqual(len(manager.connections), 5)
        self.mock_logger.info.assert_called_with("Started 5 connections")

    async def test_get_connection(self):
        """Test getting a connection with load balancing"""
        manager = DealerConnectionManager(pid=1, max_connections=2)

        # Mock the stream creation
        mock_stream1 = mock.AsyncMock()
        mock_stream2 = mock.AsyncMock()
        self.mock_create_stream.side_effect = [mock_stream1, mock_stream2]

        await manager.initialize()

        # First request
        conn1, queue1 = await manager.get_connection("req1")
        self.assertIs(conn1, mock_stream1)

        # Second request should use different connection
        conn2, queue2 = await manager.get_connection("req2")
        self.assertIs(conn2, mock_stream2)

        # Third request should go back to first connection (least loaded)
        conn3, queue3 = await manager.get_connection("req3")
        self.assertIs(conn3, mock_stream1)

    async def test_listen_connection(self):
        """Test message listening"""
        manager = DealerConnectionManager(pid=1)
        manager.running = True

        # Mock connection
        mock_stream = mock.AsyncMock()
        mock_stream.read.return_value = [b"", msgpack.packb({"request_id": "req1", "finished": True})]

        # Mock response queue
        mock_queue = mock.AsyncMock()
        manager.request_map["req1"] = mock_queue

        await manager._listen_connection(mock_stream, 0)

        # Verify message was processed
        mock_queue.put.assert_called_once()
        self.assertEqual(manager.connection_load[0], -1)

    async def test_close(self):
        """Test cleanup on close"""
        manager = DealerConnectionManager(pid=1)
        manager.running = True

        # Mock connection
        mock_stream = mock.MagicMock()
        mock_task = mock.MagicMock()
        manager.connections.append(mock_stream)
        manager.connection_tasks.append(mock_task)
        manager.request_map["req1"] = mock.AsyncMock()

        await manager.close()

        # Verify cleanup
        self.assertFalse(manager.running)
        mock_stream.close.assert_called_once()
        mock_task.cancel.assert_called_once()
        self.assertEqual(len(manager.connections), 0)
        self.assertEqual(len(manager.request_map), 0)
        self.mock_logger.info.assert_called_with("All connections and tasks closed")


if __name__ == "__main__":
    unittest.main()
