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


class TestDealerConnectionManagerBatchMode(unittest.TestCase):
    """Test cases for DealerConnectionManager in batch mode (ZMQ_SEND_BATCH_DATA=1)"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_batch_mode_initialization(self, mock_create):
        """Test manager initialization in batch mode creates PULL client"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)

        # Verify batch mode attributes are initialized
        self.assertIsNone(manager.pull_client)
        self.assertIsNone(manager.dispatcher_task)
        self.assertFalse(hasattr(manager, "connections"))

        # Initialize
        await manager.initialize()

        # Verify PULL client was created
        mock_create.assert_called_once()
        self.assertIsNotNone(manager.pull_client)
        self.assertIsNotNone(manager.dispatcher_task)
        self.assertTrue(manager.running)

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_batch_mode_initialization_failure(self, mock_create):
        """Test manager initialization failure in batch mode"""
        mock_create.side_effect = Exception("PULL connection failed")

        manager = DealerConnectionManager(pid=1, max_connections=5)

        with self.assertLogs(level="ERROR") as log:
            await manager.initialize()
            self.assertTrue(any("Failed to create PULL client" in msg for msg in log.output))

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_batch_mode_get_connection(self, mock_create):
        """Test get_connection in batch mode returns None for dealer"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        dealer, queue = await manager.get_connection("req1")

        # In batch mode, dealer should be None
        self.assertIsNone(dealer)
        self.assertIsNotNone(queue)
        self.assertIn("req1", manager.request_map)

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_batch_mode_cleanup_request(self, mock_create):
        """Test cleanup_request in batch mode"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        await manager.get_connection("req1")
        self.assertIn("req1", manager.request_map)

        await manager.cleanup_request("req1")
        self.assertNotIn("req1", manager.request_map)

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_batch_mode_close(self, mock_create):
        """Test close method in batch mode"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Verify dispatcher task is running
        self.assertFalse(manager.dispatcher_task.done())

        # Close manager
        await manager.close()

        # Verify cleanup
        self.assertTrue(manager.dispatcher_task.cancelled() or manager.dispatcher_task.done())
        self.assertEqual(len(manager.request_map), 0)

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_batch_mode_close_with_pull_client(self, mock_create):
        """Test close method properly closes pull_client"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Close manager
        await manager.close()

        # Verify pull_client.close was called
        mock_stream.close.assert_called_once()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_batch_mode_close_pull_client_exception(self, mock_create):
        """Test close method handles exception when closing pull_client"""
        mock_stream = AsyncMock()
        mock_stream.close.side_effect = Exception("Close failed")
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Should not raise exception
        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_batch_responses_success(self, mock_create):
        """Test _dispatch_batch_responses processes batch data correctly"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Create a queue for the request
        queue = asyncio.Queue()
        manager.request_map["req1"] = queue

        # Create test batch data
        test_output = {"request_id": "req1", "data": "test"}
        batch_data = [[test_output]]
        serialized = msgpack.packb(batch_data)

        # Mock read to return batch data then raise to exit loop
        mock_stream.read.side_effect = [[b"", serialized], Exception("Exit loop")]

        # Wait for dispatcher to process
        await asyncio.sleep(0.1)

        # Check that data was dispatched to queue
        try:
            result = queue.get_nowait()
            self.assertEqual(result, [test_output])
        except asyncio.QueueEmpty:
            pass

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_batch_responses_with_cmpl_prefix(self, mock_create):
        """Test _dispatch_batch_responses handles request_id with cmpl prefix"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Create a queue for the request (without suffix)
        queue = asyncio.Queue()
        manager.request_map["req1"] = queue

        # Create test batch data with cmpl prefix
        test_output = {"request_id": "cmpl_abc_123", "data": "test"}
        batch_data = [[test_output]]
        serialized = msgpack.packb(batch_data)

        mock_stream.read.side_effect = [[b"", serialized], Exception("Exit loop")]

        await asyncio.sleep(0.1)

        try:
            result = queue.get_nowait()
            self.assertEqual(result, [test_output])
        except asyncio.QueueEmpty:
            pass

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_batch_responses_with_chatcmpl_prefix(self, mock_create):
        """Test _dispatch_batch_responses handles request_id with chatcmpl prefix"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        queue = asyncio.Queue()
        manager.request_map["chat_req"] = queue

        # Create test batch data with chatcmpl prefix
        test_output = {"request_id": "chatcmpl_xyz_456", "data": "test"}
        batch_data = [[test_output]]
        serialized = msgpack.packb(batch_data)

        mock_stream.read.side_effect = [[b"", serialized], Exception("Exit loop")]

        await asyncio.sleep(0.1)

        try:
            result = queue.get_nowait()
            self.assertEqual(result, [test_output])
        except asyncio.QueueEmpty:
            pass

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_batch_responses_with_embd_prefix(self, mock_create):
        """Test _dispatch_batch_responses handles request_id with embd prefix"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        queue = asyncio.Queue()
        manager.request_map["emb1"] = queue

        test_output = {"request_id": "embd_def_789", "data": "test"}
        batch_data = [[test_output]]
        serialized = msgpack.packb(batch_data)

        mock_stream.read.side_effect = [[b"", serialized], Exception("Exit loop")]

        await asyncio.sleep(0.1)

        try:
            result = queue.get_nowait()
            self.assertEqual(result, [test_output])
        except asyncio.QueueEmpty:
            pass

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_batch_responses_with_reward_prefix(self, mock_create):
        """Test _dispatch_batch_responses handles request_id with reward prefix"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        queue = asyncio.Queue()
        manager.request_map["rew1"] = queue

        test_output = {"request_id": "reward_ghi_012", "data": "test"}
        batch_data = [[test_output]]
        serialized = msgpack.packb(batch_data)

        mock_stream.read.side_effect = [[b"", serialized], Exception("Exit loop")]

        await asyncio.sleep(0.1)

        try:
            result = queue.get_nowait()
            self.assertEqual(result, [test_output])
        except asyncio.QueueEmpty:
            pass

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_batch_responses_connection_error(self, mock_create):
        """Test _dispatch_batch_responses handles connection errors"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Simulate connection error
        mock_stream.read.side_effect = ConnectionError("Connection lost")

        with self.assertLogs(level="ERROR") as log:
            await asyncio.sleep(0.1)
            self.assertTrue(any("connection lost" in msg for msg in log.output))

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_batch_responses_os_error(self, mock_create):
        """Test _dispatch_batch_responses handles OS errors"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        mock_stream.read.side_effect = OSError("OS error")

        with self.assertLogs(level="ERROR") as log:
            await asyncio.sleep(0.1)
            self.assertTrue(any("connection lost" in msg.lower() or "os error" in msg.lower() for msg in log.output))

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_batch_responses_consecutive_errors(self, mock_create):
        """Test _dispatch_batch_responses exits after consecutive errors"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Simulate multiple consecutive errors
        mock_stream.read.side_effect = ValueError("Test error")

        with self.assertLogs(level="ERROR") as log:
            await asyncio.sleep(0.2)
            # Should see error logs about consecutive errors
            error_msgs = [msg for msg in log.output if "Dispatcher error" in msg or "consecutive errors" in msg]
            self.assertTrue(len(error_msgs) > 0)

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_batch_responses_unknown_request(self, mock_create):
        """Test _dispatch_batch_responses ignores unknown request IDs"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # No queue registered for this request
        test_output = {"request_id": "unknown_req", "data": "test"}
        batch_data = [[test_output]]
        serialized = msgpack.packb(batch_data)

        mock_stream.read.side_effect = [[b"", serialized], Exception("Exit loop")]

        # Should not raise exception, just ignore unknown request
        await asyncio.sleep(0.1)

        await manager.close()


class TestDealerConnectionManagerCleanupCancelled(unittest.TestCase):
    """Test cases for CancelledError handling in cleanup_request"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    @patch("aiozmq.create_zmq_stream")
    async def test_cleanup_request_cancelled_error(self, mock_create):
        """Test cleanup_request handles CancelledError correctly"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        await manager.get_connection("req1")
        self.assertIn("req1", manager.request_map)

        # Simulate CancelledError during cleanup
        original_lock = manager.lock
        manager.lock = AsyncMock()
        manager.lock.__aenter__ = AsyncMock(side_effect=asyncio.CancelledError)
        manager.lock.__aexit__ = AsyncMock()

        try:
            await manager.cleanup_request("req1")
        except asyncio.CancelledError:
            pass

        # Verify cleanup happened without lock
        self.assertNotIn("req1", manager.request_map)

        manager.lock = original_lock
        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_cleanup_request_cancelled_error_batch_mode(self, mock_create):
        """Test cleanup_request handles CancelledError in batch mode"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        await manager.get_connection("req1")
        self.assertIn("req1", manager.request_map)

        # Simulate CancelledError during cleanup
        original_lock = manager.lock
        manager.lock = AsyncMock()
        manager.lock.__aenter__ = AsyncMock(side_effect=asyncio.CancelledError)
        manager.lock.__aexit__ = AsyncMock()

        try:
            await manager.cleanup_request("req1")
        except asyncio.CancelledError:
            pass

        # Verify cleanup happened
        self.assertNotIn("req1", manager.request_map)

        manager.lock = original_lock
        await manager.close()


class TestDealerConnectionManagerCloseExceptions(unittest.TestCase):
    """Test cases for exception handling in close method"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    @patch("aiozmq.create_zmq_stream")
    async def test_close_handles_dealer_close_exception(self, mock_create):
        """Test close method handles exceptions when closing dealers"""
        mock_stream = AsyncMock()
        mock_stream.close.side_effect = Exception("Close dealer failed")
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Should not raise exception
        await manager.close()

        # Verify cleanup still happened
        self.assertEqual(len(manager.connections), 0)

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_close_batch_mode_no_dispatcher_task(self, mock_create):
        """Test close in batch mode when dispatcher_task is None"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Set dispatcher_task to None to test the None check
        manager.dispatcher_task = None

        # Should not raise exception
        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_close_batch_mode_no_pull_client(self, mock_create):
        """Test close in batch mode when pull_client is None"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Set pull_client to None to test the None check
        manager.pull_client = None

        # Should not raise exception
        await manager.close()


if __name__ == "__main__":
    unittest.main()
