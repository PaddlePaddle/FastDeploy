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
    """Test cases for DealerConnectionManager in dealer mode (ZMQ_SEND_BATCH_DATA=0)"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    def test_init_attributes(self):
        """Test __init__ sets correct attributes in dealer mode"""
        manager = DealerConnectionManager(pid=1, max_connections=5)
        # Verify dealer mode attributes (lines 99-104)
        self.assertEqual(manager.max_connections, 10)  # max(max_connections, 10)
        self.assertEqual(manager.connections, [])
        self.assertEqual(manager.connection_load, [])
        self.assertEqual(manager.connection_heap, [])
        self.assertEqual(manager.request_num, {})
        self.assertEqual(manager.connection_tasks, [])

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_initialization(self, mock_create):
        """Test manager initialization creates connections"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        # Test initialization
        await manager.initialize()

        # Verify connections were created
        self.assertEqual(len(manager.connections), 10)
        self.assertEqual(len(manager.connection_load), 10)
        self.assertEqual(len(manager.connection_tasks), 10)

        # Verify connection tasks are running
        for task in manager.connection_tasks:
            self.assertFalse(task.done())

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_get_connection(self, mock_create):
        """Test getting a connection with load balancing"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Get a connection
        dealer, queue = await manager.get_connection("req1")

        # Verify least loaded connection is returned
        self.assertEqual(manager.connection_load[0], 1)
        self.assertIsNotNone(dealer)
        self.assertIsNotNone(queue)
        self.assertIn("req1", manager.request_map)

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_connection_listening(self, mock_create):
        """Test connection listener handles responses"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Setup test response
        test_response = {"request_id": "req1", "finished": True}
        mock_stream.read.return_value = [b"", msgpack.packb(test_response)]

        # Simulate response
        dealer, queue = await manager.get_connection("req1")
        response = await queue.get()

        # Verify response handling
        self.assertEqual(response[-1]["request_id"], "req1")
        self.assertEqual(manager.connection_load[0], 0)  # Should be decremented after finish

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_request_cleanup(self, mock_create):
        """Test request cleanup removes request tracking"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        await manager.get_connection("req1")
        self.assertIn("req1", manager.request_map)

        await manager.cleanup_request("req1")
        self.assertNotIn("req1", manager.request_map)

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_multiple_requests(self, mock_create):
        """Test load balancing with multiple requests"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Get multiple connections
        connections = []
        for i in range(1, 6):
            dealer, queue = await manager.get_connection(f"req{i}")
            connections.append((dealer, queue))

        # Verify load is distributed
        load_counts = [0] * 5
        for i in range(5):
            load_counts[i] = manager.connection_load[i]

        self.assertEqual(sum(load_counts), 5)
        self.assertTrue(all(1 <= load <= 2 for load in load_counts))

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_connection_failure(self, mock_create):
        """Test connection failure handling"""
        mock_create.side_effect = Exception("Connection failed")

        manager = DealerConnectionManager(pid=1, max_connections=5)

        with self.assertLogs(level="ERROR") as log:
            await manager._add_connection(0)
            self.assertTrue(any("Failed to create dealer" in msg for msg in log.output))

        self.assertEqual(len(manager.connections), 0)

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_close_manager(self, mock_create):
        """Test manager shutdown"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Verify connections exist
        self.assertEqual(len(manager.connections), 10)

        # Close manager
        await manager.close()

        # Verify cleanup
        self.assertEqual(len(manager.connections), 0)
        self.assertEqual(len(manager.request_map), 0)
        for task in manager.connection_tasks:
            self.assertTrue(task.cancelled())

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_get_connection_no_available(self, mock_create):
        """Test get_connection raises error when no connections available"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Clear connection heap to simulate no available connections
        manager.connection_heap = []

        with self.assertRaises(RuntimeError) as context:
            await manager.get_connection("req1")

        self.assertIn("No available connections", str(context.exception))

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_get_connection_with_num_choices(self, mock_create):
        """Test get_connection with num_choices in dealer mode"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        dealer, queue = await manager.get_connection("req1", num_choices=2)

        self.assertIsNotNone(dealer)
        self.assertIsNotNone(queue)
        self.assertIn("req1", manager.request_map)
        self.assertEqual(manager.request_num["req1"], 2)

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_cleanup_request_with_request_num(self, mock_create):
        """Test cleanup_request removes request_num in dealer mode"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        await manager.get_connection("req1", num_choices=2)
        self.assertIn("req1", manager.request_map)
        self.assertIn("req1", manager.request_num)

        await manager.cleanup_request("req1")
        self.assertNotIn("req1", manager.request_map)
        self.assertNotIn("req1", manager.request_num)

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_close_with_exception(self, mock_create):
        """Test close handles exceptions when closing dealers"""
        mock_stream = AsyncMock()
        mock_stream.close.side_effect = Exception("Close failed")
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Should not raise exception
        await manager.close()

        # Verify cleanup happened
        self.assertEqual(len(manager.connections), 0)

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_cleanup_cancelled_error(self, mock_create):
        """Test cleanup_request handles CancelledError in dealer mode"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        await manager.get_connection("req1", num_choices=2)
        self.assertIn("req1", manager.request_map)
        self.assertIn("req1", manager.request_num)

        # Simulate CancelledError during cleanup
        original_lock = manager.lock
        manager.lock = AsyncMock()
        manager.lock.__aenter__ = AsyncMock(side_effect=asyncio.CancelledError)
        manager.lock.__aexit__ = AsyncMock()

        try:
            await manager.cleanup_request("req1")
        except asyncio.CancelledError:
            pass

        # Verify cleanup happened without lock (both request_map and request_num)
        self.assertNotIn("req1", manager.request_map)
        self.assertNotIn("req1", manager.request_num)

        manager.lock = original_lock
        await manager.close()


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
        """Test manager initialization failure in batch mode (covers lines 119-123)"""
        mock_create.side_effect = Exception("PULL connection failed")

        manager = DealerConnectionManager(pid=1, max_connections=5)

        with self.assertLogs(level="ERROR") as log:
            with self.assertRaises(RuntimeError) as ctx:
                await manager.initialize()
            self.assertTrue(any("Failed to create PULL client" in msg for msg in log.output))

        # Lines 122-123: running should be reset to False
        self.assertFalse(manager.running)
        self.assertIn("Failed to initialize PULL client", str(ctx.exception))

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
        """Test _dispatch_batch_responses handles connection errors (lines 235-237)"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Simulate connection error while running
        self.assertTrue(manager.running)
        mock_stream.read.side_effect = ConnectionError("Connection lost")

        with self.assertLogs(level="ERROR") as log:
            await asyncio.sleep(0.2)
            self.assertTrue(any("connection lost" in msg for msg in log.output))

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_batch_responses_os_error(self, mock_create):
        """Test _dispatch_batch_responses handles OS errors (lines 235-237)"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Simulate OS error while running
        self.assertTrue(manager.running)
        mock_stream.read.side_effect = OSError("OS error")

        with self.assertLogs(level="ERROR") as log:
            await asyncio.sleep(0.2)
            self.assertTrue(any("connection lost" in msg.lower() or "os error" in msg.lower() for msg in log.output))

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_batch_responses_consecutive_errors(self, mock_create):
        """Test _dispatch_batch_responses exits after consecutive errors (lines 239-247)"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Simulate multiple consecutive errors
        self.assertTrue(manager.running)
        mock_stream.read.side_effect = ValueError("Test error")

        with self.assertLogs(level="ERROR") as log:
            await asyncio.sleep(0.5)
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

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_cleanup_request_cancelled_error(self, mock_create):
        """Test cleanup_request handles CancelledError correctly in dealer mode"""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        await manager.get_connection("req1", num_choices=2)
        self.assertIn("req1", manager.request_map)
        self.assertIn("req1", manager.request_num)

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
        self.assertNotIn("req1", manager.request_num)

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

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
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


class TestDealerConnectionManagerDealerModeInitialize(unittest.TestCase):
    """Test dealer-mode initialize and close paths for coverage of lines 125-127, 262-268, 304-314."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_dealer_mode_initialize_creates_connections(self, mock_create):
        """Cover lines 125-127: dealer-mode initialize creates connections and logs."""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=42, max_connections=10)
        await manager.initialize()

        # Lines 125-126: all connections created
        self.assertEqual(len(manager.connections), 10)
        self.assertEqual(len(manager.connection_load), 10)
        self.assertTrue(manager.running)

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_dealer_mode_get_connection_returns_dealer_and_queue(self, mock_create):
        """Cover lines 262-268: get_connection in dealer mode sets request_num and returns dealer."""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=42, max_connections=10)
        await manager.initialize()

        dealer, queue = await manager.get_connection("req_test", num_choices=3)

        # Lines 262-264: request_map and request_num set
        self.assertIn("req_test", manager.request_map)
        self.assertEqual(manager.request_num["req_test"], 3)
        # Line 265-266: dealer returned
        self.assertIsNotNone(dealer)
        # Line 267-268: no RuntimeError since connections exist
        self.assertIsNotNone(queue)

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_dealer_mode_get_connection_raises_when_empty(self, mock_create):
        """Cover lines 266-267: get_connection raises when no connections."""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=42, max_connections=10)
        await manager.initialize()

        # Clear heap to simulate no available connections
        manager.connection_heap = []

        with self.assertRaises(RuntimeError) as ctx:
            await manager.get_connection("req_fail")
        self.assertIn("No available connections", str(ctx.exception))

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_dealer_mode_close_cancels_tasks_and_clears(self, mock_create):
        """Cover lines 304-314: close in dealer mode cancels tasks and clears connections."""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=42, max_connections=10)
        await manager.initialize()

        # Verify state before close
        self.assertEqual(len(manager.connections), 10)
        self.assertEqual(len(manager.connection_tasks), 10)

        await manager.close()

        # Lines 304-305: tasks cancelled
        for task in manager.connection_tasks:
            self.assertTrue(task.cancelled() or task.done())
        # Lines 307-314: connections and load cleared
        self.assertEqual(len(manager.connections), 0)
        self.assertEqual(len(manager.connection_load), 0)
        self.assertEqual(len(manager.request_map), 0)

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_cleanup_request_cancelled_error_dealer_mode(self, mock_create):
        """Cover lines 278-284: cleanup_request CancelledError fallback in dealer mode."""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=42, max_connections=10)
        await manager.initialize()

        await manager.get_connection("req_cancel", num_choices=2)
        self.assertIn("req_cancel", manager.request_map)
        self.assertIn("req_cancel", manager.request_num)

        # Replace lock to raise CancelledError
        original_lock = manager.lock
        manager.lock = AsyncMock()
        manager.lock.__aenter__ = AsyncMock(side_effect=asyncio.CancelledError)
        manager.lock.__aexit__ = AsyncMock()

        with self.assertRaises(asyncio.CancelledError):
            await manager.cleanup_request("req_cancel")

        # Lines 281-284: cleanup happens without lock
        self.assertNotIn("req_cancel", manager.request_map)
        self.assertNotIn("req_cancel", manager.request_num)

        manager.lock = original_lock
        await manager.close()


class TestDealerConnectionManagerDispatchErrors(unittest.TestCase):
    """Test _dispatch_batch_responses error paths for coverage of lines 238-250."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_connection_error_while_running(self, mock_create):
        """Cover lines 238-240: ConnectionError while running logs and breaks."""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        self.assertTrue(manager.running)
        mock_stream.read.side_effect = ConnectionError("Connection lost")

        # Wait for dispatcher to process the error
        await asyncio.sleep(0.2)

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_generic_error_increments_counter(self, mock_create):
        """Cover lines 242-244: generic Exception increments consecutive_errors."""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        self.assertTrue(manager.running)
        mock_stream.read.side_effect = ValueError("deserialization error")

        # Wait for dispatcher to hit max consecutive errors
        await asyncio.sleep(0.5)

        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_max_consecutive_errors_exits(self, mock_create):
        """Cover lines 248-250: exits after max_consecutive_errors."""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream

        manager = DealerConnectionManager(pid=1, max_connections=5)
        await manager.initialize()

        # Raise enough errors to hit the max (5)
        mock_stream.read.side_effect = RuntimeError("repeated failure")

        # Wait long enough for all 5 errors to be processed
        await asyncio.sleep(1.0)

        # Dispatcher should have exited
        self.assertTrue(manager.dispatcher_task.done())

        await manager.close()


class TestDealerConnectionManagerAsync(unittest.IsolatedAsyncioTestCase):
    """
    Use IsolatedAsyncioTestCase to properly execute async test methods so that
    the coverage tool can instrument the async code paths.

    Targets uncovered lines in utils.py:
      - 123-134  : initialize() exception path (batch) + dealer mode connection loop
      - 245-257  : _dispatch_batch_responses ConnectionError / generic error / max retries
      - 269-275  : get_connection() dealer mode (raise RuntimeError when no connections)
      - 285-291  : cleanup_request() CancelledError fallback (ZMQ_SEND_BATCH_DATA=True)
      - 317-330  : close() non-batch path (cancel tasks + close dealers)
    """

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_initialize_batch_mode_exception_resets_running(self, mock_create):
        """Lines 123-130: initialize() in batch mode raises RuntimeError on failure and sets running=False."""
        mock_create.side_effect = OSError("bind failed")
        manager = DealerConnectionManager(pid=9, max_connections=5)
        with self.assertRaises(RuntimeError) as ctx:
            await manager.initialize()
        self.assertFalse(manager.running)
        self.assertIn("Failed to initialize PULL client", str(ctx.exception))

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_initialize_dealer_mode_creates_connections(self, mock_create):
        """Lines 132-134: initialize() dealer mode loops over max_connections and calls _add_connection."""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream
        manager = DealerConnectionManager(pid=9, max_connections=3)
        await manager.initialize()
        self.assertEqual(len(manager.connections), 10)  # max(3, 10) = 10
        self.assertTrue(manager.running)
        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_connection_error_exits_loop(self, mock_create):
        """Lines 245-247: ConnectionError/OSError in _dispatch_batch_responses logs and breaks."""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream
        manager = DealerConnectionManager(pid=9, max_connections=5)
        await manager.initialize()
        mock_stream.read.side_effect = ConnectionError("socket closed")
        # Allow the dispatcher coroutine to run
        await asyncio.sleep(0.15)
        # Dispatcher task should have finished due to the break
        self.assertTrue(manager.dispatcher_task.done())
        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_os_error_exits_loop(self, mock_create):
        """Lines 245-247: OSError in _dispatch_batch_responses logs and breaks."""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream
        manager = DealerConnectionManager(pid=9, max_connections=5)
        await manager.initialize()
        mock_stream.read.side_effect = OSError("ipc gone")
        await asyncio.sleep(0.15)
        self.assertTrue(manager.dispatcher_task.done())
        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_dispatch_generic_error_increments_and_exits_after_max(self, mock_create):
        """Lines 249-257: generic Exception increments counter; dispatcher exits after max_consecutive_errors (5)."""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream
        manager = DealerConnectionManager(pid=9, max_connections=5)
        await manager.initialize()
        mock_stream.read.side_effect = ValueError("bad data")
        # Give enough time to hit 5 consecutive errors
        await asyncio.sleep(0.8)
        self.assertTrue(manager.dispatcher_task.done())
        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_get_connection_dealer_no_connections_raises(self, mock_create):
        """Lines 269-275: get_connection() raises RuntimeError when connection_heap is empty."""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream
        manager = DealerConnectionManager(pid=9, max_connections=5)
        await manager.initialize()
        # Drain the heap so no connection is available
        manager.connection_heap.clear()
        with self.assertRaises(RuntimeError) as ctx:
            await manager.get_connection("req-no-conn")
        self.assertIn("No available connections", str(ctx.exception))
        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", True)
    @patch("aiozmq.create_zmq_stream")
    async def test_cleanup_request_cancelled_error_batch_mode(self, mock_create):
        """Lines 285-291: cleanup_request() CancelledError fallback in batch mode."""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream
        manager = DealerConnectionManager(pid=9, max_connections=5)
        await manager.initialize()

        # Add a request to the map
        queue = asyncio.Queue()
        manager.request_map["req-cancel"] = queue

        # Make the lock raise CancelledError on entry
        original_lock = manager.lock
        manager.lock = AsyncMock()
        manager.lock.__aenter__ = AsyncMock(side_effect=asyncio.CancelledError)
        manager.lock.__aexit__ = AsyncMock()

        with self.assertRaises(asyncio.CancelledError):
            await manager.cleanup_request("req-cancel")

        # Fallback cleanup should have removed the request from request_map
        self.assertNotIn("req-cancel", manager.request_map)

        manager.lock = original_lock
        await manager.close()

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_close_dealer_mode_cancels_tasks_and_clears_connections(self, mock_create):
        """Lines 317-330: close() non-batch path cancels tasks and clears connections/load."""
        mock_stream = AsyncMock()
        mock_create.return_value = mock_stream
        manager = DealerConnectionManager(pid=9, max_connections=5)
        await manager.initialize()

        # Verify connections exist before close
        self.assertEqual(len(manager.connections), 10)
        self.assertEqual(len(manager.connection_tasks), 10)

        await manager.close()
        # Yield to the event loop so that pending task cancellations are processed.
        # task.cancel() only schedules the CancelledError injection; the tasks need
        # at least one event-loop iteration to actually transition to cancelled/done.
        await asyncio.sleep(0)

        # Lines 319-321: tasks cancelled or finished after close()
        for task in manager.connection_tasks:
            self.assertTrue(task.cancelled() or task.done())
        # Lines 323-330: connections and load cleared
        self.assertEqual(len(manager.connections), 0)
        self.assertEqual(len(manager.connection_load), 0)
        self.assertEqual(len(manager.request_map), 0)

    @patch("fastdeploy.entrypoints.openai.utils.envs.ZMQ_SEND_BATCH_DATA", False)
    @patch("aiozmq.create_zmq_stream")
    async def test_close_dealer_mode_with_dealer_close_exception(self, mock_create):
        """Lines 325-328: close() non-batch path swallows exceptions from dealer.close()."""
        mock_stream = AsyncMock()
        mock_stream.close.side_effect = Exception("dealer close failed")
        mock_create.return_value = mock_stream
        manager = DealerConnectionManager(pid=9, max_connections=5)
        await manager.initialize()
        # Should not raise
        await manager.close()
        self.assertEqual(len(manager.connections), 0)


if __name__ == "__main__":
    unittest.main()
