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
from unittest.mock import MagicMock, patch

from fastapi import Request
from fastapi.responses import StreamingResponse

from fastdeploy.entrypoints.openai.utils import with_cancellation


class TestWithCancellation(unittest.TestCase):
    """Test cases for with_cancellation decorator"""

    def setUp(self):
        """Set up test fixtures"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up test fixtures"""
        self.loop.close()

    @patch("fastdeploy.entrypoints.openai.utils.listen_for_disconnect")
    def test_normal_execution(self, mock_listen_disconnect):
        """Test that handler executes normally when no disconnect occurs"""

        # Setup mock request
        mock_request = MagicMock(spec=Request)

        # Create a mock that returns a coroutine that never completes
        async def never_disconnect(request):
            await asyncio.Future()  # This will never complete

        mock_listen_disconnect.side_effect = never_disconnect

        # Setup handler function
        @with_cancellation
        async def test_handler(self, raw_request):
            await asyncio.sleep(0.01)  # Simulate some work
            return "test_result"

        # Run the test - need to pass self as first arg
        result = self.loop.run_until_complete(test_handler(None, mock_request))

        # Verify results
        self.assertEqual(result, "test_result")
        mock_listen_disconnect.assert_called_once_with(mock_request)

    @patch("fastdeploy.entrypoints.openai.utils.listen_for_disconnect")
    def test_client_disconnect(self, mock_listen_disconnect):
        """Test that handler is cancelled when client disconnects"""

        # Setup mock request
        mock_request = MagicMock(spec=Request)

        # Create a future that will complete (disconnect) after a short delay
        disconnect_future = asyncio.Future()
        self.loop.call_later(0.05, disconnect_future.set_result, None)
        mock_listen_disconnect.return_value = disconnect_future

        # Setup handler function that takes longer than disconnect
        handler_called = False

        @with_cancellation
        async def test_handler(self, raw_request):
            nonlocal handler_called
            try:
                await asyncio.sleep(0.1)  # Simulate work
                handler_called = True
                return "should_not_reach_here"
            except asyncio.CancelledError:
                # Handler should be cancelled
                raise

        # Run the test - need to pass self as first arg
        result = self.loop.run_until_complete(test_handler(None, mock_request))

        # Verify results
        self.assertIsNone(result)  # Should return None when cancelled
        self.assertFalse(handler_called)  # Handler should not complete
        mock_listen_disconnect.assert_called_once_with(mock_request)

    @patch("fastdeploy.entrypoints.openai.utils.listen_for_disconnect")
    def test_handler_with_args_kwargs(self, mock_listen_disconnect):
        """Test that decorator properly handles handler with args and kwargs"""

        # Setup mock request
        mock_request = MagicMock(spec=Request)

        # Create a mock that returns a coroutine that never completes
        async def never_disconnect(request):
            await asyncio.Future()  # This will never complete

        mock_listen_disconnect.side_effect = never_disconnect

        # Setup handler function with multiple arguments
        @with_cancellation
        async def test_handler(arg1, arg2, raw_request, kwarg1=None, kwarg2=None):
            await asyncio.sleep(0.01)
            return {"arg1": arg1, "arg2": arg2, "kwarg1": kwarg1, "kwarg2": kwarg2, "request": raw_request}

        # Run the test with both positional and keyword arguments
        result = self.loop.run_until_complete(
            test_handler("value1", "value2", mock_request, kwarg1="kwvalue1", kwarg2="kwvalue2")
        )

        # Verify results
        expected = {
            "arg1": "value1",
            "arg2": "value2",
            "kwarg1": "kwvalue1",
            "kwarg2": "kwvalue2",
            "request": mock_request,
        }
        self.assertEqual(result, expected)

    @patch("fastdeploy.entrypoints.openai.utils.listen_for_disconnect")
    def test_handler_returns_streaming_response(self, mock_listen_disconnect):
        """Test that decorator handles StreamingResponse correctly"""

        # Setup mock request
        mock_request = MagicMock(spec=Request)

        # Create a mock that returns a coroutine that never completes
        async def never_disconnect(request):
            await asyncio.Future()  # This will never complete

        mock_listen_disconnect.side_effect = never_disconnect

        # Setup handler that returns StreamingResponse
        @with_cancellation
        async def test_handler(self, raw_request):
            async def generate():
                yield "chunk1"
                yield "chunk2"

            return StreamingResponse(generate())

        # Run the test - need to pass self as first arg
        result = self.loop.run_until_complete(test_handler(None, mock_request))

        # Verify results
        self.assertIsInstance(result, StreamingResponse)
        mock_listen_disconnect.assert_called_once_with(mock_request)

    @patch("fastdeploy.entrypoints.openai.utils.listen_for_disconnect")
    def test_handler_exception_propagation(self, mock_listen_disconnect):
        """Test that exceptions from handler are properly propagated"""

        # Setup mock request
        mock_request = MagicMock(spec=Request)

        # Create a mock that returns a coroutine that never completes
        async def never_disconnect(request):
            await asyncio.Future()  # This will never complete

        mock_listen_disconnect.side_effect = never_disconnect

        # Setup handler that raises an exception
        @with_cancellation
        async def test_handler(self, raw_request):
            await asyncio.sleep(0.01)
            raise ValueError("Test exception")

        # Run the test and expect exception - need to pass self as first arg
        with self.assertRaises(ValueError) as context:
            self.loop.run_until_complete(test_handler(None, mock_request))

        self.assertEqual(str(context.exception), "Test exception")
        mock_listen_disconnect.assert_called_once_with(mock_request)

    @patch("fastdeploy.entrypoints.openai.utils.listen_for_disconnect")
    def test_concurrent_cancellation_and_completion(self, mock_listen_disconnect):
        """Test edge case where cancellation and completion happen simultaneously"""

        # Setup mock request
        mock_request = MagicMock(spec=Request)

        # Create futures that complete at roughly the same time
        disconnect_future = asyncio.Future()
        handler_future = asyncio.Future()

        # Set both futures to complete almost simultaneously
        self.loop.call_later(0.05, lambda: disconnect_future.set_result(None))
        self.loop.call_later(0.05, lambda: handler_future.set_result("completed"))

        mock_listen_disconnect.return_value = disconnect_future

        @with_cancellation
        async def test_handler(self, raw_request):
            return await handler_future

        # Run the test - need to pass self as first arg
        result = self.loop.run_until_complete(test_handler(None, mock_request))

        # The result depends on which task completes first
        # This test ensures the decorator handles this edge case gracefully
        self.assertIn(result, [None, "completed"])

    def test_wrapper_preserves_function_metadata(self):
        """Test that the wrapper preserves the original function's metadata"""

        def original_handler(raw_request):
            """Original handler docstring"""
            pass

        # Apply decorator
        decorated_handler = with_cancellation(original_handler)

        # Verify metadata is preserved
        self.assertEqual(decorated_handler.__name__, "original_handler")
        self.assertEqual(decorated_handler.__doc__, "Original handler docstring")
        self.assertTrue(hasattr(decorated_handler, "__wrapped__"))


class TestListenForDisconnect(unittest.TestCase):
    """Test cases for listen_for_disconnect function"""

    def setUp(self):
        """Set up test fixtures"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up test fixtures"""
        self.loop.close()

    def test_listen_for_disconnect_normal_flow(self):
        """Test that listen_for_disconnect waits for disconnect message"""
        from fastdeploy.entrypoints.openai.utils import listen_for_disconnect

        # Setup mock request
        mock_request = MagicMock()
        receive_call_count = 0

        # Create mock messages - normal messages followed by disconnect
        messages = [
            {"type": "http.request", "body": b"some data"},
            {"type": "http.request", "body": b"more data"},
            {"type": "http.disconnect"},  # This should break the loop
        ]

        # Setup receive method to return messages sequentially
        receive_iter = iter(messages)

        async def mock_receive():
            nonlocal receive_call_count
            receive_call_count += 1
            try:
                return next(receive_iter)
            except StopIteration:
                # After all messages, return a message that keeps it waiting
                await asyncio.Future()  # Never completes

        mock_request.receive = mock_receive

        # Run the function in the event loop
        self.loop.run_until_complete(listen_for_disconnect(mock_request))

        # Verify that receive was called multiple times
        self.assertGreaterEqual(receive_call_count, 3)

    def test_listen_for_disconnect_immediate_disconnect(self):
        """Test that listen_for_disconnect returns immediately on disconnect"""
        from fastdeploy.entrypoints.openai.utils import listen_for_disconnect

        # Setup mock request
        mock_request = MagicMock()
        receive_called = False

        # Setup receive to return disconnect immediately (as a coroutine)
        async def mock_receive():
            nonlocal receive_called
            receive_called = True
            return {"type": "http.disconnect"}

        mock_request.receive = mock_receive

        # Run the function in the event loop
        self.loop.run_until_complete(listen_for_disconnect(mock_request))

        # Verify that receive was called exactly once
        self.assertTrue(receive_called)

    def test_listen_for_disconnect_timeout(self):
        """Test that listen_for_disconnect can be cancelled with timeout"""
        from fastdeploy.entrypoints.openai.utils import listen_for_disconnect

        # Setup mock request
        mock_request = MagicMock()

        # Setup receive to never return disconnect
        async def mock_receive():
            await asyncio.Future()  # Never completes

        mock_request.receive = mock_receive

        # Run with timeout to test cancellation
        with self.assertRaises(asyncio.TimeoutError):
            self.loop.run_until_complete(asyncio.wait_for(listen_for_disconnect(mock_request), timeout=0.01))

    def test_listen_for_disconnect_various_message_types(self):
        """Test that listen_for_disconnect ignores non-disconnect messages"""
        from fastdeploy.entrypoints.openai.utils import listen_for_disconnect

        # Setup mock request
        mock_request = MagicMock()
        receive_call_count = 0

        # Create various non-disconnect messages
        messages = [
            {"type": "http.request", "body": b"data"},
            {"type": "http.response", "status": 200},
            {"type": "websocket.connect"},
            {"type": "http.request", "body": b"more data"},
            {"type": "http.disconnect"},  # Final disconnect
        ]

        # Setup receive method
        receive_iter = iter(messages)

        async def mock_receive():
            nonlocal receive_call_count
            receive_call_count += 1
            try:
                return next(receive_iter)
            except StopIteration:
                await asyncio.Future()

        mock_request.receive = mock_receive

        # Run the function in the event loop
        self.loop.run_until_complete(listen_for_disconnect(mock_request))

        # Verify all messages were processed
        self.assertEqual(receive_call_count, 5)


if __name__ == "__main__":
    unittest.main()
