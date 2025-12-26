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

import sys
import time
import unittest
from multiprocessing import Queue
from unittest.mock import Mock, patch

# Mock all external dependencies before importing anything
mock_logger = Mock()


# Create a proper mock for FD_EP_BATCHED_TOKEN_TIMEOUT that can be compared with float
class MockEnv:
    FD_EP_BATCHED_TOKEN_TIMEOUT = 0.1


mock_envs = MockEnv()

# Mock threading module to prevent real thread creation
import threading

mock_threading = Mock()
sys.modules["threading"] = mock_threading
mock_threading.Thread = Mock()
mock_threading.Lock = Mock(return_value=Mock())
mock_threading.Condition = Mock(return_value=Mock())

# Create mock modules
sys.modules["fastdeploy"] = Mock()
sys.modules["fastdeploy.utils"] = Mock()
sys.modules["fastdeploy.envs"] = mock_envs
sys.modules["fastdeploy.engine"] = Mock()
sys.modules["fastdeploy.engine.request"] = Mock()
sys.modules["fastdeploy.scheduler"] = Mock()
sys.modules["fastdeploy.scheduler.local_scheduler"] = Mock()
sys.modules["fastdeploy.scheduler.data"] = Mock()

# Mock the get_logger function
sys.modules["fastdeploy.utils"].get_logger = Mock(return_value=mock_logger)


# Mock the Request, RequestOutput, and ScheduledResponse classes
class MockRequest:
    def __init__(self, request_id, prompt_tokens_ids_len=10):
        self.request_id = request_id
        self.prompt_tokens_ids_len = prompt_tokens_ids_len
        self.schedule_time = time.time()
        self.raw = self


class MockRequestOutput:
    def __init__(self, request_id, finished=False):
        self.request_id = request_id
        self.finished = finished


class MockScheduledResponse:
    def __init__(self, request_output):
        self.request_id = request_output.request_id
        self.finished = request_output.finished
        self.raw = self


# Mock LocalScheduler base class
class MockLocalScheduler:
    def __init__(
        self,
        max_size,
        ttl,
        enable_chunked_prefill,
        max_num_partial_prefills,
        max_long_partial_prefills,
        long_prefill_token_threshold,
    ):
        self.max_size = max_size
        self.ttl = ttl
        self.mutex = threading.Lock()
        self.requests = {}
        self.responses = {}
        self.ids = []
        self.ids_read_cursor = 0
        self.requests_not_empty = threading.Condition()
        self.responses_not_empty = threading.Condition()
        self.batch_responses_per_step = list()

    def calc_required_blocks(self, token_len, block_size):
        return (token_len + block_size - 1) // block_size

    def put_requests(self, requests):
        with self.mutex:
            for request in requests:
                if request.request_id not in self.requests:
                    self.requests[request.request_id] = request
                    self.ids.append(request.request_id)
        with self.requests_not_empty:
            self.requests_not_empty.notify_all()

    def get_results(self):
        with self.responses_not_empty:
            # Don't actually wait, just check if there are responses
            if any(self.responses.values()):
                results = []
                for response_list in list(self.responses.values()):
                    results.extend(response_list)
                self.responses.clear()
                return results
            return []

    def _recycle(self, request_id=None):
        """Mock implementation of _recycle method."""
        if request_id is not None:
            self.requests.pop(request_id, None)
            self.responses.pop(request_id, None)
            if hasattr(self, "splitwise_role") and self.splitwise_role == "decode":
                return
            if request_id in self.ids:
                self.ids.pop(self.ids.index(request_id))
            self.ids_read_cursor = max(0, self.ids_read_cursor - 1)
            return

        if self.max_size <= 0:
            return

        if len(self.requests) <= self.max_size:
            return

        now = time.time()
        expired_ids = []
        for req_id in self.ids:
            if req_id in self.requests:
                request = self.requests[req_id]
                if now - request.schedule_time >= self.ttl:
                    expired_ids.append(req_id)
                else:
                    break

        for expired_id in expired_ids:
            self.requests.pop(expired_id, None)
            self.responses.pop(expired_id, None)
            if expired_id in self.ids:
                self.ids.pop(self.ids.index(expired_id))

        if len(expired_ids) > 0:
            self.ids_read_cursor = max(0, self.ids_read_cursor - len(expired_ids))


# Set up the mock classes in the modules
sys.modules["fastdeploy.engine.request"].Request = MockRequest
sys.modules["fastdeploy.engine.request"].RequestOutput = MockRequestOutput
sys.modules["fastdeploy.scheduler.data"].ScheduledResponse = MockScheduledResponse
sys.modules["fastdeploy.scheduler.local_scheduler"].LocalScheduler = MockLocalScheduler

# Now we can import the dp_scheduler module with all dependencies mocked
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "dp_scheduler", os.path.join(os.path.dirname(__file__), "../../fastdeploy/scheduler/dp_scheduler.py")
)
dp_scheduler_module = importlib.util.module_from_spec(spec)

# Mock the dependencies in the module
dp_scheduler_module.envs = mock_envs
dp_scheduler_module.get_logger = Mock(return_value=mock_logger)
dp_scheduler_module.threading = mock_threading  # Add threading to the module

# Execute the module
spec.loader.exec_module(dp_scheduler_module)

# Extract the classes we want to test
DPLocalScheduler = dp_scheduler_module.DPLocalScheduler
DPScheduler = dp_scheduler_module.DPScheduler

# Override the scheduler_logger to use our mock
original_init = DPLocalScheduler.__init__


def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.scheduler_logger = mock_logger


DPLocalScheduler.__init__ = patched_init


class TestDPLocalScheduler(unittest.TestCase):
    """Test cases for DPLocalScheduler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = DPLocalScheduler(
            max_size=100,
            ttl=60,
            enable_chunked_prefill=True,
            max_num_partial_prefills=4,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=1024,
            splitwise_role="prefill",
        )

    def test_initialization_with_default_role(self):
        """Test scheduler initialization with default splitwise_role."""
        scheduler = DPLocalScheduler(
            max_size=50,
            ttl=30,
            enable_chunked_prefill=False,
            max_num_partial_prefills=2,
            max_long_partial_prefills=1,
            long_prefill_token_threshold=512,
        )
        self.assertEqual(scheduler.splitwise_role, "prefill")
        self.assertEqual(scheduler.max_size, 50)
        self.assertEqual(scheduler.ttl, 30)

    def test_initialization_with_custom_role(self):
        """Test scheduler initialization with custom splitwise_role."""
        scheduler = DPLocalScheduler(
            max_size=50,
            ttl=30,
            enable_chunked_prefill=False,
            max_num_partial_prefills=2,
            max_long_partial_prefills=1,
            long_prefill_token_threshold=512,
            splitwise_role="decode",
        )
        self.assertEqual(scheduler.splitwise_role, "decode")

    def test_put_results_with_finished_requests(self):
        """Test putting results with finished requests."""
        # Reset mock logger
        mock_logger.reset_mock()

        # Create mock request outputs
        results = [
            MockRequestOutput("req1", finished=True),
            MockRequestOutput("req2", finished=False),
            MockRequestOutput("req3", finished=True),
        ]

        # Put results - this should work without threading issues since we're using the real implementation
        with patch.object(self.scheduler, "responses_not_empty"):
            self.scheduler.put_results(results)

        # Check that finished requests were logged - the logger should have been called
        self.assertTrue(mock_logger.info.called)
        # Get the actual call arguments to verify the message format
        call_args = mock_logger.info.call_args[0][0]
        self.assertIn("finished responses", call_args)
        self.assertIn("req1", call_args)
        self.assertIn("req3", call_args)

    def test_put_results_with_new_responses(self):
        """Test putting results with new responses."""
        results = [MockRequestOutput("new_req", finished=False)]

        # Initially no responses
        self.assertNotIn("new_req", self.scheduler.responses)

        # Put results - mock the condition variable to avoid threading issues
        with patch.object(self.scheduler, "responses_not_empty"):
            self.scheduler.put_results(results)

        # Check response was added
        self.assertIn("new_req", self.scheduler.responses)
        self.assertEqual(len(self.scheduler.responses["new_req"]), 1)

    def test_put_results_with_existing_responses(self):
        """Test putting results with existing responses."""
        results1 = [MockRequestOutput("existing_req", finished=False)]
        results2 = [MockRequestOutput("existing_req", finished=True)]

        # Put first set of results - mock the condition variable to avoid threading issues
        with patch.object(self.scheduler, "responses_not_empty"):
            self.scheduler.put_results(results1)
            self.assertEqual(len(self.scheduler.responses["existing_req"]), 1)

            # Put second set of results
            self.scheduler.put_results(results2)
            self.assertEqual(len(self.scheduler.responses["existing_req"]), 2)

    def test_recycle_specific_request_id(self):
        """Test recycling a specific request ID."""
        # Add some test data
        self.scheduler.requests["req1"] = MockRequest("req1")
        self.scheduler.responses["req1"] = [MockScheduledResponse(MockRequestOutput("req1"))]
        self.scheduler.ids = ["req1", "req2"]
        self.scheduler.ids_read_cursor = 1

        # Recycle specific request
        self.scheduler._recycle("req1")

        # Verify request was removed
        self.assertNotIn("req1", self.scheduler.requests)
        self.assertNotIn("req1", self.scheduler.responses)
        self.assertEqual(self.scheduler.ids, ["req2"])
        self.assertEqual(self.scheduler.ids_read_cursor, 0)

    def test_recycle_specific_request_id_decode_role(self):
        """Test recycling a specific request ID in decode role."""
        scheduler = DPLocalScheduler(
            max_size=100,
            ttl=60,
            enable_chunked_prefill=True,
            max_num_partial_prefills=4,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=1024,
            splitwise_role="decode",
        )

        # Add some test data
        scheduler.requests["req1"] = MockRequest("req1")
        scheduler.responses["req1"] = [MockScheduledResponse(MockRequestOutput("req1"))]
        scheduler.ids = ["req1", "req2"]
        scheduler.ids_read_cursor = 1

        # Recycle specific request (should not modify ids in decode role)
        scheduler._recycle("req1")

        # Verify request and response were removed but ids unchanged
        self.assertNotIn("req1", scheduler.requests)
        self.assertNotIn("req1", scheduler.responses)
        self.assertEqual(scheduler.ids, ["req1", "req2"])  # Should not change in decode role
        self.assertEqual(scheduler.ids_read_cursor, 1)  # Should not change in decode role

    def test_recycle_with_max_size_zero(self):
        """Test recycling when max_size is 0 (unlimited)."""
        scheduler = DPLocalScheduler(
            max_size=0,
            ttl=60,
            enable_chunked_prefill=True,
            max_num_partial_prefills=4,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=1024,
        )

        # Add test data
        scheduler.requests["req1"] = MockRequest("req1")
        scheduler.responses["req1"] = [MockScheduledResponse(MockRequestOutput("req1"))]
        scheduler.ids = ["req1"]

        # Should return early without recycling
        scheduler._recycle()

        # Data should remain unchanged
        self.assertIn("req1", scheduler.requests)
        self.assertIn("req1", scheduler.responses)

    def test_recycle_under_max_size(self):
        """Test recycling when under max_size limit."""
        # Add test data under limit
        self.scheduler.requests["req1"] = MockRequest("req1")
        self.scheduler.requests["req2"] = MockRequest("req2")
        self.scheduler.ids = ["req1", "req2"]

        # Should return early without recycling
        self.scheduler._recycle()

        # Data should remain unchanged
        self.assertIn("req1", self.scheduler.requests)
        self.assertIn("req2", self.scheduler.requests)

    @patch("time.time")
    def test_recycle_expired_requests(self, mock_time):
        """Test recycling expired requests."""
        # Create a scheduler with smaller max_size to trigger recycling
        scheduler = DPLocalScheduler(
            max_size=1,  # Set to 1 to trigger recycling when we have 2 requests
            ttl=60,
            enable_chunked_prefill=True,
            max_num_partial_prefills=4,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=1024,
        )

        # Mock time to make requests appear expired
        mock_time.return_value = 100.0

        # Create expired request (schedule_time = 50.0, ttl = 60, so expired)
        expired_request = MockRequest("expired_req")
        expired_request.schedule_time = 30.0  # 70 seconds ago (beyond ttl=60)

        # Create non-expired request
        fresh_request = MockRequest("fresh_req")
        fresh_request.schedule_time = 80.0  # 20 seconds ago (within ttl=60)

        # Add test data
        scheduler.requests["expired_req"] = expired_request
        scheduler.requests["fresh_req"] = fresh_request
        scheduler.ids = ["expired_req", "fresh_req"]
        scheduler.ids_read_cursor = 2

        # Recycle expired requests
        scheduler._recycle()

        # Verify expired request was removed, fresh request remains
        self.assertNotIn("expired_req", scheduler.requests)
        self.assertIn("fresh_req", scheduler.requests)
        self.assertEqual(scheduler.ids, ["fresh_req"])
        self.assertEqual(scheduler.ids_read_cursor, 1)

    def test_get_requests_insufficient_resources(self):
        """Test getting requests when resources are insufficient."""
        mock_logger.reset_mock()

        # Test with insufficient blocks - mock the condition variable to avoid threading issues
        with patch.object(self.scheduler, "requests_not_empty"):
            requests = self.scheduler.get_requests(
                available_blocks=5, block_size=16, reserved_output_blocks=10, max_num_batched_tokens=1024, batch=1
            )

        self.assertEqual(requests, [])
        # The logger should have been called for insufficient resources
        self.assertTrue(mock_logger.debug.called)
        # Check the message contains expected content
        call_args = mock_logger.debug.call_args[0][0]
        self.assertIn("insufficient", call_args.lower())

    def test_get_requests_insufficient_batch(self):
        """Test getting requests when batch size is insufficient."""
        with patch.object(self.scheduler, "requests_not_empty"):
            requests = self.scheduler.get_requests(
                available_blocks=20, block_size=16, reserved_output_blocks=10, max_num_batched_tokens=1024, batch=0
            )

        self.assertEqual(requests, [])

    @patch("time.time")
    @patch.object(dp_scheduler_module, "envs")
    def test_get_requests_no_requests_available(self, mock_envs, mock_time):
        """Test getting requests when no requests are available."""
        # Mock envs to return our mock environment
        mock_envs.FD_EP_BATCHED_TOKEN_TIMEOUT = 0.1

        # Mock time to return consistent values - provide enough values for multiple calls
        time_values = [100.0, 100.1, 100.2, 100.3, 100.4, 100.5]  # Multiple values for the loop
        mock_time.side_effect = time_values

        # Mock the condition variable to avoid threading issues
        with patch.object(self.scheduler, "requests_not_empty"):
            requests = self.scheduler.get_requests(
                available_blocks=20, block_size=16, reserved_output_blocks=10, max_num_batched_tokens=1024, batch=1
            )

        # Should return empty list after timeout
        self.assertEqual(requests, [])

    def test_get_requests_successful_batching(self):
        """Test successful request batching."""
        # Add a mock request
        mock_request = MockRequest("test_req", prompt_tokens_ids_len=10)
        self.scheduler.requests["test_req"] = mock_request
        self.scheduler.ids = ["test_req"]

        # Mock calc_required_blocks to return small value
        self.scheduler.calc_required_blocks = Mock(return_value=1)

        requests = self.scheduler.get_requests(
            available_blocks=20, block_size=16, reserved_output_blocks=10, max_num_batched_tokens=1024, batch=1
        )

        # Should get the request
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].request_id, "test_req")

    @patch("time.time")
    @patch.object(dp_scheduler_module, "envs")
    def test_get_requests_timeout(self, mock_envs, mock_time):
        """Test request batching with timeout."""
        # Mock envs to return our mock environment
        mock_envs.FD_EP_BATCHED_TOKEN_TIMEOUT = 0.1

        # Mock time to return consistent values - provide enough values for multiple calls
        time_values = [100.0, 100.1, 100.2, 100.3, 100.4, 100.5]  # Multiple values for the loop
        mock_time.side_effect = time_values

        # Add a mock request
        mock_request = MockRequest("test_req", prompt_tokens_ids_len=10)
        self.scheduler.requests["test_req"] = mock_request
        self.scheduler.ids = ["test_req"]

        # Mock calc_required_blocks to return large value to exceed available blocks
        self.scheduler.calc_required_blocks = Mock(return_value=50)

        # Mock the condition variable to avoid threading issues
        with patch.object(self.scheduler, "requests_not_empty"):
            requests = self.scheduler.get_requests(
                available_blocks=20, block_size=16, reserved_output_blocks=10, max_num_batched_tokens=1024, batch=1
            )

        # Should return empty due to timeout
        self.assertEqual(requests, [])


class TestDPScheduler(unittest.TestCase):
    """Test cases for DPScheduler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.dp_scheduler = DPScheduler(
            max_size=100,
            ttl=60,
            enable_chunked_prefill=True,
            max_num_partial_prefills=4,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=1024,
            splitwise_role="prefill",
        )

    def test_initialization(self):
        """Test DPScheduler initialization."""
        self.assertIsNotNone(self.dp_scheduler._scheduler)
        self.assertEqual(self.dp_scheduler._scheduler.splitwise_role, "prefill")

    def test_get_unhandled_request_num(self):
        """Test getting number of unhandled requests."""
        # Initially should be 0
        self.assertEqual(self.dp_scheduler.get_unhandled_request_num(), 0)

        # Add a request to the internal scheduler
        mock_request = MockRequest("test_req")
        self.dp_scheduler._scheduler.requests["test_req"] = mock_request

        # Should return 1
        self.assertEqual(self.dp_scheduler.get_unhandled_request_num(), 1)

    def test_put_results(self):
        """Test putting results to DPScheduler."""
        results = [MockRequestOutput("test_req", finished=True)]

        # Should not raise an exception - mock the condition variable to avoid threading issues
        with patch.object(self.dp_scheduler._scheduler, "responses_not_empty"):
            self.dp_scheduler.put_results(results)

        # Verify results were added to the internal scheduler
        self.assertIn("test_req", self.dp_scheduler._scheduler.responses)

    def test_get_requests_delegates_to_scheduler(self):
        """Test that get_requests delegates to internal scheduler."""
        # Mock the internal scheduler's get_requests method
        expected_requests = [MockRequest("test_req")]
        self.dp_scheduler._scheduler.get_requests = Mock(return_value=expected_requests)

        requests = self.dp_scheduler.get_requests(
            available_blocks=20, block_size=16, reserved_output_blocks=10, max_num_batched_tokens=1024, batch=1
        )

        # Verify delegation
        self.dp_scheduler._scheduler.get_requests.assert_called_once_with(20, 16, 10, 1024, 1)
        self.assertEqual(requests, expected_requests)

    def test_put_requests_missing_dp_rank(self):
        """Test put_requests raises error when dp_rank is missing."""
        # Create a request without dp_rank attribute
        mock_request = MockRequest("test_req")

        requests = [mock_request]

        # Should raise ValueError
        with self.assertRaises(ValueError) as cm:
            self.dp_scheduler.put_requests(requests)

        self.assertIn("missing the 'dp_rank' attribute", str(cm.exception))

    @patch("threading.Thread")
    def test_put_requests_success(self, mock_thread):
        """Test successful put_requests with dp_rank."""
        # Create request queues - use Mock instead of real Queue to avoid threading issues
        request_queues = [Mock(), Mock(), Mock()]
        result_queue = Mock()

        # Start the scheduler - this will create mocked threads
        self.dp_scheduler.start(0, request_queues, result_queue)

        # Create requests with dp_rank
        mock_request1 = MockRequest("test_req1")
        mock_request1.dp_rank = 0
        mock_request2 = MockRequest("test_req2")
        mock_request2.dp_rank = 1

        requests = [mock_request1, mock_request2]

        # Should not raise an exception
        results = self.dp_scheduler.put_requests(requests)

        # Verify results format
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], ("test_req1", None))
        self.assertEqual(results[1], ("test_req2", None))

        # Verify requests were put to the correct queues
        request_queues[0].put.assert_called_once_with(mock_request1)
        request_queues[1].put.assert_called_once_with(mock_request2)

    @patch("threading.Thread")
    def test_start_creates_threads(self, mock_thread):
        """Test that start creates and starts threads."""
        mock_thread.return_value = Mock()

        request_queues = [Queue(), Queue()]
        result_queue = Queue()

        self.dp_scheduler.start(0, request_queues, result_queue)

        # Should create 2 threads
        self.assertEqual(mock_thread.call_count, 2)

        # Both threads should be started
        mock_thread.return_value.start.assert_called()


class TestDPIntegration(unittest.TestCase):
    """Integration tests for DP Scheduler functionality."""

    def test_end_to_end_request_flow(self):
        """Test end-to-end request flow through DP scheduler - without real threads."""
        # Create DP scheduler
        dp_scheduler = DPScheduler(
            max_size=10,
            ttl=30,
            enable_chunked_prefill=True,
            max_num_partial_prefills=2,
            max_long_partial_prefills=1,
            long_prefill_token_threshold=512,
        )

        # Mock the start method to avoid creating real threads
        with patch.object(dp_scheduler, "start") as mock_start:
            # Set up test data directly
            dp_scheduler.dp_rank = 0
            dp_scheduler.request_queues = [Mock(), Mock()]
            dp_scheduler.result_queue = Mock()
            dp_scheduler.scheduler_logger = mock_logger
            dp_scheduler._scheduler.scheduler_logger = mock_logger

            # Test basic functionality without real threads
            mock_request = MockRequest("integration_req")
            mock_request.dp_rank = 0

            # Mock the request_queues to avoid real Queue operations
            dp_scheduler.request_queues[0].put = Mock()

            # Test put_requests functionality
            results = dp_scheduler.put_requests([mock_request])
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0], ("integration_req", None))

            # Verify the request was put to the correct queue
            dp_scheduler.request_queues[0].put.assert_called_once_with(mock_request)

            # Verify start method was not called (to avoid threads)
            mock_start.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
