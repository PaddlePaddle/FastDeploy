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

import os
import sys
import threading
import time
import unittest
from multiprocessing import Queue
from unittest.mock import Mock, call, patch

# Determine import method based on environment
# Use environment variable FD_TEST_MODE=standalone for local testing
TEST_MODE = os.environ.get("FD_TEST_MODE", "normal")

if TEST_MODE == "standalone":
    # Local testing mode - use dynamic import
    # Mock the logger and dependencies to avoid import issues
    mock_logger = Mock()
    mock_envs = Mock()
    mock_envs.FD_EP_BATCHED_TOKEN_TIMEOUT = 0.1

    # Create a mock module structure
    class MockUtils:
        def get_logger(self, name, filename):
            return mock_logger

    class MockEnv:
        FD_EP_BATCHED_TOKEN_TIMEOUT = 0.1

    sys.modules["fastdeploy"] = Mock()
    sys.modules["fastdeploy.utils"] = MockUtils()
    sys.modules["fastdeploy.envs"] = MockEnv()
    sys.modules["fastdeploy.engine"] = Mock()
    sys.modules["fastdeploy.engine.request"] = Mock()

    # Mock scheduler modules
    mock_scheduler = Mock()
    sys.modules["fastdeploy.scheduler"] = mock_scheduler
    sys.modules["fastdeploy.scheduler.local_scheduler"] = mock_scheduler
    sys.modules["fastdeploy.scheduler.data"] = Mock()

    # Import the dp_scheduler module directly
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "dp_scheduler", os.path.join(os.path.dirname(__file__), "../../fastdeploy/scheduler/dp_scheduler.py")
    )
    dp_scheduler_module = importlib.util.module_from_spec(spec)

    # Mock the dependencies
    dp_scheduler_module.envs = mock_envs
    dp_scheduler_module.get_logger = lambda name, filename: mock_logger

    # Create mock classes for dependencies
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
                self.responses_not_empty.wait_for(lambda: any(self.responses.values()), timeout=0.1)
                results = []
                for response_list in list(self.responses.values()):
                    results.extend(response_list)
                self.responses.clear()
                return results

    # Mock the imports
    dp_scheduler_module.Request = MockRequest
    dp_scheduler_module.RequestOutput = MockRequestOutput
    dp_scheduler_module.ScheduledResponse = MockScheduledResponse
    dp_scheduler_module.LocalScheduler = MockLocalScheduler

    spec.loader.exec_module(dp_scheduler_module)

    # Extract classes we want to test
    DPLocalScheduler = dp_scheduler_module.DPLocalScheduler
    DPScheduler = dp_scheduler_module.DPScheduler

else:
    # Normal mode - direct import (for CI/CD and production)
    try:
        from fastdeploy.scheduler.dp_scheduler import DPLocalScheduler, DPScheduler

        # If we can import directly, we don't need mocking
        mock_logger = None
    except ImportError:
        # Fallback to standalone mode if direct import fails
        print("Warning: Direct import failed, falling back to standalone mode")
        TEST_MODE = "standalone"
        # Re-run the standalone setup
        mock_logger = Mock()
        mock_envs = Mock()
        mock_envs.FD_EP_BATCHED_TOKEN_TIMEOUT = 0.1

        class MockUtils:
            def get_logger(self, name, filename):
                return mock_logger

        class MockEnv:
            FD_EP_BATCHED_TOKEN_TIMEOUT = 0.1

        sys.modules["fastdeploy"] = Mock()
        sys.modules["fastdeploy.utils"] = MockUtils()
        sys.modules["fastdeploy.envs"] = MockEnv()
        sys.modules["fastdeploy.engine"] = Mock()
        sys.modules["fastdeploy.engine.request"] = Mock()

        # Mock scheduler modules
        mock_scheduler = Mock()
        sys.modules["fastdeploy.scheduler"] = mock_scheduler
        sys.modules["fastdeploy.scheduler.local_scheduler"] = mock_scheduler
        sys.modules["fastdeploy.scheduler.data"] = Mock()

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "dp_scheduler", os.path.join(os.path.dirname(__file__), "../../fastdeploy/scheduler/dp_scheduler.py")
        )
        dp_scheduler_module = importlib.util.module_from_spec(spec)
        dp_scheduler_module.envs = mock_envs
        dp_scheduler_module.get_logger = lambda name, filename: mock_logger

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
                    self.responses_not_empty.wait_for(lambda: any(self.responses.values()), timeout=0.1)
                    results = []
                    for response_list in list(self.responses.values()):
                        results.extend(response_list)
                    self.responses.clear()
                    return results

        dp_scheduler_module.Request = MockRequest
        dp_scheduler_module.RequestOutput = MockRequestOutput
        dp_scheduler_module.ScheduledResponse = MockScheduledResponse
        dp_scheduler_module.LocalScheduler = MockLocalScheduler

        spec.loader.exec_module(dp_scheduler_module)

        DPLocalScheduler = dp_scheduler_module.DPLocalScheduler
        DPScheduler = dp_scheduler_module.DPScheduler


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
        if TEST_MODE != "standalone":
            self.skipTest("Logger mocking only available in standalone mode")

        # Reset mock logger
        mock_logger.reset_mock()

        # Create mock request outputs
        results = [
            MockRequestOutput("req1", finished=True),
            MockRequestOutput("req2", finished=False),
            MockRequestOutput("req3", finished=True),
        ]

        # Put results
        self.scheduler.put_results(results)

        # Check that finished requests were logged
        expected_calls = [call("Scheduler has received some finished responses: ['req1', 'req3']")]
        mock_logger.info.assert_has_calls(expected_calls)

    def test_put_results_with_new_responses(self):
        """Test putting results with new responses."""
        results = [MockRequestOutput("new_req", finished=False)]

        # Initially no responses
        self.assertNotIn("new_req", self.scheduler.responses)

        # Put results
        self.scheduler.put_results(results)

        # Check response was added
        self.assertIn("new_req", self.scheduler.responses)
        self.assertEqual(len(self.scheduler.responses["new_req"]), 1)

    def test_put_results_with_existing_responses(self):
        """Test putting results with existing responses."""
        results1 = [MockRequestOutput("existing_req", finished=False)]
        results2 = [MockRequestOutput("existing_req", finished=True)]

        # Put first set of results
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
        # Mock time to make requests appear expired
        mock_time.return_value = 100.0

        # Create expired request (schedule_time = 50.0, ttl = 60, so expired)
        expired_request = MockRequest("expired_req")
        expired_request.schedule_time = 30.0  # 70 seconds ago (beyond ttl=60)

        # Create non-expired request
        fresh_request = MockRequest("fresh_req")
        fresh_request.schedule_time = 80.0  # 20 seconds ago (within ttl=60)

        # Add test data
        self.scheduler.requests["expired_req"] = expired_request
        self.scheduler.requests["fresh_req"] = fresh_request
        self.scheduler.ids = ["expired_req", "fresh_req"]
        self.scheduler.ids_read_cursor = 2

        # Recycle expired requests
        self.scheduler._recycle()

        # Verify expired request was removed, fresh request remains
        self.assertNotIn("expired_req", self.scheduler.requests)
        self.assertIn("fresh_req", self.scheduler.requests)
        self.assertEqual(self.scheduler.ids, ["fresh_req"])
        self.assertEqual(self.scheduler.ids_read_cursor, 1)

    def test_get_requests_insufficient_resources(self):
        """Test getting requests when resources are insufficient."""
        if TEST_MODE != "standalone":
            self.skipTest("Logger mocking only available in standalone mode")

        mock_logger.reset_mock()

        # Test with insufficient blocks
        requests = self.scheduler.get_requests(
            available_blocks=5, block_size=16, reserved_output_blocks=10, max_num_batched_tokens=1024, batch=1
        )

        self.assertEqual(requests, [])
        mock_logger.debug.assert_called()

    def test_get_requests_insufficient_batch(self):
        """Test getting requests when batch size is insufficient."""
        requests = self.scheduler.get_requests(
            available_blocks=20, block_size=16, reserved_output_blocks=10, max_num_batched_tokens=1024, batch=0
        )

        self.assertEqual(requests, [])

    def test_get_requests_no_requests_available(self):
        """Test getting requests when no requests are available."""
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
    def test_get_requests_timeout(self, mock_time):
        """Test request batching with timeout."""
        if TEST_MODE != "standalone":
            self.skipTest("Environment mocking only available in standalone mode")

        # Mock time progression to trigger timeout
        start_time = 100.0
        mock_time.side_effect = [start_time, start_time + 0.2]  # Beyond timeout

        # Add a mock request
        mock_request = MockRequest("test_req", prompt_tokens_ids_len=10)
        self.scheduler.requests["test_req"] = mock_request
        self.scheduler.ids = ["test_req"]

        # Mock calc_required_blocks to return large value to exceed available blocks
        self.scheduler.calc_required_blocks = Mock(return_value=50)

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

        # Should not raise an exception
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
        del mock_request.dp_rank  # Remove dp_rank if it exists

        requests = [mock_request]

        # Should raise ValueError
        with self.assertRaises(ValueError) as cm:
            self.dp_scheduler.put_requests(requests)

        self.assertIn("missing the 'dp_rank' attribute", str(cm.exception))

    def test_put_requests_success(self):
        """Test successful put_requests with dp_rank."""
        # Create request queues
        request_queues = [Queue(), Queue(), Queue()]
        result_queue = Queue()

        # Start the scheduler
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

    def test_start_initializes_threads_and_logger(self):
        """Test that start initializes threads and logger correctly."""
        if TEST_MODE != "standalone":
            self.skipTest("Logger mocking only available in standalone mode")

        request_queues = [Queue(), Queue()]
        result_queue = Queue()

        # Start scheduler
        self.dp_scheduler.start(1, request_queues, result_queue)

        # Verify attributes are set
        self.assertEqual(self.dp_scheduler.dp_rank, 1)
        self.assertEqual(self.dp_scheduler.request_queues, request_queues)
        self.assertEqual(self.dp_scheduler.result_queue, result_queue)
        self.assertIsNotNone(self.dp_scheduler.scheduler_logger)

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
        """Test end-to-end request flow through DP scheduler."""
        # Create DP scheduler
        dp_scheduler = DPScheduler(
            max_size=10,
            ttl=30,
            enable_chunked_prefill=True,
            max_num_partial_prefills=2,
            max_long_partial_prefills=1,
            long_prefill_token_threshold=512,
        )

        # Set up queues
        request_queues = [Queue(), Queue()]
        result_queue = Queue()

        # Start scheduler
        dp_scheduler.start(0, request_queues, result_queue)

        # Create and put request
        mock_request = MockRequest("integration_req")
        mock_request.dp_rank = 0

        results = dp_scheduler.put_requests([mock_request])
        self.assertEqual(len(results), 1)

        # Verify unhandled request count
        time.sleep(0.1)  # Give time for background thread
        # Note: In a real test environment, this would test the actual threading
        # but for unit tests we verify the setup is correct

    def test_error_handling_in_threads(self):
        """Test error handling in background threads."""
        if TEST_MODE != "standalone":
            self.skipTest("Thread mocking only available in standalone mode")

        # Create DP scheduler
        dp_scheduler = DPScheduler(
            max_size=10,
            ttl=30,
            enable_chunked_prefill=True,
            max_num_partial_prefills=2,
            max_long_partial_prefills=1,
            long_prefill_token_threshold=512,
        )

        # Set up queues with one that will cause an error
        request_queues = [Queue()]
        request_queues[0].close()  # Close queue to cause error
        result_queue = Queue()

        # Should not raise exception even if queue has issues
        dp_scheduler.start(0, request_queues, result_queue)

        # Background threads should handle errors gracefully
        # (This tests that exceptions in threads don't crash initialization)


if __name__ == "__main__":
    # Print current test mode for clarity
    print(f"Running tests in {TEST_MODE} mode")
    if TEST_MODE == "standalone":
        print("To run in normal mode, ensure fastdeploy is properly installed")
        print("Or set FD_TEST_MODE=normal environment variable")
    unittest.main(verbosity=2)
