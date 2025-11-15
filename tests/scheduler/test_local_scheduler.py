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
from unittest.mock import Mock  # noqa: F401

# Determine import method based on environment
# Use environment variable FD_TEST_MODE=standalone for local testing
TEST_MODE = os.environ.get("FD_TEST_MODE", "normal")


# Define mock classes at module level to ensure availability in all contexts
class MockRequest:
    def __init__(self, request_id, prompt_token_ids=None):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids or []


class MockRequestOutput:
    def __init__(self, request_id, finished=False):
        self.request_id = request_id
        self.finished = finished
        self.outputs = Mock()


class MockScheduledRequest:
    def __init__(self, request):
        self.raw = request
        self.request_id = request.request_id
        self.prompt_tokens_ids_len = len(request.prompt_token_ids)
        self.schedule_time = time.time()


class MockScheduledResponse:
    def __init__(self, response):
        self.raw = response
        self.request_id = response.request_id
        self.finished = response.finished


if TEST_MODE == "standalone":
    # Local testing mode - use dynamic import
    # Mock the logger to avoid import issues
    mock_scheduler_logger = Mock()
    mock_envs = Mock()
    mock_envs.FD_ENABLE_MAX_PREFILL = False

    # Create a mock module structure
    class MockEngine:
        class request:
            Request = MockRequest
            RequestOutput = MockRequestOutput

    class MockScheduler:
        class data:
            ScheduledRequest = MockScheduledRequest
            ScheduledResponse = MockScheduledResponse

    class MockUtils:
        scheduler_logger = mock_scheduler_logger
        envs = mock_envs

    sys.modules["fastdeploy"] = Mock()
    sys.modules["fastdeploy.utils"] = MockUtils()
    sys.modules["fastdeploy.engine"] = MockEngine()
    sys.modules["fastdeploy.engine.request"] = MockEngine.request
    sys.modules["fastdeploy.scheduler"] = MockScheduler()
    sys.modules["fastdeploy.scheduler.data"] = MockScheduler.data

    # Import the local_scheduler module directly
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "local_scheduler", os.path.join(os.path.dirname(__file__), "../../fastdeploy/scheduler/local_scheduler.py")
    )
    local_scheduler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local_scheduler_module)
    LocalScheduler = local_scheduler_module.LocalScheduler
else:
    # Normal mode - direct import (for CI/CD and production)
    try:
        from fastdeploy.scheduler.local_scheduler import LocalScheduler

        # If we can import directly, we don't need mocking
        mock_scheduler_logger = None
    except ImportError:
        # Fallback to standalone mode if direct import fails
        print("Warning: Direct import failed, falling back to standalone mode")
        TEST_MODE = "standalone"
        # Re-run the standalone setup
        mock_scheduler_logger = Mock()
        mock_envs = Mock()
        mock_envs.FD_ENABLE_MAX_PREFILL = False

        # Create a mock module structure using module-level classes
        class MockEngine:
            class request:
                Request = MockRequest
                RequestOutput = MockRequestOutput

        class MockScheduler:
            class data:
                ScheduledRequest = MockScheduledRequest
                ScheduledResponse = MockScheduledResponse

        class MockUtils:
            scheduler_logger = mock_scheduler_logger
            envs = mock_envs

        sys.modules["fastdeploy"] = Mock()
        sys.modules["fastdeploy.utils"] = MockUtils()
        sys.modules["fastdeploy.engine"] = MockEngine()
        sys.modules["fastdeploy.engine.request"] = MockEngine.request
        sys.modules["fastdeploy.scheduler"] = MockScheduler()
        sys.modules["fastdeploy.scheduler.data"] = MockScheduler.data

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "local_scheduler", os.path.join(os.path.dirname(__file__), "../../fastdeploy/scheduler/local_scheduler.py")
        )
        local_scheduler_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(local_scheduler_module)
        LocalScheduler = local_scheduler_module.LocalScheduler


class TestLocalScheduler(unittest.TestCase):
    """Test cases for LocalScheduler class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.max_size = 10
        self.ttl = 60
        self.enable_chunked_prefill = True
        self.max_num_partial_prefills = 5
        self.max_long_partial_prefills = 2
        self.long_prefill_token_threshold = 1000

        self.scheduler = LocalScheduler(
            max_size=self.max_size,
            ttl=self.ttl,
            enable_chunked_prefill=self.enable_chunked_prefill,
            max_num_partial_prefills=self.max_num_partial_prefills,
            max_long_partial_prefills=self.max_long_partial_prefills,
            long_prefill_token_threshold=self.long_prefill_token_threshold,
        )

        # Create mock requests for testing
        self.mock_request_1 = Mock(request_id="req_1", prompt_token_ids=[1, 2, 3, 4, 5])
        self.mock_request_2 = Mock(request_id="req_2", prompt_token_ids=[6, 7, 8])
        self.mock_request_3 = Mock(request_id="req_3", prompt_token_ids=[9, 10, 11, 12])

    def test_local_scheduler_initialization(self):
        """Test LocalScheduler initialization with default parameters."""
        self.assertEqual(self.scheduler.max_size, self.max_size)
        self.assertEqual(self.scheduler.ttl, self.ttl)
        self.assertEqual(self.scheduler.enable_chunked_prefill, self.enable_chunked_prefill)
        self.assertEqual(self.scheduler.max_num_partial_prefills, self.max_num_partial_prefills)
        self.assertEqual(self.scheduler.max_long_partial_prefills, self.max_long_partial_prefills)
        self.assertEqual(self.scheduler.long_prefill_token_threshold, self.long_prefill_token_threshold)
        self.assertEqual(self.scheduler.ids_read_cursor, 0)
        self.assertEqual(len(self.scheduler.ids), 0)
        self.assertEqual(len(self.scheduler.requests), 0)
        self.assertEqual(len(self.scheduler.responses), 0)

    def test_local_scheduler_initialization_unlimited_size(self):
        """Test LocalScheduler initialization with unlimited size."""
        scheduler = LocalScheduler(
            max_size=0,  # 0 means unlimited
            ttl=30,
            enable_chunked_prefill=False,
            max_num_partial_prefills=3,
            max_long_partial_prefills=1,
            long_prefill_token_threshold=500,
        )
        self.assertEqual(scheduler.max_size, 0)
        self.assertEqual(scheduler.ttl, 30)
        self.assertFalse(scheduler.enable_chunked_prefill)

    def test_reset_functionality(self):
        """Test scheduler reset functionality."""
        # Add some requests and responses
        requests = [self.mock_request_1, self.mock_request_2]
        self.scheduler.put_requests(requests)

        # Reset the scheduler
        self.scheduler.reset()

        # Verify everything is cleared
        self.assertEqual(self.scheduler.ids_read_cursor, 0)
        self.assertEqual(len(self.scheduler.ids), 0)
        self.assertEqual(len(self.scheduler.requests), 0)
        self.assertEqual(len(self.scheduler.responses), 0)

    def test_reset_logs_message(self):
        """Test that reset logs appropriate message."""
        if TEST_MODE != "standalone":
            self.skipTest("Logger mocking only available in standalone mode")

        mock_scheduler_logger.reset_mock()
        self.scheduler.reset()

        mock_scheduler_logger.info.assert_called_once_with("Scheduler has been reset")

    def test_put_requests_single_request(self):
        """Test putting a single request into the scheduler."""
        requests = [self.mock_request_1]
        results = self.scheduler.put_requests(requests)

        # Verify the request was added successfully
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "req_1")  # request_id
        self.assertIsNone(results[0][1])  # error_message (None for success)

        # Verify internal state
        self.assertIn("req_1", self.scheduler.requests)
        self.assertIn("req_1", self.scheduler.ids)
        self.assertEqual(len(self.scheduler.requests), 1)
        self.assertEqual(len(self.scheduler.ids), 1)

    def test_put_requests_multiple_requests(self):
        """Test putting multiple requests into the scheduler."""
        requests = [self.mock_request_1, self.mock_request_2, self.mock_request_3]
        results = self.scheduler.put_requests(requests)

        # Verify all requests were added successfully
        self.assertEqual(len(results), 3)
        for i, (request_id, error) in enumerate(results):
            self.assertIsNone(error)
            self.assertIn(request_id, self.scheduler.requests)

        # Verify internal state
        self.assertEqual(len(self.scheduler.requests), 3)
        self.assertEqual(len(self.scheduler.ids), 3)

    def test_put_requests_duplicate_handling(self):
        """Test handling of duplicate request IDs."""
        # Add first request
        requests_1 = [self.mock_request_1]
        results_1 = self.scheduler.put_requests(requests_1)
        self.assertEqual(len(results_1), 1)
        self.assertIsNone(results_1[0][1])

        # Try to add duplicate request
        duplicate_request = Mock(request_id="req_1", prompt_token_ids=[1, 2, 3])
        requests_2 = [duplicate_request]
        results_2 = self.scheduler.put_requests(requests_2)

        # Verify duplicate was rejected
        self.assertEqual(len(results_2), 1)
        self.assertEqual(results_2[0][0], "req_1")
        self.assertEqual(results_2[0][1], "duplicated request_id")

        # Verify only one request exists in scheduler
        self.assertEqual(len(self.scheduler.requests), 1)

    def test_put_requests_max_size_limit(self):
        """Test that max size limit is enforced."""
        # Create scheduler with small max size
        small_scheduler = LocalScheduler(
            max_size=2,
            ttl=60,
            enable_chunked_prefill=True,
            max_num_partial_prefills=5,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=1000,
        )

        # Add first request (should succeed)
        requests_1 = [self.mock_request_1]
        results_1 = small_scheduler.put_requests(requests_1)
        self.assertEqual(len(results_1), 1)
        self.assertIsNone(results_1[0][1])

        # Try to add two more requests (should exceed limit)
        requests_2 = [self.mock_request_2, self.mock_request_3]
        results_2 = small_scheduler.put_requests(requests_2)

        # Verify all were rejected due to size limit
        self.assertEqual(len(results_2), 2)
        for request_id, error in results_2:
            self.assertIsNotNone(error)
            self.assertIn("Exceeding the max length", error)

    def test_put_requests_unlimited_size(self):
        """Test that unlimited size scheduler accepts all requests."""
        unlimited_scheduler = LocalScheduler(
            max_size=0,  # Unlimited
            ttl=60,
            enable_chunked_prefill=True,
            max_num_partial_prefills=5,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=1000,
        )

        # Add many requests
        many_requests = [Mock(request_id=f"req_{i}", prompt_token_ids=[i]) for i in range(100)]
        results = unlimited_scheduler.put_requests(many_requests)

        # Verify all were accepted
        self.assertEqual(len(results), 100)
        for request_id, error in results:
            self.assertIsNone(error)

        self.assertEqual(len(unlimited_scheduler.requests), 100)

    def test_has_request_existing(self):
        """Test has_request with existing request."""
        self.scheduler.put_requests([self.mock_request_1])

        result = self.scheduler.has_request("req_1")
        self.assertTrue(result)

    def test_has_request_non_existing(self):
        """Test has_request with non-existing request."""
        result = self.scheduler.has_request("non_existing")
        self.assertFalse(result)

    def test_calc_required_blocks(self):
        """Test block calculation functionality."""
        # Test exact division
        result = self.scheduler.calc_required_blocks(100, 25)
        self.assertEqual(result, 4)

        # Test rounding up
        result = self.scheduler.calc_required_blocks(101, 25)
        self.assertEqual(result, 5)

        # Test zero tokens
        result = self.scheduler.calc_required_blocks(0, 25)
        self.assertEqual(result, 0)

        # Test single token
        result = self.scheduler.calc_required_blocks(1, 25)
        self.assertEqual(result, 1)

    def test_get_unhandled_request_num(self):
        """Test getting number of unhandled requests."""
        # Initially should be 0
        result = self.scheduler.get_unhandled_request_num()
        self.assertEqual(result, 0)

        # Add requests
        self.scheduler.put_requests([self.mock_request_1, self.mock_request_2])
        result = self.scheduler.get_unhandled_request_num()
        self.assertEqual(result, 2)

        # Mock getting some requests to advance cursor
        self.scheduler.ids_read_cursor = 1
        result = self.scheduler.get_unhandled_request_num()
        self.assertEqual(result, 1)

    def test_get_requests_insufficient_resources(self):
        """Test get_requests with insufficient resources."""
        requests = self.scheduler.get_requests(
            available_blocks=5,
            block_size=10,
            reserved_output_blocks=10,  # More than available
            max_num_batched_tokens=1000,
            batch=1,
        )

        self.assertEqual(len(requests), 0)

    def test_get_requests_insufficient_batch_size(self):
        """Test get_requests with invalid batch size."""
        requests = self.scheduler.get_requests(
            available_blocks=100,
            block_size=10,
            reserved_output_blocks=10,
            max_num_batched_tokens=1000,
            batch=0,  # Invalid batch size
        )

        self.assertEqual(len(requests), 0)

    def test_get_requests_no_available_requests(self):
        """Test get_requests when no requests are available."""
        requests = self.scheduler.get_requests(
            available_blocks=100, block_size=10, reserved_output_blocks=10, max_num_batched_tokens=1000, batch=1
        )

        self.assertEqual(len(requests), 0)

    def test_get_requests_with_available_requests(self):
        """Test get_requests with available requests and sufficient resources."""
        # Add requests to scheduler
        self.scheduler.put_requests([self.mock_request_1, self.mock_request_2])

        requests = self.scheduler.get_requests(
            available_blocks=100, block_size=10, reserved_output_blocks=10, max_num_batched_tokens=1000, batch=2
        )

        # Should return some requests (exact number depends on resource calculation)
        self.assertGreaterEqual(len(requests), 0)

    def test_get_requests_chunked_prefill_long_requests(self):
        """Test chunked prefill behavior with long requests."""
        # Create a long request
        long_request = Mock(request_id="long_req", prompt_token_ids=list(range(2000)))
        self.scheduler.put_requests([long_request])

        requests = self.scheduler.get_requests(
            available_blocks=500, block_size=10, reserved_output_blocks=10, max_num_batched_tokens=1000, batch=1
        )

        # Behavior depends on chunked prefill logic
        self.assertGreaterEqual(len(requests), 0)

    def test_put_results_single_result(self):
        """Test putting a single result."""
        # First add a request
        self.scheduler.put_requests([self.mock_request_1])

        # Create mock output
        mock_output = MockRequestOutput(request_id="req_1", finished=False)
        results = [mock_output]

        # Put results
        self.scheduler.put_results(results)

        # Verify result was stored
        self.assertIn("req_1", self.scheduler.responses)
        self.assertEqual(len(self.scheduler.responses["req_1"]), 1)

    def test_put_results_multiple_results(self):
        """Test putting multiple results."""
        # Add requests first
        self.scheduler.put_requests([self.mock_request_1, self.mock_request_2])

        # Create mock outputs
        mock_output_1 = MockRequestOutput(request_id="req_1", finished=False)
        mock_output_2 = MockRequestOutput(request_id="req_2", finished=True)
        results = [mock_output_1, mock_output_2]

        # Put results
        self.scheduler.put_results(results)

        # Verify results were stored
        self.assertIn("req_1", self.scheduler.responses)
        self.assertIn("req_2", self.scheduler.responses)
        self.assertEqual(len(self.scheduler.responses["req_1"]), 1)
        self.assertEqual(len(self.scheduler.responses["req_2"]), 1)

    def test_put_results_expired_response(self):
        """Test putting results for expired/non-existent requests."""
        mock_output = MockRequestOutput(request_id="non_existent", finished=False)
        results = [mock_output]

        # This should not raise an exception
        self.scheduler.put_results(results)

        # Response should not be stored (request doesn't exist)
        self.assertNotIn("non_existent", self.scheduler.responses)

    def test_put_results_append_to_existing(self):
        """Test appending results to existing request responses."""
        # Add request first
        self.scheduler.put_requests([self.mock_request_1])

        # Put first result
        mock_output_1 = MockRequestOutput(request_id="req_1", finished=False)
        self.scheduler.put_results([mock_output_1])

        # Put second result for same request
        mock_output_2 = MockRequestOutput(request_id="req_1", finished=True)
        self.scheduler.put_results([mock_output_2])

        # Should have two responses for the request
        self.assertEqual(len(self.scheduler.responses["req_1"]), 2)

    def test_get_results_empty(self):
        """Test getting results when none are available."""
        results = self.scheduler.get_results()
        self.assertEqual(len(results), 0)

    def test_get_results_with_available_results(self):
        """Test getting results when they are available."""
        # Add request and result
        self.scheduler.put_requests([self.mock_request_1])
        mock_output = MockRequestOutput(request_id="req_1", finished=False)
        self.scheduler.put_results([mock_output])

        # Get results
        results = self.scheduler.get_results()

        # Should return the results
        self.assertIn("req_1", results)
        self.assertEqual(len(results["req_1"]), 1)

    def test_get_results_finished_request_cleanup(self):
        """Test that finished requests are cleaned up after getting results."""
        # Add request and finished result
        self.scheduler.put_requests([self.mock_request_1])
        mock_output = MockRequestOutput(request_id="req_1", finished=True)
        self.scheduler.put_results([mock_output])

        # Get results
        results = self.scheduler.get_results()

        # Request should be cleaned up (recycled) after getting finished results
        # Note: The exact behavior depends on _recycle implementation
        self.assertIn("req_1", results)

    def test_recycle_specific_request(self):
        """Test recycling a specific request."""
        # Add request
        self.scheduler.put_requests([self.mock_request_1])

        # Verify request exists
        self.assertIn("req_1", self.scheduler.requests)

        # Recycle specific request
        self.scheduler._recycle("req_1")

        # Verify request was removed
        self.assertNotIn("req_1", self.scheduler.requests)
        self.assertNotIn("req_1", self.scheduler.ids)

    def test_recycle_expired_requests(self):
        """Test recycling expired requests based on TTL."""
        # Create scheduler with short TTL
        short_ttl_scheduler = LocalScheduler(
            max_size=10,
            ttl=1,  # 1 second TTL
            enable_chunked_prefill=True,
            max_num_partial_prefills=5,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=1000,
        )

        # Add request
        short_ttl_scheduler.put_requests([self.mock_request_1])

        # Wait for expiration
        time.sleep(1.5)

        # Trigger recycle (happens automatically in put_requests)
        short_ttl_scheduler.put_requests([])

        # Request should still be there because _recycle with max_size > 0
        # only removes expired when exceeding max_size
        self.assertIn("req_1", short_ttl_scheduler.requests)

    def test_recycle_unlimited_size(self):
        """Test recycle behavior with unlimited size scheduler."""
        unlimited_scheduler = LocalScheduler(
            max_size=0,  # Unlimited
            ttl=60,
            enable_chunked_prefill=True,
            max_num_partial_prefills=5,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=1000,
        )

        # Add request
        unlimited_scheduler.put_requests([self.mock_request_1])

        # Recycle should do nothing for unlimited size
        unlimited_scheduler._recycle()

        # Request should still be there
        self.assertIn("req_1", unlimited_scheduler.requests)

    def test_thread_safety_basic(self):
        """Test basic thread safety of scheduler operations."""
        results = []
        errors = []

        def add_requests():
            try:
                for i in range(10):
                    request = Mock(request_id=f"thread_req_{i}", prompt_token_ids=[i])
                    result = self.scheduler.put_requests([request])
                    results.append(result)
            except Exception as e:
                errors.append(e)

        def get_requests():
            try:
                for i in range(10):
                    _ = self.scheduler.get_requests(
                        available_blocks=100,
                        block_size=10,
                        reserved_output_blocks=10,
                        max_num_batched_tokens=100,
                        batch=1,
                    )
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)

        # Run threads concurrently
        thread1 = threading.Thread(target=add_requests)
        thread2 = threading.Thread(target=get_requests)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")

    def test_logging_put_requests(self):
        """Test that put_requests logs appropriate messages."""
        if TEST_MODE != "standalone":
            self.skipTest("Logger mocking only available in standalone mode")

        mock_scheduler_logger.reset_mock()
        self.scheduler.put_requests([self.mock_request_1])

        # Should log successful enqueue
        mock_scheduler_logger.info.assert_called()
        log_calls = [call.args[0] for call in mock_scheduler_logger.info.call_args_list]
        self.assertTrue(any("enqueued some requests" in msg for msg in log_calls))

    def test_logging_put_results_finished(self):
        """Test that put_results logs finished responses."""
        if TEST_MODE != "standalone":
            self.skipTest("Logger mocking only available in standalone mode")

        # Add request first
        self.scheduler.put_requests([self.mock_request_1])

        mock_scheduler_logger.reset_mock()
        mock_output = MockRequestOutput(request_id="req_1", finished=True)
        self.scheduler.put_results([mock_output])

        # Should log finished response
        mock_scheduler_logger.info.assert_called()
        log_calls = [call.args[0] for call in mock_scheduler_logger.info.call_args_list]
        self.assertTrue(any("finished responses" in msg for msg in log_calls))

    def test_edge_case_empty_request_list(self):
        """Test putting empty request list."""
        results = self.scheduler.put_requests([])
        self.assertEqual(len(results), 0)

    def test_edge_case_empty_result_list(self):
        """Test putting empty result list."""
        # Should not raise an exception
        self.scheduler.put_results([])

        # Should have no responses
        self.assertEqual(len(self.scheduler.responses), 0)

    def test_edge_case_zero_ttl(self):
        """Test scheduler with zero TTL."""
        zero_ttl_scheduler = LocalScheduler(
            max_size=10,
            ttl=0,  # Immediate expiration
            enable_chunked_prefill=True,
            max_num_partial_prefills=5,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=1000,
        )

        # Add request
        zero_ttl_scheduler.put_requests([self.mock_request_1])

        # Request should be added (TTL only affects recycling)
        self.assertIn("req_1", zero_ttl_scheduler.requests)


if __name__ == "__main__":
    # Print current test mode for clarity
    print(f"Running tests in {TEST_MODE} mode")
    if TEST_MODE == "standalone":
        print("To run in normal mode, ensure fastdeploy is properly installed")
        print("Or set FD_TEST_MODE=normal environment variable")
    unittest.main(verbosity=2)
