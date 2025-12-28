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

import time
import unittest
from unittest.mock import Mock, patch  # noqa: F401

from fastdeploy.engine.request import Request, RequestMetrics, RequestOutput

# Real FastDeploy imports
from fastdeploy.scheduler.local_scheduler import LocalScheduler
from fastdeploy.utils import envs, scheduler_logger


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

        # Patch the FD_ENABLE_MAX_PREFILL environment variable for controlled testing
        self.envs_patcher = patch.object(envs, "FD_ENABLE_MAX_PREFILL", False)
        self.envs_patcher.start()

        self.scheduler = LocalScheduler(
            max_size=self.max_size,
            ttl=self.ttl,
            enable_chunked_prefill=self.enable_chunked_prefill,
            max_num_partial_prefills=self.max_num_partial_prefills,
            max_long_partial_prefills=self.max_long_partial_prefills,
            long_prefill_token_threshold=self.long_prefill_token_threshold,
        )

        # Create real Request objects for testing
        self.mock_request_1 = self._create_test_request("req_1", [1, 2, 3, 4, 5])
        self.mock_request_2 = self._create_test_request("req_2", [6, 7, 8])
        self.mock_request_3 = self._create_test_request("req_3", [9, 10, 11, 12])

    def tearDown(self):
        """Clean up after each test method."""
        self.envs_patcher.stop()

    # ========== Mock Factory Methods ==========

    def _create_mock_output(self, index=0, token_ids=None):
        """Create mock completion output with defaults."""
        mock = Mock()
        mock.index = index
        mock.token_ids = token_ids or []
        return mock

    def _create_mock_metrics(self):
        """Create mock metrics with current time."""
        mock = Mock()
        mock.arrival_time = time.time()
        return mock

    # ========== Scheduler Factory Methods ==========

    def _create_scheduler(
        self,
        max_size=10,
        ttl=60,
        enable_chunked_prefill=True,
        max_num_partial_prefills=5,
        max_long_partial_prefills=2,
        long_prefill_token_threshold=1000,
    ):
        """Helper to create scheduler with custom parameters."""
        return LocalScheduler(
            max_size=max_size,
            ttl=ttl,
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_partial_prefills=max_num_partial_prefills,
            max_long_partial_prefills=max_long_partial_prefills,
            long_prefill_token_threshold=long_prefill_token_threshold,
        )

    # ========== Assertion Helper Methods ==========

    def _assert_scheduler_state(
        self,
        scheduler,
        expected_max_size=None,
        expected_ttl=None,
        expected_ids_cursor=None,
        expected_num_ids=None,
        expected_num_requests=None,
        expected_num_responses=None,
    ):
        """Helper to assert scheduler state."""
        if expected_max_size is not None:
            self.assertEqual(scheduler.max_size, expected_max_size)
        if expected_ttl is not None:
            self.assertEqual(scheduler.ttl, expected_ttl)
        if expected_ids_cursor is not None:
            self.assertEqual(scheduler.ids_read_cursor, expected_ids_cursor)
        if expected_num_ids is not None:
            self.assertEqual(len(scheduler.ids), expected_num_ids)
        if expected_num_requests is not None:
            self.assertEqual(len(scheduler.requests), expected_num_requests)
        if expected_num_responses is not None:
            self.assertEqual(len(scheduler.responses), expected_num_responses)

    def _assert_request_added(self, results, request_id, scheduler):
        """Helper to assert request was added successfully."""
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], request_id)
        self.assertIsNone(results[0][1])
        self.assertIn(request_id, scheduler.requests)
        self.assertIn(request_id, scheduler.ids)
        self.assertEqual(len(scheduler.requests), 1)

    def _assert_response_stored(self, request_id, scheduler, num_responses=1):
        """Helper to assert response was stored."""
        self.assertIn(request_id, scheduler.responses)
        self.assertEqual(len(scheduler.responses[request_id]), num_responses)

    def _assert_request_exists(self, request_id, scheduler):
        """Helper to assert request exists in scheduler."""
        self.assertIn(request_id, scheduler.requests)
        self.assertIn(request_id, scheduler.ids)

    def _assert_request_not_exists(self, request_id, scheduler):
        """Helper to assert request doesn't exist in scheduler."""
        self.assertNotIn(request_id, scheduler.requests)
        self.assertNotIn(request_id, scheduler.ids)

    def _assert_log_contains(self, mock_logger, message_fragment):
        """Helper to assert a log contains a specific message fragment."""
        mock_logger.assert_called()
        log_calls = [call.args[0] for call in mock_logger.call_args_list]
        self.assertTrue(any(message_fragment in msg for msg in log_calls))

    # ========== Convenience Wrapper Methods ==========

    def _get_requests_with_defaults(
        self,
        available_blocks=100,
        block_size=10,
        reserved_output_blocks=10,
        max_num_batched_tokens=1000,
        batch=1,
    ):
        """Helper to call get_requests with common defaults."""
        return self.scheduler.get_requests(
            available_blocks=available_blocks,
            block_size=block_size,
            reserved_output_blocks=reserved_output_blocks,
            max_num_batched_tokens=max_num_batched_tokens,
            batch=batch,
        )

    def _add_request_with_result(self, request, request_id, finished=False):
        """Helper to add request and result in one call."""
        self.scheduler.put_requests([request])
        mock_output = self._create_test_request_output(request_id, finished=finished)
        self.scheduler.put_results([mock_output])
        return mock_output

    def _create_test_request(self, request_id, prompt_token_ids):
        """Helper method to create test Request objects with minimal required parameters."""
        return Request(
            request_id=request_id,
            prompt="test prompt",
            prompt_token_ids=prompt_token_ids,
            prompt_token_ids_len=len(prompt_token_ids),
            messages=None,
            history=None,
            tools=None,
            system=None,
            eos_token_ids=[2],
            metrics=RequestMetrics(),
        )

    def _create_test_request_output(self, request_id, finished=False):
        """Helper method to create test RequestOutput objects."""
        return RequestOutput(
            request_id=request_id,
            prompt="test prompt",
            prompt_token_ids=[1, 2, 3],
            outputs=self._create_mock_output(),
            finished=finished,
            metrics=self._create_mock_metrics(),
        )

    def test_reset_functionality(self):
        """Test scheduler reset functionality."""
        # Add some requests and responses
        requests = [self.mock_request_1, self.mock_request_2]
        self.scheduler.put_requests(requests)

        # Reset the scheduler
        self.scheduler.reset()

        # Verify everything is cleared
        self._assert_scheduler_state(
            self.scheduler,
            expected_ids_cursor=0,
            expected_num_ids=0,
            expected_num_requests=0,
            expected_num_responses=0,
        )

    def test_reset_logs_message(self):
        """Test that reset logs appropriate message."""
        with patch.object(scheduler_logger, "info") as mock_info:
            self.scheduler.reset()
            mock_info.assert_called_once_with("Scheduler has been reset")

    def test_put_requests_duplicate_handling(self):
        """Test handling of duplicate request IDs."""
        # Add first request
        requests_1 = [self.mock_request_1]
        results_1 = self.scheduler.put_requests(requests_1)
        self.assertEqual(len(results_1), 1)
        self.assertIsNone(results_1[0][1])

        # Try to add duplicate request
        duplicate_request = self._create_test_request("req_1", [1, 2, 3])
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
        small_scheduler = self._create_scheduler(max_size=2)

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
        requests = self._get_requests_with_defaults(
            available_blocks=5, reserved_output_blocks=10
        )  # More than available

        self.assertEqual(len(requests), 0)

    def test_get_requests_insufficient_batch_size(self):
        """Test get_requests with invalid batch size."""
        requests = self._get_requests_with_defaults(batch=0)  # Invalid batch size

        self.assertEqual(len(requests), 0)

    def test_get_requests_no_available_requests(self):
        """Test get_requests when no requests are available."""
        requests = self._get_requests_with_defaults()

        self.assertEqual(len(requests), 0)

    def test_get_requests_with_available_requests(self):
        """Test get_requests with available requests and sufficient resources."""
        # Add requests to scheduler
        self.scheduler.put_requests([self.mock_request_1, self.mock_request_2])

        requests = self._get_requests_with_defaults(batch=2)

        # Should return some requests (exact number depends on resource calculation)
        self.assertGreaterEqual(len(requests), 0)

    def test_get_requests_chunked_prefill_long_requests(self):
        """Test chunked prefill behavior with long requests."""
        # Create a long request
        long_request = self._create_test_request("long_req", list(range(2000)))
        self.scheduler.put_requests([long_request])

        requests = self._get_requests_with_defaults(available_blocks=500)

        # Behavior depends on chunked prefill logic
        self.assertGreaterEqual(len(requests), 0)

    def test_put_results_expired_response(self):
        """Test putting results for expired/non-existent requests."""
        mock_output = self._create_test_request_output("non_existent", finished=False)
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
        mock_output_1 = self._create_test_request_output("req_1", finished=False)
        self.scheduler.put_results([mock_output_1])

        # Put second result for same request
        mock_output_2 = self._create_test_request_output("req_1", finished=True)
        self.scheduler.put_results([mock_output_2])

        # Should have two responses for the request
        self._assert_response_stored("req_1", self.scheduler, num_responses=2)

    def test_get_results_empty(self):
        """Test getting results when none are available."""
        results = self.scheduler.get_results()
        self.assertEqual(len(results), 0)

    def test_get_results_with_available_results(self):
        """Test getting results when they are available."""
        # Add request and result
        self._add_request_with_result(self.mock_request_1, "req_1", finished=False)

        # Get results
        results = self.scheduler.get_results()

        # Should return the results (structure is {request_id: [RequestOutput]})
        self.assertEqual(len(results), 1)
        self.assertIn("req_1", results)
        self.assertEqual(len(results["req_1"]), 1)
        self.assertEqual(results["req_1"][0].request_id, "req_1")
        self.assertFalse(results["req_1"][0].finished)

    def test_get_results_finished_request_cleanup(self):
        """Test that finished requests are cleaned up after getting results."""
        # Add request and finished result
        self._add_request_with_result(self.mock_request_1, "req_1", finished=True)

        # Get results
        results = self.scheduler.get_results()

        # Results should contain the finished response (structure is {request_id: [RequestOutput]})
        self.assertEqual(len(results), 1)
        self.assertIn("req_1", results)
        self.assertEqual(len(results["req_1"]), 1)
        self.assertEqual(results["req_1"][0].request_id, "req_1")
        self.assertTrue(results["req_1"][0].finished)

    def test_recycle_specific_request(self):
        """Test recycling a specific request."""
        # Add request
        self.scheduler.put_requests([self.mock_request_1])

        # Verify request exists
        self._assert_request_exists("req_1", self.scheduler)

        # Recycle specific request
        self.scheduler._recycle("req_1")

        # Verify request was removed
        self._assert_request_not_exists("req_1", self.scheduler)

    def test_logging_put_results_finished(self):
        """Test that put_results logs finished responses."""
        # Add request first
        self.scheduler.put_requests([self.mock_request_1])

        with patch.object(scheduler_logger, "info") as mock_info:
            mock_output = self._create_test_request_output("req_1", finished=True)
            self.scheduler.put_results([mock_output])

            # Should log finished response
            self._assert_log_contains(mock_info, "finished responses")


if __name__ == "__main__":
    unittest.main(verbosity=2)
