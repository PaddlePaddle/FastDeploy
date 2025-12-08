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

import time
import unittest
from unittest.mock import Mock, patch

import paddle

from fastdeploy.engine.request import RequestMetrics
from fastdeploy.output.token_processor import TokenProcessor

paddle.set_device("cpu")


# Mock classes and constants needed for the test
class MockConfig:
    class ParallelConfig:
        local_data_parallel_id = 0

    class SpeculativeConfig:
        method = None

    class ModelConfig:
        enable_logprob = False

    class SchedulerConfig:
        name = "default"

    class CacheConfig:
        enable_prefix_caching = False
        enable_output_caching = False
        block_size = 64

    parallel_config = ParallelConfig()
    speculative_config = SpeculativeConfig()
    model_config = ModelConfig()
    scheduler_config = SchedulerConfig()
    cache_config = CacheConfig()


class MockTask:
    def __init__(self, request_id="test_request"):
        self.request_id = request_id
        self.eos_token_ids = [2]
        self.output_token_ids = []
        self.messages = "Test prompt"
        self.num_cached_tokens = 0
        self.disaggregate_info = None
        self.prefill_chunk_info = None
        self.prefill_chunk_num = 0
        self.llm_engine_recv_req_timestamp = time.time()
        self.ic_req_data = {}
        self.prompt_token_ids_len = 0
        self.block_tables = []

        now = time.time()
        self.metrics = RequestMetrics(
            arrival_time=now,
            preprocess_start_time=now - 0.2,
            preprocess_end_time=now - 0.1,
            scheduler_recv_req_time=now + 0.1,
            inference_start_time=now + 0.2,
        )

    def get(self, key: str, default_value=None):
        if hasattr(self, key):
            return getattr(self, key)
        elif hasattr(self, "sampling_params") and hasattr(self.sampling_params, key):
            return getattr(self.sampling_params, key)
        else:
            return default_value


class MockResourceManager:
    def __init__(self):
        self.abort_req_ids_set = set()
        self.req_dict = {}
        self.tasks_list = []
        self.stop_flags = []
        self.max_num_seqs = 10

    def info(self):
        return "Mock resource manager info"

    def reschedule_preempt_task(self, task_id):
        pass

    def total_block_number(self):
        return 1000

    def available_batch(self):
        return 5

    def finish_requests_async(self, request_id):
        pass

    def _recycle_block_tables(self, task):
        pass

    def clear_data(self):
        pass


class MockCachedGeneratedTokens:
    def __init__(self):
        self.cache = []

    def put_results(self, results):
        self.cache.extend(results)


class TestTokenProcessorRecycleAbortedTask(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.cfg = MockConfig()
        self.processor = TokenProcessor.__new__(TokenProcessor)
        self.processor.cfg = self.cfg
        self.processor.cached_generated_tokens = MockCachedGeneratedTokens()
        self.processor.executor = Mock()
        self.processor.engine_worker_queue = Mock()
        self.processor.split_connector = Mock()
        self.processor.resource_manager = MockResourceManager()
        self.processor.tokens_counter = {}
        self.processor.prefill_result_status = {}
        self.processor._batch_result_buffer = None
        self.processor.use_logprobs = False
        self.processor.speculative_decoding = False

        # Mock _recycle_resources method
        self.processor._recycle_resources = Mock()

    def test_recycle_aborted_task_with_valid_aborted_request(self):
        """Test recycling when aborted task is out of batch bounds"""
        # Setup
        request_id = "test_request_1"
        batch_id = 5
        batch_size = 4  # batch_id >= batch_size - 1, so should be recycled

        task = MockTask(request_id)
        self.processor.resource_manager.abort_req_ids_set.add(request_id)
        self.processor.resource_manager.req_dict[request_id] = batch_id
        self.processor.resource_manager.tasks_list = [None] * 10
        self.processor.resource_manager.tasks_list[batch_id] = task

        # Execute
        self.processor._recycle_aborted_task(batch_size)

        # Verify
        self.assertNotIn(request_id, self.processor.resource_manager.abort_req_ids_set)
        self.processor._recycle_resources.assert_called_once_with(request_id, batch_id, task)

    def test_recycle_aborted_task_with_in_bounds_request(self):
        """Test no recycling when aborted task is within batch bounds"""
        # Setup
        request_id = "test_request_2"
        batch_id = 2
        batch_size = 5  # batch_id < batch_size - 1, so should NOT be recycled

        task = MockTask(request_id)
        self.processor.resource_manager.abort_req_ids_set.add(request_id)
        self.processor.resource_manager.req_dict[request_id] = batch_id
        self.processor.resource_manager.tasks_list = [None] * 10
        self.processor.resource_manager.tasks_list[batch_id] = task

        # Execute
        self.processor._recycle_aborted_task(batch_size)

        # Verify
        self.assertIn(request_id, self.processor.resource_manager.abort_req_ids_set)
        self.processor._recycle_resources.assert_not_called()

    def test_recycle_aborted_task_with_request_not_in_req_dict(self):
        """Test when request is in abort_req_ids_set but not in req_dict"""
        # Setup
        request_id = "test_request_3"
        batch_size = 4

        self.processor.resource_manager.abort_req_ids_set.add(request_id)
        # request_id not in req_dict

        # Execute
        self.processor._recycle_aborted_task(batch_size)

        # Verify
        self.assertNotIn(request_id, self.processor.resource_manager.abort_req_ids_set)
        self.processor._recycle_resources.assert_not_called()

    def test_recycle_aborted_task_with_empty_abort_set(self):
        """Test when no aborted requests exist"""
        # Setup
        batch_size = 4
        # abort_req_ids_set is empty by default

        # Execute
        self.processor._recycle_aborted_task(batch_size)

        # Verify
        self.processor._recycle_resources.assert_not_called()

    def test_recycle_aborted_task_with_multiple_requests(self):
        """Test recycling multiple aborted requests"""
        # Setup
        request_id_1 = "test_request_1"
        request_id_2 = "test_request_2"
        request_id_3 = "test_request_3"

        batch_id_1 = 5  # Should be recycled (out of bounds)
        batch_id_2 = 1  # Should NOT be recycled (in bounds)
        batch_id_3 = 6  # Should be recycled (out of bounds)

        batch_size = 4

        task_1 = MockTask(request_id_1)
        task_2 = MockTask(request_id_2)
        task_3 = MockTask(request_id_3)

        self.processor.resource_manager.abort_req_ids_set.update([request_id_1, request_id_2, request_id_3])
        self.processor.resource_manager.req_dict[request_id_1] = batch_id_1
        self.processor.resource_manager.req_dict[request_id_2] = batch_id_2
        self.processor.resource_manager.req_dict[request_id_3] = batch_id_3

        self.processor.resource_manager.tasks_list = [None] * 10
        self.processor.resource_manager.tasks_list[batch_id_1] = task_1
        self.processor.resource_manager.tasks_list[batch_id_2] = task_2
        self.processor.resource_manager.tasks_list[batch_id_3] = task_3

        # Execute
        self.processor._recycle_aborted_task(batch_size)

        # Verify
        self.assertNotIn(request_id_1, self.processor.resource_manager.abort_req_ids_set)
        self.assertIn(request_id_2, self.processor.resource_manager.abort_req_ids_set)
        self.assertNotIn(request_id_3, self.processor.resource_manager.abort_req_ids_set)

        # Should have called _recycle_resources twice (for request_1 and request_3)
        self.assertEqual(self.processor._recycle_resources.call_count, 2)

        # Verify the correct calls were made
        expected_calls = [((request_id_1, batch_id_1, task_1),), ((request_id_3, batch_id_3, task_3),)]
        actual_calls = self.processor._recycle_resources.call_args_list
        self.assertEqual(len(actual_calls), 2)

        # Check that the calls match (order might vary, so we check sets)
        actual_call_tuples = tuple(call[0] for call in actual_calls)
        expected_call_tuples = tuple(call[0] for call in expected_calls)

        # Convert to sets of tuples for comparison
        actual_call_sets = set(actual_call_tuples)
        expected_call_sets = set(expected_call_tuples)
        self.assertEqual(actual_call_sets, expected_call_sets)

    def test_recycle_aborted_task_boundary_condition(self):
        """Test boundary condition when batch_id equals batch_size - 1"""
        # Setup
        request_id = "test_request_boundary"
        batch_id = 3
        batch_size = 4  # batch_id == batch_size - 1, so should be recycled

        task = MockTask(request_id)
        self.processor.resource_manager.abort_req_ids_set.add(request_id)
        self.processor.resource_manager.req_dict[request_id] = batch_id
        self.processor.resource_manager.tasks_list = [None] * 10
        self.processor.resource_manager.tasks_list[batch_id] = task

        # Execute
        self.processor._recycle_aborted_task(batch_size)

        # Verify
        self.assertNotIn(request_id, self.processor.resource_manager.abort_req_ids_set)
        self.processor._recycle_resources.assert_called_once_with(request_id, batch_id, task)

    @patch("fastdeploy.output.token_processor.llm_logger")
    def test_recycle_aborted_task_logging(self, mock_logger):
        """Test that appropriate logging occurs when recycling aborted tasks"""
        # Setup
        request_id = "test_request_log"
        batch_id = 5
        batch_size = 4

        task = MockTask(request_id)
        self.processor.resource_manager.abort_req_ids_set.add(request_id)
        self.processor.resource_manager.req_dict[request_id] = batch_id
        self.processor.resource_manager.tasks_list = [None] * 10
        self.processor.resource_manager.tasks_list[batch_id] = task

        # Execute
        self.processor._recycle_aborted_task(batch_size)

        # Verify logging
        mock_logger.info.assert_called_once_with(
            f"Aborted task {request_id} idx {batch_id} is out of batch {batch_size}. Recycling."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=False)
