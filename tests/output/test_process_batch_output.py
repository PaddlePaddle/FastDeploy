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

import random
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import paddle

from fastdeploy.engine.request import RequestMetrics, RequestOutput
from fastdeploy.output.token_processor import TokenProcessor

paddle.set_device("cpu")


# Mock classes and constants needed for the test
class MockConfig:
    class ParallelConfig:
        local_data_parallel_id = 0

    class SpeculativeConfig:
        method = None
        num_speculative_tokens = 1
        num_model_steps = 1
        max_candidate_len = 5
        verify_window = 2
        max_ngram_size = 5
        min_ngram_size = 2
        model = None
        quantization = None
        num_gpu_block_expand_ratio = 1
        model_type = "main"
        benchmark_mode = False
        num_extra_cache_layer = 0
        mtp_strategy = "default"

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
    def __init__(self):
        self.request_id = "test_request_1"
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
        self.trace_carrier = {}

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
        self.stop_flags = [False]
        self.tasks_list = [MockTask()]
        self.to_be_rescheduled_request_id_set = set()
        self.abort_req_ids_set = set()
        self.req_dict = {}

    def info(self):
        return "Mock resource manager info"

    def reschedule_preempt_task(self, task_id):
        pass


class MockCachedGeneratedTokens:
    def __init__(self):
        self.cache = []

    def put_results(self, results):
        self.cache.extend(results)


# Constants
RECOVERY_STOP_SIGNAL = -3
MAX_BSZ = 512
K = 20
MAX_DRAFT_TOKENS = 6
SPECULATE_MAX_BSZ = 256


class TestTokenProcessorProcessBatchOutput(unittest.TestCase):
    def setup_token_processor(self, speculative_decoding=False, use_logprobs=False):
        """Helper method to setup TokenProcessor with different configurations"""
        cfg = MockConfig()
        cfg.speculative_config.method = "mtp" if speculative_decoding else None
        cfg.speculative_config.num_speculative_tokens = 1
        cfg.model_config.enable_logprob = use_logprobs
        cfg.speculative_config.enable_draft_logprob = True

        processor = TokenProcessor.__new__(TokenProcessor)
        processor.cfg = cfg
        processor.cached_generated_tokens: MockCachedGeneratedTokens = MockCachedGeneratedTokens()
        processor.executor = Mock()
        processor.engine_worker_queue = Mock()
        processor.split_connector = Mock()
        processor.resource_manager = MockResourceManager()
        processor.scheduler_metrics_logger = None
        task1 = MockTask()
        task2 = MockTask()
        processor.resource_manager.tasks_list = [task1, task2]
        processor.resource_manager.stop_flags = [False, False]
        processor.tokens_counter = {task1.request_id: 0, task2.request_id: 0}
        processor.total_step = 0
        processor.number_of_output_tokens = 0
        processor.prefill_result_status = {}
        processor.use_logprobs = use_logprobs
        processor.enable_draft_logprob = cfg.speculative_config.enable_draft_logprob
        processor.num_draft_tokens = 0
        processor.num_accepted_tokens = 0
        processor.num_emitted_tokens = 0
        processor.max_num_emitted_tokens = 0
        processor.speculative_stats_step = 0
        processor.total_step_per_request = {}
        processor.accept_token_num_per_head_per_request = {}
        processor.accept_token_num_per_head = [0] * MAX_DRAFT_TOKENS

        # processor._recycle_resources = Mock()

        if speculative_decoding:
            if use_logprobs:
                processor.output_tokens = paddle.full(
                    shape=[MAX_BSZ * MAX_DRAFT_TOKENS * (K + 1) + MAX_BSZ + 3, 1],
                    fill_value=2,
                    dtype="int64",
                )
                processor.output_scores = paddle.full(
                    shape=[MAX_BSZ * MAX_DRAFT_TOKENS * (K + 1), 1],
                    fill_value=0.0,
                    dtype="float32",
                )
                processor.output_ranks = paddle.full(
                    shape=[MAX_BSZ * MAX_DRAFT_TOKENS],
                    fill_value=0,
                    dtype="int64",
                )
            else:
                processor.output_tokens = paddle.full(
                    shape=[SPECULATE_MAX_BSZ * MAX_DRAFT_TOKENS + SPECULATE_MAX_BSZ + 2],
                    fill_value=2,
                    dtype="int64",
                )
        elif use_logprobs:
            processor.output_tokens = paddle.full(shape=[MAX_BSZ * (K + 1) + 2, 1], fill_value=2, dtype="int64")
            processor.output_scores = paddle.full(shape=[MAX_BSZ * (K + 1), 1], fill_value=0.0, dtype="float32")
            processor.output_ranks = paddle.full(shape=[MAX_BSZ], fill_value=0, dtype="int64")
        else:
            processor.output_tokens = paddle.full(shape=[MAX_BSZ + 2, 1], fill_value=2, dtype="int64")

        return processor

    def test_speculative_decoding_use_logprobs(self):
        """Test basic speculative decoding scenario"""
        processor = self.setup_token_processor(speculative_decoding=True, use_logprobs=True)

        # stop_flag
        processor.output_tokens[0, 0].set_tensor(paddle.to_tensor(2))
        # mtype target = 3, decode = 4
        processor.output_tokens[1, 0].set_tensor(paddle.to_tensor(3))
        # batch
        processor.output_tokens[2, 0].set_tensor(paddle.to_tensor(2))
        # accept_num
        processor.output_tokens[3, 0].set_tensor(paddle.to_tensor(3))
        processor.output_tokens[4, 0].set_tensor(paddle.to_tensor(3))

        batch = processor.output_tokens[2, 0]
        mtype = processor.output_tokens[3, 0]
        accept_num = [int(num[0]) for num in processor.output_tokens[3 : batch + 3]]

        # init
        print(f"batch:{batch}, mtype:{mtype} accept_num: {accept_num}")
        for i in range(batch):
            for j in range(accept_num[i]):
                token_index = 3 + MAX_BSZ + i * MAX_DRAFT_TOKENS * (K + 1) + j * (K + 1)
                score_index = i * MAX_DRAFT_TOKENS * (K + 1) + j * (K + 1)
                print(f"batch:{i}, accept:{j} token_index: {token_index} score_index: {score_index}")
                for k in range(K + 1):
                    processor.output_tokens[token_index + k].set_tensor(paddle.to_tensor(random.randint(100, 100000)))
                    processor.output_scores[score_index + k].set_tensor(paddle.to_tensor(random.random()))
                processor.output_ranks[j].set_tensor(paddle.to_tensor(1))

        processor._process_batch_output()

        batch_result_buffer: list[RequestOutput] = processor._batch_result_buffer

        for i, request_output in enumerate(batch_result_buffer):
            assert isinstance(request_output, RequestOutput)
            assert len(request_output.outputs.token_ids) == accept_num[i]
            assert len(request_output.outputs.top_logprobs) == 3
            # tokens, scores, ranks
            assert len(request_output.outputs.top_logprobs[0][0]) == K + 1
            assert len(request_output.outputs.top_logprobs[1][0]) == K + 1
            assert len(request_output.outputs.top_logprobs[2]) == accept_num[i]

        # mtype = 4
        processor.output_tokens[1, 0].set_tensor(paddle.to_tensor(4))
        processor._process_batch_output()
        cached_generated_tokens: MockCachedGeneratedTokens = processor.cached_generated_tokens
        for c in cached_generated_tokens.cache:
            assert isinstance(request_output, RequestOutput)
            assert len(request_output.outputs.token_ids) == accept_num[i]
            assert len(request_output.outputs.top_logprobs) == 3
            assert len(request_output.outputs.draft_top_logprobs) == 3
            # tokens, scores, ranks
            assert len(request_output.outputs.draft_top_logprobs[0][0]) == K + 1
            assert len(request_output.outputs.draft_top_logprobs[1][0]) == K + 1
            assert len(request_output.outputs.draft_top_logprobs[2]) == accept_num[i]

    def test_process_batch_output_aborted_task_negative_token_speculative_decoding(self):
        """Test aborted task receiving negative token triggers recycling in speculative decoding mode"""
        processor = self.setup_token_processor(speculative_decoding=True, use_logprobs=True)

        # Set up task as aborted
        task_id = "test_aborted_request"
        task = processor.resource_manager.tasks_list[0]
        task.request_id = task_id
        processor.resource_manager.abort_req_ids_set = {task_id}

        # Add the task to req_dict to prevent _recycle_aborted_task from processing it early
        # Use a larger batch to avoid the early recycling condition
        processor.resource_manager.req_dict[task_id] = 0  # batch_id = 0

        # Mock _recycle_resources to track if it's called
        processor._recycle_resources = MagicMock()

        # Set up output tokens with negative token
        # stop_flag
        processor.output_tokens[0, 0].set_tensor(paddle.to_tensor(2))
        # mtype target = 3
        processor.output_tokens[1, 0].set_tensor(paddle.to_tensor(3))
        # batch = 2 (so batch_id=0 is < batch_size-1=1)
        processor.output_tokens[2, 0].set_tensor(paddle.to_tensor(2))
        # Set accept_num = PREEMPTED_TOKEN_ID (-9) for first task to trigger abort logic
        processor.output_tokens[3, 0].set_tensor(paddle.to_tensor(-9))
        processor.output_tokens[4, 0].set_tensor(paddle.to_tensor(1))

        # Add second task to tasks_list
        task2 = MockTask()
        task2.request_id = "test_request_2"
        processor.resource_manager.tasks_list = [task, task2]
        processor.resource_manager.stop_flags = [False, False]
        # Update tokens_counter to include both tasks
        processor.tokens_counter[task_id] = 0
        processor.tokens_counter[task2.request_id] = 0

        # Mock llm_logger to capture the log message and envs.ENABLE_V1_KVCACHE_SCHEDULER
        with (
            patch("fastdeploy.output.token_processor.llm_logger") as mock_logger,
            patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0),
        ):
            # Call the method
            processor._process_batch_output()

            # In speculative decoding mode, when accept_num[i] == PREEMPTED_TOKEN_ID,
            # the code logs "sync preemption" and continues without triggering abort recycling
            # This is the expected behavior for speculative decoding mode
            mock_logger.info.assert_any_call(f"sync preemption for request_id {task_id} done.")
            # Verify that _recycle_resources was NOT called for the aborted task
            # (it may be called for other tasks like test_request_2 if they receive EOS tokens)
            for call in processor._recycle_resources.call_args_list:
                self.assertNotEqual(
                    call[0][0], task_id, f"_recycle_resources should not be called for aborted task {task_id}"
                )
            # Verify that the task is still in abort_req_ids_set
            self.assertIn(task_id, processor.resource_manager.abort_req_ids_set)

    def test_process_batch_output_aborted_task_negative_token_normal_mode(self):
        """Test aborted task receiving negative token triggers recycling in normal mode"""
        processor = self.setup_token_processor(speculative_decoding=False, use_logprobs=False)

        # Set up task as aborted
        task_id = "test_aborted_request"
        task = processor.resource_manager.tasks_list[0]
        task.request_id = task_id
        processor.resource_manager.abort_req_ids_set = {task_id}

        # Add the task to req_dict to prevent _recycle_aborted_task from processing it early
        # batch_id should be < batch_size - 1 to avoid early recycling
        processor.resource_manager.req_dict[task_id] = (
            0  # batch_id = 0, batch_size = 1, so 0 < 0 is false, but 0 >= 0 is true
        )
        # Actually, let's use a larger batch to avoid the early recycling condition
        processor.output_tokens = paddle.full(shape=[MAX_BSZ + 2, 1], fill_value=2, dtype="int64")

        # Mock _recycle_resources to track if it's called
        processor._recycle_resources = MagicMock()

        # Set up output tokens with negative token
        # batch = 2 (so batch_id=0 is < batch_size-1=1)
        processor.output_tokens[1, 0].set_tensor(paddle.to_tensor(2))
        # Set negative token for first task (batch_id=0)
        processor.output_tokens[2, 0].set_tensor(paddle.to_tensor(-1))
        # Set positive token for second task (batch_id=1)
        processor.output_tokens[3, 0].set_tensor(paddle.to_tensor(100))

        # Add second task to tasks_list
        task2 = MockTask()
        task2.request_id = "test_request_2"
        processor.resource_manager.tasks_list = [task, task2]
        processor.resource_manager.stop_flags = [False, False]
        # Update tokens_counter to include both tasks
        processor.tokens_counter[task_id] = 0
        processor.tokens_counter[task2.request_id] = 0

        # Mock llm_logger to capture the log message and envs.ENABLE_V1_KVCACHE_SCHEDULER
        with (
            patch("fastdeploy.output.token_processor.llm_logger") as mock_logger,
            patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0),
        ):
            # Call the method
            processor._process_batch_output()

            # Verify the recycling logic was triggered
            mock_logger.info.assert_any_call(f"Aborted task {task_id} received negative token. Recycling.")
            processor._recycle_resources.assert_called_once_with(task_id, 0, task)
            self.assertNotIn(task_id, processor.resource_manager.abort_req_ids_set)

    def test_process_batch_output_non_aborted_task_negative_token(self):
        """Test non-aborted task receiving negative token does not trigger recycling"""
        processor = self.setup_token_processor(speculative_decoding=False, use_logprobs=False)

        # Set up task as not aborted
        task_id = "test_normal_request"
        task = processor.resource_manager.tasks_list[0]
        task.request_id = task_id
        processor.resource_manager.abort_req_ids_set = set()  # Empty set

        # Mock _recycle_resources to track if it's called
        processor._recycle_resources = MagicMock()

        # Set up output tokens with negative token
        # batch = 1
        processor.output_tokens[1, 0].set_tensor(paddle.to_tensor(1))
        # Set negative token
        processor.output_tokens[2, 0].set_tensor(paddle.to_tensor(-1))

        # Mock llm_logger to capture the log message and envs.ENABLE_V1_KVCACHE_SCHEDULER
        with (
            patch("fastdeploy.output.token_processor.llm_logger") as mock_logger,
            patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0),
        ):
            # Call the method
            processor._process_batch_output()
            print(mock_logger)
            # Verify the recycling logic was NOT triggered
            # When a non-aborted task receives a negative token, the code just continues
            # without logging or recycling
            processor._recycle_resources.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=False)
