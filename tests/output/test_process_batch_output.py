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
from unittest.mock import Mock

import paddle

from fastdeploy.engine.request import RequestOutput
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

    parallel_config = ParallelConfig()
    speculative_config = SpeculativeConfig()
    model_config = ModelConfig()
    scheduler_config = SchedulerConfig()


class MockTask:
    def __init__(self):
        self.request_id = "test_request_1"
        self.arrival_time = time.time()
        self.inference_start_time = time.time()
        self.schedule_start_time = time.time()
        self.preprocess_end_time = time.time() - 0.1
        self.preprocess_start_time = time.time() - 0.2
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

        processor = TokenProcessor.__new__(TokenProcessor)
        processor.cfg = cfg
        processor.cached_generated_tokens: MockCachedGeneratedTokens = MockCachedGeneratedTokens()
        processor.executor = Mock()
        processor.engine_worker_queue = Mock()
        processor.split_connector = Mock()
        processor.resource_manager = MockResourceManager()
        task1 = MockTask()
        task2 = MockTask()
        processor.resource_manager.tasks_list = [task1, task2]
        processor.resource_manager.stop_flags = [False, False]
        processor.tokens_counter = {task1.request_id: 0, task2.request_id: 0}
        processor.total_step = 0
        processor.number_of_output_tokens = 0
        processor.prefill_result_status = {}
        processor.use_logprobs = use_logprobs
        processor.num_draft_tokens = 0
        processor.num_accepted_tokens = 0
        processor.num_emitted_tokens = 0
        processor.max_num_emitted_tokens = 0
        processor.num_rest_requests_per_head = [
            0,
        ] * MAX_DRAFT_TOKENS
        processor.num_accept_requests_per_head = [
            0,
        ] * MAX_DRAFT_TOKENS
        processor.speculative_stats_step = 0

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


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=False)
