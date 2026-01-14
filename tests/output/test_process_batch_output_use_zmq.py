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
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.engine.request import CompletionOutput, RequestMetrics, RequestOutput
from fastdeploy.output.token_processor import TokenProcessor
from fastdeploy.worker.output import LogprobsLists


class TestTokenProcessorLogprobs(unittest.TestCase):
    def setUp(self):
        self.cfg = MagicMock()
        self.cfg.model_config.enable_logprob = True
        self.cfg.speculative_config.method = None
        self.cfg.parallel_config.local_data_parallel_id = 0
        self.cached_generated_tokens = MagicMock()
        self.engine_worker_queue = MagicMock()
        self.split_connector = MagicMock()

        self.processor = TokenProcessor(
            self.cfg, self.cached_generated_tokens, self.engine_worker_queue, self.split_connector
        )

        # Mock resource manager
        self.processor.resource_manager = MagicMock()
        self.processor.resource_manager.stop_flags = [False]

        # Create a proper task mock with time attributes
        self.task_mock = MagicMock()
        self.task_mock.request_id = "test_request"
        self.task_mock.pooling_params = None
        self.task_mock.messages = None
        self.task_mock.disaggregate_info = None
        self.task_mock.eos_token_ids = [2]
        self.task_mock.ic_req_data = {}
        self.task_mock.prompt_token_ids_len = 0

        now = time.time()
        self.task_mock.metrics = RequestMetrics(
            arrival_time=now,
            preprocess_start_time=now - 0.2,
            preprocess_end_time=now - 0.1,
            scheduler_recv_req_time=now + 0.1,
            inference_start_time=now + 0.2,
        )

        self.processor.resource_manager.tasks_list = [self.task_mock]

        # Mock logger
        self.processor.llm_logger = MagicMock()

        # Mock metrics to avoid prometheus dependency issues
        self.processor.main_process_metrics = MagicMock()
        self.processor._recycle_resources = MagicMock()

        # Mock the _process_per_token method to avoid prometheus issues
        self.processor._process_per_token = MagicMock()
        self.processor._process_per_token.return_value = RequestOutput(
            request_id="test_request",
            outputs=CompletionOutput(
                index=0,
                send_idx=0,
                token_ids=[],
                draft_token_ids=[],
            ),
            finished=False,
            metrics=MagicMock(),
        )

    def test_process_logprobs_success(self):
        """Test successful logprobs parsing"""
        stream_data = MagicMock()
        logprobs = MagicMock()
        logprobs.tolists.return_value = LogprobsLists(
            logprobs=[[0.5]], logprob_token_ids=[[1]], sampled_token_ranks=[0]
        )
        stream_data.logprobs = logprobs
        stream_data.tokens = np.array([1])
        stream_data.batch_id = 0

        result = self.processor._process_batch_output_use_zmq([stream_data])

        self.assertEqual(len(result), 1)
        self.processor.llm_logger.warning.assert_not_called()

    def test_process_logprobs_failure(self):
        """Test failed logprobs parsing"""
        stream_data = MagicMock()
        stream_data.logprobs = MagicMock()
        stream_data.logprobs.tolists.side_effect = Exception("Test error")
        stream_data.tokens = np.array([1])
        stream_data.batch_id = 0

        with patch.object(self.processor.llm_logger, "warning"):
            result = self.processor._process_batch_output_use_zmq([stream_data])

            self.assertEqual(len(result), 1)
            self.assertIsNone(result[0].outputs.logprob)

    def test_process_prompt_logprobs_success(self):
        """Test successful prompt_logprobs parsing"""
        stream_data = MagicMock()
        stream_data.logprobs = None
        stream_data.prompt_logprobs = np.array([0.1, 0.2])
        stream_data.tokens = np.array([1])
        stream_data.batch_id = 0

        result = self.processor._process_batch_output_use_zmq([stream_data])

        self.assertEqual(len(result), 1)
        self.processor.llm_logger.warning.assert_not_called()

    def test_process_prompt_logprobs_failure(self):
        """Test failed prompt_logprobs parsing"""
        stream_data = MagicMock()
        stream_data.logprobs = None
        stream_data.prompt_logprobs = MagicMock()
        stream_data.prompt_logprobs.tolist.side_effect = AttributeError("'NoneType' object has no attribute 'tolist'")
        stream_data.tokens = np.array([1])
        stream_data.batch_id = 0

        with patch.object(self.processor.llm_logger, "warning"):
            result = self.processor._process_batch_output_use_zmq([stream_data])

            self.assertEqual(len(result), 1)
            self.assertIsNone(getattr(result[0], "prompt_logprobs", None))

    def test_process_batch_with_stop_flag(self):
        """Test processing when stop flag is True"""
        self.processor.resource_manager.stop_flags = [True]
        stream_data = MagicMock()
        stream_data.batch_id = 0

        result = self.processor._process_batch_output_use_zmq([stream_data])

        self.assertEqual(len(result), 0)

    def test_process_batch_output_use_zmq_aborted_task_negative_token(self):
        """Test aborted task receiving negative token triggers recycling logic"""
        # Set up task as aborted
        task_id = "test_aborted_request"
        self.task_mock.request_id = task_id
        self.processor.resource_manager.abort_req_ids_set = {task_id}

        # Create stream data with negative token
        stream_data = MagicMock()
        stream_data.tokens = np.array([1, 2, -1])  # Last token is negative
        stream_data.batch_id = 0

        # Mock _recycle_resources to track if it's called
        self.processor._recycle_resources = MagicMock()

        # Mock the llm_logger module and envs.ENABLE_V1_KVCACHE_SCHEDULER
        with (
            patch("fastdeploy.output.token_processor.llm_logger") as mock_logger,
            patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0),
        ):
            # Call the method
            result = self.processor._process_batch_output_use_zmq([stream_data])

            # Verify the recycling logic was triggered
            mock_logger.info.assert_any_call(f"Aborted task {task_id} received negative token. Recycling.")
            self.processor._recycle_resources.assert_called_once_with(task_id, 0, self.task_mock)
            self.assertNotIn(task_id, self.processor.resource_manager.abort_req_ids_set)
            self.assertEqual(len(result), 1)  # Should return abort result
            self.assertEqual(result[0].finished, True)
            self.assertEqual(result[0].error_code, 499)
            self.assertIn("aborted", result[0].error_msg.lower())

    def test_process_batch_output_use_zmq_non_aborted_task_negative_token(self):
        """Test non-aborted task receiving negative token does not trigger recycling"""
        # Set up task as not aborted
        task_id = "test_normal_request"
        self.task_mock.request_id = task_id
        self.processor.resource_manager.abort_req_ids_set = set()  # Empty set

        # Create stream data with negative token
        stream_data = MagicMock()
        stream_data.tokens = np.array([1, 2, -1])  # Last token is negative
        stream_data.batch_id = 0

        # Mock _recycle_resources to track if it's called
        self.processor._recycle_resources = MagicMock()

        # Call the method
        self.processor._process_batch_output_use_zmq([stream_data])

        # Verify recycling logic was NOT triggered
        self.processor._recycle_resources.assert_not_called()
        self.processor.llm_logger.info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
