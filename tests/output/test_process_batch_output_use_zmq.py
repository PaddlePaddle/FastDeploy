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

from fastdeploy.config import PREEMPTED_TOKEN_ID
from fastdeploy.engine.request import CompletionOutput, RequestMetrics, RequestOutput
from fastdeploy.output.stream_transfer_data import DecoderState, StreamTransferData
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
        self.processor.resource_manager.to_be_aborted_req_id_set = {task_id}
        self.processor.resource_manager.recycle_abort_task = MagicMock(
            side_effect=lambda rid: self.processor.resource_manager.to_be_aborted_req_id_set.discard(rid)
        )

        # Create stream data with negative token (PREEMPTED_TOKEN_ID = -9)
        stream_data = MagicMock()
        stream_data.tokens = np.array([1, 2, -9])  # Last token is PREEMPTED_TOKEN_ID
        stream_data.batch_id = 0

        # Mock _recycle_resources to track if it's called
        self.processor._recycle_resources = MagicMock()

        # Mock the llm_logger module and envs.ENABLE_V1_KVCACHE_SCHEDULER
        with (
            patch("fastdeploy.output.token_processor.llm_logger") as mock_logger,
            patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1),
        ):
            # Call the method
            result = self.processor._process_batch_output_use_zmq([stream_data])

            # Verify the recycling logic was triggered
            mock_logger.info.assert_any_call(f"start to recycle abort request_id {task_id}")
            self.processor.resource_manager.recycle_abort_task.assert_called_once_with(task_id)
            self.assertNotIn(task_id, self.processor.resource_manager.to_be_aborted_req_id_set)
            self.assertEqual(len(result), 0)  # Aborted task is skipped (continue)

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


def _make_speculative_processor():
    """Create a TokenProcessor configured for speculative decoding (ZMQ path)."""
    cfg = MagicMock()
    cfg.model_config.enable_logprob = True
    cfg.speculative_config.method = "mtp"
    cfg.speculative_config.enable_draft_logprob = True
    cfg.parallel_config.local_data_parallel_id = 0
    cfg.scheduler_config.splitwise_role = None
    cfg.scheduler_config.name = "priority"
    cfg.cache_config.enable_prefix_caching = False
    cfg.cache_config.enable_output_caching = False

    cached_generated_tokens = MagicMock()
    engine_worker_queue = MagicMock()
    split_connector = MagicMock()

    processor = TokenProcessor(cfg, cached_generated_tokens, engine_worker_queue, split_connector)
    processor.speculative_decoding = True
    processor.use_logprobs = True

    # Mock resource manager
    processor.resource_manager = MagicMock()
    processor.resource_manager.stop_flags = [False, False, False]
    processor.resource_manager.to_be_aborted_req_id_set = set()
    processor.resource_manager.to_be_rescheduled_request_id_set = set()

    # Create a proper task mock
    task = MagicMock()
    task.request_id = "spec_req_0"
    task.pooling_params = None
    task.messages = None
    task.disaggregate_info = None
    task.eos_token_ids = [2]
    task.ic_req_data = {}
    task.prompt_token_ids_len = 5
    task.num_cached_tokens = 0
    task.output_token_ids = []

    now = time.time()
    task.metrics = RequestMetrics(
        arrival_time=now,
        preprocess_start_time=now - 0.2,
        preprocess_end_time=now - 0.1,
        scheduler_recv_req_time=now + 0.1,
        inference_start_time=now + 0.2,
    )

    processor.resource_manager.tasks_list = [task]
    processor._recycle_resources = MagicMock()
    processor._record_speculative_decoding_accept_num_per_request = MagicMock()
    processor._record_first_token_metrics = MagicMock()
    processor._record_metrics = MagicMock()
    processor._record_completion_metrics = MagicMock()
    processor._compute_speculative_status = MagicMock()
    processor._record_speculative_decoding_metrics = MagicMock()

    return processor, task


def _make_stream_data(
    accept_tokens=None,
    accept_num=3,
    output_type=3,
    logprobs=None,
    prompt_logprobs=None,
    batch_id=0,
):
    """Build a StreamTransferData for speculative decoding tests."""
    if accept_tokens is None:
        accept_tokens = np.array([10, 20, 30], dtype=np.int64)
    return StreamTransferData(
        decoder_state=DecoderState.TEXT,
        batch_id=batch_id,
        tokens=accept_tokens,
        speculative_decoding=True,
        accept_tokens=accept_tokens,
        accept_num=np.array([accept_num], dtype=np.int32),
        output_type=output_type,
        logprobs=logprobs,
        prompt_logprobs=prompt_logprobs,
    )


class TestSpeculativeOutputMtype3(unittest.TestCase):
    """Tests for _process_speculative_output_use_zmq with mtype=3 (target tokens)."""

    def setUp(self):
        self.processor, self.task = _make_speculative_processor()

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0)
    def test_mtype3_normal_accept(self):
        """mtype=3 with accept_num=3 should produce a result with 3 token_ids."""
        stream_data = _make_stream_data(
            accept_tokens=np.array([10, 20, 30], dtype=np.int64),
            accept_num=3,
            output_type=3,
        )
        result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])
        self.assertIsNotNone(result)
        self.assertEqual(result.output_type, 3)
        self.assertEqual(result.outputs.token_ids, [10, 20, 30])
        self.assertFalse(result.finished)

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0)
    def test_mtype3_single_token_accept(self):
        """mtype=3 with accept_num=1 should produce a result with 1 token_id."""
        stream_data = _make_stream_data(
            accept_tokens=np.array([42], dtype=np.int64),
            accept_num=1,
            output_type=3,
        )
        result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])
        self.assertIsNotNone(result)
        self.assertEqual(result.outputs.token_ids, [42])

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0)
    def test_mtype3_with_logprobs(self):
        """mtype=3 with logprobs should populate top_logprobs on the result."""
        logprobs_mock = MagicMock()
        logprobs_mock.tolists.return_value = LogprobsLists(
            logprobs=[[0.1, 0.05], [0.3, 0.2]],
            logprob_token_ids=[[100, 101], [200, 201]],
            sampled_token_ranks=[0, 1],
        )
        stream_data = _make_stream_data(
            accept_tokens=np.array([100, 200], dtype=np.int64),
            accept_num=2,
            output_type=3,
            logprobs=logprobs_mock,
        )
        result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])
        self.assertIsNotNone(result)
        # logprob is overwritten per token in the loop, so it holds the last token's value
        self.assertAlmostEqual(result.outputs.logprob, 0.3)
        self.assertIsNotNone(result.outputs.top_logprobs)
        self.assertEqual(len(result.outputs.top_logprobs.logprobs), 2)

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0)
    def test_mtype3_logprobs_parse_failure(self):
        """mtype=3 with failing logprobs.tolists should not crash."""
        logprobs_mock = MagicMock()
        logprobs_mock.tolists.side_effect = RuntimeError("bad logprobs")
        stream_data = _make_stream_data(
            accept_tokens=np.array([100], dtype=np.int64),
            accept_num=1,
            output_type=3,
            logprobs=logprobs_mock,
        )
        with patch("fastdeploy.output.token_processor.llm_logger"):
            result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])
        self.assertIsNotNone(result)
        self.assertEqual(result.outputs.token_ids, [100])

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0)
    def test_mtype3_eos_token_finishes(self):
        """mtype=3 with an eos token should mark the result as finished."""
        stream_data = _make_stream_data(
            accept_tokens=np.array([100, 2], dtype=np.int64),  # 2 is in eos_token_ids
            accept_num=2,
            output_type=3,
        )
        result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])
        self.assertIsNotNone(result)
        self.assertTrue(result.finished)

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0)
    def test_mtype3_prefill_draft_token_ids(self):
        """mtype=3 with is_prefill and multi-token should populate draft_token_ids."""
        self.task.disaggregate_info = {"role": "prefill"}
        stream_data = _make_stream_data(
            accept_tokens=np.array([100, 200, 300], dtype=np.int64),
            accept_num=3,
            output_type=3,
        )
        # is_prefill path returns None by default (unless splitwise)
        result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])
        # With scheduler_config.name="priority", is_prefill returns None
        self.assertIsNone(result)

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0)
    def test_mtype3_prompt_logprobs(self):
        """mtype=3 with prompt_logprobs should set result.prompt_logprobs."""
        prompt_lp = np.array([0.1, 0.2])
        stream_data = _make_stream_data(
            accept_tokens=np.array([100], dtype=np.int64),
            accept_num=1,
            output_type=3,
            prompt_logprobs=prompt_lp,
        )
        result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result.prompt_logprobs, prompt_lp)


class TestSpeculativeOutputAcceptNumBoundary(unittest.TestCase):
    """Tests for boundary accept_num values (0, -3, -9) with mtype=3."""

    def setUp(self):
        self.processor, self.task = _make_speculative_processor()

    def test_accept_num_zero_returns_none(self):
        """accept_num=0 means no tokens accepted this step; should return None."""
        stream_data = _make_stream_data(
            accept_tokens=np.array([], dtype=np.int64),
            accept_num=0,
            output_type=3,
        )
        result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])
        self.assertIsNone(result)

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1)
    def test_accept_num_preempted_triggers_recycle(self):
        """accept_num=-9 (PREEMPTED_TOKEN_ID) should handle preemption and return None."""
        task_id = self.task.request_id
        self.processor.resource_manager.to_be_aborted_req_id_set = {task_id}
        self.processor.resource_manager.recycle_abort_task = MagicMock(
            side_effect=lambda rid: self.processor.resource_manager.to_be_aborted_req_id_set.discard(rid)
        )
        self.processor.resource_manager.to_be_rescheduled_request_id_set = set()

        stream_data = _make_stream_data(
            accept_tokens=np.array([], dtype=np.int64),
            accept_num=PREEMPTED_TOKEN_ID,  # -9
            output_type=3,
        )

        with patch("fastdeploy.output.token_processor.llm_logger"):
            result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])

        self.assertIsNone(result)
        self.processor.resource_manager.recycle_abort_task.assert_called_once_with(task_id)
        self.assertNotIn(task_id, self.processor.resource_manager.to_be_aborted_req_id_set)

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1)
    def test_accept_num_preempted_reschedule(self):
        """accept_num=-9 with task in to_be_rescheduled_request_id_set should reschedule."""
        task_id = self.task.request_id
        self.processor.resource_manager.to_be_aborted_req_id_set = set()
        self.processor.resource_manager.to_be_rescheduled_request_id_set = {task_id}

        stream_data = _make_stream_data(
            accept_tokens=np.array([], dtype=np.int64),
            accept_num=PREEMPTED_TOKEN_ID,
            output_type=3,
        )

        with patch("fastdeploy.output.token_processor.llm_logger"):
            result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])

        self.assertIsNone(result)
        self.processor.resource_manager.reschedule_preempt_task.assert_called_once_with(task_id)

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0)
    def test_accept_num_recovery_stop(self):
        """accept_num=-3 should trigger recovery stop: result.finished=True with error_msg."""
        stream_data = _make_stream_data(
            accept_tokens=np.array([], dtype=np.int64),
            accept_num=-3,
            output_type=3,
        )

        with patch("fastdeploy.output.token_processor.llm_logger"):
            result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])

        self.assertIsNotNone(result)
        self.assertTrue(result.finished)
        self.assertIn("Recover is not supported", result.error_msg)

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0)
    def test_accept_num_negative_other_than_boundary(self):
        """accept_num with a negative value other than -3/-9 should still be handled.
        Since accept_num_val != 0 and != -3 and != -9, it falls into the else
        branch: token_ids = stream_data.accept_tokens[:accept_num_val].tolist().
        Slicing with a negative index yields an empty list, then accept_num_val != 0,
        so the method continues but produces an empty token_ids loop."""
        stream_data = _make_stream_data(
            accept_tokens=np.array([100, 200], dtype=np.int64),
            accept_num=-1,
            output_type=3,
        )
        # accept_tokens[:-1] -> [100], then accept_num_val=-1 != 0, continues
        result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])
        # With negative accept_num_val, slicing produces [100] (not empty),
        # but accept_num_val != 0 so it doesn't early-return.
        # The token_ids list is [100] from slicing.
        self.assertIsNotNone(result)


class TestSpeculativeOutputMtype4(unittest.TestCase):
    """Tests for _process_speculative_output_use_zmq with mtype=4 (draft logprobs)."""

    def setUp(self):
        self.processor, self.task = _make_speculative_processor()

    def test_mtype4_with_logprobs(self):
        """mtype=4 with logprobs should populate draft_top_logprobs on the result."""
        logprobs_mock = MagicMock()
        logprobs_mock.tolists.return_value = LogprobsLists(
            logprobs=[[0.1, 0.05], [0.3, 0.2], [0.4, 0.15]],
            logprob_token_ids=[[10, 11], [20, 21], [30, 31]],
            sampled_token_ranks=[0, 1, 2],
        )
        stream_data = _make_stream_data(
            accept_tokens=np.array([10, 20, 30], dtype=np.int64),
            accept_num=3,
            output_type=4,
            logprobs=logprobs_mock,
        )
        result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])
        self.assertIsNotNone(result)
        self.assertEqual(result.output_type, 4)
        self.assertEqual(result.outputs.token_ids, [])  # draft path: no real tokens
        self.assertIsNotNone(result.outputs.draft_top_logprobs)
        self.assertEqual(len(result.outputs.draft_top_logprobs.logprobs), 3)
        self.assertEqual(len(result.outputs.draft_top_logprobs.logprob_token_ids), 3)
        self.assertEqual(len(result.outputs.draft_top_logprobs.sampled_token_ranks), 3)

    def test_mtype4_no_logprobs(self):
        """mtype=4 without logprobs should still return a result (no draft_top_logprobs)."""
        stream_data = _make_stream_data(
            accept_tokens=np.array([10, 20], dtype=np.int64),
            accept_num=2,
            output_type=4,
            logprobs=None,
        )
        result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])
        self.assertIsNotNone(result)
        self.assertEqual(result.output_type, 4)
        self.assertIsNone(result.outputs.draft_top_logprobs)

    def test_mtype4_zero_accept_num_returns_none(self):
        """mtype=4 with accept_num=0 should return None (no draft tokens)."""
        stream_data = _make_stream_data(
            accept_tokens=np.array([], dtype=np.int64),
            accept_num=0,
            output_type=4,
        )
        result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])
        self.assertIsNone(result)

    def test_mtype4_negative_accept_num_returns_none(self):
        """mtype=4 with negative accept_num should return None (max(neg,0)=0)."""
        stream_data = _make_stream_data(
            accept_tokens=np.array([], dtype=np.int64),
            accept_num=-3,
            output_type=4,
        )
        result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])
        self.assertIsNone(result)

    def test_mtype4_logprobs_parse_failure(self):
        """mtype=4 with failing logprobs should not crash, draft_top_logprobs stays None."""
        logprobs_mock = MagicMock()
        logprobs_mock.tolists.side_effect = RuntimeError("parse failure")
        stream_data = _make_stream_data(
            accept_tokens=np.array([10], dtype=np.int64),
            accept_num=1,
            output_type=4,
            logprobs=logprobs_mock,
        )
        with patch("fastdeploy.output.token_processor.llm_logger"):
            result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])
        self.assertIsNotNone(result)
        self.assertIsNone(result.outputs.draft_top_logprobs)

    def test_mtype4_logprobs_fewer_rows_than_accept_num(self):
        """mtype=4 with fewer logprobs rows than accept_num should only iterate available rows."""
        logprobs_mock = MagicMock()
        logprobs_mock.tolists.return_value = LogprobsLists(
            logprobs=[[0.1]],
            logprob_token_ids=[[10]],
            sampled_token_ranks=[0],
        )
        stream_data = _make_stream_data(
            accept_tokens=np.array([10, 20, 30], dtype=np.int64),
            accept_num=3,
            output_type=4,
            logprobs=logprobs_mock,
        )
        result = self.processor._process_speculative_output_use_zmq(stream_data, self.task, 0, [])
        self.assertIsNotNone(result)
        # Only 1 logprob row available, so draft_top_logprobs has 1 entry
        self.assertEqual(len(result.outputs.draft_top_logprobs.logprobs), 1)


class TestSpeculativeMtype3And4Combined(unittest.TestCase):
    """Tests for the combined mtype=3 + mtype=4 flow through _process_batch_output_use_zmq."""

    def setUp(self):
        self.processor, self.task = _make_speculative_processor()

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0)
    def test_combined_mtype3_then_mtype4(self):
        """Process mtype=3 (target) and mtype=4 (draft) stream data in sequence.
        mtype=3 should produce real tokens; mtype=4 should produce draft_top_logprobs."""
        # mtype=3: target tokens
        target_data = _make_stream_data(
            accept_tokens=np.array([100, 200], dtype=np.int64),
            accept_num=2,
            output_type=3,
        )
        # mtype=4: draft logprobs
        logprobs_mock = MagicMock()
        logprobs_mock.tolists.return_value = LogprobsLists(
            logprobs=[[0.1, 0.05], [0.3, 0.2]],
            logprob_token_ids=[[100, 101], [200, 201]],
            sampled_token_ranks=[0, 1],
        )
        draft_data = _make_stream_data(
            accept_tokens=np.array([100, 200], dtype=np.int64),
            accept_num=2,
            output_type=4,
            logprobs=logprobs_mock,
        )

        results = self.processor._process_batch_output_use_zmq([target_data, draft_data])

        self.assertEqual(len(results), 2)
        # First result: mtype=3 target
        self.assertEqual(results[0].output_type, 3)
        self.assertEqual(results[0].outputs.token_ids, [100, 200])
        # Second result: mtype=4 draft
        self.assertEqual(results[1].output_type, 4)
        self.assertIsNotNone(results[1].outputs.draft_top_logprobs)

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0)
    def test_combined_mtype3_preempted_then_mtype4(self):
        """If mtype=3 is preempted (accept_num=-9), it returns None (skipped),
        but mtype=4 draft logprobs for the same request may still arrive and
        should not be confused with target output."""
        target_data = _make_stream_data(
            accept_tokens=np.array([], dtype=np.int64),
            accept_num=PREEMPTED_TOKEN_ID,
            output_type=3,
        )
        draft_data = _make_stream_data(
            accept_tokens=np.array([10], dtype=np.int64),
            accept_num=1,
            output_type=4,
        )

        with patch("fastdeploy.output.token_processor.llm_logger"):
            results = self.processor._process_batch_output_use_zmq([target_data, draft_data])

        # target result is None (preempted), only draft remains
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].output_type, 4)

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0)
    def test_combined_mtype3_recovery_then_mtype4(self):
        """If mtype=3 has accept_num=-3 (recovery stop), it returns a finished result
        with error_msg. mtype=4 draft should still be distinguishable."""
        target_data = _make_stream_data(
            accept_tokens=np.array([], dtype=np.int64),
            accept_num=-3,
            output_type=3,
        )
        draft_data = _make_stream_data(
            accept_tokens=np.array([10], dtype=np.int64),
            accept_num=1,
            output_type=4,
        )

        with patch("fastdeploy.output.token_processor.llm_logger"):
            results = self.processor._process_batch_output_use_zmq([target_data, draft_data])

        self.assertEqual(len(results), 2)
        # target is recovery stop
        self.assertTrue(results[0].finished)
        self.assertIn("Recover is not supported", results[0].error_msg)
        # draft is separate
        self.assertEqual(results[1].output_type, 4)

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0)
    def test_mtype3_accept_zero_mtype4_normal(self):
        """mtype=3 with accept_num=0 returns None (skipped).
        mtype=4 with accept_num>0 should still produce draft logprobs result."""
        target_data = _make_stream_data(
            accept_tokens=np.array([], dtype=np.int64),
            accept_num=0,
            output_type=3,
        )
        logprobs_mock = MagicMock()
        logprobs_mock.tolists.return_value = LogprobsLists(
            logprobs=[[0.5]],
            logprob_token_ids=[[10]],
            sampled_token_ranks=[0],
        )
        draft_data = _make_stream_data(
            accept_tokens=np.array([10], dtype=np.int64),
            accept_num=1,
            output_type=4,
            logprobs=logprobs_mock,
        )

        results = self.processor._process_batch_output_use_zmq([target_data, draft_data])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].output_type, 4)

    @patch("fastdeploy.output.token_processor.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0)
    def test_mtype4_preempted_still_returns_none(self):
        """mtype=4 with accept_num=-9: max(-9, 0) = 0, so _process_draft_output_use_zmq
        returns None. This prevents draft logprobs from being emitted for a preempted
        request, avoiding draft/target mismatch."""
        draft_data = _make_stream_data(
            accept_tokens=np.array([], dtype=np.int64),
            accept_num=PREEMPTED_TOKEN_ID,
            output_type=4,
        )

        results = self.processor._process_batch_output_use_zmq([draft_data])

        self.assertEqual(len(results), 0)


class TestBuildSpeculativeStreamTransferData(unittest.TestCase):
    """Tests for _build_speculative_stream_transfer_data in pre_and_post_process.py."""

    def test_basic_target_build(self):
        """Build StreamTransferData list for mtype=3 with 2 requests."""
        import paddle

        from fastdeploy.model_executor.pre_and_post_process import (
            _build_speculative_stream_transfer_data,
        )

        accept_tokens_cpu = paddle.to_tensor([[100, 200, 0], [300, 0, 0]], dtype="int64")
        accept_num_cpu = paddle.to_tensor([2, 1], dtype="int64")

        result = _build_speculative_stream_transfer_data(
            accept_tokens_cpu=accept_tokens_cpu,
            accept_num_cpu=accept_num_cpu,
            output_type=3,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].output_type, 3)
        self.assertTrue(result[0].speculative_decoding)
        # Request 0: accept_num=2 -> tokens [100, 200]
        self.assertEqual(result[0].accept_num[0], 2)
        np.testing.assert_array_equal(result[0].accept_tokens, np.array([100, 200]))
        # Request 1: accept_num=1 -> tokens [300]
        self.assertEqual(result[1].accept_num[0], 1)
        np.testing.assert_array_equal(result[1].accept_tokens, np.array([300]))

    def test_draft_build(self):
        """Build StreamTransferData list for mtype=4 (draft)."""
        import paddle

        from fastdeploy.model_executor.pre_and_post_process import (
            _build_speculative_stream_transfer_data,
        )

        accept_tokens_cpu = paddle.to_tensor([[10, 20, 30]], dtype="int64")
        accept_num_cpu = paddle.to_tensor([3], dtype="int64")

        result = _build_speculative_stream_transfer_data(
            accept_tokens_cpu=accept_tokens_cpu,
            accept_num_cpu=accept_num_cpu,
            output_type=4,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].output_type, 4)

    def test_preempted_request_injected(self):
        """If last_preempted_idx is nonzero for a slot, accept_num should be set to PREEMPTED_TOKEN_ID."""
        import paddle

        from fastdeploy.model_executor.pre_and_post_process import (
            _build_speculative_stream_transfer_data,
        )

        accept_tokens_cpu = paddle.to_tensor([[100, 200, 0], [300, 400, 500]], dtype="int64")
        accept_num_cpu = paddle.to_tensor([2, 3], dtype="int64")
        # Request 1 is preempted
        last_preempted_idx = paddle.to_tensor([0, 99], dtype="int64")

        result = _build_speculative_stream_transfer_data(
            accept_tokens_cpu=accept_tokens_cpu,
            accept_num_cpu=accept_num_cpu,
            output_type=3,
            last_preempted_idx=last_preempted_idx,
        )

        # Request 0: not preempted, accept_num=2
        self.assertEqual(result[0].accept_num[0], 2)
        # Request 1: preempted, accept_num overwritten to PREEMPTED_TOKEN_ID (-9)
        self.assertEqual(result[1].accept_num[0], PREEMPTED_TOKEN_ID)

    def test_zero_accept_num_empty_tokens(self):
        """accept_num=0 should produce empty tokens array."""
        import paddle

        from fastdeploy.model_executor.pre_and_post_process import (
            _build_speculative_stream_transfer_data,
        )

        accept_tokens_cpu = paddle.to_tensor([[100, 200, 0]], dtype="int64")
        accept_num_cpu = paddle.to_tensor([0], dtype="int64")

        result = _build_speculative_stream_transfer_data(
            accept_tokens_cpu=accept_tokens_cpu,
            accept_num_cpu=accept_num_cpu,
            output_type=3,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].accept_num[0], 0)
        self.assertEqual(len(result[0].accept_tokens), 0)

    def test_prompt_logprobs_assigned(self):
        """prompt_logprobs_list should be assigned to the corresponding stream_data."""
        import paddle

        from fastdeploy.model_executor.pre_and_post_process import (
            _build_speculative_stream_transfer_data,
        )

        accept_tokens_cpu = paddle.to_tensor([[100, 200]], dtype="int64")
        accept_num_cpu = paddle.to_tensor([2], dtype="int64")
        prompt_logprobs_list = [np.array([0.1, 0.2])]

        result = _build_speculative_stream_transfer_data(
            accept_tokens_cpu=accept_tokens_cpu,
            accept_num_cpu=accept_num_cpu,
            prompt_logprobs_list=prompt_logprobs_list,
            output_type=3,
        )

        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0].prompt_logprobs, prompt_logprobs_list[0])


if __name__ == "__main__":
    unittest.main()
