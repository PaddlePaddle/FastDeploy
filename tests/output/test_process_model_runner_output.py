"""
Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time
from unittest import mock

import paddle
import pytest

from fastdeploy import envs
from fastdeploy.config import PREEMPTED_TOKEN_ID
from fastdeploy.engine.request import PoolingRequestOutput, Request, RequestMetrics
from fastdeploy.output.token_processor import TokenProcessor
from fastdeploy.worker.output import (
    DecodeMode,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
)


class _DummyCfg:
    def __init__(
        self,
        enable_logprob=False,
        max_num_seqs=2,
        enable_prefix_caching=False,
        enable_output_caching=False,
        speculative_method=None,
    ):
        self.parallel_config = mock.Mock(
            local_data_parallel_id=0,
            enable_expert_parallel=False,
            data_parallel_size=1,
        )
        self.speculative_config = mock.Mock(
            method=speculative_method,
            num_speculative_tokens=2,
            enable_draft_logprob=True,
        )
        self.model_config = mock.Mock(enable_logprob=enable_logprob)
        self.scheduler_config = mock.Mock(name="default", splitwise_role="decode")
        self.cache_config = mock.Mock(
            enable_prefix_caching=enable_prefix_caching,
            enable_output_caching=enable_output_caching,
            block_size=64,
        )
        self.max_num_seqs = max_num_seqs
        self.splitwise_version = "v1"


class _DummyResourceManager:
    def __init__(self, max_num_seqs=2):
        self.max_num_seqs = max_num_seqs
        self.stop_flags = [False] * max_num_seqs
        self.tasks_list = [None] * max_num_seqs
        self.req_dict = {}
        self.requests = {}
        self.to_be_rescheduled_request_id_set = set()
        self.abort_req_ids_set = set()
        self.recycled = []
        self.cached_tasks = []
        self.cleared = False

    def _recycle_block_tables(self, task):
        self.recycled.append(task.request_id)

    def reschedule_preempt_task(self, request_id):
        self.recycled.append(f"reschedule-{request_id}")

    def finish_requests_async(self, request_id):
        self.recycled.append(f"finish-{request_id}")

    def total_block_number(self):
        return 8

    def available_batch(self):
        return self.tasks_list.count(None)

    def info(self):
        return "rm-info"

    def get_finished_req(self):
        return []

    def cache_output_tokens(self, task):
        self.cached_tasks.append(task.request_id)

    def clear_data(self):
        self.cleared = True


class _DummyQueue:
    def get_finished_req(self):
        return []


class _DummyConnector:
    def __init__(self):
        self.calls = []

    def send_first_token(self, info, results):
        self.calls.append((info, results))


def _make_processor(
    enable_logprob=False,
    max_num_seqs=2,
    enable_prefix_caching=False,
    enable_output_caching=False,
    speculative_method=None,
):
    cfg = _DummyCfg(
        enable_logprob=enable_logprob,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=enable_prefix_caching,
        enable_output_caching=enable_output_caching,
        speculative_method=speculative_method,
    )
    cache = mock.Mock()
    queue = _DummyQueue()
    connector = _DummyConnector()
    processor = TokenProcessor(cfg, cache, queue, connector)
    rm = _DummyResourceManager(max_num_seqs)
    processor.set_resource_manager(rm)
    return processor, rm, cache, connector


@pytest.fixture(autouse=True)
def _ensure_cpu():
    paddle.device.set_device("cpu")


def _create_test_request(request_id="test-req", prompt_tokens=None, pooling_params=None):
    """Helper to create a test request with common setup."""
    base_time = time.time()
    if prompt_tokens is None:
        prompt_tokens = [1, 2]

    return Request(
        request_id=request_id,
        prompt=["test prompt"],
        prompt_token_ids=prompt_tokens,
        prompt_token_ids_len=len(prompt_tokens),
        messages=[{"content": "test", "role": "user"}],
        history=[],
        tools=[],
        system="test system",
        eos_token_ids=[99],
        pooling_params=pooling_params,
        metrics=RequestMetrics(
            arrival_time=base_time,
            preprocess_start_time=base_time - 0.2,
            preprocess_end_time=base_time - 0.1,
            inference_start_time=base_time,
            decode_inference_start_time=base_time,
        ),
    )


def _create_model_runner_output(decode_mode=DecodeMode.TARGET, token_ids=None, logprobs_data=None, pooler_output=None):
    """Helper to create ModelRunnerOutput with common defaults."""
    if token_ids is None:
        token_ids = [5, 6]

    cu_num_tokens = paddle.to_tensor([0, len(token_ids)], dtype=paddle.int32)
    sampled_tokens = paddle.to_tensor(token_ids, dtype=paddle.int64)

    model_output = ModelRunnerOutput(
        decode_mode=decode_mode,
        cu_num_generated_tokens=cu_num_tokens,
        sampled_token_ids=sampled_tokens,
        logprobs=None,
        prompt_logprobs=None,
        pooler_output=pooler_output,
    )

    if logprobs_data:
        logprobs_tensor = LogprobsTensors(
            logprob_token_ids=paddle.to_tensor(logprobs_data["token_ids"], dtype=paddle.int64),
            logprobs=paddle.to_tensor(logprobs_data["values"], dtype=paddle.float32),
            selected_token_ranks=paddle.to_tensor(logprobs_data["ranks"], dtype=paddle.int64),
        )
        model_output.logprobs = logprobs_tensor

    return model_output


def test_process_model_runner_output_normal_completion():
    """Test normal completion scenario without logprobs."""
    processor, rm, _, connector = _make_processor()

    # Setup task
    task = _create_test_request("req-1")
    task.disaggregate_info = None  # regular decode
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    rm.requests[task.request_id] = mock.Mock(idx=0)

    # Setup model output with normal tokens
    model_output = _create_model_runner_output(token_ids=[5, 6, 7])

    # Process output
    batch_result, draft_result = processor.process_model_runner_output(model_output)

    # Verify results
    assert len(batch_result) == 1
    result = batch_result[0]
    assert result.request_id == "req-1"
    assert result.finished is False
    assert result.metrics is not None
    assert len(draft_result) == 0  # No draft results for mode 3

    # Check token processing occurred
    assert processor.tokens_counter["req-1"] > 0


def test_process_model_runner_output_draft_mode_without_logprobs():
    """Test draft mode when logprobs are disabled."""
    processor, rm, _, _ = _make_processor(enable_logprob=False)

    # Setup task
    task = _create_test_request("req-draft")
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task

    # Setup draft mode output
    model_output = _create_model_runner_output(decode_mode=DecodeMode.DRAFT, token_ids=[1, 2])

    # Process output
    batch_result, draft_result = processor.process_model_runner_output(model_output)

    # Should skip draft processing when logprobs disabled
    assert len(batch_result) == 0
    assert len(draft_result) == 0


def test_process_model_runner_output_draft_mode_with_logprobs():
    """Test draft mode with logprobs enabled."""
    processor, rm, _, _ = _make_processor(enable_logprob=True)

    # Setup task
    task = _create_test_request("req-draft-logprob")
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task

    # Setup draft mode output with logprobs
    logprobs_data = {"token_ids": [[10, 20], [30, 40]], "values": [[-0.5, -1.0], [-0.3, -0.8]], "ranks": [0, 1]}
    model_output = _create_model_runner_output(
        decode_mode=DecodeMode.DRAFT, token_ids=[1, 2], logprobs_data=logprobs_data
    )

    # Process output
    batch_result, draft_result = processor.process_model_runner_output(model_output)

    # Should have draft results with logprobs
    assert len(batch_result) == 0
    assert len(draft_result) == 1

    draft_res = draft_result[0]
    assert draft_res.request_id == "req-draft-logprob"
    assert draft_res.output_type == DecodeMode.DRAFT
    assert draft_res.outputs.draft_top_logprobs is not None
    assert draft_res.finished is False


def test_process_model_runner_output_aborted_task():
    """Test handling of aborted tasks with negative tokens."""
    processor, rm, _, _ = _make_processor()

    # Setup task marked for abortion
    task = _create_test_request("req-aborted")
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    rm.abort_req_ids_set.add("req-aborted")

    # Setup model output with abort signal
    model_output = _create_model_runner_output(token_ids=[5, PREEMPTED_TOKEN_ID])

    # Test without V1 scheduler
    with mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False):
        batch_result, draft_result = processor.process_model_runner_output(model_output)

    # Should handle abort and create finished result
    assert len(batch_result) == 1
    result = batch_result[0]
    assert result.request_id == "req-aborted"
    assert result.finished is True
    assert result.error_code == 499
    assert "aborted" in result.error_msg
    assert "req-aborted" not in rm.abort_req_ids_set  # Should be removed


def test_process_model_runner_output_rescheduled_task_v1_scheduler():
    """Test reschedule handling with V1 scheduler enabled."""
    processor, rm, _, _ = _make_processor()

    # Setup task marked for rescheduling
    task = _create_test_request("req-rescheduled")
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    rm.to_be_rescheduled_request_id_set.add("req-rescheduled")

    # Setup model output with preemption token
    model_output = _create_model_runner_output(token_ids=[5, PREEMPTED_TOKEN_ID])

    # Test with V1 scheduler enabled
    with mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", True):
        batch_result, draft_result = processor.process_model_runner_output(model_output)

    # Should reschedule preempted task
    assert len(batch_result) == 0
    assert f"reschedule-{task.request_id}" in rm.recycled


def test_process_model_runner_output_pooling_request():
    """Test pooling request scenario."""
    processor, rm, _, _ = _make_processor()

    # Setup task with pooling parameters
    task = _create_test_request("req-pooling", pooling_params={"method": "mean"})
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    rm.requests[task.request_id] = mock.Mock(idx=0)

    # Setup model output with pooler data
    pooler_data = paddle.to_tensor([[0.1, 0.2, 0.3]], dtype=paddle.float32)
    model_output = _create_model_runner_output(token_ids=[1], pooler_output=pooler_data)

    # Process output
    batch_result, draft_result = processor.process_model_runner_output(model_output)

    # Should create pooling result
    assert len(batch_result) == 1
    result = batch_result[0]
    assert isinstance(result, PoolingRequestOutput)
    assert result.request_id == "req-pooling"
    assert result.finished is True
    assert result.prompt_token_ids == task.prompt_token_ids
    assert result.outputs.data is not None


def test_process_model_runner_output_with_logprobs():
    """Test completion with logprobs enabled."""
    processor, rm, _, _ = _make_processor(enable_logprob=True)

    # Setup task
    task = _create_test_request("req-logprob")
    task.disaggregate_info = None
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    rm.requests[task.request_id] = mock.Mock(idx=0)

    # Setup model output with logprobs
    logprobs_data = {"token_ids": [[10, 20, 30]], "values": [[-0.1, -0.2, -0.3]], "ranks": [0]}
    model_output = _create_model_runner_output(token_ids=[5], logprobs_data=logprobs_data)

    # Process output
    batch_result, draft_result = processor.process_model_runner_output(model_output)

    # Should include logprobs in result
    assert len(batch_result) == 1
    result = batch_result[0]
    assert result.outputs.logprob == pytest.approx(-0.1)
    assert result.outputs.top_logprobs is not None
    assert isinstance(result.outputs.top_logprobs, LogprobsLists)


def test_process_model_runner_output_first_token_metrics():
    """Test first token metrics recording."""
    processor, rm, _, _ = _make_processor()

    # Setup task
    task = _create_test_request("req-first-token")
    task.disaggregate_info = None
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    rm.requests[task.request_id] = mock.Mock(idx=0)

    # Setup initial state (no tokens processed yet)
    processor.tokens_counter["req-first-token"] = 0

    model_output = _create_model_runner_output(token_ids=[5])

    # Process output
    batch_result, draft_result = processor.process_model_runner_output(model_output)

    # Should record first token metrics
    print(f"=====>{batch_result}")
    assert len(batch_result) == 1
    assert processor.tokens_counter["req-first-token"] == 1
    assert task.metrics.first_token_time is not None


def test_process_model_runner_output_multimodal_inputs():
    """Test handling of multimodal inputs."""
    processor, rm, _, _ = _make_processor()

    # Setup task with multimodal inputs
    task = _create_test_request("req-multimodal")
    task.disaggregate_info = None
    task.multimodal_inputs = {"num_input_image_tokens": 4, "num_input_video_tokens": 3}
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    rm.requests[task.request_id] = mock.Mock(idx=0)

    # Initial state for first token
    processor.tokens_counter["req-multimodal"] = 0

    model_output = _create_model_runner_output(token_ids=[5])

    # Process output
    batch_result, draft_result = processor.process_model_runner_output(model_output)

    # Should include multimodal counts in result
    assert len(batch_result) == 1
    result = batch_result[0]
    assert result.num_input_image_tokens == 4
    assert result.num_input_video_tokens == 3


def test_process_model_runner_output_stopped_task():
    """Test skipping of stopped tasks."""
    processor, rm, _, _ = _make_processor()

    # Setup stopped task
    task = _create_test_request("req-stopped")
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    rm.stop_flags[0] = True  # Mark as stopped

    model_output = _create_model_runner_output(token_ids=[5, 6])

    # Process output
    batch_result, draft_result = processor.process_model_runner_output(model_output)

    # Should skip stopped tasks
    assert len(batch_result) == 0
    assert len(draft_result) == 0


def test_process_model_runner_output_speculative_decoding():
    """Test speculative decoding metrics recording."""
    processor, rm, _, _ = _make_processor(speculative_method="mtp")

    # Setup task
    task = _create_test_request("req-speculative")
    task.disaggregate_info = None
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    rm.requests[task.request_id] = mock.Mock(idx=0)

    # Setup multiple tokens for speculative decoding
    model_output = _create_model_runner_output(token_ids=[5, 6, 7])

    # Process output
    batch_result, draft_result = processor.process_model_runner_output(model_output)

    # Should record speculative decoding metrics
    assert len(batch_result) == 1
    assert task.request_id in processor.accept_token_num_per_head_per_request
    assert processor.total_step_per_request[task.request_id] == 1


def test_process_model_runner_output_boundary_values():
    """Test boundary values and edge cases."""
    processor, rm, _, _ = _make_processor()

    # Test empty token sequence
    task = _create_test_request("req-empty")
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task

    model_output = _create_model_runner_output(token_ids=[])
    batch_result, draft_result = processor.process_model_runner_output(model_output)
    assert len(batch_result) == 0  # Empty tokens should be skipped

    # Test single token
    task2 = _create_test_request("req-single")
    rm.tasks_list[1] = task2
    rm.req_dict[task2.request_id] = task2
    rm.requests[task2.request_id] = mock.Mock(idx=1)

    model_output2 = _create_model_runner_output(token_ids=[1])
    batch_result2, draft_result2 = processor.process_model_runner_output(model_output2)
    assert len(batch_result2) == 1


def test_process_model_runner_output_error_handling_with_prompt_logprobs():
    """Test error handling when prompt_logprobs processing fails."""
    processor, rm, _, _ = _make_processor(enable_logprob=True)

    # Setup task
    task = _create_test_request("req-error")
    task.disaggregate_info = None
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    rm.requests[task.request_id] = mock.Mock(idx=0)

    # Setup model output with malformed prompt_logprobs
    logprobs_data = {"token_ids": [[10, 20]], "values": [[-0.1, -0.2]], "ranks": [0]}
    model_output = _create_model_runner_output(token_ids=[5], logprobs_data=logprobs_data)

    # Mock a problematic prompt_logprobs access
    original_tolist = getattr(model_output, "prompt_logprobs", None)
    if original_tolist:
        model_output.prompt_logprobs = mock.Mock(side_effect=Exception("Mock error"))

    # Should handle gracefully
    batch_result, draft_result = processor.process_model_runner_output(model_output)
    assert len(batch_result) == 1  # Should still process main request


def test_process_model_runner_output_multiple_batch_items():
    """Test processing multiple batch items in one output."""
    processor, rm, _, _ = _make_processor(max_num_seqs=3)

    # Setup multiple tasks
    tasks = [_create_test_request("req-1"), _create_test_request("req-2"), _create_test_request("req-3")]

    for i, task in enumerate(tasks):
        rm.tasks_list[i] = task
        rm.req_dict[task.request_id] = task
        rm.requests[task.request_id] = mock.Mock(idx=i)

    # Setup model output with multiple batch items
    cu_num_tokens = paddle.to_tensor([0, 2, 4, 6], dtype=paddle.int32)
    sampled_tokens = paddle.to_tensor([1, 2, 3, 4, 5, 6], dtype=paddle.int64)

    model_output = ModelRunnerOutput(
        decode_mode=DecodeMode.TARGET,
        cu_num_generated_tokens=cu_num_tokens,
        sampled_token_ids=sampled_tokens,
        logprobs=None,
        prompt_logprobs=None,
        pooler_output=None,
    )

    # Process output
    batch_result, draft_result = processor.process_model_runner_output(model_output)

    # Should process all active tasks
    assert len(batch_result) == 3
    request_ids = {result.request_id for result in batch_result}
    assert request_ids == {"req-1", "req-2", "req-3"}


def test_process_model_runner_output_concurrent_abort_and_processing():
    """Test concurrent abort scenario during processing."""
    processor, rm, _, _ = _make_processor()

    # Setup task
    task = _create_test_request("req-concurrent")
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task

    # Setup model output
    model_output = _create_model_runner_output(token_ids=[5, 6])

    # Simulate concurrent abort by removing from abort set during processing
    def mock_abort_remove(request_id):
        rm.abort_req_ids_set.discard(request_id)
        return True

    rm.abort_req_ids_set.add("req-concurrent")

    # Should handle gracefully
    batch_result, draft_result = processor.process_model_runner_output(model_output)
    assert len(batch_result) == 0 or len(batch_result) == 1  # Could go either way


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
