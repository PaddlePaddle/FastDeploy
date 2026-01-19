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

import copy
import time
import types
from collections import Counter
from unittest import mock

import numpy as np
import paddle
import pytest

from fastdeploy import envs
from fastdeploy.engine.request import Request, RequestMetrics, RequestOutput
from fastdeploy.output import token_processor
from fastdeploy.output.token_processor import (
    MAX_BSZ,
    MAX_DRAFT_TOKENS,
    SPECULATE_MAX_BSZ,
    K,
    TokenProcessor,
)


class _DummyCfg:
    def __init__(
        self,
        speculative_method=None,
        enable_logprob=False,
        max_num_seqs=2,
        enable_prefix_caching=False,
        enable_output_caching=False,
    ):
        self.parallel_config = types.SimpleNamespace(
            local_data_parallel_id=0,
            enable_expert_parallel=False,
            data_parallel_size=1,
        )
        self.speculative_config = types.SimpleNamespace(
            method=speculative_method,
            num_speculative_tokens=2,
            enable_draft_logprob=True,
        )
        self.model_config = types.SimpleNamespace(enable_logprob=enable_logprob)
        self.scheduler_config = types.SimpleNamespace(name="default", splitwise_role="decode")
        self.cache_config = types.SimpleNamespace(
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


@pytest.fixture(autouse=True)
def _ensure_cpu():
    paddle.device.set_device("cpu")


def _make_processor(
    speculative_method=None,
    enable_logprob=False,
    max_num_seqs=2,
    enable_prefix_caching=False,
    enable_output_caching=False,
):
    cfg = _DummyCfg(
        speculative_method=speculative_method,
        enable_logprob=enable_logprob,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=enable_prefix_caching,
        enable_output_caching=enable_output_caching,
    )
    cache = mock.Mock()
    queue = _DummyQueue()
    connector = _DummyConnector()
    processor = TokenProcessor(cfg, cache, queue, connector)
    rm = _DummyResourceManager(max_num_seqs)
    processor.set_resource_manager(rm)
    return processor, rm, cache, connector


class _Metric:
    def __init__(self):
        self.value = None

    def set(self, v):
        self.value = v

    def inc(self, v=1):
        self.value = (self.value or 0) + v

    def dec(self, v=1):
        self.value = (self.value or 0) - v

    def observe(self, v):
        self.value = v


class _Metrics:
    def __init__(self):
        self.spec_decode_num_accepted_tokens_total = _Metric()
        self.spec_decode_num_emitted_tokens_total = _Metric()
        self.spec_decode_draft_acceptance_rate = _Metric()
        self.spec_decode_efficiency = _Metric()
        self.spec_decode_num_draft_tokens_total = _Metric()
        self.spec_decode_draft_single_head_acceptance_rate = [_Metric() for _ in range(MAX_DRAFT_TOKENS)]
        self.time_per_output_token = _Metric()
        self.generation_tokens_total = _Metric()
        self.time_to_first_token = _Metric()
        self.request_queue_time = _Metric()
        self.request_prefill_time = _Metric()
        self.request_decode_time = _Metric()
        self.request_inference_time = _Metric()
        self.request_generation_tokens = _Metric()
        self.num_requests_running = _Metric()
        self.request_success_total = _Metric()
        self.available_gpu_block_num = _Metric()
        self.batch_size = _Metric()
        self.available_batch_size = _Metric()

    def _init_speculative_metrics(self, method, num_speculative_tokens):
        return None


def test_init_allocates_expected_buffers():
    processor, _, _, _ = _make_processor()
    assert list(processor.output_tokens.shape) == [MAX_BSZ + 2, 1]

    processor_logprob, _, _, _ = _make_processor(enable_logprob=True)
    assert list(processor_logprob.output_scores.shape) == [MAX_BSZ * (K + 1), 1]

    processor_spec, _, _, _ = _make_processor(speculative_method="mtp", enable_logprob=False)
    assert processor_spec.output_tokens.shape[0] == SPECULATE_MAX_BSZ * MAX_DRAFT_TOKENS + SPECULATE_MAX_BSZ + 2


def test_run_uses_correct_worker_based_on_flag():
    processor, _, _, _ = _make_processor()
    processor.worker = None
    with (
        mock.patch.object(envs, "FD_USE_GET_SAVE_OUTPUT_V1", True),
        mock.patch("fastdeploy.output.token_processor.threading.Thread") as thread_cls,
    ):
        fake_thread = mock.Mock()
        thread_cls.return_value = fake_thread
        processor.run()
    target = thread_cls.call_args.kwargs["target"]
    assert target.__func__ is processor.process_sampling_results_use_zmq.__func__
    assert fake_thread.daemon is True

    processor.worker = object()
    with pytest.raises(Exception):
        processor.run()


def test_cleanup_resources_shuts_down_executor():
    processor, _, _, _ = _make_processor()
    processor.executor = mock.Mock()
    processor._cleanup_resources()
    processor.executor.shutdown.assert_called_once_with(wait=False)


def test_reschedule_preempt_task_use_zmq_reschedules_missing_batch():
    processor, rm, _, _ = _make_processor()
    rm.to_be_rescheduled_request_id_set = {"req-a"}
    rm.requests = {"req-a": types.SimpleNamespace(idx=1)}
    with mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", True):
        processor._reschedule_preempt_task_use_zmq([types.SimpleNamespace(batch_id=0)])
    assert "reschedule-req-a" in rm.recycled


def test_process_batch_draft_tokens_collects_top_logprobs():
    processor, rm, _, _ = _make_processor(speculative_method="mtp", enable_logprob=True)
    rm.tasks_list[0] = types.SimpleNamespace(request_id="task-0", block_tables=[1])
    ranks = paddle.to_tensor(np.array([[0, 1, 1]]))
    scores = paddle.ones([1, 3, 1], dtype="float32")
    tokens = paddle.arange(9, dtype="int64").reshape([1, 3, 3])

    results = processor._process_batch_draft_tokens(
        4, batch=1, accept_num=[2], tokens=tokens, scores=scores, ranks=ranks
    )

    assert len(results) == 1
    assert results[0].outputs.draft_top_logprobs.logprob_token_ids[0][0] == 0
    assert results[0].outputs.draft_top_logprobs.sampled_token_ranks[-1] == 1


def test_process_batch_output_use_zmq_finishes_on_eos():
    processor, rm, cache, connector = _make_processor()
    base_time = time.time()
    task = Request(
        request_id="req-zmq",
        prompt=["hi"],
        prompt_token_ids=[1, 2],
        prompt_token_ids_len=2,
        messages=[[{"content": "hi", "role": "user"}]],
        history=[],
        tools=[],
        system="system",
        eos_token_ids=[6],
        metrics=RequestMetrics(
            arrival_time=base_time,
            preprocess_start_time=base_time - 0.2,
            preprocess_end_time=base_time - 0.1,
            inference_start_time=base_time,
        ),
    )
    task.metrics.decode_inference_start_time = base_time
    task.disaggregate_info = None
    task.ic_req_data = None
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    rm.requests[task.request_id] = types.SimpleNamespace(idx=0)

    tokens = np.array([5, 6], dtype=np.int64)
    stream = types.SimpleNamespace(batch_id=0, tokens=tokens, pooler_output=None)
    with mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False):
        results = processor._process_batch_output_use_zmq([stream])

    assert results[0].finished is True
    assert task.output_token_ids == [5, 6]
    assert rm.stop_flags[0] is True
    assert connector.calls == []


def test_process_batch_output_use_zmq_parses_logprobs():
    processor, rm, _, _ = _make_processor(enable_logprob=True)
    base_time = time.time()
    task = Request(
        request_id="req-zmq-logprob",
        prompt=["hi"],
        prompt_token_ids=[1],
        prompt_token_ids_len=1,
        messages=[[{"content": "hi", "role": "user"}]],
        history=[],
        tools=[],
        system="system",
        eos_token_ids=[6],
        metrics=RequestMetrics(
            arrival_time=base_time,
            preprocess_start_time=base_time - 0.2,
            preprocess_end_time=base_time - 0.1,
            inference_start_time=base_time,
        ),
    )
    task.metrics.decode_inference_start_time = base_time
    task.disaggregate_info = None
    task.ic_req_data = None
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    rm.requests[task.request_id] = types.SimpleNamespace(idx=0)

    logprob_list = token_processor.LogprobsLists(
        logprob_token_ids=[[1, 2]],
        logprobs=[[0.1, 0.2]],
        sampled_token_ranks=[0],
    )
    logprob_holder = types.SimpleNamespace(tolists=lambda: logprob_list)
    stream = types.SimpleNamespace(
        batch_id=0,
        tokens=np.array([5], dtype=np.int64),
        pooler_output=None,
        logprobs=logprob_holder,
        prompt_logprobs={"0": -0.1},
    )
    with mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False):
        results = processor._process_batch_output_use_zmq([stream])

    assert results[0].outputs.logprob == 0.1
    assert results[0].outputs.top_logprobs is logprob_list
    assert results[0].prompt_logprobs == {"0": -0.1}


def test_recycle_resources_updates_metrics_and_state():
    processor, rm, _, _ = _make_processor()
    task = types.SimpleNamespace(request_id="req-1", block_tables=[1], disaggregate_info=None)
    task.trace_carrier = None
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task

    metrics = RequestMetrics(
        arrival_time=time.time(), preprocess_start_time=time.time(), preprocess_end_time=time.time()
    )
    result = RequestOutput(request_id=task.request_id, outputs=None, finished=False, metrics=metrics)
    with mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False):
        processor._recycle_resources(task.request_id, 0, task, result, is_prefill=False)

    assert rm.stop_flags[0] is True
    assert task.request_id not in rm.req_dict
    assert rm.recycled[-1] == task.request_id
    assert processor.tokens_counter.get(task.request_id) is None


def test_compute_speculative_status_builds_metrics():
    processor, rm, _, _ = _make_processor(speculative_method="mtp", enable_logprob=True)
    req_id = "req-spec"
    rm.tasks_list[0] = types.SimpleNamespace(request_id=req_id, block_tables=[1])
    metrics = RequestMetrics(
        arrival_time=time.time(), preprocess_start_time=time.time(), preprocess_end_time=time.time()
    )
    result = RequestOutput(request_id=req_id, outputs=None, finished=False, metrics=metrics)

    processor.total_step = 2
    processor.number_of_output_tokens = 4
    processor.speculative_stats_step = 0
    processor.accept_token_num_per_head = [2, 1] + [0] * (MAX_DRAFT_TOKENS - 2)
    processor.accept_token_num_per_head_per_request[req_id] = [2, 1]
    processor.total_step_per_request[req_id] = 2
    processor._compute_speculative_status(result)

    assert hasattr(result.metrics, "speculate_metrics")
    assert result.metrics.speculate_metrics.accepted_tokens == 3


def test_process_per_token_handles_recovery_stop_and_cleanup():
    processor, rm, _, _ = _make_processor()
    metrics = RequestMetrics(
        arrival_time=time.time(), preprocess_start_time=time.time(), preprocess_end_time=time.time()
    )
    metrics.inference_start_time = time.time()
    metrics.decode_inference_start_time = metrics.inference_start_time
    task = types.SimpleNamespace(
        request_id="req-recover",
        prompt=["hi"],
        prompt_token_ids=[1],
        prompt_token_ids_len=1,
        messages=[],
        history=[],
        tools=[],
        system="sys",
        eos_token_ids=[99],
        metrics=metrics,
        output_token_ids=[],
        block_tables=[1],
    )
    task.trace_carrier = None
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    rm.requests[task.request_id] = types.SimpleNamespace(idx=0)

    result = RequestOutput(
        request_id=task.request_id,
        outputs=types.SimpleNamespace(token_ids=[], tool_calls=[]),
        finished=False,
        metrics=copy.copy(task.metrics),
    )

    with mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False):
        stopped = processor._process_per_token(
            task,
            batch_id=0,
            token_ids=np.array([token_processor.RECOVERY_STOP_SIGNAL]),
            result=result,
            is_prefill=False,
        )

    assert stopped.finished is True
    assert "incomplete" in stopped.error_msg
    assert rm.stop_flags[0] is True
    assert rm.tasks_list[0] is None
    assert processor.tokens_counter.get(task.request_id) is None


def test_postprocess_buffers_and_merges_speculative_results():
    processor, _, _, _ = _make_processor(speculative_method="mtp", enable_logprob=True)
    processor.cached_generated_tokens = mock.Mock()

    target_output = RequestOutput(
        request_id="req-t",
        outputs=types.SimpleNamespace(draft_top_logprobs=None, tool_calls=[]),
        finished=False,
        metrics=RequestMetrics(arrival_time=time.time(), preprocess_start_time=0, preprocess_end_time=0),
    )
    draft_output = RequestOutput(
        request_id="req-t",
        outputs=types.SimpleNamespace(draft_top_logprobs="draft-logprobs", tool_calls=[]),
        finished=False,
        metrics=RequestMetrics(arrival_time=time.time(), preprocess_start_time=0, preprocess_end_time=0),
    )

    processor.postprocess([target_output], mtype=3)
    assert processor._batch_result_buffer == [target_output]

    processor.postprocess([draft_output], mtype=4)
    processor.cached_generated_tokens.put_results.assert_called_once()
    merged = processor.cached_generated_tokens.put_results.call_args.args[0][0]
    assert merged.outputs.draft_top_logprobs == "draft-logprobs"
    assert processor._batch_result_buffer is None


def test_postprocess_emits_finished_speculative_batch():
    processor, _, _, _ = _make_processor(speculative_method="mtp", enable_logprob=True)
    processor.cached_generated_tokens = mock.Mock()

    finished_output = RequestOutput(
        request_id="req-finished",
        outputs=types.SimpleNamespace(draft_top_logprobs=None, tool_calls=[]),
        finished=True,
        metrics=RequestMetrics(arrival_time=time.time(), preprocess_start_time=0, preprocess_end_time=0),
    )

    processor.postprocess([finished_output], mtype=3)

    processor.cached_generated_tokens.put_results.assert_called_once_with([finished_output])
    assert processor._batch_result_buffer is None


def test_postprocess_passes_through_unknown_type():
    processor, _, _, _ = _make_processor(speculative_method="mtp", enable_logprob=True)
    processor.cached_generated_tokens = mock.Mock()

    output = RequestOutput(
        request_id="req-direct",
        outputs=types.SimpleNamespace(draft_top_logprobs=None, tool_calls=[]),
        finished=False,
        metrics=RequestMetrics(arrival_time=time.time(), preprocess_start_time=0, preprocess_end_time=0),
    )

    processor.postprocess([output], mtype=99)

    processor.cached_generated_tokens.put_results.assert_called_once_with([output])


def test_postprocess_logs_and_swallows_exception():
    processor, _, _, _ = _make_processor()
    processor.cached_generated_tokens = mock.Mock()
    processor.cached_generated_tokens.put_results.side_effect = RuntimeError("boom")

    output = RequestOutput(
        request_id="req-error",
        outputs=None,
        finished=False,
        metrics=RequestMetrics(arrival_time=time.time(), preprocess_start_time=0, preprocess_end_time=0),
    )

    processor.postprocess([output])

    processor.cached_generated_tokens.put_results.assert_called_once()


def test_record_speculative_decoding_metrics_tracks_acceptance():
    processor, _, _, _ = _make_processor(speculative_method="mtp", enable_logprob=True)
    with mock.patch.object(token_processor, "main_process_metrics", _Metrics()):
        processor.accept_token_num_per_head = [2, 3, 0, 0, 0, 0]
        processor.num_draft_tokens = 0
        processor.num_emitted_tokens = 0
        processor.num_accepted_tokens = 0

        processor._record_speculative_decoding_metrics(accept_num=[1, 2])

        metrics = token_processor.main_process_metrics
        assert metrics.spec_decode_num_accepted_tokens_total.value == 3
        assert metrics.spec_decode_num_emitted_tokens_total.value == 5
        assert pytest.approx(metrics.spec_decode_draft_acceptance_rate.value) == 0.75
        assert pytest.approx(metrics.spec_decode_efficiency.value) == pytest.approx(5 / 6)
        assert pytest.approx(metrics.spec_decode_draft_single_head_acceptance_rate[0].value) == 1.5


def test_recycle_resources_prefill_sends_first_token():
    processor, rm, _, connector = _make_processor()
    task_id = "req-prefill"
    metrics = RequestMetrics(
        arrival_time=time.time(),
        preprocess_start_time=time.time(),
        preprocess_end_time=time.time(),
        inference_start_time=time.time(),
    )
    task = types.SimpleNamespace(
        request_id=task_id,
        metrics=metrics,
        block_tables=[1],
        disaggregate_info={"role": "prefill"},
        eos_token_ids=[1],
    )
    task.trace_carrier = None
    rm.tasks_list[0] = task
    rm.req_dict[task_id] = task
    result = RequestOutput(request_id=task_id, outputs=None, finished=False, metrics=metrics)
    processor.engine_worker_queue = mock.Mock()
    processor.engine_worker_queue.get_finished_req.side_effect = [[(task_id, "finished")]]

    with mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False):
        processor._recycle_resources(task_id, 0, task, result, is_prefill=True)

    assert rm.stop_flags[0] is True
    assert connector.calls and connector.calls[0][1][0] is result


def test_recycle_resources_prefill_failure_sets_error():
    processor, rm, _, connector = _make_processor()
    task_id = "req-prefill-failed"
    metrics = RequestMetrics(
        arrival_time=time.time(),
        preprocess_start_time=time.time(),
        preprocess_end_time=time.time(),
        inference_start_time=time.time(),
    )
    task = types.SimpleNamespace(
        request_id=task_id,
        metrics=metrics,
        block_tables=[1],
        disaggregate_info={"role": "prefill"},
        eos_token_ids=[1],
    )
    task.trace_carrier = None
    rm.tasks_list[0] = task
    rm.req_dict[task_id] = task
    result = RequestOutput(request_id=task_id, outputs=None, finished=False, metrics=metrics)
    processor.engine_worker_queue = mock.Mock()
    processor.engine_worker_queue.get_finished_req.side_effect = [[(task_id, "failed")]]

    with mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False):
        processor._recycle_resources(task_id, 0, task, result, is_prefill=True)

    assert result.error_code == 400
    assert "failed" in result.error_message
    assert connector.calls and connector.calls[0][1][0] is result


def test_clear_data_marks_all_tasks_finished():
    processor, rm, _, _ = _make_processor()
    metrics = RequestMetrics(
        arrival_time=time.time(), preprocess_start_time=time.time(), preprocess_end_time=time.time()
    )
    task_a = types.SimpleNamespace(
        request_id="req-a",
        eos_token_ids=[0],
        metrics=metrics,
        disaggregate_info=None,
        block_tables=[1],
        arrival_time=time.time(),
    )
    task_b = types.SimpleNamespace(
        request_id="req-b",
        eos_token_ids=[0],
        metrics=metrics,
        disaggregate_info=None,
        block_tables=[2],
        arrival_time=time.time(),
    )
    rm.tasks_list[0] = task_a
    rm.tasks_list[1] = task_b
    rm.req_dict = {"req-a": task_a, "req-b": task_b}
    processor.tokens_counter = Counter({"req-a": 2, "req-b": 1})

    with mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False):
        processor.clear_data()

    assert rm.tasks_list == [None, None]
    assert not processor.tokens_counter
    assert task_a.request_id in rm.recycled and task_b.request_id in rm.recycled


def test_record_speculative_decoding_accept_num_per_request_updates_maps():
    processor, _, _, _ = _make_processor(speculative_method="mtp")
    processor._record_speculative_decoding_accept_num_per_request("req-acc", 3)

    assert processor.total_step_per_request["req-acc"] == 1
    assert processor.accept_token_num_per_head_per_request["req-acc"][0] == 1
    assert processor.accept_token_num_per_head[2] == 1


def test_process_batch_output_consumes_tokens_and_finishes_task():
    processor, rm, _, _ = _make_processor()
    metrics = RequestMetrics(
        arrival_time=time.time(),
        preprocess_start_time=time.time(),
        preprocess_end_time=time.time(),
        inference_start_time=time.time(),
    )
    metrics.decode_inference_start_time = metrics.inference_start_time
    task = types.SimpleNamespace(
        request_id="req-out",
        disaggregate_info=None,
        eos_token_ids=[7],
        metrics=metrics,
        output_token_ids=[],
        messages=[{"role": "user", "content": "hi"}],
        num_cached_tokens=0,
        ic_req_data=None,
        prompt_token_ids_len=0,
        num_total_tokens=1,
        block_tables=[1],
    )
    task.trace_carrier = None
    task.get = lambda key, default=None: getattr(task, key, default)
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    processor.output_tokens[1, 0] = 1
    processor.output_tokens[2, 0] = 7

    with mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False):
        processor._process_batch_output()

    assert rm.stop_flags[0] is True
    assert task.output_token_ids == [7]


def test_process_batch_output_logprob_records_topk_and_caching():
    processor, rm, _, _ = _make_processor(enable_logprob=True, enable_prefix_caching=True, enable_output_caching=True)
    metrics = RequestMetrics(
        arrival_time=time.time(),
        preprocess_start_time=time.time(),
        preprocess_end_time=time.time(),
        inference_start_time=time.time(),
    )
    metrics.decode_inference_start_time = metrics.inference_start_time
    task = types.SimpleNamespace(
        request_id="req-logprob",
        disaggregate_info=None,
        eos_token_ids=[3],
        metrics=metrics,
        output_token_ids=[],
        messages=[],
        num_cached_tokens=0,
        ic_req_data=None,
        prompt_token_ids_len=0,
        num_total_tokens=1,
        block_tables=[1],
        get=lambda key, default=None: None,
    )
    task.trace_carrier = None
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    processor.output_tokens[1, 0] = 1
    token_block = np.arange(K + 1, dtype=np.int64) + 3
    processor.output_tokens[2 : 2 + K + 1] = paddle.to_tensor(token_block.reshape([-1, 1]))
    processor.output_scores[: K + 1] = paddle.ones([K + 1, 1], dtype="float32")
    processor.output_ranks[0] = paddle.to_tensor(0, dtype="int64")
    processor.cached_generated_tokens.put_results = mock.Mock()

    with (
        mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", True),
        mock.patch.object(token_processor, "main_process_metrics", _Metrics()),
    ):
        processor._process_batch_output()

    assert rm.cached_tasks[-1] == "req-logprob"
    sent = processor.cached_generated_tokens.put_results.call_args.args[0][0]
    assert sent.outputs.top_logprobs is not None


def test_process_batch_output_speculative_logprob_handles_draft_batch():
    processor, rm, _, _ = _make_processor(speculative_method="mtp", enable_logprob=True)
    rm.tasks_list[0] = types.SimpleNamespace(request_id="req-draft", block_tables=[1], disaggregate_info=None)
    target = RequestOutput(
        request_id="req-draft",
        outputs=types.SimpleNamespace(draft_top_logprobs=None, tool_calls=[]),
        finished=False,
        metrics=None,
    )
    processor._batch_result_buffer = [target]
    processor.cached_generated_tokens = mock.Mock()
    processor.output_tokens[1, 0] = 4
    processor.output_tokens[2, 0] = 1
    processor.output_tokens[3, 0] = 1

    draft_tokens = np.arange(MAX_DRAFT_TOKENS * (K + 1), dtype=np.int64).reshape([-1, 1]) + 5
    processor.output_tokens[3 + MAX_BSZ : 3 + MAX_BSZ + len(draft_tokens)] = paddle.to_tensor(draft_tokens)
    processor.output_scores[: MAX_DRAFT_TOKENS * (K + 1)] = paddle.ones(
        [MAX_DRAFT_TOKENS * (K + 1), 1], dtype="float32"
    )
    processor.output_ranks[:MAX_DRAFT_TOKENS] = paddle.arange(MAX_DRAFT_TOKENS, dtype="int64")

    with (
        mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False),
        mock.patch.object(token_processor, "main_process_metrics", _Metrics()),
    ):
        processor._process_batch_output()

    sent_batch = processor.cached_generated_tokens.put_results.call_args.args[0]
    assert sent_batch and sent_batch[0].outputs.draft_top_logprobs is not None


def test_process_batch_output_speculative_recovery_stop_finishes():
    processor, rm, _, _ = _make_processor(speculative_method="mtp")
    metrics = RequestMetrics(
        arrival_time=time.time(),
        preprocess_start_time=time.time(),
        preprocess_end_time=time.time(),
        inference_start_time=time.time(),
    )
    metrics.decode_inference_start_time = metrics.inference_start_time
    task = types.SimpleNamespace(
        request_id="req-recover-spec",
        disaggregate_info=None,
        eos_token_ids=[2],
        metrics=metrics,
        output_token_ids=[],
        messages=[],
        num_cached_tokens=0,
        ic_req_data=None,
        prompt_token_ids_len=0,
        num_total_tokens=1,
        block_tables=[1],
        get=lambda key, default=None: None,
    )
    task.trace_carrier = None
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    processor.output_tokens[1] = 1
    processor.output_tokens[2] = -3
    processor.number_of_output_tokens = 1
    processor.total_step = 1
    processor.accept_token_num_per_head_per_request[task.request_id] = [1] + [0] * (MAX_DRAFT_TOKENS - 1)
    processor.total_step_per_request[task.request_id] = 1
    processor.cached_generated_tokens.put_results = mock.Mock()

    with (
        mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False),
        mock.patch.object(token_processor, "main_process_metrics", _Metrics()),
    ):
        processor._process_batch_output()

    assert rm.stop_flags[0] is True
    sent = processor.cached_generated_tokens.put_results.call_args.args[0][0]
    assert sent.finished is True
    assert "incomplete" in sent.error_msg


def test_process_batch_output_prefill_chunk_and_adapter_skip():
    processor, rm, _, _ = _make_processor(enable_logprob=True)
    metrics = RequestMetrics(
        arrival_time=time.time(), preprocess_start_time=time.time(), preprocess_end_time=time.time()
    )
    metrics.inference_start_time = time.time()
    metrics.decode_inference_start_time = metrics.inference_start_time
    processor.cfg.scheduler_config.splitwise_role = "prefill"
    task = types.SimpleNamespace(
        request_id="req-prefill-chunk",
        disaggregate_info={"role": "prefill"},
        eos_token_ids=[1],
        metrics=metrics,
        output_token_ids=[],
        messages=[{"role": "user", "content": "hi"}],
        num_cached_tokens=0,
        ic_req_data=None,
        prompt_token_ids_len=1,
        num_total_tokens=2,
        block_tables=[1],
        prefill_chunk_info=[{"idx": 0}, {"idx": 1}],
    )
    task.trace_carrier = None
    task.get = lambda key, default=None: getattr(task, key, default)
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    processor.output_tokens[1, 0] = 1
    processor.output_tokens[2 : 2 + K + 1] = paddle.to_tensor(np.ones([K + 1, 1], dtype=np.int64))
    processor.output_scores[: K + 1] = paddle.ones([K + 1, 1], dtype="float32")
    processor.output_ranks[0] = paddle.to_tensor(0, dtype="int64")
    processor.cached_generated_tokens.put_results = mock.Mock()

    with (
        mock.patch.object(envs, "FD_ENABLE_INTERNAL_ADAPTER", True),
        mock.patch.object(token_processor, "main_process_metrics", _Metrics()),
    ):
        processor._process_batch_output()

    assert getattr(task, "prefill_chunk_num") == 1
    assert processor.cached_generated_tokens.put_results.call_args.args[0] == []


def test_process_batch_output_handles_multimodal_and_negative_token():
    processor, rm, _, _ = _make_processor()
    metrics = RequestMetrics(
        arrival_time=time.time(), preprocess_start_time=time.time(), preprocess_end_time=time.time()
    )
    metrics.inference_start_time = time.time()
    metrics.decode_inference_start_time = metrics.inference_start_time
    processor.cfg.scheduler_config.splitwise_role = "prefill"
    task = types.SimpleNamespace(
        request_id="req-negative",
        disaggregate_info=None,
        eos_token_ids=[5],
        metrics=metrics,
        output_token_ids=[],
        messages=None,
        num_cached_tokens=0,
        ic_req_data=None,
        prompt_token_ids_len=0,
        num_total_tokens=1,
        block_tables=[1],
        prefill_chunk_info=None,
        multimodal_inputs={"num_input_image_tokens": 2, "num_input_video_tokens": 3},
        get=lambda key, default=None: None,
    )
    task.trace_carrier = None
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    rm.to_be_rescheduled_request_id_set = {task.request_id}
    rm.requests = {task.request_id: types.SimpleNamespace(idx=0)}
    processor.output_tokens[1, 0] = 1
    processor.output_tokens[2, 0] = -9

    with (
        mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", True),
        mock.patch.object(token_processor, "main_process_metrics", _Metrics()),
    ):
        processor._process_batch_output()

    assert rm.recycled[-1] == f"reschedule-{task.request_id}"


def test_process_batch_output_speculative_logprob_targets_topk_scores():
    processor, rm, _, _ = _make_processor(speculative_method="mtp", enable_logprob=True)
    metrics = RequestMetrics(
        arrival_time=time.time(), preprocess_start_time=time.time(), preprocess_end_time=time.time()
    )
    metrics.inference_start_time = time.time()
    metrics.decode_inference_start_time = metrics.inference_start_time
    task = types.SimpleNamespace(
        request_id="req-spec-logprob",
        disaggregate_info=None,
        eos_token_ids=[9],
        metrics=metrics,
        output_token_ids=[],
        messages=None,
        num_cached_tokens=0,
        ic_req_data=None,
        prompt_token_ids_len=0,
        num_total_tokens=1,
        block_tables=[1],
        get=lambda key, default=None: None,
    )
    task.trace_carrier = None
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    processor.output_tokens[1, 0] = 3
    processor.output_tokens[2, 0] = 1
    processor.output_tokens[3, 0] = 2
    token_block = np.arange(MAX_DRAFT_TOKENS * (K + 1), dtype=np.int64).reshape([-1, 1]) + 3
    processor.output_tokens[3 + MAX_BSZ : 3 + MAX_BSZ + len(token_block)] = paddle.to_tensor(token_block)
    score_block = paddle.arange(MAX_DRAFT_TOKENS * (K + 1), dtype="float32").reshape([-1, 1])
    processor.output_scores[: MAX_DRAFT_TOKENS * (K + 1)] = score_block
    processor.output_ranks[:MAX_DRAFT_TOKENS] = paddle.arange(MAX_DRAFT_TOKENS, dtype="int64")
    processor.cached_generated_tokens.put_results = mock.Mock()

    with (
        mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False),
        mock.patch.object(token_processor, "main_process_metrics", _Metrics()),
    ):
        processor._process_batch_output()

    assert processor.tokens_counter[task.request_id] == 2


def test_record_metrics_and_speculative_ngram_metrics():
    processor, _, _, _ = _make_processor(speculative_method="ngram", enable_logprob=True)
    metrics = _Metrics()
    task = types.SimpleNamespace(
        request_id="req-metrics",
        metrics=RequestMetrics(arrival_time=time.time(), preprocess_start_time=0, preprocess_end_time=0),
        last_token_time=time.time(),
    )
    with mock.patch.object(token_processor, "main_process_metrics", metrics):
        processor._record_metrics(task, current_time=time.time(), token_ids=[1, 2])
        processor.accept_token_num_per_head = [0, 2] + [0] * (MAX_DRAFT_TOKENS - 2)
        processor.num_accepted_tokens = 3
        processor.num_emitted_tokens = 3
        processor._record_speculative_decoding_metrics(accept_num=[1, 1])

    assert metrics.generation_tokens_total.value == 2
    assert metrics.spec_decode_draft_acceptance_rate.value == 1


def test_clear_data_invokes_scheduler_cleanup():
    processor, rm, _, _ = _make_processor()
    metrics = RequestMetrics(
        arrival_time=time.time(), preprocess_start_time=time.time(), preprocess_end_time=time.time()
    )
    task = types.SimpleNamespace(
        request_id="req-clear",
        arrival_time=time.time(),
        disaggregate_info=None,
        eos_token_ids=[0],
        metrics=metrics,
        output_token_ids=[],
        messages=None,
        num_cached_tokens=0,
        ic_req_data=None,
        prompt_token_ids_len=0,
        num_total_tokens=1,
        block_tables=[1],
        get=lambda key, default=None: getattr(task, key, default),
    )
    task.trace_carrier = None
    rm.tasks_list[0] = task
    rm.stop_flags = [True] * rm.max_num_seqs
    processor.tokens_counter[task.request_id] = 0

    with (
        mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", True),
        mock.patch.object(token_processor, "main_process_metrics", _Metrics()),
    ):
        processor.clear_data()

    assert rm.cleared is True


def test_process_batch_output_skips_already_stopped_slot():
    processor, rm, _, _ = _make_processor()
    rm.stop_flags[0] = True
    processor.output_tokens[1, 0] = 1
    processor.output_tokens[2, 0] = 5

    with (
        mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False),
        mock.patch.object(token_processor, "main_process_metrics", _Metrics()),
    ):
        processor._process_batch_output()

    assert processor.cached_generated_tokens.put_results.called


def test_process_batch_output_speculative_negative_token_reschedules():
    processor, rm, _, _ = _make_processor(speculative_method="mtp")
    task_id = "req-spec-neg"
    metrics = RequestMetrics(
        arrival_time=time.time(), preprocess_start_time=time.time(), preprocess_end_time=time.time()
    )
    metrics.inference_start_time = time.time()
    metrics.decode_inference_start_time = metrics.inference_start_time
    task = types.SimpleNamespace(
        request_id=task_id,
        disaggregate_info=None,
        eos_token_ids=[1],
        metrics=metrics,
        output_token_ids=[],
        messages=None,
        num_cached_tokens=0,
        ic_req_data=None,
        prompt_token_ids_len=0,
        num_total_tokens=1,
        block_tables=[1],
        get=lambda key, default=None: None,
    )
    task.trace_carrier = None
    rm.tasks_list[0] = task
    rm.req_dict[task_id] = task
    rm.to_be_rescheduled_request_id_set = {task_id}
    rm.requests = {task_id: types.SimpleNamespace(idx=0)}
    processor.output_tokens[1] = 1
    processor.output_tokens[2] = -9
    processor.output_tokens[3] = -1
    processor.output_tokens[2 + SPECULATE_MAX_BSZ] = -1

    with (
        mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", True),
        mock.patch.object(token_processor, "main_process_metrics", _Metrics()),
    ):
        processor._process_batch_output()

    assert rm.recycled[-1] == f"reschedule-{task_id}"


def test_process_batch_output_use_zmq_reschedules_negative_token():
    processor, rm, _, _ = _make_processor()
    task = types.SimpleNamespace(request_id="req-zmq-neg")
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    rm.to_be_rescheduled_request_id_set = {task.request_id}

    stream = types.SimpleNamespace(batch_id=0, tokens=np.array([-9], dtype=np.int64), pooler_output=None)
    with mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", True):
        results = processor._process_batch_output_use_zmq([stream])

    assert results == []
    assert rm.recycled[-1] == f"reschedule-{task.request_id}"


def test_process_batch_output_records_second_decode_token():
    processor, rm, _, _ = _make_processor()
    processor.cfg.scheduler_config.splitwise_role = "decode"
    task = types.SimpleNamespace(
        request_id="req-second",
        disaggregate_info=None,
        eos_token_ids=[2],
        metrics=RequestMetrics(
            arrival_time=time.time(), preprocess_start_time=time.time(), preprocess_end_time=time.time()
        ),
        output_token_ids=[],
        messages=None,
        num_cached_tokens=0,
        ic_req_data=None,
        prompt_token_ids_len=0,
        num_total_tokens=1,
        block_tables=[1],
        get=lambda key, default=None: None,
    )
    task.trace_carrier = None
    task.metrics.inference_start_time = time.time()
    task.metrics.decode_inference_start_time = task.metrics.inference_start_time
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    processor.tokens_counter[task.request_id] = 1
    processor.output_tokens[1, 0] = 1
    processor.output_tokens[2, 0] = 2

    with (
        mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False),
        mock.patch.object(token_processor, "main_process_metrics", _Metrics()),
    ):
        processor._process_batch_output()

    assert task.metrics.decode_recv_second_token_time is not None


def test_record_speculative_metrics_calls_init_when_missing():
    processor, _, _, _ = _make_processor(speculative_method="mtp")

    class _MinimalMetrics:
        def __init__(self):
            self.init_called = False

        def _init_speculative_metrics(self, method, num_speculative_tokens):
            self.spec_decode_num_accepted_tokens_total = _Metric()
            self.spec_decode_num_emitted_tokens_total = _Metric()
            self.spec_decode_draft_acceptance_rate = _Metric()
            self.spec_decode_efficiency = _Metric()
            self.spec_decode_num_draft_tokens_total = _Metric()
            self.spec_decode_draft_single_head_acceptance_rate = [_Metric() for _ in range(MAX_DRAFT_TOKENS)]
            self.init_called = True

    processor.accept_token_num_per_head = [1, 1] + [0] * (MAX_DRAFT_TOKENS - 2)
    processor.num_accepted_tokens = 2
    processor.num_emitted_tokens = 2

    metrics = _MinimalMetrics()
    with mock.patch.object(token_processor, "main_process_metrics", metrics):
        processor._record_speculative_decoding_metrics(accept_num=[1])

    assert metrics.init_called is True


def test_process_batch_output_prefill_sets_draft_tokens():
    processor, rm, _, connector = _make_processor(speculative_method="mtp")
    processor.cfg.scheduler_config.splitwise_role = "prefill"
    metrics = RequestMetrics(
        arrival_time=time.time(),
        preprocess_start_time=time.time(),
        preprocess_end_time=time.time(),
        inference_start_time=time.time(),
    )
    metrics.decode_inference_start_time = metrics.inference_start_time
    task = types.SimpleNamespace(
        request_id="req-prefill-draft",
        disaggregate_info={"role": "prefill"},
        eos_token_ids=[99],
        metrics=metrics,
        output_token_ids=[],
        messages=None,
        num_cached_tokens=0,
        ic_req_data=None,
        prompt_token_ids_len=0,
        num_total_tokens=1,
        block_tables=[1],
        get=lambda key, default=None: None,
    )
    task.trace_carrier = None
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    processor.engine_worker_queue = mock.Mock()
    processor.engine_worker_queue.get_finished_req.side_effect = [[(task.request_id, "finished")]]
    processor.output_tokens[1] = 1
    processor.output_tokens[2] = 2
    processor.output_tokens[2 + SPECULATE_MAX_BSZ] = 11
    processor.output_tokens[2 + SPECULATE_MAX_BSZ + 1] = 12

    with (
        mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False),
        mock.patch.object(token_processor, "main_process_metrics", _Metrics()),
    ):
        processor._process_batch_output()

    assert connector.calls
    sent = connector.calls[0][1][0]
    assert sent.outputs.draft_token_ids == [11, 12]


def test_process_batch_output_logs_recovery_stop_for_non_speculative():
    processor, rm, _, _ = _make_processor()
    metrics = RequestMetrics(
        arrival_time=time.time(), preprocess_start_time=time.time(), preprocess_end_time=time.time()
    )
    metrics.inference_start_time = time.time()
    metrics.decode_inference_start_time = metrics.inference_start_time
    task = types.SimpleNamespace(
        request_id="req-recovery",
        disaggregate_info=None,
        eos_token_ids=[1],
        metrics=metrics,
        output_token_ids=[],
        messages=None,
        num_cached_tokens=0,
        ic_req_data=None,
        prompt_token_ids_len=0,
        num_total_tokens=1,
        block_tables=[1],
    )
    task.trace_carrier = None
    task.get = lambda k, d=None: getattr(task, k, d)
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    processor.output_tokens[1, 0] = 1
    processor.output_tokens[2, 0] = token_processor.RECOVERY_STOP_SIGNAL

    with (
        mock.patch.object(envs, "ENABLE_V1_KVCACHE_SCHEDULER", False),
        mock.patch.object(token_processor, "main_process_metrics", _Metrics()),
    ):
        processor._process_batch_output()

    assert rm.stop_flags[0] is True


def test_process_batch_output_sets_multimodal_token_counts():
    processor, rm, _, _ = _make_processor()
    metrics = RequestMetrics(
        arrival_time=time.time(), preprocess_start_time=time.time(), preprocess_end_time=time.time()
    )
    metrics.inference_start_time = time.time()
    metrics.decode_inference_start_time = metrics.inference_start_time
    task = types.SimpleNamespace(
        request_id="req-mm",
        disaggregate_info=None,
        eos_token_ids=[7],
        metrics=metrics,
        output_token_ids=[],
        messages=None,
        num_cached_tokens=0,
        ic_req_data=None,
        prompt_token_ids_len=0,
        num_total_tokens=1,
        block_tables=[1],
        multimodal_inputs={"num_input_image_tokens": 4, "num_input_video_tokens": 5},
    )
    task.trace_carrier = None
    task.get = lambda key, default=None: getattr(task, key, default)
    rm.tasks_list[0] = task
    rm.req_dict[task.request_id] = task
    processor.output_tokens[1, 0] = 1
    processor.output_tokens[2, 0] = 7

    with mock.patch.object(token_processor, "main_process_metrics", _Metrics()):
        processor._process_batch_output()

    sent = processor.cached_generated_tokens.put_results.call_args.args[0][0]
    assert sent.num_input_image_tokens == 4 and sent.num_input_video_tokens == 5


def test_warmup_token_processor_initialization():
    cfg = _DummyCfg()
    with mock.patch.object(token_processor.TokenProcessor, "__init__", lambda self, _cfg: None):
        warm = token_processor.WarmUpTokenProcessor(cfg)
    assert warm._is_running is True and warm._is_blocking is True
    warm.postprocess([])


def test_warmup_processor_stop_joins_worker():
    warm = token_processor.WarmUpTokenProcessor.__new__(token_processor.WarmUpTokenProcessor)
    warm._is_running = True
    worker = mock.Mock()
    warm.worker = worker
    warm.stop()
    worker.join.assert_called_once()


def test_healthy_behaviour_respects_timeout(monkeypatch):
    processor, _, _, _ = _make_processor()
    processor.timestamp_for_alive_before_handle_batch = time.time() - 1
    processor.timestamp_for_alive_after_handle_batch = None
    monkeypatch.setattr(envs, "FD_TOKEN_PROCESSOR_HEALTH_TIMEOUT", 0.1)

    assert processor.healthy() is False


def test_healthy_detects_engine_hang():
    processor, _, _, _ = _make_processor()
    processor.timestamp_for_alive_before_handle_batch = None
    processor.timestamp_for_alive_after_handle_batch = time.time()
    processor.engine_output_token_hang = True

    assert processor.healthy() is False


def test_healthy_recent_prehandle_activity_is_ok(monkeypatch):
    processor, _, _, _ = _make_processor()
    processor.timestamp_for_alive_before_handle_batch = time.time()
    processor.timestamp_for_alive_after_handle_batch = None
    monkeypatch.setattr(envs, "FD_TOKEN_PROCESSOR_HEALTH_TIMEOUT", 5)

    assert processor.healthy() is True


def test_record_completion_metrics_updates_counters():
    processor, _, _, _ = _make_processor()
    task_id = "req-complete"
    metrics = RequestMetrics(
        arrival_time=time.time(), preprocess_start_time=time.time(), preprocess_end_time=time.time()
    )
    metrics.inference_start_time = time.time() - 0.2
    metrics.engine_recv_first_token_time = time.time() - 0.1
    task = types.SimpleNamespace(request_id=task_id, metrics=metrics, user="user-a")
    processor.tokens_counter[task_id] = 4

    with (
        mock.patch.object(token_processor, "main_process_metrics", _Metrics()) as metrics_obj,
        mock.patch.object(token_processor, "trace_print"),
    ):
        processor._record_completion_metrics(task, current_time=time.time())

        assert metrics_obj.request_decode_time.value is not None
        assert metrics_obj.request_success_total.value == 1
        assert metrics_obj.request_generation_tokens.value == 4


def test_process_sampling_results_use_zmq_rejects_speculative():
    processor, _, _, _ = _make_processor(speculative_method="mtp")
    with pytest.raises(NotImplementedError):
        processor.process_sampling_results_use_zmq()
