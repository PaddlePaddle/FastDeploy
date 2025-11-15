from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import Any, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakeTensor:
    def __init__(self, array: Any):
        self.array = np.array(array)

    def numpy(self):
        return self.array

    def __getitem__(self, item):
        value = self.array.__getitem__(item)
        if isinstance(value, np.ndarray):
            return _FakeTensor(value)
        return value

    def __setitem__(self, key, value):
        self.array.__setitem__(key, value)

    def reshape(self, *args, **kwargs):  # pragma: no cover - compatibility helper
        return self.array.reshape(*args, **kwargs)


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


class _Metric:
    def __init__(self):
        self.values: List[Any] = []

    def observe(self, value):
        self.values.append(("observe", value))

    def set(self, value):
        self.values.append(("set", value))

    def inc(self, value=1):
        self.values.append(("inc", value))

    def dec(self, value=1):
        self.values.append(("dec", value))


class _MainMetrics:
    def __init__(self):
        self.spec_decode_draft_single_head_acceptance_rate = []

    def __getattr__(self, name):  # pragma: no cover - simple factory
        if name == "spec_decode_draft_acceptance_rate":
            raise AttributeError(name)
        metric = _Metric()
        setattr(self, name, metric)
        return metric

    def _init_speculative_metrics(self, _method, num_speculative_tokens):
        self.spec_decode_draft_single_head_acceptance_rate = [_Metric() for _ in range(num_speculative_tokens)]
        self.spec_decode_draft_acceptance_rate = _Metric()
        self.spec_decode_efficiency = _Metric()
        self.spec_decode_num_draft_tokens_total = _Metric()
        self.spec_decode_num_accepted_tokens_total = _Metric()
        self.spec_decode_num_emitted_tokens_total = _Metric()


class _Logger:
    def __init__(self):
        self.messages = []

    def debug(self, msg):  # pragma: no cover - helper for interface compatibility
        self.messages.append(("debug", msg))

    def info(self, msg):
        self.messages.append(("info", msg))

    def warning(self, msg):
        self.messages.append(("warning", msg))

    def error(self, msg):
        self.messages.append(("error", msg))


class _LogprobsLists:
    def __init__(self, logprob_token_ids=None, logprobs=None, sampled_token_ranks=None):
        self.logprob_token_ids = logprob_token_ids or []
        self.logprobs = logprobs or []
        self.sampled_token_ranks = sampled_token_ranks or []


class _IPCSignal:
    def __init__(self, name, array, dtype, suffix, create):
        self.name = name
        self.value = array
        self.dtype = dtype
        self.suffix = suffix
        self.create = create

    def clear(self):
        self.value[:] = 0


class _ZmqServer:
    def __init__(self, name, mode):  # pragma: no cover - compatibility helper
        self.name = name
        self.mode = mode

    def recv_pyobj(self):  # pragma: no cover - unused helper
        return []


class _Request(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self


class _RequestMetrics:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _CompletionOutput:
    def __init__(self, index, send_idx, token_ids, draft_token_ids):
        self.index = index
        self.send_idx = send_idx
        self.token_ids = token_ids
        self.draft_token_ids = draft_token_ids
        self.logprob = None
        self.top_logprobs = None
        self.draft_top_logprobs = None


class _PoolingOutput:
    def __init__(self, data):
        self.data = data


class _RequestOutput:
    def __init__(
        self,
        request_id,
        outputs,
        finished=False,
        metrics=None,
        output_type=None,
        **extra,
    ):
        self.request_id = request_id
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.output_type = output_type
        self.prompt = None
        self.num_cached_tokens = 0
        self.num_input_image_tokens = 0
        self.num_input_video_tokens = 0
        self.error_msg = None
        self.error_code = None
        for key, value in extra.items():
            setattr(self, key, value)


class _PoolingRequestOutput(_RequestOutput):
    pass


def _install_stub_modules():
    fake_paddle = types.ModuleType("paddle")
    fake_paddle.device = types.SimpleNamespace(set_device=lambda *_args, **_kwargs: None)
    fake_paddle.full = lambda shape, fill_value=0, dtype=None: _FakeTensor(np.full(shape, fill_value, dtype=dtype))
    sys.modules["paddle"] = fake_paddle

    fake_zmq = types.SimpleNamespace(PULL=1)
    sys.modules["zmq"] = fake_zmq

    fastdeploy_pkg = _ensure_module("fastdeploy")
    fastdeploy_pkg.__path__ = []
    sys.modules["fastdeploy.output"] = _ensure_module("fastdeploy.output")
    _ensure_module("fastdeploy.engine")
    request_module = _ensure_module("fastdeploy.engine.request")
    request_module.CompletionOutput = _CompletionOutput
    request_module.PoolingOutput = _PoolingOutput
    request_module.PoolingRequestOutput = _PoolingRequestOutput
    request_module.Request = _Request
    request_module.RequestMetrics = _RequestMetrics
    request_module.RequestOutput = _RequestOutput

    envs_module = types.SimpleNamespace(
        FD_USE_GET_SAVE_OUTPUT_V1=False,
        ENABLE_V1_KVCACHE_SCHEDULER=False,
        FD_DEBUG=0,
        FD_ENABLE_INTERNAL_ADAPTER=False,
    )
    sys.modules["fastdeploy.envs"] = envs_module

    inter_comm = _ensure_module("fastdeploy.inter_communicator")
    inter_comm.IPCSignal = _IPCSignal
    inter_comm.ZmqIpcServer = _ZmqServer

    metrics_module = _ensure_module("fastdeploy.metrics.metrics")
    metrics_module.main_process_metrics = _MainMetrics()

    platforms_module = _ensure_module("fastdeploy.platforms")
    platforms_module.current_platform = types.SimpleNamespace(
        is_xpu=lambda: False,
        is_iluvatar=lambda: False,
        is_gcu=lambda: False,
        is_intel_hpu=lambda: False,
    )

    utils_module = _ensure_module("fastdeploy.utils")
    utils_module.llm_logger = _Logger()
    utils_module.spec_logger = _Logger()

    worker_module = _ensure_module("fastdeploy.worker.output")
    worker_module.LogprobsLists = _LogprobsLists

    return metrics_module.main_process_metrics, utils_module


def _load_token_processor():
    for name in list(sys.modules):
        if name.startswith("fastdeploy.output.token_processor"):
            sys.modules.pop(name)
    metrics, utils_module = _install_stub_modules()
    spec = importlib.util.spec_from_file_location(
        "fastdeploy.output.token_processor",
        PROJECT_ROOT / "fastdeploy" / "output" / "token_processor.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["fastdeploy.output.token_processor"] = module
    spec.loader.exec_module(module)
    return module, metrics, utils_module


class _DummyResourceManager:
    def __init__(self, max_num_seqs):
        self.max_num_seqs = max_num_seqs
        self.stop_flags = [False] * max_num_seqs
        self.tasks_list = [None] * max_num_seqs
        self.req_dict = {}
        self.requests = {}
        self.to_be_rescheduled_request_id_set = set()
        self.recycled = []
        self.finished_async = []
        self.cleared = False

    def info(self):
        return "resource-info"

    def total_block_number(self):
        return 8

    def available_batch(self):
        return self.tasks_list.count(None)

    def _recycle_block_tables(self, task):
        self.recycled.append(task.request_id)

    def finish_requests_async(self, request_id):
        self.finished_async.append(request_id)

    def reschedule_preempt_task(self, request_id):
        self.recycled.append(f"reschedule-{request_id}")

    def clear_data(self):
        self.cleared = True


class _DummyCache:
    def __init__(self):
        self.results = []

    def put_results(self, batch_result):
        self.results.append(batch_result)


class _DummyQueue:
    def __init__(self, finished=None):
        self._finished = list(finished or [])

    def get_finished_req(self):
        if self._finished:
            return [self._finished.pop(0)]
        return []


class _DummyConnector:
    def __init__(self):
        self.calls = []

    def send_first_token(self, info, results):
        self.calls.append((info, results))


def _build_cfg(max_num_seqs=2, speculative_method=None, enable_logprob=False):
    parallel_config = types.SimpleNamespace(
        local_data_parallel_id=0,
        enable_expert_parallel=False,
        data_parallel_size=1,
    )
    spec_config = types.SimpleNamespace(
        method=speculative_method,
        num_speculative_tokens=2,
    )
    model_config = types.SimpleNamespace(enable_logprob=enable_logprob)
    scheduler_config = types.SimpleNamespace(name="default")
    cfg = types.SimpleNamespace(
        parallel_config=parallel_config,
        speculative_config=spec_config,
        model_config=model_config,
        scheduler_config=scheduler_config,
        max_num_seqs=max_num_seqs,
        splitwise_version="v1",
    )
    return cfg


def _make_request(request_id="req-0", **overrides):
    base = dict(
        request_id=request_id,
        arrival_time=0.1,
        inference_start_time=0.2,
        schedule_start_time=0.3,
        preprocess_start_time=0.05,
        preprocess_end_time=0.15,
        llm_engine_recv_req_timestamp=0.2,
        llm_engine_send_req_to_engine_timestamp=0.25,
        prompt_token_ids=[1, 2],
        output_token_ids=[],
        num_cached_tokens=0,
        messages=[{"role": "user", "content": "hi"}],
        pooling_params=None,
        pooling_outputs=None,
        disaggregate_info=None,
        eos_token_ids=[0],
        block_tables=[1],
        idx=0,
        num_input_image_tokens=0,
        num_input_video_tokens=0,
        prefill_chunk_info=None,
        prefill_chunk_num=0,
        multimodal_inputs={},
    )
    base.update(overrides)
    return _Request(**base)


class TokenProcessorTestCase(unittest.TestCase):
    def setUp(self):
        self.module, self.metrics, self.utils_module = _load_token_processor()

    def _create_processor(self, **cfg_kwargs):
        cfg = _build_cfg(**cfg_kwargs)
        cache = _DummyCache()
        queue = _DummyQueue()
        connector = _DummyConnector()
        processor = self.module.TokenProcessor(cfg, cache, queue, connector)
        rm = _DummyResourceManager(cfg.max_num_seqs)
        processor.set_resource_manager(rm)
        return processor, rm, cache, queue, connector

    def test_init_with_zmq_and_speculative_buffers(self):
        envs = sys.modules["fastdeploy.envs"]
        envs.FD_USE_GET_SAVE_OUTPUT_V1 = True
        processor, _, _, _, _ = self._create_processor(speculative_method="mtp", enable_logprob=False)
        envs.FD_USE_GET_SAVE_OUTPUT_V1 = False
        self.assertIsNotNone(getattr(processor, "zmq_server", None))
        self.assertEqual(
            processor.output_tokens.array.shape[0],
            self.module.SPECULATE_MAX_BSZ * self.module.MAX_DRAFT_TOKENS + self.module.SPECULATE_MAX_BSZ + 2,
        )

    def test_cleanup_resources_and_run_paths(self):
        processor, rm, _, _, _ = self._create_processor()
        envs = sys.modules["fastdeploy.envs"]
        envs.FD_USE_GET_SAVE_OUTPUT_V1 = True

        created = {}
        original_threading = getattr(self.module, "threading", None)

        class _FakeThread:
            def __init__(self, target):
                created["target"] = target
                self.daemon = False

            def start(self):
                created["started"] = True

        self.module.threading = types.SimpleNamespace(Thread=_FakeThread)
        processor.run()
        self.assertTrue(created["started"])
        self.assertEqual(created["target"], processor.process_sampling_results_use_zmq)

        cleared = []
        processor.prefill_time_signal = types.SimpleNamespace(clear=lambda: cleared.append("signal"))
        processor.executor = types.SimpleNamespace(shutdown=lambda wait=False: cleared.append("executor"))
        processor._cleanup_resources()
        envs.FD_USE_GET_SAVE_OUTPUT_V1 = False
        if original_threading is not None:
            self.module.threading = original_threading
        self.assertEqual(cleared, ["signal", "executor"])

    def test_run_prevents_duplicate_workers(self):
        processor, _, _, _, _ = self._create_processor()
        envs = sys.modules["fastdeploy.envs"]
        envs.FD_USE_GET_SAVE_OUTPUT_V1 = False
        created = {}
        original_threading = getattr(self.module, "threading", None)

        class _FakeThread:
            def __init__(self, target):
                created["target"] = target
                self.daemon = False

            def start(self):
                created["started"] = True

        self.module.threading = types.SimpleNamespace(Thread=_FakeThread)
        processor.run()
        with self.assertRaises(Exception):
            processor.run()
        self.module.threading = original_threading
        self.assertEqual(created["target"], processor.process_sampling_results)

    def test_process_batch_output_handles_logprobs(self):
        processor, rm, cache, _, _ = self._create_processor(enable_logprob=True)
        task = _make_request(request_id="req-a")
        rm.tasks_list[0] = task
        rm.requests[task.request_id] = types.SimpleNamespace(idx=0)
        rm.req_dict[task.request_id] = task

        tensor = processor.output_tokens
        tensor.array[1, 0] = 1
        sequence = np.arange(self.module.K + 1)
        tensor.array[2 : 2 + len(sequence), 0] = sequence
        processor.output_scores.array[: len(sequence), 0] = np.linspace(0.5, 1.5, len(sequence))
        processor.output_ranks.array[0] = 3

        processor._process_batch_output()

        self.assertEqual(len(cache.results[-1]), 1)
        result = cache.results[-1][0]
        self.assertTrue(result.finished)
        self.assertEqual(result.outputs.token_ids, [0])
        self.assertEqual(task.output_token_ids, [0])
        self.assertTrue(rm.stop_flags[0])
        self.assertIsNone(rm.tasks_list[0])

    def test_process_batch_output_use_zmq_pooling(self):
        processor, rm, _, _, _ = self._create_processor()
        task = _make_request(request_id="req-b", pooling_params=True)
        rm.tasks_list[0] = task
        stream = types.SimpleNamespace(
            batch_id=0,
            tokens=np.array([1, 2, 3], dtype=np.int64),
            pooler_output=np.array([0.25, 0.75], dtype=np.float32),
        )
        results = processor._process_batch_output_use_zmq([stream])
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].finished)
        self.assertEqual(results[0].outputs.data, [0.25, 0.75])

    def test_process_batch_output_use_zmq_normal_path(self):
        processor, rm, _, _, _ = self._create_processor()
        task = _make_request(request_id="req-z", multimodal_inputs={"num_input_image_tokens": 1}, eos_token_ids=[6])
        rm.tasks_list[0] = task
        rm.req_dict[task.request_id] = task
        stream = types.SimpleNamespace(
            batch_id=0,
            tokens=np.array([5, 6], dtype=np.int64),
            pooler_output=None,
        )
        results = processor._process_batch_output_use_zmq([stream])
        self.assertTrue(results)
        self.assertTrue(results[0].finished)
        self.assertEqual(results[0].outputs.token_ids, [5, 6])

    def test_process_batch_output_use_zmq_negative_tokens_reschedule(self):
        processor, rm, _, _, _ = self._create_processor()
        envs = sys.modules["fastdeploy.envs"]
        envs.ENABLE_V1_KVCACHE_SCHEDULER = True
        rm.to_be_rescheduled_request_id_set = {"req-neg"}
        rm.requests["req-neg"] = types.SimpleNamespace(idx=0)
        task = _make_request(request_id="req-neg")
        rm.tasks_list[0] = task
        stream = types.SimpleNamespace(batch_id=0, tokens=np.array([1, -1], dtype=np.int64), pooler_output=None)
        results = processor._process_batch_output_use_zmq([stream])
        envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        self.assertFalse(results)
        self.assertIn("reschedule-req-neg", rm.recycled)

    def test_postprocess_merges_draft_results(self):
        processor, _, cache, _, _ = self._create_processor(speculative_method="ngram", enable_logprob=True)
        unfinished = _RequestOutput("r1", _CompletionOutput(0, 0, [], []))
        finished = _RequestOutput("r2", _CompletionOutput(0, 0, [], []), finished=True)
        processor.postprocess([unfinished], mtype=3)
        self.assertEqual(processor._batch_result_buffer, [unfinished])

        processor.postprocess([finished], mtype=4)
        self.assertEqual(cache.results[-1][0].request_id, "r1")
        self.assertIsNone(processor._batch_result_buffer)

    def test_postprocess_finished_and_error_paths(self):
        processor, _, cache, _, _ = self._create_processor(speculative_method="ngram", enable_logprob=True)
        finished = _RequestOutput("r3", _CompletionOutput(0, 0, [], []), finished=True)
        processor.postprocess([finished], mtype=3)
        self.assertEqual(cache.results[-1], [finished])

        class _ExplodingCache(_DummyCache):
            def put_results(self, batch_result):
                raise RuntimeError("explode")

        processor.cached_generated_tokens = _ExplodingCache()
        processor.postprocess([finished], mtype=0)
        self.assertIn("explode", str(self.utils_module.llm_logger.messages[-1][1]))

    def test_recycle_resources_prefill_and_decode(self):
        processor, rm, _, queue, connector = self._create_processor()
        task = _make_request(request_id="req-c", disaggregate_info={"role": "prefill"})
        rm.tasks_list[0] = task
        rm.req_dict[task.request_id] = task
        queue._finished = [(task.request_id, "finished")]
        result = _RequestOutput(task.request_id, _CompletionOutput(0, 0, [], []))
        processor._recycle_resources(task.request_id, 0, task, result, is_prefill=True)
        self.assertTrue(connector.calls)
        self.assertTrue(processor.prefill_result_status)

        task_decode = _make_request(request_id="req-d")
        rm.tasks_list[1] = task_decode
        rm.req_dict[task_decode.request_id] = task_decode
        processor.tokens_counter[task_decode.request_id] = 1
        processor._recycle_resources(task_decode.request_id, 1, task_decode, result, is_prefill=False)
        self.assertNotIn(task_decode.request_id, rm.req_dict)
        self.assertNotIn(task_decode.request_id, processor.tokens_counter)

    def test_reschedule_helpers_handle_requests(self):
        processor, rm, _, _, _ = self._create_processor()
        envs = sys.modules["fastdeploy.envs"]
        envs.ENABLE_V1_KVCACHE_SCHEDULER = True
        rm.to_be_rescheduled_request_id_set = {"req-r"}
        rm.requests["req-r"] = types.SimpleNamespace(idx=2)
        data = types.SimpleNamespace(batch_id=0)
        processor._reschedule_preempt_task_use_zmq([data])
        processor._reschedule_preempt_task(batch_size=1)
        envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        self.assertIn("reschedule-req-r", rm.recycled)

    def test_process_per_token_handles_recovery_stop(self):
        processor, rm, _, _, _ = self._create_processor(speculative_method="ngram", enable_logprob=True)
        task = _make_request(request_id="req-stop")
        rm.tasks_list[0] = task
        rm.req_dict[task.request_id] = task
        result = _RequestOutput(task.request_id, _CompletionOutput(0, 0, [], []))
        processor.number_of_output_tokens = 1
        processor.total_step = 1
        processor._process_per_token(task, 0, np.array([self.module.RECOVERY_STOP_SIGNAL]), result, is_prefill=False)
        self.assertTrue(result.finished)
        self.assertEqual(result.error_msg, "Recover is not supported, the result is incomplete!")

    def test_prefill_metrics_processed(self):
        processor, _, _, _, _ = self._create_processor()
        processor.prefill_time_signal.value[0] = 0.5

        executed = []

        class _ImmediateExecutor:
            def submit(self, func):
                executed.append("run")
                func()

            def shutdown(self, wait=False):  # pragma: no cover - cleanup helper
                pass

        processor.executor = _ImmediateExecutor()
        processor._process_prefill_metrics()
        self.assertIn(("observe", 0.5), self.metrics.request_prefill_time.values)
        self.assertEqual(executed, ["run"])

    def test_prefill_metrics_handles_errors(self):
        processor, _, _, _, _ = self._create_processor()
        processor.prefill_time_signal = types.SimpleNamespace(value=None, clear=lambda: None)

        class _ImmediateExecutor:
            def submit(self, func):
                func()

            def shutdown(self, wait=False):  # pragma: no cover - cleanup helper
                pass

        processor.executor = _ImmediateExecutor()
        processor._process_prefill_metrics()
        self.assertTrue(self.utils_module.llm_logger.messages)

    def test_speculative_metrics_and_status(self):
        processor, _, _, _, _ = self._create_processor(speculative_method="ngram", enable_logprob=True)
        processor.number_of_output_tokens = 10
        processor.total_step = 5
        processor.speculative_stats_step = 0
        processor._compute_speculative_status()
        processor._record_speculative_decoding_mertics([2, 3])
        self.assertTrue(self.utils_module.spec_logger.messages)
        self.assertTrue(self.metrics.spec_decode_draft_single_head_acceptance_rate)

    def test_speculative_mtp_metrics(self):
        processor, _, _, _, _ = self._create_processor(speculative_method="mtp", enable_logprob=True)
        processor._record_speculative_decoding_mertics([3, 2, 0])
        self.assertTrue(self.metrics.spec_decode_efficiency.values)

    def test_compute_speculative_status_mtp_tracks_heads(self):
        processor, _, _, _, _ = self._create_processor(speculative_method="mtp", enable_logprob=True)
        processor.number_of_output_tokens = 10
        processor.total_step = 5
        processor.speculative_stats_step = 0
        processor.num_rest_requests_per_head = [2, 1] + [0] * (self.module.MAX_DRAFT_TOKENS - 2)
        processor.num_accept_requests_per_head = [1, 0] + [0] * (self.module.MAX_DRAFT_TOKENS - 2)
        processor._compute_speculative_status()
        self.assertTrue(any(msg for msg in self.utils_module.spec_logger.messages if "Single head" in msg[1]))

    def test_process_batch_output_speculative_draft_flow(self):
        processor, rm, cache, _, _ = self._create_processor(speculative_method="ngram", enable_logprob=True)
        task = _make_request(request_id="req-spec", messages=None)
        rm.tasks_list[0] = task
        rm.req_dict[task.request_id] = task
        rm.requests[task.request_id] = types.SimpleNamespace(idx=0)
        processor._batch_result_buffer = [
            _RequestOutput(task.request_id, _CompletionOutput(0, 0, [], []), finished=False)
        ]
        tensor = processor.output_tokens
        tensor.array[1, 0] = 4  # draft type
        tensor.array[2, 0] = 1  # batch size
        tensor.array[3, 0] = 1  # accept num
        start = 3 + self.module.MAX_BSZ
        tensor.array[start, 0] = 9
        processor.output_scores.array[0, 0] = 0.25
        processor.output_ranks.array[0] = 1
        processor._process_batch_output()
        result = cache.results[-1][0]
        self.assertEqual(result.outputs.draft_top_logprobs.logprob_token_ids[0][0], 9)

    def test_process_batch_output_top_logprobs_extend_and_finish(self):
        processor, rm, cache, _, _ = self._create_processor(speculative_method="ngram", enable_logprob=True)
        task = _make_request(request_id="req-top")
        rm.tasks_list[0] = task
        rm.req_dict[task.request_id] = task
        rm.requests[task.request_id] = types.SimpleNamespace(idx=0)
        tensor = processor.output_tokens
        tensor.array[1, 0] = 3  # target type
        tensor.array[2, 0] = 1
        tensor.array[3, 0] = 2
        start = 3 + self.module.MAX_BSZ
        stride = self.module.K + 1
        tensor.array[start + 0 * stride, 0] = 1
        tensor.array[start + 1 * stride, 0] = task.eos_token_ids[0]
        processor.output_scores.array[: 2 * (self.module.K + 1), 0] = np.arange(2 * (self.module.K + 1))
        processor.output_ranks.array[0 : 2 * self.module.MAX_DRAFT_TOKENS] = 0
        processor._process_batch_output()
        result = cache.results[-1][0]
        self.assertTrue(result.finished)
        self.assertGreater(len(result.outputs.top_logprobs.logprob_token_ids), 1)

    def test_process_batch_output_speculative_without_logprobs(self):
        processor, rm, cache, _, _ = self._create_processor(speculative_method="mtp", enable_logprob=False)
        envs = sys.modules["fastdeploy.envs"]
        envs.ENABLE_V1_KVCACHE_SCHEDULER = True
        rm.to_be_rescheduled_request_id_set = {"req-a"}
        tasks = [
            _make_request(request_id="req-a", prefill_chunk_info=[1, 2]),
            _make_request(request_id="req-b", messages=None, eos_token_ids=[8]),
        ]
        for idx, task in enumerate(tasks):
            rm.tasks_list[idx] = task
            rm.req_dict[task.request_id] = task
            rm.requests[task.request_id] = types.SimpleNamespace(idx=idx)
        processor.tokens_counter[tasks[1].request_id] = 1
        tensor = processor.output_tokens
        tensor.array[1] = 2  # batch size
        tensor.array[2] = -3
        tensor.array[3] = 2
        base = 2 + self.module.SPECULATE_MAX_BSZ
        second_start = base + self.module.MAX_DRAFT_TOKENS
        tensor.array[second_start] = 7
        tensor.array[second_start + 1] = 8
        processor._process_batch_output()
        envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        self.assertTrue(cache.results[-1])
        self.assertEqual(cache.results[-1][0].outputs.token_ids, [7, 8])

    def test_clear_data_completes_tasks(self):
        processor, rm, _, _, _ = self._create_processor()
        task = _make_request(request_id="req-clear")
        rm.tasks_list[0] = task
        rm.req_dict[task.request_id] = task
        processor.tokens_counter[task.request_id] = 0
        for idx in range(1, len(rm.stop_flags)):
            rm.stop_flags[idx] = True
        processor.clear_data()
        self.assertTrue(rm.recycled)
        self.assertTrue(rm.stop_flags[0])

    def test_warmup_token_processor_process_loop(self):
        module = self.module
        cfg = _build_cfg()
        cache = _DummyCache()
        queue = _DummyQueue()
        connector = _DummyConnector()
        warm = module.WarmUpTokenProcessor.__new__(module.WarmUpTokenProcessor)
        module.TokenProcessor.__init__(warm, cfg, cache, queue, connector)
        warm._is_running = True
        warm._is_blocking = False

        def _stop_get_output(*_args, **_kwargs):
            warm.output_tokens.__setitem__((0, 0), -2)
            warm._is_running = False

        sys.modules["fastdeploy.model_executor.ops.gpu"] = types.SimpleNamespace(
            get_output=_stop_get_output,
            speculate_get_output=_stop_get_output,
        )
        warm.process_sampling_results()
        warm.worker = types.SimpleNamespace(join=lambda: None)
        warm.stop()


if __name__ == "__main__":
    unittest.main()
