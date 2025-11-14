"""Unit tests for :mod:`fastdeploy.engine.common_engine`."""

from __future__ import annotations

import importlib
import sys
import threading
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _FakeTensor:
    def __init__(self, data: Any):
        self.data = np.array(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def __iter__(self):
        return iter(self.data)

    def __array__(self, dtype=None):
        if dtype:
            return self.data.astype(dtype)
        return self.data

    def __eq__(self, other):

        if isinstance(other, _FakeTensor):
            other = other.data
        return _FakeTensor(self.data == other)

    def cast(self, dtype: str) -> "_FakeTensor":

        return _FakeTensor(self.data.astype(dtype))

    def numpy(self):
        return self.data.copy()

    def reshape(self, shape):
        return _FakeTensor(self.data.reshape(shape))


def _install_stub_modules() -> None:
    if getattr(_install_stub_modules, "_installed", False):
        return

    package = types.ModuleType("fastdeploy")
    package.__path__ = [str(PROJECT_ROOT / "fastdeploy")]
    sys.modules.setdefault("fastdeploy", package)

    paddle_mod = types.ModuleType("paddle")

    def _to_tensor(data, dtype=None):
        if dtype:
            return _FakeTensor(np.array(data, dtype=dtype))
        return _FakeTensor(np.array(data))

    def _full(shape, fill_value=0, dtype=None):
        return _FakeTensor(np.full(shape, fill_value, dtype=dtype))

    def _cumsum(tensor, dtype=None):
        arr = np.cumsum(np.array(tensor), dtype=dtype)
        return _FakeTensor(arr)

    paddle_mod.to_tensor = _to_tensor  # type: ignore[attr-defined]
    paddle_mod.full = _full  # type: ignore[attr-defined]
    paddle_mod.cumsum = _cumsum  # type: ignore[attr-defined]
    paddle_mod.distributed = types.SimpleNamespace()
    sys.modules["paddle"] = paddle_mod

    requests_mod = types.SimpleNamespace()

    class _Resp:
        def __init__(self, ok=True):
            self.ok = ok
            self.status_code = 200
            self.text = "ok"

    def _post(*_args, **_kwargs):
        return _Resp()

    requests_mod.post = _post

    class _ReqError(Exception):
        pass

    requests_mod.exceptions = types.SimpleNamespace(RequestException=_ReqError)
    sys.modules["requests"] = requests_mod  # type: ignore[assignment]

    zmq_mod = types.SimpleNamespace(PULL=1, ROUTER=2)
    sys.modules["zmq"] = zmq_mod  # type: ignore[assignment]

    utils_mod = types.ModuleType("fastdeploy.utils")

    class _Logger:
        def __init__(self):
            self.messages: List[str] = []

        def info(self, msg):
            self.messages.append(str(msg))

        debug = warning = error = exception = info

    class EngineError(Exception):
        def __init__(self, msg, error_code=400):
            super().__init__(msg)
            self.error_code = error_code

    utils_mod.EngineError = EngineError
    utils_mod.check_download_links = lambda *args, **kwargs: True
    utils_mod.envs = types.SimpleNamespace(
        FD_ENABLE_CACHE_TASK="0",
        ENABLE_V1_KVCACHE_SCHEDULER=False,
        FD_ENGINE_TASK_QUEUE_WITH_SHM=False,
        FD_ENABLE_INTERNAL_ADAPTER=False,
        FD_OFFLINE_PERF_TEST_FOR_PD=False,
        FD_ZMQ_RECV_REQUEST_SERVER_PORT=6000,
        FD_ZMQ_SEND_RESPONSE_SERVER_PORT=6001,
        FD_ENABLE_RETURN_TEXT=True,
    )
    utils_mod.get_logger = lambda *_args, **_kwargs: _Logger()
    utils_mod.init_bos_client = lambda *_args, **_kwargs: object()
    utils_mod.llm_logger = _Logger()
    sys.modules["fastdeploy.utils"] = utils_mod

    fd_request_mod = types.ModuleType("fastdeploy.engine.request")

    class RequestOutput:
        def __init__(self, request_id, outputs=None, finished=False, error_code=200, error_msg=""):
            self.request_id = request_id
            self.finished = finished
            self.error_code = error_code
            self.error_msg = error_msg
            if outputs is None:
                outputs = types.SimpleNamespace(token_ids=[], decode_type=0, text="")
            self.outputs = outputs

    class Request:
        def __init__(self, request_id, prompt_token_ids_len, multimodal_inputs=None):
            self.request_id = request_id
            self.prompt_token_ids_len = prompt_token_ids_len
            self.prompt_token_ids = [[1]]
            self.multimodal_inputs = multimodal_inputs or {}
            self.outputs = types.SimpleNamespace(token_ids=[[]], decode_type=0, text="")
            self.num_cached_tokens = 0
            self.idx = 0
            self.prefill_chunk_info = None
            self.max_tokens = 0
            self.min_tokens = 0
            self.disaggregate_info = None

        def set(self, key, value):
            setattr(self, key, value)

    class RequestType:
        PREFILL = 0
        DECODE = 1

    fd_request_mod.Request = Request
    fd_request_mod.RequestOutput = RequestOutput
    fd_request_mod.RequestType = RequestType
    sys.modules["fastdeploy.engine.request"] = fd_request_mod

    rm_mod = types.ModuleType("fastdeploy.engine.resource_manager")

    class _ResourceManager:
        def __init__(self, *_args, **_kwargs):
            self.req_dict: Dict[str, int] = {}
            self.tasks_list: List[Any] = []
            self.stop_flags: List[bool] = []
            self.requests: Dict[str, Any] = {}
            self.real_bsz = 1
            self.recycled: List[str] = []

        def available_batch(self):
            return 2

        def available_block_num(self):
            return 4

        def check_and_free_block_tables(self):
            return True

        def _recycle_block_tables(self, task):
            self.recycled.append(task.request_id)

        def _free_blocks(self, task):
            self.recycled.append(task.request_id)

        def is_resource_sufficient(self, *_args):
            return True

        def preallocate_resource_in_d(self, *_args):
            return True

        def preallocate_resource_in_p(self, *_args):
            return True

        def prerelease_resource(self, *_args):
            return True

        def add_request(self, task):
            self.requests[task.request_id] = task

        def add_request_in_p(self, tasks):
            for task in tasks:
                self.add_request(task)

        def allocate_resources_for_new_tasks(self, tasks):
            for task in tasks:
                idx = len(self.tasks_list)
                task.idx = idx
                self.req_dict[task.request_id] = idx
                self.tasks_list.append(task)
                self.stop_flags.append(False)
            return tasks

    rm_mod.ResourceManager = _ResourceManager
    sys.modules["fastdeploy.engine.resource_manager"] = rm_mod

    rm_v1_mod = types.ModuleType("fastdeploy.engine.sched.resource_manager_v1")
    rm_v1_mod.ResourceManagerV1 = _ResourceManager
    sys.modules["fastdeploy.engine.sched.resource_manager_v1"] = rm_v1_mod

    preprocess_mod = types.ModuleType("fastdeploy.input.preprocess")

    class _InputPreprocessor:
        def __init__(self, *args, **kwargs):
            self.args = args

        def create_processor(self):
            class _Processor:
                image_patch_id = 7
                decode_status: Dict[str, List[int]] = {}

                def ids2tokens(self, token_ids, req_id):
                    text = "|".join(str(t) for t in token_ids)
                    cum = list(range(len(token_ids)))
                    self.decode_status.setdefault(req_id, [0, len(cum)])
                    return text, cum, None

            return _Processor()

    preprocess_mod.InputPreprocessor = _InputPreprocessor
    sys.modules["fastdeploy.input.preprocess"] = preprocess_mod

    inter_mod = types.ModuleType("fastdeploy.inter_communicator")

    class _BaseQueue:
        def __init__(self, *args, **kwargs):
            self.address = kwargs.get("address") or args[0]
            self.is_server = kwargs.get("is_server", False)

        def get_server_port(self):
            return 1234

        def cleanup(self):
            self.cleaned = True

    class EngineWorkerQueue(_BaseQueue):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tasks: List[Any] = []
            self.cache_infos: List[Any] = []
            self.disaggregated: List[Any] = []
            self.finished_ids: List[str] = []

        def put_tasks(self, payload):
            self.tasks.append(payload)

        def num_tasks(self):
            return len(self.tasks)

        def num_cache_infos(self):
            return len(self.cache_infos)

        def disaggregate_queue_empty(self):
            return not self.disaggregated

        def get_disaggregated_tasks(self):
            data = self.disaggregated
            self.disaggregated = []
            return data

        def get_finished_add_cache_task_req(self):
            data = self.finished_ids
            self.finished_ids = []
            return data

    class EngineCacheQueue(_BaseQueue):
        pass

    class IPCSignal:
        def __init__(self, name, array, dtype, suffix, create):
            self.name = name
            self.value = array

        def clear(self):
            self.value[:] = 0

    class ZmqIpcServer:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def close(self):
            self.closed = True

    class ZmqTcpServer(ZmqIpcServer):
        def send_response(self, *_args):
            return True

    inter_mod.EngineCacheQueue = EngineCacheQueue
    inter_mod.EngineWorkerQueue = EngineWorkerQueue
    inter_mod.IPCSignal = IPCSignal
    inter_mod.ZmqIpcServer = ZmqIpcServer
    inter_mod.ZmqTcpServer = ZmqTcpServer
    sys.modules["fastdeploy.inter_communicator"] = inter_mod

    metrics_mod = types.ModuleType("fastdeploy.metrics.metrics")

    class _Counter:
        def __init__(self):
            self.value = 0

        def inc(self, amount=1):
            self.value += amount

        def dec(self, amount=1):
            self.value -= amount

    class _Metrics:
        num_requests_waiting = _Counter()
        num_requests_running = _Counter()
        requests_number = _Counter()

    metrics_mod.main_process_metrics = _Metrics()
    sys.modules["fastdeploy.metrics.metrics"] = metrics_mod

    trace_mod = types.ModuleType("fastdeploy.metrics.trace_util")

    class _Span:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    trace_mod.start_span = lambda *args, **kwargs: _Span()
    trace_mod.start_span_request = lambda *args, **kwargs: None
    sys.modules["fastdeploy.metrics.trace_util"] = trace_mod

    guided_mod = types.ModuleType("fastdeploy.model_executor.guided_decoding")
    guided_mod.schema_checker = lambda *args, **kwargs: lambda text: text
    sys.modules["fastdeploy.model_executor.guided_decoding"] = guided_mod

    token_mod = types.ModuleType("fastdeploy.plugins.token_processor")

    class _TokenProcessor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.tasks_queue = None
            self.tokens_counter: Dict[str, int] = {}
            self.number_of_tasks = 0
            self.number_of_input_tokens = 0

        def set_resource_manager(self, rm):
            self.rm = rm

        def run(self):
            self.started = True

    token_mod.load_token_processor_plugins = lambda: _TokenProcessor
    sys.modules["fastdeploy.plugins.token_processor"] = token_mod

    router_utils = types.ModuleType("fastdeploy.router.utils")
    router_utils.check_service_health = lambda *_args, **_kwargs: True
    sys.modules["fastdeploy.router.utils"] = router_utils

    internal_adapter = types.ModuleType("fastdeploy.splitwise.internal_adapter_utils")
    internal_adapter.InternalAdapter = lambda *args, **kwargs: object()
    sys.modules["fastdeploy.splitwise.internal_adapter_utils"] = internal_adapter

    split_connector = types.ModuleType("fastdeploy.splitwise.splitwise_connector")

    class _SplitwiseConnector:
        def __init__(self, *args, **kwargs):
            self.current_request_ids: List[str] = []

        def has_splitwise_tasks(self):
            return False

        def send_splitwise_tasks(self, *_args, **_kwargs):
            return True

        def send_cache_infos(self, *_args, **_kwargs):
            return True

        def check_decode_allocated(self, *_args, **_kwargs):
            return True, ""

    split_connector.SplitwiseConnector = _SplitwiseConnector
    sys.modules["fastdeploy.splitwise.splitwise_connector"] = split_connector

    ops_mod = types.ModuleType("fastdeploy.model_executor.ops.gpu")

    def _get_mm_split_fuse(*_args, **_kwargs):
        return _FakeTensor([1]), _FakeTensor([2])

    ops_mod.get_mm_split_fuse = _get_mm_split_fuse
    sys.modules["fastdeploy.model_executor.ops.gpu"] = ops_mod

    output_mod = types.ModuleType("fastdeploy.output.token_processor")
    output_mod.TokenProcessor = token_mod.load_token_processor_plugins()
    sys.modules["fastdeploy.output.token_processor"] = output_mod

    _install_stub_modules._installed = True


def _import_common_engine():
    _install_stub_modules()
    if "fastdeploy.engine.common_engine" in sys.modules:
        return sys.modules["fastdeploy.engine.common_engine"]
    return importlib.import_module("fastdeploy.engine.common_engine")


class _DummyScheduler:
    def __init__(self):
        self.requests: List[Any] = []
        self.results: Dict[str, List[Any]] = {}

    def get_requests(self, **_kwargs):
        data = self.requests
        self.requests = []
        return data

    def put_requests(self, reqs):
        self.requests.extend(reqs)
        return []

    def put_results(self, reqs):
        for req in reqs:
            self.results.setdefault(req.request_id, []).append(req)

    def get_results(self):
        data = {rid: items for rid, items in self.results.items()}
        self.results.clear()
        return data

    def has_request(self, req_id):
        return any(req.request_id == req_id for req in self.requests)


def _build_config():
    class _Parallel:
        tensor_parallel_size = 1
        data_parallel_size = 1
        local_data_parallel_id = 0
        engine_worker_queue_port = ["5557"]
        enable_expert_parallel = False

    class _Cache:
        enable_prefix_caching = True
        cache_queue_port = "5560"
        block_size = 8
        enc_dec_block_num = 1
        enable_chunked_prefill = True
        max_block_num_per_seq = 4

    class _Scheduler:
        splitwise_role = "mixed"
        max_num_seqs = 2
        max_num_batched_tokens = 32
        name = "default"

        def scheduler(self):
            return _DummyScheduler()

    class _Structured:
        guided_decoding_backend = "off"
        disable_any_whitespace = False
        reasoning_parser = None

    class _Router:
        router = "http://router"
        api_server_host = "localhost"
        api_server_port = 1234

    class _Speculative:
        method = "mtp"

    cfg = types.SimpleNamespace()
    cfg.parallel_config = _Parallel()
    cfg.cache_config = _Cache()
    cfg.scheduler_config = _Scheduler()
    cfg.model_config = types.SimpleNamespace(enable_mm=False)
    cfg.structured_outputs_config = _Structured()
    cfg.limit_mm_per_prompt = None
    cfg.mm_processor_kwargs = {}
    cfg.tool_parser = None
    cfg.max_num_partial_prefills = 2
    cfg.master_ip = "127.0.0.1"
    cfg.host_ip = "127.0.0.1"
    cfg.worker_num_per_node = 1
    cfg.node_rank = 0
    cfg.cache_config.cache_queue_port = "6003"
    cfg.parallel_config.engine_worker_queue_port = ["7001"]
    cfg.max_prefill_batch = 2
    cfg.splitwise_version = "v0"
    cfg.cache_config.enable_prefix_caching = True
    cfg.cache_config.cache_queue_port = "6003"
    cfg.cache_config.enable_chunked_prefill = True
    cfg.router_config = _Router()
    cfg.register_info = {"name": "engine"}
    cfg.speculative_config = _Speculative()
    return cfg


class EngineServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _import_common_engine()

    def setUp(self):
        self.cfg = _build_config()
        self.service = self.module.EngineService(self.cfg, start_queue=False)
        self.service.create_data_processor()

    def test_update_requests_chunk_size_assigns_chunks(self):
        Request = sys.modules["fastdeploy.engine.request"].Request
        requests = [Request("r1", 16), Request("r2", 8)]
        for req in requests:
            req.prompt_token_ids_len = 16 if req.request_id == "r1" else 8

        self.service.update_requests_chunk_size(requests)
        chunks = [req.prefill_chunk_info for req in requests]
        self.assertTrue(all(chunks))

    def test_update_mm_requests_chunk_size_handles_images(self):
        Request = sys.modules["fastdeploy.engine.request"].Request
        inputs = {
            "input_ids": np.array([1, 7, 2, 7, 3]),
            "token_type_ids": np.zeros(5, dtype="int32"),
            "images": np.array([[1], [2]]),
            "image_type_ids": np.array([[1, 0, 0], [1, 0, 0]]),
            "grid_thw": np.array([[1, 1, 1], [2, 1, 1]]),
            "position_ids": np.arange(5),
        }
        req = Request("mm", 8, multimodal_inputs=inputs)
        self.service.update_mm_requests_chunk_size([req])
        self.assertIsNotNone(req.prefill_chunk_info)

    def test_decode_token_returns_delta(self):
        req_id = "decode"
        self.service.data_processor.decode_status[req_id] = [0, 2]
        delta, tokens = self.service._decode_token([1, 2, 3], req_id, is_end=True)
        self.assertIn("|", delta)
        self.assertEqual(tokens, [0, 1])

    def test_has_features_info(self):
        Request = sys.modules["fastdeploy.engine.request"].Request
        req = Request("feat", 1)
        req.multimodal_inputs = {
            "video_feature_urls": ["v"],
            "image_feature_urls": [],
            "audio_feature_urls": [],
        }
        self.assertTrue(self.service._has_features_info(req))

    def test_register_to_router_spawns_thread(self):
        calls = {}

        def fake_thread(target, daemon):
            calls["target"] = target
            thread = types.SimpleNamespace(start=lambda: calls.setdefault("start", True))
            return thread

        self.cfg.router_config.router = "http://router"
        with mock.patch.object(threading, "Thread", side_effect=fake_thread):
            self.service._register_to_router()
        self.assertIn("target", calls)

    def test_exit_sub_services_cleans_signals(self):
        self.service.exist_task_signal.value[0] = 5
        self.service.exist_swapped_task_signal.value[0] = 2
        self.service.start_worker_queue_service(start_queue=False)
        self.service.send_response_server = types.SimpleNamespace(close=lambda: setattr(self, "closed", True))
        self.service.recv_request_server = types.SimpleNamespace(close=lambda: None)
        self.service._exit_sub_services()
        self.assertEqual(self.service.exist_task_signal.value[0], 0)

    def test_start_worker_queue_service_with_server(self):
        self.service.start_worker_queue_service(start_queue=True)
        self.assertIsNotNone(self.service.engine_worker_queue)
        self.assertIsNotNone(self.service.engine_worker_queue_server)

    def test_insert_tasks_handles_allocated_and_regular(self):
        Request = sys.modules["fastdeploy.engine.request"].Request
        RequestOutput = sys.modules["fastdeploy.engine.request"].RequestOutput

        task = Request("alloc", 4)
        self.service.resource_manager.req_dict[task.request_id] = 0
        self.service.resource_manager.tasks_list = [
            types.SimpleNamespace(prompt_token_ids=[[0, 0]], num_cached_tokens=0, request_id=task.request_id)
        ]
        self.service.resource_manager.stop_flags = [False]
        output = RequestOutput(
            request_id=task.request_id,
            outputs=types.SimpleNamespace(token_ids=[[1, 2]], draft_token_ids=None),
        )
        output.num_cached_tokens = 0
        self.assertTrue(self.service.insert_tasks([output], current_id=0, allocated=True))

        normal_task = Request("normal", 8)
        normal_task.prompt_token_ids = [[1, 2, 3]]
        normal_task.prompt_token_ids_len = 4
        self.service.resource_manager.stop_flags = [True, True]
        self.service.scheduler.requests.append(normal_task)
        inserted = self.service.insert_tasks([normal_task])
        self.assertTrue(inserted)
        self.assertGreater(len(self.service.engine_worker_queue.tasks), 0)

    def test_task_completion_helpers(self):
        self.service.resource_manager.stop_flags = [True, False]
        self.assertTrue(self.service.task_is_finished(0))
        self.assertFalse(self.service.all_tasks_finished())


if __name__ == "__main__":
    unittest.main()
