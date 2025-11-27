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

import pathlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_required_stubs():
    if "paddle" not in sys.modules:
        paddle_mod = types.ModuleType("paddle")
        dist_mod = types.ModuleType("paddle.distributed")
        collective_mod = types.SimpleNamespace(_set_custom_gid=lambda *_: None)
        dist_mod.collective = collective_mod
        dist_mod.new_group = lambda *_, **__: None
        paddle_mod.distributed = dist_mod

        class _FakeTensor:
            def __init__(self, data):
                self._array = np.array(data)

            def numpy(self):
                return np.array(self._array)

            def __getitem__(self, item):
                return self._array.__getitem__(item)

            def __eq__(self, other):
                return self._array == other

            def any(self):
                return self._array.any()

        paddle_mod.Tensor = _FakeTensor
        paddle_mod.is_compiled_with_rocm = lambda: False
        paddle_mod.is_compiled_with_cuda = lambda: False
        paddle_mod.is_compiled_with_xpu = lambda: False
        paddle_mod.is_compiled_with_custom_device = lambda *_: False
        paddle_mod.to_tensor = lambda data, dtype=None: _FakeTensor(data)
        paddle_mod.sum = lambda value: np.array(value).sum()
        sys.modules["paddle"] = paddle_mod
        sys.modules["paddle.distributed"] = dist_mod

    if "paddleformers" not in sys.modules:
        paddleformers_mod = types.ModuleType("paddleformers")
        sys.modules["paddleformers"] = paddleformers_mod

        utils_mod = types.ModuleType("paddleformers.utils")
        sys.modules["paddleformers.utils"] = utils_mod
        paddleformers_mod.utils = utils_mod

        log_mod = types.ModuleType("paddleformers.utils.log")
        log_mod.logger = types.SimpleNamespace(logger=types.SimpleNamespace(setLevel=lambda *_: None))
        sys.modules["paddleformers.utils.log"] = log_mod
        utils_mod.log = log_mod

        transformers_mod = types.ModuleType("paddleformers.transformers")
        sys.modules["paddleformers.transformers"] = transformers_mod

        config_utils_mod = types.ModuleType("paddleformers.transformers.configuration_utils")

        class _PretrainedConfig:
            pass

        config_utils_mod.PretrainedConfig = _PretrainedConfig
        sys.modules["paddleformers.transformers.configuration_utils"] = config_utils_mod
        transformers_mod.configuration_utils = config_utils_mod


_install_required_stubs()

import fastdeploy.engine.sched.resource_manager_v1 as rm_v1
from fastdeploy.engine.request import ImagePosition, Request, RequestStatus, RequestType


class _MetricRecorder:
    """Test double used to record metric values in resource manager tests."""

    def __init__(self):
        self.value = 0
        self.calls = []

    def set(self, value):
        self.value = value
        self.calls.append(("set", value))

    def inc(self, value):
        self.value += value
        self.calls.append(("inc", value))


class _FakePrefixCacheManager:
    """Lightweight cache manager stub tracking GPU/CPU block usage."""

    def __init__(self, config, tensor_parallel_size, splitwise_role, local_data_parallel_id):
        cache_cfg = config.cache_config
        total_blocks = getattr(cache_cfg, "initial_gpu_blocks", 64)
        self.num_gpu_blocks = total_blocks
        self.gpu_free_block_list = list(range(total_blocks))
        self.num_cpu_blocks = getattr(cache_cfg, "fake_num_cpu_blocks", 0)
        self.release_calls = []

    def can_allocate_gpu_blocks(self, num_blocks):
        return len(self.gpu_free_block_list) >= num_blocks

    def allocate_gpu_blocks(self, num_blocks):
        allocated = []
        for _ in range(num_blocks):
            if not self.gpu_free_block_list:
                break
            allocated.append(self.gpu_free_block_list.pop(0))
        return allocated

    def recycle_gpu_blocks(self, block_ids):
        if block_ids:
            self.gpu_free_block_list.extend(block_ids)

    def release_block_ids(self, request):
        self.release_calls.append(request.request_id)

    def release_block_ids_async(self, _):
        pass

    def request_match_blocks(self, request, block_size):
        return getattr(
            request,
            "match_result",
            ([], 0, {"gpu_match_token_num": 0, "cpu_match_token_num": 0}),
        )

    def get_required_block_num(self, token_num, block_size):
        if token_num <= 0:
            return 0
        return (token_num + block_size - 1) // block_size

    def update_cache_blocks(self, request, block_size, num_computed_tokens):
        request.cached_block_num = getattr(request, "cached_block_num", 0)

    def update_cache_config(self, cfg):
        pass


class _FakeSignal:
    """IPCSignal stub capturing shared numeric arrays for assertions."""

    def __init__(self, name, array, dtype, suffix=None, create=True):
        del name, dtype, suffix, create
        self.value = np.array(array, copy=True)


@pytest.fixture(autouse=True)
def _patch_dependencies(monkeypatch):
    metrics = SimpleNamespace(
        max_batch_size=_MetricRecorder(),
        available_gpu_block_num=_MetricRecorder(),
        batch_size=_MetricRecorder(),
        gpu_cache_usage_perc=_MetricRecorder(),
        num_requests_running=_MetricRecorder(),
        num_requests_waiting=_MetricRecorder(),
        prefix_cache_token_num=_MetricRecorder(),
        prefix_gpu_cache_token_num=_MetricRecorder(),
        prefix_cpu_cache_token_num=_MetricRecorder(),
    )
    monkeypatch.setattr("fastdeploy.engine.sched.resource_manager_v1.main_process_metrics", metrics)
    monkeypatch.setattr("fastdeploy.engine.resource_manager.main_process_metrics", metrics)
    monkeypatch.setattr("fastdeploy.engine.resource_manager.PrefixCacheManager", _FakePrefixCacheManager)
    monkeypatch.setattr("fastdeploy.engine.sched.resource_manager_v1.IPCSignal", _FakeSignal)
    mm_cache = types.SimpleNamespace(apply_cache=lambda hashes, positions: [])
    monkeypatch.setattr(
        "fastdeploy.cache_manager.multimodal_cache_manager.EncoderCacheManager",
        lambda *_, **__: mm_cache,
    )
    monkeypatch.setattr(
        "fastdeploy.cache_manager.multimodal_cache_manager.ProcessorCacheManager",
        lambda *_, **__: mm_cache,
    )
    monkeypatch.setattr(
        "fastdeploy.engine.sched.resource_manager_v1.current_platform",
        SimpleNamespace(is_xpu=lambda: False),
    )
    return metrics


@pytest.fixture
def resource_manager_factory():
    def _factory(
        *,
        max_num_seqs=2,
        splitwise_role="mixed",
        enable_prefix=True,
        enable_hierarchical=False,
        model_enable_mm=False,
        block_size=4,
        enc_dec_block_num=2,
        initial_gpu_blocks=64,
        num_cpu_blocks=0,
        max_num_batched_tokens=16,
        speculative_method=None,
        max_encoder_cache=0,
        max_processor_cache=0,
    ):
        cache_cfg = SimpleNamespace(
            block_size=block_size,
            dec_token_num=block_size,
            enc_dec_block_num=enc_dec_block_num,
            enable_prefix_caching=enable_prefix,
            enable_hierarchical_cache=enable_hierarchical,
            max_block_num_per_seq=8,
            prealloc_dec_block_slot_num_threshold=1,
            max_encoder_cache=max_encoder_cache,
            max_processor_cache=max_processor_cache,
            initial_gpu_blocks=initial_gpu_blocks,
            fake_num_cpu_blocks=num_cpu_blocks,
        )
        config = SimpleNamespace(
            cache_config=cache_cfg,
            model_config=SimpleNamespace(enable_mm=model_enable_mm),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=max_num_batched_tokens, splitwise_role=splitwise_role
            ),
            speculative_config=SimpleNamespace(method=speculative_method),
        )
        return rm_v1.ResourceManagerV1(max_num_seqs, config, tensor_parallel_size=1, splitwise_role=splitwise_role)

    return _factory


def _make_request(request_id, prompt_token_ids, **kwargs):
    req = Request.from_dict(
        {
            "request_id": request_id,
            "prompt_token_ids": prompt_token_ids,
            "prompt_token_ids_len": len(prompt_token_ids),
        }
    )
    req.disaggregate_info = kwargs.get("disaggregate_info", {})
    req.cached_block_num = kwargs.get("cached_block_num", 0)
    req.multimodal_inputs = kwargs.get("multimodal_inputs", {})
    req.output_token_ids = kwargs.get("output_token_ids", [])
    req.reasoning_max_tokens = kwargs.get("reasoning_max_tokens")
    req.use_extend_tables = kwargs.get("use_extend_tables", False)
    req.extend_block_tables = kwargs.get("extend_block_tables", [])
    return req


def _setup_running_request(manager, request, idx=0, need_block_num=0):
    """Setup a request in running state with proper internal bookkeeping."""

    request.idx = idx
    manager.running.append(request)
    manager.requests[request.request_id] = request
    manager.req_dict[request.request_id] = idx
    manager.tasks_list[idx] = request
    manager.stop_flags[idx] = False
    if need_block_num:
        manager.need_block_num_signal.value[idx] = need_block_num


def test_signal_consumer_resets_after_limit():
    consumer = rm_v1.SignalConsumer(signal=3, consume_limit=2)
    assert consumer.watch() == 3
    assert consumer.consume() == 3
    assert consumer.consume() == 3
    assert consumer.consume() == 0


def test_get_new_block_nums_with_speculative_budget(resource_manager_factory):
    manager = resource_manager_factory(speculative_method="mtp")
    request = _make_request("spec", list(range(5)))
    request.num_computed_tokens = 2

    num_blocks = manager.get_new_block_nums(request, num_new_tokens=4)

    assert num_blocks == 3  # extra speculative block added


def test_get_num_new_tokens_tracks_patch_boundaries(resource_manager_factory):
    manager = resource_manager_factory(model_enable_mm=True)
    inputs = {
        "patch_idx": [0, 0, 1, 1, 2, 3, 3, 4],
        "patch_map": [
            {"image_num": 0, "video_num": 0, "audio_num": 0, "modal_id": 0, "end_idx": 2},
            {"image_num": 1, "video_num": 0, "audio_num": 0, "modal_id": 0, "end_idx": 4},
            {"image_num": 2, "video_num": 5, "audio_num": 0, "modal_id": 2, "end_idx": 6},
            {"image_num": 3, "video_num": 6, "audio_num": 0, "modal_id": 0, "end_idx": 7},
            {"image_num": 4, "video_num": 7, "audio_num": 0, "modal_id": 0, "end_idx": 8},
        ],
        "image_end_id": 99,
        "video_end_id": 98,
        "audio_end_id": 97,
    }
    request = _make_request(
        "mm-req",
        [5, 6, 7, 8, 9, 99, 10, 11],
        multimodal_inputs=inputs,
    )
    request.num_computed_tokens = 2
    token_budget = 3

    num_tokens = manager._get_num_new_tokens(request, token_budget)

    assert num_tokens == 4  # Modal boundary extended the budget
    assert request.image_start == inputs["patch_map"][1]["image_num"]
    assert request.image_end == inputs["patch_map"][2]["image_num"]
    assert request.video_end == inputs["patch_map"][2]["video_num"]


def test_manager_initializes_mm_caches(resource_manager_factory):
    manager = resource_manager_factory(model_enable_mm=True, max_encoder_cache=1, max_processor_cache=1)
    assert manager.encoder_cache is not None
    assert manager.processor_cache is not None


def test_get_num_new_tokens_with_image_regions(resource_manager_factory):
    manager = resource_manager_factory(model_enable_mm=True)
    request = _make_request("image-mm", [1, 2, 3, 4, 5, 6, 7, 8, 9, 99, 10, 11])
    request.num_computed_tokens = 4
    request.multimodal_img_boundaries = (
        np.array([4, 8, 12], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int64),
    )
    request.multimodal_inputs = {
        "images": [b"chunk"],
        "image_patch_id": 99,
        "grid_thw": np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
        "mm_hashes": ["h1", "h2", "h3"],
        "mm_positions": [ImagePosition(0, 1), ImagePosition(1, 1), ImagePosition(2, 1)],
    }

    num_tokens = manager._get_num_new_tokens(request, token_budget=6)

    assert num_tokens == 8
    assert bool(request.with_image)
    assert request.num_image_start == 1
    assert request.num_image_end == 3
    assert request.image_type_ids_start == 1
    assert request.image_type_ids_end == 6
    assert request.image_start == 1
    assert request.image_end == 36


def test_get_num_new_tokens_builds_missing_boundaries(resource_manager_factory, monkeypatch):
    manager = resource_manager_factory(model_enable_mm=True)
    request = _make_request("image-mm-missing", [99, 5, 99, 6])
    request.num_computed_tokens = 0
    request.multimodal_inputs = {
        "images": [b"a", b"b"],
        "grid_thw": [[1, 1, 1], [1, 1, 1]],
        "mm_positions": [ImagePosition(0, 1), ImagePosition(1, 1)],
        "mm_hashes": [b"x", b"y"],
        "image_patch_id": 99,
    }

    class _BoundaryTensor:
        def numpy(self):
            return np.array([[2, 4], [1, 2]], dtype=np.int64)

    gpu_mod = types.ModuleType("fastdeploy.model_executor.ops.gpu")
    gpu_mod.get_img_boundaries = lambda **_: _BoundaryTensor()
    monkeypatch.setitem(sys.modules, "fastdeploy.model_executor.ops.gpu", gpu_mod)

    num_tokens = manager._get_num_new_tokens(request, token_budget=3)

    assert num_tokens == 4
    assert bool(request.with_image) is True
    assert request.num_image_start >= 1
    assert request.num_image_end >= request.num_image_start
    assert request.image_start > 0


def test_update_mm_hashes_rebuilds_video_positions(resource_manager_factory, monkeypatch):
    manager = resource_manager_factory(model_enable_mm=True)
    request = _make_request("mm-update", [1])
    request.multimodal_inputs = {
        "images": list(range(20)),
        "grid_thw": [[2, 1, 1], [1, 1, 1]],
        "mm_positions": [ImagePosition(0, 1), ImagePosition(1, 1)],
        "mm_hashes": ["a", "b"],
        "image_patch_id": 99,
    }
    monkeypatch.setattr(
        "fastdeploy.engine.sched.resource_manager_v1.MultimodalHasher.hash_features",
        lambda data: b"hash" + bytes([len(data)]),
    )

    manager._update_mm_hashes(request)

    assert len(request.multimodal_inputs["mm_positions"]) == 2
    assert len(request.multimodal_inputs["mm_hashes"]) == 2


def test_update_mm_hashes_ignores_missing_inputs(resource_manager_factory):
    manager = resource_manager_factory(model_enable_mm=True)
    request = _make_request("mm-empty", [1])
    request.multimodal_inputs = None

    manager._update_mm_hashes(request)

    assert request.multimodal_inputs is None


def test_is_mm_request_detects_feature_urls(resource_manager_factory):
    manager = resource_manager_factory(model_enable_mm=True)
    request = _make_request("mm-flag", [1])
    request.multimodal_inputs = {"video_feature_urls": ["v"], "image_feature_urls": [], "audio_feature_urls": []}
    assert manager._is_mm_request(request)


def test_is_mm_request_detects_image_inputs(resource_manager_factory):
    manager = resource_manager_factory(model_enable_mm=True)
    request = _make_request("mm-image", [1])
    request.multimodal_inputs = {"images": [b"x"], "image_patch_id": 9, "grid_thw": [[1, 1, 1]]}

    assert manager._is_mm_request(request)


def test_schedule_prefill_and_decode_roundtrip(resource_manager_factory):
    manager = resource_manager_factory(enable_prefix=False, max_num_seqs=2, max_num_batched_tokens=12)
    req1 = _make_request("prefill-1", list(range(6)))
    req2 = _make_request("prefill-2", list(range(4)))
    manager.add_request(req1)
    manager.add_request(req2)

    scheduled = manager.schedule()
    assert [task.request_id for task in scheduled] == ["prefill-1", "prefill-2"]
    assert all(task.task_type == RequestType.PREFILL for task in scheduled)
    assert len(manager.running) == 2

    req1.num_computed_tokens = req1.need_prefill_tokens
    req2.num_computed_tokens = req2.need_prefill_tokens
    req1.output_token_ids = [42]

    decode_round = manager.schedule()
    decode_types = {task.task_type for task in decode_round}
    assert RequestType.DECODE in decode_types


def test_schedule_handles_preempted_request(resource_manager_factory):
    manager = resource_manager_factory(enable_prefix=False, max_num_seqs=1, max_num_batched_tokens=4)
    req = _make_request("preempted", [1, 2, 3, 4])
    manager.add_request(req)
    req.status = RequestStatus.PREEMPTED
    req.output_token_ids = [5]

    scheduled = manager.schedule()
    assert scheduled and scheduled[0].request_id == req.request_id


def test_schedule_allocates_extend_blocks(resource_manager_factory):
    manager = resource_manager_factory(enable_prefix=False, max_num_seqs=1)
    request = _make_request("extend-flow", list(range(8)))
    request.block_tables = manager.cache_manager.allocate_gpu_blocks(4)
    request.num_computed_tokens = request.need_prefill_tokens
    request.output_token_ids = [10, 11, 12, 13]
    request.use_extend_tables = True
    _setup_running_request(manager, request, need_block_num=2)

    scheduled = manager.schedule()
    extend_tasks = [task for task in scheduled if getattr(task, "task_type", None) == RequestType.EXTEND]
    assert extend_tasks and extend_tasks[0].request_id == request.request_id
    assert request.request_id in manager.using_extend_tables_req_id


def test_schedule_waiting_with_prefix_cache(resource_manager_factory):
    manager = resource_manager_factory(enable_hierarchical=True, num_cpu_blocks=1)
    request = _make_request("cached-wait", list(range(8)))
    request.match_result = ([101], 4, {"gpu_match_token_num": 2, "cpu_match_token_num": 2})
    manager.add_request(request)

    scheduled = manager.schedule()

    assert scheduled and scheduled[0].request_id == request.request_id
    assert request.status == RequestStatus.RUNNING
    assert request.block_tables


def test_trigger_preempt_recycles_and_marks_requests(resource_manager_factory):
    manager = resource_manager_factory(enable_prefix=False)
    manager.cache_manager.gpu_free_block_list.clear()
    running_req = _make_request("run", [1, 1, 1])
    running_req.block_tables = [0, 1]

    tail_req = _make_request("tail", [2, 2, 2])
    tail_req.block_tables = [2, 3]

    _setup_running_request(manager, running_req, idx=0)
    _setup_running_request(manager, tail_req, idx=1)

    scheduled, preempted = [], []
    request_to_schedule = _make_request("waiting", [0])
    result = manager._trigger_preempt(request_to_schedule, 2, preempted, scheduled)

    assert result is True
    assert preempted == [tail_req]
    assert scheduled[-1].task_type.value == RequestStatus.PREEMPTED.value
    assert tail_req.status == RequestStatus.PREEMPTED
    assert tail_req.request_id in manager.to_be_rescheduled_request_id_set
    assert len(manager.cache_manager.gpu_free_block_list) == 2
    assert manager.running == [running_req]


def test_trigger_preempt_skips_extend_tables(resource_manager_factory):
    manager = resource_manager_factory(enable_prefix=False, max_num_seqs=2)
    manager.cache_manager.gpu_free_block_list.clear()
    survivor = _make_request("survivor", [1])
    survivor.block_tables = [0]
    extender = _make_request("extender", [1])
    extender.block_tables = [1]
    extender.use_extend_tables = True

    _setup_running_request(manager, survivor, idx=0)
    _setup_running_request(manager, extender, idx=1)

    preempted, scheduled = [], []
    can_schedule = manager._trigger_preempt(extender, 2, preempted, scheduled)

    assert can_schedule is False
    assert preempted == [survivor]
    assert scheduled[-1].request_id == survivor.request_id
    assert extender in manager.running


def test_trigger_preempt_stops_when_preempting_self(resource_manager_factory):
    manager = resource_manager_factory(enable_prefix=False, max_num_seqs=1)
    manager.cache_manager.gpu_free_block_list.clear()
    request = _make_request("self", [1])
    request.block_tables = [0]
    _setup_running_request(manager, request, idx=0)

    preempted, scheduled = [], []
    can_schedule = manager._trigger_preempt(request, 1, preempted, scheduled)

    assert can_schedule is False
    assert preempted[0].request_id == request.request_id


def test_trigger_preempt_in_decode_role(resource_manager_factory):
    manager = resource_manager_factory(enable_prefix=False, splitwise_role="decode")
    victim = _make_request("victim", [1, 1])
    victim.block_tables = [0, 1]
    _setup_running_request(manager, victim, idx=0)
    manager.cache_manager.gpu_free_block_list.clear()

    scheduled, preempted = [], []
    incoming = _make_request("incoming", [2])
    result = manager._trigger_preempt(incoming, 2, preempted, scheduled)

    assert result is False
    assert preempted and preempted[0] is victim
    assert victim.request_id not in manager.requests
    assert manager.tasks_list[0] is None


def test_preallocate_resource_in_p_uses_prefix_cache(resource_manager_factory):
    manager = resource_manager_factory(
        splitwise_role="prefill",
        enable_hierarchical=True,
        num_cpu_blocks=1,
    )
    request = _make_request("prefill", list(range(12)))
    request.match_result = ([101, 102], 8, {"gpu_match_token_num": 4, "cpu_match_token_num": 4})

    assert manager.preallocate_resource_in_p(request) is True
    assert len(request.block_tables) == 5  # 2 cached + 3 allocated
    assert request.idx is not None
    assert manager.tasks_list[request.idx] is request
    assert manager.requests[request.request_id] is request


def test_preallocate_resource_in_p_returns_false_when_exhausted(resource_manager_factory, monkeypatch):
    manager = resource_manager_factory(
        splitwise_role="prefill",
        enable_hierarchical=True,
        num_cpu_blocks=1,
        initial_gpu_blocks=1,
    )
    request = _make_request("prefill-fail", list(range(6)))
    request.match_result = ([], 0, {"gpu_match_token_num": 0, "cpu_match_token_num": 0})
    monkeypatch.setattr(manager.cache_manager, "can_allocate_gpu_blocks", lambda *_: False)
    monkeypatch.setattr(manager.cache_manager, "allocate_gpu_blocks", lambda *_: [])

    assert manager.preallocate_resource_in_p(request) is False


def test_get_prefix_cached_blocks_all_hit(resource_manager_factory):
    manager = resource_manager_factory()
    request = _make_request("cached", list(range(8)))
    total_tokens = request.need_prefill_tokens
    request.match_result = ([101], total_tokens, {"gpu_match_token_num": total_tokens, "cpu_match_token_num": 0})

    assert manager.get_prefix_cached_blocks(request) is True
    assert request.skip_allocate is True
    assert request.num_computed_tokens == total_tokens - manager.config.cache_config.block_size


def test_get_prefix_cached_blocks_handles_exception(resource_manager_factory, monkeypatch):
    manager = resource_manager_factory()
    request = _make_request("cached-fail", list(range(4)))
    monkeypatch.setattr(
        manager.cache_manager,
        "request_match_blocks",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    success = manager.get_prefix_cached_blocks(request)

    assert success is False


def test_finish_requests_releases_blocks(resource_manager_factory):
    manager = resource_manager_factory(enable_prefix=False)
    request = _make_request("to-finish", [1, 2, 3])
    request.block_tables = manager.cache_manager.allocate_gpu_blocks(2)
    _setup_running_request(manager, request, idx=0)
    manager.to_be_rescheduled_request_id_set.add(request.request_id)

    manager.finish_requests([request.request_id, "missing"])

    assert request.status == RequestStatus.FINISHED
    assert manager.running == []
    assert manager.requests == {}
    assert manager.stop_flags[0] is True
    assert manager.tasks_list[0] is None
    assert request.request_id not in manager.to_be_rescheduled_request_id_set
    assert len(manager.cache_manager.gpu_free_block_list) >= 2


def test_finish_requests_logs_free_block_error(resource_manager_factory, monkeypatch):
    manager = resource_manager_factory(enable_prefix=False)
    request = _make_request("finish-error", [1])
    request.block_tables = manager.cache_manager.allocate_gpu_blocks(1)
    _setup_running_request(manager, request, idx=0)
    monkeypatch.setattr(manager, "_free_blocks", lambda *_: (_ for _ in ()).throw(RuntimeError("oops")))

    manager.finish_requests(request.request_id)

    assert manager.stop_flags[0] is True


def test_finish_requests_async_and_clear_data(resource_manager_factory):
    manager = resource_manager_factory(enable_prefix=False)
    request = _make_request("async", [1, 2])
    request.block_tables = manager.cache_manager.allocate_gpu_blocks(1)
    _setup_running_request(manager, request, idx=0)

    future = manager.finish_requests_async(request.request_id)
    future.result(timeout=1)

    manager.waiting.append(_make_request("cleanup", [3]))
    manager.clear_data()
    assert len(manager.waiting) == 0


def test_prerelease_resource_removes_req_dict(resource_manager_factory):
    manager = resource_manager_factory(enable_prefix=False)
    request = _make_request("prerelease", [1, 2])
    _setup_running_request(manager, request, idx=0)

    manager.prerelease_resource(request)

    assert request.request_id not in manager.req_dict
    assert manager.tasks_list[0] is None


def test_preallocate_resource_in_d_tracks_disagg_info(resource_manager_factory):
    manager = resource_manager_factory(splitwise_role="decode", enable_prefix=False)
    request = _make_request("decode-prealloc", [1, 2, 3], disaggregate_info={}, reasoning_max_tokens=3)

    assert manager.preallocate_resource_in_d(request) is True
    assert request.reasoning_max_tokens == 2
    assert request.num_computed_tokens == request.need_prefill_tokens
    assert request.disaggregate_info["block_tables"] == request.block_tables
    assert manager.requests[request.request_id] is request


def test_insert_task_for_decoding_adds_tokens(resource_manager_factory):
    manager = resource_manager_factory(splitwise_role="decode", speculative_method="mtp")
    request = _make_request("decode", [1, 2, 3])
    request.output_token_ids = []
    request.draft_token_ids = []
    manager.requests[request.request_id] = request

    output = SimpleNamespace(
        request_id=request.request_id,
        outputs=SimpleNamespace(token_ids=[42], draft_token_ids=[9, 8, 7]),
        num_cached_tokens=5,
    )

    manager.insert_task_for_decoding(output)

    assert request.output_token_ids == [42]
    assert request.num_cached_tokens == 5
    assert request.draft_token_ids == [9, 8, 7]
    assert request.draft_token_ids is not output.outputs.draft_token_ids
    assert manager.running[-1] is request
    assert request.need_prefill_tokens == len(request.prompt_token_ids) + 1


def test_free_blocks_release_extend_tables(resource_manager_factory):
    manager = resource_manager_factory()
    request = _make_request("extend", [1, 2, 3], cached_block_num=1)
    request.block_tables = [10, 11, 12]
    request.extend_block_tables = [20, 21, 22, 23]
    manager.using_extend_tables_req_id.add(request.request_id)
    manager.reuse_block_num_map[request.request_id] = 2
    manager.need_block_num_map[request.request_id] = rm_v1.SignalConsumer(2, 1)

    manager._free_blocks(request)

    assert request.block_tables == []
    assert request.extend_block_tables == []
    assert request.request_id not in manager.using_extend_tables_req_id
    assert request.request_id not in manager.reuse_block_num_map
    assert request.request_id not in manager.need_block_num_map
    assert request.request_id in manager.cache_manager.release_calls


def test_reschedule_and_prerelease_flow(resource_manager_factory):
    manager = resource_manager_factory(enable_prefix=False)
    request = _make_request("resched", [1, 2, 3])
    request.idx = 0
    request.block_tables = manager.cache_manager.allocate_gpu_blocks(2)
    manager.waiting.append(request)
    manager.requests[request.request_id] = request
    manager.to_be_rescheduled_request_id_set.add(request.request_id)

    manager.reschedule_preempt_task(request.request_id)
    assert manager.waiting[0] is request

    manager.prerelease_resource(request)
    assert manager.tasks_list[0] is None
    assert request.request_id not in manager.requests


def test_add_request_in_p_appends_running(resource_manager_factory):
    manager = resource_manager_factory(enable_prefix=False)
    request1 = _make_request("p1", [1])
    request2 = _make_request("p2", [2])

    manager.add_request_in_p([request1, request2])

    assert request1 in manager.running and request2 in manager.running
    assert request1.inference_start_time is not None


def test_schedule_preempted_waiting_with_prefix_cache(resource_manager_factory):
    manager = resource_manager_factory(enable_hierarchical=True, num_cpu_blocks=1)
    request = _make_request("cached-preempt", list(range(6)))
    request.match_result = ([111], 2, {"gpu_match_token_num": 1, "cpu_match_token_num": 1})
    manager.add_request(request)
    request.status = RequestStatus.PREEMPTED
    request.output_token_ids = [9, 9]

    scheduled = manager.schedule()

    assert scheduled and scheduled[0].request_id == request.request_id
    assert request.status == RequestStatus.RUNNING


def test_schedule_respects_xpu_prefill_gate(resource_manager_factory, monkeypatch):
    manager = resource_manager_factory(enable_prefix=False, max_num_seqs=2)
    req1 = _make_request("xpu-1", [1, 2])
    req2 = _make_request("xpu-2", [3, 4])
    manager.add_request(req1)
    manager.add_request(req2)

    monkeypatch.setattr(rm_v1.paddle, "is_compiled_with_xpu", lambda: True)

    scheduled = manager.schedule()

    assert scheduled and scheduled[0].request_id == "xpu-1"
    assert req2 in manager.waiting
