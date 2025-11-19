"""
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
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
from pytest import MonkeyPatch


class _RecordingBufferConfig:
    def __init__(self, world_size: int):
        self.world_size = world_size

    def get_nvl_buffer_size_hint(self, hidden_bytes, world_size):
        return hidden_bytes * max(world_size, 1)

    def get_rdma_buffer_size_hint(self, hidden_bytes, world_size):
        return hidden_bytes * (max(world_size, 1) + 1)


class _RecordingBuffer:
    """Minimal DeepEP buffer stub that records every interaction."""

    DEFAULT_RDMA_HINT = 1536
    DEFAULT_NVL_HINT = 1024
    TWO_STAGE_RDMA_HINT = 4096
    TWO_STAGE_NVL_HINT = 2048

    init_history: list[dict] = []

    def __init__(self, group, num_nvl_bytes, num_rdma_bytes, *, low_latency_mode, num_qps_per_rank):
        self.group = group
        self.kwargs = {
            "num_nvl_bytes": num_nvl_bytes,
            "num_rdma_bytes": num_rdma_bytes,
            "low_latency_mode": low_latency_mode,
            "num_qps_per_rank": num_qps_per_rank,
        }
        self.num_sms = None
        self.dispatch_layout_calls: list[dict] = []
        self.dispatch_calls: list[dict] = []
        self.combine_calls: list[dict] = []
        self.clean_calls: list[dict] = []
        self.low_latency_dispatch_calls: list[dict] = []
        self.low_latency_dispatch_two_stage_calls: list[dict] = []
        self.low_latency_combine_calls: list[dict] = []
        self.low_latency_combine_two_stage_calls: list[dict] = []
        self.barrier_count = 0
        type(self).init_history.append({"kwargs": self.kwargs, "instance": self})

    @classmethod
    def reset(cls):
        cls.init_history.clear()

    @classmethod
    def get_dispatch_config(cls, world_size):
        return _RecordingBufferConfig(world_size)

    @classmethod
    def get_combine_config(cls, world_size):
        return _RecordingBufferConfig(world_size)

    @staticmethod
    def get_low_latency_rdma_size_hint(*_args):
        return _RecordingBuffer.DEFAULT_RDMA_HINT

    @staticmethod
    def get_low_latency_rdma_size_hint_two_stage(*_args):
        return _RecordingBuffer.TWO_STAGE_RDMA_HINT

    @staticmethod
    def get_low_latency_nvl_size_hint_two_stage(*_args, **_kwargs):
        return _RecordingBuffer.TWO_STAGE_NVL_HINT

    def set_num_sms(self, num_sms):
        self.num_sms = num_sms

    def get_dispatch_layout(self, topk_idx, num_experts, async_finish):
        call = {
            "topk_idx": topk_idx,
            "num_experts": num_experts,
            "async_finish": async_finish,
        }
        self.dispatch_layout_calls.append(call)
        return ("rank", "rdma", "expert", "in_rank", "prefill_event")

    def dispatch(self, **kwargs):
        self.dispatch_calls.append(kwargs)
        return "dispatched_prefill"

    def combine(self, **kwargs):
        self.combine_calls.append(kwargs)
        return "combined_prefill", None, "prefill_finished"

    def low_latency_dispatch(
        self,
        hidden_states,
        topk_idx,
        expertwise_scale,
        max_tokens,
        num_experts,
        *,
        use_fp8,
        async_finish,
        return_recv_hook,
    ):
        call = {
            "hidden_states": hidden_states,
            "topk_idx": topk_idx,
            "expertwise_scale": expertwise_scale,
            "max_tokens": max_tokens,
            "num_experts": num_experts,
            "use_fp8": use_fp8,
            "async_finish": async_finish,
            "return_recv_hook": return_recv_hook,
            "hook_called": False,
        }
        self.low_latency_dispatch_calls.append(call)

        def _hook():
            call["hook_called"] = True

        return ("recv_hidden", "recv_count", ("src", "layout", max_tokens, num_experts), None, _hook)

    def low_latency_dispatch_two_stage(
        self,
        hidden_states,
        topk_idx,
        topk_weights,
        max_tokens,
        num_experts,
        *,
        use_fp8,
        async_finish,
        return_recv_hook,
    ):
        call = {
            "hidden_states": hidden_states,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
            "max_tokens": max_tokens,
            "num_experts": num_experts,
            "use_fp8": use_fp8,
            "async_finish": async_finish,
            "return_recv_hook": return_recv_hook,
            "hook_called": False,
        }
        self.low_latency_dispatch_two_stage_calls.append(call)

        def _hook():
            call["hook_called"] = True

        return (
            "recv_two_stage",
            "recv_two_stage_count",
            None,
            ("src", "layout", max_tokens, num_experts),
            None,
            _hook,
        )

    def low_latency_combine(self, hidden_states, topk_idx, topk_weights, handle, *, async_finish, return_recv_hook):
        call = {
            "hidden_states": hidden_states,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
            "handle": handle,
            "async_finish": async_finish,
            "return_recv_hook": return_recv_hook,
            "hook_called": False,
        }
        self.low_latency_combine_calls.append(call)

        def _hook():
            call["hook_called"] = True

        return "combined_decode", None, _hook

    def low_latency_combine_two_stage(
        self,
        hidden_states,
        topk_idx,
        topk_weights,
        handle,
        *,
        async_finish,
        dispatch_use_fp8,
        return_recv_hook,
    ):
        call = {
            "hidden_states": hidden_states,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
            "handle": handle,
            "async_finish": async_finish,
            "dispatch_use_fp8": dispatch_use_fp8,
            "return_recv_hook": return_recv_hook,
            "hook_called": False,
        }
        self.low_latency_combine_two_stage_calls.append(call)

        def _hook():
            call["hook_called"] = True

        return "combined_two_stage", None, _hook

    def clean_low_latency_buffer(self, *args):
        self.clean_calls.append({"method": "single", "args": args})

    def clean_low_latency_two_stage_buffer(self, *args):
        self.clean_calls.append({"method": "two_stage", "args": args})

    def barrier_all(self):
        self.barrier_count += 1


class _FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.logger = types.SimpleNamespace(setLevel=lambda *_args, **_kwargs: None)

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)


@pytest.fixture(scope="module")
def _ep_env():
    """Install scoped stubs required to import the ep module."""

    monkeypatch = MonkeyPatch()

    project_root = Path(__file__).resolve().parents[2]

    def ensure_module(name: str, *, package: bool = False, path: str | None = None) -> types.ModuleType:
        module = types.ModuleType(name)
        if package:
            module.__path__ = [] if path is None else [path]
        monkeypatch.setitem(sys.modules, name, module)
        return module

    paddle = ensure_module("paddle")
    paddle.__version__ = "3.0.0"
    paddle.Tensor = type("Tensor", (), {})
    paddle.is_compiled_with_rocm = lambda: False
    paddle.is_compiled_with_cuda = lambda: False
    paddle.is_compiled_with_xpu = lambda: False
    paddle.is_compiled_with_custom_device = lambda _name: False

    nn_module = ensure_module("paddle.nn")
    nn_module.Layer = object
    paddle.nn = nn_module

    dist_module = ensure_module("paddle.distributed")

    class _Group:
        def __init__(self, ranks):
            ranks = list(ranks)
            self.ranks = tuple(ranks)
            self.world_size = max(len(ranks), 1)

    dist_module.new_group = lambda ranks: _Group(ranks)
    paddle.distributed = dist_module

    comm_module = ensure_module("paddle.distributed.communication", package=True)
    deep_ep_module = ensure_module("paddle.distributed.communication.deep_ep")
    deep_ep_module.Buffer = _RecordingBuffer
    comm_module.deep_ep = deep_ep_module
    dist_module.communication = comm_module

    paddleformers = ensure_module("paddleformers", package=True)
    pf_utils = ensure_module("paddleformers.utils", package=True)
    log_module = ensure_module("paddleformers.utils.log")
    log_module.logger = _FakeLogger()
    pf_utils.log = log_module
    paddleformers.utils = pf_utils
    transformers = ensure_module("paddleformers.transformers", package=True)
    configuration_utils = ensure_module("paddleformers.transformers.configuration_utils")

    class PretrainedConfig:
        pass

    configuration_utils.PretrainedConfig = PretrainedConfig
    transformers.configuration_utils = configuration_utils
    paddleformers.transformers = transformers

    fastdeploy_module = ensure_module("fastdeploy", package=True, path=str(project_root / "fastdeploy"))
    utils_module = ensure_module("fastdeploy.utils")

    def singleton(cls):
        return cls

    utils_module.singleton = singleton
    fastdeploy_module.utils = utils_module

    config_module = ensure_module("fastdeploy.config")

    class MoEPhase:
        """Simple stub mirroring the production API."""

        def __init__(self, phase="prefill"):
            self.phase = phase

        @property
        def phase(self):
            return self._phase

        @phase.setter
        def phase(self, value):
            if value not in ["prefill", "decode"]:
                raise ValueError(f"The moe_phase is invalid, only support prefill and decode, but got {value}")
            self._phase = value

    config_module.MoEPhase = MoEPhase
    fastdeploy_module.config = config_module

    fd_model_executor = ensure_module(
        "fastdeploy.model_executor", package=True, path=str(project_root / "fastdeploy" / "model_executor")
    )
    fd_layers = ensure_module(
        "fastdeploy.model_executor.layers",
        package=True,
        path=str(project_root / "fastdeploy" / "model_executor" / "layers"),
    )
    fd_moe_pkg = ensure_module(
        "fastdeploy.model_executor.layers.moe",
        package=True,
        path=str(project_root / "fastdeploy" / "model_executor" / "layers" / "moe"),
    )
    fd_ops_pkg = ensure_module(
        "fastdeploy.model_executor.ops",
        package=True,
        path=str(project_root / "fastdeploy" / "model_executor" / "ops"),
    )

    gpu_module = ensure_module("fastdeploy.model_executor.ops.gpu")
    gpu_module.calls = {"redundant": [], "topk": []}

    def moe_redundant_topk_select(**kwargs):
        gpu_module.calls["redundant"].append(kwargs)
        return ("redundant_idx", "redundant_weights")

    def moe_topk_select(*args):
        gpu_module.calls["topk"].append(args)
        return ("plain_idx", "plain_weights")

    gpu_module.moe_redundant_topk_select = moe_redundant_topk_select
    gpu_module.moe_topk_select = moe_topk_select

    moe_module = ensure_module("fastdeploy.model_executor.layers.moe.moe")
    moe_module.calls = []

    def get_moe_scores(*args, **kwargs):
        record = {"args": args, "kwargs": kwargs}
        moe_module.calls.append(record)
        return ("score", "weights", "indices")

    moe_module.get_moe_scores = get_moe_scores

    fd_ops_pkg.gpu = gpu_module
    fd_moe_pkg.moe = moe_module
    fd_layers.moe = fd_moe_pkg
    fd_model_executor.layers = fd_layers
    fd_model_executor.ops = fd_ops_pkg
    fastdeploy_module.model_executor = fd_model_executor

    ep_module = importlib.import_module("fastdeploy.model_executor.layers.moe.ep")
    ep_module = importlib.reload(ep_module)

    try:
        yield {"ep_module": ep_module, "gpu_module": gpu_module, "moe_module": moe_module}
    finally:
        monkeypatch.undo()


@pytest.fixture()
def ep_module(_ep_env):
    module = importlib.reload(_ep_env["ep_module"])
    module.DeepEPBufferManager._engine = None
    return module


@pytest.fixture()
def gpu_ops_module(_ep_env):
    return _ep_env["gpu_module"]


@pytest.fixture()
def moe_scores_module(_ep_env):
    return _ep_env["moe_module"]


@pytest.fixture()
def moe_phase_cls(_ep_env):
    from fastdeploy.config import MoEPhase

    return MoEPhase


@pytest.fixture(autouse=True)
def reset_recorders(gpu_ops_module, moe_scores_module):
    _RecordingBuffer.reset()
    gpu_ops_module.calls["redundant"].clear()
    gpu_ops_module.calls["topk"].clear()
    moe_scores_module.calls.clear()
    yield
    _RecordingBuffer.reset()


def test_buffer_two_stage_allocations_and_cleanup(ep_module, moe_phase_cls):
    phase = moe_phase_cls("prefill")
    group = types.SimpleNamespace(world_size=2)
    buffer = ep_module.DeepEPBuffer(
        group=group,
        hidden_size=16,
        num_experts=8,
        ep_size=2,
        num_max_dispatch_tokens_per_rank=32,
        splitwise_role="mixed",
        moe_phase=phase,
        use_internode_ll_two_stage=True,
        top_k=4,
    )
    assert buffer.num_rdma_bytes == _RecordingBuffer.TWO_STAGE_RDMA_HINT
    assert buffer.num_nvl_bytes == _RecordingBuffer.TWO_STAGE_NVL_HINT

    buffer.create_buffer()
    instance = buffer.deepep_buffer
    assert instance.kwargs["low_latency_mode"] is True
    assert instance.kwargs["num_qps_per_rank"] == 24

    buffer.clean_low_latency_buffer()
    assert instance.clean_calls[-1]["method"] == "two_stage"

    buffer.barrier_all()
    assert instance.barrier_count == 1


def test_buffer_create_unknown_phase(ep_module):
    odd_phase = types.SimpleNamespace(phase="unknown")
    buffer = ep_module.DeepEPBuffer(
        group=types.SimpleNamespace(world_size=1),
        hidden_size=8,
        num_experts=2,
        ep_size=1,
        num_max_dispatch_tokens_per_rank=8,
        splitwise_role="prefill",
        moe_phase=odd_phase,
        use_internode_ll_two_stage=False,
        top_k=2,
    )
    with pytest.raises(ValueError):
        buffer.create_buffer()


def test_low_latency_buffer_qps_scaling(ep_module, moe_phase_cls):
    phase = moe_phase_cls("decode")
    buffer = ep_module.DeepEPBuffer(
        group=types.SimpleNamespace(world_size=4),
        hidden_size=32,
        num_experts=32,
        ep_size=32,
        num_max_dispatch_tokens_per_rank=64,
        splitwise_role="prefill",
        moe_phase=phase,
        use_internode_ll_two_stage=False,
        top_k=8,
    )
    buffer._create_low_latency_buffer()
    record = _RecordingBuffer.init_history[-1]
    assert record["kwargs"]["num_qps_per_rank"] == 4


def test_deepep_engine_low_latency_combine_rewrites_handle(ep_module, moe_phase_cls):
    engine = ep_module.DeepEPEngine(
        num_max_dispatch_tokens_per_rank=4,
        hidden_size=32,
        num_experts=8,
        ep_size=2,
        ep_rank=0,
        splitwise_role="prefill",
        moe_phase=moe_phase_cls("decode"),
    )
    combined, hook = engine.low_latency_combine("ffn", "idx", "weights", ("src", "layout", 5, 7))
    call = engine.deepep_engine.low_latency_combine_calls[-1]
    assert call["handle"][3] is None
    assert combined == "combined_decode"
    hook()
    assert call["hook_called"] is True


def test_prefill_runner_dispatch_and_combine_flow(ep_module):
    runner = ep_module.EPPrefillRunner(
        top_k=2,
        hidden_size=16,
        num_experts=4,
        splitwise_role="prefill",
        num_max_dispatch_tokens_per_rank=4,
        ep_size=2,
        ep_rank=0,
    )
    dispatch_result = runner.dispatch(
        "hidden",
        topk_idx="idx",
        topk_weights="weights",
        expert_alignment=2,
        x_scale_tensor="scale",
    )
    instance = runner.ep_engine.deepep_engine
    layout_call = instance.dispatch_layout_calls[-1]
    assert layout_call["num_experts"] == runner.num_experts
    dispatch_call = instance.dispatch_calls[-1]
    assert dispatch_call["x"] == ("hidden", "scale")
    assert dispatch_result == "dispatched_prefill"

    fused, event = runner.combine("tmp", handle="handle", recv_topk_weights="weights")
    combine_call = instance.combine_calls[-1]
    assert combine_call["topk_weights"] == "weights"
    assert (fused, event) == ("combined_prefill", "prefill_finished")


def test_decoder_runner_dispatch_and_combine_two_stage(ep_module):
    runner = ep_module.EPDecoderRunner(
        top_k=2,
        hidden_size=16,
        num_experts=4,
        splitwise_role="decode",
        num_max_dispatch_tokens_per_rank=4,
        ep_size=2,
        ep_rank=0,
        use_internode_ll_two_stage=True,
    )
    recv_hidden, recv_count, handle = runner.dispatch(
        "hidden",
        topk_idx="idx",
        topk_weights="weights",
        expertwise_scale="scale",
        use_fp8=True,
    )
    instance = runner.ep_engine.deepep_engine
    dispatch_call = instance.low_latency_dispatch_two_stage_calls[-1]
    assert dispatch_call["topk_weights"] == "weights"
    assert dispatch_call["hook_called"] is True
    assert recv_hidden == "recv_two_stage"

    combined = runner.combine("ffn", "idx", "weights", handle)
    combine_call = instance.low_latency_combine_two_stage_calls[-1]
    assert combine_call["dispatch_use_fp8"] is True
    assert combine_call["hook_called"] is True
    assert combined == "combined_two_stage"


def test_moe_select_prefers_redundant_tables(ep_module, gpu_ops_module):
    runner = ep_module.EPPrefillRunner(
        top_k=2,
        hidden_size=8,
        num_experts=4,
        splitwise_role="prefill",
        num_max_dispatch_tokens_per_rank=2,
    )

    class _RedundantTable:
        def __init__(self):
            self.requests = []

        def get_ep_rank_to_expert_id_list_by_layer(self, layer_idx):
            self.requests.append(layer_idx)
            return ([0], [0], [1], [2])

    layer = types.SimpleNamespace(
        redundant_table_manger=_RedundantTable(),
        layer_idx=3,
        gate_correction_bias="bias",
        fd_config=types.SimpleNamespace(model_config=types.SimpleNamespace(redundant_experts_num=1)),
        topk_method="any",
    )

    topk_idx, topk_weights = runner.moe_select(layer, gate_out="logits")
    assert (topk_idx, topk_weights) == ("redundant_idx", "redundant_weights")
    assert len(gpu_ops_module.calls["redundant"]) == 1
    assert layer.redundant_table_manger.requests == [3]


def test_moe_select_uses_moe_scores_with_noaux(ep_module, moe_scores_module):
    runner = ep_module.EPPrefillRunner(
        top_k=2,
        hidden_size=8,
        num_experts=4,
        splitwise_role="prefill",
        num_max_dispatch_tokens_per_rank=2,
    )
    layer = types.SimpleNamespace(
        redundant_table_manger=None,
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        top_k=2,
        routed_scaling_factor=0.5,
        gate_correction_bias="bias",
        renormalize=False,
    )
    topk_idx, topk_weights = runner.moe_select(layer, gate_out="logits")
    assert (topk_idx, topk_weights) == ("indices", "weights")
    assert len(moe_scores_module.calls) == 1
    call = moe_scores_module.calls[0]
    assert call["args"][0] == "logits"


def test_moe_select_falls_back_to_gpu_topk(ep_module, gpu_ops_module):
    runner = ep_module.EPPrefillRunner(
        top_k=2,
        hidden_size=8,
        num_experts=4,
        splitwise_role="prefill",
        num_max_dispatch_tokens_per_rank=2,
    )
    layer = types.SimpleNamespace(
        redundant_table_manger=None,
        topk_method="default",
        gate_correction_bias="bias",
    )
    topk_idx, topk_weights = runner.moe_select(layer, gate_out="logits")
    assert (topk_idx, topk_weights) == ("plain_idx", "plain_weights")
    assert len(gpu_ops_module.calls["topk"]) == 1
