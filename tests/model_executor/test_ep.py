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

from __future__ import annotations

from types import SimpleNamespace

import paddle
import pytest

from fastdeploy.config import MoEPhase
from fastdeploy.model_executor.layers.moe import ep


class FakeConfig:
    def __init__(self, nvl_base: int, rdma_base: int):
        self.nvl_base = nvl_base
        self.rdma_base = rdma_base

    def get_nvl_buffer_size_hint(self, hidden_bytes: int, world_size: int) -> int:
        return hidden_bytes + self.nvl_base + world_size

    def get_rdma_buffer_size_hint(self, hidden_bytes: int, world_size: int) -> int:
        return hidden_bytes + self.rdma_base + world_size


class FakeBuffer:
    def __init__(self, group, nvl_bytes, rdma_bytes, low_latency_mode=False, num_qps_per_rank=None):
        self.group = group
        self.nvl_bytes = nvl_bytes
        self.rdma_bytes = rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.num_qps_per_rank = num_qps_per_rank
        self.num_sms = None
        self.cleaned = None
        self.dispatch_args = None
        self.combine_args = None
        self.barrier_called = False
        self._dispatch_hook_called = False
        self._combine_handle = None

    @classmethod
    def get_dispatch_config(cls, _world_size):
        return FakeConfig(1, 2)

    @classmethod
    def get_combine_config(cls, _world_size):
        return FakeConfig(3, 4)

    @classmethod
    def get_low_latency_rdma_size_hint(cls, *_args):
        return 1000

    @classmethod
    def get_low_latency_rdma_size_hint_two_stage(cls, *_args):
        return 2000

    @classmethod
    def get_low_latency_nvl_size_hint_two_stage(cls, *_args):
        return 3000

    def set_num_sms(self, num_sms: int):
        self.num_sms = num_sms

    def clean_low_latency_buffer(self, *args):
        self.cleaned = ("single", args)

    def clean_low_latency_two_stage_buffer(self, *args):
        self.cleaned = ("two_stage", args)

    def barrier_all(self):
        self.barrier_called = True

    def get_dispatch_layout(self, *args, **kwargs):
        num_tokens_per_rank = paddle.to_tensor([1], dtype="int32")
        num_tokens_per_rdma_rank = paddle.to_tensor([1], dtype="int32")
        num_tokens_per_expert = paddle.to_tensor([1], dtype="int32")
        is_token_in_rank = paddle.to_tensor([1], dtype="bool")
        event = "dispatch_event"
        return (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        )

    def dispatch(self, **kwargs):
        self.dispatch_args = kwargs
        return "dispatch_result"

    def combine(self, **kwargs):
        self.combine_args = kwargs
        return "combined", None, "combine_event"

    def low_latency_dispatch(self, *_args, **_kwargs):
        def _dispatch_hook():
            self._dispatch_hook_called = True

        return "packed", "count", ("handle",), None, _dispatch_hook

    def low_latency_dispatch_two_stage(self, *_args, **_kwargs):
        def _dispatch_hook():
            self._dispatch_hook_called = True

        return "packed", "count", None, ("handle",), None, _dispatch_hook

    def low_latency_combine(self, *_args, **_kwargs):
        handle = _args[3]
        self._combine_handle = handle
        return "combined", None, None

    def low_latency_combine_two_stage(self, *_args, **_kwargs):
        handle = _args[3]
        self._combine_handle = handle
        return "combined", None, None


class FakeDeepEP:
    Buffer = FakeBuffer


def _patch_deep_ep(monkeypatch):
    monkeypatch.setattr(ep, "deep_ep", FakeDeepEP, raising=False)


def test_deepep_buffer_manager_calls_engine(monkeypatch):
    class FakeEngine:
        def __init__(self):
            self.cleared = False
            self.created = False

        def clear_deep_ep_buffer(self):
            self.cleared = True

        def create_deep_ep_buffer(self):
            self.created = True

    engine = FakeEngine()
    ep.DeepEPBufferManager.set_engine(engine)
    ep.DeepEPBufferManager.clear_buffer()
    ep.DeepEPBufferManager.recreate_buffer()

    assert engine.cleared is True
    assert engine.created is True


def test_deepep_buffer_mixed_two_stage_buffers(monkeypatch):
    _patch_deep_ep(monkeypatch)
    group = SimpleNamespace(world_size=2)
    buffer = ep.DeepEPBuffer(
        group=group,
        hidden_size=4,
        num_experts=8,
        ep_size=2,
        num_max_dispatch_tokens_per_rank=2,
        splitwise_role="mixed",
        moe_phase=MoEPhase("prefill"),
        use_internode_ll_two_stage=True,
        top_k=2,
    )

    assert buffer.num_nvl_bytes >= 3000
    assert buffer.num_rdma_bytes >= 2000

    buffer.create_buffer()
    assert buffer.deepep_buffer.low_latency_mode is True
    assert buffer.deepep_buffer.num_qps_per_rank == 24
    assert buffer.deepep_buffer.num_sms == 14

    buffer.clean_low_latency_buffer()
    assert buffer.deepep_buffer.cleaned[0] == "two_stage"

    buffer.clear_buffer()
    assert buffer.deepep_buffer is None


def test_deepep_buffer_decode_low_latency_buffer(monkeypatch):
    _patch_deep_ep(monkeypatch)
    group = SimpleNamespace(world_size=4)
    buffer = ep.DeepEPBuffer(
        group=group,
        hidden_size=8,
        num_experts=16,
        ep_size=16,
        num_max_dispatch_tokens_per_rank=1,
        splitwise_role="prefill",
        moe_phase=MoEPhase("decode"),
        use_internode_ll_two_stage=False,
        top_k=4,
    )

    buffer.create_buffer()
    assert buffer.deepep_buffer.low_latency_mode is True
    assert buffer.deepep_buffer.num_qps_per_rank == 2

    buffer.clean_low_latency_buffer()
    assert buffer.deepep_buffer.cleaned[0] == "single"


def test_deepep_buffer_prefill_and_invalid_phase(monkeypatch):
    _patch_deep_ep(monkeypatch)
    group = SimpleNamespace(world_size=2)
    buffer = ep.DeepEPBuffer(
        group=group,
        hidden_size=4,
        num_experts=8,
        ep_size=2,
        num_max_dispatch_tokens_per_rank=2,
        splitwise_role="prefill",
        moe_phase=MoEPhase("prefill"),
        use_internode_ll_two_stage=False,
        top_k=2,
    )

    buffer.create_buffer()
    assert buffer.deepep_buffer.low_latency_mode is True
    assert buffer.deepep_buffer.num_qps_per_rank == 24

    buffer.barrier_all()
    assert buffer.deepep_buffer.barrier_called is True
    assert buffer.get_buffer() is buffer.deepep_buffer

    invalid_phase_buffer = ep.DeepEPBuffer(
        group=group,
        hidden_size=4,
        num_experts=8,
        ep_size=2,
        num_max_dispatch_tokens_per_rank=2,
        splitwise_role="prefill",
        moe_phase=MoEPhase("unknown"),
        use_internode_ll_two_stage=False,
        top_k=2,
    )
    with pytest.raises(ValueError, match="Unknown generation phase"):
        invalid_phase_buffer.create_buffer()


def test_deepep_engine_combine_rewrites_handle_and_errors(monkeypatch):
    _patch_deep_ep(monkeypatch)
    group = SimpleNamespace(world_size=1)
    engine = ep.DeepEPEngine(
        num_max_dispatch_tokens_per_rank=1,
        hidden_size=4,
        num_experts=2,
        ep_size=1,
        ep_rank=0,
        splitwise_role="prefill",
        moe_phase=MoEPhase("decode"),
        group=group,
    )

    hidden_states = paddle.randn([1, 4], dtype="float32")
    topk_idx = paddle.zeros([1, 1], dtype="int64")
    topk_weights = paddle.ones([1, 1], dtype="float32")
    handle = ("src", "layout", 4, 2)
    engine.low_latency_combine(hidden_states, topk_idx, topk_weights, handle)

    assert engine.deepep_engine._combine_handle == handle
    assert len(engine.deepep_engine._combine_handle) == 4

    engine.buffer.deepep_buffer = None
    with pytest.raises(RuntimeError, match="DeepEP buffer not initialized"):
        engine.low_latency_dispatch(hidden_states, topk_idx, None)


def test_prefill_runner_dispatch_and_combine(monkeypatch):
    _patch_deep_ep(monkeypatch)

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            self.async_finish = True
            self.ep_config = "ep_config"
            self.deepep_engine = FakeBuffer(SimpleNamespace(world_size=1), 1, 1)

        def clean_low_latency_buffer(self):
            self.deepep_engine.cleaned = ("single", ())

        def clear_deep_ep_buffer(self):
            self.deepep_engine = None

        def create_deep_ep_buffer(self):
            self.deepep_engine = FakeBuffer(SimpleNamespace(world_size=1), 1, 1)

    monkeypatch.setattr(ep, "DeepEPEngine", FakeEngine)

    ep.EPPrefillRunner.set_allocate_on_comm_stream(True)
    ep.EPPrefillRunner.set_allocate_on_comm_stream(True)

    runner = ep.EPPrefillRunner(
        top_k=2,
        hidden_size=4,
        num_experts=2,
        splitwise_role="prefill",
        num_max_dispatch_tokens_per_rank=1,
    )
    x = paddle.randn([2, 4], dtype="float32")
    topk_idx = paddle.zeros([2, 2], dtype="int64")
    topk_weights = paddle.ones([2, 2], dtype="float32")

    dispatch_result = runner.dispatch(x, topk_idx, topk_weights, expert_alignment=8)
    assert dispatch_result == "dispatch_result"
    assert runner.ep_engine.deepep_engine.dispatch_args["allocate_on_comm_stream"] is True
    assert runner.ep_engine.deepep_engine.dispatch_args["expert_alignment"] == 8

    combined, event = runner.combine(x, ("handle",), topk_weights)
    assert combined == "combined"
    assert event == "combine_event"


def test_decoder_runner_dispatch_and_combine_hooks(monkeypatch):
    _patch_deep_ep(monkeypatch)

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            self.dispatch_called = False
            self.combine_called = False
            self.two_stage_dispatch_called = False
            self.two_stage_combine_called = False

        def low_latency_dispatch(self, *args, **kwargs):
            self.dispatch_called = True
            return "recv", "count", ("handle",), lambda: self._mark_hook("dispatch")

        def low_latency_dispatch_two_stage(self, *args, **kwargs):
            self.two_stage_dispatch_called = True
            return "recv", "count", ("handle",), lambda: self._mark_hook("dispatch")

        def low_latency_combine(self, *args, **kwargs):
            self.combine_called = True
            return "combined", lambda: self._mark_hook("combine")

        def low_latency_combine_two_stage(self, *args, **kwargs):
            self.two_stage_combine_called = True
            return "combined", lambda: self._mark_hook("combine")

        def _mark_hook(self, name):
            setattr(self, f"{name}_hook_called", True)

    monkeypatch.setattr(ep, "DeepEPEngine", FakeEngine)

    runner = ep.EPDecoderRunner(
        top_k=2,
        hidden_size=4,
        num_experts=2,
        splitwise_role="prefill",
        num_max_dispatch_tokens_per_rank=1,
    )
    x = paddle.randn([1, 4], dtype="float32")
    topk_idx = paddle.zeros([1, 1], dtype="int64")
    topk_weights = paddle.ones([1, 1], dtype="float32")

    recv_hidden, recv_count, handle = runner.dispatch(x, topk_idx, topk_weights)
    assert recv_hidden == "recv"
    assert recv_count == "count"
    assert handle == ("handle",)
    assert runner.ep_engine.dispatch_called is True
    assert runner.ep_engine.dispatch_hook_called is True

    combined = runner.combine(x, topk_idx, topk_weights, handle)
    assert combined == "combined"
    assert runner.ep_engine.combine_called is True
    assert runner.ep_engine.combine_hook_called is True

    runner_two_stage = ep.EPDecoderRunner(
        top_k=2,
        hidden_size=4,
        num_experts=2,
        splitwise_role="prefill",
        num_max_dispatch_tokens_per_rank=1,
        use_internode_ll_two_stage=True,
    )
    recv_hidden, recv_count, handle = runner_two_stage.dispatch(
        x, topk_idx, topk_weights, expertwise_scale=None, use_fp8=True
    )
    assert recv_hidden == "recv"
    assert runner_two_stage.ep_engine.two_stage_dispatch_called is True

    combined = runner_two_stage.combine(x, topk_idx, topk_weights, handle, quant_group_size=64)
    assert combined == "combined"
    assert runner_two_stage.ep_engine.two_stage_combine_called is True


def test_eprunner_moe_select_noaux_tc_without_redundant(monkeypatch):
    _patch_deep_ep(monkeypatch)

    def fake_get_moe_scores(*_args, **_kwargs):
        return "score", paddle.to_tensor([[0.5]]), paddle.to_tensor([[1]], dtype="int64")

    from fastdeploy.model_executor.layers.moe import moe as moe_module

    monkeypatch.setattr(moe_module, "get_moe_scores", fake_get_moe_scores, raising=True)

    runner = ep.EPPrefillRunner(
        top_k=2,
        hidden_size=4,
        num_experts=2,
        splitwise_role="prefill",
        num_max_dispatch_tokens_per_rank=1,
    )

    layer = SimpleNamespace(
        redundant_table_manger=None,
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        top_k=2,
        routed_scaling_factor=1.0,
        gate_correction_bias=None,
        renormalize=False,
    )
    gate_out = paddle.randn([1, 4], dtype="float32")

    topk_idx, topk_weights = runner.moe_select(layer, gate_out)
    assert list(topk_idx.shape) == [1, 1]
    assert list(topk_weights.shape) == [1, 1]
    assert paddle.allclose(topk_idx, paddle.to_tensor([[1]], dtype="int64"))
    assert paddle.allclose(topk_weights, paddle.to_tensor([[0.5]]))


def test_eprunner_moe_select_redundant_and_topk(monkeypatch):
    _patch_deep_ep(monkeypatch)

    def fake_redundant_topk_select(**_kwargs):
        return paddle.to_tensor([[2]], dtype="int64"), paddle.to_tensor([[0.25]])

    from fastdeploy.model_executor.ops import gpu as gpu_ops

    monkeypatch.setattr(gpu_ops, "moe_redundant_topk_select", fake_redundant_topk_select, raising=True)

    runner = ep.EPPrefillRunner(
        top_k=2,
        hidden_size=4,
        num_experts=2,
        splitwise_role="prefill",
        num_max_dispatch_tokens_per_rank=1,
    )

    class FakeRedundantTableManager:
        def get_ep_rank_to_expert_id_list_by_layer(self, _layer_idx):
            return [0], paddle.to_tensor([0], dtype="int64"), [1], [1]

    layer = SimpleNamespace(
        redundant_table_manger=FakeRedundantTableManager(),
        layer_idx=0,
        topk_method="aux",
        n_group=1,
        topk_group=1,
        top_k=2,
        routed_scaling_factor=1.0,
        gate_correction_bias=None,
        fd_config=SimpleNamespace(eplb_config=SimpleNamespace(redundant_experts_num=0)),
    )
    gate_out = paddle.randn([1, 4], dtype="float32")

    topk_idx, topk_weights = runner.moe_select(layer, gate_out)
    assert list(topk_idx.shape) == [1, 1]
    assert list(topk_weights.shape) == [1, 1]
    assert paddle.allclose(topk_idx, paddle.to_tensor([[2]], dtype="int64"))
    assert paddle.allclose(topk_weights, paddle.to_tensor([[0.25]]))


def test_eprunner_moe_select_topk_without_redundant(monkeypatch):
    _patch_deep_ep(monkeypatch)

    def fake_topk_select(*_args, **_kwargs):
        return paddle.to_tensor([[3]], dtype="int64"), paddle.to_tensor([[0.75]])

    from fastdeploy.model_executor.ops import gpu as gpu_ops

    monkeypatch.setattr(gpu_ops, "moe_topk_select", fake_topk_select, raising=True)

    runner = ep.EPPrefillRunner(
        top_k=2,
        hidden_size=4,
        num_experts=2,
        splitwise_role="prefill",
        num_max_dispatch_tokens_per_rank=1,
    )

    layer = SimpleNamespace(
        redundant_table_manger=None,
        topk_method="aux",
        gate_correction_bias=None,
    )
    gate_out = paddle.randn([1, 4], dtype="float32")

    topk_idx, topk_weights = runner.moe_select(layer, gate_out)
    assert list(topk_idx.shape) == [1, 1]
    assert list(topk_weights.shape) == [1, 1]
    assert paddle.allclose(topk_idx, paddle.to_tensor([[3]], dtype="int64"))
    assert paddle.allclose(topk_weights, paddle.to_tensor([[0.75]]))
