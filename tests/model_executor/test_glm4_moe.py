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

from fastdeploy.model_executor.models import glm4_moe

# ── Stubs ────────────────────────────────────────────────────────────────────


class _StubLinear(paddle.nn.Layer):
    """Stub for ReplicatedLinear — accepts fd_config/prefix kwargs, does nothing."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class _StubFusedMoE(paddle.nn.Layer):
    """Stub for FusedMoE — returns input tensor unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, gate, forward_meta=None):
        return x


class _StubMLP(paddle.nn.Layer):
    """Stub for Glm4MoeMLP — returns input tensor unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_fd_config(
    num_hidden_layers=4,
    tensor_parallel_size=1,
    expert_parallel_size=1,
    enable_flashinfer_allreduce_fusion=False,
    n_routed_experts=4,
    n_shared_experts=0,
    hidden_size=16,
    moe_intermediate_size=8,
    num_experts_per_tok=2,
    topk_group=1,
    n_group=1,
    routed_scaling_factor=1.0,
    norm_topk_prob=True,
    moe_gate_fp32=False,
):
    mc = SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared_experts,
        moe_intermediate_size=moe_intermediate_size,
        num_experts_per_tok=num_experts_per_tok,
        topk_group=topk_group,
        n_group=n_group,
        routed_scaling_factor=routed_scaling_factor,
        norm_topk_prob=norm_topk_prob,
        moe_gate_fp32=moe_gate_fp32,
    )
    pc = SimpleNamespace(
        tensor_parallel_size=tensor_parallel_size,
        expert_parallel_size=expert_parallel_size,
        tensor_parallel_rank=0,
        tp_group=None,
        enable_flashinfer_allreduce_fusion=enable_flashinfer_allreduce_fusion,
    )
    return SimpleNamespace(model_config=mc, parallel_config=pc)


@pytest.fixture(autouse=True)
def _patch_heavy_layers(monkeypatch):
    """Replace GPU-intensive layer constructors with lightweight stubs."""
    monkeypatch.setattr(glm4_moe, "ReplicatedLinear", _StubLinear)
    monkeypatch.setattr(glm4_moe, "FusedMoE", _StubFusedMoE)
    monkeypatch.setattr(glm4_moe, "Glm4MoeMLP", _StubMLP)


# ── Tests: __init__ (lines 136-137) ──────────────────────────────────────────


class TestGlm4MoeInit:
    def test_last_layer_id_equals_num_hidden_layers_minus_one(self):
        """Line 136: last_layer_id must be num_hidden_layers - 1."""
        cfg = _make_fd_config(num_hidden_layers=6)
        moe = glm4_moe.Glm4Moe(fd_config=cfg, layer_id=0)
        assert moe.last_layer_id == 5  # 6 - 1

    def test_enable_all_reduce_fusion_true_for_non_last_layer(self):
        """Line 137: fusion enabled when flag is set and layer is not the last."""
        cfg = _make_fd_config(num_hidden_layers=4, enable_flashinfer_allreduce_fusion=True)
        # layer_id=0 is not the last (last=3), so fusion should be True
        moe = glm4_moe.Glm4Moe(fd_config=cfg, layer_id=0)
        assert moe.enable_all_reduce_fusion is True

    def test_enable_all_reduce_fusion_false_for_last_layer(self):
        """Line 137: fusion disabled when current layer IS the last layer."""
        cfg = _make_fd_config(num_hidden_layers=4, enable_flashinfer_allreduce_fusion=True)
        # layer_id=3 equals last_layer_id=3, so fusion must be False
        moe = glm4_moe.Glm4Moe(fd_config=cfg, layer_id=3)
        assert moe.enable_all_reduce_fusion is False

    def test_enable_all_reduce_fusion_false_when_flag_disabled(self):
        """Line 137: fusion disabled when enable_flashinfer_allreduce_fusion=False."""
        cfg = _make_fd_config(num_hidden_layers=4, enable_flashinfer_allreduce_fusion=False)
        moe = glm4_moe.Glm4Moe(fd_config=cfg, layer_id=0)
        assert moe.enable_all_reduce_fusion is False


# ── Tests: forward (lines 208-211) ───────────────────────────────────────────


class TestGlm4MoeForward:
    """
    The branch under test (lines 207-211) is only entered when merge_ffn_tp=True,
    which requires tensor_parallel_size > 1 and expert_parallel_size == 1.
    """

    def _make_tp_moe(self, enable_fusion=False, layer_id=0, num_hidden_layers=4):
        """Build a Glm4Moe in pure-TP mode (tp=2, ep=1)."""
        cfg = _make_fd_config(
            num_hidden_layers=num_hidden_layers,
            tensor_parallel_size=2,
            expert_parallel_size=1,
            enable_flashinfer_allreduce_fusion=enable_fusion,
        )
        moe = glm4_moe.Glm4Moe(fd_config=cfg, layer_id=layer_id)
        assert moe.merge_ffn_tp is True, "precondition: must be in pure-TP mode"
        return moe

    def test_all_reduce_called_when_fusion_disabled(self, monkeypatch):
        """Lines 208-211: when fusion is off, tensor_model_parallel_all_reduce must be called."""
        moe = self._make_tp_moe(enable_fusion=False)

        calls = []

        def _fake_all_reduce(tensor, group):
            calls.append(tensor)
            return tensor + 0  # identity

        monkeypatch.setattr(glm4_moe, "tensor_model_parallel_all_reduce", _fake_all_reduce)

        x = paddle.ones([4, 16], dtype="float32")
        moe.forward(x)
        assert len(calls) == 1, "all-reduce should have been called exactly once"

    def test_all_reduce_called_when_batch_too_large_for_fusion(self, monkeypatch):
        """Lines 208-209: even with fusion enabled, large batch (>2048) must still all-reduce."""
        moe = self._make_tp_moe(enable_fusion=True, layer_id=0)
        # layer_id=0 < last_layer_id=3, so enable_all_reduce_fusion=True
        # But shape[0]=4096 > 2048, so need_tp_all_reduce_fusion=False → all-reduce runs.

        calls = []

        def _fake_all_reduce(tensor, group):
            calls.append(tensor)
            return tensor + 0

        monkeypatch.setattr(glm4_moe, "tensor_model_parallel_all_reduce", _fake_all_reduce)

        x = paddle.ones([4096, 16], dtype="float32")
        moe.forward(x)
        assert len(calls) == 1

    def test_all_reduce_skipped_when_fusion_active_and_small_batch(self, monkeypatch):
        """Lines 208-209: fusion active + small batch → all-reduce must be skipped."""
        moe = self._make_tp_moe(enable_fusion=True, layer_id=0)

        calls = []

        def _fake_all_reduce(tensor, group):
            calls.append(tensor)
            return tensor + 0

        monkeypatch.setattr(glm4_moe, "tensor_model_parallel_all_reduce", _fake_all_reduce)

        x = paddle.ones([128, 16], dtype="float32")  # 128 <= 2048
        moe.forward(x)
        assert len(calls) == 0, "all-reduce should be skipped when fusion handles it"

    def test_all_reduce_not_called_when_merge_ffn_tp_false(self, monkeypatch):
        """Sanity: when merge_ffn_tp=False (ep>1), the all-reduce block is never entered."""
        cfg = _make_fd_config(
            tensor_parallel_size=2,
            expert_parallel_size=2,  # use_ep=True → merge_ffn_tp=False
            enable_flashinfer_allreduce_fusion=False,
        )
        moe = glm4_moe.Glm4Moe(fd_config=cfg, layer_id=0)
        assert moe.merge_ffn_tp is False

        calls = []

        def _fake_all_reduce(tensor, group):
            calls.append(tensor)
            return tensor + 0

        monkeypatch.setattr(glm4_moe, "tensor_model_parallel_all_reduce", _fake_all_reduce)

        x = paddle.ones([4, 16], dtype="float32")
        moe.forward(x)
        assert len(calls) == 0
