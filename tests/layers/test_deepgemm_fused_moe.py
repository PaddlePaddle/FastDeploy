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

import sys
import types

import paddle
import pytest

# ---------------------------------------------------------------------------
# Minimal stub before any fastdeploy import: deep_ep requires distributed setup
# ---------------------------------------------------------------------------

deep_ep_stub = types.ModuleType("fastdeploy.model_executor.layers.moe.ep.deep_ep")
deep_ep_stub.Buffer = types.SimpleNamespace(capture=lambda: object())
sys.modules["fastdeploy.model_executor.layers.moe.ep.deep_ep"] = deep_ep_stub

from fastdeploy.model_executor.layers.moe import (  # noqa: E402
    fused_moe_deepgemm_backend as backend,
)

# ---------------------------------------------------------------------------
# Detect whether deepgemm JIT compilation works on this machine.
# It requires the host compiler to support C++17 (GCC >= 7).
# CI machines with older GCC will fail to compile the kernel.
# ---------------------------------------------------------------------------


def _deepgemm_available() -> bool:
    """Try to JIT-compile a minimal deepgemm kernel; return False on failure."""
    try:
        from fastdeploy.model_executor.layers.quantization.fp8_utils import deep_gemm

        lhs = paddle.zeros([128, 128], dtype="float8_e4m3fn")
        lhs_scale = paddle.ones([128, 1], dtype="float32")
        rhs = paddle.zeros([1, 128, 128], dtype="float8_e4m3fn")
        rhs_scale = paddle.ones([1, 1, 1], dtype="float32")
        out = paddle.empty([128, 128], dtype="bfloat16")
        m_indices = paddle.zeros([128], dtype="int32")
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous((lhs, lhs_scale), (rhs, rhs_scale), out, m_indices)
        return True
    except Exception:
        return False


_DEEPGEMM_AVAILABLE = _deepgemm_available()

requires_deepgemm = pytest.mark.skipif(
    not _DEEPGEMM_AVAILABLE,
    reason="deepgemm JIT compilation requires C++17-capable host compiler (GCC >= 7)",
)

# ---------------------------------------------------------------------------
# Test parameters – deepgemm requires:
#   M alignment = 128 (tokens dispatched to each expert)
#   N, K must be multiples of 128
# ---------------------------------------------------------------------------
NUM_EXPERTS = 2
HIDDEN_SIZE = 128  # K
MOE_INTER = 128  # moe_intermediate_size  →  N_up = 256, N_down = 128
TOP_K = 2
EP_SIZE = 1
# Use ≥128 tokens so that after top-k expansion M≥128 (deepgemm alignment)
NUM_TOKENS = 128  # ensures token_all_num = NUM_TOKENS * TOP_K / ... ≥ 128

# Weight block size matching deepgemm: 128×128
WEIGHT_BLOCK_SIZE = (128, 128)


# ---------------------------------------------------------------------------
# Dummy helpers
# ---------------------------------------------------------------------------


class DummyQuantConfig:
    def __init__(self):
        self.weight_block_size = WEIGHT_BLOCK_SIZE
        self.deepgemm_scale_ue8m0 = False
        self.is_checkpoint_bf16 = False

    def name(self):
        return "blockwise_fp8"


class DummyFDConfig:
    def __init__(self):
        self.load_config = types.SimpleNamespace(load_choices="default_v1", dynamic_load_weight=False)
        self.model_config = types.SimpleNamespace(
            enable_cache=False,
            model="dummy",
            # ep_size * this = max tokens buffer for masked GEMM; must be ≥ aligned M
            num_max_dispatch_tokens_per_rank=128,
        )
        self.scheduler_config = types.SimpleNamespace(max_num_batched_tokens=NUM_TOKENS)
        self.parallel_config = types.SimpleNamespace(tensor_parallel_size=1)


class DummyLayer(paddle.nn.Layer):
    """Layer with properly-shaped fp8 weights for deepgemm."""

    def __init__(self):
        super().__init__()
        qc = DummyQuantConfig()
        E = NUM_EXPERTS
        K = HIDDEN_SIZE
        N_up = MOE_INTER * 2  # 256
        N_down = HIDDEN_SIZE  # 128
        K_down = MOE_INTER  # 128

        self.num_local_experts = E
        self.num_experts = E
        self.hidden_size = K
        self.moe_intermediate_size = MOE_INTER
        self.top_k = TOP_K
        self.ep_size = EP_SIZE
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.renormalize = True
        self.gate_correction_bias = paddle.zeros([E], dtype="float32")
        self.topk_method = "noaux_tc"
        self.fd_config = DummyFDConfig()
        self.quant_method = types.SimpleNamespace(quant_config=qc)

        # up_gate_proj_weight: [E, N_up, K]  (deepgemm NT: each expert [N, K])
        self.up_gate_proj_weight = self.create_parameter(
            shape=[E, N_up, K],
            dtype="float8_e4m3fn",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        # down_proj_weight: [E, N_down, K_down]
        self.down_proj_weight = self.create_parameter(
            shape=[E, N_down, K_down],
            dtype="float8_e4m3fn",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        # Scales: [E, ceil(N/128), ceil(K/128)]
        self.up_gate_proj_weight_scale_inv = self.create_parameter(
            shape=[E, N_up // 128, K // 128],  # [2, 2, 1]
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.down_proj_weight_scale_inv = self.create_parameter(
            shape=[E, N_down // 128, K_down // 128],  # [2, 1, 1]
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )


class DummyGate(paddle.nn.Layer):
    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, x):
        return paddle.ones([x.shape[0], self.num_experts], dtype="float32")


def _make_method():
    qc = DummyQuantConfig()
    method = backend.DeepGemmFusedMoeMethod(qc)
    method.added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
    method.added_scale_attrs = ["up_gate_proj_weight_scale_inv", "down_proj_weight_scale_inv"]
    return method


# ---------------------------------------------------------------------------
# Tests: apply_tp
# ---------------------------------------------------------------------------


class TestApplyTp:
    """apply_tp with FD_USE_PHI_FP8_QUANT=True, FD_USE_PHI_MOE_PERMUTE=True."""

    @requires_deepgemm
    def test_apply_tp_noaux_tc_path(self):
        """noaux_tc: get_moe_scores → fp8_quant_blockwise → moe_permute → deepgemm → moe_unpermute."""
        layer = DummyLayer()
        gate = DummyGate(layer.num_local_experts)
        method = _make_method()

        x = paddle.randn([NUM_TOKENS, HIDDEN_SIZE], dtype="bfloat16")
        captured = {}

        def hook(topk_ids):
            captured["topk_ids"] = topk_ids

        out = method.apply_tp(layer, x, gate, topk_ids_hookfunc=hook)

        assert "topk_ids" in captured
        assert list(out.shape) == [NUM_TOKENS, HIDDEN_SIZE]

    @requires_deepgemm
    def test_apply_tp_aux_path(self):
        """Non-noaux_tc: moe_topk_select → fp8_quant_blockwise → moe_permute → deepgemm → moe_unpermute."""
        layer = DummyLayer()
        layer.topk_method = "greedy"
        gate = DummyGate(layer.num_local_experts)
        method = _make_method()

        x = paddle.randn([NUM_TOKENS, HIDDEN_SIZE], dtype="bfloat16")
        out = method.apply_tp(layer, x, gate)

        assert list(out.shape) == [NUM_TOKENS, HIDDEN_SIZE]


# ---------------------------------------------------------------------------
# Tests: apply_ep_prefill
# ---------------------------------------------------------------------------


class TestApplyEpPrefill:
    """apply_ep_prefill: stub only the EP communication runner."""

    def _make_zero_runner(self, layer):
        """Runner that returns 0 tokens per expert → zero-token branch."""

        class ZeroRunner:
            num_worst_tokens = 0
            ep_engine = types.SimpleNamespace(async_finish=False)

            def moe_select(self, _layer, gate_out):
                n = gate_out.shape[0]
                return (
                    paddle.zeros([n, _layer.top_k], dtype="int64"),
                    paddle.ones([n, _layer.top_k], dtype="float32"),
                )

            def dispatch(self, x, topk_idx, topk_weights, **kwargs):
                # x is already fp8 (after fp8_quant_blockwise), scale comes via x_scale_tensor kwarg
                n = x.shape[0]
                scale = kwargs.get("x_scale_tensor", paddle.ones([n, 1], dtype="float32"))
                return (
                    (x, scale),
                    topk_idx,
                    topk_weights,
                    [0, 0],
                    object(),
                    types.SimpleNamespace(current_stream_wait=lambda: None),
                )

            def combine(self, out, handle, weights, event):
                return out, types.SimpleNamespace(current_stream_wait=lambda: None)

        return ZeroRunner()

    def _make_contiguous_runner(self, layer):
        """Runner that returns token_all_num > 0 → contiguous GEMM branch."""

        class ContiguousRunner:
            num_worst_tokens = 0
            ep_engine = types.SimpleNamespace(async_finish=False)

            def moe_select(self, _layer, gate_out):
                n = gate_out.shape[0]
                # Route all tokens to expert 0 so count is deterministic
                topk_ids = paddle.zeros([n, _layer.top_k], dtype="int64")
                topk_weights = paddle.ones([n, _layer.top_k], dtype="float32")
                return topk_ids, topk_weights

            def dispatch(self, x, topk_idx, topk_weights, **kwargs):
                n = x.shape[0]
                scale = kwargs.get("x_scale_tensor", paddle.ones([n, 1], dtype="float32"))
                # non-zero counts so token_all_num > 0
                num_per_expert = [n * layer.top_k // layer.num_local_experts] * layer.num_local_experts
                return (
                    (x, scale),
                    topk_idx,
                    topk_weights,
                    num_per_expert,
                    object(),
                    types.SimpleNamespace(current_stream_wait=lambda: None),
                )

            def combine(self, out, handle, weights, event):
                return out, types.SimpleNamespace(current_stream_wait=lambda: None)

        return ContiguousRunner()

    def test_ep_prefill_zero_token_path(self):
        """All experts get 0 tokens → returns empty [0, hidden_size] tensor."""
        layer = DummyLayer()
        layer.topk_method = "greedy"
        gate = DummyGate(layer.num_local_experts)
        method = _make_method()
        method.ep_prefill_runner = self._make_zero_runner(layer)

        x = paddle.randn([NUM_TOKENS, HIDDEN_SIZE], dtype="bfloat16")
        out = method.apply_ep_prefill(layer, x, gate)
        assert list(out.shape) == [0, HIDDEN_SIZE]

    @requires_deepgemm
    def test_ep_prefill_contiguous_path(self):
        """token_all_num > 0, num_worst_tokens == 0 → moe_permute + contiguous deepgemm."""
        layer = DummyLayer()
        layer.topk_method = "greedy"
        gate = DummyGate(layer.num_local_experts)
        method = _make_method()
        method.ep_prefill_runner = self._make_contiguous_runner(layer)

        x = paddle.randn([NUM_TOKENS, HIDDEN_SIZE], dtype="bfloat16")
        out = method.apply_ep_prefill(layer, x, gate)
        assert len(out.shape) == 2
        assert out.shape[1] == HIDDEN_SIZE


# ---------------------------------------------------------------------------
# Tests: apply_ep_decode
# ---------------------------------------------------------------------------


class TestApplyEpDecode:
    """apply_ep_decode: stub only the EP communication runner."""

    def _make_decode_runner(self, layer):
        """Decode runner: dispatch returns fp8 tuple + token counts, combine aggregates."""
        max_dispatch = layer.fd_config.model_config.num_max_dispatch_tokens_per_rank

        class DecodeRunner:
            ep_engine = types.SimpleNamespace(async_finish=False)

            def moe_select(self, _layer, gate_out):
                n = gate_out.shape[0]
                top_k = _layer.top_k
                return (
                    paddle.zeros([n, top_k], dtype="int64"),
                    paddle.ones([n, top_k], dtype="float32"),
                )

            def dispatch(self, x, topk_idx, topk_weights, use_fp8=False, use_ue8m0=False):
                E = layer.num_local_experts
                ep = layer.ep_size
                K = layer.hidden_size
                # Return (fp8_tensor, scale) tuple as expected by apply_ep_decode
                x_fp8 = paddle.zeros([E, ep * max_dispatch, K], dtype="float8_e4m3fn")
                scale = paddle.ones([E, ep * max_dispatch, 1], dtype="float32")
                token_nums = paddle.zeros([E], dtype="int32")
                return (x_fp8, scale), token_nums, object()

            def combine(self, ffn_out, topk_idx, topk_weights, handle):
                n_tok = topk_idx.shape[0]
                return paddle.zeros([n_tok, layer.hidden_size], dtype="bfloat16")

        return DecodeRunner()

    @requires_deepgemm
    def test_ep_decode_masked_gemm_path(self):
        """dispatch → masked deepgemm → fused_mask_swiglu_fp8_quant → masked deepgemm → combine."""
        layer = DummyLayer()
        layer.topk_method = "greedy"
        gate = DummyGate(layer.num_local_experts)
        method = _make_method()
        method.ep_decoder_runner = self._make_decode_runner(layer)

        x = paddle.randn([NUM_TOKENS, HIDDEN_SIZE], dtype="bfloat16")
        captured = {}

        def hook(topk_ids):
            captured["topk_ids"] = topk_ids

        out = method.apply_ep_decode(layer, x, gate, topk_ids_hookfunc=hook)

        assert "topk_ids" in captured
        assert list(out.shape) == [NUM_TOKENS, HIDDEN_SIZE]
