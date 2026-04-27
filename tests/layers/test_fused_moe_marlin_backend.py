"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import pytest

if not hasattr(paddle, "enable_compat"):
    paddle.enable_compat = lambda scope=None: None
if not hasattr(paddle.nn.functional, "swiglu"):
    paddle.nn.functional.swiglu = lambda x: x

from fastdeploy.model_executor.layers.moe import fused_moe_marlin_backend as backend
from fastdeploy.model_executor.layers.moe import moe as moe_module


class TestMarlinBackendCoverage:
    """Test coverage for fused_moe_marlin_backend.py.

    Coverage lines:
    - Line 27: current_platform import
    - Lines 263, 264, 265: noaux_tc path with use_fused=False (FD_ENABLE_RL=True or non-CUDA)
    - Line 277: non-noaux_tc path (else branch)
    """

    @pytest.fixture()
    def fake_ops(self, monkeypatch):
        """Mock ops for marlin backend."""

        def fake_moe_topk_select(gate_out, gate_correction_bias, top_k, apply_norm_weight, use_softmax):
            token_num = gate_out.shape[0]
            topk_ids = paddle.zeros([token_num, top_k], dtype="int64")
            topk_weights = paddle.ones([token_num, top_k], dtype="float32")
            return topk_ids, topk_weights

        def fake_get_moe_scores(*args, **kwargs):
            gate_out = args[0]
            token_num = gate_out.shape[0]
            top_k = args[3]
            topk_ids = paddle.zeros([token_num, top_k], dtype="int64")
            topk_weights = paddle.ones([token_num, top_k], dtype="float32")
            return gate_out, topk_weights, topk_ids

        def fake_tritonmoe_preprocess(topk_ids, num_local_experts, block_size):
            token_num = topk_ids.shape[0]
            top_k = topk_ids.shape[1]
            sorted_token_ids = paddle.arange(token_num * top_k, dtype="int32")
            expert_ids = paddle.zeros_like(sorted_token_ids)
            num_tokens_post_padded = paddle.to_tensor([token_num * top_k], dtype="int32")
            return sorted_token_ids, expert_ids, num_tokens_post_padded

        def fake_moe_wna16_marlin_gemm_api(*args, **kwargs):
            x = args[0]
            token_num = x.shape[0]
            hidden_size = x.shape[1]
            return paddle.zeros([token_num, hidden_size], dtype="float16")

        monkeypatch.setattr(
            backend.fastdeploy.model_executor.ops.gpu,
            "moe_topk_select",
            fake_moe_topk_select,
            raising=False,
        )
        # get_moe_scores is imported in marlin_backend from moe.moe, need to mock at that location
        monkeypatch.setattr(moe_module, "get_moe_scores", fake_get_moe_scores)
        monkeypatch.setattr(backend, "tritonmoe_preprocess_func", fake_tritonmoe_preprocess, raising=False)
        monkeypatch.setattr(
            backend.fastdeploy.model_executor.ops.gpu,
            "tritonmoe_preprocess_func",
            fake_tritonmoe_preprocess,
            raising=False,
        )
        monkeypatch.setattr(
            backend.fastdeploy.model_executor.ops.gpu,
            "MoeWna16MarlinGemmApi",
            fake_moe_wna16_marlin_gemm_api,
            raising=False,
        )
        return monkeypatch

    class DummyLayerMarlin(paddle.nn.Layer):
        def __init__(self, topk_method="noaux_tc"):
            super().__init__()
            self.num_local_experts = 2
            self.num_experts = 2
            self.hidden_size = 4
            self.moe_intermediate_size = 3
            self.top_k = 2
            self.n_group = 1
            self.topk_group = 1
            self.routed_scaling_factor = 1.0
            self.renormalize = True
            self.gate_correction_bias = paddle.zeros([2], dtype="float32")
            self.topk_method = topk_method

            # Marlin weights
            self.up_gate_proj_weight = paddle.zeros([2, 4 * 2], dtype="float16")
            self.down_proj_weight = paddle.zeros([3, 4], dtype="float16")
            self.up_gate_proj_weight_scale = paddle.ones([2, 4, 1], dtype="float32")
            self.down_proj_weight_scale = paddle.ones([2, 3, 1], dtype="float32")

    class DummyGateMarlin(paddle.nn.Layer):
        def __init__(self, num_experts):
            super().__init__()
            self.num_experts = num_experts

        def forward(self, x):
            return paddle.ones([x.shape[0], self.num_experts], dtype="float32")

    def test_marlin_current_platform_import_coverage(self):
        """Test that current_platform is imported (line 27)."""
        assert hasattr(backend, "current_platform"), "current_platform should be imported"

    def test_marlin_apply_noaux_tc_with_fd_enable_rl(self, fake_ops, monkeypatch):
        """Test noaux_tc path with FD_ENABLE_RL=True to trigger lines 263-265.

        Line 263: use_fused = not fastdeploy.envs.FD_ENABLE_RL and current_platform.is_cuda()
        Line 264: if not use_fused:
        Line 265: gate_out = gate_out.cast("float32")
        """
        layer = self.DummyLayerMarlin(topk_method="noaux_tc")
        method = backend.MarlinWeightOnlyMoEMethod(quant_method=None)
        method.moe_intermediate_size = layer.moe_intermediate_size
        method.moe_quant_type = "w16a16"

        # Set FD_ENABLE_RL=True to trigger lines 263-265
        monkeypatch.setattr(backend.fastdeploy.envs, "FD_ENABLE_RL", True)

        x = paddle.randn([1, layer.hidden_size], dtype="float32")
        gate = self.DummyGateMarlin(layer.num_local_experts)

        _ = method.apply(layer, x, gate)
        # If this runs without error, lines 263-265 were covered

    def test_marlin_apply_noaux_tc_non_cuda_platform(self, fake_ops, monkeypatch):
        """Test noaux_tc path with non-CUDA platform to trigger lines 263-265.

        Line 263: use_fused = not fastdeploy.envs.FD_ENABLE_RL and current_platform.is_cuda()
        Line 264: if not use_fused:
        Line 265: gate_out = gate_out.cast("float32")
        """
        layer = self.DummyLayerMarlin(topk_method="noaux_tc")
        method = backend.MarlinWeightOnlyMoEMethod(quant_method=None)
        method.moe_intermediate_size = layer.moe_intermediate_size
        method.moe_quant_type = "w16a16"

        # Mock current_platform.is_cuda() to return False to trigger lines 263-265
        monkeypatch.setattr(backend.current_platform, "is_cuda", lambda: False)

        x = paddle.randn([1, layer.hidden_size], dtype="float32")
        gate = self.DummyGateMarlin(layer.num_local_experts)

        _ = method.apply(layer, x, gate)
        # If this runs without error, lines 263-265 were covered

    def test_marlin_apply_non_noaux_tc_path(self, fake_ops):
        """Test non-noaux_tc path (else branch) to trigger line 277.

        Line 277: gate_out = gate_out.cast("float32")
        """
        layer = self.DummyLayerMarlin(topk_method="aux")
        method = backend.MarlinWeightOnlyMoEMethod(quant_method=None)
        method.moe_intermediate_size = layer.moe_intermediate_size
        method.moe_quant_type = "w16a16"

        x = paddle.randn([1, layer.hidden_size], dtype="float32")
        gate = self.DummyGateMarlin(layer.num_local_experts)

        _ = method.apply(layer, x, gate)
        # If this runs without error, line 277 was covered
