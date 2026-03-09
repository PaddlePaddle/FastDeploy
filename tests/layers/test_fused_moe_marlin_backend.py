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

"""Unit tests for fastdeploy/model_executor/layers/moe/fused_moe_marlin_backend.py"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import paddle
from paddle import nn

# Mock GPU ops before importing the module under test.
# These ops require compiled CUDA extensions not available on CPU.
_mock_gpu = MagicMock()
sys.modules.setdefault("fastdeploy.model_executor.ops.gpu", _mock_gpu)

from fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend import (  # noqa: E402
    MarlinWeightOnlyMoEMethod,
    get_scale_perms,
    gptq_marlin_moe_repack,
    marlin_moe_permute_scales,
    marlin_permute_scales,
)

MB = "fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend"


# ── helpers ──────────────────────────────────────────────────────────────
class _DummyLayer(nn.Layer):
    """Minimal layer mock matching FusedMoE's attributes."""

    def __init__(
        self,
        hidden_size=128,
        moe_intermediate_size=64,
        num_local_experts=2,
        num_experts=2,
        top_k=2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_local_experts = num_local_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = 1
        self.topk_group = 1
        self.topk_method = "topk"
        self.routed_scaling_factor = 1.0
        self.gate_correction_bias = paddle.zeros([num_experts], dtype="float32")

    def extract_moe_ffn_weights(self, state_dict):
        return state_dict["up"], state_dict["down"], None, None


# ── get_scale_perms ──────────────────────────────────────────────────────
class TestGetScalePerms(unittest.TestCase):
    def test_returns_two_lists(self):
        sp, sps = get_scale_perms()
        self.assertIsInstance(sp, list)
        self.assertIsInstance(sps, list)

    def test_scale_perm_length(self):
        sp, _ = get_scale_perms()
        self.assertEqual(len(sp), 64)

    def test_scale_perm_single_length(self):
        _, sps = get_scale_perms()
        self.assertEqual(len(sps), 32)

    def test_scale_perm_contains_all_values(self):
        sp, _ = get_scale_perms()
        self.assertEqual(sorted(sp), list(range(64)))

    def test_scale_perm_single_elements_in_range(self):
        _, sps = get_scale_perms()
        for v in sps:
            self.assertGreaterEqual(v, 0)
            self.assertLess(v, 32)

    def test_deterministic(self):
        sp1, sps1 = get_scale_perms()
        sp2, sps2 = get_scale_perms()
        self.assertEqual(sp1, sp2)
        self.assertEqual(sps1, sps2)


# ── marlin_permute_scales ────────────────────────────────────────────────
class TestMarlinPermuteScales(unittest.TestCase):
    def test_per_channel_shape(self):
        """group_size=-1 → uses scale_perm_single (32 elems)."""
        s = paddle.arange(32, dtype="float32").reshape([1, 32])
        result = marlin_permute_scales(s, size_k=128, size_n=32, group_size=-1)
        self.assertEqual(list(result.shape), [1, 32])

    def test_per_group_shape(self):
        """group_size < size_k → uses scale_perm (64 elems)."""
        s = paddle.arange(256, dtype="float32").reshape([4, 64])
        result = marlin_permute_scales(s, size_k=128, size_n=64, group_size=32)
        self.assertEqual(list(result.shape), [4, 64])

    def test_per_channel_permutes_values(self):
        s = paddle.arange(32, dtype="float32").reshape([1, 32])
        result = marlin_permute_scales(s, size_k=128, size_n=32, group_size=-1)
        # Values should be permuted (not identity)
        original = s.numpy().flatten()
        permuted = result.numpy().flatten()
        self.assertFalse(np.array_equal(original, permuted))
        # But same set of values
        np.testing.assert_array_equal(sorted(original), sorted(permuted))

    def test_per_group_permutes_values(self):
        s = paddle.arange(64, dtype="float32").reshape([1, 64])
        result = marlin_permute_scales(s, size_k=128, size_n=64, group_size=32)
        original = s.numpy().flatten()
        permuted = result.numpy().flatten()
        self.assertFalse(np.array_equal(original, permuted))
        np.testing.assert_array_equal(sorted(original), sorted(permuted))

    def test_preserves_dtype(self):
        s = paddle.ones([1, 32], dtype="float16")
        result = marlin_permute_scales(s, size_k=128, size_n=32, group_size=-1)
        self.assertEqual(result.dtype, paddle.float16)


# ── marlin_moe_permute_scales ────────────────────────────────────────────
class TestMarlinMoePermuteScales(unittest.TestCase):
    def test_output_shape_matches_input(self):
        num_experts = 4
        s = paddle.ones([num_experts, 1, 32], dtype="float32")
        result = marlin_moe_permute_scales(s, size_k=128, size_n=32, group_size=-1)
        self.assertEqual(list(result.shape), [num_experts, 1, 32])

    def test_each_expert_permuted_independently(self):
        s = paddle.stack(
            [
                paddle.arange(32, dtype="float32").reshape([1, 32]),
                paddle.arange(32, 64, dtype="float32").reshape([1, 32]),
            ],
            axis=0,
        )
        result = marlin_moe_permute_scales(s, size_k=128, size_n=32, group_size=-1)
        # Each expert should have different values
        e0 = result[0].numpy().flatten()
        e1 = result[1].numpy().flatten()
        self.assertFalse(np.array_equal(e0, e1))

    def test_preserves_dtype_float16(self):
        s = paddle.ones([2, 1, 32], dtype="float16")
        result = marlin_moe_permute_scales(s, size_k=128, size_n=32, group_size=-1)
        self.assertEqual(result.dtype, paddle.float16)

    def test_single_expert(self):
        s = paddle.arange(32, dtype="float32").reshape([1, 1, 32])
        result = marlin_moe_permute_scales(s, size_k=128, size_n=32, group_size=-1)
        self.assertEqual(list(result.shape), [1, 1, 32])


# ── gptq_marlin_moe_repack ──────────────────────────────────────────────
class TestGptqMarlinMoeRepack(unittest.TestCase):
    def test_calls_repack_per_expert(self):
        num_experts = 3
        size_k = 64
        size_n = 32
        num_bits = 4
        b_q_weight = paddle.zeros([num_experts, size_k, size_n], dtype="int32")
        perm = paddle.zeros([num_experts, 0], dtype="int32")

        mock_repack = MagicMock(return_value=paddle.ones([size_k // 16, size_n * (num_bits // 2)], dtype="int32"))
        _mock_gpu.gptq_marlin_repack = mock_repack
        result = gptq_marlin_moe_repack(b_q_weight, perm, size_k, size_n, num_bits)

        self.assertEqual(mock_repack.call_count, num_experts)
        self.assertEqual(list(result.shape), [num_experts, size_k // 16, size_n * (num_bits // 2)])

    def test_size_k_alignment(self):
        """size_k must be divisible by 16."""
        b_q_weight = paddle.zeros([1, 15, 8], dtype="int32")
        perm = paddle.zeros([1, 0], dtype="int32")
        with self.assertRaises(AssertionError):
            gptq_marlin_moe_repack(b_q_weight, perm, 15, 8, 4)


# ── MarlinWeightOnlyMoEMethod.__init__ ───────────────────────────────────
class TestMarlinInit(unittest.TestCase):
    def test_default_attrs(self):
        method = MarlinWeightOnlyMoEMethod()
        self.assertIsNone(method.quant_method)
        self.assertEqual(len(method.added_weight_attrs), 2)
        self.assertEqual(len(method.added_scale_attrs), 2)
        self.assertEqual(len(method.added_zeros_attrs), 2)

    def test_custom_quant_method(self):
        method = MarlinWeightOnlyMoEMethod(quant_method="gptq")
        self.assertEqual(method.quant_method, "gptq")

    def test_weight_attr_names(self):
        method = MarlinWeightOnlyMoEMethod()
        self.assertIn("up_gate_proj_weight", method.added_weight_attrs)
        self.assertIn("down_proj_weight", method.added_weight_attrs)

    def test_scale_attr_names(self):
        method = MarlinWeightOnlyMoEMethod()
        self.assertIn("up_gate_proj_weight_scale", method.added_scale_attrs)
        self.assertIn("down_proj_weight_scale", method.added_scale_attrs)


# ── MarlinWeightOnlyMoEMethod.create_weights ─────────────────────────────
class TestMarlinCreateWeights(unittest.TestCase):
    def test_creates_weight_parameters(self):
        method = MarlinWeightOnlyMoEMethod()
        layer = _DummyLayer(hidden_size=128, moe_intermediate_size=64, num_local_experts=2)
        method.create_weights(layer)

        self.assertTrue(hasattr(layer, "up_gate_proj_weight"))
        self.assertTrue(hasattr(layer, "down_proj_weight"))

    def test_weight_shapes(self):
        method = MarlinWeightOnlyMoEMethod()
        layer = _DummyLayer(hidden_size=128, moe_intermediate_size=64, num_local_experts=2)
        method.create_weights(layer)

        # up_gate: [E, hidden//16, intermediate*4]
        self.assertEqual(list(layer.up_gate_proj_weight.shape), [2, 128 // 16, 64 * 4])
        # down: [E, intermediate//16, hidden*2]
        self.assertEqual(list(layer.down_proj_weight.shape), [2, 64 // 16, 128 * 2])

    def test_scale_shapes(self):
        method = MarlinWeightOnlyMoEMethod()
        layer = _DummyLayer(hidden_size=128, moe_intermediate_size=64, num_local_experts=2)
        method.create_weights(layer)

        self.assertTrue(hasattr(layer, "up_gate_proj_weight_scale"))
        self.assertTrue(hasattr(layer, "down_proj_weight_scale"))
        # up_gate_scale: [E, 1, intermediate*2]
        self.assertEqual(list(layer.up_gate_proj_weight_scale.shape), [2, 1, 64 * 2])
        # down_scale: [E, 1, hidden]
        self.assertEqual(list(layer.down_proj_weight_scale.shape), [2, 1, 128])

    def test_weight_dtype_is_int32(self):
        method = MarlinWeightOnlyMoEMethod()
        layer = _DummyLayer()
        method.create_weights(layer)
        self.assertEqual(layer.up_gate_proj_weight.dtype, paddle.int32)
        self.assertEqual(layer.down_proj_weight.dtype, paddle.int32)

    def test_stores_shapes(self):
        method = MarlinWeightOnlyMoEMethod()
        layer = _DummyLayer(hidden_size=256, moe_intermediate_size=128, num_local_experts=4)
        method.create_weights(layer)
        self.assertEqual(method.up_gate_proj_weight_shape[0], 4)
        self.assertEqual(method.down_proj_weight_shape[0], 4)


# ── MarlinWeightOnlyMoEMethod.process_loaded_weights ─────────────────────
class TestMarlinProcessLoadedWeights(unittest.TestCase):
    @patch(f"{MB}.marlin_moe_permute_scales")
    @patch(f"{MB}.gptq_marlin_moe_repack")
    def test_processes_all_experts(self, mock_repack, mock_permute):
        method = MarlinWeightOnlyMoEMethod()
        hidden = 128
        inter = 64
        num_experts = 2
        layer = _DummyLayer(hidden_size=hidden, moe_intermediate_size=inter, num_local_experts=num_experts)
        method.create_weights(layer)

        up_weights = [paddle.randn([hidden, inter * 2]) for _ in range(num_experts)]
        down_weights = [paddle.randn([inter, hidden]) for _ in range(num_experts)]
        state_dict = {"up": up_weights, "down": down_weights}

        mock_repack.side_effect = lambda w, p, k, n, b: paddle.zeros(
            [num_experts, k // 16, n * (b // 2)], dtype="int32"
        )
        mock_permute.side_effect = lambda s, **kw: s

        method.process_loaded_weights(layer, state_dict)

        # gptq_marlin_moe_repack called once per weight type (up_gate + down)
        self.assertEqual(mock_repack.call_count, 2)
        # marlin_moe_permute_scales called once per scale type
        self.assertEqual(mock_permute.call_count, 2)

    @patch(f"{MB}.marlin_moe_permute_scales")
    @patch(f"{MB}.gptq_marlin_moe_repack")
    def test_assertion_on_wrong_expert_count(self, mock_repack, mock_permute):
        method = MarlinWeightOnlyMoEMethod()
        layer = _DummyLayer(hidden_size=128, moe_intermediate_size=64, num_local_experts=2)
        method.create_weights(layer)

        # Provide wrong number of experts
        up_weights = [paddle.randn([128, 128])]  # only 1 expert
        down_weights = [paddle.randn([64, 128]), paddle.randn([64, 128])]
        state_dict = {"up": up_weights, "down": down_weights}

        with self.assertRaises(AssertionError):
            method.process_loaded_weights(layer, state_dict)

    @patch(f"{MB}.marlin_moe_permute_scales")
    @patch(f"{MB}.gptq_marlin_moe_repack")
    def test_assertion_on_wrong_weight_shape(self, mock_repack, mock_permute):
        method = MarlinWeightOnlyMoEMethod()
        layer = _DummyLayer(hidden_size=128, moe_intermediate_size=64, num_local_experts=1)
        method.create_weights(layer)

        # Wrong shape: [hidden, wrong_size] instead of [hidden, inter*2]
        up_weights = [paddle.randn([128, 999])]
        down_weights = [paddle.randn([64, 128])]
        state_dict = {"up": up_weights, "down": down_weights}

        with self.assertRaises(AssertionError):
            method.process_loaded_weights(layer, state_dict)


# ── MarlinWeightOnlyMoEMethod.apply ──────────────────────────────────────
class TestMarlinApply(unittest.TestCase):
    def _setup_method_and_layer(self):
        method = MarlinWeightOnlyMoEMethod()
        layer = _DummyLayer(
            hidden_size=128,
            moe_intermediate_size=64,
            num_local_experts=4,
            num_experts=4,
            top_k=2,
        )
        method.create_weights(layer)
        return method, layer

    @patch(f"{MB}.tritonmoe_preprocess_func")
    @patch(f"{MB}.MoeWna16MarlinGemmApi")
    @patch("fastdeploy.model_executor.ops.gpu.moe_topk_select")
    def test_apply_output_shape(self, mock_topk, mock_gemm, mock_preproc):
        method, layer = self._setup_method_and_layer()
        batch, hidden = 4, 128
        x = paddle.randn([batch, hidden])
        gate = nn.Linear(hidden, layer.num_experts, bias_attr=False)

        # Setup mocks
        topk_ids = paddle.zeros([batch, layer.top_k], dtype="int32")
        topk_weights = paddle.ones([batch, layer.top_k], dtype="float32")
        mock_topk.return_value = (topk_ids, topk_weights)
        mock_preproc.return_value = (
            paddle.zeros([16], dtype="int32"),
            paddle.zeros([4], dtype="int32"),
            paddle.to_tensor([16], dtype="int32"),
        )
        # MoeWna16MarlinGemmApi returns tuple; first call = up_gate+swiglu, second = down
        final_out = paddle.randn([batch * layer.top_k, hidden])
        mock_gemm.side_effect = [
            (paddle.randn([batch * layer.top_k, layer.moe_intermediate_size * 2]),),
            (final_out,),
        ]

        result = method.apply(layer, x, gate)
        self.assertEqual(list(result.shape), [batch, hidden])

    @patch(f"{MB}.tritonmoe_preprocess_func")
    @patch(f"{MB}.MoeWna16MarlinGemmApi")
    @patch("fastdeploy.model_executor.ops.gpu.moe_topk_select")
    def test_apply_calls_gemm_twice(self, mock_topk, mock_gemm, mock_preproc):
        """apply() should call MoeWna16MarlinGemmApi exactly twice (up_gate + down)."""
        method, layer = self._setup_method_and_layer()
        batch, hidden = 2, 128
        x = paddle.randn([batch, hidden])
        gate = nn.Linear(hidden, layer.num_experts, bias_attr=False)

        topk_ids = paddle.zeros([batch, layer.top_k], dtype="int32")
        topk_weights = paddle.ones([batch, layer.top_k], dtype="float32")
        mock_topk.return_value = (topk_ids, topk_weights)
        mock_preproc.return_value = (
            paddle.zeros([8], dtype="int32"),
            paddle.zeros([4], dtype="int32"),
            paddle.to_tensor([8], dtype="int32"),
        )
        mock_gemm.side_effect = [
            (paddle.randn([batch * layer.top_k, layer.moe_intermediate_size * 2]),),
            (paddle.randn([batch * layer.top_k, hidden]),),
        ]

        method.apply(layer, x, gate)
        self.assertEqual(mock_gemm.call_count, 2)

    @patch(f"{MB}.tritonmoe_preprocess_func")
    @patch(f"{MB}.MoeWna16MarlinGemmApi")
    @patch("fastdeploy.model_executor.ops.gpu.moe_topk_select")
    def test_apply_with_hookfunc(self, mock_topk, mock_gemm, mock_preproc):
        """topk_ids_hookfunc should be called with topk_ids."""
        method, layer = self._setup_method_and_layer()
        batch, hidden = 2, 128
        x = paddle.randn([batch, hidden])
        gate = nn.Linear(hidden, layer.num_experts, bias_attr=False)

        topk_ids = paddle.zeros([batch, layer.top_k], dtype="int32")
        topk_weights = paddle.ones([batch, layer.top_k], dtype="float32")
        mock_topk.return_value = (topk_ids, topk_weights)
        mock_preproc.return_value = (
            paddle.zeros([8], dtype="int32"),
            paddle.zeros([4], dtype="int32"),
            paddle.to_tensor([8], dtype="int32"),
        )
        mock_gemm.side_effect = [
            (paddle.randn([batch * layer.top_k, layer.moe_intermediate_size * 2]),),
            (paddle.randn([batch * layer.top_k, hidden]),),
        ]

        hook = MagicMock()
        method.apply(layer, x, gate, topk_ids_hookfunc=hook)
        hook.assert_called_once()

    @patch(f"{MB}.tritonmoe_preprocess_func")
    @patch(f"{MB}.MoeWna16MarlinGemmApi")
    def test_apply_noaux_tc_topk_method(self, mock_gemm, mock_preproc):
        """When topk_method='noaux_tc', should use get_moe_scores instead of moe_topk_select."""
        method, layer = self._setup_method_and_layer()
        layer.topk_method = "noaux_tc"
        batch, hidden = 2, 128
        x = paddle.randn([batch, hidden])
        gate = nn.Linear(hidden, layer.num_experts, bias_attr=False)

        mock_preproc.return_value = (
            paddle.zeros([8], dtype="int32"),
            paddle.zeros([4], dtype="int32"),
            paddle.to_tensor([8], dtype="int32"),
        )
        mock_gemm.side_effect = [
            (paddle.randn([batch * layer.top_k, layer.moe_intermediate_size * 2]),),
            (paddle.randn([batch * layer.top_k, hidden]),),
        ]

        mock_scores = MagicMock(
            return_value=(
                None,
                paddle.ones([batch, layer.top_k], dtype="float32"),
                paddle.zeros([batch, layer.top_k], dtype="int32"),
            )
        )
        with patch(f"{MB}.get_moe_scores", mock_scores, create=True):
            with patch("fastdeploy.model_executor.layers.moe.moe.get_moe_scores", mock_scores, create=True):
                result = method.apply(layer, x, gate)
        self.assertEqual(list(result.shape), [batch, hidden])

    @patch(f"{MB}.tritonmoe_preprocess_func")
    @patch(f"{MB}.MoeWna16MarlinGemmApi")
    @patch("fastdeploy.model_executor.ops.gpu.moe_topk_select")
    def test_block_size_selection(self, mock_topk, mock_gemm, mock_preproc):
        """Block size should be selected based on token_num * top_k / num_experts ratio."""
        method, layer = self._setup_method_and_layer()
        # Use small batch to trigger small block_size
        batch, hidden = 1, 128
        x = paddle.randn([batch, hidden])
        gate = nn.Linear(hidden, layer.num_experts, bias_attr=False)

        topk_ids = paddle.zeros([batch, layer.top_k], dtype="int32")
        topk_weights = paddle.ones([batch, layer.top_k], dtype="float32")
        mock_topk.return_value = (topk_ids, topk_weights)
        mock_preproc.return_value = (
            paddle.zeros([8], dtype="int32"),
            paddle.zeros([4], dtype="int32"),
            paddle.to_tensor([8], dtype="int32"),
        )
        mock_gemm.side_effect = [
            (paddle.randn([batch * layer.top_k, layer.moe_intermediate_size * 2]),),
            (paddle.randn([batch * layer.top_k, hidden]),),
        ]

        method.apply(layer, x, gate)
        # Verify preprocess was called (block_size_m passed implicitly)
        mock_preproc.assert_called_once()


# ── Edge cases / integration ─────────────────────────────────────────────
class TestEdgeCases(unittest.TestCase):
    def test_single_expert_permute(self):
        s = paddle.arange(32, dtype="float32").reshape([1, 1, 32])
        result = marlin_moe_permute_scales(s, size_k=128, size_n=32, group_size=-1)
        self.assertEqual(list(result.shape), [1, 1, 32])

    def test_large_num_experts(self):
        num_experts = 16
        s = paddle.ones([num_experts, 1, 32], dtype="float32")
        result = marlin_moe_permute_scales(s, size_k=128, size_n=32, group_size=-1)
        self.assertEqual(list(result.shape), [num_experts, 1, 32])

    def test_method_inherits_quant_base(self):
        from fastdeploy.model_executor.layers.quantization.quant_base import (
            QuantMethodBase,
        )

        self.assertTrue(issubclass(MarlinWeightOnlyMoEMethod, QuantMethodBase))

    def test_create_weights_different_sizes(self):
        """Test create_weights with various hidden/intermediate size combos."""
        for hidden, inter in [(256, 128), (512, 256), (64, 32)]:
            method = MarlinWeightOnlyMoEMethod()
            layer = _DummyLayer(hidden_size=hidden, moe_intermediate_size=inter, num_local_experts=1)
            method.create_weights(layer)
            self.assertEqual(list(layer.up_gate_proj_weight.shape), [1, hidden // 16, inter * 4])
            self.assertEqual(list(layer.down_proj_weight.shape), [1, inter // 16, hidden * 2])


if __name__ == "__main__":
    unittest.main()
