"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import unittest
from unittest.mock import MagicMock, patch

import paddle

from fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend import (
    MarlinWeightOnlyMoEMethod,
    get_scale_perms,
    gptq_marlin_moe_repack,
    marlin_moe_permute_scales,
    marlin_permute_scales,
)


class TestGetScalePerms(unittest.TestCase):
    """Test get_scale_perms function."""

    def test_returns_two_lists(self):
        """get_scale_perms returns two lists."""
        scale_perm, scale_perm_single = get_scale_perms()
        self.assertIsInstance(scale_perm, list)
        self.assertIsInstance(scale_perm_single, list)

    def test_scale_perm_length(self):
        """scale_perm has 64 elements (8*8)."""
        scale_perm, _ = get_scale_perms()
        self.assertEqual(len(scale_perm), 64)

    def test_scale_perm_single_length(self):
        """scale_perm_single has 32 elements (4*8)."""
        _, scale_perm_single = get_scale_perms()
        self.assertEqual(len(scale_perm_single), 32)

    def test_scale_perm_values(self):
        """scale_perm contains correct permutation pattern."""
        scale_perm, _ = get_scale_perms()
        # First 8 elements: [0, 8, 16, 24, 32, 40, 48, 56]
        expected_first_8 = [0 + 8 * j for j in range(8)]
        self.assertEqual(scale_perm[:8], expected_first_8)
        # Second 8 elements: [1, 9, 17, 25, 33, 41, 49, 57]
        expected_second_8 = [1 + 8 * j for j in range(8)]
        self.assertEqual(scale_perm[8:16], expected_second_8)

    def test_scale_perm_single_values(self):
        """scale_perm_single contains correct permutation pattern."""
        _, scale_perm_single = get_scale_perms()
        # First 8 elements for i=0: [0, 1, 8, 9, 16, 17, 24, 25]
        expected_first_8 = [0, 1, 8, 9, 16, 17, 24, 25]
        self.assertEqual(scale_perm_single[:8], expected_first_8)

    def test_scale_perm_no_duplicates(self):
        """scale_perm has no duplicate values."""
        scale_perm, _ = get_scale_perms()
        self.assertEqual(len(scale_perm), len(set(scale_perm)))

    def test_scale_perm_single_no_duplicates(self):
        """scale_perm_single has no duplicate values."""
        _, scale_perm_single = get_scale_perms()
        self.assertEqual(len(scale_perm_single), len(set(scale_perm_single)))


class TestMarlinPermuteScales(unittest.TestCase):
    """Test marlin_permute_scales function."""

    def test_group_size_less_than_size_k(self):
        """Uses scale_perm when group_size < size_k."""
        # scale_perm has 64 elements, so input needs to be reshapable to [-1, 64]
        s = paddle.randn([128, 256])
        result = marlin_permute_scales(s, size_k=256, size_n=256, group_size=128)
        self.assertEqual(result.shape, [128, 256])

    def test_group_size_equals_size_k(self):
        """Uses scale_perm_single when group_size == size_k."""
        # scale_perm_single has 32 elements, so input needs to be reshapable to [-1, 32]
        s = paddle.randn([1, 32])
        result = marlin_permute_scales(s, size_k=32, size_n=32, group_size=32)
        self.assertEqual(result.shape, [1, 32])

    def test_group_size_minus_one(self):
        """Uses scale_perm_single when group_size == -1 (per-channel)."""
        s = paddle.randn([1, 64])
        result = marlin_permute_scales(s, size_k=64, size_n=64, group_size=-1)
        self.assertEqual(result.shape, [1, 64])


class TestMarlinMoePermuteScales(unittest.TestCase):
    """Test marlin_moe_permute_scales function."""

    def test_per_expert_permutation(self):
        """marlin_moe_permute_scales applies permutation per expert."""
        num_experts = 4
        s = paddle.randn([num_experts, 1, 64])
        result = marlin_moe_permute_scales(s, size_k=64, size_n=64, group_size=-1)
        self.assertEqual(result.shape, [num_experts, 1, 64])

    def test_single_expert(self):
        """marlin_moe_permute_scales handles single expert."""
        s = paddle.randn([1, 1, 32])
        result = marlin_moe_permute_scales(s, size_k=32, size_n=32, group_size=-1)
        self.assertEqual(result.shape, [1, 1, 32])


class TestGptqMarlinMoeRepack(unittest.TestCase):
    """Test gptq_marlin_moe_repack function."""

    @patch("fastdeploy.model_executor.ops.gpu.gptq_marlin_repack")
    def test_repacks_per_expert(self, mock_repack):
        """gptq_marlin_moe_repack calls repack for each expert."""
        num_experts = 4
        size_k = 512
        size_n = 256
        num_bits = 4
        output_last_dim = size_n * (num_bits // 2)  # 512

        mock_repack.return_value = paddle.zeros([size_k // 16, output_last_dim], dtype="int32")

        b_q_weight = paddle.zeros([num_experts, size_k // 16, output_last_dim], dtype="int32")
        perm = paddle.zeros([num_experts, 0], dtype="int32")

        result = gptq_marlin_moe_repack(b_q_weight, perm, size_k, size_n, num_bits)

        self.assertEqual(mock_repack.call_count, num_experts)
        self.assertEqual(result.shape, [num_experts, size_k // 16, output_last_dim])

    def test_asserts_size_k_multiple_of_16(self):
        """gptq_marlin_moe_repack asserts size_k % 16 == 0."""
        with self.assertRaises(AssertionError):
            gptq_marlin_moe_repack(
                paddle.zeros([2, 10, 20], dtype="int32"),
                paddle.zeros([2, 0], dtype="int32"),
                size_k=100,  # not multiple of 16
                size_n=10,
                num_bits=4,
            )


class TestMarlinWeightOnlyMoEMethodInit(unittest.TestCase):
    """Test MarlinWeightOnlyMoEMethod.__init__."""

    def test_init_default(self):
        """__init__ sets default attributes."""
        method = MarlinWeightOnlyMoEMethod()
        self.assertIsNone(method.quant_method)
        self.assertEqual(method.added_weight_attrs, ["up_gate_proj_weight", "down_proj_weight"])
        self.assertEqual(method.added_scale_attrs, ["up_gate_proj_weight_scale", "down_proj_weight_scale"])
        self.assertEqual(method.added_zeros_attrs, ["zeros0", "zeros1"])

    def test_init_with_quant_method(self):
        """__init__ stores quant_method."""
        mock_qm = MagicMock()
        method = MarlinWeightOnlyMoEMethod(quant_method=mock_qm)
        self.assertIs(method.quant_method, mock_qm)


class TestMarlinWeightOnlyMoEMethodCreateWeights(unittest.TestCase):
    """Test MarlinWeightOnlyMoEMethod.create_weights."""

    def test_create_weights_shapes(self):
        """create_weights sets correct weight and scale shapes."""
        method = MarlinWeightOnlyMoEMethod()

        layer = MagicMock()
        layer.num_local_experts = 8
        layer.hidden_size = 4096
        layer.moe_intermediate_size = 2048
        layer._helper.get_default_dtype.return_value = "float16"

        method.create_weights(layer)

        # up_gate: [8, 4096//16, 2048*4] = [8, 256, 8192]
        self.assertEqual(method.up_gate_proj_weight_shape, [8, 256, 8192])
        # down: [8, 2048//16, 4096*2] = [8, 128, 8192]
        self.assertEqual(method.down_proj_weight_shape, [8, 128, 8192])
        self.assertEqual(method.weight_dtype, "int32")
        self.assertEqual(method.default_dtype, "float16")

        # Verify setattr was called (create_parameter for 4 attrs: 2 weights + 2 scales)
        self.assertEqual(layer.create_parameter.call_count, 4)

    def test_create_weights_scale_shapes(self):
        """create_weights sets correct scale parameter shapes."""
        method = MarlinWeightOnlyMoEMethod()

        layer = MagicMock()
        layer.num_local_experts = 4
        layer.hidden_size = 2048
        layer.moe_intermediate_size = 1024
        layer._helper.get_default_dtype.return_value = "bfloat16"

        method.create_weights(layer)

        # Check the scale shapes from create_parameter calls
        calls = layer.create_parameter.call_args_list
        # Call 0: up_gate weight [4, 128, 4096]
        # Call 1: down weight [4, 64, 4096]
        # Call 2: up_gate scale [4, 1, 2048]
        self.assertEqual(calls[2][1]["shape"], [4, 1, 2048])
        # Call 3: down scale [4, 1, 2048]
        self.assertEqual(calls[3][1]["shape"], [4, 1, 2048])


class TestMarlinWeightOnlyMoEMethodApply(unittest.TestCase):
    """Test MarlinWeightOnlyMoEMethod.apply."""

    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.MoeWna16MarlinGemmApi")
    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.tritonmoe_preprocess_func")
    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.fastdeploy")
    def test_apply_non_noaux_tc(self, mock_fd, mock_preprocess, mock_gemm_api):
        """apply uses moe_topk_select for non-noaux_tc topk_method."""
        method = MarlinWeightOnlyMoEMethod()

        layer = MagicMock()
        layer.top_k = 2
        layer.moe_intermediate_size = 2048
        layer.hidden_size = 4096
        layer.num_experts = 8
        layer.topk_method = "greedy"
        layer.gate_correction_bias = None

        x = MagicMock()
        x.shape = [4, 4096]  # 4 tokens

        gate = MagicMock()
        gate.return_value = MagicMock()
        gate.return_value.cast.return_value = "gate_out_fp32"

        mock_fd.model_executor.ops.gpu.moe_topk_select.return_value = ("topk_ids", "topk_weights")
        mock_preprocess.return_value = ("sorted_ids", "expert_ids", "num_tokens_padded")

        # First gemm returns ffn_out, second gemm returns final
        mock_gemm_api.return_value = [MagicMock()]

        with patch("paddle.incubate.nn.functional.swiglu", return_value=MagicMock()):
            method.apply(layer, x, gate)

        mock_fd.model_executor.ops.gpu.moe_topk_select.assert_called_once_with("gate_out_fp32", None, 2, True, False)
        # gemm called twice (up_gate + down)
        self.assertEqual(mock_gemm_api.call_count, 2)

    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.MoeWna16MarlinGemmApi")
    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.tritonmoe_preprocess_func")
    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.fastdeploy")
    def test_apply_noaux_tc(self, mock_fd, mock_preprocess, mock_gemm_api):
        """apply uses get_moe_scores for noaux_tc topk_method."""
        method = MarlinWeightOnlyMoEMethod()

        layer = MagicMock()
        layer.top_k = 2
        layer.moe_intermediate_size = 2048
        layer.hidden_size = 4096
        layer.num_experts = 8
        layer.topk_method = "noaux_tc"
        layer.n_group = 4
        layer.topk_group = 2
        layer.routed_scaling_factor = 1.0
        layer.gate_correction_bias = None
        layer.renormalize = True

        x = MagicMock()
        x.shape = [4, 4096]

        gate = MagicMock()
        gate.return_value = MagicMock()
        gate.return_value.cast.return_value = "gate_out_fp32"

        mock_preprocess.return_value = ("sorted_ids", "expert_ids", "num_tokens_padded")
        mock_gemm_api.return_value = [MagicMock()]

        with patch(
            "fastdeploy.model_executor.layers.moe.moe.get_moe_scores",
            return_value=(None, "topk_weights", "topk_ids"),
        ) as mock_get_scores:
            with patch("paddle.incubate.nn.functional.swiglu", return_value=MagicMock()):
                method.apply(layer, x, gate)

            mock_get_scores.assert_called_once()

    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.MoeWna16MarlinGemmApi")
    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.tritonmoe_preprocess_func")
    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.fastdeploy")
    def test_apply_calls_topk_hookfunc(self, mock_fd, mock_preprocess, mock_gemm_api):
        """apply calls topk_ids_hookfunc when provided."""
        method = MarlinWeightOnlyMoEMethod()

        layer = MagicMock()
        layer.top_k = 2
        layer.moe_intermediate_size = 2048
        layer.hidden_size = 4096
        layer.num_experts = 8
        layer.topk_method = "greedy"
        layer.gate_correction_bias = None

        x = MagicMock()
        x.shape = [4, 4096]

        gate = MagicMock()
        gate.return_value = MagicMock()
        gate.return_value.cast.return_value = "gate_out_fp32"

        mock_fd.model_executor.ops.gpu.moe_topk_select.return_value = ("topk_ids", "topk_weights")
        mock_preprocess.return_value = ("sorted_ids", "expert_ids", "num_tokens_padded")
        mock_gemm_api.return_value = [MagicMock()]

        hookfunc = MagicMock()

        with patch("paddle.incubate.nn.functional.swiglu", return_value=MagicMock()):
            method.apply(layer, x, gate, topk_ids_hookfunc=hookfunc)

        hookfunc.assert_called_once_with(topk_ids="topk_ids")

    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.MoeWna16MarlinGemmApi")
    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.tritonmoe_preprocess_func")
    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.fastdeploy")
    def test_apply_block_size_selection(self, mock_fd, mock_preprocess, mock_gemm_api):
        """apply selects correct block_size_m based on token ratio."""
        method = MarlinWeightOnlyMoEMethod()

        layer = MagicMock()
        layer.top_k = 2
        layer.moe_intermediate_size = 2048
        layer.hidden_size = 4096
        layer.num_experts = 64
        layer.topk_method = "greedy"
        layer.gate_correction_bias = None

        # With 1 token, top_k=2, num_experts=64:
        # ratio = 1*2/64/m => for m=8: 0.0039 < 0.9 -> block_size_m=8
        x = MagicMock()
        x.shape = [1, 4096]

        gate = MagicMock()
        gate.return_value = MagicMock()
        gate.return_value.cast.return_value = "gate_out_fp32"

        mock_fd.model_executor.ops.gpu.moe_topk_select.return_value = ("topk_ids", "topk_weights")
        mock_preprocess.return_value = ("sorted_ids", "expert_ids", "num_tokens_padded")
        mock_gemm_api.return_value = [MagicMock()]

        with patch("paddle.incubate.nn.functional.swiglu", return_value=MagicMock()):
            method.apply(layer, x, gate)

        # Verify preprocess was called with block_size_m=8
        mock_preprocess.assert_called_once_with("topk_ids", 64, 8)


class TestMarlinWeightOnlyMoEMethodProcessLoadedWeights(unittest.TestCase):
    """Test MarlinWeightOnlyMoEMethod.process_loaded_weights."""

    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.marlin_moe_permute_scales")
    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.gptq_marlin_moe_repack")
    def test_process_loaded_weights(self, mock_repack, mock_permute_scales):
        """process_loaded_weights quantizes and repacks weights."""
        method = MarlinWeightOnlyMoEMethod()
        method.added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        method.added_scale_attrs = ["up_gate_proj_weight_scale", "down_proj_weight_scale"]

        layer = MagicMock()
        layer.num_local_experts = 2
        layer.hidden_size = 64
        layer.moe_intermediate_size = 32

        # Mock extract_moe_ffn_weights
        up_gate_weights = [paddle.randn([64, 64]) for _ in range(2)]
        down_weights = [paddle.randn([32, 64]) for _ in range(2)]
        layer.extract_moe_ffn_weights.return_value = (up_gate_weights, down_weights, None, None)

        mock_repack.return_value = paddle.zeros([2, 4, 128], dtype="int32")
        mock_permute_scales.return_value = paddle.zeros([2, 1, 64])

        method.process_loaded_weights(layer, state_dict={})

        # Should have been called twice (up_gate + down)
        self.assertEqual(mock_repack.call_count, 2)
        self.assertEqual(mock_permute_scales.call_count, 2)

    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.marlin_moe_permute_scales")
    @patch("fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend.gptq_marlin_moe_repack")
    def test_process_loaded_weights_assertion_experts(self, mock_repack, mock_permute_scales):
        """process_loaded_weights asserts expert count matches."""
        method = MarlinWeightOnlyMoEMethod()
        method.added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        method.added_scale_attrs = ["up_gate_proj_weight_scale", "down_proj_weight_scale"]

        layer = MagicMock()
        layer.num_local_experts = 4

        # Only 2 experts returned
        up_gate_weights = [paddle.randn([64, 64]) for _ in range(2)]
        down_weights = [paddle.randn([32, 64]) for _ in range(2)]
        layer.extract_moe_ffn_weights.return_value = (up_gate_weights, down_weights, None, None)

        with self.assertRaises(AssertionError):
            method.process_loaded_weights(layer, state_dict={})


if __name__ == "__main__":
    unittest.main()
