# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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


import sys
import types
import unittest
from unittest import mock

import paddle

import fastdeploy.model_executor.layers.moe.fused_moe_triton_backend as moe_backend_module
import fastdeploy.model_executor.layers.quantization.fp8_utils as fp8_utils_module
from fastdeploy.model_executor.layers.moe.fused_moe_triton_backend import (
    BlockWiseFP8MoEMethod,
)
from fastdeploy.model_executor.layers.quantization.block_wise_fp8 import (
    BlockWiseFP8Config,
    BlockWiseFP8LinearMethod,
)


class DummyLinearLayer(paddle.nn.Layer):
    def __init__(self, fd_config, weight_shape, with_bias=False):
        super().__init__()
        self.weight_shape = weight_shape
        self.with_bias = with_bias
        self.bias = None
        self.fd_config = fd_config
        self.weight = paddle.randn(self.weight_shape, paddle.bfloat16)


class DummyFusedMoELayer(paddle.nn.Layer):
    def __init__(self, fd_config, num_local_experts, moe_intermediate_size, hidden_size):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.hidden_size = hidden_size
        self.gate_correction_bias = paddle.zeros([1], dtype=paddle.float32)
        self.top_k = 1
        self.ep_size = 1
        self.ep_rank = 0
        self.tp_size = 1
        self.tp_rank = 0
        self.fd_config = fd_config

        self.up_gate_proj_weight = paddle.randn(
            [num_local_experts, hidden_size, moe_intermediate_size * 2], dtype="bfloat16"
        )
        self.down_proj_weight = paddle.randn([num_local_experts, moe_intermediate_size, hidden_size], dtype="bfloat16")


class TestFP8LinearWithUe8m0Scale(unittest.TestCase):
    def setUp(self):
        self.quant_config = BlockWiseFP8Config(weight_block_size=[128, 128], is_checkpoint_bf16=True)
        self.quant_config.deepgemm_scale_ue8m0 = True  # set deepgemm_scale_ue8m0 to True

    def test_create_layer_with_ue8m0_scale(self):
        def fake_per_block_cast_to_fp8(x, use_ue8m0=True):
            out_w = x.astype(paddle.float8_e4m3fn)
            out_s = paddle.ones([(x.shape[0] // 128), (x.shape[1] // 128)], dtype=paddle.float32)
            return out_w, out_s

        fd_config = mock.MagicMock()
        fd_config.load_config.load_choices.return_value = "default_v1"
        layer = DummyLinearLayer(fd_config=fd_config, weight_shape=[128, 1024])
        method = BlockWiseFP8LinearMethod(quant_config=self.quant_config)

        if "fastdeploy.model_executor.ops.gpu.deep_gemm.utils" in sys.modules:
            # This is for sm90, which DeepGEMM does not support ue8m0 scale
            fake = types.ModuleType("fastdeploy.model_executor.ops.gpu.deep_gemm")
            fake2 = types.ModuleType("fastdeploy.model_executor.ops.gpu.deep_gemm.utils.math")
            fake2.per_block_cast_to_fp8 = fake_per_block_cast_to_fp8
            fake3 = types.ModuleType("fastdeploy.model_executor.ops.gpu.deep_gemm.utils")
            fake3.align = lambda x, y: (x + y - 1) // y * y
            fake3.get_tma_aligned_size = lambda x, y: (x + 16 // y - 1) // (16 // y) * (16 // y)

            fake.utils = fake3
            fake3.math = fake2

            sys.modules["fastdeploy.model_executor.ops.gpu.deep_gemm"].utils = fake3

        method.model_format = "torch"
        method.process_weights_after_loading(layer)
        self.assertTrue(layer.weight_scale_inv.dtype == paddle.int32)
        self.assertEqual(layer.weight_scale_inv.shape, [128, 2])  # 1024 / 128 / 4


class TestFP8FusedMoeWithUe8m0Scale(unittest.TestCase):
    def setUp(self):
        self.quant_config = BlockWiseFP8Config(weight_block_size=[128, 128], is_checkpoint_bf16=True)
        self.quant_config.deepgemm_scale_ue8m0 = True  # set deepgemm_scale_ue8m0 to True

    def test_create_layer_with_ue8m0_scale(self):
        # This test covers the quant_weight_ue8m0 branch in BlockWiseFP8MoEMethod.process_weights_after_loading
        # The branch is entered when:
        # - deepgemm_scale_ue8m0=True
        # - FD_USE_PHI_FP8_QUANT=False (so we don't use fused_stack_transpose_quant)
        # - is_checkpoint_bf16=True

        def fake_quant_weight_ue8m0(weight_dequant, weight_block_size):
            # Mock quant_weight_ue8m0 behavior
            n, k = weight_dequant.shape[-2], weight_dequant.shape[-1]
            out_w = weight_dequant.astype(paddle.float8_e4m3fn)
            # Scale shape: [ceil_div(n, 128), ceil_div(k, 128)]
            out_s = paddle.ones([(n + 127) // 128, (k + 127) // 128], dtype="float32")
            return out_w, out_s

        def fake_transform_scale_ue8m0(sf, mn, weight_block_size=None):
            # Mock transform_scale_ue8m0 behavior
            # For input [mn, k] where k is small (e.g., 2):
            # After index_select: sf becomes [mn, k]
            # After TMA align and pack: result is [mn, align(k, 4)//4]
            # For k=2, align(2,4)=4, so result is [mn, 1]
            if weight_block_size:
                indices = paddle.arange(mn) // 128
                sf = paddle.index_select(sf, -2, indices)
            # Final shape: [mn, align(k,4)//4]
            aligned_k = ((sf.shape[-1] + 3) // 4) * 4 if sf.shape[-1] < 4 else ((sf.shape[-1] + 4 - 1) // 4) * 4
            result_shape = [sf.shape[0], aligned_k // 4]
            return paddle.zeros(result_shape, dtype=paddle.int32)

        fd_config = mock.MagicMock()
        fd_config.load_config.load_choices.return_value = "default_v1"
        layer = DummyFusedMoELayer(
            fd_config=fd_config, num_local_experts=1, moe_intermediate_size=256, hidden_size=256
        )
        method = BlockWiseFP8MoEMethod(quant_config=self.quant_config)

        # Call create_weights to initialize method attributes and layer parameters
        method.create_weights(layer, model_format="torch")

        # Override the down_proj scale shape to match expected [1, 256, 1]
        method.down_proj_scale_shape = [1, 256, 1]

        # Mock quant_weight_ue8m0 and transform_scale_ue8m0 for the else branch
        # Patch them after create_weights is called
        with mock.patch.object(moe_backend_module, "quant_weight_ue8m0", fake_quant_weight_ue8m0):
            with mock.patch.object(moe_backend_module, "transform_scale_ue8m0", fake_transform_scale_ue8m0):
                with mock.patch.object(moe_backend_module, "free_tensor", lambda tensor: None):
                    import fastdeploy.envs as _fd_envs

                    # Set FD_USE_PHI_FP8_QUANT=False to enter the quant_weight_ue8m0 branch
                    with mock.patch.object(_fd_envs, "FD_USE_PHI_FP8_QUANT", False):
                        method.model_format = "torch"
                        method.process_weights_after_loading(layer)

        self.assertTrue(layer.down_proj_weight_scale_inv.dtype == paddle.int32)
        self.assertEqual(layer.down_proj_weight_scale_inv.shape, method.down_proj_scale_shape)

    def _make_fake_fused_stack_transpose_quant(self, weight_shape):
        """Return a fake fused_stack_transpose_quant compatible with weight_shape [N, H, W]."""

        def fake_fused(expert_list, use_ue8m0=False):
            n = len(expert_list)
            # weight_shape: [total_experts, dim_a, dim_b]
            # fused_stack_transpose_quant returns a flat tensor that gets reshaped to
            # [chunk_n, dim_a, dim_b] inside _process_quantize.
            # We return [n*dim_a, dim_b] so reshape([n, -1, dim_b]) works.
            _, dim_a, dim_b = weight_shape
            w = paddle.zeros([n * dim_a, dim_b], dtype=paddle.float8_e4m3fn)
            s = paddle.ones([n * dim_a, max(1, dim_b // 128)], dtype=paddle.float32)
            return w, s

        return fake_fused

    def test_fleet_fp8_quant_single_chunk(self):
        """FD_USE_PHI_FP8_QUANT=True, num_experts <= 64: single chunk path runs without error."""
        import fastdeploy.envs as _fd_envs

        fd_config = mock.MagicMock()
        fd_config.load_config.load_choices.return_value = "default_v1"
        num_experts = 4
        hidden_size = 256
        intermediate_size = 256

        layer = DummyFusedMoELayer(
            fd_config=fd_config,
            num_local_experts=num_experts,
            moe_intermediate_size=intermediate_size,
            hidden_size=hidden_size,
        )
        method = BlockWiseFP8MoEMethod(quant_config=self.quant_config)
        up_gate_shape = [num_experts, hidden_size, intermediate_size * 2]
        down_shape = [num_experts, intermediate_size, hidden_size]
        method.up_gate_proj_weight_shape = up_gate_shape
        method.down_proj_weight_shape = down_shape
        method.up_gate_proj_scale_shape = [num_experts, intermediate_size * 2, 1]
        method.down_proj_scale_shape = [num_experts, hidden_size, 1]

        method.model_format = "torch"
        # Force gate_up branch so the fleet-FP8 chunk path is exercised for up_gate_proj_weight
        with mock.patch.object(_fd_envs, "FD_USE_PHI_FP8_QUANT", True):
            with mock.patch.object(moe_backend_module, "weight_fully_copied", return_value=True):
                with mock.patch.object(
                    moe_backend_module,
                    "fused_stack_transpose_quant",
                    self._make_fake_fused_stack_transpose_quant(up_gate_shape),
                ):
                    method.process_weights_after_loading(layer)

        # Weight must be fp8 and scale must cover all experts
        self.assertEqual(layer.up_gate_proj_weight.dtype, paddle.float8_e4m3fn)
        self.assertEqual(layer.up_gate_proj_weight_scale_inv.shape[0], num_experts)

    def test_fleet_fp8_quant_multi_chunk(self):
        """FD_USE_PHI_FP8_QUANT=True, num_experts=70>64: two chunks processed and concat'd."""
        import fastdeploy.envs as _fd_envs

        fd_config = mock.MagicMock()
        fd_config.load_config.load_choices.return_value = "default_v1"
        num_experts = 70  # chunk_size=64 → two chunks: 64 + 6
        hidden_size = 128
        intermediate_size = 128

        layer = DummyFusedMoELayer(
            fd_config=fd_config,
            num_local_experts=num_experts,
            moe_intermediate_size=intermediate_size,
            hidden_size=hidden_size,
        )
        method = BlockWiseFP8MoEMethod(quant_config=self.quant_config)
        up_gate_shape = [num_experts, hidden_size, intermediate_size * 2]
        down_shape = [num_experts, intermediate_size, hidden_size]
        method.up_gate_proj_weight_shape = up_gate_shape
        method.down_proj_weight_shape = down_shape
        method.up_gate_proj_scale_shape = [num_experts, intermediate_size * 2, 1]
        method.down_proj_scale_shape = [num_experts, hidden_size, 1]

        chunks_seen = []
        original_fake = self._make_fake_fused_stack_transpose_quant(up_gate_shape)

        def recording_fake(expert_list, use_ue8m0=False):
            chunks_seen.append(len(expert_list))
            return original_fake(expert_list, use_ue8m0)

        method.model_format = "torch"
        # Force gate_up branch to exercise the multi-chunk concat logic
        with mock.patch.object(_fd_envs, "FD_USE_PHI_FP8_QUANT", True):
            with mock.patch.object(moe_backend_module, "weight_fully_copied", return_value=True):
                with mock.patch.object(moe_backend_module, "fused_stack_transpose_quant", recording_fake):
                    method.process_weights_after_loading(layer)

        # Expect exactly two chunks: 64 then 6
        self.assertIn(64, chunks_seen, "First chunk should be 64 experts")
        self.assertIn(6, chunks_seen, "Second chunk should be remaining 6 experts")

        # Final scale param shape[0] must equal num_experts (result of concat across chunks)
        self.assertEqual(layer.up_gate_proj_weight_scale_inv.shape[0], num_experts)


class TestFusedStackTransposeQuant(unittest.TestCase):
    """Unit tests for fp8_utils.fused_stack_transpose_quant."""

    def _make_expert_weights(self, num_experts=4, out_features=128, in_features=64):
        """Create a list of bfloat16 weight tensors as expert inputs."""
        return [paddle.randn([out_features, in_features], dtype="bfloat16") for _ in range(num_experts)]

    # ------------------------------------------------------------------
    # Helper: build a minimal fake paddlefleet_ops namespace
    # ------------------------------------------------------------------
    def _fake_paddlefleet_ops(self, *, has_op=True, use_pow2_scale_result=False, num_experts=4, out=128, inp=64):
        """Return a mock object that optionally exposes fuse_stack_transpose_fp8_quant."""
        fake_ops = mock.MagicMock()
        if has_op:
            stacked_w = paddle.zeros([num_experts, inp, out], dtype=paddle.float8_e4m3fn)
            scale = paddle.ones([num_experts * inp, out // 128 if out >= 128 else 1], dtype=paddle.float32)

            def fake_quant(expert_weight_list, use_pow2_scale, use_ue8m0_w, use_ue8m0_s):
                return stacked_w, scale

            fake_ops.fuse_stack_transpose_fp8_quant = fake_quant
        else:
            # Simulate that the attribute is absent
            del fake_ops.fuse_stack_transpose_fp8_quant
        return fake_ops

    # ------------------------------------------------------------------
    # Test: op not available → RuntimeError
    # ------------------------------------------------------------------
    def test_raises_when_op_unavailable(self):
        """fused_stack_transpose_quant should raise RuntimeError when paddlefleet_ops
        does not expose fuse_stack_transpose_fp8_quant."""
        fake_ops = mock.MagicMock(spec=[])  # empty spec → no attributes
        expert_weights = self._make_expert_weights()
        with mock.patch.object(fp8_utils_module, "paddlefleet_ops", fake_ops):
            with self.assertRaises(RuntimeError) as ctx:
                fp8_utils_module.fused_stack_transpose_quant(expert_weights)
        self.assertIn("fuse_stack_transpose_fp8_quant", str(ctx.exception))

    # ------------------------------------------------------------------
    # Test: normal path (use_ue8m0=False, non-Blackwell)
    # ------------------------------------------------------------------
    def test_normal_path_no_ue8m0(self):
        """Returns (w, scale) without transposing scale when use_ue8m0=False."""
        num_experts, out, inp = 4, 128, 64
        stacked_w = paddle.zeros([num_experts, inp, out], dtype=paddle.float8_e4m3fn)
        scale = paddle.ones([num_experts, out // 128 if out >= 128 else 1, inp], dtype=paddle.float32)
        scale_shape_before = list(scale.shape)

        call_kwargs = {}

        def fake_quant(expert_weight_list, use_pow2_scale, use_ue8m0_w, use_ue8m0_s):
            call_kwargs["use_pow2_scale"] = use_pow2_scale
            call_kwargs["use_ue8m0_w"] = use_ue8m0_w
            call_kwargs["use_ue8m0_s"] = use_ue8m0_s
            return stacked_w, scale

        fake_ops = mock.MagicMock()
        fake_ops.fuse_stack_transpose_fp8_quant = fake_quant
        expert_weights = self._make_expert_weights(num_experts, out, inp)

        with mock.patch.object(fp8_utils_module, "paddlefleet_ops", fake_ops):
            # Force non-Blackwell (SM < 100)
            with mock.patch.object(fp8_utils_module, "get_sm_version", return_value=90):
                with mock.patch.object(fp8_utils_module.current_platform, "is_cuda", return_value=True):
                    w_out, scale_out = fp8_utils_module.fused_stack_transpose_quant(expert_weights, use_ue8m0=False)

        self.assertIs(w_out, stacked_w)
        self.assertIs(scale_out, scale)
        # Scale must NOT be transposed when use_ue8m0 is False
        self.assertEqual(list(scale_out.shape), scale_shape_before)
        self.assertFalse(call_kwargs["use_pow2_scale"])
        self.assertFalse(call_kwargs["use_ue8m0_w"])
        self.assertFalse(call_kwargs["use_ue8m0_s"])

    # ------------------------------------------------------------------
    # Test: use_ue8m0=True → scale is transposed
    # ------------------------------------------------------------------
    def test_ue8m0_transposes_scale(self):
        """When use_ue8m0=True the returned scale tensor should be transposed."""
        num_experts, out, inp = 2, 256, 128
        stacked_w = paddle.zeros([num_experts, inp, out], dtype=paddle.float8_e4m3fn)
        # Deliberate non-square shape so transposition is detectable
        scale = paddle.ones([num_experts * inp, out // 128], dtype=paddle.float32)
        original_shape = list(scale.shape)

        def fake_quant(expert_weight_list, use_pow2_scale, use_ue8m0_w, use_ue8m0_s):
            return stacked_w, scale

        fake_ops = mock.MagicMock()
        fake_ops.fuse_stack_transpose_fp8_quant = fake_quant
        expert_weights = self._make_expert_weights(num_experts, out, inp)

        with mock.patch.object(fp8_utils_module, "paddlefleet_ops", fake_ops):
            with mock.patch.object(fp8_utils_module, "get_sm_version", return_value=90):
                with mock.patch.object(fp8_utils_module.current_platform, "is_cuda", return_value=True):
                    w_out, scale_out = fp8_utils_module.fused_stack_transpose_quant(expert_weights, use_ue8m0=True)

        # Shape should be the transpose of original
        self.assertEqual(list(scale_out.shape), list(reversed(original_shape)))

    # ------------------------------------------------------------------
    # Test: Blackwell GPU (SM 10) → use_pow2_scale=True
    # ------------------------------------------------------------------
    def test_blackwell_sets_pow2_scale(self):
        """On SM 10 (Blackwell) the op must be called with use_pow2_scale=True."""
        num_experts, out, inp = 2, 128, 64
        stacked_w = paddle.zeros([num_experts, inp, out], dtype=paddle.float8_e4m3fn)
        scale = paddle.ones([num_experts, 1, inp], dtype=paddle.float32)

        received = {}

        def fake_quant(expert_weight_list, use_pow2_scale, use_ue8m0_w, use_ue8m0_s):
            received["use_pow2_scale"] = use_pow2_scale
            return stacked_w, scale

        fake_ops = mock.MagicMock()
        fake_ops.fuse_stack_transpose_fp8_quant = fake_quant
        expert_weights = self._make_expert_weights(num_experts, out, inp)

        with mock.patch.object(fp8_utils_module, "paddlefleet_ops", fake_ops):
            with mock.patch.object(fp8_utils_module, "get_sm_version", return_value=100):
                with mock.patch.object(fp8_utils_module.current_platform, "is_cuda", return_value=True):
                    fp8_utils_module.fused_stack_transpose_quant(expert_weights, use_ue8m0=False)

        self.assertTrue(received["use_pow2_scale"])

    # ------------------------------------------------------------------
    # Test: non-Blackwell GPU (SM 9) → use_pow2_scale=False
    # ------------------------------------------------------------------
    def test_non_blackwell_no_pow2_scale(self):
        """On SM < 10 the op must be called with use_pow2_scale=False."""
        num_experts, out, inp = 2, 128, 64
        stacked_w = paddle.zeros([num_experts, inp, out], dtype=paddle.float8_e4m3fn)
        scale = paddle.ones([num_experts, 1, inp], dtype=paddle.float32)

        received = {}

        def fake_quant(expert_weight_list, use_pow2_scale, use_ue8m0_w, use_ue8m0_s):
            received["use_pow2_scale"] = use_pow2_scale
            return stacked_w, scale

        fake_ops = mock.MagicMock()
        fake_ops.fuse_stack_transpose_fp8_quant = fake_quant
        expert_weights = self._make_expert_weights(num_experts, out, inp)

        with mock.patch.object(fp8_utils_module, "paddlefleet_ops", fake_ops):
            with mock.patch.object(fp8_utils_module, "get_sm_version", return_value=90):
                with mock.patch.object(fp8_utils_module.current_platform, "is_cuda", return_value=True):
                    fp8_utils_module.fused_stack_transpose_quant(expert_weights, use_ue8m0=False)

        self.assertFalse(received["use_pow2_scale"])

    # ------------------------------------------------------------------
    # Test: op receives the correct expert_weight_list argument
    # ------------------------------------------------------------------
    def test_expert_weight_list_forwarded(self):
        """The expert weight list must be passed as-is to the underlying op."""
        stacked_w = paddle.zeros([2, 64, 128], dtype=paddle.float8_e4m3fn)
        scale = paddle.ones([2, 1, 64], dtype=paddle.float32)
        received_list = {}

        def fake_quant(expert_weight_list, use_pow2_scale, use_ue8m0_w, use_ue8m0_s):
            received_list["weights"] = expert_weight_list
            return stacked_w, scale

        fake_ops = mock.MagicMock()
        fake_ops.fuse_stack_transpose_fp8_quant = fake_quant
        expert_weights = self._make_expert_weights(num_experts=2, out_features=128, in_features=64)

        with mock.patch.object(fp8_utils_module, "paddlefleet_ops", fake_ops):
            with mock.patch.object(fp8_utils_module, "get_sm_version", return_value=90):
                with mock.patch.object(fp8_utils_module.current_platform, "is_cuda", return_value=True):
                    fp8_utils_module.fused_stack_transpose_quant(expert_weights)

        self.assertIs(received_list["weights"], expert_weights)

    # ------------------------------------------------------------------
    # Test: ue8m0=True on Blackwell → both flags propagate correctly
    # ------------------------------------------------------------------
    def test_ue8m0_and_blackwell_combined(self):
        """use_ue8m0=True on Blackwell: pow2_scale=True and ue8m0 flags=True, scale transposed."""
        num_experts, out, inp = 2, 256, 128
        stacked_w = paddle.zeros([num_experts, inp, out], dtype=paddle.float8_e4m3fn)
        scale = paddle.ones([num_experts * inp, out // 128], dtype=paddle.float32)
        original_shape = list(scale.shape)
        received = {}

        def fake_quant(expert_weight_list, use_pow2_scale, use_ue8m0_w, use_ue8m0_s):
            received.update(use_pow2_scale=use_pow2_scale, use_ue8m0_w=use_ue8m0_w, use_ue8m0_s=use_ue8m0_s)
            return stacked_w, scale

        fake_ops = mock.MagicMock()
        fake_ops.fuse_stack_transpose_fp8_quant = fake_quant
        expert_weights = self._make_expert_weights(num_experts, out, inp)

        with mock.patch.object(fp8_utils_module, "paddlefleet_ops", fake_ops):
            with mock.patch.object(fp8_utils_module, "get_sm_version", return_value=100):
                with mock.patch.object(fp8_utils_module.current_platform, "is_cuda", return_value=True):
                    w_out, scale_out = fp8_utils_module.fused_stack_transpose_quant(expert_weights, use_ue8m0=True)

        self.assertTrue(received["use_pow2_scale"])
        self.assertTrue(received["use_ue8m0_w"])
        self.assertTrue(received["use_ue8m0_s"])
        self.assertEqual(list(scale_out.shape), list(reversed(original_shape)))


if __name__ == "__main__":
    unittest.main()
