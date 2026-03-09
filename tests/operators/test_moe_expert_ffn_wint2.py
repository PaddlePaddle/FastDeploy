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

"""Unit tests for moe_expert_ffn_wint2 custom op.

Tests the CUTLASS Weight-Only INT2 quantized MoE FFN operator:
  1) First GEMM:  input x dequant(up_gate_proj_weight) -> fc1_out
  2) SwiGLU activation: fc1_out -> act_out
  3) Second GEMM: act_out x dequant(down_proj_weight) -> output

Reference source for the WINT2 dequant algorithm:
  - Triton kernel: fastdeploy/model_executor/ops/triton_ops/wint2_fused_moe_kernel.py
  - CUTLASS layout: fastdeploy/model_executor/layers/moe/fused_moe_wint2_backend.py
"""

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import moe_expert_ffn_wint2

paddle.seed(2026)
np.random.seed(2026)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cutlass_rearrange(w):
    """Apply CUTLASS WINT2 weight layout rearrangement.

    Matches CutlassWint2FusedMoeMethod.process_weights_after_loading():
      reshape [E, K//16, 16, N//8, 8] -> transpose [0,3,1,4,2] -> reshape
    """
    shape = w.shape
    E, Kp, N = shape
    w = w.reshape([E, Kp // 16, 16, N // 8, 8])
    w = paddle.transpose(w, perm=[0, 3, 1, 4, 2])
    return w.reshape(shape)


def _build_inputs(
    num_experts,
    hidden_size,
    inter_size,
    tokens_per_expert,
    dtype="bfloat16",
    use_3d=False,
    zero_input=False,
):
    """Create correctly-shaped tensors for moe_expert_ffn_wint2.

    Args:
        num_experts: Number of experts.
        hidden_size: Hidden dimension (must be divisible by 128).
        inter_size: Intermediate size after SwiGLU split.
        tokens_per_expert: List of token counts per expert.
        dtype: "bfloat16" or "float16".
        use_3d: Use 3D input [E, max_tokens, H] instead of 2D.
        zero_input: Set input to zeros (for zero-input invariant test).
    """
    gated_inter = inter_size * 2
    total_tokens = sum(tokens_per_expert)

    # --- Input ---
    if use_3d:
        max_tok = max(tokens_per_expert) if tokens_per_expert else 1
        shape = [num_experts, max_tok, hidden_size]
    else:
        shape = [total_tokens, hidden_size]
    if zero_input:
        permute_input = paddle.zeros(shape, dtype=dtype)
    else:
        permute_input = paddle.randn(shape, dtype=dtype)

    # --- Prefix sum ---
    tokens_expert_prefix_sum = paddle.to_tensor(np.cumsum(tokens_per_expert).astype("int64"))

    # --- Packed uint8 weights with CUTLASS rearrangement ---
    w_up = _cutlass_rearrange(
        paddle.randint(0, 256, [num_experts, hidden_size // 4, gated_inter], dtype="int32").cast("uint8")
    )
    w_down = _cutlass_rearrange(
        paddle.randint(0, 256, [num_experts, inter_size // 4, hidden_size], dtype="int32").cast("uint8")
    )

    # --- Super scales (channel-wise, input dtype) ---
    super_up = paddle.randn([num_experts, gated_inter], dtype=dtype) * 0.01
    super_down = paddle.randn([num_experts, hidden_size], dtype=dtype) * 0.01

    # --- Local scales (group-wise, uint8) ---
    local_up = paddle.randint(0, 256, [num_experts, hidden_size // 128, gated_inter], dtype="int32").cast("uint8")
    local_down = paddle.randint(0, 256, [num_experts, inter_size // 128, hidden_size], dtype="int32").cast("uint8")

    # --- Code scale and zero-point (channel-wise, float32) ---
    code_scale_up = paddle.randn([num_experts, gated_inter], dtype="float32") * 0.01
    code_zp_up = paddle.randn([num_experts, gated_inter], dtype="float32") * 0.01
    code_scale_down = paddle.randn([num_experts, hidden_size], dtype="float32") * 0.01
    code_zp_down = paddle.randn([num_experts, hidden_size], dtype="float32") * 0.01

    return dict(
        permute_input=permute_input,
        tokens_expert_prefix_sum=tokens_expert_prefix_sum,
        up_gate_proj_weight=w_up,
        down_proj_weight=w_down,
        up_gate_proj_bias=None,
        up_gate_proj_scale=super_up,
        down_proj_scale=super_down,
        up_gate_proj_local_scale=local_up,
        up_gate_proj_code_scale=code_scale_up,
        up_gate_proj_code_zp=code_zp_up,
        down_proj_local_scale=local_down,
        down_proj_code_scale=code_scale_down,
        down_proj_code_zp=code_zp_down,
    )


def _call_op(inputs, used_in_ep_low_latency=False):
    """Invoke moe_expert_ffn_wint2 with the given inputs dict."""
    return moe_expert_ffn_wint2(
        inputs["permute_input"],
        inputs["tokens_expert_prefix_sum"],
        inputs["up_gate_proj_weight"],
        inputs["down_proj_weight"],
        inputs["up_gate_proj_bias"],
        inputs["up_gate_proj_scale"],
        inputs["down_proj_scale"],
        inputs["up_gate_proj_local_scale"],
        inputs["up_gate_proj_code_scale"],
        inputs["up_gate_proj_code_zp"],
        inputs["down_proj_local_scale"],
        inputs["down_proj_code_scale"],
        inputs["down_proj_code_zp"],
        used_in_ep_low_latency,
    )


# ===================================================================
# Test Cases
# ===================================================================


class TestMoeExpertFFNWint2(unittest.TestCase):
    """Correctness and regression tests for the WINT2 MoE FFN op."""

    # Small dimensions for fast CI (all must be divisible by 128)
    E = 4
    H = 256
    INTER = 128
    TOKENS = [4, 6, 2, 4]  # per expert, total = 16

    def setUp(self):
        paddle.set_device("gpu")

    # -- Numerical correctness -----------------------------------------

    def test_zero_input_produces_zero_output(self):
        """Zero input => matmul=0, SwiGLU(0)=0, matmul=0 => output = 0.

        This is a mathematical invariant independent of weight values.
        """
        for dtype in ["bfloat16", "float16"]:
            with self.subTest(dtype=dtype):
                inputs = _build_inputs(
                    self.E,
                    self.H,
                    self.INTER,
                    self.TOKENS,
                    dtype=dtype,
                    zero_input=True,
                )
                out = _call_op(inputs).cast("float32").numpy()
                np.testing.assert_allclose(
                    out,
                    np.zeros_like(out),
                    atol=1e-5,
                    err_msg=f"Zero input must produce zero output ({dtype})",
                )

    def test_determinism(self):
        """Identical inputs must produce bit-identical outputs."""
        inputs = _build_inputs(self.E, self.H, self.INTER, self.TOKENS)
        out1 = _call_op(inputs).cast("float32").numpy()
        out2 = _call_op(inputs).cast("float32").numpy()
        np.testing.assert_array_equal(
            out1,
            out2,
            err_msg="Non-deterministic: two runs with same inputs differ",
        )

    def test_nonzero_input_gives_finite_nonzero_output(self):
        """Random non-zero inputs must produce finite, non-zero values."""
        inputs = _build_inputs(self.E, self.H, self.INTER, self.TOKENS)
        out = _call_op(inputs).cast("float32").numpy()
        self.assertTrue(np.all(np.isfinite(out)), "Output contains NaN or Inf")
        self.assertGreater(
            np.abs(out).max(),
            0,
            "All-zero output from non-zero input",
        )

    # -- Shape and dtype -----------------------------------------------

    def test_output_shape_2d(self):
        """2D input [total_tokens, H] => output shape matches."""
        inputs = _build_inputs(self.E, self.H, self.INTER, self.TOKENS)
        out = _call_op(inputs)
        self.assertEqual(list(out.shape), list(inputs["permute_input"].shape))
        self.assertEqual(out.dtype, inputs["permute_input"].dtype)

    def test_output_shape_3d(self):
        """3D input [E, max_tokens, H] => output shape matches."""
        inputs = _build_inputs(
            self.E,
            self.H,
            self.INTER,
            self.TOKENS,
            use_3d=True,
        )
        out = _call_op(inputs)
        self.assertEqual(list(out.shape), list(inputs["permute_input"].shape))

    def test_dtype_bf16(self):
        """Op supports bfloat16 input/output."""
        inputs = _build_inputs(
            self.E,
            self.H,
            self.INTER,
            self.TOKENS,
            dtype="bfloat16",
        )
        out = _call_op(inputs)
        self.assertEqual(out.dtype, paddle.bfloat16)

    def test_dtype_fp16(self):
        """Op supports float16 input/output."""
        inputs = _build_inputs(
            self.E,
            self.H,
            self.INTER,
            self.TOKENS,
            dtype="float16",
        )
        out = _call_op(inputs)
        self.assertEqual(out.dtype, paddle.float16)

    # -- Edge cases ----------------------------------------------------

    def test_sparse_experts(self):
        """Experts with zero tokens are handled correctly."""
        sparse = [8, 0, 0, 8]
        inputs = _build_inputs(self.E, self.H, self.INTER, sparse)
        out = _call_op(inputs)
        self.assertEqual(list(out.shape), list(inputs["permute_input"].shape))
        self.assertTrue(np.all(np.isfinite(out.cast("float32").numpy())))

    def test_single_token_single_expert(self):
        """Minimal case: 1 expert, 1 token."""
        inputs = _build_inputs(1, self.H, self.INTER, [1])
        out = _call_op(inputs)
        self.assertEqual(list(out.shape), [1, self.H])

    def test_low_latency_mode(self):
        """Low-latency mode (GroupSwigluWithMasked) with 3D input."""
        inputs = _build_inputs(
            self.E,
            self.H,
            self.INTER,
            self.TOKENS,
            use_3d=True,
        )
        out = _call_op(inputs, used_in_ep_low_latency=True)
        self.assertEqual(list(out.shape), list(inputs["permute_input"].shape))
        # In 3D mode, padded slots (beyond each expert's token count) may
        # overflow in the first GEMM before GroupSwigluWithMasked zeros them,
        # causing NaN propagation.  Only validate the unpadded positions.
        out_np = out.cast("float32").numpy()
        for i, n_tok in enumerate(self.TOKENS):
            valid = out_np[i, :n_tok, :]
            self.assertTrue(
                np.all(np.isfinite(valid)),
                f"Expert {i}: non-finite values in first {n_tok} valid tokens",
            )

    def test_uneven_tokens(self):
        """Different number of tokens per expert."""
        uneven = [1, 5, 3, 7]
        inputs = _build_inputs(self.E, self.H, self.INTER, uneven)
        out = _call_op(inputs)
        self.assertEqual(list(out.shape), [sum(uneven), self.H])
        self.assertTrue(np.all(np.isfinite(out.cast("float32").numpy())))


if __name__ == "__main__":
    unittest.main()
