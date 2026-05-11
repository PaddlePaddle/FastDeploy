"""
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
"""

from typing import Any, Optional

import paddle

paddle.enable_compat(scope={"flashinfer"})


def _dtype_str(dtype) -> str:
    """Normalize dtype to string, handling both paddle and torch proxy dtypes."""
    return str(dtype).split(".")[-1]


def _is_dtype(tensor, *dtype_names: str) -> bool:
    """Check tensor dtype by name, compatible with both paddle and torch proxy tensors."""
    return _dtype_str(tensor.dtype) in dtype_names


def _perm(tensor, *dims):
    """Permute tensor dims, compatible with both paddle (transpose) and torch proxy (permute)."""
    try:
        return tensor.transpose(list(dims))
    except TypeError:
        return tensor.permute(*dims)


def get_cute_dtype(input) -> str:
    s = _dtype_str(input.dtype)
    if s == "bfloat16":
        return "bfloat16"
    elif s == "float16":
        return "float16"
    elif s == "float32":
        return "float32"
    else:
        raise ValueError(f"Unsupported cute dtype {input.dtype}")


def flashinfer_cutedsl_moe_masked(
    hidden_states: tuple,
    input_global_scale: paddle.Tensor,
    w1: paddle.Tensor,
    w1_blockscale: paddle.Tensor,
    w1_alpha: paddle.Tensor,
    w2: paddle.Tensor,
    a2_global_scale: paddle.Tensor,
    w2_blockscale: paddle.Tensor,
    w2_alpha: paddle.Tensor,
    masked_m: paddle.Tensor,
    down_sm_count: Optional[int] = None,
    down_signals: Optional[paddle.Tensor] = None,
    down_start_event: Optional[Any] = None,
):
    """
    Perform masked Mixture-of-Experts computation with FlashInfer's CuteDSL kernels.

    Args:
        hidden_states: Either of the following:
            * (paddle.Tensor, None): [num_experts, m, k] bf16 — not pre-quantized
            * (paddle.Tensor, paddle.Tensor): [m, k//2, num_experts] uint8,
              [m, k//16, num_experts] float8_e4m3fn — pre-quantized FP4 from dispatch
        input_global_scale: (l,) float32, value is 1/input_scale per expert
        w1: [l, 2*n, k//2] uint8, FP4-packed gate+up projection weights
        w1_blockscale: float8_e4m3fn blockscale for w1
        w1_alpha: (l,) float32, = input_scale * w1_weight_scale_2
        w2: [l, k, n//2] uint8, FP4-packed down projection weights
        a2_global_scale: (l,) float32, 1/input_scale for GEMM2
        w2_blockscale: float8_e4m3fn blockscale for w2
        w2_alpha: (l,) float32, = input_scale * w2_weight_scale_2
        masked_m: (l,) int32, valid token count per expert; max(masked_m) == m

    Returns:
        paddle.Tensor: [num_experts, m, k] bf16
    """
    from flashinfer import (
        scaled_fp4_grouped_quantize,
        silu_and_mul_scaled_nvfp4_experts_quantize,
    )
    from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

    # === Dtype assertions ===
    # Use string-based dtype check to be compatible with both paddle and torch proxy tensors
    assert _is_dtype(w1, "uint8"), f"w1 must be uint8 (fp4 packed), got {w1.dtype}"
    assert _is_dtype(w1_blockscale, "float8_e4m3fn"), f"w1_blockscale must be float8_e4m3fn, got {w1_blockscale.dtype}"
    assert _is_dtype(w1_alpha, "float32"), f"w1_alpha must be float32, got {w1_alpha.dtype}"
    assert _is_dtype(w2, "uint8"), f"w2 must be uint8 (fp4 packed), got {w2.dtype}"
    assert _is_dtype(a2_global_scale, "float32"), f"a2_global_scale must be float32, got {a2_global_scale.dtype}"
    assert _is_dtype(w2_blockscale, "float8_e4m3fn"), f"w2_blockscale must be float8_e4m3fn, got {w2_blockscale.dtype}"
    assert _is_dtype(w2_alpha, "float32"), f"w2_alpha must be float32, got {w2_alpha.dtype}"
    assert len(hidden_states) == 2, f"hidden_states must be a tuple of length 2, got {len(hidden_states)}"

    # intermediate_size derived from w2 last dimension
    n = w2.shape[-1] * 2

    if hidden_states[1] is not None:
        # Pre-quantized path: tokens already FP4-packed by dispatch
        # a_q:   [m, k//2, num_experts] uint8
        # a_q_sf:[m, k//16, num_experts] float8_e4m3fn
        a_q = hidden_states[0].view(paddle.uint8)
        a_q_sf = hidden_states[1].view(paddle.float8_e4m3fn)
        m, k_by_2, num_experts = a_q.shape
        k = k_by_2 * 2
    else:
        # Standard path: bf16 [num_experts, m, k], quantize to FP4 here
        num_experts, m, k = hidden_states[0].shape

        assert _is_dtype(
            input_global_scale, "float32"
        ), f"input_global_scale must be float32, got {input_global_scale.dtype}"
        assert list(input_global_scale.shape) == [
            num_experts
        ], f"input_global_scale must be (l,), got {input_global_scale.shape}"

        a_q, a_q_sf = scaled_fp4_grouped_quantize(
            hidden_states[0],
            masked_m,
            input_global_scale,
        )

    assert w1.shape[-2] == 2 * n, f"w1 last-2 dim must be 2*n={2*n}, got {w1.shape[-2]}"
    assert w1.shape[-1] * 2 == k, f"w1 last dim * 2 must equal k={k}, got {w1.shape[-1] * 2}"
    assert (
        w2.shape[-2] == k and w2.shape[-1] == n // 2
    ), f"w2 shape mismatch, got {list(w2.shape[-2:])}, expected [{k}, {n // 2}]"
    assert list(w1_alpha.shape) == [num_experts], f"w1_alpha must be (l,), got {w1_alpha.shape}"
    assert list(a2_global_scale.shape) == [num_experts], f"a2_global_scale must be (l,), got {a2_global_scale.shape}"
    assert list(w2_alpha.shape) == [num_experts], f"w2_alpha must be (l,), got {w2_alpha.shape}"

    assert _is_dtype(a_q, "uint8")
    assert _is_dtype(a_q_sf, "float8_e4m3fn")

    ab_dtype = "float4_e2m1fn"
    sf_dtype = "float8_e4m3fn"
    c_dtype = "bfloat16"
    sf_vec_size = 16

    # === GEMM1: gate+up projection ===
    # grouped_gemm_nt_masked requires output in [m, 2*n, l] layout
    gateup_output = paddle.empty([num_experts, m, n * 2], dtype=paddle.bfloat16)
    gateup_output = gateup_output.transpose([1, 2, 0])  # [m, 2*n, num_experts]

    # w1:           [E, 2*n, k//2]  → _perm(., 1, 2, 0) → [2*n, k//2, E]
    # w1_blockscale:[E, 2*n, k//G]  → _perm(., 1, 2, 0) → [2*n, k//G, E]
    # Both must share the same expert-last layout for grouped_gemm_nt_masked.
    grouped_gemm_nt_masked(
        (a_q, a_q_sf),
        (_perm(w1, 1, 2, 0), _perm(w1_blockscale, 1, 2, 0)),
        gateup_output,
        masked_m,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        alpha=w1_alpha.reshape([1, 1, num_experts]),
        alpha_dtype=get_cute_dtype(w1_alpha),
    )  # fills gateup_output in logical [m, 2*n, l]

    # === SiLU + mul + quantize intermediate activations to FP4 ===
    # Input expected as [num_experts, m, 2*n]
    diq, diq_sf = silu_and_mul_scaled_nvfp4_experts_quantize(
        gateup_output.transpose([2, 0, 1]),  # [num_experts, m, 2*n]
        masked_m,
        a2_global_scale,
    )

    if down_start_event is not None:
        down_start_event.record()

    # === GEMM2: down projection ===
    # grouped_gemm_nt_masked requires output in [m, k, l] layout
    out = paddle.empty([num_experts, m, k], dtype=paddle.bfloat16)
    out = out.transpose([1, 2, 0])  # [m, k, num_experts]

    # w2:           [E, k, n//2]  → _perm(., 1, 2, 0) → [k, n//2, E]
    # w2_blockscale:[E, k, n//G]  → _perm(., 1, 2, 0) → [k, n//G, E]
    # Both must share the same expert-last layout for grouped_gemm_nt_masked.
    grouped_gemm_nt_masked(
        (diq, diq_sf),
        (_perm(w2, 1, 2, 0), _perm(w2_blockscale, 1, 2, 0)),
        out,
        masked_m,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        alpha=w2_alpha.reshape([1, 1, num_experts]),
        alpha_dtype=get_cute_dtype(w2_alpha),
        **(
            dict(
                sm_count=down_sm_count,
                dst_signals=down_signals,
            )
            if down_sm_count is not None or down_signals is not None
            else {}
        ),
    )  # fills out in logical [m, k, l]

    # Return [num_experts, m, k]
    return out.transpose([2, 0, 1])
