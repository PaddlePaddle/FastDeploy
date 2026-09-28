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
Fused GDN gating Triton kernel.

Ported from SGLang (sglang/srt/layers/attention/fla/fused_gdn_gating.py).
Computes in a single kernel launch:
    g = -exp(A_log) * softplus(a + dt_bias)
    beta_output = sigmoid(b)
"""

from __future__ import annotations

from typing import Tuple

import paddle
import triton
import triton.language as tl


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
# beta_output = b.sigmoid()
@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: paddle.Tensor,
    a: paddle.Tensor,
    b: paddle.Tensor,
    dt_bias: paddle.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Fused GDN gating: g = -exp(A_log)*softplus(a+dt_bias), beta = sigmoid(b).

    Args:
        A_log: [num_heads] - log of A matrix
        a: [num_tokens, num_heads] - alpha values
        b: [num_tokens, num_heads] - beta values
        dt_bias: [num_heads] - delta-time bias

    Returns:
        g: [num_tokens, num_heads] float32
        beta_output: [num_tokens, num_heads] float32
    """
    num_tokens, num_heads = a.shape
    seq_len = 1
    grid = (num_tokens, seq_len, triton.cdiv(num_heads, 8))
    g = paddle.empty([num_tokens, num_heads], dtype=paddle.float32)
    beta_output = paddle.empty([num_tokens, num_heads], dtype=paddle.float32)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
