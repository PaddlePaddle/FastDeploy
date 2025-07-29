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

import triton
import triton.language as tl


@triton.jit
def repetition_early_stopper_kernel(
    trunc_ptr,  # float32[B, W]
    probs_ptr,  # float32[B, V]
    next_tokens_ptr,  # int32[B]
    stop_flags,  # bool[B]
    threshold,
    B,  # batch size
    W,  # windows size
    V,  # vocab size
    stride_bw,
    stride_bv,
    BLOCK_W: tl.constexpr,
):
    b = tl.program_id(0)
    w_offs = tl.arange(0, BLOCK_W)

    # current ptr
    trunc_row = trunc_ptr + b * stride_bw
    probs_row = probs_ptr + b * stride_bv

    # step1: use index_sample to get next_score
    next_token = tl.load(next_tokens_ptr + b)
    next_score = tl.load(probs_row + next_token)

    # step2: move window left（w = 0 ~ W-2）←（w = 1 ~ W-1）
    mask = w_offs < W - 1
    val = tl.load(trunc_row + w_offs + 1, mask=mask)
    tl.store(trunc_row + w_offs, val, mask=mask)

    # step3: Insert the current score at the end
    tl.store(trunc_row + W - 1, next_score)

    # step4: determine whether all are greater than threshold
    scores = tl.load(trunc_row + w_offs, mask=w_offs < W, other=0.0)
    is_over = scores > threshold
    all_over = tl.sum(is_over & (w_offs < W)) == W

    # step5: set stop flags and reset trunc scores
    if all_over:
        tl.store(stop_flags + b, True)
        zero = tl.full([BLOCK_W], 0.0, tl.float32)
        tl.store(trunc_row + w_offs, zero, mask=w_offs < W)
