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

import paddle
import triton
import triton.language as tl

from fastdeploy.platforms import current_platform


@triton.jit
def count_greater_kernel(
    x_ptr,  # [num_tokens, n_elements]
    y_ptr,  # [num_tokens, 1]
    out_ptr,  # [num_tokens, 1]
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    sum_val = 0.0
    y = tl.load(y_ptr + b * 1 + 0)
    for col_start_idx in range(0, tl.cdiv(n_elements, BLOCK_SIZE)):
        col_ids = col_start_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        col_mask = col_ids < n_elements
        x = tl.load(x_ptr + b * n_elements + col_ids, mask=col_mask, other=-float("inf"))
        compare_mask = x >= y
        cmp_mask = tl.where(compare_mask & col_mask, 1, 0)
        sum_val += tl.sum(cmp_mask, axis=0)
    tl.store(out_ptr + b, sum_val.to(tl.int64))


def batched_count_greater_than(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    Triton implementation: (x >= y).sum(-1)

    Args:
        x (paddle.Tensor): 2D tensor，shape [num_tokens, n_elements]，float32.
        y (paddle.Tensor): 2D tensor，shape [num_tokens, 1]，float32.

    Returns:
        paddle.Tensor: 1D tensor，shape [num_tokens].
    """
    assert x.dim() == 2, f"x must be 2D, got {x.dim()}D"
    assert y.dim() == 2 and y.shape[1] == 1, f"y must be 2D with shape [num_tokens, 1], got {y.shape}"
    assert x.shape[0] == y.shape[0], f"shape[0] mismatch: x has {x.shape[0]}, y has {y.shape[0]}"
    assert x.dtype == y.dtype, f"dtype mismatch: x is {x.dtype}, y is {y.dtype}"

    if current_platform.is_cuda():

        num_tokens, n_elements = x.shape
        dtype = paddle.int64

        out = paddle.empty([num_tokens], dtype=dtype, device=x.place)

        config = {"BLOCK_SIZE": 4096, "num_warps": 16}
        grid = (num_tokens,)

        count_greater_kernel[grid](
            x_ptr=x,
            y_ptr=y,
            out_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=config["BLOCK_SIZE"],
            num_warps=config["num_warps"],
        )
    else:
        out = (x >= y).sum(-1)

    return out
