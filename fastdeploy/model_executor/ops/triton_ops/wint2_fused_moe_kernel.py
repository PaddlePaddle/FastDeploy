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

import triton.language as tl

from fastdeploy.model_executor.ops.triton_ops.triton_utils_v2 import (
    paddle_use_triton_v2,
)


@paddle_use_triton_v2()
def moe_wint2_ffn_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    bs_ptr,
    superbs_ptr,
    codebs_ptr,
    codebzp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_valid_tokens,
    # Matrix dimensions
    max_possible_num_post_padded,
    N: tl.constexpr,
    K: tl.constexpr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_bce: tl.constexpr,
    stride_bck: tl.constexpr,
    stride_bcn: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    USE_DOUBLE_QUANT: tl.constexpr,
    top_k: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """

    if USE_DOUBLE_QUANT:
        # INT4 scale
        s_packnums: tl.constexpr = 2
    bzp: tl.constexpr = 32
    w_mask: tl.constexpr = 0x3F
    pack_num: tl.constexpr = 4
    real_k_size: tl.constexpr = (BLOCK_SIZE_K - 1) // pack_num + 1

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(max_possible_num_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    compute_type = c_ptr.dtype.element_ty

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    # offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_bk = tl.arange(0, real_k_size)

    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_bk[None, :] * pack_num * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    bs_ptrs = bs_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn  # group-wise, need advanced

    off_set = off_experts * stride_bce + offs_bn[None, :] * stride_bcn
    # load channel-wise scale & zero-point
    if USE_DOUBLE_QUANT:
        superbs_ptrs = superbs_ptr + off_set  # channel-wise
        super_bs = tl.load(superbs_ptrs)  # super scale

    codebs_ptrs = codebs_ptr + off_set  # channel-wise
    code_bs = tl.load(codebs_ptrs)  # code scale
    codebzp_ptrs = codebzp_ptr + off_set  # channel-wise
    code_bzp = tl.load(codebzp_ptrs)  # code zp

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        b = tl.load(b_ptrs)

        bs = tl.load(bs_ptrs)
        if USE_DOUBLE_QUANT:
            s_shift_bits = (1 - k % s_packnums) * 4
            bs = ((bs >> s_shift_bits) & 0xF) * super_bs

        # reverse to int16
        b = tl.floor((b.to(tl.float32) * code_bs + code_bzp) + 0.5).to(tl.int16)
        # dequant
        b1 = (((b >> 9) & w_mask) - bzp) * bs
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(a, b1.to(a.dtype))

        b1 = (((b >> 6) & w_mask) - bzp) * bs
        a = tl.load(
            a_ptrs + 1,
            mask=token_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(a, b1.to(a.dtype))

        b1 = (((b >> 3) & w_mask) - bzp) * bs
        a = tl.load(
            a_ptrs + 2,
            mask=token_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(a, b1.to(a.dtype))

        b = ((b & w_mask) - bzp) * bs
        a = tl.load(
            a_ptrs + 3,
            mask=token_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(a, b.to(a.dtype))

        b_ptrs += real_k_size * stride_bk
        a_ptrs += BLOCK_SIZE_K * stride_ak

        # advance scale ptr
        if USE_DOUBLE_QUANT:
            bs_ptrs += stride_bsk * (k % s_packnums)
        else:
            bs_ptrs += stride_bsk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
