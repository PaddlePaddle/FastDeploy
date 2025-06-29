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
import triton.language as tl
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.framework import in_dynamic_or_pir_mode

from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    get_dtype_str, paddle_use_triton, rendering_common_template)

BLOCK_SIZE_M = 16


def invoke_fused_moe_kernel(
    A,
    B,
    C,
    B_scale,
    B_super_scale,
    B_code_scale,
    B_code_zp,
    topk_weights,
    topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    mul_routed_weight=False,
    top_k=-1,
    group_size=-1,
):
    """
    Invoke Fused Moe Kernel
    """
    KK = A.shape[-1]
    NN = B.shape[-1]
    sstride_am, sstride_ak = A.shape[1], 1
    sstride_be, sstride_bk, sstride_bn = B.shape[1] * B.shape[2], B.shape[2], 1
    sstride_cm, sstride_cn = C.shape[-1], 1
    sstride_bse, sstride_bsk, sstride_bsn = B_scale.shape[1] * B_scale.shape[
        2], B_scale.shape[2], 1
    sstride_bce, sstride_bck, sstride_bcn = B_code_scale.shape[1], 1, 1

    ddouble_quant = B_super_scale is not None

    prepare_attr_for_triton_kernel = """
        auto N = B.shape()[2];
        auto K = A.shape()[1];
        auto EM = sorted_token_ids.shape()[0];
        auto num_valid_tokens = (topk_ids.shape()[0]) * (topk_ids.shape()[1]);
        auto stride_am = A.strides()[0];
        auto stride_ak = A.strides()[1];
        auto stride_be = B.strides()[0];
        auto stride_bk = B.strides()[1];
        auto stride_bn = B.strides()[2];
        auto stride_cm = C.strides()[1];
        auto stride_cn = C.strides()[2];
        auto stride_bse = B_scale.strides()[0];
        auto stride_bsk = B_scale.strides()[1];
        auto stride_bsn = 1;
        auto stride_bce = B_code_scale.strides()[0];
        auto stride_bck = 1;
        auto stride_bcn = 1;
        auto double_quant = true;
    """
    if mul_routed_weight:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "GROUP_SIZE_M": 2,
            "num_warps": 4,
            "num_stages": 8,
        }
    else:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 512,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 12,
        }
    configs = []

    configs.append(dict(config))

    op_name = "wint2_moe_ffn"
    op_name += f"{get_dtype_str(A.dtype)}"
    op_name += f"{B.shape[0]}"
    op_name += f"{B.shape[1]}"
    op_name += f"{B.shape[2]}"

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        prepare_ptr_for_triton_kernel = """
            CUdeviceptr input_ptrs[11] = {
                get_tensor_ptr(A),
                get_tensor_ptr(B),
                get_tensor_ptr(C),
                get_tensor_ptr(B_scale),
                get_tensor_ptr(B_super_scale),
                get_tensor_ptr(B_code_scale),
                get_tensor_ptr(B_code_zp),
                get_tensor_ptr(topk_weights),
                get_tensor_ptr(sorted_token_ids),
                get_tensor_ptr(expert_ids),
                get_tensor_ptr(num_tokens_post_padded),
            };
            """
        template_used = rendering_common_template(
            invoke_fused_moe_kernel,
            prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel,
        )
        grid = (
            "(EM+BLOCK_SIZE_M-1)/BLOCK_SIZE_M * ((N+BLOCK_SIZE_N-1)/BLOCK_SIZE_N)",
        )

        moe_wint2_ffn_kernel[(op_name, template_used, grid, configs)](
            A,
            B,
            C,
            B_scale,
            B_super_scale,
            B_code_scale,
            B_code_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            NN,
            KK,
            -1,  #EEM,
            -1,  #nnum_valid_tokens,
            sstride_am,
            sstride_ak,
            sstride_be,
            sstride_bk,
            sstride_bn,
            sstride_cm,
            sstride_cn,
            sstride_bse,
            sstride_bsk,
            sstride_bsn,
            sstride_bce,
            sstride_bck,
            sstride_bcn,
            MUL_ROUTED_WEIGHT=(int)(mul_routed_weight),
            USE_DOUBLE_QUANT=(int)(ddouble_quant),
            top_k=top_k,
            BLOCK_SIZE_K=group_size,
        )
    if in_dynamic_or_pir_mode():

        outs = _C_ops._run_custom_op(
            op_name,
            A,
            B,
            C,
            B_scale,
            B_super_scale,
            B_code_scale,
            B_code_zp,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            group_size,
        )
        return outs[0]


@paddle_use_triton(key=["1"], )
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
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bce,
    stride_bck,
    stride_bcn,
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

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
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

    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_bk[None, :] * pack_num * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_bk[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    bs_ptrs = bs_ptr + off_experts * stride_bse + offs_bn[
        None, :] * stride_bsn  # group-wise, need advanced

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
        b = tl.floor((b.to(tl.float32) * code_bs + code_bzp) + 0.5).to(
            tl.int16)
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
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def fused_moe_wint2_impl(
    hidden_states,
    ffn1_quant_weight,
    ffn2_quant_weight,
    topk_weights,
    topk_ids,
    # inplace: bool = False,
    ffn1_weight_scale=None,
    ffn2_weight_scale=None,
    ffn1_super_scales=None,
    ffn2_super_scales=None,
    ffn1_code_scale=None,
    ffn2_code_scale=None,
    ffn1_code_zp=None,
    ffn2_code_zp=None,
    group_size=64,
    bit="wint2",
):
    """
    Implementation of Fused MoE kernels on GPU.
    """
    # Check constraints.
    # A: [M, K]
    # B: [E, K, N]
    # assert hidden_states.shape[1] == ffn1_weight_scale.shape[1],
    # f"Hidden size mismatch, {hidden_states.shape[1]} != {ffn1_quant_weight.shape[1]}"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert ffn1_quant_weight.is_contiguous(
    ), "Expert weights1 must be contiguous"
    assert ffn2_quant_weight.is_contiguous(
    ), "Expert weights2 must be contiguous"
    assert group_size > 0, "Group size must be greater than 0"

    num_tokens, K = hidden_states.shape
    E, _, N = ffn1_quant_weight.shape
    M = num_tokens

    if group_size < 0:
        group_size = K // ffn1_weight_scale.shape[1]

    top_k = topk_ids.shape[1]

    intermediate_cache1 = paddle.empty(
        [M, top_k, N],
        dtype=hidden_states.dtype,
    )
    intermediate_cache2 = paddle.empty(
        (M * top_k, N // 2),
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = paddle.empty(
        (M, top_k, K),
        dtype=hidden_states.dtype,
    )

    from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess

    sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess(
        topk_ids, E, BLOCK_SIZE_M)


    invoke_fused_moe_kernel(
        A=hidden_states,
        B=ffn1_quant_weight,
        C=intermediate_cache1,
        B_scale=ffn1_weight_scale,
        B_super_scale=ffn1_super_scales,
        B_code_scale=ffn1_code_scale,
        B_code_zp=ffn1_code_zp,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=top_k,
        group_size=group_size,
    )

    intermediate_cache2 = paddle.incubate.nn.functional.swiglu(
        intermediate_cache1.reshape([-1, N]))

    invoke_fused_moe_kernel(
        A=intermediate_cache2,
        B=ffn2_quant_weight,
        C=intermediate_cache3,
        B_scale=ffn2_weight_scale,
        B_super_scale=ffn2_super_scales,
        B_code_scale=ffn2_code_scale,
        B_code_zp=ffn2_code_zp,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
        group_size=group_size,
    )

    out_hidden_states = paddle.sum(intermediate_cache3, axis=1)
    return out_hidden_states


def fused_moe_wint2_triton(
    hidden_states,
    ffn1_quant_weight,
    ffn2_quant_weight,
    scores,
    gate_correction_bias,
    topk,
    ffn1_weight_scale,
    ffn2_weight_scale,
    ffn1_super_scales,
    ffn2_super_scales,
    ffn1_code_scale,
    ffn2_code_scale,
    ffn1_code_zp,
    ffn2_code_zp,
):
    """
    Fuse MoE with WINT2 quantization scheme and Triton backend.
    Args:
        hidden_states: input tensor.
        ffn1_quant_weight: ffn1 weight matrix for experts.
        ffn2_quant_weight: ffn2 weight matrix for experts.
        scores: gate scores.
        gate_correction_bias: bias correction for gates.
        topk: number of experts to use.
        ffn1_weight_scale: scaling factor for ffn1_quant_weight.
        ffn2_weight_scale: scaling factor for ffn2_quant_weight.
        ffn1_super_scales: super scaling factor for ffn1_scale.
        ffn2_super_scales: super scaling factor for ffn2_weight_scale.
        ffn1_code_scale: code scaling factor for ffn1_quant_weight.
        ffn2_code_scale: code scaling factor for ffn2_quant_weight.
        ffn1_code_zp: code zero point for ffn1_quant_weight.
        ffn2_code_zp: code zero point for ffn2_quant_weight.
    Returns:
        output tensor.
    """

    score = gate_correction_bias + scores
    _, topk_ids = paddle.topk(score, k=topk, axis=-1)
    topk_weights, _ = paddle.topk(scores, k=topk, axis=-1)
    topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdim=True)

    return fused_moe_wint2_impl(
        hidden_states,
        ffn1_quant_weight,
        ffn2_quant_weight,
        topk_weights,
        topk_ids,
        ffn1_weight_scale,
        ffn2_weight_scale,
        ffn1_super_scales,
        ffn2_super_scales,
        ffn1_code_scale,
        ffn2_code_scale,
        ffn1_code_zp,
        ffn2_code_zp,
        bit="wint2",
    )
