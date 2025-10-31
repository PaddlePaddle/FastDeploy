/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/extension.h"
#include "kernel_traits.h"
#include "flash_mask_attn_kernel.hpp"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <typename paddle_type>
struct cuteType;

template <>
struct cuteType<phi::dtype::float16> {
    using type = cutlass::half_t;
};

template <>
struct cuteType<phi::dtype::bfloat16> {
    using type = cutlass::bfloat16_t;
};

template <typename T>
std::vector<paddle::Tensor> DispatchFlashAttentionMask(
        const paddle::Tensor& q_input,
        const paddle::Tensor& k_input,
        const paddle::Tensor& v_input,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const paddle::Tensor& seq_len_encoder,
        const paddle::optional<paddle::Tensor>& mask,
        const int head_num,
        const int kv_head_num,
        const int head_dim,
        const int max_seq_len,
        const int max_enc_len_this_time,
        const int max_dec_len_this_time) {

    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    const int batch_size = cu_seq_q.dims()[0];

    paddle::Tensor out = paddle::empty(
        {q_input.dims()[0], head_num * head_dim}, q_input.dtype(), q_input.place());

    Flash_mask_params params;
    memset(&params, 0, sizeof(Flash_mask_params));

    params.q_ptr = const_cast<T*>(q_input.data<T>());
    params.k_ptr = const_cast<T*>(k_input.data<T>());
    params.v_ptr = const_cast<T*>(v_input.data<T>());
    params.o_ptr = const_cast<T*>(out.data<T>());
    params.cu_seq_q = const_cast<int*>(cu_seq_q.data<int>());
    params.cu_seq_k = const_cast<int*>(cu_seq_k.data<int>());
    params.seq_len_encoder = const_cast<int*>(seq_len_encoder.data<int>());
    params.head_num = head_num;
    params.kv_head_num = kv_head_num;
    params.max_seq_len_q = max_enc_len_this_time;
    params.max_seq_len_k = max_enc_len_this_time + max_dec_len_this_time;
    params.batch_size = batch_size;
    params.gqa_group_size = head_num / kv_head_num;
    constexpr float kLog2e = 1.4426950408889634074;
    params.scale_softmax_log2 = 1.0f / std::sqrt(head_dim) * kLog2e;

    using cute_type = typename cuteType<T>::type;

    if (mask) {
        params.mask = const_cast<int*>(mask.get().data<int>());
        flash_attn_headdim128<kBlockM, kBlockN, true, cute_type>(params, 0);
    } else {
        flash_attn_headdim128<kBlockM, kBlockN, false, cute_type>(params, 0);
    }

    return {out};
}


std::vector<paddle::Tensor> FlashAttentionMask(
        const paddle::Tensor& q_input,
        const paddle::Tensor& k_input,
        const paddle::Tensor& v_input,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const paddle::Tensor& seq_len_encoder,
        const paddle::optional<paddle::Tensor> &mask,
        const int head_num,
        const int kv_head_num,
        const int head_dim,
        const int max_seq_len,
        const int max_enc_len_this_time,
        const int max_dec_len_this_time) {

    if (q_input.dtype() == paddle::DataType::FLOAT16) {
        using T = phi::dtype::float16;
        return std::move(
            DispatchFlashAttentionMask<T>(
                q_input,
                k_input,
                v_input,
                cu_seq_q,
                cu_seq_k,
                seq_len_encoder,
                mask,
                head_num,
                kv_head_num,
                head_dim,
                max_seq_len,
                max_enc_len_this_time,
                max_dec_len_this_time));
    } else if (q_input.dtype() == paddle::DataType::BFLOAT16) {
        using T = phi::dtype::bfloat16;
        return std::move(
            DispatchFlashAttentionMask<T>(
                q_input,
                k_input,
                v_input,
                cu_seq_q,
                cu_seq_k,
                seq_len_encoder,
                mask,
                head_num,
                kv_head_num,
                head_dim,
                max_seq_len,
                max_enc_len_this_time,
                max_dec_len_this_time));
    }

}


PD_BUILD_STATIC_OP(flash_attention_mask)
    .Inputs({
        "q_input",
        "k_input",
        "v_input",
        "cu_seq_q",
        "cu_seq_k",
        "seq_len_encoder",
        paddle::Optional("mask")})
    .Attrs({
        "head_num: int",
        "kv_head_num: int",
        "head_dim: int",
        "max_seq_len: int",
        "max_enc_len_this_time: int",
        "max_dec_len_this_time: int"})
    .Outputs({
        "out"})
    .SetKernelFn(PD_KERNEL(FlashAttentionMask));
