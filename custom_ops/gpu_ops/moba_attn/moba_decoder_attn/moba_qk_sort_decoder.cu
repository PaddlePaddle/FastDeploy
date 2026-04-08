// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "moba_attn/moba_attn_utils.hpp"
#include "moba_attn/moba_attn.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <typename T, int knthreads, int moba_block_size, int kBlockMaxN, int searchtimes>
__global__ void qk_gate_sort_decoder_kernel(
        const T* qk_gate_weight,
        int * qk_gate_topk_idx,
        const int *decoder_seq_lens,
        const int head_num,
        const int kv_head_num,
        const int kGqaGroupSize,
        const int top_k_left,
        const int top_k_right,
        const int use_moba_seq_limit) {

    const int bidb = blockIdx.x;
    const int bidh = blockIdx.y;
    const int tidx = threadIdx.x;
    const int bidh_kv = bidh / kGqaGroupSize;

    if (decoder_seq_lens[bidb] == 0 || decoder_seq_lens[bidb] < use_moba_seq_limit) {
        return;
    }
    const int seq_len = (decoder_seq_lens[bidb] + moba_block_size - 1) / moba_block_size;

    constexpr int kPackSize = kBlockMaxN / knthreads;

    static_assert(kBlockMaxN % knthreads == 0);

    T token_mean[kPackSize];

    using SrcType = Vec<T, kPackSize>;
    using SrcType_f = Vec<float, kPackSize>;
    using SrcType_i = Vec<int, kPackSize>;

    SrcType src;
    SrcType_f src_f;
    SrcType_i select_idx;

    select_idx.set_zero();

    const int load_offset = bidb * head_num * kBlockMaxN + bidh * kBlockMaxN + tidx * kPackSize;

    src.load_from(qk_gate_weight + load_offset);

    float max_global = -FLT_MAX;
    float min_global = FLT_MAX;

    const int data_len = seq_len - tidx * kPackSize;

    #pragma unroll
    for (int i = 0; i < kPackSize; i++) {
        if (i < data_len) {
            src_f.data.elt[i] = float(src.data.elt[i]);
            min_global = min(min_global, src_f.data.elt[i]);
        } else {
            src_f.data.elt[i] = -FLT_MAX;
        }
        max_global = max(max_global, src_f.data.elt[i]);
    }


    max_global = BlockAllReduce<float, MaxOp<float>, knthreads>(max_global);
    min_global = BlockAllReduce<float, MinOp<float>, knthreads>(min_global);


    float right_limit = max_global;
    float left_limit = min_global;

    float mid_limit;
    int count;

    #pragma unroll
    for (int i = 0; i < searchtimes; i++) {
        mid_limit = (left_limit + right_limit) * 0.5f;
        count = get_data_count<kPackSize, knthreads>(src_f.data.elt, mid_limit);
        if (count < top_k_left) {
            right_limit = mid_limit;
        } else if (count > top_k_right) {
            left_limit = mid_limit;
        } else {
            break;
        }
    }

    const int store_idx = bidb * kv_head_num * kBlockMaxN + bidh_kv * kBlockMaxN + tidx * kPackSize;

    #pragma unroll
    for (int i = 0; i < kPackSize; i++) {
        if (src_f.data.elt[i] >= mid_limit) {
            qk_gate_topk_idx[store_idx + i] = 1;
        }
    }

    if (tidx == 0) {
        qk_gate_topk_idx[store_idx] = 1;
        qk_gate_topk_idx[store_idx + seq_len - 1] = 1;
        qk_gate_topk_idx[store_idx + seq_len - 2] = 1;
    }
}

template <int kBlockMaxN, int moba_block_size, typename T>
void qk_gate_sort_decoder(
        const T* qk_gate_weight,
        int * qk_gate_topk_idx,
        const int *decoder_seq_lens,
        const int head_num,
        const int kv_head_num,
        const int batch_size,
        const int top_k_left,
        const int top_k_right,
        const int use_moba_seq_limit,
        cudaStream_t stream) {

    const int gqa_group_size = head_num / kv_head_num;
    constexpr int kPackSize = 16 / sizeof(T);
    const int knthreads = kBlockMaxN / kPackSize;
    dim3 grid_dims;
    grid_dims.x = batch_size;
    grid_dims.y = head_num;
    const int searchtimes = 6;

    constexpr auto kernel = qk_gate_sort_decoder_kernel<T, knthreads, moba_block_size, kBlockMaxN, searchtimes>;

    kernel<<<grid_dims, knthreads, 0, 0>>>(
        qk_gate_weight,
        qk_gate_topk_idx,
        decoder_seq_lens,
        head_num,
        kv_head_num,
        gqa_group_size,
        top_k_left,
        top_k_right,
        use_moba_seq_limit);
}


template <typename T>
std::vector<paddle::Tensor> DispatchQkSortDecoder(
        const paddle::Tensor& qk_gate_weight,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const int head_num,
        const int kv_head_num,
        const int top_k_left,
        const int top_k_right,
        const int use_moba_seq_limit) {

    constexpr int kMobaBlockSize = 128;
    constexpr int kMaxN = 1024;

    const int batch_size = seq_len_decoder.dims()[0];
    paddle::Tensor qk_gate_topk_idx = paddle::empty({batch_size, kv_head_num, kMaxN}, paddle::DataType::INT32, qk_gate_weight.place());

    qk_gate_sort_decoder<kMaxN, kMobaBlockSize, T>(
        qk_gate_weight.data<T>(),
        qk_gate_topk_idx.data<int>(),
        seq_len_decoder.data<int>(),
        head_num,
        kv_head_num,
        batch_size,
        top_k_left,
        top_k_right,
        use_moba_seq_limit,
        qk_gate_weight.stream()
    );

    return {qk_gate_topk_idx};
}

std::vector<paddle::Tensor> QkSortDecoder(
        const paddle::Tensor& qk_gate_weight,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const int head_num,
        const int kv_head_num,
        const int top_k_left,
        const int top_k_right,
        const int use_moba_seq_limit) {

    if (qk_gate_weight.dtype() == paddle::DataType::FLOAT16) {
        return std::move(
            DispatchQkSortDecoder<phi::dtype::float16>(
                qk_gate_weight,
                seq_len_encoder,
                seq_len_decoder,
                head_num,
                kv_head_num,
                top_k_left,
                top_k_right,
                use_moba_seq_limit)
        );
    } else if (qk_gate_weight.dtype() == paddle::DataType::BFLOAT16) {
        return std::move(
            DispatchQkSortDecoder<phi::dtype::bfloat16>(
                qk_gate_weight,
                seq_len_encoder,
                seq_len_decoder,
                head_num,
                kv_head_num,
                top_k_left,
                top_k_right,
                use_moba_seq_limit)
        );
    }
}

PD_BUILD_STATIC_OP(moba_qk_sort_decoder)
    .Inputs({
        "qk_gate_weight",
        "seq_len_encoder",
        "seq_len_decoder"})
    .Attrs({
        "head_num: int",
        "kv_head_num: int",
        "top_k_left: int",
        "top_k_right: int",
        "use_moba_seq_limit: int"})
    .Outputs({"qk_gate_topk_idx"})
    .SetKernelFn(PD_KERNEL(QkSortDecoder));
