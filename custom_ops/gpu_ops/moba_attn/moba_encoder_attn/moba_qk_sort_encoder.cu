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
#include "moba_attn/moba_attn_utils.hpp"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <typename T, int knthreads, int moba_block_size, int kBlockM, int kBlockMaxN, int searchtimes>
__global__ void qk_gate_sort_encoder_kernel(
        const T* qk_gate_weight,
        int * qk_gate_topk_idx,
        const int *seq_len_encoder,
        const int *seq_len_decoder,
        const int* cu_seq_q,
        const int* cu_seq_k,
        const int* cu_seq_q_pack,
        const int use_moba_seq_limit,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const int kGqaGroupSize,
        const int top_k_left,
        const int top_k_right) {

    const int bidt = blockIdx.x * kBlockM;
    const int bidh = blockIdx.y;
    const int bidb = blockIdx.z;
    const int tidx = threadIdx.x;

    constexpr int kPackSize = kBlockMaxN / knthreads;

    static_assert(kBlockMaxN % knthreads == 0);

    const int seq_len_q = seq_len_encoder[bidb];

    if (seq_len_q == 0 || bidt >= seq_len_q) {
        return;
    }

    const int seq_len_k = (bidt + kBlockM + seq_len_decoder[bidb]);

    const int seq_len_moba = seq_len_k / moba_block_size;

    using SrcType = Vec<T, kPackSize>;
    using SrcType_f = Vec<float, kPackSize>;
    using SrcType_i = Vec<int, kPackSize>;

    SrcType src;
    SrcType_f src_f;

    SrcType_i select_idx;

    select_idx.set_zero();

    const int store_idx = cu_seq_q_pack[bidb] / kBlockM * head_num * kBlockMaxN + bidh * kBlockMaxN + blockIdx.x * head_num * kBlockMaxN + tidx * kPackSize;

    if (seq_len_k < use_moba_seq_limit) {
        #pragma unroll
        for (int i = 0; i < kPackSize; i++) {
            select_idx.data.elt[i] = 1;
        }
        select_idx.store_to(qk_gate_topk_idx + store_idx);
        return;
    }

    const int load_offset = (cu_seq_q[bidb] + bidt) * head_num * kBlockMaxN + bidh * kBlockMaxN + tidx * kPackSize;
    const int data_len = seq_len_moba - tidx * kPackSize;

    #pragma unroll
    for (int t = 0; t < kBlockM; t++) {
        if (bidt + t >= seq_len_q) {
            break;
        }
        src.load_from(qk_gate_weight + load_offset + t * head_num * kBlockMaxN);
        float min_global = FLT_MAX;
        float max_global = -FLT_MAX;
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

        if (right_limit == left_limit) {
            mid_limit = (left_limit + right_limit) * 0.5f;
        } else {
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
        }

        #pragma unroll
        for (int i = 0; i < kPackSize; i++) {
            if (src_f.data.elt[i] >= mid_limit) {
                select_idx.data.elt[i] = 1;
            }
        }
    }

    if (tidx == 0) {
        select_idx.data.elt[0] = 1;
    }

    __align__(16) __shared__ int qk_gate_mem[kBlockMaxN];
    __align__(16) __shared__ int qk_continue_idx_mem[kBlockMaxN];
    select_idx.store_to(qk_gate_mem + tidx * kPackSize);

    __syncthreads();

    if (tidx == 0) {
        int cur_idx = 0;
        int idx = -1;
        const int last_idx = seq_len_moba - 1;
        while (last_idx + idx >= 0 && qk_gate_mem[last_idx + idx] == 0) {
            idx--;
        }
        qk_continue_idx_mem[cur_idx] = -idx;
        cur_idx++;

        for (int i = last_idx - 1; i >= 0; --i) {
            if (qk_gate_mem[i] == 1) {
                int idx = -1;
                while (i + idx >= 0 && qk_gate_mem[i + idx] == 0) {
                    idx--;
                }
                qk_continue_idx_mem[cur_idx] = -idx;
                cur_idx++;
            }
        }
        qk_continue_idx_mem[cur_idx] = 10000000;
    }

    __syncthreads();

    *reinterpret_cast<SrcType_i *>(qk_gate_topk_idx + store_idx) = reinterpret_cast<SrcType_i *>(qk_continue_idx_mem)[tidx];
}

template <int kBlockM, int kMaxN, int moba_block_size, typename T>
void qk_gate_sort_encoder(
        const T* qk_gate_weight,
        int * qk_gate_topk_idx,
        const int *seq_len_encoder,
        const int *seq_len_decoder,
        const int* cu_seq_q,
        const int* cu_seq_k,
        const int* cu_seq_q_pack,
        const int use_moba_seq_limit,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const int batch_size,
        const int top_k_left,
        const int top_k_right,
        cudaStream_t stream) {

    constexpr int kPackSize = 16 / sizeof(T);

    const int gqa_group_size = head_num / kv_head_num;
    const int knthreads = kMaxN / kPackSize;
    const int searchtimes = 6;

    dim3 grid_dims;
    grid_dims.x = (max_seq_q + kBlockM - 1) / kBlockM;
    grid_dims.y = head_num;
    grid_dims.z = batch_size;

    constexpr auto kernel = qk_gate_sort_encoder_kernel<T, knthreads, moba_block_size, kBlockM, kMaxN, searchtimes>;

    kernel<<<grid_dims, knthreads, 0, stream>>>(
        qk_gate_weight,
        qk_gate_topk_idx,
        seq_len_encoder,
        seq_len_decoder,
        cu_seq_q,
        cu_seq_k,
        cu_seq_q_pack,
        use_moba_seq_limit,
        max_seq_q,
        max_seq_k,
        head_num,
        kv_head_num,
        gqa_group_size,
        top_k_left,
        top_k_right);
}
template <typename T>
std::vector<paddle::Tensor> DispatchQkSortEncoder(
        const paddle::Tensor& qk_gate_weight,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const paddle::Tensor& cu_seq_q_pack,
        const paddle::Tensor& q_pack_tokens,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const int top_k_left,
        const int top_k_right,
        const int use_moba_seq_limit) {
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    constexpr int kMobaBlockSize = 128;
    constexpr int kMaxN = 1024;
    using cute_type = typename cuteType<T>::type;
    const int batch_size = seq_len_encoder.dims()[0];

    paddle::Tensor qk_gate_topk_idx = paddle::empty({q_pack_tokens.data<int>()[0] / kBlockM, head_num, kMaxN}, paddle::DataType::INT32, qk_gate_weight.place());

    qk_gate_sort_encoder<kBlockM, kMaxN, kMobaBlockSize, cute_type>(
            reinterpret_cast<const cute_type *>(qk_gate_weight.data<T>()),
            qk_gate_topk_idx.data<int>(),
            seq_len_encoder.data<int>(),
            seq_len_decoder.data<int>(),
            cu_seq_q.data<int>(),
            cu_seq_k.data<int>(),
            cu_seq_q_pack.data<int>(),
            use_moba_seq_limit,
            max_seq_q,
            max_seq_k,
            head_num,
            kv_head_num,
            batch_size,
            top_k_left,
            top_k_right,
            qk_gate_weight.stream());

    return {qk_gate_topk_idx};
}


std::vector<paddle::Tensor> QkSortEncoder(
        const paddle::Tensor& qk_gate_weight,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const paddle::Tensor& cu_seq_q_pack,
        const paddle::Tensor& q_pack_tokens,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const int top_k_left,
        const int top_k_right,
        const int use_moba_seq_limit) {
    if (qk_gate_weight.dtype() == paddle::DataType::FLOAT16) {
        return std::move(
            DispatchQkSortEncoder<phi::dtype::float16>(
                qk_gate_weight,
                seq_len_encoder,
                seq_len_decoder,
                cu_seq_q,
                cu_seq_k,
                cu_seq_q_pack,
                q_pack_tokens,
                max_seq_q,
                max_seq_k,
                head_num,
                kv_head_num,
                top_k_left,
                top_k_right,
                use_moba_seq_limit
            )
        );
    } else if (qk_gate_weight.dtype() == paddle::DataType::BFLOAT16) {
        return std::move(
            DispatchQkSortEncoder<phi::dtype::bfloat16>(
                qk_gate_weight,
                seq_len_encoder,
                seq_len_decoder,
                cu_seq_q,
                cu_seq_k,
                cu_seq_q_pack,
                q_pack_tokens,
                max_seq_q,
                max_seq_k,
                head_num,
                kv_head_num,
                top_k_left,
                top_k_right,
                use_moba_seq_limit
            )
        );
    }
}

PD_BUILD_STATIC_OP(moba_qk_sort_encoder)
    .Inputs({
        "qk_gate_weight",
        "seq_len_encoder",
        "seq_len_decoder",
        "cu_seq_q",
        "cu_seq_k",
        "cu_seq_q_pack",
        "q_pack_tokens"})
    .Attrs({
        "max_seq_q: int",
        "max_seq_k: int",
        "head_num: int",
        "kv_head_num: int",
        "top_k_left: int",
        "top_k_right: int",
        "use_moba_seq_limit: int"})
    .Outputs({"qk_gate_topk_idx"})
    .SetKernelFn(PD_KERNEL(QkSortEncoder));
