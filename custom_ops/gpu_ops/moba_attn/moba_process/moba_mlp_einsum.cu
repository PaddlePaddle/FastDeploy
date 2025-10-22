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
#include "moba_attn/moba_attn.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <typename T, int moba_block_size, int kHeadDim, int kMaxN>
__global__ void moba_mlp_einsum_kernel(
        const T * src_data,
        const T * weight_data,
        const int * seq_lens_encoder,
        const int * seq_lens_decoder,
        const int * cu_seq_k,
        T * dst_data,
        const int head_num) {

    constexpr int kPackSize = 16 / sizeof(T);
    const int block_idx = blockIdx.x;
    const int bidh = blockIdx.y;
    const int bidb = blockIdx.z;
    const int tidx = threadIdx.x;
    const int lane_id = tidx % 32;
    const int warp_id = tidx / 32;

    __align__(16) __shared__ T local_sum_mem[128 / 32 * kHeadDim];

    const int seq_len_encoder = seq_lens_encoder[bidb];
    const int seq_len_decoder = seq_len_encoder + seq_lens_decoder[bidb];

    const int seq_len_this_block = seq_len_decoder - block_idx * moba_block_size;

    if (seq_len_encoder == 0 || seq_len_this_block <= 0) {
        return;
    }


    using SrcType = Vec<T, kPackSize>;

    constexpr int tidx_per_row = kHeadDim / kPackSize;

    const int row_idx = tidx / tidx_per_row;
    const int col_idx = tidx % tidx_per_row * kPackSize;

    const int src_base_idx = cu_seq_k[bidb] * head_num * kHeadDim + block_idx * moba_block_size * head_num * kHeadDim + bidh * kHeadDim + row_idx * head_num * kHeadDim + col_idx;
    const int weight_base_idx = bidh * kHeadDim * moba_block_size + row_idx * kHeadDim + col_idx;

    constexpr int step = 128 / tidx_per_row;

    SrcType sums, src, weight;

    sums.set_zero();

    for (int i = 0; i < moba_block_size; i += step) {
        if (i >= seq_len_this_block) {
            break;
        }
        src.load_from(src_data + src_base_idx + i * head_num * kHeadDim);
        weight.load_from(weight_data + weight_base_idx + i * kHeadDim);
        sums.fma(src, weight);
    }

    SrcType neighbor;

    #pragma unroll
    for (int i = 0; i < kPackSize; i+=2) {
        *reinterpret_cast<int32_t*>(neighbor.data.elt + i) = __shfl_down_sync(0xffffffff, *reinterpret_cast<int32_t*>(sums.data.elt + i), 16);
    }

    sums.add(neighbor);

    if (lane_id < 16) {
        sums.store_to(local_sum_mem + warp_id * kHeadDim + lane_id * kPackSize);
    }

    __syncthreads();
    using pack_half = std::conditional_t<std::is_same<T, phi::dtype::float16>::value, __half2, nv_bfloat162>;
    pack_half * local_sum_mem_half = reinterpret_cast<pack_half*>(local_sum_mem);

    if (tidx < kHeadDim / 2) {
        pack_half local_sum_half = local_sum_mem_half[tidx];
        #pragma unroll
        for (int i = 1; i < 4; i++) {
            local_sum_half += local_sum_mem_half[tidx + i * (kHeadDim / 2)];
        }
        local_sum_mem_half[tidx] = local_sum_half;
    }

    __syncthreads();

    const int store_row_id = tidx / (kHeadDim / kPackSize);
    const int store_col_id = tidx % (kHeadDim / kPackSize) * kPackSize;

    sums.load_from(local_sum_mem + store_col_id);

    const int base_store_idx = bidb * kMaxN * head_num * kHeadDim + (block_idx * (moba_block_size / 128) + store_row_id) * head_num * kHeadDim + bidh * kHeadDim + store_col_id;

    if (store_row_id < moba_block_size / 128) {
        sums.store_to(dst_data + base_store_idx);
    }
}


template <typename T, int kHeadDim, int kMaxN>
void moba_mlp_einsum(
        const T * src_data,
        const T * weight_data,
        const int * seq_lens_encoder,
        const int * seq_lens_decoder,
        const int * cu_seq_k,
        T * dst_data,
        const int moba_block_size,
        const int max_seq_len,
        const int head_num,
        const int batch_size,
        cudaStream_t stream) {

    dim3 grid_dims;
    grid_dims.x = (max_seq_len + moba_block_size - 1) / moba_block_size;
    grid_dims.y = head_num;
    grid_dims.z = batch_size;

    if (moba_block_size == 1024) {
        moba_mlp_einsum_kernel<T, 1024, kHeadDim, kMaxN><<<grid_dims, 128, 0, stream>>>(
            src_data,
            weight_data,
            seq_lens_encoder,
            seq_lens_decoder,
            cu_seq_k,
            dst_data,
            head_num);
    } else if (moba_block_size == 128) {
        moba_mlp_einsum_kernel<T, 128, kHeadDim, kMaxN><<<grid_dims, 128, 0, stream>>>(
            src_data,
            weight_data,
            seq_lens_encoder,
            seq_lens_decoder,
            cu_seq_k,
            dst_data,
            head_num);
    } else {
        PADDLE_THROW(phi::errors::Unimplemented(
            "MobaMlpEinsum not implemented for moba_block_size = %d", moba_block_size));
    }

}


std::vector<paddle::Tensor> MobaMlpEinsum(
        const paddle::Tensor& k_input,
        const paddle::Tensor& attn_gate_weight,
        const paddle::Tensor& seq_lens_encoder,
        const paddle::Tensor& seq_lens_decoder,
        const paddle::Tensor& cu_seq_k,
        const int max_seq_len,
        const int kv_head_num) {

    const int kHeadDim = 128;
    const int kMaxN = 1024;
    const int moba_block_size = attn_gate_weight.dims()[1];
    const int batch_size = seq_lens_encoder.dims()[0];
    paddle::Tensor k_gate_weight = paddle::zeros({batch_size, kMaxN, kv_head_num, kHeadDim}, k_input.dtype(), k_input.place());

    if (k_input.dtype() == paddle::DataType::FLOAT16) {
        using T = phi::dtype::float16;
        moba_mlp_einsum<T, kHeadDim, kMaxN>(
            const_cast<T*>(k_input.data<T>()),
            const_cast<T*>(attn_gate_weight.data<T>()),
            const_cast<int*>(seq_lens_encoder.data<int>()),
            const_cast<int*>(seq_lens_decoder.data<int>()),
            const_cast<int*>(cu_seq_k.data<int>()),
            k_gate_weight.data<T>(),
            moba_block_size,
            max_seq_len,
            kv_head_num,
            batch_size,
            k_input.stream()
        );
    } else if (k_input.dtype() == paddle::DataType::BFLOAT16) {
        using T = phi::dtype::bfloat16;
        moba_mlp_einsum<T, kHeadDim, kMaxN>(
            const_cast<T*>(k_input.data<T>()),
            const_cast<T*>(attn_gate_weight.data<T>()),
            const_cast<int*>(seq_lens_encoder.data<int>()),
            const_cast<int*>(seq_lens_decoder.data<int>()),
            const_cast<int*>(cu_seq_k.data<int>()),
            k_gate_weight.data<T>(),
            moba_block_size,
            max_seq_len,
            kv_head_num,
            batch_size,
            k_input.stream()
        );
    }
    return {k_gate_weight};
}

PD_BUILD_STATIC_OP(moba_mlp_einsum)
    .Inputs({
        "k_input",
        "attn_gate_weight",
        "seq_lens_encoder",
        "seq_lens_decoder",
        "cu_seq_k"})
    .Attrs({
        "max_seq_len: int",
        "kv_head_num: int"})
    .Outputs({"k_gate"})
    .SetKernelFn(PD_KERNEL(MobaMlpEinsum));
