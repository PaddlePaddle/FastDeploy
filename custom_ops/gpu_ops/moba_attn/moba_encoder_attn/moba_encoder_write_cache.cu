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
#include "moba_attn/moba_attn.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <typename T, int kBlockSize, int kHeadDim>
__global__ void write_encoder_cachekv_c16(
        const T * k_input,
        const T * v_input,
        const int * cu_seq_k,
        const int * seq_len_encoder,
        const int * seq_len_decoder,
        T * cache_k,
        T * cache_v,
        const int * block_tables,
        const int kv_head_num,
        const int max_blocks_per_seq) {

    constexpr int kPackSize = 16 / sizeof(T);
    const int block_idx = blockIdx.x * kBlockSize;
    int bidh = blockIdx.y;
    const int bidb = blockIdx.z;
    const int tidx = threadIdx.x;
    const int row_idx = tidx / (kHeadDim / kPackSize);
    const int col_idx = tidx % (kHeadDim / kPackSize) * kPackSize;
    const int seq_len = seq_len_encoder[bidb];

    if (seq_len == 0) return;

    const int remain_tokens = seq_len - block_idx;

    const int32_t *block_table_now = block_tables + bidb * max_blocks_per_seq;
    const uint32_t physical_block_number = block_table_now[blockIdx.x + seq_len_decoder[bidb] / kBlockSize];

    if (bidh < kv_head_num) {
        T * cache = cache_k + physical_block_number * kv_head_num * kBlockSize * kHeadDim + bidh * kBlockSize * kHeadDim + col_idx;
        const int base_load_idx = (block_idx + cu_seq_k[bidb]) * kv_head_num * kHeadDim + bidh * kHeadDim + col_idx;

        #pragma unroll
        for (int i = row_idx; i < kBlockSize; i += 128 / (kHeadDim / kPackSize)) {
            if (i < remain_tokens) {
                *reinterpret_cast<float4*>(cache + i * kHeadDim) = *reinterpret_cast<const float4*>(k_input + base_load_idx + i * kv_head_num * kHeadDim);
            }
        }
    } else {
        bidh -= kv_head_num;
        const int base_load_idx = (block_idx + cu_seq_k[bidb]) * kv_head_num * kHeadDim + bidh * kHeadDim + col_idx;
        T * cache = cache_v + physical_block_number * kv_head_num * kBlockSize * kHeadDim + bidh * kBlockSize * kHeadDim + col_idx;

        #pragma unroll
        for (int i = row_idx; i < kBlockSize; i += 128 / (kHeadDim / kPackSize)) {
            if (i < remain_tokens) {
                *reinterpret_cast<float4*>(cache + i * kHeadDim) = *reinterpret_cast<const float4*>(v_input + base_load_idx + i * kv_head_num * kHeadDim);
            }
        }

    }
}

void MobaEncoderAttnWriteCacheKv(
        const paddle::Tensor& k_input,
        const paddle::Tensor& v_input,
        const paddle::Tensor& cu_seq_k,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cache_k,
        const paddle::Tensor& cache_v,
        const paddle::Tensor& block_tables,
        const paddle::optional<paddle::Tensor>& cache_k_quant_scale,
        const paddle::optional<paddle::Tensor>& cache_v_quant_scale,
        const paddle::optional<paddle::Tensor>& cache_k_dequant_scale,
        const paddle::optional<paddle::Tensor>& cache_v_dequant_scale,
        const paddle::optional<paddle::Tensor>& cache_k_zero_points,
        const paddle::optional<paddle::Tensor>& cache_v_zero_points,
        const int head_num,
        const int kv_head_num,
        const int head_dim,
        const int max_seq_q,
        const std::string &cache_quant_type_str) {

    constexpr int kThreads = 128;
    constexpr int kHeadDim = 128;
    assert(kHeadDim == head_dim);
    constexpr int kBlockSize = 64;
    const int batch_size = block_tables.dims()[0];
    const int max_blocks_per_seq = block_tables.dims()[1];
    if (cache_quant_type_str == "none") {
        dim3 grid_dims;
        grid_dims.x = (max_seq_q + kBlockSize - 1) / kBlockSize;
        grid_dims.y = kv_head_num * 2;
        grid_dims.z = batch_size;
        if (k_input.dtype() == paddle::DataType::FLOAT16) {
            using T = phi::dtype::float16;
            write_encoder_cachekv_c16<T, kBlockSize, kHeadDim><<<grid_dims, kThreads, 0, k_input.stream()>>>(
                const_cast<T*>(k_input.data<T>()),
                const_cast<T*>(v_input.data<T>()),
                cu_seq_k.data<int>(),
                seq_len_encoder.data<int>(),
                seq_len_decoder.data<int>(),
                const_cast<T*>(cache_k.data<T>()),
                const_cast<T*>(cache_v.data<T>()),
                block_tables.data<int>(),
                kv_head_num,
                max_blocks_per_seq);
        } else if (k_input.dtype() == paddle::DataType::BFLOAT16) {
            using T = phi::dtype::bfloat16;
            write_encoder_cachekv_c16<T, kBlockSize, kHeadDim><<<grid_dims, kThreads, 0, k_input.stream()>>>(
                const_cast<T*>(k_input.data<T>()),
                const_cast<T*>(v_input.data<T>()),
                cu_seq_k.data<int>(),
                seq_len_encoder.data<int>(),
                seq_len_decoder.data<int>(),
                const_cast<T*>(cache_k.data<T>()),
                const_cast<T*>(cache_v.data<T>()),
                block_tables.data<int>(),
                kv_head_num,
                max_blocks_per_seq);
        }
    } else {
        PADDLE_THROW(phi::errors::Unimplemented(
            "Quantized cache not implemented for cache_quant_type = %s", cache_quant_type_str.c_str()));
    }
}

PD_BUILD_STATIC_OP(moba_encoder_attn_write_cache_kv)
    .Inputs({
        "k_input",
        "v_input",
        "cu_seq_k",
        "seq_len_encoder",
        "seq_len_decoder",
        "cache_k",
        "cache_v",
        "block_tables",
        paddle::Optional("cache_k_quant_scale"),
        paddle::Optional("cache_v_quant_scale"),
        paddle::Optional("cache_k_dequant_scale"),
        paddle::Optional("cache_v_dequant_scale"),
        paddle::Optional("cache_k_zero_points"),
        paddle::Optional("cache_v_zero_points")})
    .Attrs({
        "head_num: int",
        "kv_head_num: int",
        "head_dim: int",
        "max_seq_q: int",
        "cache_quant_type_str: std::string"})
    .Outputs({"cache_k_out", "cache_v_out"})
    .SetInplaceMap({{"cache_k", "cache_k_out"},
                    {"cache_v", "cache_v_out"}})
    .SetKernelFn(PD_KERNEL(MobaEncoderAttnWriteCacheKv));
