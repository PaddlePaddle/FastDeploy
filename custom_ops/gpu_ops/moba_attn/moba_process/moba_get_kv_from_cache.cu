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

template <typename T, int kBlockSize, int kHeadDim>
__global__ void get_kv_from_cache_c16_kernel(
        T * k_input,
        T * v_input,
        const int * seq_len_encoder,
        const int * seq_len_decoder,
        const int * cu_seq_k,
        const T * cache_k,
        const T * cache_v,
        const int * block_tables,
        const int kv_head_num,
        const int head_dim,
        const int batch_size,
        const int max_input_length,
        const int max_blocks_per_seq) {

    const int block_idx = blockIdx.x;
    int bidh = blockIdx.y;
    const int bidb = blockIdx.z;
    const int seq_len = seq_len_decoder[bidb] + seq_len_encoder[bidb];
    const int tidx = threadIdx.x;
    const int base_token_idx = block_idx * kBlockSize;

    if (base_token_idx >= seq_len || seq_len_encoder[bidb] == 0) {
        return;
    }

    constexpr int kPackSize = 16 / sizeof(T);

    const int row_idx = tidx / (kHeadDim / kPackSize);
    const int col_idx = tidx % (kHeadDim / kPackSize) * kPackSize;
    const int physical_block_number = block_tables[bidb * max_blocks_per_seq + block_idx];


    const int remain_tokens = seq_len - base_token_idx;

    if (bidh < kv_head_num) {
        const int cache_offset = physical_block_number * kv_head_num * kBlockSize * kHeadDim + bidh * kBlockSize * kHeadDim + col_idx;
        const int base_store_idx = (base_token_idx + cu_seq_k[bidb]) * kv_head_num * kHeadDim + bidh * kHeadDim + col_idx;
        #pragma unroll
        for (int i = row_idx; i < kBlockSize; i += 128 / (kHeadDim / kPackSize)) {
            if (i < remain_tokens) {
                *reinterpret_cast<float4*>(k_input + base_store_idx + i * kv_head_num * kHeadDim) = *reinterpret_cast<const float4*>(cache_k + cache_offset + i * kHeadDim);
            }
        }
    } else {
        bidh -= kv_head_num;
        const int cache_offset = physical_block_number * kv_head_num * kBlockSize * kHeadDim + bidh * kBlockSize * kHeadDim + col_idx;
        const int base_store_idx = (base_token_idx + cu_seq_k[bidb]) * kv_head_num * kHeadDim + bidh * kHeadDim + col_idx;
        #pragma unroll
        for (int i = row_idx; i < kBlockSize; i += 128 / (kHeadDim / kPackSize)) {
            if (i < remain_tokens) {
                *reinterpret_cast<float4*>(v_input + base_store_idx + i * kv_head_num * kHeadDim) = *reinterpret_cast<const float4*>(cache_v + cache_offset + i * kHeadDim);
            }
        }
    }
}

template <typename T>
void get_kv_from_cache(
        T * k_input,
        T * v_input,
        const int * seq_len_encoder,
        const int * seq_len_decoder,
        const int * cu_seq_k,
        const void * cache_k,
        const void * cache_v,
        const int * block_tables,
        const T * cache_k_dequant_scale,
        const T * cache_v_dequant_scale,
        const T * cache_k_zero_points,
        const T * cache_v_zero_points,
        const int kv_head_num,
        const int head_dim,
        const int max_seq_k,
        const int batch_size,
        const int max_input_length,
        const int max_blocks_per_seq,
        const std::string &cache_quant_type_str,
        cudaStream_t stream) {

    constexpr int kThreads = 128;
    constexpr int kHeadDim = 128;
    assert(kHeadDim == head_dim);
    constexpr int kBlockSize = 64;
    if (cache_quant_type_str == "none") {
        dim3 grid_dims;
        grid_dims.x = (max_seq_k + kBlockSize - 1) / kBlockSize;
        grid_dims.y = kv_head_num * 2;
        grid_dims.z = batch_size;
        get_kv_from_cache_c16_kernel<T, kBlockSize, kHeadDim><<<grid_dims, kThreads, 0, stream>>>(
            k_input,
            v_input,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_k,
            reinterpret_cast<const T*>(cache_k),
            reinterpret_cast<const T*>(cache_v),
            block_tables,
            kv_head_num,
            head_dim,
            batch_size,
            max_input_length,
            max_blocks_per_seq);
    } else {
        PD_THROW("Only supported cache_quant_type_str in ['none'].");
    }
}

void GetKVFromCache(
        const paddle::Tensor& k_input,
        const paddle::Tensor& v_input,
        const paddle::Tensor& cu_seq_k,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cache_k,
        const paddle::Tensor& cache_v,
        const paddle::Tensor& block_tables,
        const paddle::optional<paddle::Tensor>& cache_k_dequant_scale,
        const paddle::optional<paddle::Tensor>& cache_v_dequant_scale,
        const paddle::optional<paddle::Tensor>& cache_k_zero_points,
        const paddle::optional<paddle::Tensor>& cache_v_zero_points,
        const int head_num,
        const int kv_head_num,
        const int head_dim,
        const int max_input_length,
        const int max_seq_k,
        const std::string &cache_quant_type_str) {

    if (k_input.dtype() == paddle::DataType::FLOAT16) {
        using T = phi::dtype::float16;
        using cute_type = typename cuteType<T>::type;
        get_kv_from_cache<cute_type>(
            reinterpret_cast<cute_type*>(const_cast<T*>(k_input.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(v_input.data<T>())),
            seq_len_encoder.data<int>(),
            seq_len_decoder.data<int>(),
            cu_seq_k.data<int>(),
            cache_k.data(),
            cache_v.data(),
            block_tables.data<int>(),
            cache_k_dequant_scale ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_k_dequant_scale.get().data<T>())) : nullptr,
            cache_v_dequant_scale ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_v_dequant_scale.get().data<T>())) : nullptr,
            cache_k_zero_points ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_k_zero_points.get().data<T>())) : nullptr,
            cache_v_zero_points ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_v_zero_points.get().data<T>())) : nullptr,
            kv_head_num,
            head_dim,
            max_seq_k,
            seq_len_encoder.dims()[0],
            max_input_length,
            block_tables.dims()[1],
            cache_quant_type_str,
            k_input.stream());
    } else if (k_input.dtype() == paddle::DataType::BFLOAT16) {
        using T = phi::dtype::bfloat16;
        using cute_type = typename cuteType<T>::type;
        get_kv_from_cache<cute_type>(
            reinterpret_cast<cute_type*>(const_cast<T*>(k_input.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(v_input.data<T>())),
            seq_len_encoder.data<int>(),
            seq_len_decoder.data<int>(),
            cu_seq_k.data<int>(),
            cache_k.data(),
            cache_v.data(),
            block_tables.data<int>(),
            cache_k_dequant_scale ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_k_dequant_scale.get().data<T>())) : nullptr,
            cache_v_dequant_scale ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_v_dequant_scale.get().data<T>())) : nullptr,
            cache_k_zero_points ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_k_zero_points.get().data<T>())) : nullptr,
            cache_v_zero_points ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_v_zero_points.get().data<T>())) : nullptr,
            kv_head_num,
            head_dim,
            max_seq_k,
            seq_len_encoder.dims()[0],
            max_input_length,
            block_tables.dims()[1],
            cache_quant_type_str,
            k_input.stream());
    }
}

__global__ void get_cur_cu_seq_len_k_kernel(
        const int* __restrict__ seq_lens_encoder,
        const int* __restrict__ seq_lens_decoder,
        const int* __restrict__ seq_lens_this_time,
        int* __restrict__ cu_seqlens_k,
        int* __restrict__ cu_seq_q_pack,
        int* __restrict__ q_pack_tokens,
        const int pack_size,
        const int bsz) {

    int total_tokens = 0;
    cu_seqlens_k[0] = 0;
    cu_seq_q_pack[0] = 0;

    for (uint32_t bid = 0; bid < bsz; bid++) {
        int cache_len = seq_lens_decoder[bid];
        const int q_len = seq_lens_encoder[bid];
        if (q_len <= 0) {
            cache_len = 0;
        }
        total_tokens += (cache_len + q_len);
        cu_seqlens_k[bid + 1] = total_tokens;
        cu_seq_q_pack[bid + 1] = cu_seq_q_pack[bid] + (q_len + pack_size -1) / pack_size * pack_size;
    }
    q_pack_tokens[0] = cu_seq_q_pack[bsz];
}

std::vector<paddle::Tensor> GetCurCuSeqLenk(
        const paddle::Tensor& seq_lens_encoder,
        const paddle::Tensor& seq_lens_decoder,
        const paddle::Tensor& seq_lens_this_time,
        const int pack_size) {
    auto stream = seq_lens_decoder.stream();
    auto place = seq_lens_decoder.place();
    int bsz = seq_lens_this_time.shape()[0];

    paddle::Tensor cu_seq_q_pack = paddle::empty({bsz + 1}, paddle::DataType::INT32, place);
    paddle::Tensor cu_seqlens_k = paddle::empty({bsz + 1}, paddle::DataType::INT32, place);
    paddle::Tensor q_pack_tokens = paddle::empty({1}, paddle::DataType::INT32, place);

    get_cur_cu_seq_len_k_kernel<<<1, 1, 0, stream>>>(
        seq_lens_encoder.data<int>(),
        seq_lens_decoder.data<int>(),
        seq_lens_this_time.data<int>(),
        cu_seqlens_k.data<int>(),
        cu_seq_q_pack.data<int>(),
        q_pack_tokens.data<int>(),
        pack_size,
        bsz
    );

    auto q_pack_tokens_cpu = q_pack_tokens.copy_to(paddle::CPUPlace(), true);
    return {cu_seq_q_pack, cu_seqlens_k, q_pack_tokens_cpu};
}

PD_BUILD_STATIC_OP(get_kv_from_cache)
    .Inputs({
        "k_input",
        "v_input",
        "cu_seq_k",
        "seq_len_encoder",
        "seq_len_decoder",
        "cache_k",
        "cache_v",
        "block_tables",
        paddle::Optional("cache_k_dequant_scale"),
        paddle::Optional("cache_v_dequant_scale"),
        paddle::Optional("cache_k_zero_points"),
        paddle::Optional("cache_v_zero_points")})
    .Attrs({
        "head_num: int",
        "kv_head_num: int",
        "head_dim: int",
        "max_input_length: int",
        "max_seq_k: int",
        "cache_quant_type_str: std::string"})
    .Outputs({"k_input_out", "v_input_out"})
    .SetInplaceMap({{"k_input", "k_input_out"},
                    {"v_input", "v_input_out"}})
    .SetKernelFn(PD_KERNEL(GetKVFromCache));

PD_BUILD_STATIC_OP(get_cur_cu_seq_len_k)
    .Inputs({
            "seq_lens_encoder",
            "seq_lens_decoder",
            "seq_lens_this_time"})
    .Attrs({
        "pack_size: int"})
    .Outputs({"cu_seq_q_pack", "cu_seqlens_k", "q_pack_tokens"})
    .SetKernelFn(PD_KERNEL(GetCurCuSeqLenk));
