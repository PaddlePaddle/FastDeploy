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
#include "../moba_attn_utils.hpp"
#include "moba_attn/moba_attn.h"

template <typename T, int kBlockSize, int kHeadDim, int moba_block_size, int kMaxN>
__global__ void moba_decoder_attn_write_c16(
        const T * qkv_out,
        const T * qkv_bias,
        T * q_input,
        const int * cu_seq_q,
        const int * cu_seq_k,
        const int * seq_len_encoder,
        const int * seq_len_decoder,
        T * cache_k,
        T * cache_v,
        const int * block_tables,
        const float * rope_sin_cos,
        T *k_block_means,
        const int head_num,
        const int kv_head_num,
        const int max_blocks_per_seq,
        const int max_input_length) {

    int bidh = blockIdx.x;
    const int bidb = blockIdx.y;
    const int tidx = threadIdx.x;
    const int seq_len = seq_len_decoder[bidb];

    if (seq_len == 0) {
        return;
    }

    constexpr int kPackSize = 4;
    using SrcType = Vec<T, kPackSize>;
    using rope_type = Vec<float, kPackSize / 2>;
    SrcType src, bias, k_prev;
    rope_type sin, cos;
    const int bias_idx = bidh * kHeadDim + tidx * kPackSize;
    const int ori_token_idx = cu_seq_q[bidb];
    src.load_from(qkv_out + ori_token_idx * (head_num + 2 * kv_head_num) * kHeadDim + bias_idx);
    if (qkv_bias != nullptr) {
        bias.load_from(qkv_bias + bias_idx);
        src.add(bias);
    }

    const int32_t *block_table_now = block_tables + bidb * max_blocks_per_seq;
    const int32_t physical_block_number = block_table_now[seq_len / kBlockSize];


    if (bidh < head_num) {
        const float * cos_rope = rope_sin_cos + seq_len * (kHeadDim / 2) + tidx * (kPackSize / 2);
        const float * sin_rope = cos_rope + max_input_length * (kHeadDim / 2);
        sin.load_from(sin_rope);
        cos.load_from(cos_rope);
        apply_rotary_embedding<T, kPackSize>(src, cos, sin);

        src.store_to(q_input + cu_seq_q[bidb] * head_num * kHeadDim + bias_idx);
    } else if (bidh < head_num + kv_head_num) {
        bidh -= head_num;
        const int token_in_blocks = seq_len % kBlockSize;
        const float * cos_rope = rope_sin_cos + seq_len * (kHeadDim / 2) + tidx * (kPackSize / 2);
        const float * sin_rope = cos_rope + max_input_length * (kHeadDim / 2);
        sin.load_from(sin_rope);
        cos.load_from(cos_rope);
        apply_rotary_embedding<T, kPackSize>(src, cos, sin);

        T * cache = cache_k + physical_block_number * kv_head_num * kBlockSize * kHeadDim + bidh * kBlockSize * kHeadDim + tidx * kPackSize + token_in_blocks * kHeadDim;
        src.store_to(cache);

        const int seq_len_block = seq_len / moba_block_size;

        const int store_mean_idx = (bidb * kMaxN + seq_len_block) * kv_head_num * kHeadDim + bidh * kHeadDim + tidx * kPackSize;

        if (seq_len % moba_block_size != 0) {
            const int token_num_prev = seq_len % moba_block_size;
            const float inv_tokens_sum = fdividef(1.0f, token_num_prev + 1);
            k_prev.load_from(k_block_means + store_mean_idx);

            #pragma unroll
            for (int i = 0; i < kPackSize; i++) {
                src.data.elt[i] = T(inv_tokens_sum * (float(src.data.elt[i]) + float(k_prev.data.elt[i]) * token_num_prev));
            }
        }

        src.store_to(k_block_means + store_mean_idx);

    } else {
        bidh -= head_num + kv_head_num;
        const int token_in_blocks = seq_len % kBlockSize;
        T * cache = cache_v + physical_block_number * kv_head_num * kBlockSize * kHeadDim + bidh * kBlockSize * kHeadDim + tidx * kPackSize + token_in_blocks * kHeadDim;
        src.store_to(cache);
    }

}

void MobaDecoderAttnWriteCacheKv(
        const paddle::Tensor& qkv_out,
        const paddle::Tensor& q_input,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cache_k,
        const paddle::Tensor& cache_v,
        const paddle::Tensor& block_tables,
        const paddle::Tensor& rope_sin_cos,
        const paddle::Tensor& k_block_means,
        const paddle::optional<paddle::Tensor>& qkv_bias,
        const paddle::optional<paddle::Tensor>& cache_k_quant_scale,
        const paddle::optional<paddle::Tensor>& cache_v_quant_scale,
        const paddle::optional<paddle::Tensor>& cache_k_dequant_scale,
        const paddle::optional<paddle::Tensor>& cache_v_dequant_scale,
        const paddle::optional<paddle::Tensor>& cache_k_zero_points,
        const paddle::optional<paddle::Tensor>& cache_v_zero_points,
        const int head_num,
        const int kv_head_num,
        const int head_dim,
        const int max_input_length,
        const std::string &cache_quant_type_str) {

    constexpr int kThreads = 32;
    constexpr int kHeadDim = 128;
    constexpr int kMobaBlockSize = 128;
    constexpr int kMaxN = 1024;
    assert(kHeadDim == head_dim);
    constexpr int kBlockSize = 64;
    const int max_blocks_per_seq = block_tables.dims()[1];
    const int batch_size = block_tables.dims()[0];
    if (cache_quant_type_str == "none") {
        dim3 grid_dims;
        grid_dims.x = head_num + kv_head_num * 2;
        grid_dims.y = batch_size;
        if (qkv_out.dtype() == paddle::DataType::FLOAT16) {
            using T = phi::dtype::float16;
            moba_decoder_attn_write_c16<T, kBlockSize, kHeadDim, kMobaBlockSize, kMaxN><<<grid_dims, kThreads, 0, qkv_out.stream()>>>(
                qkv_out.data<T>(),
                qkv_bias ? qkv_bias.get().data<T>() : nullptr,
                const_cast<T*>(q_input.data<T>()),
                cu_seq_q.data<int>(),
                cu_seq_k.data<int>(),
                seq_len_encoder.data<int>(),
                seq_len_decoder.data<int>(),
                const_cast<T *>(cache_k.data<T>()),
                const_cast<T *>(cache_v.data<T>()),
                block_tables.data<int>(),
                rope_sin_cos.data<float>(),
                const_cast<T*>(k_block_means.data<T>()),
                head_num,
                kv_head_num,
                max_blocks_per_seq,
                max_input_length);
        } else if (qkv_out.dtype() == paddle::DataType::BFLOAT16) {
            using T = phi::dtype::bfloat16;
            moba_decoder_attn_write_c16<T, kBlockSize, kHeadDim, kMobaBlockSize, kMaxN><<<grid_dims, kThreads, 0, qkv_out.stream()>>>(
                qkv_out.data<T>(),
                qkv_bias ? qkv_bias.get().data<T>() : nullptr,
                const_cast<T*>(q_input.data<T>()),
                cu_seq_q.data<int>(),
                cu_seq_k.data<int>(),
                seq_len_encoder.data<int>(),
                seq_len_decoder.data<int>(),
                const_cast<T *>(cache_k.data<T>()),
                const_cast<T *>(cache_v.data<T>()),
                block_tables.data<int>(),
                rope_sin_cos.data<float>(),
                const_cast<T*>(k_block_means.data<T>()),
                head_num,
                kv_head_num,
                max_blocks_per_seq,
                max_input_length);
        }
    } else {
        PD_THROW("Only supported cache_quant_type_str in ['none'].");
    }
}
