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

#pragma once

// 宏定义用于简化模板实例化
#define INSTANTIATE_APPEND_ATTENTION_C4(T, OutT) \
  template void CascadeAppendAttentionC4Kernel<T, OutT>( \
    const AppendAttnMetaData& meta_data, \
    const paddle::Tensor& qkv, \
    const paddle::Tensor& cache_k, \
    const paddle::Tensor& cache_v, \
    const paddle::optional<paddle::Tensor>& attn_mask, \
    const paddle::optional<paddle::Tensor>& cache_k_scale, \
    const paddle::optional<paddle::Tensor>& cache_v_scale, \
    const paddle::optional<paddle::Tensor>& cache_k_zp, \
    const paddle::optional<paddle::Tensor>& cache_v_zp, \
    const paddle::optional<paddle::Tensor>& shift_bias, \
    const paddle::optional<paddle::Tensor>& smooth_weight, \
    const paddle::Tensor& seq_lens_q, \
    const paddle::Tensor& seq_lens_kv, \
    const paddle::Tensor& seq_lens_encoder, \
    const paddle::Tensor& batch_id_per_token, \
    const paddle::Tensor& cu_seqlens_q, \
    const paddle::Tensor& block_table, \
    const paddle::Tensor& batch_ids, \
    const paddle::Tensor& tile_ids_per_batch, \
    const int num_blocks, \
    const int block_shape_q, \
    const int max_seq_len, \
    const int max_dec_len, \
    const float quant_max_bound, \
    const float quant_min_bound, \
    const float in_scale, \
    const int max_partition_size, \
    const int encoder_max_partition_size, \
    const int speculate_max_draft_token_num, \
    const bool causal, \
    const bool is_decoder, \
    const bool enable_prefill, \
    cudaStream_t& stream, \
    paddle::Tensor* out);

#define INSTANTIATE_APPEND_ATTENTION_C8(T, OutT, IsFP8) \
  template void CascadeAppendAttentionC8Kernel<T, OutT, IsFP8>( \
    const AppendAttnMetaData& meta_data, \
    const paddle::Tensor& qkv, \
    const paddle::Tensor& cache_k, \
    const paddle::Tensor& cache_v, \
    const paddle::optional<paddle::Tensor>& attn_mask, \
    const paddle::optional<paddle::Tensor>& cache_k_scale, \
    const paddle::optional<paddle::Tensor>& cache_v_scale, \
    const paddle::optional<paddle::Tensor>& cache_k_zp, \
    const paddle::optional<paddle::Tensor>& cache_v_zp, \
    const paddle::optional<paddle::Tensor>& shift_bias, \
    const paddle::optional<paddle::Tensor>& smooth_weight, \
    const paddle::Tensor& seq_lens_q, \
    const paddle::Tensor& seq_lens_kv, \
    const paddle::Tensor& seq_lens_encoder, \
    const paddle::Tensor& batch_id_per_token, \
    const paddle::Tensor& cu_seqlens_q, \
    const paddle::Tensor& block_table, \
    const paddle::Tensor& batch_ids, \
    const paddle::Tensor& tile_ids_per_batch, \
    const int num_blocks, \
    const int block_shape_q, \
    const int max_seq_len, \
    const int max_dec_len, \
    const float quant_max_bound, \
    const float quant_min_bound, \
    const float in_scale, \
    const int max_partition_size, \
    const int encoder_max_partition_size, \
    const int speculate_max_draft_token_num, \
    const bool causal, \
    const bool is_decoder, \
    const bool enable_prefill, \
    cudaStream_t& stream, \
    paddle::Tensor* out);

#define INSTANTIATE_APPEND_ATTENTION_C16(T, OutT) \
  template void CascadeAppendAttentionC16Kernel<T, OutT>( \
    const AppendAttnMetaData& meta_data, \
    const paddle::Tensor& qkv, \
    const paddle::Tensor& cache_k, \
    const paddle::Tensor& cache_v, \
    const paddle::optional<paddle::Tensor>& attn_mask, \
    const paddle::optional<paddle::Tensor>& cache_k_scale, \
    const paddle::optional<paddle::Tensor>& cache_v_scale, \
    const paddle::optional<paddle::Tensor>& cache_k_zp, \
    const paddle::optional<paddle::Tensor>& cache_v_zp, \
    const paddle::optional<paddle::Tensor>& shift_bias, \
    const paddle::optional<paddle::Tensor>& smooth_weight, \
    const paddle::Tensor& seq_lens_q, \
    const paddle::Tensor& seq_lens_kv, \
    const paddle::Tensor& seq_lens_encoder, \
    const paddle::Tensor& batch_id_per_token, \
    const paddle::Tensor& cu_seqlens_q, \
    const paddle::Tensor& block_table, \
    const paddle::Tensor& batch_ids, \
    const paddle::Tensor& tile_ids_per_batch, \
    const int num_blocks, \
    const int block_shape_q, \
    const int max_seq_len, \
    const int max_dec_len, \
    const float quant_max_bound, \
    const float quant_min_bound, \
    const float in_scale, \
    const int max_partition_size, \
    const int encoder_max_partition_size, \
    const int speculate_max_draft_token_num, \
    const bool causal, \
    const bool is_decoder, \
    const bool enable_prefill, \
    cudaStream_t& stream, \
    paddle::Tensor* out);