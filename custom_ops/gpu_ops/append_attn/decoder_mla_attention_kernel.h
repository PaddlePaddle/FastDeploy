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
#pragma once

#include "helper.h"
#include "multiquery_decoder_attention_kernel.h"
#include "utils.cuh"

template <typename T>
void DecodeMLAAttentionKernel(
    const AppendAttnMetaData &meta_data,
    const paddle::Tensor &q,  // [token_num, num_heads, head_dim]
    const paddle::Tensor &cache_k,
    const paddle::Tensor &cache_v,
    const paddle::optional<paddle::Tensor> &attn_mask,
    const paddle::optional<paddle::Tensor> &shift_bias,
    const paddle::optional<paddle::Tensor> &smooth_weight,
    const paddle::Tensor &seq_lens_q,  // q_seq_len is 1
    const paddle::Tensor &seq_lens_kv,
    const paddle::Tensor &batch_id_per_token,
    const paddle::Tensor &cu_seqlens_q,
    const paddle::Tensor &block_table,
    int max_seq_len,
    int max_dec_len,
    float softmax_scale,
    float in_scale,
    bool causal,
    cudaStream_t &stream,
    paddle::Tensor *out) {
  const auto token_num = meta_data.token_nums;
  const auto block_size = meta_data.block_size;
  const auto bsz = meta_data.batch_size;
  const auto num_heads = meta_data.q_num_heads;
  const auto group_size = meta_data.q_num_heads / meta_data.kv_num_heads;
  const auto head_dim_qk = meta_data.head_dims;
  const auto head_dim_v = meta_data.head_dims_v;
  const float rope_scale = 0.0;
  const float rope_theta = 0.0;
  const uint32_t deal_each_time = get_cascade_attention_deal_each_time();
  const uint32_t num_stage = get_cascade_attention_num_stages();
  const uint32_t num_threads = get_cascade_attention_num_threads();

  DISPATCH_CAUSAL(
      causal,
      CAUSAL,
      {DISPATCH_MLA_GROUP_SIZE(
          group_size,
          GROUP_SIZE,
          {DISPATCH_MLA_HEAD_DIM(
              head_dim_qk,
              HEAD_DIM_QK,
              {DISPATCH_MLA_HEAD_DIM(
                  head_dim_v,
                  HEAD_DIM_V,
                  {DISPATCH_BLOCK_SIZE(
                      block_size,
                      BLOCK_SIZE,
                      {DISPATCH_DEAL_EACH_TIME(deal_each_time, DEAL_EACH_TIME, {
                        MultiQueryDecoderAttention<T,
                                                   GROUP_SIZE,
                                                   HEAD_DIM_QK,
                                                   HEAD_DIM_V,
                                                   BLOCK_SIZE,
                                                   CAUSAL,
                                                   2,
                                                   16,
                                                   DEAL_EACH_TIME>(
                            meta_data,
                            stream,
                            q,
                            cache_k,
                            cache_v,
                            attn_mask,
                            shift_bias,
                            smooth_weight,
                            seq_lens_q,
                            seq_lens_kv,
                            batch_id_per_token,
                            cu_seqlens_q,
                            block_table,
                            max_seq_len,
                            max_dec_len,
                            rope_scale,
                            rope_theta,
                            softmax_scale,
                            in_scale,
                            out);
                      })})})})})});
}

template void DecodeMLAAttentionKernel<paddle::bfloat16>(
    const AppendAttnMetaData &meta_data,
    const paddle::Tensor &q,  // [token_num, num_heads, head_dim]
    const paddle::Tensor &cache_k,
    const paddle::Tensor &cache_v,
    const paddle::optional<paddle::Tensor> &attn_mask,
    const paddle::optional<paddle::Tensor> &shift_bias,
    const paddle::optional<paddle::Tensor> &smooth_weight,
    const paddle::Tensor &seq_lens_q,  // q_seq_len is 1
    const paddle::Tensor &seq_lens_kv,
    const paddle::Tensor &batch_id_per_token,
    const paddle::Tensor &cu_seqlens_q,
    const paddle::Tensor &block_table,
    int max_seq_len,
    int max_dec_len,
    float softmax_scale,
    float in_scale,
    bool causal,
    cudaStream_t &stream,
    paddle::Tensor *out);

template void DecodeMLAAttentionKernel<paddle::float16>(
    const AppendAttnMetaData &meta_data,
    const paddle::Tensor &q,  // [token_num, num_heads, head_dim]
    const paddle::Tensor &cache_k,
    const paddle::Tensor &cache_v,
    const paddle::optional<paddle::Tensor> &attn_mask,
    const paddle::optional<paddle::Tensor> &shift_bias,
    const paddle::optional<paddle::Tensor> &smooth_weight,
    const paddle::Tensor &seq_lens_q,  // q_seq_len is 1
    const paddle::Tensor &seq_lens_kv,
    const paddle::Tensor &batch_id_per_token,
    const paddle::Tensor &cu_seqlens_q,
    const paddle::Tensor &block_table,
    int max_seq_len,
    int max_dec_len,
    float softmax_scale,
    float in_scale,
    bool causal,
    cudaStream_t &stream,
    paddle::Tensor *out);
