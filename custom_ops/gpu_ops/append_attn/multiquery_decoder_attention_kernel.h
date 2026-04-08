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

#include "decode_attention_func.cuh"

template <typename T, uint32_t GROUP_SIZE, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_V, uint32_t BLOCK_SIZE, bool CAUSAL, uint32_t NUM_STAGE, uint32_t cache_bytes, uint32_t DEAL_EACH_TIME>
void MultiQueryDecoderAttention(
  const AppendAttnMetaData& meta_data,
  cudaStream_t &stream,
  const paddle::Tensor &q,
  const paddle::Tensor &cache_k, // [max_block_num, num_kv_heads, block_size, head_dim]
  const paddle::Tensor &cache_v, // [num_kv_heads, head_dim]
  const paddle::optional<paddle::Tensor>& attn_mask,
  const paddle::optional<paddle::Tensor>& shift_bias,
  const paddle::optional<paddle::Tensor>& smooth_weight,
  const paddle::Tensor &seq_lens_q,
  const paddle::Tensor &seq_lens_kv,
  const paddle::Tensor &batch_id_per_token,
  const paddle::Tensor &cu_seqlens_q,
  const paddle::Tensor &block_table,
  const int max_seq_len,
  const int max_dec_len,
  const float rope_scale,
  const float rope_theta,
  const float softmax_scale,
  const float in_scale,
  paddle::Tensor *out);
