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

#include "append_attention_func.cuh"

template <typename T,
          uint32_t GROUP_SIZE,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          bool CAUSAL,
          uint32_t BLOCK_SHAPE_Q,
          uint32_t NUM_WARP_Q,
          typename OutT = T,
          bool ENABLE_PREFILL = true>
void MultiQueryAppendC4Attention(
    const AppendAttnMetaData &meta_data,
    const paddle::Tensor &qkv,
    const paddle::Tensor &cache_k,
    const paddle::Tensor &cache_v,
    const paddle::optional<paddle::Tensor> &attn_mask,
    const paddle::Tensor &cache_k_scale,
    const paddle::Tensor &cache_v_scale,
    const paddle::optional<paddle::Tensor> &cache_k_zp,
    const paddle::optional<paddle::Tensor> &cache_v_zp,
    const paddle::optional<paddle::Tensor> &shift_bias,
    const paddle::optional<paddle::Tensor> &smooth_weight,
    const paddle::optional<paddle::Tensor> &sinks,
    const paddle::Tensor &seq_lens_q,
    const paddle::Tensor &seq_lens_kv,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &batch_id_per_token,
    const paddle::Tensor &cu_seqlens_q,
    const paddle::Tensor &block_table,
    const paddle::Tensor &batch_ids,
    const paddle::Tensor &tile_ids_per_batch,
    const int num_blocks_x_cpu,
    const int max_seq_len,
    const int max_dec_len,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool is_decoder,
    cudaStream_t &stream,
    paddle::Tensor *out,
    const int sliding_window);
