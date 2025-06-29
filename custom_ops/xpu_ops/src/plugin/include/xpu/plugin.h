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
/*
 * copyright (C) 2022 KUNLUNXIN, Inc
 */

#pragma once
#include "xpu/xdnn.h"

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T>
DLL_EXPORT int set_stop_value_multi_ends(Context *ctx, bool *stop_flags,
                                         T *topk_ids, T *next_tokens,
                                         const T *end_ids, const int *seq_lens,
                                         const int bs, const int end_length,
                                         const bool beam_search);

DLL_EXPORT int set_value_by_flags_and_idx(Context *ctx, const bool *stop_flags,
                                          int64_t *pre_ids_all,
                                          const int64_t *input_ids,
                                          const int *seq_lens_encoder,
                                          const int *seq_lens_decoder,
                                          const int64_t *step_idx, int bs,
                                          int length, int length_input_ids);

template <typename T>
DLL_EXPORT int token_penalty_multi_scores(
    Context *ctx, const int64_t *pre_ids, T *logits, const T *penalty_scores,
    const T *frequency_scores, const T *presence_scores,
    const float *temperatures, const int64_t *cur_len, const int64_t *min_len,
    const int64_t *eos_token_id, const int64_t *bad_words, const int64_t bs,
    const int64_t length, const int64_t length_id, const int64_t end_length,
    const int64_t length_bad_words);

DLL_EXPORT int get_padding_offset(Context *ctx, int *padding_offset,
                                  int *cum_offsets_out, int *cu_seqlens_q,
                                  int *cu_seqlens_k, int64_t *x_remove_padding,
                                  const int64_t *input_ids,
                                  const int *cum_offsets, const int *seq_lens,
                                  const int max_seq_len, const int bs);

DLL_EXPORT int update_inputs(Context *ctx, bool *not_need_stop,
                             int *seq_lens_this_time, int *seq_lens_encoder,
                             int *seq_lens_decoder, int64_t *input_ids,
                             const int64_t *stop_nums, const bool *stop_flags,
                             const bool *is_block_step,
                             const int64_t *next_tokens, const int bsz,
                             const int max_bsz, const int input_ids_stride);

DLL_EXPORT int free_and_dispatch_block(
    Context *ctx, bool *stop_flags, int *seq_lens_this_time,
    int *seq_lens_decoder, int *block_tables, int *encoder_block_lens,
    bool *is_block_step,
    int *step_block_list, // [bsz]
    int *step_len, int *recover_block_list, int *recover_len,
    int *need_block_list, int *need_block_len, int *used_list_len,
    int *free_list, int *free_list_len, int64_t *first_token_ids, const int bsz,
    const int block_size, const int block_num_per_seq,
    const int max_decoder_block_num);

DLL_EXPORT int
recover_block(Context *ctx,
              int *recover_block_list, // [bsz]
              int *recover_len, bool *stop_flags, int *seq_lens_this_time,
              const int *ori_seq_lens_encoder, int *seq_lens_encoder,
              const int *seq_lens_decoder, int *block_tables, int *free_list,
              int *free_list_len, int64_t *input_ids, const int64_t *pre_ids,
              const int64_t *step_idx, const int *encoder_block_lens,
              const int *used_list_len, const int64_t *next_tokens,
              const int64_t *first_token_ids, const int bsz,
              const int block_num_per_seq, const int length,
              const int pre_id_length);

template <typename TX, typename TY>
DLL_EXPORT int
eb_adjust_batch(Context *ctx, const TX *x, TY *y,
                VectorParam<int32_t> &encoder_seqs_lods, // NOLINT
                VectorParam<int32_t> &encoder_batch_map, // NOLINT
                VectorParam<int32_t> &decoder_batch_map, // NOLINT
                int64_t hidden_dim);

template <typename TX, typename TY>
DLL_EXPORT int
eb_gather_next_token(Context *ctx, const TX *x, TY *y,
                     VectorParam<int32_t> &encoder_seqs_lods, // NOLINT
                     VectorParam<int32_t> &encoder_batch_map, // NOLINT
                     VectorParam<int32_t> &decoder_batch_map, // NOLINT
                     int64_t hidden_dim);

template <typename TX, typename TSCALE = float, typename TY = int8_t>
DLL_EXPORT int quant2d_per_channel(api::Context *ctx, const TX *x,
                                   const TSCALE *scale_in, TY *y,
                                   TSCALE *scale_out, int64_t m, int64_t n);
} // namespace plugin
} // namespace api
} // namespace xpu
} // namespace baidu
