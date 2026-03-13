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

#include <algorithm>
#include <numeric>
#include "xpu/plugin.h"
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace fd_xpu3 {

__attribute__((global)) void update_attn_mask_offsets(
    int* attn_mask_offsets,
    const int* seq_lens_this_time,
    const int* seq_lens_encoder,
    const int* seq_lens_decoder,
    const int* cu_seqlens_q,
    const int* attn_mask_offsets_full,
    int* attn_mask_offsets_decoder,
    const bool* is_block_step,
    int* decode_states,
    int* mask_rollback,
    const int real_bsz,
    const int max_model_len,
    const int decode_states_len);

}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

static int cpu_wrapper(int* attn_mask_offsets,
                       const int* seq_lens_this_time,
                       const int* seq_lens_encoder,
                       const int* seq_lens_decoder,
                       const int* cu_seqlens_q,
                       const int* attn_mask_offsets_full,
                       int* attn_mask_offsets_decoder,
                       const bool* is_block_step,
                       int* decode_states,
                       int* mask_rollback,
                       int real_bsz,
                       int max_model_len,
                       int decode_states_len) {
  for (int bid = 0; bid < real_bsz; bid++) {
    if (is_block_step[bid]) continue;

    int seq_len_this_time_val = seq_lens_this_time[bid];
    int seq_len_encoder_val = seq_lens_encoder[bid];
    int seq_len_decoder_val = seq_lens_decoder[bid];
    int query_start_id = cu_seqlens_q[bid];

    const int* attn_mask_offsets_full_now =
        attn_mask_offsets_full + bid * max_model_len;
    int* decode_states_now = decode_states + bid * decode_states_len;
    // Status: stop
    if (seq_len_encoder_val == 0 && seq_len_decoder_val == 0) {
      continue;
    } else if (seq_len_encoder_val > 0) {
      for (int i = 0; i < seq_len_this_time_val; i++) {
        if (*decode_states_now == 2 && seq_len_decoder_val > 0) {
          // Status: vision generate phase
          attn_mask_offsets[(query_start_id + i) * 2 + 1] =
              seq_len_decoder_val + seq_len_this_time_val;
        } else {
          // Status: prefill -- normal or chunk_prefill
          attn_mask_offsets[(query_start_id + i) * 2 + 1] =
              attn_mask_offsets_full_now[i] + 1;
        }
      }
    } else if (seq_len_decoder_val > 0) {
      attn_mask_offsets_decoder[bid] -= mask_rollback[bid];
      mask_rollback[bid] = 0;
      for (int i = 0; i < seq_len_this_time_val; i++) {
        attn_mask_offsets[(query_start_id + i) * 2 + 1] =
            attn_mask_offsets_decoder[bid] + 1 + i;
      }
      attn_mask_offsets_decoder[bid] += seq_len_this_time_val;

      // Speculative decoding in text_generation
      if (seq_len_this_time_val > 1) {
        for (int i = 0; i < decode_states_len; i++) {
          decode_states_now[i] = (i < seq_len_this_time_val) ? 0 : -1;
        }
      }
    }
  }
  return api::SUCCESS;
}

static int xpu3_wrapper(api::Context* ctx,
                        int* attn_mask_offsets,
                        const int* seq_lens_this_time,
                        const int* seq_lens_encoder,
                        const int* seq_lens_decoder,
                        const int* cu_seqlens_q,
                        const int* attn_mask_offsets_full,
                        int* attn_mask_offsets_decoder,
                        const bool* is_block_step,
                        int* decode_states,
                        int* mask_rollback,
                        int real_bsz,
                        int max_model_len,
                        int decode_states_len) {
  int32_t ret_xre = fd_xpu3::
      update_attn_mask_offsets<<<ctx->ncluster(), 1, ctx->xpu_stream>>>(
          attn_mask_offsets,
          seq_lens_this_time,
          seq_lens_encoder,
          seq_lens_decoder,
          cu_seqlens_q,
          attn_mask_offsets_full,
          attn_mask_offsets_decoder,
          is_block_step,
          decode_states,
          mask_rollback,
          real_bsz,
          max_model_len,
          decode_states_len);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int update_attn_mask_offsets(api::Context* ctx,
                             int* attn_mask_offsets,
                             const int* seq_lens_this_time,
                             const int* seq_lens_encoder,
                             const int* seq_lens_decoder,
                             const int* cu_seqlens_q,
                             const int* attn_mask_offsets_full,
                             int* attn_mask_offsets_decoder,
                             const bool* is_block_step,
                             int* decode_states,
                             int* mask_rollback,
                             int real_bsz,
                             int max_model_len,
                             int decode_states_len) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "update_attn_mask_offsets", int);
  WRAPPER_DUMP_PARAM5(ctx,
                      attn_mask_offsets,
                      seq_lens_this_time,
                      seq_lens_encoder,
                      seq_lens_decoder,
                      cu_seqlens_q);
  WRAPPER_DUMP_PARAM5(ctx,
                      attn_mask_offsets_full,
                      attn_mask_offsets_decoder,
                      is_block_step,
                      decode_states,
                      mask_rollback);
  WRAPPER_DUMP_PARAM3(ctx, real_bsz, max_model_len, decode_states_len);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_encoder);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_decoder);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, cu_seqlens_q);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz* max_model_len, attn_mask_offsets_full);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, attn_mask_offsets_decoder);
  WRAPPER_CHECK_PTR(ctx, bool, real_bsz, is_block_step);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz* decode_states_len, decode_states);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, mask_rollback);

  WRAPPER_ASSERT_GT(ctx, real_bsz, 0);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(attn_mask_offsets,
                       seq_lens_this_time,
                       seq_lens_encoder,
                       seq_lens_decoder,
                       cu_seqlens_q,
                       attn_mask_offsets_full,
                       attn_mask_offsets_decoder,
                       is_block_step,
                       decode_states,
                       mask_rollback,
                       real_bsz,
                       max_model_len,
                       decode_states_len);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        attn_mask_offsets,
                        seq_lens_this_time,
                        seq_lens_encoder,
                        seq_lens_decoder,
                        cu_seqlens_q,
                        attn_mask_offsets_full,
                        attn_mask_offsets_decoder,
                        is_block_step,
                        decode_states,
                        mask_rollback,
                        real_bsz,
                        max_model_len,
                        decode_states_len);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace fastdeploy
