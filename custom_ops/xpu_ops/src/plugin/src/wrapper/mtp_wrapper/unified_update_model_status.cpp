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

#include "xpu/plugin.h"
#include "xpu/refactor/impl/xdnn_impl.h"
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace fd_xpu3 {

__attribute__((global)) void unified_update_model_status_kernel(
    int *seq_lens_encoder,
    int *seq_lens_decoder,
    bool *has_running_seqs,
    int *mask_rollback,
    int64_t *step_input_ids,
    int *adaptive_step_input_len,
    int64_t *step_output_ids,
    int *step_output_len,
    bool *stop_flags,
    int *seq_lens_this_time,
    const bool *is_paused,
    int64_t *token_ids_all,
    const int64_t *prompt_lens,
    int64_t *step_idx,
    const int64_t *end_tokens,
    const int64_t *max_dec_len,
    int real_bsz,
    int max_bsz,
    int max_step_tokens,
    int max_model_len,
    int num_end_tokens,
    bool is_naive_mode,
    bool prefill_one_step_stop);
}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

bool is_end_token(int64_t token,
                  const int64_t *end_tokens,
                  int num_end_tokens) {
#pragma unroll 4
  for (int i = 0; i < num_end_tokens; i++) {
    if (token == end_tokens[i]) return true;
  }
  return false;
}

static int cpu_wrapper(api::Context *ctx,
                       int *seq_lens_encoder,
                       int *seq_lens_decoder,
                       bool *has_running_seqs,
                       int *mask_rollback,
                       int64_t *step_input_ids,
                       int *adaptive_step_input_len,
                       int64_t *step_output_ids,
                       int *step_output_len,
                       bool *stop_flags,
                       int *seq_lens_this_time,
                       const bool *is_paused,
                       int64_t *token_ids_all,
                       const int64_t *prompt_lens,
                       int64_t *step_idx,
                       const int64_t *end_tokens,
                       const int64_t *max_dec_len,
                       int real_bsz,
                       int max_bsz,
                       int max_step_tokens,
                       int max_model_len,
                       int num_end_tokens,
                       bool is_naive_mode,
                       bool prefill_one_step_stop) {
  int stop_flag_int = 0;

  for (int batch_id = 0; batch_id < max_bsz; batch_id++) {
    // Read state
    int cur_seq_len_encoder = seq_lens_encoder[batch_id];
    int cur_seq_len_decoder = seq_lens_decoder[batch_id];
    bool cur_stop_flag = stop_flags[batch_id];
    int output_len = 0;
    int64_t cur_step_idx = step_idx[batch_id];
    bool cur_is_paused = is_paused[batch_id];

    bool is_running = !cur_stop_flag && !cur_is_paused;

    // Compute output length
    if (is_running) {
      if (is_naive_mode) {
        output_len = 1;
      } else {
        output_len = step_output_len[batch_id];
      }
    }

    // EOS detection
    if (is_running && output_len > 0) {
      bool hit_stop = false;
      int64_t *output_ids = &step_output_ids[batch_id * max_step_tokens];

      for (int i = 0; i < output_len; i++) {
        cur_step_idx++;
        int64_t token = output_ids[i];
        bool is_eos = is_end_token(token, end_tokens, num_end_tokens);
        bool max_len_hit = (cur_step_idx >= max_dec_len[batch_id]);

        if (is_eos || max_len_hit) {
          if (!is_eos) output_ids[i] = end_tokens[0];
          output_len = i + 1;
          cur_stop_flag = true;
          hit_stop = true;
          break;
        }
      }

      if (!hit_stop && prefill_one_step_stop && cur_seq_len_encoder > 0) {
        cur_stop_flag = true;
      }
    }

    // Update state and write back
    if (is_running) {
      if (cur_stop_flag) {
        stop_flag_int += 1;
        if (output_len == 0) cur_seq_len_decoder = 0;
        stop_flags[batch_id] = true;
        mask_rollback[batch_id] = 0;
      } else if (cur_seq_len_encoder == 0) {
        cur_seq_len_decoder += output_len;
        mask_rollback[batch_id] = seq_lens_this_time[batch_id] - output_len;
      } else {
        mask_rollback[batch_id] = 0;
      }

      if (cur_seq_len_encoder > 0) {
        cur_seq_len_decoder += cur_seq_len_encoder;
        cur_seq_len_encoder = 0;
      }

      seq_lens_encoder[batch_id] = cur_seq_len_encoder;
      seq_lens_decoder[batch_id] = cur_seq_len_decoder;
      step_output_len[batch_id] = output_len;
      step_idx[batch_id] = cur_step_idx;

      // Write history to token_ids_all
      if (cur_step_idx > 0 && output_len > 0) {
        // Bounds check: highest write index is prompt_lens + cur_step_idx
        if (prompt_lens[batch_id] + cur_step_idx < max_model_len) {
          int64_t *token_ids_all_now =
              &token_ids_all[batch_id * max_model_len + prompt_lens[batch_id]];
          int64_t *output_ids = &step_output_ids[batch_id * max_step_tokens];
          for (int i = 0; i < output_len; i++) {
            token_ids_all_now[cur_step_idx - i] =
                output_ids[output_len - 1 - i];
          }
        }
      }

      // Setup next input
      if (output_len > 0) {
        step_input_ids[batch_id * max_step_tokens] =
            step_output_ids[batch_id * max_step_tokens + output_len - 1];
      }

      if (is_naive_mode) {
        seq_lens_this_time[batch_id] = cur_stop_flag ? 0 : 1;
      }
    } else if (batch_id >= real_bsz) {
      // Padding slot: just count as stopped, don't modify state
      stop_flag_int += 1;
    } else {
      // Stopped or paused slot (batch_id < real_bsz)
      stop_flag_int += 1;
      stop_flags[batch_id] = true;
      seq_lens_decoder[batch_id] = 0;
      seq_lens_this_time[batch_id] = 0;
      step_output_len[batch_id] = 0;
    }
  }
  has_running_seqs[0] = stop_flag_int < max_bsz;
  return api::SUCCESS;
}

static int xpu3_wrapper(api::Context *ctx,
                        int *seq_lens_encoder,
                        int *seq_lens_decoder,
                        bool *has_running_seqs,
                        int *mask_rollback,
                        int64_t *step_input_ids,
                        int *adaptive_step_input_len,
                        int64_t *step_output_ids,
                        int *step_output_len,
                        bool *stop_flags,
                        int *seq_lens_this_time,
                        const bool *is_paused,
                        int64_t *token_ids_all,
                        const int64_t *prompt_lens,
                        int64_t *step_idx,
                        const int64_t *end_tokens,
                        const int64_t *max_dec_len,
                        int real_bsz,
                        int max_bsz,
                        int max_step_tokens,
                        int max_model_len,
                        int num_end_tokens,
                        bool is_naive_mode,
                        bool prefill_one_step_stop) {
  using XPU_INT64 = typename api::XPUIndexType<int64_t>::type;
  int32_t ret_xre =
      fd_xpu3::unified_update_model_status_kernel<<<ctx->ncluster(),
                                                    64,
                                                    ctx->xpu_stream>>>(
          seq_lens_encoder,
          seq_lens_decoder,
          has_running_seqs,
          mask_rollback,
          reinterpret_cast<XPU_INT64 *>(step_input_ids),
          adaptive_step_input_len,
          reinterpret_cast<XPU_INT64 *>(step_output_ids),
          step_output_len,
          stop_flags,
          seq_lens_this_time,
          is_paused,
          reinterpret_cast<XPU_INT64 *>(token_ids_all),
          reinterpret_cast<const XPU_INT64 *>(prompt_lens),
          reinterpret_cast<XPU_INT64 *>(step_idx),
          reinterpret_cast<const XPU_INT64 *>(end_tokens),
          reinterpret_cast<const XPU_INT64 *>(max_dec_len),
          real_bsz,
          max_bsz,
          max_step_tokens,
          max_model_len,
          num_end_tokens,
          is_naive_mode,
          prefill_one_step_stop);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int unified_update_model_status(api::Context *ctx,
                                int *seq_lens_encoder,
                                int *seq_lens_decoder,
                                bool *has_running_seqs,
                                int *mask_rollback,
                                int64_t *step_input_ids,
                                int *adaptive_step_input_len,
                                int64_t *step_output_ids,
                                int *step_output_len,
                                bool *stop_flags,
                                int *seq_lens_this_time,
                                const bool *is_paused,
                                int64_t *token_ids_all,
                                const int64_t *prompt_lens,
                                int64_t *step_idx,
                                const int64_t *end_tokens,
                                const int64_t *max_dec_len,
                                int real_bsz,
                                int max_bsz,
                                int max_step_tokens,
                                int max_model_len,
                                int num_end_tokens,
                                bool is_naive_mode,
                                bool prefill_one_step_stop) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "unified_update_model_status", int);
  WRAPPER_DUMP_PARAM6(ctx,
                      seq_lens_encoder,
                      seq_lens_decoder,
                      has_running_seqs,
                      mask_rollback,
                      step_input_ids,
                      adaptive_step_input_len);
  WRAPPER_DUMP_PARAM6(ctx,
                      step_output_ids,
                      step_output_len,
                      stop_flags,
                      seq_lens_this_time,
                      is_paused,
                      token_ids_all);
  WRAPPER_DUMP_PARAM6(
      ctx, prompt_lens, step_idx, end_tokens, max_dec_len, real_bsz, max_bsz);
  WRAPPER_DUMP_PARAM5(ctx,
                      max_step_tokens,
                      max_model_len,
                      num_end_tokens,
                      is_naive_mode,
                      prefill_one_step_stop);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, int, max_bsz, seq_lens_encoder);
  WRAPPER_CHECK_PTR(ctx, int, max_bsz, seq_lens_decoder);
  WRAPPER_CHECK_PTR(ctx, bool, 1, has_running_seqs);
  WRAPPER_CHECK_PTR(ctx, int, max_bsz, mask_rollback);
  WRAPPER_CHECK_PTR(ctx, int64_t, max_bsz * max_step_tokens, step_input_ids);
  // WRAPPER_CHECK_PTR(ctx, int, 0, adaptive_step_input_len); // Temporarily
  // unused
  WRAPPER_CHECK_PTR(ctx, int64_t, max_bsz * max_step_tokens, step_output_ids);
  WRAPPER_CHECK_PTR(ctx, int, max_bsz, step_output_len);
  WRAPPER_CHECK_PTR(ctx, bool, max_bsz, stop_flags);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, bool, max_bsz, is_paused);
  WRAPPER_CHECK_PTR(ctx, int64_t, max_bsz * max_model_len, token_ids_all);
  WRAPPER_CHECK_PTR(ctx, int64_t, max_bsz, prompt_lens);
  WRAPPER_CHECK_PTR(ctx, int64_t, max_bsz, step_idx);
  WRAPPER_CHECK_PTR(ctx, int64_t, num_end_tokens, end_tokens);
  WRAPPER_CHECK_PTR(ctx, int64_t, max_bsz, max_dec_len);
  WRAPPER_ASSERT_GE(ctx, max_bsz, real_bsz);
  WRAPPER_ASSERT_GE(ctx, 1024, num_end_tokens);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       seq_lens_encoder,
                       seq_lens_decoder,
                       has_running_seqs,
                       mask_rollback,
                       step_input_ids,
                       adaptive_step_input_len,
                       step_output_ids,
                       step_output_len,
                       stop_flags,
                       seq_lens_this_time,
                       is_paused,
                       token_ids_all,
                       prompt_lens,
                       step_idx,
                       end_tokens,
                       max_dec_len,
                       real_bsz,
                       max_bsz,
                       max_step_tokens,
                       max_model_len,
                       num_end_tokens,
                       is_naive_mode,
                       prefill_one_step_stop);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        seq_lens_encoder,
                        seq_lens_decoder,
                        has_running_seqs,
                        mask_rollback,
                        step_input_ids,
                        adaptive_step_input_len,
                        step_output_ids,
                        step_output_len,
                        stop_flags,
                        seq_lens_this_time,
                        is_paused,
                        token_ids_all,
                        prompt_lens,
                        step_idx,
                        end_tokens,
                        max_dec_len,
                        real_bsz,
                        max_bsz,
                        max_step_tokens,
                        max_model_len,
                        num_end_tokens,
                        is_naive_mode,
                        prefill_one_step_stop);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace fastdeploy
