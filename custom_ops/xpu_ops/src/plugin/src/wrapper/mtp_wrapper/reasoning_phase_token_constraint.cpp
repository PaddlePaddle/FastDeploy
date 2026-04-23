// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#include <cstring>
#include "xpu/plugin.h"
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace fd_xpu3 {

__attribute__((global)) void update_reasoning_status(
    const bool* stop_flags,
    const int* seq_lens_encoder,
    const int64_t* step_idx,
    const int64_t* token_ids_all,
    const int64_t* prompt_lens,
    const bool* enable_thinking,
    int* reasoning_status,
    int bs,
    int max_seq_len,
    int64_t think_end_id,
    int64_t line_break_id);

template <typename T>
__attribute__((global)) void apply_token_enforce_generation_scores(
    const T* logits_src,
    T* logits_dst,
    const int64_t* allowed_tokens,
    const int* reasoning_status,
    const int* batch_id_per_token_output,
    const int* cu_seqlens_q_output,
    int max_bsz,
    int vocab_size,
    int allowed_tokens_len,
    int token_num);

}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

template <typename T>
static int cpu_wrapper(api::Context* ctx,
                       const T* logits_src,
                       T* logits_dst,
                       const int64_t* token_ids_all,
                       const int64_t* prompt_lens,
                       const bool* stop_flags,
                       const int* seq_lens_encoder,
                       const int64_t* step_idx,
                       const int64_t* allowed_tokens,
                       int* reasoning_status,
                       const int* batch_id_per_token_output,
                       const int* cu_seqlens_q_output,
                       const bool* enable_thinking,
                       int64_t think_end_id,
                       int64_t line_break_id,
                       int bs,
                       int token_num,
                       int vocab_size,
                       int max_seq_len,
                       int allowed_tokens_len) {
  // Step 1: Update reasoning status
  for (int i = 0; i < bs; i++) {
    if (stop_flags[i] || reasoning_status[i] == 3) continue;

    int64_t cur_step = step_idx[i];
    const int64_t* pre_ids_now =
        token_ids_all + i * max_seq_len + prompt_lens[i];
    int64_t t0 = (cur_step >= 1) ? pre_ids_now[cur_step - 1] : -1;
    int64_t t1 = (cur_step >= 2) ? pre_ids_now[cur_step - 2] : -1;
    int64_t t2 = (cur_step >= 3) ? pre_ids_now[cur_step - 3] : -1;
    int64_t t3 = (cur_step >= 4) ? pre_ids_now[cur_step - 4] : -1;

    int new_status = reasoning_status[i];

    // x = 0 -> x = 1 or x = 2
    if (reasoning_status[i] == 0) {
      if (!enable_thinking[i] && seq_lens_encoder[i] > 0 && cur_step == 0) {
        new_status = 2;
      } else if (t0 == think_end_id || t1 == think_end_id ||
                 t2 == think_end_id || t3 == think_end_id) {
        new_status = 1;
      }
    }

    // x = 1 -> x = 2 or x = 3
    if (new_status == 1 && cur_step >= 4) {
      if (t3 == line_break_id && t2 == think_end_id && t1 == line_break_id &&
          t0 == line_break_id) {
        new_status = 2;
      } else if (t3 != think_end_id && t2 != think_end_id &&
                 t1 != think_end_id && t0 != think_end_id) {
        new_status = 3;
      }
    } else if (reasoning_status[i] == 2) {
      new_status = 3;
    }

    reasoning_status[i] = new_status;
  }

  // Step 2: Apply token enforce generation scores
  float fill_val_f = std::is_same<T, float16>::value ? -1e4f : -1e10f;
  T fill_val_t;
  api::cast<float, T>(ctx, &fill_val_f, &fill_val_t, 1);

  for (int token_idx = 0; token_idx < token_num; token_idx++) {
    int bs_idx = batch_id_per_token_output[token_idx];
    if (bs_idx < 0) continue;
    int query_start = cu_seqlens_q_output[bs_idx];
    bool is_first = (token_idx == query_start);
    if (!is_first || allowed_tokens_len == 0) continue;
    if (reasoning_status[bs_idx] != 2) continue;

    T* dst = logits_dst + token_idx * vocab_size;
    const T* src = logits_src + token_idx * vocab_size;

    // Clear all logits
    for (int v = 0; v < vocab_size; v++) {
      dst[v] = fill_val_t;
    }
    // Restore allowed tokens
    for (int j = 0; j < allowed_tokens_len; j++) {
      int64_t token_id = allowed_tokens[j];
      if ((unsigned)token_id < (unsigned)vocab_size) {
        dst[token_id] = src[token_id];
      }
    }
  }

  return api::SUCCESS;
}

template <typename T>
static int xpu3_wrapper(api::Context* ctx,
                        const T* logits_src,
                        T* logits_dst,
                        const int64_t* token_ids_all,
                        const int64_t* prompt_lens,
                        const bool* stop_flags,
                        const int* seq_lens_encoder,
                        const int64_t* step_idx,
                        const int64_t* allowed_tokens,
                        int* reasoning_status,
                        const int* batch_id_per_token_output,
                        const int* cu_seqlens_q_output,
                        const bool* enable_thinking,
                        int64_t think_end_id,
                        int64_t line_break_id,
                        int bs,
                        int token_num,
                        int vocab_size,
                        int max_seq_len,
                        int allowed_tokens_len) {
  using XPU_INT64 = typename api::XPUIndexType<int64_t>::type;

  // Kernel 1: Update reasoning status
  int32_t ret_xre = fd_xpu3::
      update_reasoning_status<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          stop_flags,
          seq_lens_encoder,
          reinterpret_cast<const XPU_INT64*>(step_idx),
          reinterpret_cast<const XPU_INT64*>(token_ids_all),
          reinterpret_cast<const XPU_INT64*>(prompt_lens),
          enable_thinking,
          reasoning_status,
          bs,
          max_seq_len,
          think_end_id,
          line_break_id);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);

  // Kernel 2: Apply token enforce generation scores
  ret_xre = fd_xpu3::apply_token_enforce_generation_scores<T>
      <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          logits_src,
          logits_dst,
          reinterpret_cast<const XPU_INT64*>(allowed_tokens),
          reasoning_status,
          batch_id_per_token_output,
          cu_seqlens_q_output,
          bs,
          vocab_size,
          allowed_tokens_len,
          token_num);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);

  return api::SUCCESS;
}

template <typename T>
int reasoning_phase_token_constraint(api::Context* ctx,
                                     const T* logits_src,
                                     T* logits_dst,
                                     const int64_t* token_ids_all,
                                     const int64_t* prompt_lens,
                                     const bool* stop_flags,
                                     const int* seq_lens_encoder,
                                     const int64_t* step_idx,
                                     const int64_t* allowed_tokens,
                                     int* reasoning_status,
                                     const int* batch_id_per_token_output,
                                     const int* cu_seqlens_q_output,
                                     const bool* enable_thinking,
                                     int64_t think_end_id,
                                     int64_t line_break_id,
                                     int bs,
                                     int token_num,
                                     int vocab_size,
                                     int max_seq_len,
                                     int allowed_tokens_len) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "reasoning_phase_token_constraint", T);
  WRAPPER_DUMP_PARAM6(ctx,
                      logits_src,
                      logits_dst,
                      token_ids_all,
                      prompt_lens,
                      stop_flags,
                      seq_lens_encoder);
  WRAPPER_DUMP_PARAM6(ctx,
                      step_idx,
                      allowed_tokens,
                      reasoning_status,
                      batch_id_per_token_output,
                      cu_seqlens_q_output,
                      enable_thinking);
  WRAPPER_DUMP_PARAM4(ctx, think_end_id, line_break_id, bs, max_seq_len);
  WRAPPER_DUMP_PARAM3(ctx, token_num, vocab_size, allowed_tokens_len);
  WRAPPER_DUMP(ctx);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T>(ctx,
                          logits_src,
                          logits_dst,
                          token_ids_all,
                          prompt_lens,
                          stop_flags,
                          seq_lens_encoder,
                          step_idx,
                          allowed_tokens,
                          reasoning_status,
                          batch_id_per_token_output,
                          cu_seqlens_q_output,
                          enable_thinking,
                          think_end_id,
                          line_break_id,
                          bs,
                          token_num,
                          vocab_size,
                          max_seq_len,
                          allowed_tokens_len);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper<T>(ctx,
                           logits_src,
                           logits_dst,
                           token_ids_all,
                           prompt_lens,
                           stop_flags,
                           seq_lens_encoder,
                           step_idx,
                           allowed_tokens,
                           reasoning_status,
                           batch_id_per_token_output,
                           cu_seqlens_q_output,
                           enable_thinking,
                           think_end_id,
                           line_break_id,
                           bs,
                           token_num,
                           vocab_size,
                           max_seq_len,
                           allowed_tokens_len);
  }

  WRAPPER_UNIMPLEMENTED(ctx);
}

template int reasoning_phase_token_constraint<float>(
    api::Context* ctx,
    const float* logits_src,
    float* logits_dst,
    const int64_t* token_ids_all,
    const int64_t* prompt_lens,
    const bool* stop_flags,
    const int* seq_lens_encoder,
    const int64_t* step_idx,
    const int64_t* allowed_tokens,
    int* reasoning_status,
    const int* batch_id_per_token_output,
    const int* cu_seqlens_q_output,
    const bool* enable_thinking,
    int64_t think_end_id,
    int64_t line_break_id,
    int bs,
    int token_num,
    int vocab_size,
    int max_seq_len,
    int allowed_tokens_len);
template int reasoning_phase_token_constraint<float16>(
    api::Context* ctx,
    const float16* logits_src,
    float16* logits_dst,
    const int64_t* token_ids_all,
    const int64_t* prompt_lens,
    const bool* stop_flags,
    const int* seq_lens_encoder,
    const int64_t* step_idx,
    const int64_t* allowed_tokens,
    int* reasoning_status,
    const int* batch_id_per_token_output,
    const int* cu_seqlens_q_output,
    const bool* enable_thinking,
    int64_t think_end_id,
    int64_t line_break_id,
    int bs,
    int token_num,
    int vocab_size,
    int max_seq_len,
    int allowed_tokens_len);
template int reasoning_phase_token_constraint<bfloat16>(
    api::Context* ctx,
    const bfloat16* logits_src,
    bfloat16* logits_dst,
    const int64_t* token_ids_all,
    const int64_t* prompt_lens,
    const bool* stop_flags,
    const int* seq_lens_encoder,
    const int64_t* step_idx,
    const int64_t* allowed_tokens,
    int* reasoning_status,
    const int* batch_id_per_token_output,
    const int* cu_seqlens_q_output,
    const bool* enable_thinking,
    int64_t think_end_id,
    int64_t line_break_id,
    int bs,
    int token_num,
    int vocab_size,
    int max_seq_len,
    int allowed_tokens_len);

}  // namespace plugin
}  // namespace fastdeploy
