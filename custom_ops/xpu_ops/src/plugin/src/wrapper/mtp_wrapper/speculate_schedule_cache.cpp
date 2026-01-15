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

namespace xpu3 {
namespace plugin {
__attribute__((global)) void speculate_schedule_cache(
    const int64_t *draft_tokens,
    int *block_tables,
    bool *stop_flags,
    const int64_t *prompt_lens,
    int *seq_lens_this_time,
    int *seq_lens_encoder,
    int *seq_lens_decoder,
    int *step_seq_lens_decoder,
    int64_t *step_draft_tokens,
    int *step_seq_lens_this_time,
    int *accept_num,
    int64_t *accept_tokens,
    bool *is_block_step,
    bool *not_need_stop,
    const int64_t *stop_nums,
    const int real_bsz,
    const int max_bsz,
    const int max_next_step_tokens,
    const int draft_tokens_len,
    const int accept_tokens_len,
    const int block_size,
    const int block_num_per_seq,
    const bool prefill_one_step_stop);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context *ctx,
                       const int64_t *draft_tokens,
                       int *block_tables,
                       bool *stop_flags,
                       const int64_t *prompt_lens,
                       int *seq_lens_this_time,
                       int *seq_lens_encoder,
                       int *seq_lens_decoder,
                       int *step_seq_lens_decoder,
                       int64_t *step_draft_tokens,
                       int *step_seq_lens_this_time,
                       int *accept_num,
                       int64_t *accept_tokens,
                       bool *is_block_step,
                       bool *not_need_stop,
                       const int64_t *stop_nums,
                       const int real_bsz,
                       const int max_bsz,
                       const int max_next_step_tokens,
                       const int draft_tokens_len,
                       const int accept_tokens_len,
                       const int block_size,
                       const int block_num_per_seq,
                       const bool prefill_one_step_stop) {
  int stop_sum_now = 0;
  for (int bid = 0; bid < real_bsz; bid++) {
    if (!stop_flags[bid]) {
      const int64_t *draft_tokens_now = draft_tokens + bid * draft_tokens_len;
      int64_t *step_draft_tokens_now =
          step_draft_tokens + bid * draft_tokens_len;
      int *block_table_now = block_tables + bid * block_num_per_seq;
      int64_t *accept_tokens_now = accept_tokens + bid * accept_tokens_len;

      if (seq_lens_decoder[bid] >= prompt_lens[bid]) {
        const int max_possible_block_idx =
            (seq_lens_decoder[bid] + max_next_step_tokens) / block_size;

        if (prefill_one_step_stop) {
          stop_flags[bid] = true;
          seq_lens_this_time[bid] = 0;
          seq_lens_decoder[bid] = 0;
          seq_lens_encoder[bid] = 0;
          accept_num[bid] = 0;
          stop_sum_now += 1;
        } else if (max_possible_block_idx < block_num_per_seq &&
                   block_table_now[max_possible_block_idx] == -1) {
          is_block_step[bid] = true;
          step_seq_lens_this_time[bid] = seq_lens_this_time[bid];
          seq_lens_this_time[bid] = 0;
          stop_flags[bid] = true;
          stop_sum_now += 1;
          step_seq_lens_decoder[bid] = seq_lens_decoder[bid];
          seq_lens_decoder[bid] = 0;
          accept_num[bid] = 0;
          for (int i = 0; i < accept_tokens_len; i++) {
            accept_tokens_now[i] = -1;
          }
          for (int i = 0; i < draft_tokens_len; i++) {
            step_draft_tokens_now[i] = draft_tokens_now[i];
          }
        }
      } else {
        // prefill
        stop_flags[bid] = true;
        seq_lens_this_time[bid] = 0;
        seq_lens_decoder[bid] = 0;
        seq_lens_encoder[bid] = 0;
        accept_num[bid] = 0;
        stop_sum_now += 1;
      }
    } else {
      stop_sum_now += 1;
    }
  }

  // for (int bid = real_bsz; i < max_bsz; bid++) {
  //   stop_sum_now += 1;
  // }

  // printf("stop_sum %d \n", stop_sum);
  not_need_stop[0] = stop_sum_now < stop_nums[0];
  return api::SUCCESS;
}

static int xpu3_wrapper(Context *ctx,
                        const int64_t *draft_tokens,
                        int *block_tables,
                        bool *stop_flags,
                        const int64_t *prompt_lens,
                        int *seq_lens_this_time,
                        int *seq_lens_encoder,
                        int *seq_lens_decoder,
                        int *step_seq_lens_decoder,
                        int64_t *step_draft_tokens,
                        int *step_seq_lens_this_time,
                        int *accept_num,
                        int64_t *accept_tokens,
                        bool *is_block_step,
                        bool *not_need_stop,
                        const int64_t *stop_nums,
                        const int real_bsz,
                        const int max_bsz,
                        const int max_next_step_tokens,
                        const int draft_tokens_len,
                        const int accept_tokens_len,
                        const int block_size,
                        const int block_num_per_seq,
                        const bool prefill_one_step_stop) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  using XPU_TI = typename XPUIndexType<int64_t>::type;
  xpu3::plugin::speculate_schedule_cache<<<1, 64, ctx->xpu_stream>>>(
      (const XPU_TI *)draft_tokens,
      block_tables,
      stop_flags,
      (const XPU_TI *)prompt_lens,
      seq_lens_this_time,
      seq_lens_encoder,
      seq_lens_decoder,
      step_seq_lens_decoder,
      reinterpret_cast<XPU_TI *>(step_draft_tokens),
      step_seq_lens_this_time,
      accept_num,
      reinterpret_cast<XPU_TI *>(accept_tokens),
      is_block_step,
      not_need_stop,
      (const XPU_TI *)stop_nums,
      real_bsz,
      max_bsz,
      max_next_step_tokens,
      draft_tokens_len,
      accept_tokens_len,
      block_size,
      block_num_per_seq,
      prefill_one_step_stop);
  return api::SUCCESS;
}

int speculate_schedule_cache(Context *ctx,
                             const int64_t *draft_tokens,
                             int *block_tables,
                             bool *stop_flags,
                             const int64_t *prompt_lens,
                             int *seq_lens_this_time,
                             int *seq_lens_encoder,
                             int *seq_lens_decoder,
                             int *step_seq_lens_decoder,
                             int64_t *step_draft_tokens,
                             int *step_seq_lens_this_time,
                             int *accept_num,
                             int64_t *accept_tokens,
                             bool *is_block_step,
                             bool *not_need_stop,
                             const int64_t *stop_nums,
                             const int real_bsz,
                             const int max_bsz,
                             const int max_next_step_tokens,
                             const int draft_tokens_len,
                             const int accept_tokens_len,
                             const int block_size,
                             const int block_num_per_seq,
                             const bool prefill_one_step_stop) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_schedule_cache", float);
  WRAPPER_DUMP_PARAM6(ctx,
                      draft_tokens,
                      block_tables,
                      stop_flags,
                      prompt_lens,
                      seq_lens_this_time,
                      seq_lens_encoder);
  WRAPPER_DUMP_PARAM6(ctx,
                      seq_lens_decoder,
                      step_seq_lens_decoder,
                      step_draft_tokens,
                      step_seq_lens_this_time,
                      accept_num,
                      accept_tokens);
  WRAPPER_DUMP_PARAM6(ctx,
                      is_block_step,
                      not_need_stop,
                      stop_nums,
                      real_bsz,
                      max_bsz,
                      max_next_step_tokens);
  WRAPPER_DUMP_PARAM5(ctx,
                      draft_tokens_len,
                      accept_tokens_len,
                      block_size,
                      block_num_per_seq,
                      prefill_one_step_stop);

  WRAPPER_ASSERT_GT(ctx, draft_tokens_len, 0);
  WRAPPER_ASSERT_GT(ctx, accept_tokens_len, 0);
  WRAPPER_ASSERT_GT(ctx, block_num_per_seq, 0);
  WRAPPER_ASSERT_GT(ctx, real_bsz, 0);
  WRAPPER_ASSERT_GT(ctx, block_size, 0);
  WRAPPER_ASSERT_GT(ctx, max_next_step_tokens, 0);
  WRAPPER_ASSERT_GE(ctx, max_bsz, real_bsz);
  WRAPPER_CHECK_PTR(ctx, int64_t, draft_tokens_len * real_bsz, draft_tokens);
  WRAPPER_CHECK_PTR(
      ctx, int64_t, draft_tokens_len * real_bsz, step_draft_tokens);
  WRAPPER_CHECK_PTR(ctx, int64_t, accept_tokens_len * real_bsz, accept_tokens);
  WRAPPER_CHECK_PTR(ctx, int, block_num_per_seq *real_bsz, block_tables);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_decoder);
  WRAPPER_CHECK_PTR(ctx, int64_t, real_bsz, prompt_lens);
  WRAPPER_CHECK_PTR(ctx, bool, real_bsz, stop_flags);
  WRAPPER_DUMP(ctx);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       draft_tokens,
                       block_tables,
                       stop_flags,
                       prompt_lens,
                       seq_lens_this_time,
                       seq_lens_encoder,
                       seq_lens_decoder,
                       step_seq_lens_decoder,
                       step_draft_tokens,
                       step_seq_lens_this_time,
                       accept_num,
                       accept_tokens,
                       is_block_step,
                       not_need_stop,
                       stop_nums,
                       real_bsz,
                       max_bsz,
                       max_next_step_tokens,
                       draft_tokens_len,
                       accept_tokens_len,
                       block_size,
                       block_num_per_seq,
                       prefill_one_step_stop);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        draft_tokens,
                        block_tables,
                        stop_flags,
                        prompt_lens,
                        seq_lens_this_time,
                        seq_lens_encoder,
                        seq_lens_decoder,
                        step_seq_lens_decoder,
                        step_draft_tokens,
                        step_seq_lens_this_time,
                        accept_num,
                        accept_tokens,
                        is_block_step,
                        not_need_stop,
                        stop_nums,
                        real_bsz,
                        max_bsz,
                        max_next_step_tokens,
                        draft_tokens_len,
                        accept_tokens_len,
                        block_size,
                        block_num_per_seq,
                        prefill_one_step_stop);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
