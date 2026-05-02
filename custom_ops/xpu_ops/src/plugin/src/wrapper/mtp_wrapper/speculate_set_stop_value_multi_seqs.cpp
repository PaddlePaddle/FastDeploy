// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
__attribute__((global)) void speculate_set_stop_value_multi_seqs(
    bool* stop_flags,
    int64_t* accept_tokens,
    int* accept_nums,
    const int64_t* token_ids_all,
    const int64_t* prompt_lens,
    const int64_t* step_idx,
    const int64_t* stop_seqs,
    const int* stop_seqs_len,
    const int* seq_lens,
    const int64_t* end_ids,
    const int64_t* min_tokens,
    const int bs,
    const int accept_tokens_len,
    const int stop_seqs_bs,
    const int stop_seqs_max_len,
    const int max_model_len);
}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

static int cpu_wrapper(api::Context* ctx,
                       bool* stop_flags,
                       int64_t* accept_tokens,
                       int* accept_nums,
                       const int64_t* token_ids_all,
                       const int64_t* prompt_lens,
                       const int64_t* step_idx,
                       const int64_t* stop_seqs,
                       const int* stop_seqs_len,
                       const int* seq_lens,
                       const int64_t* end_ids,
                       const int64_t* min_tokens,
                       const int bs,
                       const int accept_tokens_len,
                       const int stop_seqs_bs,
                       const int stop_seqs_max_len,
                       const int max_model_len) {
  for (int bid = 0; bid < bs; ++bid) {
    // Align with GPU: pre_ids_now = token_ids_all + bid * max_model_len +
    // prompt_lens[bid]
    const int64_t* pre_ids_now =
        token_ids_all + bid * max_model_len + prompt_lens[bid];
    int64_t* accept_tokens_now = accept_tokens + bid * accept_tokens_len;
    const int accept_num = accept_nums[bid];
    const int64_t step_idx_now = step_idx[bid];
    const int64_t min_token_limit = min_tokens[bid];

    // Align with GPU: can_stop = (step_idx_now + accept_num >= min_token_limit)
    const bool can_stop = (step_idx_now + accept_num >= min_token_limit);
    if (!can_stop) continue;
    if (stop_flags[bid]) continue;
    for (int tid = 0; tid < stop_seqs_bs; ++tid) {
      // Align with GPU: per-batch stop_seqs_len
      const int stop_seq_len = stop_seqs_len[bid * stop_seqs_bs + tid];
      if (stop_seq_len <= 0) continue;
      // Align with GPU: per-batch stop_seqs
      const int64_t* stop_seq_now = stop_seqs +
                                    bid * stop_seqs_max_len * stop_seqs_bs +
                                    tid * stop_seqs_max_len;

      /*
        Align with GPU:
        accept_idx = -1 means the last token of stop_seq is at the end
        of pre_ids (delayed match from the previous round).
        Loop range: when accept_num > 0, [-1, accept_num-2];
                    when accept_num = 0, [-1].
      */
      int accept_idx = -1;
      bool is_end = false;

      int loop_end = (accept_num > 0) ? accept_num - 2 : -1;
      for (; accept_idx <= loop_end && !is_end; accept_idx++) {
        if (step_idx_now + accept_idx + 1 < stop_seq_len) {
          continue;
        }
        for (int i = stop_seq_len - 1; i >= 0; --i) {
          int64_t cur_token_idx = -1;

          int offset = stop_seq_len - 1 - i;
          int accept_tokens_idx = accept_idx - offset;

          if (accept_tokens_idx >= 0) {
            cur_token_idx = accept_tokens_now[accept_tokens_idx];
          } else {
            int pre_ids_idx = step_idx_now + accept_tokens_idx;
            // Align with GPU: use < 0 instead of <= 0
            if (pre_ids_idx < 0) {
              break;
            }
            cur_token_idx = pre_ids_now[pre_ids_idx];
          }
          if (cur_token_idx != stop_seq_now[i]) {
            break;
          }
          if (i == 0) {
            is_end = true;
          }
        }
      }
      if (is_end) {
        // Align with GPU: truncate accept_nums, write eos at accept_idx,
        // do NOT set stop_flags here.
        accept_nums[bid] = accept_idx + 1;
        accept_tokens_now[accept_idx] = end_ids[0];
      }
    }
  }

  return api::SUCCESS;
}

static int xpu3_wrapper(api::Context* ctx,
                        bool* stop_flags,
                        int64_t* accept_tokens,
                        int* accept_nums,
                        const int64_t* token_ids_all,
                        const int64_t* prompt_lens,
                        const int64_t* step_idx,
                        const int64_t* stop_seqs,
                        const int* stop_seqs_len,
                        const int* seq_lens,
                        const int64_t* end_ids,
                        const int64_t* min_tokens,
                        const int bs,
                        const int accept_tokens_len,
                        const int stop_seqs_bs,
                        const int stop_seqs_max_len,
                        const int max_model_len) {
  using XPU_INT64 = typename api::XPUIndexType<int64_t>::type;
  int32_t ret_xre =
      fd_xpu3::speculate_set_stop_value_multi_seqs<<<1, 64, ctx->xpu_stream>>>(
          stop_flags,
          reinterpret_cast<XPU_INT64*>(accept_tokens),
          accept_nums,
          reinterpret_cast<const XPU_INT64*>(token_ids_all),
          reinterpret_cast<const XPU_INT64*>(prompt_lens),
          reinterpret_cast<const XPU_INT64*>(step_idx),
          reinterpret_cast<const XPU_INT64*>(stop_seqs),
          stop_seqs_len,
          seq_lens,
          reinterpret_cast<const XPU_INT64*>(end_ids),
          reinterpret_cast<const XPU_INT64*>(min_tokens),
          bs,
          accept_tokens_len,
          stop_seqs_bs,
          stop_seqs_max_len,
          max_model_len);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int speculate_set_stop_value_multi_seqs(api::Context* ctx,
                                        bool* stop_flags,
                                        int64_t* accept_tokens,
                                        int* accept_nums,
                                        const int64_t* token_ids_all,
                                        const int64_t* prompt_lens,
                                        const int64_t* step_idx,
                                        const int64_t* stop_seqs,
                                        const int* stop_seqs_len,
                                        const int* seq_lens,
                                        const int64_t* end_ids,
                                        const int64_t* min_tokens,
                                        const int bs_now,
                                        const int accept_tokens_len,
                                        const int stop_seqs_bs,
                                        const int stop_seqs_max_len,
                                        const int max_model_len) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_set_stop_value_multi_seqs", int64_t);
  WRAPPER_DUMP_PARAM3(ctx, stop_flags, accept_tokens, accept_nums);
  WRAPPER_DUMP_PARAM6(ctx,
                      token_ids_all,
                      prompt_lens,
                      step_idx,
                      stop_seqs,
                      stop_seqs_len,
                      seq_lens);
  WRAPPER_DUMP_PARAM2(ctx, end_ids, min_tokens);
  WRAPPER_DUMP_PARAM5(ctx,
                      bs_now,
                      accept_tokens_len,
                      stop_seqs_bs,
                      stop_seqs_max_len,
                      max_model_len);
  WRAPPER_DUMP(ctx);
  WRAPPER_CHECK_PTR(ctx, int64_t, bs_now * accept_tokens_len, accept_tokens);
  WRAPPER_CHECK_PTR(
      ctx, int64_t, bs_now * stop_seqs_bs * stop_seqs_max_len, stop_seqs);
  WRAPPER_ASSERT_GT(ctx, bs_now, 0);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       stop_flags,
                       accept_tokens,
                       accept_nums,
                       token_ids_all,
                       prompt_lens,
                       step_idx,
                       stop_seqs,
                       stop_seqs_len,
                       seq_lens,
                       end_ids,
                       min_tokens,
                       bs_now,
                       accept_tokens_len,
                       stop_seqs_bs,
                       stop_seqs_max_len,
                       max_model_len);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        stop_flags,
                        accept_tokens,
                        accept_nums,
                        token_ids_all,
                        prompt_lens,
                        step_idx,
                        stop_seqs,
                        stop_seqs_len,
                        seq_lens,
                        end_ids,
                        min_tokens,
                        bs_now,
                        accept_tokens_len,
                        stop_seqs_bs,
                        stop_seqs_max_len,
                        max_model_len);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace fastdeploy
