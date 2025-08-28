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

namespace xpu3 {
namespace plugin {
__attribute__((global)) void speculate_set_stop_value_multi_seqs(
    bool* stop_flags,
    int64_t* accept_tokens,
    int* accept_nums,
    const int64_t* pre_ids,
    const int64_t* step_idx,
    const int64_t* stop_seqs,
    const int* stop_seqs_len,
    const int* seq_lens,
    const int64_t* end_ids,
    const int bs,
    const int accept_tokens_len,
    const int stop_seqs_bs,
    const int stop_seqs_max_len,
    const int pre_ids_len);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context* ctx,
                       bool* stop_flags,
                       int64_t* accept_tokens,
                       int* accept_nums,
                       const int64_t* pre_ids,
                       const int64_t* step_idx,
                       const int64_t* stop_seqs,
                       const int* stop_seqs_len,
                       const int* seq_lens,
                       const int64_t* end_ids,
                       const int bs,
                       const int accept_tokens_len,
                       const int stop_seqs_bs,
                       const int stop_seqs_max_len,
                       const int pre_ids_len) {
  for (int bid = 0; bid < bs; ++bid) {
    const int64_t* pre_ids_now = pre_ids + bid * pre_ids_len;
    int64_t* accept_tokens_now = accept_tokens + bid * accept_tokens_len;
    const int accept_num = accept_nums[bid];
    const int64_t step_idx_now = step_idx[bid];
    for (int tid = 0; tid < stop_seqs_bs; ++tid) {
      const int stop_seq_len = stop_seqs_len[tid];
      if (stop_seq_len <= 0) continue;
      const int64_t* stop_seq_now = stop_seqs + tid * stop_seqs_max_len;
      if (!stop_flags[bid]) {
        int accept_idx = 0;
        bool is_end = false;
        // 遍历起始位置
        for (; accept_idx <= accept_num - 1 && !is_end; accept_idx++) {
          if (step_idx_now - accept_num + accept_idx + 1 < stop_seq_len) {
            continue;
          }
          // 遍历一个 stop_seqs
          for (int i = stop_seq_len - 1; i >= 0; --i) {
            int64_t cur_token_idx = -1;

            // 通过当前值判断 token 是在 pre_ids 还是 accept_token 里
            if (stop_seq_len - 1 - i < accept_idx) {
              cur_token_idx =
                  accept_tokens_now[accept_idx - (stop_seq_len - 1 - i) - 1];
            } else {
              int pre_ids_idx = step_idx_now - accept_num + accept_idx -
                                (stop_seq_len - 1 - i);
              // EC3
              // 特殊拼接会导致input_ids最后一位无特殊token，即pre_ids[0]可能为23,
              // 导致异常结束
              if (pre_ids_idx <= 0) {
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
          accept_nums[bid] = accept_idx;
          accept_tokens_now[accept_idx - 1] = end_ids[0];
          stop_flags[bid] = true;
        }
      }
    }
  }

  return api::SUCCESS;
}

static int xpu2or3_wrapper(Context* ctx,
                           bool* stop_flags,
                           int64_t* accept_tokens,
                           int* accept_nums,
                           const int64_t* pre_ids,
                           const int64_t* step_idx,
                           const int64_t* stop_seqs,
                           const int* stop_seqs_len,
                           const int* seq_lens,
                           const int64_t* end_ids,
                           const int bs,
                           const int accept_tokens_len,
                           const int stop_seqs_bs,
                           const int stop_seqs_max_len,
                           const int pre_ids_len) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  xpu3::plugin::speculate_set_stop_value_multi_seqs<<<ctx->ncluster(),
                                                      64,
                                                      ctx->xpu_stream>>>(
      stop_flags,
      reinterpret_cast<XPU_INT64*>(accept_tokens),
      accept_nums,
      reinterpret_cast<const XPU_INT64*>(pre_ids),
      reinterpret_cast<const XPU_INT64*>(step_idx),
      reinterpret_cast<const XPU_INT64*>(stop_seqs),
      stop_seqs_len,
      seq_lens,
      reinterpret_cast<const XPU_INT64*>(end_ids),
      bs,
      accept_tokens_len,
      stop_seqs_bs,
      stop_seqs_max_len,
      pre_ids_len);
  return api::SUCCESS;
}

int speculate_set_stop_value_multi_seqs(Context* ctx,
                                        bool* stop_flags,
                                        int64_t* accept_tokens,
                                        int* accept_nums,
                                        const int64_t* pre_ids,
                                        const int64_t* step_idx,
                                        const int64_t* stop_seqs,
                                        const int* stop_seqs_len,
                                        const int* seq_lens,
                                        const int64_t* end_ids,
                                        const int bs_now,
                                        const int accept_tokens_len,
                                        const int stop_seqs_bs,
                                        const int stop_seqs_max_len,
                                        const int pre_ids_len) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_set_stop_value_multi_seqs", int64_t);
  WRAPPER_DUMP_PARAM3(ctx, stop_flags, accept_tokens, accept_nums);
  WRAPPER_DUMP_PARAM6(
      ctx, pre_ids, step_idx, stop_seqs, stop_seqs_len, seq_lens, end_ids);
  WRAPPER_DUMP_PARAM5(ctx,
                      bs_now,
                      accept_tokens_len,
                      stop_seqs_bs,
                      stop_seqs_max_len,
                      pre_ids_len);
  WRAPPER_DUMP(ctx);
  WRAPPER_CHECK_PTR(ctx, int64_t, bs_now * accept_tokens_len, accept_tokens);
  WRAPPER_CHECK_PTR(ctx, int64_t, stop_seqs_bs * stop_seqs_max_len, stop_seqs);
  WRAPPER_ASSERT_GT(ctx, bs_now, 0);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       stop_flags,
                       accept_tokens,
                       accept_nums,
                       pre_ids,
                       step_idx,
                       stop_seqs,
                       stop_seqs_len,
                       seq_lens,
                       end_ids,
                       bs_now,
                       accept_tokens_len,
                       stop_seqs_bs,
                       stop_seqs_max_len,
                       pre_ids_len);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu2or3_wrapper(ctx,
                           stop_flags,
                           accept_tokens,
                           accept_nums,
                           pre_ids,
                           step_idx,
                           stop_seqs,
                           stop_seqs_len,
                           seq_lens,
                           end_ids,
                           bs_now,
                           accept_tokens_len,
                           stop_seqs_bs,
                           stop_seqs_max_len,
                           pre_ids_len);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
