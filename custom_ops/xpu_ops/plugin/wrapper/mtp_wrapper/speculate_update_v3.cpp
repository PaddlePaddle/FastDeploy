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

#include "xpu/plugin.h"
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace xpu3 {
namespace plugin {
template <int THREADBLOCK_SIZE>
__attribute__((global)) void speculate_update_v3(
    int *seq_lens_encoder,          // 输入 [B_max, ]
    int *seq_lens_decoder,          // 输出 [B_max, ]
    bool *not_need_stop,            // 输出 [1,]
    int64_t *draft_tokens,          // 输出 [B_max, T_max]
    int *actual_draft_token_nums,   // 输出 [B_max, ]
    const int64_t *accept_tokens,   // 输入 [B_max, T_max]
    const int *accept_num,          // 输入 [B_max, ]
    const bool *stop_flags,         // 输入 [B_max, ]
    const int *seq_lens_this_time,  // 输入 [B_real,]
    const bool *is_block_step,      // 输入 [B_max, ]
    const int64_t *stop_nums,       // 输入 [1,]
    const int real_bsz,
    const int max_bsz,
    const int max_draft_tokens);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context *ctx,
                       int *seq_lens_encoder,          // 输入 [B_max, ]
                       int *seq_lens_decoder,          // 输出 [B_max, ]
                       bool *not_need_stop,            // [1,]
                       int64_t *draft_tokens,          // [B_max, T_max]
                       int *actual_draft_token_nums,   // [B_max, ]
                       const int64_t *accept_tokens,   // [B_max, T_max]
                       const int *accept_num,          // [B_max, ]
                       const bool *stop_flags,         // [B_max, ]
                       const int *seq_lens_this_time,  // [B_real,]
                       const bool *is_block_step,      // [B_max, ]
                       const int64_t *stop_nums,       // [1,]
                       const int real_bsz,
                       const int max_bsz,
                       const int max_draft_tokens) {
  int64_t stop_sum = 0;

  for (int bid = 0; bid < max_bsz; ++bid) {
    int stop_flag_now_int = 0;

    const bool inactive = (bid >= real_bsz);
    const bool block_step = (!inactive && is_block_step[bid]);

    if (!block_step && !inactive) {
      // 1. 本样本是否已触发 stop
      if (stop_flags[bid]) stop_flag_now_int = 1;

      // 2. encoder len == 0 时可直接累加 decoder
      if (seq_lens_encoder[bid] == 0) {
        seq_lens_decoder[bid] += accept_num[bid];
      }

      // 3. 根据「是否全部接受」动态调整 draft 长度
      if (seq_lens_encoder[bid] == 0 &&  // append-mode 才走
          seq_lens_this_time[bid] > 1) {
        int cur_len = actual_draft_token_nums[bid];

        if (accept_num[bid] - 1 == cur_len) {
          // 全部接受：尝试 +2 / +1
          if (cur_len + 2 <= max_draft_tokens - 1)
            cur_len += 2;
          else if (cur_len + 1 <= max_draft_tokens - 1)
            cur_len += 1;
          else
            cur_len = max_draft_tokens - 1;
        } else {
          // 有拒绝：-1，最小 1
          cur_len = std::max(1, cur_len - 1);
        }
        actual_draft_token_nums[bid] = cur_len;
      }

      // 4. 偿还 encoder 欠账
      if (seq_lens_encoder[bid] != 0) {
        seq_lens_decoder[bid] += seq_lens_encoder[bid];
        const_cast<int *>(seq_lens_encoder)[bid] = 0;  // cast 因原指针是 const
      }

      // 6. 如果 stop，decoder 长度清零
      if (stop_flag_now_int) {
        seq_lens_decoder[bid] = 0;
      } else {
        // 5. 写回下一轮首 token，但理论上只需要更新有效draft即可
        draft_tokens[bid * max_draft_tokens] =
            accept_tokens[bid * max_draft_tokens + accept_num[bid] - 1];
      }

    } else if (inactive) {
      // padding slot：直接当作 stop
      stop_flag_now_int = 1;
    }

    stop_sum += stop_flag_now_int;
  }

  // 7. 写出全局标志
  not_need_stop[0] = (stop_sum < stop_nums[0]);

  return api::SUCCESS;
}

static int xpu3_wrapper(Context *ctx,
                        int *seq_lens_encoder,          // 输入 [B_max, ]
                        int *seq_lens_decoder,          // 输出 [B_max, ]
                        bool *not_need_stop,            // [1,]
                        int64_t *draft_tokens,          // [B_max, T_max]
                        int *actual_draft_token_nums,   // [B_max, ]
                        const int64_t *accept_tokens,   // [B_max, T_max]
                        const int *accept_num,          // [B_max, ]
                        const bool *stop_flags,         // [B_max, ]
                        const int *seq_lens_this_time,  // [B_real,]
                        const bool *is_block_step,      // [B_max, ]
                        const int64_t *stop_nums,       // [1,]
                        const int real_bsz,
                        const int max_bsz,
                        const int max_draft_tokens) {
  constexpr int BlockSize = 512;
  using XPU_TI = typename XPUIndexType<int64_t>::type;
  xpu3::plugin::speculate_update_v3<BlockSize>
      <<<1, 64, ctx->xpu_stream>>>(seq_lens_encoder,
                                   seq_lens_decoder,
                                   not_need_stop,
                                   reinterpret_cast<XPU_TI *>(draft_tokens),
                                   actual_draft_token_nums,
                                   (const XPU_TI *)accept_tokens,
                                   accept_num,
                                   stop_flags,
                                   seq_lens_this_time,
                                   is_block_step,
                                   (const XPU_TI *)stop_nums,
                                   real_bsz,
                                   max_bsz,
                                   max_draft_tokens);
  return api::SUCCESS;
}

int speculate_update_v3(Context *ctx,
                        int *seq_lens_encoder,          // 输入 [B_max, ]
                        int *seq_lens_decoder,          // 输出 [B_max, ]
                        bool *not_need_stop,            // [1,]
                        int64_t *draft_tokens,          // [B_max, T_max]
                        int *actual_draft_token_nums,   // [B_max, ]
                        const int64_t *accept_tokens,   // [B_max, T_max]
                        const int *accept_num,          // [B_max, ]
                        const bool *stop_flags,         // [B_max, ]
                        const int *seq_lens_this_time,  // [B_real,]
                        const bool *is_block_step,      // [B_max, ]
                        const int64_t *stop_nums,       // [1,]
                        const int real_bsz,
                        const int max_bsz,
                        const int max_draft_tokens) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_update_v3", int);
  WRAPPER_DUMP_PARAM4(
      ctx, seq_lens_encoder, seq_lens_decoder, not_need_stop, draft_tokens);
  WRAPPER_DUMP_PARAM4(
      ctx, actual_draft_token_nums, accept_tokens, accept_num, stop_flags);
  WRAPPER_DUMP_PARAM4(
      ctx, seq_lens_this_time, is_block_step, stop_nums, real_bsz);
  WRAPPER_DUMP_PARAM2(ctx, max_bsz, max_draft_tokens);
  WRAPPER_DUMP(ctx);
  WRAPPER_ASSERT_GT(ctx, real_bsz, 0);
  WRAPPER_ASSERT_GT(ctx, max_bsz, 0);
  WRAPPER_ASSERT_LE(ctx, max_bsz, 512);
  WRAPPER_ASSERT_GT(ctx, max_draft_tokens, 0);
  WRAPPER_ASSERT_GE(ctx, max_bsz, real_bsz);
  WRAPPER_CHECK_PTR(ctx, int, max_bsz, seq_lens_encoder);
  WRAPPER_CHECK_PTR(ctx, int, max_bsz, seq_lens_decoder);
  WRAPPER_CHECK_PTR(ctx, bool, 1, not_need_stop);
  WRAPPER_CHECK_PTR(ctx, int64_t, max_bsz * max_draft_tokens, draft_tokens);
  WRAPPER_CHECK_PTR(ctx, int, max_bsz, actual_draft_token_nums);
  WRAPPER_CHECK_PTR(ctx, int64_t, max_bsz * max_draft_tokens, accept_tokens);
  WRAPPER_CHECK_PTR(ctx, int, max_bsz, accept_num);
  WRAPPER_CHECK_PTR(ctx, bool, max_bsz, stop_flags);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, bool, max_bsz, is_block_step);
  WRAPPER_CHECK_PTR(ctx, int64_t, 1, stop_nums);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       seq_lens_encoder,
                       seq_lens_decoder,
                       not_need_stop,
                       draft_tokens,
                       actual_draft_token_nums,
                       accept_tokens,
                       accept_num,
                       stop_flags,
                       seq_lens_this_time,
                       is_block_step,
                       stop_nums,
                       real_bsz,
                       max_bsz,
                       max_draft_tokens);
  } else if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        seq_lens_encoder,
                        seq_lens_decoder,
                        not_need_stop,
                        draft_tokens,
                        actual_draft_token_nums,
                        accept_tokens,
                        accept_num,
                        stop_flags,
                        seq_lens_this_time,
                        is_block_step,
                        stop_nums,
                        real_bsz,
                        max_bsz,
                        max_draft_tokens);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
