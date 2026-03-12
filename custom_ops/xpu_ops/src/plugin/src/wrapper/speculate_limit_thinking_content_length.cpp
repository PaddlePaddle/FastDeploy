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
#include <numeric>
#include "xpu/plugin.h"
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace xpu3 {
namespace plugin {

__attribute__((global)) void speculate_limit_thinking_content_length_kernel(
    int64_t* next_tokens,
    const int* max_think_lens,
    int* max_reply_lens,
    int64_t* step_idx,
    const int64_t* eos_token_ids,
    int* limit_status,
    int* accept_num,
    const bool* stop_flags,
    const int64_t think_end_id,
    const int64_t* inject_token_ids,
    const int tokens_per_step,
    const int bs,
    const int eos_token_id_len,
    const int inject_len,
    const bool splitwise_role_is_decode);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context* ctx,
                       int64_t* next_tokens,
                       const int* max_think_lens,
                       int* max_reply_lens,
                       int64_t* step_idx,
                       const int64_t* eos_token_ids,
                       int* limit_status,
                       int* accept_num,
                       const bool* stop_flags,
                       const int64_t think_end_id,
                       const int64_t* inject_token_ids,
                       const int tokens_per_step,
                       const int bs,
                       const int eos_token_id_len,
                       const int inject_len,
                       const bool splitwise_role_is_decode) {
  for (int bid = 0; bid < bs; bid++) {
    const int original_accept_num = accept_num[bid];
    if (original_accept_num <= 0) continue;
    if (stop_flags[bid]) continue;

    const int max_think_len = max_think_lens[bid];
    int max_reply_len = max_reply_lens[bid];
    if (max_think_len < 0 && max_reply_len < 0) continue;

    const int done_status = (inject_len > 0) ? (inject_len + 1) : 1;
    const int reply_base = done_status + 1;

    int status = limit_status[bid];
    if (status < 0) status = 0;

    int new_accept_num = original_accept_num;

    const int64_t current_base_step = step_idx[bid] - original_accept_num + 1;

    for (int token_offset = 0; token_offset < original_accept_num;
         token_offset++) {
      const int token_idx = bid * tokens_per_step + token_offset;
      int64_t next_token = next_tokens[token_idx];
      const int64_t current_step = current_base_step + token_offset;

      const int prev_status = status;
      bool condition_triggered = false;

      if (status == 0 && next_token == think_end_id) {
        status = done_status;
        if (max_reply_len >= 0) {
          max_reply_len += 2;
        }
      }

      if (max_think_len >= 0 && status < reply_base) {
        if (max_think_len > 0) {
          if (status == 0 && (current_step - 1) == max_think_len) {
            status = (inject_len > 0) ? 1 : done_status;
          }
        } else if (max_think_len == 0) {
          if (status == 0 && !splitwise_role_is_decode) {
            status = (inject_len > 0) ? 1 : done_status;
          } else if (status == 0 && splitwise_role_is_decode) {
            status = (inject_len > 0) ? 2 : done_status + 1;
          }
        }

        if (status == 0 && inject_len > 0) {
          for (int i = 0; i < eos_token_id_len; i++) {
            if (eos_token_ids[i] == next_token) {
              status = 1;
              break;
            }
          }
        }

        if (inject_len > 0 && status >= 1 && status <= inject_len) {
          next_token = inject_token_ids[status - 1];
          status += 1;
          if (status > done_status) status = done_status;
          condition_triggered = true;
        }
      }

      const bool became_done_this_token = (status == done_status) &&
                                          (prev_status != done_status) &&
                                          (prev_status < reply_base);

      if (max_reply_len >= 0) {
        if (!became_done_this_token) {
          if (status == done_status) {
            status = reply_base;
          }

          if (status >= reply_base) {
            int reply_len = status - reply_base;

            if (reply_len >= max_reply_len) {
              if (eos_token_id_len > 0) next_token = eos_token_ids[0];
              status = reply_base + max_reply_len;
              condition_triggered = true;
            } else {
              status = reply_base + (reply_len + 1);
            }
          }
        }
      }

      next_tokens[token_idx] = next_token;

      if (condition_triggered) {
        new_accept_num = token_offset + 1;
        break;
      }
    }

    const int discarded_tokens = original_accept_num - new_accept_num;
    if (discarded_tokens > 0) {
      step_idx[bid] -= discarded_tokens;
    }

    accept_num[bid] = new_accept_num;
    limit_status[bid] = status;
    max_reply_lens[bid] = max_reply_len;
  }
  return api::SUCCESS;
}

static int xpu3_wrapper(Context* ctx,
                        int64_t* next_tokens,
                        const int* max_think_lens,
                        int* max_reply_lens,
                        int64_t* step_idx,
                        const int64_t* eos_token_ids,
                        int* limit_status,
                        int* accept_num,
                        const bool* stop_flags,
                        const int64_t think_end_id,
                        const int64_t* inject_token_ids,
                        const int tokens_per_step,
                        const int bs,
                        const int eos_token_id_len,
                        const int inject_len,
                        const bool splitwise_role_is_decode) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  auto kernel = xpu3::plugin::speculate_limit_thinking_content_length_kernel;
  int32_t ret_xre = kernel<<<1, 64, ctx->xpu_stream>>>(
      reinterpret_cast<XPU_INT64*>(next_tokens),
      max_think_lens,
      max_reply_lens,
      reinterpret_cast<XPU_INT64*>(step_idx),
      reinterpret_cast<const XPU_INT64*>(eos_token_ids),
      limit_status,
      accept_num,
      stop_flags,
      think_end_id,
      reinterpret_cast<const XPU_INT64*>(inject_token_ids),
      tokens_per_step,
      bs,
      eos_token_id_len,
      inject_len,
      splitwise_role_is_decode);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int speculate_limit_thinking_content_length_kernel(
    Context* ctx,
    int64_t* next_tokens,
    const int* max_think_lens,
    int* max_reply_lens,
    int64_t* step_idx,
    const int64_t* eos_token_ids,
    int* limit_status,
    int* accept_num,
    const bool* stop_flags,
    const int64_t think_end_id,
    const int64_t* inject_token_ids,
    const int tokens_per_step,
    const int bs,
    const int eos_token_id_len,
    const int inject_len,
    const bool splitwise_role_is_decode) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(
      ctx, "speculate_limit_thinking_content_length_kernel", int);
  WRAPPER_DUMP_PARAM5(ctx,
                      next_tokens,
                      max_think_lens,
                      max_reply_lens,
                      step_idx,
                      eos_token_ids);
  WRAPPER_DUMP_PARAM5(ctx,
                      limit_status,
                      accept_num,
                      stop_flags,
                      think_end_id,
                      inject_token_ids);
  WRAPPER_DUMP_PARAM5(ctx,
                      tokens_per_step,
                      bs,
                      eos_token_id_len,
                      inject_len,
                      splitwise_role_is_decode);
  WRAPPER_DUMP(ctx);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       next_tokens,
                       max_think_lens,
                       max_reply_lens,
                       step_idx,
                       eos_token_ids,
                       limit_status,
                       accept_num,
                       stop_flags,
                       think_end_id,
                       inject_token_ids,
                       tokens_per_step,
                       bs,
                       eos_token_id_len,
                       inject_len,
                       splitwise_role_is_decode);
  }
  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        next_tokens,
                        max_think_lens,
                        max_reply_lens,
                        step_idx,
                        eos_token_ids,
                        limit_status,
                        accept_num,
                        stop_flags,
                        think_end_id,
                        inject_token_ids,
                        tokens_per_step,
                        bs,
                        eos_token_id_len,
                        inject_len,
                        splitwise_role_is_decode);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
