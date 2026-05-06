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

namespace fd_xpu3 {

__attribute__((global)) void limit_thinking_content_length_kernel(
    int64_t* next_tokens,
    const int* max_think_lens,
    int* max_reply_lens,
    const int64_t* step_idx,
    const int64_t* eos_token_ids,
    int* limit_status,
    const bool* stop_flags,
    const int64_t think_end_id,
    const int64_t* inject_token_ids,
    const int bs,
    const int eos_token_id_len,
    const int inject_len,
    const bool splitwise_role_is_decode);

}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

static int cpu_wrapper(api::Context* ctx,
                       int64_t* next_tokens,
                       const int* max_think_lens,
                       int* max_reply_lens,
                       const int64_t* step_idx,
                       const int64_t* eos_token_ids,
                       int* limit_status,
                       const bool* stop_flags,
                       const int64_t think_end_id,
                       const int64_t* inject_token_ids,
                       const int bs,
                       const int eos_token_id_len,
                       const int inject_len,
                       const bool splitwise_role_is_decode) {
  auto is_in_end = [](int64_t token_id, const int64_t* end_ids, int length) {
    for (int i = 0; i < length; i++) {
      if (token_id == end_ids[i]) {
        return true;
      }
    }
    return false;
  };
  for (int bid = 0; bid < bs; bid++) {
    if (stop_flags[bid]) continue;

    const int max_think_len = max_think_lens[bid];
    int max_reply_len = max_reply_lens[bid];
    if (max_think_len < 0 && max_reply_len < 0) continue;

    const int done_status = (inject_len > 0) ? (inject_len + 1) : 1;
    const int reply_base = done_status + 1;

    int status = limit_status[bid];
    if (status < 0) status = 0;
    const int prev_status = status;

    int64_t next_token = next_tokens[bid];
    const int64_t step = step_idx[bid];

    // ======================= 1) Think phase: always monitor think_end_id
    // =======================
    if (status == 0 && next_token == think_end_id) {
      status = done_status;
      if (max_reply_len >= 0) {
        max_reply_len += 2;
      }
    }

    // ======================= 2) Only when thinking truncation is enabled
    // (max_think_len >= 0) =======================
    if (max_think_len >= 0 && status < reply_base) {
      // A) Timeout trigger: start injection when reaching max_think_len
      if (max_think_len > 0) {
        if (status == 0 && step == max_think_len) {
          status = (inject_len > 0) ? 1 : done_status;
        }
      } else if (max_think_len == 0) {
        if (status == 0 && !splitwise_role_is_decode) {
          status = (inject_len > 0) ? 1 : done_status;
        } else if (status == 0 && splitwise_role_is_decode) {
          status = (inject_len > 0) ? 2 : done_status + 1;
        }
      }

      // B) Early EOS in thinking phase: start injection (override eos)
      if (status == 0 && inject_len > 0) {
        if (is_in_end(next_token, eos_token_ids, eos_token_id_len)) {
          status = 1;
        }
      }

      // Injection sequence
      if (inject_len > 0 && status >= 1 && status <= inject_len) {
        next_token = inject_token_ids[status - 1];
        status += 1;
        if (status > done_status) status = done_status;
      }
    }

    // Whether this step "just entered done_status"
    const bool became_done_this_step = (status == done_status) &&
                                       (prev_status != done_status) &&
                                       (prev_status < reply_base);

    // ======================= 3) Reply length limiting
    // =======================
    if (max_reply_len >= 0) {
      if (!became_done_this_step) {
        if (status == done_status) {
          status = reply_base;
        }

        if (status >= reply_base) {
          int reply_len = status - reply_base;

          if (reply_len >= max_reply_len) {
            if (eos_token_id_len > 0) next_token = eos_token_ids[0];
            status = reply_base + max_reply_len;
          } else {
            status = reply_base + (reply_len + 1);
          }
        }
      }
    }

    next_tokens[bid] = next_token;
    limit_status[bid] = status;
    max_reply_lens[bid] = max_reply_len;
  }
  return api::SUCCESS;
}

static int xpu3_wrapper(api::Context* ctx,
                        int64_t* next_tokens,
                        const int* max_think_lens,
                        int* max_reply_lens,
                        const int64_t* step_idx,
                        const int64_t* eos_token_ids,
                        int* limit_status,
                        const bool* stop_flags,
                        const int64_t think_end_id,
                        const int64_t* inject_token_ids,
                        const int bs,
                        const int eos_token_id_len,
                        const int inject_len,
                        const bool splitwise_role_is_decode) {
  using XPU_INT64 = typename api::XPUIndexType<int64_t>::type;
  auto kernel = fd_xpu3::limit_thinking_content_length_kernel;
  int32_t ret_xre = kernel<<<1, 64, ctx->xpu_stream>>>(
      reinterpret_cast<XPU_INT64*>(next_tokens),
      max_think_lens,
      max_reply_lens,
      reinterpret_cast<const XPU_INT64*>(step_idx),
      reinterpret_cast<const XPU_INT64*>(eos_token_ids),
      limit_status,
      stop_flags,
      think_end_id,
      reinterpret_cast<const XPU_INT64*>(inject_token_ids),
      bs,
      eos_token_id_len,
      inject_len,
      splitwise_role_is_decode);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int limit_thinking_content_length_kernel(api::Context* ctx,
                                         int64_t* next_tokens,
                                         const int* max_think_lens,
                                         int* max_reply_lens,
                                         const int64_t* step_idx,
                                         const int64_t* eos_token_ids,
                                         int* limit_status,
                                         const bool* stop_flags,
                                         const int64_t think_end_id,
                                         const int64_t* inject_token_ids,
                                         const int bs,
                                         const int eos_token_id_len,
                                         const int inject_len,
                                         const bool splitwise_role_is_decode) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "limit_thinking_content_length_kernel", int);
  WRAPPER_DUMP_PARAM5(ctx,
                      next_tokens,
                      max_think_lens,
                      max_reply_lens,
                      step_idx,
                      eos_token_ids);
  WRAPPER_DUMP_PARAM5(
      ctx, limit_status, stop_flags, think_end_id, inject_token_ids, bs);
  WRAPPER_DUMP_PARAM3(
      ctx, eos_token_id_len, inject_len, splitwise_role_is_decode);
  WRAPPER_DUMP(ctx);
  WRAPPER_ASSERT_GT(ctx, bs, 0);
  WRAPPER_ASSERT_GT(ctx, eos_token_id_len, 0);
  WRAPPER_ASSERT_GE(ctx, inject_len, 0);
  WRAPPER_ASSERT_LE(ctx, eos_token_id_len, 64);
  WRAPPER_CHECK_PTR(ctx, int64_t, bs, next_tokens);
  WRAPPER_CHECK_PTR(ctx, int, bs, max_think_lens);
  WRAPPER_CHECK_PTR(ctx, int, bs, max_reply_lens);
  WRAPPER_CHECK_PTR(ctx, int64_t, bs, step_idx);
  WRAPPER_CHECK_PTR(ctx, int64_t, eos_token_id_len, eos_token_ids);
  WRAPPER_CHECK_PTR(ctx, int, bs, limit_status);
  WRAPPER_CHECK_PTR(ctx, bool, bs, stop_flags);
  if (inject_len > 0) {
    WRAPPER_CHECK_PTR(ctx, int64_t, inject_len, inject_token_ids);
  }
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       next_tokens,
                       max_think_lens,
                       max_reply_lens,
                       step_idx,
                       eos_token_ids,
                       limit_status,
                       stop_flags,
                       think_end_id,
                       inject_token_ids,
                       bs,
                       eos_token_id_len,
                       inject_len,
                       splitwise_role_is_decode);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        next_tokens,
                        max_think_lens,
                        max_reply_lens,
                        step_idx,
                        eos_token_ids,
                        limit_status,
                        stop_flags,
                        think_end_id,
                        inject_token_ids,
                        bs,
                        eos_token_id_len,
                        inject_len,
                        splitwise_role_is_decode);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace fastdeploy
