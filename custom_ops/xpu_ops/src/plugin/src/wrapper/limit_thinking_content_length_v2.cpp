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

__attribute__((global)) void limit_thinking_content_length_kernel_v2(
    int64_t* next_tokens,
    const int* max_think_lens,
    const int64_t* step_idx,
    int* limit_think_status,
    const bool* stop_flags,
    const int64_t think_end_id,
    const int64_t line_break_id,
    const int bs);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context* ctx,
                       int64_t* next_tokens,
                       const int* max_think_lens,
                       const int64_t* step_idx,
                       int* limit_think_status,
                       const bool* stop_flags,
                       const int64_t think_end_id,
                       const int64_t line_break_id,
                       const int bs) {
  for (int bid = 0; bid < bs; bid++) {
    const int max_think_len = max_think_lens[bid];
    if (max_think_len < 0) continue;
    int current_limit_think_status = limit_think_status[bid];
    if (current_limit_think_status == 3 && stop_flags[bid]) {
      continue;
    }

    int64_t next_token = next_tokens[bid];
    const int64_t step = step_idx[bid];

    if (current_limit_think_status <= 1) {
      if (step == max_think_len) {
        next_token = line_break_id;
        current_limit_think_status = 1;
      } else if (step == max_think_len + 1) {
        next_token = think_end_id;
        current_limit_think_status = 1;
      } else if (step == max_think_len + 2) {
        next_token = line_break_id;
        current_limit_think_status = 1;
      } else if (step == max_think_len + 3) {
        next_token = line_break_id;
        current_limit_think_status = 2;
      }
    }
    if (current_limit_think_status == 0) {
      if (next_token == think_end_id) {
        current_limit_think_status = 3;
      }
    }
    if (current_limit_think_status == 2) {
      current_limit_think_status = 3;
    }
    next_tokens[bid] = next_token;
    limit_think_status[bid] = current_limit_think_status;
  }
  return api::SUCCESS;
}
static int xpu3_wrapper(Context* ctx,
                        int64_t* next_tokens,
                        const int* max_think_lens,
                        const int64_t* step_idx,
                        int* limit_think_status,
                        const bool* stop_flags,
                        const int64_t think_end_id,
                        const int64_t line_break_id,
                        const int bs) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  auto limit_thinking_content_length_kernel_v2 =
      xpu3::plugin::limit_thinking_content_length_kernel_v2;
  limit_thinking_content_length_kernel_v2<<<1, 64, ctx->xpu_stream>>>(
      reinterpret_cast<XPU_INT64*>(next_tokens),
      max_think_lens,
      reinterpret_cast<const XPU_INT64*>(step_idx),
      limit_think_status,
      stop_flags,
      think_end_id,
      line_break_id,
      bs);
  return api::SUCCESS;
}

int limit_thinking_content_length_kernel_v2(Context* ctx,
                                            int64_t* next_tokens,
                                            const int* max_think_lens,
                                            const int64_t* step_idx,
                                            int* limit_think_status,
                                            const bool* stop_flags,
                                            const int64_t think_end_id,
                                            const int64_t line_break_id,
                                            const int bs) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "limit_thinking_content_length_kernel_v2", int);
  WRAPPER_DUMP_PARAM5(ctx,
                      next_tokens,
                      max_think_lens,
                      step_idx,
                      limit_think_status,
                      stop_flags);
  WRAPPER_DUMP_PARAM3(ctx, think_end_id, line_break_id, bs);
  WRAPPER_DUMP(ctx);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       next_tokens,
                       max_think_lens,
                       step_idx,
                       limit_think_status,
                       stop_flags,
                       think_end_id,
                       line_break_id,
                       bs);
  }
  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        next_tokens,
                        max_think_lens,
                        step_idx,
                        limit_think_status,
                        stop_flags,
                        think_end_id,
                        line_break_id,
                        bs);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
