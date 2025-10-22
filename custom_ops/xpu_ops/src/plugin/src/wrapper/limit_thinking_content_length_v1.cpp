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

__attribute__((global)) void limit_thinking_content_length_kernel_v1(
    int64_t* next_tokens,
    const int* max_think_lens,
    const int64_t* step_idx,
    int* limit_think_status,
    const int64_t think_end_id,
    const int bs);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int xpu3_wrapper(Context* ctx,
                        int64_t* next_tokens,
                        const int* max_think_lens,
                        const int64_t* step_idx,
                        int* limit_think_status,
                        const int64_t think_end_id,
                        const int bs) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  auto limit_thinking_content_length_kernel_v1 =
      xpu3::plugin::limit_thinking_content_length_kernel_v1;
  limit_thinking_content_length_kernel_v1<<<1, 64, ctx->xpu_stream>>>(
      reinterpret_cast<XPU_INT64*>(next_tokens),
      max_think_lens,
      reinterpret_cast<const XPU_INT64*>(step_idx),
      limit_think_status,
      think_end_id,
      bs);
  return api::SUCCESS;
}

int limit_thinking_content_length_kernel_v1(Context* ctx,
                                            int64_t* next_tokens,
                                            const int* max_think_lens,
                                            const int64_t* step_idx,
                                            int* limit_think_status,
                                            const int64_t think_end_id,
                                            const int bs) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "limit_thinking_content_length_kernel_v1", int);
  WRAPPER_DUMP_PARAM5(ctx,
                      next_tokens,
                      max_think_lens,
                      step_idx,
                      limit_think_status,
                      think_end_id);
  WRAPPER_DUMP_PARAM1(ctx, bs);

  WRAPPER_DUMP(ctx);
  if (ctx->dev().type() == api::kCPU) {
    assert(false);
  }
  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        next_tokens,
                        max_think_lens,
                        step_idx,
                        limit_think_status,
                        think_end_id,
                        bs);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
