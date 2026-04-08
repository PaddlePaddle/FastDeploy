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
__attribute__((global)) void text_image_index_out_kernel(
    const int* token_type_ids,  // x
    int* text_index,            // y1
    int* image_index,           // y2
    const int64_t token_num);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context* ctx,
                       const int* token_type_ids,  // x
                       int* text_index,            // y1
                       int* image_index,           // y2
                       const int64_t token_num) {
  int text_count = 0;
  int image_count = 0;

  for (int64_t i = 0; i < token_num; ++i) {
    if (token_type_ids[i] == 0) {
      text_index[i] = text_count;
      ++text_count;
    } else {
      image_index[i] = image_count;
      ++image_count;
    }
  }
  return api::SUCCESS;
}

static int xpu3_wrapper(Context* ctx,
                        const int* token_type_ids,  // x
                        int* text_index,            // y1
                        int* image_index,           // y2
                        const int64_t token_num) {
  xpu3::plugin::text_image_index_out_kernel<<<1, 1, ctx->xpu_stream>>>(
      token_type_ids, text_index, image_index, token_num);
  return api::SUCCESS;
}

int text_image_index_out(Context* ctx,
                         const int* token_type_ids,  // x
                         int* text_index,            // y1
                         int* image_index,           // y2
                         const int64_t token_num) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "text_image_index_out", int);
  WRAPPER_DUMP_PARAM4(ctx, token_type_ids, text_index, image_index, token_num);
  WRAPPER_DUMP(ctx);
  WRAPPER_ASSERT_GT(ctx, token_num, 0);
  WRAPPER_CHECK_PTR(ctx, int, token_num, token_type_ids);
  WRAPPER_CHECK_PTR(ctx, int, token_num, text_index);
  WRAPPER_CHECK_PTR(ctx, int, token_num, image_index);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx, token_type_ids, text_index, image_index, token_num);
  } else if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(
        ctx, token_type_ids, text_index, image_index, token_num);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
