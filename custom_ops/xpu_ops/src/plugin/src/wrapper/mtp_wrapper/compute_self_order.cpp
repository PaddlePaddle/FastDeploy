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

#include "xpu/plugin.h"
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace xpu3 {
namespace plugin {
__attribute__((global)) void ComputeSelfOrderKernel(
    const int* last_seq_lens_this_time,
    const int* seq_lens_this_time,
    const int64_t* step_idx,
    int* src_map,
    int* output_token_num,
    int bsz);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context* ctx,
                       const int* last_seq_lens_this_time,
                       const int* seq_lens_this_time,
                       const int64_t* step_idx,
                       int* src_map,
                       int* output_token_num,
                       int bsz) {
  int in_offset = 0;
  int out_offset = 0;
  for (int i = 0; i < bsz; i++) {
    int cur_seq_lens_this_time = seq_lens_this_time[i];
    int cur_last_seq_lens_this_time = last_seq_lens_this_time[i];

    // 1. encoder
    if (step_idx[i] == 1 && cur_seq_lens_this_time > 0) {
      in_offset += 1;
      src_map[out_offset++] = in_offset - 1;
      // 2. decoder
    } else if (cur_seq_lens_this_time > 0) /* =1 */ {
      in_offset += cur_last_seq_lens_this_time;
      src_map[out_offset++] = in_offset - 1;
      // 3. stop
    } else {
      // first token end
      if (step_idx[i] == 1) {
        in_offset += cur_last_seq_lens_this_time > 0 ? 1 : 0;
        // normal end
      } else {
        in_offset += cur_last_seq_lens_this_time;
      }
    }
  }
  output_token_num[0] = out_offset;
  return api::SUCCESS;
}

static int xpu3_wrapper(Context* ctx,
                        const int* last_seq_lens_this_time,
                        const int* seq_lens_this_time,
                        const int64_t* step_idx,
                        int* src_map,
                        int* output_token_num,
                        int bsz) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  xpu3::plugin::ComputeSelfOrderKernel<<<1, 1, ctx->xpu_stream>>>(
      last_seq_lens_this_time,
      seq_lens_this_time,
      reinterpret_cast<const XPU_INT64*>(step_idx),
      src_map,
      output_token_num,
      bsz);
  return api::SUCCESS;
}

int compute_self_order(Context* ctx,
                       const int* last_seq_lens_this_time,
                       const int* seq_lens_this_time,
                       const int64_t* step_idx,
                       int* src_map,
                       int* output_token_num,
                       int bsz) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_PARAM6(ctx,
                      last_seq_lens_this_time,
                      seq_lens_this_time,
                      step_idx,
                      src_map,
                      output_token_num,
                      bsz);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, int, bsz, last_seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int64_t, bsz, step_idx);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       last_seq_lens_this_time,
                       seq_lens_this_time,
                       step_idx,
                       src_map,
                       output_token_num,
                       bsz);
  } else if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        last_seq_lens_this_time,
                        seq_lens_this_time,
                        step_idx,
                        src_map,
                        output_token_num,
                        bsz);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
