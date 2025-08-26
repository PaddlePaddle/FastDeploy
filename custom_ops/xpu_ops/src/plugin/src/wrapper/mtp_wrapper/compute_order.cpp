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
__attribute__((global)) void ComputeOrderKernel(
    const int* seq_lens_this_time,
    const int* seq_lens_encoder,
    const int* base_model_seq_lens_this_time,
    const int* base_model_seq_lens_encoder,
    const int* accept_nums,
    int* position_map,
    int* output_token_num,
    const int bsz,
    const int actual_draft_token_num,
    const int input_token_num);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context* ctx,
                       const int* seq_lens_this_time,
                       const int* seq_lens_encoder,
                       const int* base_model_seq_lens_this_time,
                       const int* base_model_seq_lens_encoder,
                       const int* accept_nums,
                       int* position_map,
                       int* output_token_num,
                       const int bsz,
                       const int actual_draft_token_num,
                       const int input_token_num) {
  int in_offset = 0;   // input_offset(long)
  int out_offset = 0;  // output_offset(short)
  for (int i = 0; i < bsz; ++i) {
    int cur_base_model_seq_lens_this_time = base_model_seq_lens_this_time[i];
    int cur_base_model_seq_lens_encoder = base_model_seq_lens_encoder[i];
    int cur_seq_lens_this_time = seq_lens_this_time[i];
    int accept_num = accept_nums[i];
    int cur_seq_lens_encoder = seq_lens_encoder[i];

    // 1. eagle encoder. Base step=1
    if (cur_seq_lens_encoder > 0) {
      for (int j = 0; j < cur_seq_lens_encoder; j++) {
        position_map[in_offset++] = out_offset++;
      }
      // 2. base model encoder. Base step=0
    } else if (cur_base_model_seq_lens_encoder != 0) {
      // nothing happens
      // 3. New end
    } else if (cur_base_model_seq_lens_this_time != 0 &&
               cur_seq_lens_this_time == 0) {
      in_offset += cur_base_model_seq_lens_this_time;
      // 4. stopped
    } else if (cur_base_model_seq_lens_this_time == 0 &&
               cur_seq_lens_this_time == 0) /* end */ {
      // nothing happens
    } else {
      if (accept_num <=
          actual_draft_token_num) /*Accept partial draft tokens*/ {
        position_map[in_offset + accept_num - 1] = out_offset++;
        in_offset += cur_base_model_seq_lens_this_time;
      } else /*Accept all draft tokens*/ {
        position_map[in_offset + accept_num - 2] = out_offset++;
        position_map[in_offset + accept_num - 1] = out_offset++;
        in_offset += cur_base_model_seq_lens_this_time;
      }
    }
  }
  output_token_num[0] = out_offset;
  return api::SUCCESS;
}

static int xpu3_wrapper(Context* ctx,
                        const int* seq_lens_this_time,
                        const int* seq_lens_encoder,
                        const int* base_model_seq_lens_this_time,
                        const int* base_model_seq_lens_encoder,
                        const int* accept_nums,
                        int* position_map,
                        int* output_token_num,
                        const int bsz,
                        const int actual_draft_token_num,
                        const int input_token_num) {
  xpu3::plugin::ComputeOrderKernel<<<1, 1, ctx->xpu_stream>>>(
      seq_lens_this_time,
      seq_lens_encoder,
      base_model_seq_lens_this_time,
      base_model_seq_lens_encoder,
      accept_nums,
      position_map,
      output_token_num,
      bsz,
      actual_draft_token_num,
      input_token_num);
  return api::SUCCESS;
}

int compute_order(Context* ctx,
                  const int* seq_lens_this_time,
                  const int* seq_lens_encoder,
                  const int* base_model_seq_lens_this_time,
                  const int* base_model_seq_lens_encoder,
                  const int* accept_nums,
                  int* position_map,
                  int* output_token_num,
                  const int bsz,
                  const int actual_draft_token_num,
                  const int input_token_num) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_PARAM5(ctx,
                      seq_lens_this_time,
                      seq_lens_encoder,
                      base_model_seq_lens_this_time,
                      base_model_seq_lens_encoder,
                      accept_nums);
  WRAPPER_DUMP_PARAM5(ctx,
                      position_map,
                      output_token_num,
                      bsz,
                      actual_draft_token_num,
                      input_token_num);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_encoder);
  WRAPPER_CHECK_PTR(ctx, int, bsz, base_model_seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int, bsz, base_model_seq_lens_encoder);
  WRAPPER_CHECK_PTR(ctx, int, bsz, accept_nums);
  WRAPPER_CHECK_PTR(ctx, int, input_token_num, position_map);
  WRAPPER_CHECK_PTR(ctx, int, 1, output_token_num);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       seq_lens_this_time,
                       seq_lens_encoder,
                       base_model_seq_lens_this_time,
                       base_model_seq_lens_encoder,
                       accept_nums,
                       position_map,
                       output_token_num,
                       bsz,
                       actual_draft_token_num,
                       input_token_num);
  } else if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        seq_lens_this_time,
                        seq_lens_encoder,
                        base_model_seq_lens_this_time,
                        base_model_seq_lens_encoder,
                        accept_nums,
                        position_map,
                        output_token_num,
                        bsz,
                        actual_draft_token_num,
                        input_token_num);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
