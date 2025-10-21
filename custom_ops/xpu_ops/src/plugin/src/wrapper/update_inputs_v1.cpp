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

__attribute__((global)) void update_inputs_v1(bool *not_need_stop,
                                              int *seq_lens_this_time,
                                              int *seq_lens_encoder,
                                              int *seq_lens_decoder,
                                              int *step_seq_lens_decoder,
                                              int64_t *prompt_lens,
                                              int64_t *topk_ids,
                                              int64_t *input_ids,
                                              int *block_tables,
                                              const int64_t *stop_nums,
                                              bool *stop_flags,
                                              bool *is_block_step,
                                              const int64_t *next_tokens,
                                              const int bsz,
                                              const int max_bsz,
                                              const int input_ids_stride,
                                              const int block_num_per_seq,
                                              const int block_size);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int xpu3_wrapper(Context *ctx,
                        bool *not_need_stop,
                        int *seq_lens_this_time,
                        int *seq_lens_encoder,
                        int *seq_lens_decoder,
                        int *step_seq_lens_decoder,
                        int64_t *prompt_lens,
                        int64_t *topk_ids,
                        int64_t *input_ids,
                        int *block_tables,
                        const int64_t *stop_nums,
                        bool *stop_flags,
                        bool *is_block_step,
                        const int64_t *next_tokens,
                        const int bsz,
                        const int max_bsz,
                        const int input_ids_stride,
                        const int block_num_per_seq,
                        const int block_size) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  auto update_inputs_v1 = xpu3::plugin::update_inputs_v1;
  // kernel 内要做 reduce，只能用 1 个 cluster
  update_inputs_v1<<<1, 64, ctx->xpu_stream>>>(
      not_need_stop,
      seq_lens_this_time,
      seq_lens_encoder,
      seq_lens_decoder,
      step_seq_lens_decoder,
      reinterpret_cast<XPU_INT64 *>(prompt_lens),
      reinterpret_cast<XPU_INT64 *>(topk_ids),
      reinterpret_cast<XPU_INT64 *>(input_ids),
      block_tables,
      reinterpret_cast<const XPU_INT64 *>(stop_nums),
      stop_flags,
      is_block_step,
      reinterpret_cast<const XPU_INT64 *>(next_tokens),
      bsz,
      max_bsz,
      input_ids_stride,
      block_num_per_seq,
      block_size);
  return api::SUCCESS;
}

int update_inputs_v1(Context *ctx,
                     bool *not_need_stop,
                     int *seq_lens_this_time,
                     int *seq_lens_encoder,
                     int *seq_lens_decoder,
                     int *step_seq_lens_decoder,
                     int64_t *prompt_lens,
                     int64_t *topk_ids,
                     int64_t *input_ids,
                     int *block_tables,
                     const int64_t *stop_nums,
                     bool *stop_flags,
                     bool *is_block_step,
                     const int64_t *next_tokens,
                     const int bsz,
                     const int max_bsz,
                     const int input_ids_stride,
                     const int block_num_per_seq,
                     const int block_size) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "update_inputs_v1", int);
  WRAPPER_DUMP_PARAM5(ctx,
                      not_need_stop,
                      seq_lens_this_time,
                      seq_lens_encoder,
                      seq_lens_decoder,
                      step_seq_lens_decoder);
  WRAPPER_DUMP_PARAM5(
      ctx, prompt_lens, topk_ids, input_ids, block_tables, stop_nums);
  WRAPPER_DUMP_PARAM3(ctx, stop_flags, is_block_step, next_tokens);
  WRAPPER_DUMP_PARAM5(
      ctx, bsz, max_bsz, input_ids_stride, block_num_per_seq, block_size);
  WRAPPER_DUMP(ctx);
  if (ctx->dev().type() == api::kCPU) {
    assert(false);
  }
  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        not_need_stop,
                        seq_lens_this_time,
                        seq_lens_encoder,
                        seq_lens_decoder,
                        step_seq_lens_decoder,
                        prompt_lens,
                        topk_ids,
                        input_ids,
                        block_tables,
                        stop_nums,
                        stop_flags,
                        is_block_step,
                        next_tokens,
                        bsz,
                        max_bsz,
                        input_ids_stride,
                        block_num_per_seq,
                        block_size);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
