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

__attribute__((global)) void recover_spec_decode_task(
    bool *stop_flags,
    int *seq_lens_this_time,
    int *seq_lens_encoder,
    int *seq_lens_decoder,
    int *step_seq_lens_decoder,
    int *block_tables,
    bool *is_block_step,
    int64_t *draft_tokens,
    const int64_t *step_draft_tokens,
    const int *step_seq_lens_this_time,
    const int bsz,
    const int block_num_per_seq,
    const int block_size,
    const int draft_tokens_len,
    const int num_extra_tokens);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int xpu3_wrapper(Context *ctx,
                        bool *stop_flags,
                        int *seq_lens_this_time,
                        int *seq_lens_encoder,
                        int *seq_lens_decoder,
                        int *step_seq_lens_decoder,
                        int *block_tables,
                        bool *is_block_step,
                        int64_t *draft_tokens,
                        const int64_t *step_draft_tokens,
                        const int *step_seq_lens_this_time,
                        const int bsz,
                        const int block_num_per_seq,
                        const int block_size,
                        const int draft_tokens_len,
                        const int num_extra_tokens) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  auto recover_spec_decode_task = xpu3::plugin::recover_spec_decode_task;
  int32_t ret_xre =
      recover_spec_decode_task<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          stop_flags,
          seq_lens_this_time,
          seq_lens_encoder,
          seq_lens_decoder,
          step_seq_lens_decoder,
          block_tables,
          is_block_step,
          reinterpret_cast<XPU_INT64 *>(draft_tokens),
          reinterpret_cast<const XPU_INT64 *>(step_draft_tokens),
          step_seq_lens_this_time,
          bsz,
          block_num_per_seq,
          block_size,
          draft_tokens_len,
          num_extra_tokens);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int recover_spec_decode_task(Context *ctx,
                             bool *stop_flags,
                             int *seq_lens_this_time,
                             int *seq_lens_encoder,
                             int *seq_lens_decoder,
                             int *step_seq_lens_decoder,
                             int *block_tables,
                             bool *is_block_step,
                             int64_t *draft_tokens,
                             const int64_t *step_draft_tokens,
                             const int *step_seq_lens_this_time,
                             const int bsz,
                             const int block_num_per_seq,
                             const int block_size,
                             const int draft_tokens_len,
                             const int num_extra_tokens) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "recover_spec_decode_task", int);
  WRAPPER_DUMP_PARAM5(ctx,
                      stop_flags,
                      seq_lens_this_time,
                      seq_lens_encoder,
                      seq_lens_decoder,
                      step_seq_lens_decoder);
  WRAPPER_DUMP_PARAM4(
      ctx, block_tables, is_block_step, draft_tokens, step_draft_tokens);
  WRAPPER_DUMP_PARAM6(ctx,
                      step_seq_lens_this_time,
                      bsz,
                      block_num_per_seq,
                      block_size,
                      draft_tokens_len,
                      num_extra_tokens);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, bool, bsz, stop_flags);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_encoder);
  WRAPPER_CHECK_PTR(ctx, int, bsz, step_seq_lens_decoder);
  WRAPPER_CHECK_PTR(ctx, int, bsz *block_num_per_seq, block_tables);
  WRAPPER_CHECK_PTR(ctx, bool, bsz, is_block_step);
  WRAPPER_CHECK_PTR(ctx, int64_t, bsz * draft_tokens_len, step_draft_tokens);
  WRAPPER_CHECK_PTR(ctx, int64_t, bsz * draft_tokens_len, draft_tokens);

  if (ctx->dev().type() == api::kCPU) {
    assert(false);
  }
  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        stop_flags,
                        seq_lens_this_time,
                        seq_lens_encoder,
                        seq_lens_decoder,
                        step_seq_lens_decoder,
                        block_tables,
                        is_block_step,
                        draft_tokens,
                        step_draft_tokens,
                        step_seq_lens_this_time,
                        bsz,
                        block_num_per_seq,
                        block_size,
                        draft_tokens_len,
                        num_extra_tokens);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
