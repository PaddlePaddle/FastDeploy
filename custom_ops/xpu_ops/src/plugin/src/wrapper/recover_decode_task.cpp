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

namespace fd_xpu3 {

__attribute__((global)) void recover_decode_task(bool *stop_flags,
                                                 int *seq_lens_this_time,
                                                 int *seq_lens_encoder,
                                                 int *seq_lens_decoder,
                                                 int *step_seq_lens_decoder,
                                                 int *block_tables,
                                                 bool *is_block_step,
                                                 const int bsz,
                                                 const int block_num_per_seq,
                                                 const int block_size);

}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

static int xpu3_wrapper(api::Context *ctx,
                        bool *stop_flags,
                        int *seq_lens_this_time,
                        int *seq_lens_encoder,
                        int *seq_lens_decoder,
                        int *step_seq_lens_decoder,
                        int *block_tables,
                        bool *is_block_step,
                        const int bsz,
                        const int block_num_per_seq,
                        const int block_size) {
  using XPU_INT64 = typename api::XPUIndexType<int64_t>::type;
  auto recover_decode_task = fd_xpu3::recover_decode_task;
  int32_t ret_xre =
      recover_decode_task<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          stop_flags,
          seq_lens_this_time,
          seq_lens_encoder,
          seq_lens_decoder,
          step_seq_lens_decoder,
          block_tables,
          is_block_step,
          bsz,
          block_num_per_seq,
          block_size);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int recover_decode_task(api::Context *ctx,
                        bool *stop_flags,
                        int *seq_lens_this_time,
                        int *seq_lens_encoder,
                        int *seq_lens_decoder,
                        int *step_seq_lens_decoder,
                        int *block_tables,
                        bool *is_block_step,
                        const int bsz,
                        const int block_num_per_seq,
                        const int block_size) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "recover_decode_task", int);
  WRAPPER_DUMP_PARAM5(ctx,
                      stop_flags,
                      seq_lens_this_time,
                      seq_lens_encoder,
                      seq_lens_decoder,
                      step_seq_lens_decoder);
  WRAPPER_DUMP_PARAM2(ctx, block_tables, is_block_step);
  WRAPPER_DUMP_PARAM3(ctx, bsz, block_num_per_seq, block_size);
  WRAPPER_DUMP(ctx);
  WRAPPER_ASSERT_GT(ctx, bsz, 0);
  WRAPPER_ASSERT_GT(ctx, block_num_per_seq, 0);
  WRAPPER_ASSERT_GT(ctx, block_size, 0);
  WRAPPER_CHECK_PTR(ctx, bool, bsz, stop_flags);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_encoder);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_decoder);
  WRAPPER_CHECK_PTR(ctx, int, bsz *block_num_per_seq, block_tables);
  // TODO(mayang02): more check ptrs
  if (ctx->dev().type() == api::kCPU) {
    // TODO(mayang02): cpu implement
    WRAPPER_UNIMPLEMENTED(ctx);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        stop_flags,
                        seq_lens_this_time,
                        seq_lens_encoder,
                        seq_lens_decoder,
                        step_seq_lens_decoder,
                        block_tables,
                        is_block_step,
                        bsz,
                        block_num_per_seq,
                        block_size);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace fastdeploy
