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
#include "xpu/refactor/impl/xdnn_impl.h"
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace fd_xpu3 {

__attribute__((global)) void naive_update_model_status_kernel(
    int64_t *accept_tokens,
    int *accept_num,
    int *seq_lens_this_time,
    const int64_t *next_tokens,
    const int *cu_seqlens_q_output,
    int real_bsz,
    int max_step_tokens);
}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

static int cpu_wrapper(api::Context *ctx,
                       int64_t *accept_tokens,
                       int *accept_num,
                       int *seq_lens_this_time,
                       const int64_t *next_tokens,
                       const int *cu_seqlens_q_output,
                       int real_bsz,
                       int max_step_tokens) {
  for (int bid = 0; bid < real_bsz; bid++) {
    if (seq_lens_this_time[bid] > 0) {
      accept_tokens[bid * max_step_tokens] =
          next_tokens[cu_seqlens_q_output[bid + 1] - 1];
      accept_num[bid] = 1;
      seq_lens_this_time[bid] = 1;
    } else {
      accept_num[bid] = 0;
      seq_lens_this_time[bid] = 0;
    }
  }
  return api::SUCCESS;
}

static int xpu3_wrapper(api::Context *ctx,
                        int64_t *accept_tokens,
                        int *accept_num,
                        int *seq_lens_this_time,
                        const int64_t *next_tokens,
                        const int *cu_seqlens_q_output,
                        int real_bsz,
                        int max_step_tokens) {
  using XPU_INT64 = typename api::XPUIndexType<int64_t>::type;
  int32_t ret_xre =
      fd_xpu3::naive_update_model_status_kernel<<<ctx->ncluster(),
                                                  64,
                                                  ctx->xpu_stream>>>(
          reinterpret_cast<XPU_INT64 *>(accept_tokens),
          accept_num,
          seq_lens_this_time,
          reinterpret_cast<const XPU_INT64 *>(next_tokens),
          cu_seqlens_q_output,
          real_bsz,
          max_step_tokens);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int naive_update_model_status(api::Context *ctx,
                              int64_t *accept_tokens,
                              int *accept_num,
                              int *seq_lens_this_time,
                              const int64_t *next_tokens,
                              const int *cu_seqlens_q_output,
                              int real_bsz,
                              int max_step_tokens) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "naive_update_model_status", int);
  WRAPPER_DUMP_PARAM5(ctx,
                      accept_tokens,
                      accept_num,
                      seq_lens_this_time,
                      next_tokens,
                      cu_seqlens_q_output);
  WRAPPER_DUMP_PARAM2(ctx, real_bsz, max_step_tokens);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, int64_t, real_bsz * max_step_tokens, accept_tokens);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, accept_num);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz + 1, cu_seqlens_q_output);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       accept_tokens,
                       accept_num,
                       seq_lens_this_time,
                       next_tokens,
                       cu_seqlens_q_output,
                       real_bsz,
                       max_step_tokens);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        accept_tokens,
                        accept_num,
                        seq_lens_this_time,
                        next_tokens,
                        cu_seqlens_q_output,
                        real_bsz,
                        max_step_tokens);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace fastdeploy
