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

namespace fd_xpu3 {
__attribute__((global)) void speculate_clear_accept_nums(
    int* accept_num, const int* seq_lens_decoder, const int max_bsz);
}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

static int cpu_wrapper(api::Context* ctx,
                       int* accept_num,
                       const int* seq_lens_decoder,
                       const int max_bsz) {
  for (int i = 0; i < max_bsz; i++) {
    accept_num[i] = seq_lens_decoder[i] == 0 ? 0 : accept_num[i];
  }
  return api::SUCCESS;
}

static int xpu3_wrapper(api::Context* ctx,
                        int* accept_num,
                        const int* seq_lens_decoder,
                        const int max_bsz) {
  api::ctx_guard RAII_GUARD(ctx);
  int32_t ret_xre =
      fd_xpu3::speculate_clear_accept_nums<<<1, 64, ctx->xpu_stream>>>(
          accept_num, seq_lens_decoder, max_bsz);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);

  return api::SUCCESS;
}

int speculate_clear_accept_nums(api::Context* ctx,
                                int* accept_num,
                                const int* seq_lens_decoder,
                                const int max_bsz) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_clear_accept_nums", int);
  WRAPPER_DUMP_PARAM3(ctx, accept_num, seq_lens_decoder, max_bsz);
  WRAPPER_DUMP(ctx);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx, accept_num, seq_lens_decoder, max_bsz);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx, accept_num, seq_lens_decoder, max_bsz);
  }

  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace fastdeploy
