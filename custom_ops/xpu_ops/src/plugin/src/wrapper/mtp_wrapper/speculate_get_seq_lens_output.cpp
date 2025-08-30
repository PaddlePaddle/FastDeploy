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
__attribute__((global)) void speculate_get_seq_lens_output(
    int* seq_lens_output,
    const int* seq_lens_this_time,
    const int* seq_lens_encoder,
    const int* seq_lens_decoder,
    const int real_bsz);
}
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context* ctx,
                       int* seq_lens_output,
                       const int* seq_lens_this_time,
                       const int* seq_lens_encoder,
                       const int* seq_lens_decoder,
                       const int real_bsz) {
  for (int bid = 0; bid < real_bsz; ++bid) {
    if (seq_lens_this_time[bid] == 0) {
      continue;
    } else if (seq_lens_this_time[bid] == 1) {
      seq_lens_output[bid] = 1;
    } else if (seq_lens_encoder[bid] != 0) {
      seq_lens_output[bid] = 1;
    } else {
      seq_lens_output[bid] = seq_lens_this_time[bid];
    }
  }
  return SUCCESS;
}

static int xpu2or3_wrapper(Context* ctx,
                           int* seq_lens_output,
                           const int* seq_lens_this_time,
                           const int* seq_lens_encoder,
                           const int* seq_lens_decoder,
                           const int real_bsz) {
  ctx_guard RAII_GUARD(ctx);
  xpu3::plugin::
      speculate_get_seq_lens_output<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          seq_lens_output,
          seq_lens_this_time,
          seq_lens_encoder,
          seq_lens_decoder,
          real_bsz);

  return api::SUCCESS;
}

int speculate_get_seq_lens_output(Context* ctx,
                                  int* seq_lens_output,
                                  const int* seq_lens_this_time,
                                  const int* seq_lens_encoder,
                                  const int* seq_lens_decoder,
                                  const int real_bsz) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_get_seq_lens_output", int);
  WRAPPER_DUMP_PARAM5(ctx,
                      seq_lens_output,
                      seq_lens_this_time,
                      seq_lens_encoder,
                      seq_lens_decoder,
                      real_bsz);
  WRAPPER_DUMP(ctx);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       seq_lens_output,
                       seq_lens_this_time,
                       seq_lens_encoder,
                       seq_lens_decoder,
                       real_bsz);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu2or3_wrapper(ctx,
                           seq_lens_output,
                           seq_lens_this_time,
                           seq_lens_encoder,
                           seq_lens_decoder,
                           real_bsz);
  }

  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
