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
__attribute__((global)) void speculate_get_output_padding_offset(
    int* output_padding_offset,
    int* output_cum_offsets,
    const int* output_cum_offsets_tmp,
    const int* seq_lens_output,
    const int bsz,
    const int max_seq_len);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context* ctx,
                       int* output_padding_offset,
                       int* output_cum_offsets,
                       const int* output_cum_offsets_tmp,
                       const int* seq_lens_output,
                       const int bsz,
                       const int max_seq_len) {
  for (int bi = 0; bi < bsz; bi++) {
    int cum_offset = 0;
    if (bi > 0) {
      cum_offset = output_cum_offsets_tmp[bi - 1];
    }
    output_cum_offsets[bi] = cum_offset;
    for (int token_i = 0; token_i < seq_lens_output[bi]; token_i++) {
      output_padding_offset[bi * max_seq_len - cum_offset + token_i] =
          cum_offset;
    }
  }
  return SUCCESS;
}

static int xpu2or3_wrapper(Context* ctx,
                           int* output_padding_offset,
                           int* output_cum_offsets,
                           const int* output_cum_offsets_tmp,
                           const int* seq_lens_output,
                           const int bsz,
                           const int max_seq_len) {
  ctx_guard RAII_GUARD(ctx);
  xpu3::plugin::speculate_get_output_padding_offset<<<ctx->ncluster(),
                                                      64,
                                                      ctx->xpu_stream>>>(
      output_padding_offset,
      output_cum_offsets,
      output_cum_offsets_tmp,
      seq_lens_output,
      bsz,
      max_seq_len);
  return api::SUCCESS;
}

int speculate_get_output_padding_offset(Context* ctx,
                                        int* output_padding_offset,
                                        int* output_cum_offsets,
                                        const int* output_cum_offsets_tmp,
                                        const int* seq_lens_output,
                                        const int bsz,
                                        const int max_seq_len) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_get_output_padding_offset", int);
  WRAPPER_DUMP_PARAM5(ctx,
                      output_padding_offset,
                      output_cum_offsets,
                      output_cum_offsets_tmp,
                      seq_lens_output,
                      max_seq_len);
  WRAPPER_DUMP(ctx);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       output_padding_offset,
                       output_cum_offsets,
                       output_cum_offsets_tmp,
                       seq_lens_output,
                       bsz,
                       max_seq_len);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu2or3_wrapper(ctx,
                           output_padding_offset,
                           output_cum_offsets,
                           output_cum_offsets_tmp,
                           seq_lens_output,
                           bsz,
                           max_seq_len);
  }

  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
