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
__attribute__((global)) void speculate_set_value_by_flag_and_id(
    int64_t *pre_ids_all,
    const int64_t *accept_tokens,
    int *accept_num,
    const bool *stop_flags,
    const int *seq_lens_encoder,
    int *seq_lens_decoder,
    const int64_t *step_idx,
    int bs,
    int length,
    int max_draft_tokens);
}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

static int cpu_wrapper(api::Context *ctx,
                       int64_t *pre_ids_all,          // bs * length
                       const int64_t *accept_tokens,  // bs * max_draft_tokens
                       int *accept_num,               // bs
                       const bool *stop_flags,
                       const int *seq_lens_encoder,
                       int *seq_lens_decoder,
                       const int64_t *step_idx,
                       int bs,
                       int length,
                       int max_draft_tokens) {
  for (int i = 0; i < bs; i++) {
    if (!stop_flags[i]) {
      int64_t *pre_ids_all_now = pre_ids_all + i * length;
      const int64_t *accept_tokens_now = accept_tokens + i * max_draft_tokens;
      int accept_num_now = accept_num[i];
      int64_t step_idx_now = step_idx[i];
      if (seq_lens_encoder[i] == 0 && seq_lens_decoder[i] == 0) continue;
      if (step_idx_now >= 0) {
        for (int j = 0; j < accept_num_now; j++) {
          pre_ids_all_now[step_idx_now - j] =
              accept_tokens_now[accept_num_now - 1 - j];
        }
      }
    } else {
      accept_num[i] = 0;
      seq_lens_decoder[i] = 0;
    }
  }
  return api::SUCCESS;
}

static int xpu3_wrapper(api::Context *ctx,
                        int64_t *pre_ids_all,
                        const int64_t *accept_tokens,
                        int *accept_num,
                        const bool *stop_flags,
                        const int *seq_lens_encoder,
                        int *seq_lens_decoder,
                        const int64_t *step_idx,
                        int bs,
                        int length,
                        int max_draft_tokens) {
  api::ctx_guard RAII_GUARD(ctx);
  using XPU_INT64 = typename api::XPUIndexType<int64_t>::type;

  int32_t ret_xre =
      fd_xpu3::speculate_set_value_by_flag_and_id<<<ctx->ncluster(),
                                                    64,
                                                    ctx->xpu_stream>>>(
          reinterpret_cast<XPU_INT64 *>(pre_ids_all),
          reinterpret_cast<const XPU_INT64 *>(accept_tokens),
          accept_num,
          stop_flags,
          seq_lens_encoder,
          seq_lens_decoder,
          reinterpret_cast<const XPU_INT64 *>(step_idx),
          bs,
          length,
          max_draft_tokens);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int speculate_set_value_by_flag_and_id(api::Context *ctx,
                                       int64_t *pre_ids_all,
                                       const int64_t *accept_tokens,
                                       int *accept_num,
                                       const bool *stop_flags,
                                       const int *seq_lens_encoder,
                                       int *seq_lens_decoder,
                                       const int64_t *step_idx,
                                       int bs,
                                       int length,
                                       int max_draft_tokens) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_set_value_by_flag_and_id", int);
  WRAPPER_DUMP_PARAM6(ctx,
                      pre_ids_all,
                      accept_tokens,
                      accept_num,
                      stop_flags,
                      seq_lens_encoder,
                      seq_lens_decoder);
  WRAPPER_DUMP_PARAM4(ctx, step_idx, bs, length, max_draft_tokens);
  WRAPPER_DUMP(ctx);

  WRAPPER_ASSERT_LE(ctx, max_draft_tokens, 500);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       pre_ids_all,
                       accept_tokens,
                       accept_num,
                       stop_flags,
                       seq_lens_encoder,
                       seq_lens_decoder,
                       step_idx,
                       bs,
                       length,
                       max_draft_tokens);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        pre_ids_all,
                        accept_tokens,
                        accept_num,
                        stop_flags,
                        seq_lens_encoder,
                        seq_lens_decoder,
                        step_idx,
                        bs,
                        length,
                        max_draft_tokens);
  }

  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace fastdeploy
