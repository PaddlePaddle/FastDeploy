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
__attribute__((global)) void speculate_insert_first_token_kernel(
    int64_t* token_ids,
    const int64_t* accept_tokens,
    const int64_t* next_tokens,
    const int* cu_next_token_offset,
    const int* cu_batch_token_offset,
    const int* seq_lens_this_time,
    const int* seq_lens_encoder,
    const int max_draft_tokens,
    const int real_bsz);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context* ctx,
                       int64_t* token_ids,
                       const int64_t* accept_tokens,
                       const int64_t* next_tokens,
                       const int* cu_next_token_offset,
                       const int* cu_batch_token_offset,
                       const int* seq_lens_this_time,
                       const int* seq_lens_encoder,
                       const int max_draft_tokens,
                       const int real_bsz) {
  return api::SUCCESS;
}

static int xpu3_wrapper(Context* ctx,
                        int64_t* token_ids,
                        const int64_t* accept_tokens,
                        const int64_t* next_tokens,
                        const int* cu_next_token_offset,
                        const int* cu_batch_token_offset,
                        const int* seq_lens_this_time,
                        const int* seq_lens_encoder,
                        const int max_draft_tokens,
                        const int real_bsz) {
  ctx_guard RAII_GUARD(ctx);
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  int32_t ret_xre =
      xpu3::plugin::speculate_insert_first_token_kernel<<<ctx->ncluster(),
                                                          64,
                                                          ctx->xpu_stream>>>(
          reinterpret_cast<XPU_INT64*>(token_ids),
          reinterpret_cast<const XPU_INT64*>(accept_tokens),
          reinterpret_cast<const XPU_INT64*>(next_tokens),
          cu_next_token_offset,
          cu_batch_token_offset,
          seq_lens_this_time,
          seq_lens_encoder,
          max_draft_tokens,
          real_bsz);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int speculate_insert_first_token(Context* ctx,
                                 int64_t* token_ids,
                                 const int64_t* accept_tokens,
                                 const int64_t* next_tokens,
                                 const int* cu_next_token_offset,
                                 const int* cu_batch_token_offset,
                                 const int* seq_lens_this_time,
                                 const int* seq_lens_encoder,
                                 const int max_draft_tokens,
                                 const int real_bsz) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_insert_first_token", int);
  // real size = cu_next_token_offset[-1] > real_bsz
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, next_tokens);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, token_ids);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz* max_draft_tokens, accept_tokens);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       token_ids,
                       accept_tokens,
                       next_tokens,
                       cu_next_token_offset,
                       cu_batch_token_offset,
                       seq_lens_this_time,
                       seq_lens_encoder,
                       max_draft_tokens,
                       real_bsz);
  } else if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        token_ids,
                        accept_tokens,
                        next_tokens,
                        cu_next_token_offset,
                        cu_batch_token_offset,
                        seq_lens_this_time,
                        seq_lens_encoder,
                        max_draft_tokens,
                        real_bsz);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
