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

__attribute__((global)) void speculate_get_logits(
    float* draft_logits,
    int* next_token_num,
    int* batch_token_num,
    int* cu_next_token_offset,
    int* cu_batch_token_offset,
    const float* logits,
    const float* first_token_logits,
    const int* seq_lens_this_time,
    const int* seq_lens_encoder,
    const int real_bsz,
    const int vocab_size);
}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

static int cpu_wrapper(float* draft_logits,
                       int* next_token_num,
                       int* batch_token_num,
                       int* cu_next_token_offset,
                       int* cu_batch_token_offset,
                       const float* logits,
                       const float* first_token_logits,
                       const int* seq_lens_this_time,
                       const int* seq_lens_encoder,
                       const int real_bsz,
                       const int vocab_size) {
  int batch_token_num_sum = 0;
  int next_token_num_sum = 0;
  for (int bid = 0; bid < real_bsz; bid++) {
    // prefix sum
    cu_batch_token_offset[bid] = batch_token_num_sum;
    cu_next_token_offset[bid] = next_token_num_sum;

    batch_token_num[bid] =
        seq_lens_encoder[bid] > 0 ? 2 : seq_lens_this_time[bid];
    next_token_num[bid] =
        seq_lens_encoder[bid] > 0 ? 1 : seq_lens_this_time[bid];

    batch_token_num_sum += batch_token_num[bid];
    next_token_num_sum += next_token_num[bid];

    auto* draft_logits_now =
        draft_logits + cu_batch_token_offset[bid] * vocab_size;
    auto* logits_now = logits + cu_next_token_offset[bid] * vocab_size;
    auto* first_token_logits_now = first_token_logits + bid * vocab_size;
    for (int i = 0; i < vocab_size; i++) {
      if (seq_lens_encoder[bid] > 0) {
        draft_logits_now[i] = first_token_logits_now[i];
        draft_logits_now[vocab_size + i] = logits_now[i];
      } else {
        for (int j = 0; j < seq_lens_this_time[bid]; j++) {
          draft_logits_now[j * vocab_size + i] = logits_now[j * vocab_size + i];
        }
      }
    }
  }
  return api::SUCCESS;
}

static int xpu3_wrapper(api::Context* ctx,
                        float* draft_logits,
                        int* next_token_num,
                        int* batch_token_num,
                        int* cu_next_token_offset,
                        int* cu_batch_token_offset,
                        const float* logits,
                        const float* first_token_logits,
                        const int* seq_lens_this_time,
                        const int* seq_lens_encoder,
                        const int real_bsz,
                        const int vocab_size) {
  int32_t ret_xre =
      fd_xpu3::speculate_get_logits<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          draft_logits,
          next_token_num,
          batch_token_num,
          cu_next_token_offset,
          cu_batch_token_offset,
          logits,
          first_token_logits,
          seq_lens_this_time,
          seq_lens_encoder,
          real_bsz,
          vocab_size);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int speculate_get_logits(api::Context* ctx,
                         float* draft_logits,
                         int* next_token_num,
                         int* batch_token_num,
                         int* cu_next_token_offset,
                         int* cu_batch_token_offset,
                         const float* logits,
                         const float* first_token_logits,
                         const int* seq_lens_this_time,
                         const int* seq_lens_encoder,
                         const int real_bsz,
                         const int vocab_size) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_get_logits", float);
  WRAPPER_DUMP_PARAM6(ctx,
                      draft_logits,
                      next_token_num,
                      batch_token_num,
                      cu_next_token_offset,
                      cu_batch_token_offset,
                      logits);
  WRAPPER_DUMP_PARAM5(ctx,
                      first_token_logits,
                      seq_lens_this_time,
                      seq_lens_encoder,
                      real_bsz,
                      vocab_size);
  WRAPPER_DUMP(ctx);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, next_token_num);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, batch_token_num);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, cu_next_token_offset);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, cu_batch_token_offset);
  WRAPPER_CHECK_PTR(ctx, float, real_bsz* vocab_size, first_token_logits);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_encoder);
  WRAPPER_ASSERT_LE(ctx, real_bsz, 256 * 1024 / sizeof(int) / 5);
  WRAPPER_ASSERT_GT(ctx, vocab_size, 0);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(draft_logits,
                       next_token_num,
                       batch_token_num,
                       cu_next_token_offset,
                       cu_batch_token_offset,
                       logits,
                       first_token_logits,
                       seq_lens_this_time,
                       seq_lens_encoder,
                       real_bsz,
                       vocab_size);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        draft_logits,
                        next_token_num,
                        batch_token_num,
                        cu_next_token_offset,
                        cu_batch_token_offset,
                        logits,
                        first_token_logits,
                        seq_lens_this_time,
                        seq_lens_encoder,
                        real_bsz,
                        vocab_size);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace fastdeploy
