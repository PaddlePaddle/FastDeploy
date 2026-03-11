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
__attribute__((global)) void speculate_get_target_logits_kernel(
    float* target_logtis,
    const float* logits,
    const int* cu_batch_token_offset,
    const int* ori_cu_batch_token_offset,
    const int* seq_lens_this_time,
    const int* seq_lens_encoder,
    const int* accept_num,
    const int vocab_size,
    const int real_bsz);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context* ctx,
                       float* target_logtis,
                       const float* logits,
                       const int* cu_batch_token_offset,
                       const int* ori_cu_batch_token_offset,
                       const int* seq_lens_this_time,
                       const int* seq_lens_encoder,
                       const int* accept_num,
                       const int vocab_size,
                       const int real_bsz) {
  for (int bid = 0; bid < real_bsz; bid++) {
    auto* target_logtis_now =
        target_logtis + cu_batch_token_offset[bid] * vocab_size;
    auto* logits_now = logits + ori_cu_batch_token_offset[bid] * vocab_size;
    for (int i = 0; i < vocab_size; i++) {
      if (seq_lens_encoder[bid] > 0) {
        target_logtis_now[i] = logits_now[i];
      } else {
        for (int j = 0; j < accept_num[bid]; j++) {
          target_logtis_now[j * vocab_size + i] =
              logits_now[j * vocab_size + i];
        }
      }
    }
  }
  return api::SUCCESS;
}

static int xpu3_wrapper(Context* ctx,
                        float* target_logtis,
                        const float* logits,
                        const int* cu_batch_token_offset,
                        const int* ori_cu_batch_token_offset,
                        const int* seq_lens_this_time,
                        const int* seq_lens_encoder,
                        const int* accept_num,
                        const int vocab_size,
                        const int real_bsz) {
  ctx_guard RAII_GUARD(ctx);
  xpu3::plugin::speculate_get_target_logits_kernel<<<ctx->ncluster(),
                                                     64,
                                                     ctx->xpu_stream>>>(
      target_logtis,
      logits,
      cu_batch_token_offset,
      ori_cu_batch_token_offset,
      seq_lens_this_time,
      seq_lens_encoder,
      accept_num,
      vocab_size,
      real_bsz);
  return api::SUCCESS;
}

int speculate_get_target_logits(Context* ctx,
                                float* target_logtis,
                                const float* logits,
                                const int* cu_batch_token_offset,
                                const int* ori_cu_batch_token_offset,
                                const int* seq_lens_this_time,
                                const int* seq_lens_encoder,
                                const int* accept_num,
                                const int vocab_size,
                                const int real_bsz) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_get_target_logits", int);
  WRAPPER_DUMP_PARAM5(ctx,
                      target_logtis,
                      logits,
                      cu_batch_token_offset,
                      ori_cu_batch_token_offset,
                      seq_lens_this_time);
  WRAPPER_DUMP_PARAM4(ctx, seq_lens_encoder, accept_num, vocab_size, real_bsz);
  // only part, real size = sum(accept_num) * vocab_size
  WRAPPER_CHECK_PTR(ctx, float, vocab_size* real_bsz, logits);
  WRAPPER_CHECK_PTR(ctx, float, vocab_size* real_bsz, target_logtis);

  WRAPPER_CHECK_PTR(ctx, int, real_bsz, cu_batch_token_offset);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_encoder);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, accept_num);
  WRAPPER_ASSERT_GT(ctx, vocab_size, 0);

  WRAPPER_DUMP(ctx);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       target_logtis,
                       logits,
                       cu_batch_token_offset,
                       ori_cu_batch_token_offset,
                       seq_lens_this_time,
                       seq_lens_encoder,
                       accept_num,
                       vocab_size,
                       real_bsz);
  } else if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        target_logtis,
                        logits,
                        cu_batch_token_offset,
                        ori_cu_batch_token_offset,
                        seq_lens_this_time,
                        seq_lens_encoder,
                        accept_num,
                        vocab_size,
                        real_bsz);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
