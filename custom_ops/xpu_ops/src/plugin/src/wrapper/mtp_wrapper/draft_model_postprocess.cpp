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

namespace xpu2 {
namespace plugin {
__attribute__((global)) void draft_model_postprocess(
    const int64_t* base_model_draft_tokens,
    int* base_model_seq_lens_this_time,
    const int* base_model_seq_lens_encoder,
    const bool* base_model_stop_flags,
    int bsz,
    int base_model_draft_token_len);
}  // namespace plugin
}  // namespace xpu2

namespace xpu3 {
namespace plugin {
__attribute__((global)) void draft_model_postprocess(
    const int64_t* base_model_draft_tokens,
    int* base_model_seq_lens_this_time,
    const int* base_model_seq_lens_encoder,
    const bool* base_model_stop_flags,
    int bsz,
    int base_model_draft_token_len);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {
static int cpu_wrapper(
    Context* ctx,
    const int64_t*
        base_model_draft_tokens,  // size = [bsz, base_model_draft_token_len]
    int* base_model_seq_lens_this_time,      // size = [bsz]
    const int* base_model_seq_lens_encoder,  // size = [bsz]
    const bool* base_model_stop_flags,       // size = [bsz]
    int bsz,
    int base_model_draft_token_len) {
  // 遍历每个样本
  for (int tid = 0; tid < bsz; ++tid) {
    if (!base_model_stop_flags[tid] && base_model_seq_lens_encoder[tid] == 0) {
      // 获取当前样本的草稿token指针
      const int64_t* base_model_draft_tokens_now =
          base_model_draft_tokens + tid * base_model_draft_token_len;
      // 计算有效token数量（非-1的token）
      int token_num = 0;
      for (int i = 0; i < base_model_draft_token_len; ++i) {
        if (base_model_draft_tokens_now[i] != -1) {
          token_num++;
        }
      }
      // 更新序列长度
      base_model_seq_lens_this_time[tid] = token_num;
    } else if (base_model_stop_flags[tid]) {
      // 已停止的样本序列长度为0
      base_model_seq_lens_this_time[tid] = 0;
    }
  }
  return api::SUCCESS;
}

static int xpu3_wrapper(Context* ctx,
                        const int64_t* base_model_draft_tokens,
                        int* base_model_seq_lens_this_time,
                        const int* base_model_seq_lens_encoder,
                        const bool* base_model_stop_flags,
                        int bsz,
                        int base_model_draft_token_len) {
  xpu3::plugin::
      draft_model_postprocess<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          reinterpret_cast<const xpu3::int64_t*>(base_model_draft_tokens),
          base_model_seq_lens_this_time,
          base_model_seq_lens_encoder,
          base_model_stop_flags,
          bsz,
          base_model_draft_token_len);
  return api::SUCCESS;
}

int draft_model_postprocess(Context* ctx,
                            const int64_t* base_model_draft_tokens,
                            int* base_model_seq_lens_this_time,
                            const int* base_model_seq_lens_encoder,
                            const bool* base_model_stop_flags,
                            int bsz,
                            int base_model_draft_token_len) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_PARAM6(ctx,
                      base_model_draft_tokens,
                      base_model_seq_lens_this_time,
                      base_model_seq_lens_encoder,
                      base_model_stop_flags,
                      bsz,
                      base_model_draft_token_len);
  WRAPPER_DUMP(ctx);
  WRAPPER_CHECK_PTR(
      ctx, int64_t, bsz * base_model_draft_token_len, base_model_draft_tokens);
  WRAPPER_CHECK_PTR(ctx, int, bsz, base_model_seq_lens_encoder);
  WRAPPER_CHECK_PTR(ctx, bool, bsz, base_model_stop_flags);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       base_model_draft_tokens,
                       base_model_seq_lens_this_time,
                       base_model_seq_lens_encoder,
                       base_model_stop_flags,
                       bsz,
                       base_model_draft_token_len);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        base_model_draft_tokens,
                        base_model_seq_lens_this_time,
                        base_model_seq_lens_encoder,
                        base_model_stop_flags,
                        bsz,
                        base_model_draft_token_len);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

// template int draft_model_postprocess(
//     Context*, const int64_t*, int*, const int*, const bool*, int, int);
}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
