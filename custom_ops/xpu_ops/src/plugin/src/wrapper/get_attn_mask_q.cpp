// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#include "xpu/plugin.h"
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace fd_xpu3 {

__attribute__((global)) void get_attn_mask_q(int* startend_row_indices,
                                             const int* attn_mask_kv,
                                             const int* cu_seqlens_q,
                                             const int* cu_seqlens_k,
                                             const int kv_token_num,
                                             const int max_batch_size);

}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

static int cpu_wrapper(api::Context* ctx,
                       int* startend_row_indices,
                       const int* attn_mask_kv,
                       const int* cu_seqlens_q,
                       const int* cu_seqlens_k,
                       const int kv_token_num,
                       const int max_batch_size) {
  // Mirrors the GPU kernel logic serially for CPU debug usage.
  for (int cu_seqlens_k_idx = 0; cu_seqlens_k_idx < kv_token_num;
       cu_seqlens_k_idx++) {
    int batch_id = 0;
    for (int i = 0; i < max_batch_size; ++i) {
      if (cu_seqlens_k_idx >= cu_seqlens_k[i] &&
          cu_seqlens_k_idx < cu_seqlens_k[i + 1]) {
        batch_id = i;
        break;
      }
    }
    const int this_batch_q_start = cu_seqlens_q[batch_id];
    const int this_batch_q_end = cu_seqlens_q[batch_id + 1];
    const int this_batch_q_len = this_batch_q_end - this_batch_q_start;
    const int kv_start = cu_seqlens_k[batch_id];
    const int kv_end = cu_seqlens_k[batch_id + 1];
    const int kv_len = kv_end - kv_start;
    const int cache_k_idx = cu_seqlens_k_idx - kv_start;

    int row_start = this_batch_q_end;
    int row_end = this_batch_q_end;
    for (int this_batch_q_idx = this_batch_q_start;
         this_batch_q_idx < this_batch_q_end;
         ++this_batch_q_idx) {
      const int append_mask_k_end =
          attn_mask_kv ? attn_mask_kv[this_batch_q_idx * 2 + 1] - 1
                       : this_batch_q_idx - this_batch_q_start + kv_len -
                             this_batch_q_len;
      if (cache_k_idx <= append_mask_k_end) {
        row_end = std::min(row_end, this_batch_q_idx);
        break;
      }
    }
    startend_row_indices[cu_seqlens_k_idx * 2 + 0] = row_start;
    startend_row_indices[cu_seqlens_k_idx * 2 + 1] = row_end;
  }
  return api::SUCCESS;
}

static int xpu3_wrapper(api::Context* ctx,
                        int* startend_row_indices,
                        const int* attn_mask_kv,
                        const int* cu_seqlens_q,
                        const int* cu_seqlens_k,
                        const int kv_token_num,
                        const int max_batch_size) {
  int32_t ret_xre =
      fd_xpu3::get_attn_mask_q<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          startend_row_indices,
          attn_mask_kv,
          cu_seqlens_q,
          cu_seqlens_k,
          kv_token_num,
          max_batch_size);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int get_attn_mask_q(api::Context* ctx,
                    int* startend_row_indices,
                    const int* attn_mask_kv,
                    const int* cu_seqlens_q,
                    const int* cu_seqlens_k,
                    const int kv_token_num,
                    const int max_batch_size) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "get_attn_mask_q", int);
  WRAPPER_DUMP_PARAM4(
      ctx, startend_row_indices, attn_mask_kv, cu_seqlens_q, cu_seqlens_k);
  WRAPPER_DUMP_PARAM2(ctx, kv_token_num, max_batch_size);
  WRAPPER_DUMP(ctx);
  WRAPPER_ASSERT_GT(ctx, kv_token_num, 0);
  WRAPPER_ASSERT_GT(ctx, max_batch_size, 0);
  WRAPPER_CHECK_PTR(ctx, int, kv_token_num * 2, startend_row_indices);
  WRAPPER_CHECK_PTR(ctx, int, max_batch_size + 1, cu_seqlens_q);
  WRAPPER_CHECK_PTR(ctx, int, max_batch_size + 1, cu_seqlens_k);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       startend_row_indices,
                       attn_mask_kv,
                       cu_seqlens_q,
                       cu_seqlens_k,
                       kv_token_num,
                       max_batch_size);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        startend_row_indices,
                        attn_mask_kv,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        kv_token_num,
                        max_batch_size);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace fastdeploy
