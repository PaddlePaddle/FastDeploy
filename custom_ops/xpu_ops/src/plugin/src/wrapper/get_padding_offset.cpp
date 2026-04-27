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

__attribute__((global)) void get_padding_offset(int64_t* ids_remove_padding,
                                                int* batch_id_per_token,
                                                int* cu_seqlens_q,
                                                int* cu_seqlens_k,
                                                const int64_t* input_data,
                                                const int* seq_lens,
                                                const int max_seq_len,
                                                const int bs);

}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

static int cpu_wrapper(api::Context* ctx,
                       int* batch_id_per_token,
                       int* cu_seqlens_q,
                       int* cu_seqlens_k,
                       int64_t* x_remove_padding,
                       const int64_t* input_ids,
                       const int* seq_lens,
                       const int max_seq_len,
                       const int bs) {
  int cum_seq_len = 0;
  cu_seqlens_q[0] = 0;
  cu_seqlens_k[0] = 0;
  for (int i = 0; i < bs; i++) {
    for (int j = 0; j < seq_lens[i]; j++) {
      const int tgt = cum_seq_len + j;
      x_remove_padding[tgt] = input_ids[i * max_seq_len + j];
      batch_id_per_token[tgt] = i;
    }
    cum_seq_len += seq_lens[i];
    cu_seqlens_q[i + 1] = cum_seq_len;
    cu_seqlens_k[i + 1] = cum_seq_len;
  }
  return api::SUCCESS;
}

static int xpu3_wrapper(api::Context* ctx,
                        int* batch_id_per_token,
                        int* cu_seqlens_q,
                        int* cu_seqlens_k,
                        int64_t* x_remove_padding,
                        const int64_t* input_ids,
                        const int* seq_lens,
                        const int max_seq_len,
                        const int bs) {
  using XPU_INT64 = typename api::XPUIndexType<int64_t>::type;
  int32_t ret_xre =
      fd_xpu3::get_padding_offset<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          reinterpret_cast<XPU_INT64*>(x_remove_padding),
          batch_id_per_token,
          cu_seqlens_q,
          cu_seqlens_k,
          reinterpret_cast<const XPU_INT64*>(input_ids),
          seq_lens,
          max_seq_len,
          bs);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int get_padding_offset(api::Context* ctx,
                       int* batch_id_per_token,
                       int* cu_seqlens_q,
                       int* cu_seqlens_k,
                       int64_t* x_remove_padding,
                       const int64_t* input_ids,
                       const int* seq_lens,
                       const int max_seq_len,
                       const int bs,
                       const int64_t token_num) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "get_padding_offset", int);
  WRAPPER_DUMP_PARAM3(ctx, batch_id_per_token, cu_seqlens_q, cu_seqlens_k);
  WRAPPER_DUMP_PARAM4(ctx, x_remove_padding, input_ids, seq_lens, max_seq_len);
  WRAPPER_DUMP_PARAM2(ctx, bs, token_num);
  WRAPPER_DUMP(ctx);
  WRAPPER_ASSERT_GT(ctx, bs, 0);
  WRAPPER_ASSERT_GT(ctx, max_seq_len, 0);
  WRAPPER_CHECK_PTR(ctx, int64_t, token_num, x_remove_padding);
  WRAPPER_CHECK_PTR(ctx, int, token_num, batch_id_per_token);
  WRAPPER_CHECK_PTR(ctx, int, bs + 1, cu_seqlens_q);
  WRAPPER_CHECK_PTR(ctx, int, bs + 1, cu_seqlens_k);
  WRAPPER_CHECK_PTR(ctx, int64_t, bs * max_seq_len, input_ids);
  WRAPPER_CHECK_PTR(ctx, int, bs, seq_lens);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       batch_id_per_token,
                       cu_seqlens_q,
                       cu_seqlens_k,
                       x_remove_padding,
                       input_ids,
                       seq_lens,
                       max_seq_len,
                       bs);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        batch_id_per_token,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        x_remove_padding,
                        input_ids,
                        seq_lens,
                        max_seq_len,
                        bs);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace fastdeploy
