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
#include "xpu/refactor/impl/xdnn_impl.h"
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace fd_xpu3 {

__attribute__((global)) void speculate_preprocess_kernel(
    int64_t* ids_remove_padding,
    int* batch_id_per_token,
    int* cu_seqlens_q,
    int* cu_seqlens_k,
    int* seq_lens_output,
    int* cu_seq_lens_q_output,
    int* batch_id_per_token_output,
    int* real_output_token_num,
    const int64_t* input_data,
    const int* seq_lens,
    const int64_t* draft_tokens,
    const int* seq_lens_encoder,
    const int max_seq_len,
    const int max_draft_tokens_per_batch,
    const int real_bs);
}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

static int cpu_wrapper(api::Context* ctx,
                       int64_t* ids_remove_padding,
                       int* batch_id_per_token,
                       int* cu_seqlens_q,
                       int* cu_seqlens_k,
                       int* seq_lens_output,
                       int* cu_seq_lens_q_output,
                       int* batch_id_per_token_output,
                       int* real_output_token_num,
                       const int64_t* input_data,
                       const int* seq_lens,
                       const int64_t* draft_tokens,
                       const int* seq_lens_encoder,
                       const int max_seq_len,
                       const int max_draft_tokens_per_batch,
                       const int token_num_data,
                       const int real_bs) {
  cu_seqlens_q[0] = 0;
  cu_seqlens_k[0] = 0;
  for (int i = 0; i < real_bs; ++i) {
    const int seq_len = seq_lens[i];
    cu_seqlens_q[i + 1] = cu_seqlens_q[i] + seq_len;
    cu_seqlens_k[i + 1] = cu_seqlens_k[i] + seq_len;
  }

  for (int bi = 0; bi < real_bs; ++bi) {
    for (int i = 0; i < seq_lens[bi]; ++i) {
      const int tgt_seq_id = cu_seqlens_q[bi + 1] - seq_lens[bi] + i;
      if (max_draft_tokens_per_batch > 0 && seq_lens_encoder[bi] <= 0) {
        // speculative decoding
        const int src_seq_id = bi * max_draft_tokens_per_batch + i;
        ids_remove_padding[tgt_seq_id] = draft_tokens[src_seq_id];
      } else {
        // Non-speculative decoding
        const int src_seq_id = bi * max_seq_len + i;
        ids_remove_padding[tgt_seq_id] = input_data[src_seq_id];
      }
      batch_id_per_token[tgt_seq_id] = bi;
    }
  }

  for (int bid = 0; bid < real_bs; ++bid) {
    if (seq_lens[bid] == 0) {
      seq_lens_output[bid] = 0;
    } else if (seq_lens[bid] == 1) {
      seq_lens_output[bid] = 1;
    } else if (seq_lens_encoder[bid] != 0) {
      seq_lens_output[bid] = 1;
    } else {
      seq_lens_output[bid] = seq_lens[bid];
    }
  }

  cu_seq_lens_q_output[0] = 0;
  for (int i = 0; i < real_bs; ++i) {
    cu_seq_lens_q_output[i + 1] = cu_seq_lens_q_output[i] + seq_lens_output[i];
  }
  real_output_token_num[0] = cu_seq_lens_q_output[real_bs];

  for (int bi = 0; bi < real_bs; ++bi) {
    for (int i = 0; i < seq_lens_output[bi]; ++i) {
      const int tgt_seq_id_output =
          cu_seq_lens_q_output[bi + 1] - seq_lens_output[bi] + i;
      batch_id_per_token_output[tgt_seq_id_output] = bi;
    }
  }

  return api::SUCCESS;
}

static int xpu3_wrapper(api::Context* ctx,
                        int64_t* ids_remove_padding,
                        int* batch_id_per_token,
                        int* cu_seqlens_q,
                        int* cu_seqlens_k,
                        int* seq_lens_output,
                        int* cu_seq_lens_q_output,
                        int* batch_id_per_token_output,
                        int* real_output_token_num,
                        const int64_t* input_data,
                        const int* seq_lens,
                        const int64_t* draft_tokens,
                        const int* seq_lens_encoder,
                        const int max_seq_len,
                        const int max_draft_tokens_per_batch,
                        const int token_num_data,
                        const int real_bs) {
  using XPU_INT64 = typename api::XPUIndexType<int64_t>::type;
  int32_t ret_xre = fd_xpu3::
      speculate_preprocess_kernel<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          reinterpret_cast<XPU_INT64*>(ids_remove_padding),
          batch_id_per_token,
          cu_seqlens_q,
          cu_seqlens_k,
          seq_lens_output,
          cu_seq_lens_q_output,
          batch_id_per_token_output,
          real_output_token_num,
          reinterpret_cast<const XPU_INT64*>(input_data),
          seq_lens,
          reinterpret_cast<const XPU_INT64*>(draft_tokens),
          seq_lens_encoder,
          max_seq_len,
          max_draft_tokens_per_batch,
          real_bs);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int speculate_preprocess(api::Context* ctx,
                         int64_t* ids_remove_padding,
                         int* batch_id_per_token,
                         int* cu_seqlens_q,
                         int* cu_seqlens_k,
                         int* seq_lens_output,
                         int* cu_seq_lens_q_output,
                         int* batch_id_per_token_output,
                         int* real_output_token_num,
                         const int64_t* input_data,
                         const int* seq_lens,
                         const int64_t* draft_tokens,
                         const int* seq_lens_encoder,
                         const int max_seq_len,
                         const int max_draft_tokens_per_batch,
                         const int token_num_data,
                         const int real_bs) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_preprocess", int);
  WRAPPER_DUMP_PARAM6(ctx,
                      ids_remove_padding,
                      batch_id_per_token,
                      cu_seqlens_q,
                      cu_seqlens_k,
                      seq_lens_output,
                      cu_seq_lens_q_output);
  WRAPPER_DUMP_PARAM6(ctx,
                      batch_id_per_token_output,
                      real_output_token_num,
                      input_data,
                      seq_lens,
                      draft_tokens,
                      seq_lens_encoder);
  WRAPPER_DUMP_PARAM3(ctx, max_seq_len, max_draft_tokens_per_batch, real_bs);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, int64_t, token_num_data, ids_remove_padding);
  WRAPPER_CHECK_PTR(ctx, int, token_num_data, batch_id_per_token);
  WRAPPER_CHECK_PTR(ctx, int, real_bs + 1, cu_seqlens_q);
  WRAPPER_CHECK_PTR(ctx, int, real_bs + 1, cu_seqlens_k);
  WRAPPER_CHECK_PTR(ctx, int, real_bs, seq_lens_output);
  WRAPPER_CHECK_PTR(ctx, int, real_bs + 1, cu_seq_lens_q_output);
  WRAPPER_CHECK_PTR(
      ctx, int, real_bs* max_draft_tokens_per_batch, batch_id_per_token_output);
  WRAPPER_CHECK_PTR(ctx, int, 1, real_output_token_num);
  WRAPPER_CHECK_PTR(ctx, int64_t, real_bs * max_seq_len, input_data);
  WRAPPER_CHECK_PTR(ctx, int, real_bs, seq_lens);
  WRAPPER_CHECK_PTR(
      ctx, int, real_bs* max_draft_tokens_per_batch, draft_tokens);
  WRAPPER_CHECK_PTR(ctx, int, real_bs, seq_lens_encoder);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       ids_remove_padding,
                       batch_id_per_token,
                       cu_seqlens_q,
                       cu_seqlens_k,
                       seq_lens_output,
                       cu_seq_lens_q_output,
                       batch_id_per_token_output,
                       real_output_token_num,
                       input_data,
                       seq_lens,
                       draft_tokens,
                       seq_lens_encoder,
                       max_seq_len,
                       max_draft_tokens_per_batch,
                       token_num_data,
                       real_bs);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        ids_remove_padding,
                        batch_id_per_token,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        seq_lens_output,
                        cu_seq_lens_q_output,
                        batch_id_per_token_output,
                        real_output_token_num,
                        input_data,
                        seq_lens,
                        draft_tokens,
                        seq_lens_encoder,
                        max_seq_len,
                        max_draft_tokens_per_batch,
                        token_num_data,
                        real_bs);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace fastdeploy
