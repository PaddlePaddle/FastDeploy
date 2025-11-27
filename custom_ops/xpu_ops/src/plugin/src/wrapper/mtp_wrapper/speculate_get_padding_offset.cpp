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

namespace xpu3 {
namespace plugin {

template <typename T>
__attribute__((global)) void speculate_remove_padding(
    T* output_data,
    const T* input_data,
    const T* draft_tokens,
    const int* seq_lens,
    const int* seq_lens_encoder,
    const int* cum_offsets,
    int sequence_length,
    int max_draft_tokens,
    int bsz,
    int token_num_data);

__attribute__((global)) void speculate_get_padding_offset(
    int* batch_id_per_token,
    int* cum_offsets_out,
    int* cu_seqlens_q,
    int* cu_seqlens_k,
    const int* cum_offsets,
    const int* seq_lens,
    const int max_seq_len,
    int bsz);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T>
static int cpu_wrapper_remove_padding(Context* ctx,
                                      T* output_data,
                                      const T* input_data,
                                      const T* draft_tokens,
                                      const int* seq_lens,
                                      const int* seq_lens_encoder,
                                      const int* cum_offsets,
                                      int sequence_length,
                                      int max_draft_tokens,
                                      int bsz,
                                      int token_num_data) {
  for (int bi = 0; bi < bsz; ++bi) {
    for (int i = 0; i < seq_lens[bi]; i++) {
      const int tgt_seq_id = bi * sequence_length - cum_offsets[bi] + i;
      if (seq_lens_encoder[bi] > 0) {
        const int src_seq_id = bi * sequence_length + i;
        output_data[tgt_seq_id] = input_data[src_seq_id];
      } else {
        const int src_seq_id = bi * max_draft_tokens + i;
        output_data[tgt_seq_id] = draft_tokens[src_seq_id];
      }
    }
  }
  return api::SUCCESS;
}

static int cpu_wrapper_get_padding_offset(Context* ctx,
                                          int* batch_id_per_token,
                                          int* cum_offsets_out,
                                          int* cu_seqlens_q,
                                          int* cu_seqlens_k,
                                          const int* cum_offsets,
                                          const int* seq_lens,
                                          const int max_seq_len,
                                          int bsz) {
  for (int bi = 0; bi < bsz; ++bi) {
    int cum_offset = bi == 0 ? 0 : cum_offsets[bi - 1];
    for (int i = 0; i < seq_lens[bi]; i++) {
      batch_id_per_token[bi * max_seq_len - cum_offset + i] = bi;
    }
    cum_offsets_out[bi] = cum_offset;
    int cum_seq_len = (bi + 1) * max_seq_len - cum_offsets[bi];
    cu_seqlens_q[bi + 1] = cum_seq_len;
    cu_seqlens_k[bi + 1] = cum_seq_len;
  }
  return api::SUCCESS;
}

template <typename T>
static int xpu3_wrapper_remove_padding(Context* ctx,
                                       T* output_data,
                                       const T* input_data,
                                       const T* draft_tokens,
                                       const int* seq_lens,
                                       const int* seq_lens_encoder,
                                       const int* cum_offsets,
                                       int sequence_length,
                                       int max_draft_tokens,
                                       int bsz,
                                       int token_num_data) {
  using XPU_T = typename XPUIndexType<T>::type;
  xpu3::plugin::speculate_remove_padding<XPU_T>
      <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          static_cast<XPU_T*>(static_cast<void*>(output_data)),
          static_cast<const XPU_T*>(static_cast<const void*>(input_data)),
          static_cast<const XPU_T*>(static_cast<const void*>(draft_tokens)),
          seq_lens,
          seq_lens_encoder,
          cum_offsets,
          sequence_length,
          max_draft_tokens,
          bsz,
          token_num_data);

  return api::SUCCESS;
}

static int xpu3_wrapper_get_padding_offset(Context* ctx,
                                           int* batch_id_per_token,
                                           int* cum_offsets_out,
                                           int* cu_seqlens_q,
                                           int* cu_seqlens_k,
                                           const int* cum_offsets,
                                           const int* seq_lens,
                                           const int max_seq_len,
                                           int bsz) {
  xpu3::plugin::
      speculate_get_padding_offset<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          batch_id_per_token,
          cum_offsets_out,
          cu_seqlens_q,
          cu_seqlens_k,
          cum_offsets,
          seq_lens,
          max_seq_len,
          bsz);
  return api::SUCCESS;
}

template <typename T>
int speculate_remove_padding(Context* ctx,
                             T* x_remove_padding,
                             const T* input_ids,
                             const T* draft_tokens,
                             const int* seq_lens,
                             const int* seq_lens_encoder,
                             const int* cum_offsets_out,
                             int seq_length,
                             int max_draft_tokens,
                             int bsz,
                             int token_num_data) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_remove_padding", T);
  WRAPPER_DUMP_PARAM6(ctx,
                      x_remove_padding,
                      input_ids,
                      draft_tokens,
                      seq_lens,
                      seq_lens_encoder,
                      cum_offsets_out);
  WRAPPER_DUMP_PARAM4(ctx, seq_length, max_draft_tokens, bsz, token_num_data);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, T, bsz * seq_length, input_ids);
  WRAPPER_CHECK_PTR(ctx, T, bsz * max_draft_tokens, draft_tokens);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_encoder);
  WRAPPER_CHECK_PTR(ctx, int, bsz, cum_offsets_out);
  WRAPPER_CHECK_PTR(ctx, T, token_num_data, x_remove_padding);

  WRAPPER_ASSERT_GT(ctx, bsz, 0);
  WRAPPER_ASSERT_GT(ctx, seq_length, 0);
  WRAPPER_ASSERT_GT(ctx, max_draft_tokens, 0);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper_remove_padding(ctx,
                                      x_remove_padding,
                                      input_ids,
                                      draft_tokens,
                                      seq_lens,
                                      seq_lens_encoder,
                                      cum_offsets_out,
                                      seq_length,
                                      max_draft_tokens,
                                      bsz,
                                      token_num_data);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper_remove_padding(ctx,
                                       x_remove_padding,
                                       input_ids,
                                       draft_tokens,
                                       seq_lens,
                                       seq_lens_encoder,
                                       cum_offsets_out,
                                       seq_length,
                                       max_draft_tokens,
                                       bsz,
                                       token_num_data);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

int speculate_get_padding_offset(Context* ctx,
                                 int* batch_id_per_token,
                                 int* cum_offsets_out,
                                 int* cu_seqlens_q,
                                 int* cu_seqlens_k,
                                 const int* cum_offsets,
                                 const int* seq_lens,
                                 const int max_seq_len,
                                 int bsz) {
  WRAPPER_CHECK_CTX(ctx);

  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_get_padding_offset", float);
  WRAPPER_DUMP_PARAM6(ctx,
                      batch_id_per_token,
                      cum_offsets_out,
                      cu_seqlens_q,
                      cu_seqlens_k,
                      cum_offsets,
                      seq_lens);
  WRAPPER_DUMP_PARAM2(ctx, max_seq_len, bsz);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, int, bsz, cum_offsets);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens);
  WRAPPER_CHECK_PTR(ctx, int, bsz, cum_offsets_out);
  WRAPPER_CHECK_PTR(ctx, int, bsz + 1, cu_seqlens_q);
  WRAPPER_CHECK_PTR(ctx, int, bsz + 1, cu_seqlens_k);

  WRAPPER_ASSERT_GT(ctx, bsz, 0);
  WRAPPER_ASSERT_GT(ctx, max_seq_len, 0);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper_get_padding_offset(ctx,
                                          batch_id_per_token,
                                          cum_offsets_out,
                                          cu_seqlens_q,
                                          cu_seqlens_k,
                                          cum_offsets,
                                          seq_lens,
                                          max_seq_len,
                                          bsz);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper_get_padding_offset(ctx,
                                           batch_id_per_token,
                                           cum_offsets_out,
                                           cu_seqlens_q,
                                           cu_seqlens_k,
                                           cum_offsets,
                                           seq_lens,
                                           max_seq_len,
                                           bsz);
  }

  WRAPPER_UNIMPLEMENTED(ctx);
}

#define INSTANTIATION_SPECULATE_REMOVE_PADDING(T)                       \
  template int speculate_remove_padding<T>(Context * ctx,               \
                                           T * x_remove_padding,        \
                                           const T* input_ids,          \
                                           const T* draft_tokens,       \
                                           const int* seq_len,          \
                                           const int* seq_lens_encoder, \
                                           const int* cum_offsets_out,  \
                                           int seq_length,              \
                                           int max_draft_tokens,        \
                                           int bsz,                     \
                                           int token_num_data)

INSTANTIATION_SPECULATE_REMOVE_PADDING(float);
INSTANTIATION_SPECULATE_REMOVE_PADDING(float16);
INSTANTIATION_SPECULATE_REMOVE_PADDING(bfloat16);
INSTANTIATION_SPECULATE_REMOVE_PADDING(int64_t);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
