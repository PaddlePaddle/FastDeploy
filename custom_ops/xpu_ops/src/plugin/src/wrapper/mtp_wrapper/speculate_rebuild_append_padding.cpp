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
template <typename T>
__attribute__((global)) void RebuildAppendPaddingKernel(
    const T* full_hidden_states,
    const int* cum_offsets,
    const int* seq_len_encoder,
    const int* seq_len_decoder,
    const int* output_padding_offset,
    int max_seq_len,
    int dim_embed,
    int elem_nums,
    T* out);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T>
static int cpu_wrapper(Context* ctx,
                       T* full_hidden_states,
                       int* cum_offsets,
                       int* seq_len_encoder,
                       int* seq_len_decoder,
                       int* output_padding_offset,
                       int max_seq_len,
                       int dim_embed,
                       int elem_nums,
                       T* out) {
  for (int64_t i = 0; i < elem_nums; ++i) {
    int64_t out_token_id = i / dim_embed;
    int64_t ori_token_id = out_token_id + output_padding_offset[out_token_id];
    int64_t bi = ori_token_id / max_seq_len;

    int64_t seq_id = 0;
    if (seq_len_decoder[bi] == 0 && seq_len_encoder[bi] == 0) {
      continue;
    } else if (seq_len_encoder[bi] != 0) {
      seq_id = seq_len_encoder[bi] - 1;
    }

    int64_t input_token_id = ori_token_id - cum_offsets[bi] + seq_id;
    int64_t bias_idx = i % dim_embed;

    out[i] = full_hidden_states[input_token_id * dim_embed + bias_idx];
  }
  return api::SUCCESS;
}

template <typename T>
static int xpu3_wrapper(Context* ctx,
                        T* full_hidden_states,
                        int* cum_offsets,
                        int* seq_len_encoder,
                        int* seq_len_decoder,
                        int* output_padding_offset,
                        int max_seq_len,
                        int dim_embed,
                        int elem_nums,
                        T* out) {
  xpu3::plugin::RebuildAppendPaddingKernel<T>
      <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(full_hidden_states,
                                                 cum_offsets,
                                                 seq_len_encoder,
                                                 seq_len_decoder,
                                                 output_padding_offset,
                                                 max_seq_len,
                                                 dim_embed,
                                                 elem_nums,
                                                 out);
  return api::SUCCESS;
}

template <typename T>
int speculate_rebuild_append_padding(Context* ctx,
                                     T* full_hidden_states,
                                     int* cum_offsets,
                                     int* seq_len_encoder,
                                     int* seq_len_decoder,
                                     int* output_padding_offset,
                                     int max_seq_len,
                                     int dim_embed,
                                     int elem_nums,
                                     T* out) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_rebuild_append_padding", T);
  WRAPPER_DUMP_PARAM5(ctx,
                      full_hidden_states,
                      cum_offsets,
                      seq_len_encoder,
                      seq_len_decoder,
                      output_padding_offset);
  WRAPPER_DUMP_PARAM4(ctx, max_seq_len, dim_embed, elem_nums, out);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, T, elem_nums, out);
  WRAPPER_ASSERT_GT(ctx, max_seq_len, 0);
  WRAPPER_ASSERT_GT(ctx, dim_embed, 0);
  WRAPPER_ASSERT_GT(ctx, elem_nums, 0);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T>(ctx,
                          full_hidden_states,
                          cum_offsets,
                          seq_len_encoder,
                          seq_len_decoder,
                          output_padding_offset,
                          max_seq_len,
                          dim_embed,
                          elem_nums,
                          out);
  } else if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper<T>(ctx,
                           full_hidden_states,
                           cum_offsets,
                           seq_len_encoder,
                           seq_len_decoder,
                           output_padding_offset,
                           max_seq_len,
                           dim_embed,
                           elem_nums,
                           out);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int speculate_rebuild_append_padding(
    Context*, bfloat16*, int*, int*, int*, int*, int, int, int, bfloat16*);
template int speculate_rebuild_append_padding(
    Context*, float16*, int*, int*, int*, int*, int, int, int, float16*);
template int speculate_rebuild_append_padding(
    Context*, float*, int*, int*, int*, int*, int, int, int, float*);
}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
