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

#include <paddle/phi/backends/xpu/xpu_context.h>
#include "paddle/extension.h"
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

namespace api = baidu::xpu::api;

std::vector<paddle::Tensor> SpeculatePreProcess(
    const int64_t cpu_token_num,
    const paddle::Tensor &input_ids,
    const paddle::Tensor &seq_len,
    const paddle::Tensor &draft_tokens,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_decoder) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
  api::Context *ctx = xpu_ctx->x_context();

  // just for ut to run base line
  std::unique_ptr<baidu::xpu::api::Context> cpu_ctx;
  if (input_ids.place().GetType() == phi::AllocationType::CPU) {
    cpu_ctx = std::make_unique<baidu::xpu::api::Context>(baidu::xpu::api::kCPU);
    ctx = cpu_ctx.get();
  }

  std::vector<int64_t> input_ids_shape = input_ids.shape();
  const int bsz = seq_len.shape()[0];
  const int max_seq_len = input_ids_shape[1];
  const int token_num_data = cpu_token_num;
  auto ids_remove_padding = paddle::empty(
      {token_num_data}, paddle::DataType::INT64, input_ids.place());
  auto batch_id_per_token = paddle::empty(
      {token_num_data}, paddle::DataType::INT32, input_ids.place());
  auto cu_seqlens_q =
      paddle::empty({bsz + 1}, paddle::DataType::INT32, input_ids.place());
  auto cu_seqlens_k =
      paddle::empty({bsz + 1}, paddle::DataType::INT32, input_ids.place());
  const int max_draft_tokens_per_batch = draft_tokens.shape()[1];

  auto seq_lens_output =
      paddle::empty({bsz}, paddle::DataType::INT32, input_ids.place());
  auto cu_seq_lens_q_output =
      paddle::empty({bsz + 1}, paddle::DataType::INT32, input_ids.place());
  auto batch_id_per_token_output =
      paddle::empty({bsz * max_draft_tokens_per_batch},
                    paddle::DataType::INT32,
                    input_ids.place());
  auto real_output_token_num =
      paddle::empty({1}, paddle::DataType::INT32, input_ids.place());
  if (token_num_data == 0) {
    return {ids_remove_padding,
            batch_id_per_token,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_seq_lens_q_output,
            batch_id_per_token_output,
            real_output_token_num};
  }

  int64_t *ids_remove_padding_ptr = ids_remove_padding.data<int64_t>();
  int *batch_id_per_token_ptr = batch_id_per_token.data<int>();
  int *cu_seqlens_q_ptr = cu_seqlens_q.data<int>();
  int *cu_seqlens_k_ptr = cu_seqlens_k.data<int>();
  int *seq_lens_output_ptr = seq_lens_output.data<int>();
  int *cu_seq_lens_q_output_ptr = cu_seq_lens_q_output.data<int>();
  int *batch_id_per_token_output_ptr = batch_id_per_token_output.data<int>();
  int *real_output_token_num_ptr = real_output_token_num.data<int>();
  const int64_t *input_data_ptr = input_ids.data<int64_t>();
  const int *seq_len_ptr = seq_len.data<int>();
  const int64_t *draft_tokens_ptr = draft_tokens.data<int64_t>();
  const int *seq_lens_encoder_ptr = seq_lens_encoder.data<int>();

  int r =
      fastdeploy::plugin::speculate_preprocess(ctx,
                                               ids_remove_padding_ptr,
                                               batch_id_per_token_ptr,
                                               cu_seqlens_q_ptr,
                                               cu_seqlens_k_ptr,
                                               seq_lens_output_ptr,
                                               cu_seq_lens_q_output_ptr,
                                               batch_id_per_token_output_ptr,
                                               real_output_token_num_ptr,
                                               input_data_ptr,
                                               seq_len_ptr,
                                               draft_tokens_ptr,
                                               seq_lens_encoder_ptr,
                                               max_seq_len,
                                               max_draft_tokens_per_batch,
                                               token_num_data,
                                               bsz);

  return {ids_remove_padding,
          batch_id_per_token,
          cu_seqlens_q,
          cu_seqlens_k,
          cu_seq_lens_q_output,
          batch_id_per_token_output,
          real_output_token_num};
}

PD_BUILD_STATIC_OP(speculate_pre_process)
    .Inputs({"input_ids",
             "seq_len",
             "draft_tokens",
             "seq_lens_encoder",
             "seq_lens_decoder"})
    .Outputs({"ids_remove_padding",
              "batch_id_per_token",
              "cu_seqlens_q",
              "cu_seqlens_k",
              "cu_seq_lens_q_output",
              "batch_id_per_token_output",
              "real_output_token_num"})
    .Attrs({"cpu_token_num: int64_t"})
    .SetKernelFn(PD_KERNEL(SpeculatePreProcess));
