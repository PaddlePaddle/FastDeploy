// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

std::vector<paddle::Tensor> SpeculateGetPaddingOffset(
    const paddle::Tensor& input_ids,
    const paddle::Tensor& draft_tokens,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& token_num,
    const paddle::Tensor& seq_len,
    const paddle::Tensor& seq_lens_encoder) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  std::vector<int64_t> input_ids_shape = input_ids.shape();
  const int bsz = seq_len.shape()[0];
  const int seq_length = input_ids_shape[1];
  const int max_draft_tokens = draft_tokens.shape()[1];
  auto cum_offsets_out = cum_offsets.copy_to(cum_offsets.place(), false);
  auto cpu_token_num = token_num.copy_to(paddle::CPUPlace(), false);

  const int token_num_data = cpu_token_num.data<int64_t>()[0];
  auto x_remove_padding = paddle::empty(
      {token_num_data}, paddle::DataType::INT64, input_ids.place());
  auto padding_offset = paddle::empty(
      {token_num_data}, paddle::DataType::INT32, input_ids.place());
  auto batch_id_per_token = paddle::empty(
      {token_num_data}, paddle::DataType::INT32, input_ids.place());
  auto cu_seqlens_q =
      paddle::empty({bsz + 1}, paddle::DataType::INT32, input_ids.place());
  auto cu_seqlens_k =
      paddle::empty({bsz + 1}, paddle::DataType::INT32, input_ids.place());

  PD_CHECK(input_ids.is_contiguous(), "Input ids tensor must be contiguous");
  PD_CHECK(draft_tokens.is_contiguous(),
           "Draft tokens tensor must be contiguous");
  PD_CHECK(cum_offsets.is_contiguous(),
           "Cum offsets tensor must be contiguous");
  PD_CHECK(seq_len.is_contiguous(), "Seq lens tensor must be contiguous");

  int r = baidu::xpu::api::plugin::speculate_get_padding_offset(
      xpu_ctx->x_context(),
      batch_id_per_token.data<int>(),
      cum_offsets_out.data<int>(),
      cu_seqlens_q.data<int>(),
      cu_seqlens_k.data<int>(),
      cum_offsets.data<int>(),
      seq_len.data<int>(),
      seq_length,
      bsz);
  PD_CHECK(r == 0, "XPU speculate_get_padding_offset failed");

  r = baidu::xpu::api::plugin::speculate_remove_padding<int64_t>(
      xpu_ctx->x_context(),
      x_remove_padding.data<int64_t>(),
      input_ids.data<int64_t>(),
      draft_tokens.data<int64_t>(),
      seq_len.data<int>(),
      seq_lens_encoder.data<int>(),
      cum_offsets_out.data<int>(),
      seq_length,
      max_draft_tokens,
      bsz,
      token_num_data);
  PD_CHECK(r == 0, "XPU speculate_remove_padding failed");

  return {x_remove_padding,
          cum_offsets_out,
          batch_id_per_token,
          cu_seqlens_q,
          cu_seqlens_k};  // , enc_token_num, dec_token_num};
}

std::vector<std::vector<int64_t>> SpeculateGetPaddingOffsetInferShape(
    const std::vector<int64_t>& input_ids_shape,
    const std::vector<int64_t>& draft_tokens_shape,
    const std::vector<int64_t>& cum_offsets_shape,
    const std::vector<int64_t>& token_num_shape,
    const std::vector<int64_t>& seq_len_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape) {
  int64_t bsz = seq_len_shape[0];
  int64_t seq_len = input_ids_shape[1];
  return {{-1}, {bsz}, {-1}, {bsz + 1}, {bsz + 1}};
}

std::vector<paddle::DataType> SpeculateGetPaddingOffsetInferDtype(
    const paddle::DataType& input_ids_dtype,
    const paddle::DataType& draft_tokens_dtype,
    const paddle::DataType& cum_offsets_dtype,
    const paddle::DataType& token_num_dtype,
    const paddle::DataType& seq_len_dtype,
    const paddle::DataType& seq_lens_encoder_dtype) {
  return {input_ids_dtype,
          seq_len_dtype,
          seq_len_dtype,
          seq_len_dtype,
          seq_len_dtype};
}

PD_BUILD_STATIC_OP(speculate_get_padding_offset)
    .Inputs({"input_ids",
             "draft_tokens",
             "cum_offsets",
             "token_num",
             "seq_len",
             "seq_lens_encoder"})
    .Outputs({"x_remove_padding",
              "cum_offsets_out",
              "batch_id_per_token",
              "cu_seqlens_q",
              "cu_seqlens_k"})
    .SetKernelFn(PD_KERNEL(SpeculateGetPaddingOffset))
    .SetInferShapeFn(PD_INFER_SHAPE(SpeculateGetPaddingOffsetInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SpeculateGetPaddingOffsetInferDtype));
