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

#include <paddle/phi/backends/xpu/xpu_context.h>
#include "paddle/extension.h"
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

std::vector<paddle::Tensor> GetPaddingOffset(
    const paddle::Tensor& input_ids,
    const paddle::Tensor& seq_len,
    const paddle::optional<paddle::Tensor>& draft_tokens,
    const paddle::optional<paddle::Tensor>& seq_lens_encoder,
    const int64_t cpu_token_num) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  baidu::xpu::api::Context* ctx = xpu_ctx->x_context();

  std::unique_ptr<baidu::xpu::api::Context> cpu_ctx_guard;
  if (input_ids.is_cpu()) {
    cpu_ctx_guard.reset(new baidu::xpu::api::Context(baidu::xpu::api::kCPU));
    ctx = cpu_ctx_guard.get();
  }

  std::vector<int64_t> input_ids_shape = input_ids.shape();
  const int bsz = seq_len.shape()[0];
  const int max_seq_len = input_ids_shape[1];
  const int token_num_data = static_cast<int>(cpu_token_num);

  auto x_remove_padding = paddle::full(
      {token_num_data}, 2, paddle::DataType::INT64, input_ids.place());
  auto batch_id_per_token = paddle::full(
      {token_num_data}, -1, paddle::DataType::INT32, input_ids.place());
  auto cu_seqlens_q =
      paddle::full({bsz + 1}, 0, paddle::DataType::INT32, input_ids.place());
  auto cu_seqlens_k =
      paddle::full({bsz + 1}, 0, paddle::DataType::INT32, input_ids.place());

  if (token_num_data > 0) {
    if (draft_tokens || seq_lens_encoder) {
      // TODO(chenhuan09) : support speculate mode
      PD_THROW("Draft tokens are not supported on XPU currently.");
    }
    int r =
        fastdeploy::plugin::get_padding_offset(ctx,
                                               batch_id_per_token.data<int>(),
                                               cu_seqlens_q.data<int>(),
                                               cu_seqlens_k.data<int>(),
                                               x_remove_padding.data<int64_t>(),
                                               input_ids.data<int64_t>(),
                                               seq_len.data<int>(),
                                               max_seq_len,
                                               bsz,
                                               token_num_data);
    PD_CHECK(r == 0, "fastdeploy::plugin::get_padding_offset failed.");
  }
  return {x_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k};
}

std::vector<std::vector<int64_t>> GetPaddingOffsetInferShape(
    const std::vector<int64_t>& input_ids_shape,
    const std::vector<int64_t>& seq_len_shape,
    const std::vector<int64_t>& draft_tokens_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape) {
  int64_t bsz = seq_len_shape[0];
  return {{-1}, {-1}, {bsz + 1}, {bsz + 1}};
}

std::vector<paddle::DataType> GetPaddingOffsetInferDtype(
    const paddle::DataType& input_ids_dtype,
    const paddle::DataType& seq_len_dtype,
    const paddle::DataType& draft_tokens_dtype,
    const paddle::DataType& seq_lens_encoder_dtype) {
  return {input_ids_dtype, seq_len_dtype, seq_len_dtype, seq_len_dtype};
}

PD_BUILD_STATIC_OP(get_padding_offset)
    .Inputs({"input_ids",
             "seq_len",
             paddle::Optional("draft_tokens"),
             paddle::Optional("seq_lens_encoder")})
    .Outputs({"x_remove_padding",
              "batch_id_per_token",
              "cu_seqlens_q",
              "cu_seqlens_k"})
    .Attrs({"cpu_token_num: int64_t"})
    .SetKernelFn(PD_KERNEL(GetPaddingOffset))
    .SetInferShapeFn(PD_INFER_SHAPE(GetPaddingOffsetInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetPaddingOffsetInferDtype));
