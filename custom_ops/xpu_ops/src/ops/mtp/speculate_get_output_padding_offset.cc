// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

std::vector<paddle::Tensor> SpeculateGetOutputPaddingOffset(
    const paddle::Tensor& output_cum_offsets_tmp,
    const paddle::Tensor& out_token_num,
    const paddle::Tensor& seq_lens_output,
    const int max_seq_len) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  baidu::xpu::api::Context* ctx =
      static_cast<const phi::XPUContext*>(dev_ctx)->x_context();

  if (output_cum_offsets_tmp.is_cpu()) {
    ctx = new baidu::xpu::api::Context(baidu::xpu::api::kCPU);
  }
  std::vector<int64_t> output_cum_offsets_tmp_shape =
      output_cum_offsets_tmp.shape();
  const int bsz = output_cum_offsets_tmp_shape[0];
  auto cpu_out_token_num = out_token_num.copy_to(paddle::CPUPlace(), false);

  auto output_padding_offset = paddle::full({cpu_out_token_num},
                                            0,
                                            paddle::DataType::INT32,
                                            output_cum_offsets_tmp.place());
  auto output_cum_offsets =
      output_cum_offsets_tmp.copy_to(output_cum_offsets_tmp.place(), false);

  int r = baidu::xpu::api::plugin::speculate_get_output_padding_offset(
      ctx,
      output_padding_offset.mutable_data<int>(),
      output_cum_offsets.mutable_data<int>(),
      output_cum_offsets_tmp.data<int>(),
      seq_lens_output.data<int>(),
      bsz,
      max_seq_len);
  PD_CHECK(r == 0, "speculate_clear_accept_nums_kernel  failed.");

  return {output_padding_offset, output_cum_offsets};
}

std::vector<std::vector<int64_t>> SpeculateGetOutputPaddingOffsetInferShape(
    const std::vector<int64_t>& output_cum_offsets_tmp_shape,
    const std::vector<int64_t>& out_token_num_shape,
    const std::vector<int64_t>& seq_lens_output_shape) {
  int64_t bsz = output_cum_offsets_tmp_shape[0];
  return {{-1}, {bsz}};
}

std::vector<paddle::DataType> SpeculateGetOutputPaddingOffsetInferDtype(
    const paddle::DataType& output_cum_offsets_tmp_dtype,
    const paddle::DataType& out_token_num_dtype,
    const paddle::DataType& seq_lens_output_dtype) {
  return {output_cum_offsets_tmp_dtype, output_cum_offsets_tmp_dtype};
}

PD_BUILD_STATIC_OP(speculate_get_output_padding_offset)
    .Inputs({"output_cum_offsets_tmp", "out_token_num", "seq_lens_output"})
    .Outputs({"output_padding_offset", "output_cum_offsets"})
    .Attrs({"max_seq_len: int"})
    .SetKernelFn(PD_KERNEL(SpeculateGetOutputPaddingOffset))
    .SetInferShapeFn(PD_INFER_SHAPE(SpeculateGetOutputPaddingOffsetInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SpeculateGetOutputPaddingOffsetInferDtype));
