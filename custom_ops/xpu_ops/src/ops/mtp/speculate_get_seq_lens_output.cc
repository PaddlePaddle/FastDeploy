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

std::vector<paddle::Tensor> SpeculateGetSeqLensOutput(
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  baidu::xpu::api::Context* ctx =
      static_cast<const phi::XPUContext*>(dev_ctx)->x_context();

  if (seq_lens_this_time.is_cpu()) {
    ctx = new baidu::xpu::api::Context(baidu::xpu::api::kCPU);
  }
  std::vector<int64_t> seq_lens_this_time_shape = seq_lens_this_time.shape();
  const int bsz = seq_lens_this_time_shape[0];

  auto seq_lens_output = paddle::full(
      {bsz}, 0, paddle::DataType::INT32, seq_lens_this_time.place());

  int r = baidu::xpu::api::plugin::speculate_get_seq_lens_output(
      ctx,
      seq_lens_output.data<int>(),
      seq_lens_this_time.data<int>(),
      seq_lens_encoder.data<int>(),
      seq_lens_decoder.data<int>(),
      bsz);
  PD_CHECK(r == 0, "speculate_get_seq_lens_output  failed.");

  return {seq_lens_output};
}

std::vector<std::vector<int64_t>> SpeculateGetSeqLensOutputInferShape(
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape) {
  int64_t bsz = seq_lens_this_time_shape[0];
  return {{bsz}};
}

std::vector<paddle::DataType> SpeculateGetSeqLensOutputInferDtype(
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype) {
  return {seq_lens_this_time_dtype};
}

PD_BUILD_STATIC_OP(speculate_get_seq_lens_output)
    .Inputs({"seq_lens_this_time", "seq_lens_encoder", "seq_lens_decoder"})
    .Outputs({"seq_lens_output"})
    .SetKernelFn(PD_KERNEL(SpeculateGetSeqLensOutput))
    .SetInferShapeFn(PD_INFER_SHAPE(SpeculateGetSeqLensOutputInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SpeculateGetSeqLensOutputInferDtype));
