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

std::vector<paddle::Tensor> BuildSamplingParams(
    const paddle::Tensor& top_p,
    const paddle::Tensor& top_k,
    paddle::Tensor& infer_seed,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const int64_t token_num_output_cpu,
    const int64_t increment_value) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  api::Context* ctx = xpu_ctx->x_context();
  std::unique_ptr<api::Context> cpu_ctx;
  if (top_p.is_cpu()) {
    cpu_ctx = std::make_unique<api::Context>(api::kCPU);
    ctx = cpu_ctx.get();
  }

  int real_bsz = static_cast<int>(seq_lens_this_time.shape()[0]);

  auto top_p_padding = paddle::empty(
      {token_num_output_cpu, 1}, paddle::DataType::FLOAT32, top_p.place());
  auto top_k_padding = paddle::empty(
      {token_num_output_cpu, 1}, paddle::DataType::INT64, top_p.place());
  auto topp_seed = paddle::empty(
      {token_num_output_cpu, 1}, paddle::DataType::INT64, top_p.place());

  int r =
      fastdeploy::plugin::build_sampling_params(ctx,
                                                top_p_padding.data<float>(),
                                                top_k_padding.data<int64_t>(),
                                                topp_seed.data<int64_t>(),
                                                top_p.data<float>(),
                                                top_k.data<int64_t>(),
                                                infer_seed.data<int64_t>(),
                                                seq_lens_this_time.data<int>(),
                                                seq_lens_encoder.data<int>(),
                                                real_bsz,
                                                token_num_output_cpu,
                                                increment_value);
  PD_CHECK(r == 0, "fastdeploy::plugin::build_sampling_params failed.");

  return {top_p_padding, top_k_padding, topp_seed};
}

std::vector<std::vector<int64_t>> BuildSamplingParamsInferShape(
    const std::vector<int64_t>& top_p_shape,
    const std::vector<int64_t>& top_k_shape,
    const std::vector<int64_t>& infer_seed_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape) {
  // token_num is dynamic; return a placeholder shape of [-1, 1]
  return {{-1, 1}, {-1, 1}, {-1, 1}};
}

std::vector<paddle::DataType> BuildSamplingParamsInferDtype(
    const paddle::DataType& top_p_dtype,
    const paddle::DataType& top_k_dtype,
    const paddle::DataType& infer_seed_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& seq_lens_encoder_dtype) {
  return {paddle::DataType::FLOAT32,
          paddle::DataType::INT64,
          paddle::DataType::INT64};
}

PD_BUILD_STATIC_OP(build_sampling_params)
    .Inputs({"top_p",
             "top_k",
             "infer_seed",
             "seq_lens_this_time",
             "seq_lens_encoder"})
    .Outputs({"top_p_padding", "top_k_padding", "topp_seed"})
    .Attrs({"token_num_output_cpu: int64_t", "increment_value: int64_t"})
    .SetKernelFn(PD_KERNEL(BuildSamplingParams))
    .SetInferShapeFn(PD_INFER_SHAPE(BuildSamplingParamsInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(BuildSamplingParamsInferDtype));
