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

#include <core/check.h>
#include <core/context.h>
#include <core/param.h>
#include <infer_ops.h>
#include <xft_api.h>
#include "paddle/extension.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "utility/debug.h"
#include "utility/env.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

namespace xftblock = baidu::xpu::xftblock;
namespace api = baidu::xpu::api;

template <typename TX>
std::vector<paddle::Tensor> Quant2dPerTokenKernel(const paddle::Tensor& x) {
  using XPU_TX = typename XPUTypeTrait<TX>::Type;
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  xftblock::XFTContext xctx(xpu_ctx->x_context(), nullptr);
  auto rt_guard = xctx.get_rt_guard();

  auto input_shape = x.shape();
  auto x_scale =
      paddle::empty({input_shape[0]}, paddle::DataType::FLOAT32, x.place());
  auto quant_x = paddle::empty(
      {input_shape[0], input_shape[1]}, paddle::DataType::INT8, x.place());
  if (input_shape[0] > 0) {
    int ret = infer_ops::quant2d_per_token<XPU_TX, float, int8_t>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPU_TX*>(x.data<TX>()),
        nullptr,
        reinterpret_cast<int8_t*>(quant_x.data<int8_t>()),
        reinterpret_cast<float*>(x_scale.data<float>()),
        input_shape[0],
        input_shape[1]);
    PD_CHECK(ret == api::SUCCESS);
  }

  return {quant_x, x_scale};
}

std::vector<paddle::Tensor> Quant2dPerToken(const paddle::Tensor& x) {
  const auto x_type = x.dtype();
  if (x_type == paddle::DataType::BFLOAT16) {
    return Quant2dPerTokenKernel<paddle::bfloat16>(x);
  } else if (x_type == paddle::DataType::FLOAT16) {
    return Quant2dPerTokenKernel<paddle::float16>(x);
  } else {
    PD_THROW("Quant2dPerToken not support x_type=", static_cast<int>(x_type));
    return {};
  }
}

std::vector<std::vector<int64_t>> Quant2dPerTokenInferShape(
    const std::vector<int64_t>& x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> Quant2dPerTokenInferDtype(
    const paddle::DataType& x_dtype) {
  return {paddle::DataType::INT8};
}

PD_BUILD_STATIC_OP(quant2d_per_token)
    .Inputs({"x"})
    .Outputs({"quant_x", "x_scale"})
    .SetKernelFn(PD_KERNEL(Quant2dPerToken))
    .SetInferShapeFn(PD_INFER_SHAPE(Quant2dPerTokenInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(Quant2dPerTokenInferDtype));
