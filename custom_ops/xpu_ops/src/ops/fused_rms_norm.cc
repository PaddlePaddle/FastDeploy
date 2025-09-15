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

#include <infer_ops.h>
#include <functional>
#include "paddle/extension.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "utility/debug.h"
#include "utility/env.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

XPU_DECLARE_BOOL(ENABLE_XVLLM_SDNN_INFER, false);
namespace api = baidu::xpu::api;

template <typename T>
std::vector<paddle::Tensor> RmsNormKernel(
    const paddle::Tensor& x,
    const paddle::optional<paddle::Tensor>& bias,
    const paddle::optional<paddle::Tensor>& residual,
    const paddle::Tensor& norm_weight,
    const paddle::optional<paddle::Tensor>& norm_bias,
    const float epsilon,
    const int begin_norm_axis,
    const float quant_scale,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound) {
  using XPU_T = typename XPUTypeTrait<T>::Type;
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  int ret = -1;
  auto x_shape = x.shape();
  PD_CHECK(quant_scale <= 0, "Quantization is not supported");
  PD_CHECK(begin_norm_axis > 0 && begin_norm_axis <= x_shape.size(),
           "begin_norm_axis check fail");
  PD_CHECK(norm_bias.get_ptr() == nullptr,
           "rms norm kernel don't support norm_bias");

  int64_t m = std::accumulate(x_shape.begin(),
                              x_shape.begin() + begin_norm_axis,
                              static_cast<int64_t>(1),
                              std::multiplies<int64_t>());
  int64_t n = std::accumulate(x_shape.begin() + begin_norm_axis,
                              x_shape.end(),
                              static_cast<int64_t>(1),
                              std::multiplies<int64_t>());

  PD_CHECK(n == norm_weight.shape()[0],
           "The product from begin_norm_axis to the last axis of x must be "
           "equal to the norm_weight's shape[0]");
  if (bias.get_ptr()) {
    PD_CHECK(n == bias.get_ptr()->shape()[0],
             "The product from begin_norm_axis to the last axis of x must be "
             "equal to the bias's shape[0]");
  }

  paddle::Tensor out = paddle::empty(x_shape, x.dtype(), x.place());
  paddle::Tensor residual_out = paddle::empty(x_shape, x.dtype(), x.place());
  const XPU_T* x_data = reinterpret_cast<const XPU_T*>(x.data<T>());
  const XPU_T* norm_weight_data =
      reinterpret_cast<const XPU_T*>(norm_weight.data<T>());
  const XPU_T* bias_data =
      bias.get_ptr() ? reinterpret_cast<const XPU_T*>(bias.get_ptr()->data<T>())
                     : nullptr;
  const XPU_T* residual_data =
      residual.get_ptr()
          ? reinterpret_cast<const XPU_T*>(residual.get_ptr()->data<T>())
          : nullptr;
  XPU_T* out_data = reinterpret_cast<XPU_T*>(const_cast<T*>(out.data<T>()));
  XPU_T* residual_out_data = nullptr;
  if (residual_data) {
    residual_out_data =
        reinterpret_cast<XPU_T*>(const_cast<T*>(residual_out.data<T>()));
  }

  XPU_T* add_out_data = const_cast<XPU_T*>(x_data);
  if (bias_data) {
    ret = api::broadcast_add(
        xpu_ctx->x_context(), x_data, bias_data, out_data, {m, n}, {n});
    PD_CHECK(ret == 0, "broadcast_add");
    add_out_data = out_data;
  }

  bool use_sdnn = FLAGS_ENABLE_XVLLM_SDNN_INFER;
  if (residual_data) {
    ret = infer_ops::add_rms_layer_norm<XPU_T, XPU_T>(xpu_ctx->x_context(),
                                                      add_out_data,
                                                      residual_data,
                                                      out_data,
                                                      m,
                                                      n,
                                                      epsilon,
                                                      norm_weight_data,
                                                      nullptr,
                                                      nullptr,
                                                      residual_out_data,
                                                      nullptr,
                                                      use_sdnn);
    PD_CHECK(ret == 0, "add_rms_layer_norm");
  } else {
    ret = api::rms_layer_norm<XPU_T, XPU_T>(xpu_ctx->x_context(),
                                            add_out_data,
                                            out_data,
                                            m,
                                            n,
                                            epsilon,
                                            norm_weight_data,
                                            nullptr,
                                            nullptr,
                                            false);
    PD_CHECK(ret == 0, "rms_layer_norm");
  }

  return {out, residual_out};
}

std::vector<paddle::Tensor> RmsNorm(
    const paddle::Tensor& x,
    const paddle::optional<paddle::Tensor>& bias,
    const paddle::optional<paddle::Tensor>& residual,
    const paddle::Tensor& norm_weight,
    const paddle::optional<paddle::Tensor>& norm_bias,
    const float epsilon,
    const int begin_norm_axis,
    const float quant_scale,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound) {
  const auto x_type = x.dtype();

#define APPLY_RMS_NORM_KERNEL(TX)            \
  return RmsNormKernel<TX>(x,                \
                           bias,             \
                           residual,         \
                           norm_weight,      \
                           norm_bias,        \
                           epsilon,          \
                           begin_norm_axis,  \
                           quant_scale,      \
                           quant_round_type, \
                           quant_max_bound,  \
                           quant_min_bound);

  if (x_type == paddle::DataType::BFLOAT16) {
    APPLY_RMS_NORM_KERNEL(paddle::bfloat16);
  } else if (x_type == paddle::DataType::FLOAT16) {
    APPLY_RMS_NORM_KERNEL(paddle::float16);
  } else if (x_type == paddle::DataType::FLOAT32) {
    APPLY_RMS_NORM_KERNEL(float);
  } else {
    PD_THROW("RmsNorm not support x_type=", static_cast<int>(x_type));
    return {};
  }
#undef APPLY_RMS_NORM_KERNEL
}

std::vector<std::vector<int64_t>> RmsNormInferShape(
    const std::vector<int64_t>& x_shape,
    const paddle::optional<std::vector<int64_t>>& bias_shape,
    const paddle::optional<std::vector<int64_t>>& residual_shape,
    const std::vector<int64_t>& norm_weight_shape,
    const paddle::optional<std::vector<int64_t>>& norm_bias_shape,
    const float epsilon,
    const int begin_norm_axis,
    const float quant_scale,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound) {
  PD_CHECK(begin_norm_axis > 0 && begin_norm_axis <= x_shape.size(),
           "begin_norm_axis check fail");
  int64_t m = std::accumulate(x_shape.begin(),
                              x_shape.begin() + begin_norm_axis,
                              static_cast<int64_t>(1),
                              std::multiplies<int64_t>());
  return {x_shape, x_shape, {m}};
}

std::vector<paddle::DataType> RmsNormInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::optional<paddle::DataType>& bias_dtype,
    const paddle::optional<paddle::DataType>& residual_dtype,
    const paddle::DataType& norm_weight_dtype,
    const paddle::optional<paddle::DataType>& norm_bias_dtype,
    const float epsilon,
    const int begin_norm_axis,
    const float quant_scale,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound) {
  // out, residual_out
  return {x_dtype, x_dtype};
}

PD_BUILD_STATIC_OP(fused_rms_norm_xpu)
    .Inputs({"x",
             paddle::Optional("bias"),
             paddle::Optional("residual"),
             "norm_weight",
             paddle::Optional("norm_bias")})
    .Outputs({"out", "residul_out"})
    .Attrs({"epsilon:float",
            "begin_norm_axis:int",
            "quant_scale:float",
            "quant_round_type:int",
            "quant_max_bound:float",
            "quant_min_bound:float"})
    .SetKernelFn(PD_KERNEL(RmsNorm))
    .SetInferShapeFn(PD_INFER_SHAPE(RmsNormInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RmsNormInferDtype));
