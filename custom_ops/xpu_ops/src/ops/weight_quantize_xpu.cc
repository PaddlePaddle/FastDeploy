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
#include <infer_ops_eb.h>
#include <paddle/extension.h>
#include <paddle/phi/backends/xpu/xpu_context.h>
#include "xpu/plugin.h"

template <typename T>
std::vector<paddle::Tensor> WeightQuantizeKernel(const paddle::Tensor &x,
                                                 const std::string &algo,
                                                 const int32_t arch,
                                                 const int32_t group_size) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
  int64_t k = x.shape()[0];
  int64_t n = x.shape()[1];

  paddle::Tensor scale =
      paddle::empty({n}, paddle::DataType::FLOAT32, x.place());
  if (algo == "weight_only_int8") {
    paddle::Tensor out =
        paddle::empty({k, n}, paddle::DataType::INT8, x.place());
    paddle::Tensor x_trans = paddle::empty({k, n}, x.dtype(), x.place());
    paddle::Tensor out_trans =
        paddle::empty({k, n}, paddle::DataType::INT8, x.place());
    XPUType *x_trans_ptr = const_cast<XPUType *>(
        reinterpret_cast<const XPUType *>(x_trans.data<T>()));
    int ret = baidu::xpu::api::transpose<XPUType>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType *>(x.data<T>()),
        x_trans_ptr,
        {k, n},
        {1, 0});
    PD_CHECK(ret == 0);
    ret = infer_ops::quant2d_per_token<XPUType, float, int8_t>(
        xpu_ctx->x_context(),
        x_trans_ptr,
        nullptr,
        out_trans.data<int8_t>(),
        scale.data<float>(),
        n,
        k);
    PD_CHECK(ret == 0);
    ret = baidu::xpu::api::transpose<int8_t>(xpu_ctx->x_context(),
                                             out_trans.data<int8_t>(),
                                             out.data<int8_t>(),
                                             {n, k},
                                             {1, 0});
    PD_CHECK(ret == 0);
    return {out, scale};
  } else if (algo == "weight_only_int4") {
    // TODO(mayang02): fix quant2d_per_channel int4 bugs, use transpose +
    // quant2d_per_token + transpose at now
    PD_CHECK(k % 2 == 0);
    paddle::Tensor out =
        paddle::empty({(k + 1) / 2, n}, paddle::DataType::INT8, x.place());
    paddle::Tensor x_trans = paddle::empty({k, n}, x.dtype(), x.place());
    paddle::Tensor out_trans =
        paddle::empty({(k + 1) / 2, n}, paddle::DataType::INT8, x.place());
    XPUType *x_trans_ptr = const_cast<XPUType *>(
        reinterpret_cast<const XPUType *>(x_trans.data<T>()));
    int ret = baidu::xpu::api::transpose<XPUType>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType *>(x.data<T>()),
        x_trans_ptr,
        {k, n},
        {1, 0});
    PD_CHECK(ret == 0);
    ret = infer_ops::quant2d_per_token<XPUType, float, int4_t>(
        xpu_ctx->x_context(),
        x_trans_ptr,
        nullptr,
        reinterpret_cast<int4_t *>(out_trans.data<int8_t>()),
        scale.data<float>(),
        n,
        k);
    PD_CHECK(ret == 0);
    ret = baidu::xpu::api::transpose<int8_t>(xpu_ctx->x_context(),
                                             out_trans.data<int8_t>(),
                                             out.data<int8_t>(),
                                             {n, k / 2},
                                             {1, 0});
    PD_CHECK(ret == 0);
    return {out, scale};
  } else {
    PD_THROW("Weight quantize only supports weight_only_int8 on XPU now.");
    return {};
  }
}

std::vector<paddle::Tensor> WeightQuantize(const paddle::Tensor &x,
                                           const std::string &algo,
                                           const int32_t arch,
                                           const int32_t group_size) {
  const auto x_type = x.dtype();
#define APPLY_WEIGHT_QUANTIZE_KERNEL(TX) \
  return WeightQuantizeKernel<TX>(x, algo, arch, group_size);

  if (x_type == paddle::DataType::BFLOAT16) {
    APPLY_WEIGHT_QUANTIZE_KERNEL(paddle::bfloat16);
  } else if (x_type == paddle::DataType::FLOAT32) {
    APPLY_WEIGHT_QUANTIZE_KERNEL(float);
  } else {
    PD_THROW("WeightQuantize not support x_type==%d", static_cast<int>(x_type));
    return {};
  }
}

std::vector<std::vector<int64_t>> WeightQuantizeInferShape(
    const std::vector<int64_t> &x_shape,
    const std::string &algo,
    const int32_t arch,
    const int32_t group_size) {
  if (algo == "weight_only_int8") {
    return {x_shape, {x_shape[1]}};
  } else if (algo == "weight_only_int4") {
    return {{x_shape[0] / 2, x_shape[1]}, {x_shape[1]}};
  } else {
    PD_THROW("weight_quantize not support algo=%s", algo);
  }
}

std::vector<paddle::DataType> WeightQuantizeInferDtype(
    const paddle::DataType &x_dtype,
    const std::string &algo,
    const int32_t arch,
    const int32_t group_size) {
  if (algo == "weight_only_int8") {
    return {paddle::DataType::INT8, paddle::DataType::FLOAT32};
  } else if (algo == "weight_only_int4") {
    return {paddle::DataType::INT8, paddle::DataType::FLOAT32};
  } else {
    PD_THROW("weight_quantize not support algo=%s", algo);
  }
}

PD_BUILD_OP(weight_quantize_xpu)
    .Inputs({"x"})
    .Outputs({"out", "scale"})
    .Attrs({"algo: std::string", "arch: int", "group_size: int"})
    .SetKernelFn(PD_KERNEL(WeightQuantize))
    .SetInferShapeFn(PD_INFER_SHAPE(WeightQuantizeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WeightQuantizeInferDtype));
