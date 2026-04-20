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
#include "paddle/phi/core/enforce.h"
#include "utility/helper.h"
#include "xpu/plugin.h"
#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

namespace api = baidu::xpu::api;

// api::fast_swiglu signature (from aten_api.h):
//   int fast_swiglu(Context* ctx, const T* x, T* y,
//                   const std::vector<int64_t>& xshape, int64_t axis,
//                   bool turn = false, const float* max_x = nullptr,
//                   float* max_y = nullptr);
//
// Input x: [rows, 2 * inter_dim]  (gate and up concatenated along last dim)
// Output:  [rows, inter_dim]
// fast_swiglu is in-place friendly: reads from x, writes to y.

template <typename T>
std::vector<paddle::Tensor> FastSwigluKernel(const paddle::Tensor& x) {
  using XPU_T = typename XPUTypeTrait<T>::Type;
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  auto x_shape = x.shape();
  PD_CHECK(x_shape.size() >= 2,
           "fast_swiglu_xpu: input must have at least 2 dimensions, got ",
           x_shape.size());
  int64_t last_dim = x_shape[x_shape.size() - 1];
  PD_CHECK(last_dim % 2 == 0,
           "fast_swiglu_xpu: last dim must be even, got ",
           last_dim);

  // Compute rows (product of all dims except last)
  int64_t rows = 1;
  for (size_t i = 0; i < x_shape.size() - 1; ++i) {
    rows *= x_shape[i];
  }
  int64_t inter_dim = last_dim / 2;

  // Output shape: same as input but last dim halved
  auto out_shape = x_shape;
  out_shape[out_shape.size() - 1] = inter_dim;
  paddle::Tensor out = paddle::empty(out_shape, x.dtype(), x.place());

  const XPU_T* x_data = reinterpret_cast<const XPU_T*>(x.data<T>());
  XPU_T* out_data = reinterpret_cast<XPU_T*>(const_cast<T*>(out.data<T>()));

  int ret = api::fast_swiglu<XPU_T>(
      xpu_ctx->x_context(), x_data, out_data, {rows, last_dim}, 1, true);
  PD_CHECK(ret == 0, "fast_swiglu_xpu: api::fast_swiglu failed, ret=", ret);

  return {out};
}

std::vector<paddle::Tensor> FastSwiglu(const paddle::Tensor& x) {
  const auto x_type = x.dtype();
  if (x_type == paddle::DataType::BFLOAT16) {
    return FastSwigluKernel<paddle::bfloat16>(x);
  } else if (x_type == paddle::DataType::FLOAT16) {
    return FastSwigluKernel<paddle::float16>(x);
  } else if (x_type == paddle::DataType::FLOAT32) {
    return FastSwigluKernel<float>(x);
  } else {
    PD_THROW("fast_swiglu_xpu: unsupported dtype=", static_cast<int>(x_type));
    return {};
  }
}

std::vector<std::vector<int64_t>> FastSwigluInferShape(
    const std::vector<int64_t>& x_shape) {
  auto out_shape = x_shape;
  out_shape[out_shape.size() - 1] /= 2;
  return {out_shape};
}

std::vector<paddle::DataType> FastSwigluInferDtype(
    const paddle::DataType& x_dtype) {
  return {x_dtype};
}

PD_BUILD_STATIC_OP(fast_swiglu_xpu)
    .Inputs({"x"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(FastSwiglu))
    .SetInferShapeFn(PD_INFER_SHAPE(FastSwigluInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FastSwigluInferDtype));
