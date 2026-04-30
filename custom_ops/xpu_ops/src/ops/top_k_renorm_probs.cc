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

namespace api = baidu::xpu::api;

std::vector<paddle::Tensor> TopKRenorm(const paddle::Tensor& probs,
                                       const paddle::Tensor& top_k) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  api::Context* ctx = xpu_ctx->x_context();
  if (probs.is_cpu()) {
    ctx = new api::Context(api::kCPU);
  }

  const int batch_size = probs.shape()[0];
  const int vocab_size = probs.shape()[1];

  auto renorm_probs =
      paddle::empty(probs.shape(), probs.dtype(), probs.place());

  int r = 0;
  switch (probs.dtype()) {
    case paddle::DataType::BFLOAT16: {
      using XPUTypeBF16 = typename XPUTypeTrait<bfloat16>::Type;
      typedef paddle::bfloat16 bf16_data_t;
      r = fastdeploy::plugin::top_k_renorm_probs<XPUTypeBF16>(
          ctx,
          reinterpret_cast<const XPUTypeBF16*>(probs.data<bf16_data_t>()),
          reinterpret_cast<XPUTypeBF16*>(renorm_probs.data<bf16_data_t>()),
          top_k.data<int64_t>(),
          batch_size,
          vocab_size);
      break;
    }
    case paddle::DataType::FLOAT16: {
      using XPUTypeFP16 = typename XPUTypeTrait<float16>::Type;
      typedef paddle::float16 fp16_data_t;
      r = fastdeploy::plugin::top_k_renorm_probs<XPUTypeFP16>(
          ctx,
          reinterpret_cast<const XPUTypeFP16*>(probs.data<fp16_data_t>()),
          reinterpret_cast<XPUTypeFP16*>(renorm_probs.data<fp16_data_t>()),
          top_k.data<int64_t>(),
          batch_size,
          vocab_size);
      break;
    }
    case paddle::DataType::FLOAT32: {
      r = fastdeploy::plugin::top_k_renorm_probs<float>(
          ctx,
          probs.data<float>(),
          renorm_probs.data<float>(),
          top_k.data<int64_t>(),
          batch_size,
          vocab_size);
      break;
    }
    default:
      PD_THROW("Unsupported data type.");
  }
  PD_CHECK(r == 0, "fastdeploy::plugin::top_k_renorm_probs failed.");
  return {renorm_probs};
}

std::vector<std::vector<int64_t>> TopKRenormInferShape(
    const std::vector<int64_t>& probs_shape,
    const std::vector<int64_t>& top_k_shape) {
  return {probs_shape};
}

std::vector<paddle::DataType> TopKRenormInferDtype(
    const paddle::DataType& probs_dtype, const paddle::DataType& top_k_dtype) {
  return {probs_dtype};
}

PD_BUILD_STATIC_OP(top_k_renorm_probs)
    .Inputs({"probs", "top_k"})
    .Outputs({"renorm_probs"})
    .SetKernelFn(PD_KERNEL(TopKRenorm))
    .SetInferShapeFn(PD_INFER_SHAPE(TopKRenormInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TopKRenormInferDtype));
