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
#include <xft_api.h>
#include "paddle/extension.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "utility/debug.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <typename T>
std::vector<paddle::Tensor> MoeEPCombineKernel(
    const paddle::Tensor&
        ffn_out,  // expand_token_num * hidden_dim   dtype is fp16/bf16
    const paddle::Tensor& moe_index,  // token_num * topk   dtype is int
    const paddle::Tensor&
        weights,  // token_num * topk  dtype is same as ffn_out
    int64_t recv_token_num,
    int64_t expand_token_num,
    int64_t hidden_dim,
    int64_t topk) {
  using XPU_T = typename XPUTypeTrait<T>::Type;
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  auto combined_out = paddle::empty(
      {recv_token_num, hidden_dim}, ffn_out.dtype(), ffn_out.place());
  const float* dequant_score = nullptr;
  if (recv_token_num > 0) {
    int ret = infer_ops::moe_ep_ffn_post_fusion(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPU_T*>(ffn_out.data<T>()),
        moe_index.data<int32_t>(),
        reinterpret_cast<const XPU_T*>(weights.data<T>()),
        dequant_score,
        reinterpret_cast<XPU_T*>(combined_out.mutable_data<T>()),
        recv_token_num,
        hidden_dim,
        topk,
        expand_token_num);
    PD_CHECK(ret == 0);
  }

  return {combined_out};
}

std::vector<paddle::Tensor> MoeEPCombine(const paddle::Tensor& ffn_out,
                                         const paddle::Tensor& moe_index,
                                         const paddle::Tensor& weights,
                                         const int recv_token_num,
                                         const int expand_token_num,
                                         const int hidden_dim,
                                         const int topk) {
#define APPLY_KERNEL(TX)                          \
  return MoeEPCombineKernel<TX>(ffn_out,          \
                                moe_index,        \
                                weights,          \
                                recv_token_num,   \
                                expand_token_num, \
                                hidden_dim,       \
                                topk);

  const auto ffn_out_dtype = ffn_out.dtype();
  if (ffn_out_dtype == paddle::DataType::FLOAT16) {
    APPLY_KERNEL(paddle::float16);
  } else if (ffn_out_dtype == paddle::DataType::BFLOAT16) {
    APPLY_KERNEL(paddle::bfloat16);
  } else {
    PD_THROW("MoeEPCombine not support ffn_out_type==%d",
             static_cast<int>(ffn_out_dtype));
    return {};
  }

#undef APPLY_KERNEL
}

std::vector<std::vector<int64_t>> MoeEPCombineInferShape(
    const std::vector<int64_t>& ffn_out_shape,
    const std::vector<int64_t>& moe_index_shape,
    const std::vector<int64_t>& weights_shape,
    const int recv_token_num,
    const int expand_token_num,
    const int hidden_dim,
    const int topk) {
  std::vector<int64_t> combined_out_shape = {recv_token_num, hidden_dim};
  return {combined_out_shape};
}

std::vector<paddle::DataType> MoeEPCombineInferDtype(
    const paddle::DataType& ffn_out_dtype,
    const paddle::DataType& moe_index_dtype,
    const paddle::DataType& weights_dtype) {
  return {ffn_out_dtype};
}

PD_BUILD_STATIC_OP(ep_moe_expert_combine)
    .Inputs({"ffn_out", "moe_index", "weights"})
    .Outputs({"combined_out"})
    .Attrs({"recv_token_num: int",
            "expand_token_num: int",
            "hidden_dim: int",
            "topk: int"})
    .SetKernelFn(PD_KERNEL(MoeEPCombine))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeEPCombineInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeEPCombineInferDtype));
