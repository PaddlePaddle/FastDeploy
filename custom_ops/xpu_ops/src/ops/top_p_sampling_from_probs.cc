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
#include <paddle/phi/backends/xpu/xpu_context.h>
#include "paddle/extension.h"
#include "paddle/phi/core/enforce.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

// infer_ops::top_p_sampling_from_probs<T, TID> signature:
//   int top_p_sampling_from_probs(
//       api::Context* ctx,
//       T* probs,            // [batch_size, d]
//       TID* output,         // [batch_size]
//       TID* indices,        // [batch_size] or nullptr
//       T* top_p_arr,        // [batch_size] or nullptr
//       int64_t batch_size,
//       T top_p_val,
//       int64_t d,
//       bool deterministic,
//       int64_t philox_seed,
//       int64_t philox_offset,
//       int64_t topk = 0);
//
// gtest reference (the call we must match):
//   gtest_top_p_sampling_from_probs<float, int>(
//       api::kXPU3, "GM", "GM", "NULL", "GM",
//       1, 1, 151552, true, 0, 0, 0);
//   => probs=GM, output=GM, indices=NULL, top_p_arr=GM,
//      batch_size=1, top_p_val=1.0f, d=151552,
//      deterministic=true, philox_seed=0, philox_offset=0, topk=0

std::vector<paddle::Tensor> TopPSamplingFromProbsKernel(
    const paddle::Tensor& probs,
    const paddle::Tensor& top_p_arr,
    float top_p_val,
    bool deterministic,
    int64_t philox_seed,
    int64_t philox_offset,
    int64_t topk) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  auto probs_shape = probs.shape();
  PD_CHECK(
      probs_shape.size() == 2,
      "top_p_sampling_from_probs_xpu: probs must be 2D [batch_size, d], got ",
      probs_shape.size(),
      "D");

  int64_t batch_size = probs_shape[0];
  int64_t d = probs_shape[1];

  // output: [batch_size], int32
  paddle::Tensor output =
      paddle::empty({batch_size}, paddle::DataType::INT32, probs.place());

  // probs is float32 (as per gtest: T=float, TID=int)
  float* probs_data = const_cast<float*>(probs.data<float>());
  int32_t* output_data = output.data<int32_t>();
  int32_t* indices_data = nullptr;  // "NULL" in gtest
  float* top_p_arr_data = const_cast<float*>(top_p_arr.data<float>());

  int ret = infer_ops::top_p_sampling_from_probs<float, int32_t>(
      xpu_ctx->x_context(),
      probs_data,
      output_data,
      indices_data,
      top_p_arr_data,
      batch_size,
      static_cast<float>(top_p_val),
      d,
      deterministic,
      philox_seed,
      philox_offset,
      topk);
  PD_CHECK(ret == 0,
           "top_p_sampling_from_probs_xpu: infer_ops call failed, ret=",
           ret);

  return {output};
}

std::vector<std::vector<int64_t>> TopPSamplingFromProbsInferShape(
    const std::vector<int64_t>& probs_shape,
    const std::vector<int64_t>& top_p_arr_shape) {
  // output shape: [batch_size]
  return {{probs_shape[0]}};
}

std::vector<paddle::DataType> TopPSamplingFromProbsInferDtype(
    const paddle::DataType& probs_dtype,
    const paddle::DataType& top_p_arr_dtype) {
  return {paddle::DataType::INT32};
}

PD_BUILD_STATIC_OP(top_p_sampling_from_probs_xpu)
    .Inputs({"probs", "top_p_arr"})
    .Outputs({"output"})
    .Attrs({"top_p_val:float",
            "deterministic:bool",
            "philox_seed:int64_t",
            "philox_offset:int64_t",
            "topk:int64_t"})
    .SetKernelFn(PD_KERNEL(TopPSamplingFromProbsKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(TopPSamplingFromProbsInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TopPSamplingFromProbsInferDtype));
