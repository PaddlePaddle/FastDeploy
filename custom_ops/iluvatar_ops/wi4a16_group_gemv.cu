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

#include "helper.h"
#include "iluvatar_context.h"

std::vector<paddle::Tensor> WI4A16GroupGemv(
    const paddle::Tensor& x,
    const paddle::Tensor& weight,
    const paddle::Tensor& weight_scale,
    const paddle::Tensor& weight_zeros,
    const paddle::Tensor& tokens_per_expert,
    const int32_t group_size) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));
  auto stream = static_cast<const cudaStream_t>(dev_ctx->stream());

  const auto& x_dims = x.dims();
  const auto& w_dims = weight.dims();
  const auto& ws_dims = weight_scale.dims();
  const auto& tokens_per_expert_dims = tokens_per_expert.dims();
  const auto& zeros_dims = weight_zeros.dims();
  // [m, k]
  PD_CHECK(x_dims.size() == 2, "x should be 2D");
  // [n_experts, n // 2, k]
  PD_CHECK(w_dims.size() == 3, "weight should be 3D");
  // [n_experts, k // group_size, n]
  PD_CHECK(ws_dims.size() == 3, "weight_scale should be 3D");
  // [n_experts, k // group_size, n]
  PD_CHECK(zeros_dims.size() == 3, "weight_zeros should be 3D");
  // [n_experts]
  PD_CHECK(tokens_per_expert_dims.size() == 1,
           "tokens_per_expert should be 1D");
  PD_CHECK(group_size == 128);
  auto m = x_dims[0];
  auto k = x_dims[1];
  auto n_experts = w_dims[0];
  auto n = w_dims[1] * 2;
  PD_CHECK(w_dims[2] == k);
  PD_CHECK(ws_dims[0] == n_experts);
  PD_CHECK(ws_dims[1] == k / group_size);
  PD_CHECK(ws_dims[2] == n);
  PD_CHECK(zeros_dims[0] == n_experts);
  PD_CHECK(zeros_dims[1] == k / group_size);
  PD_CHECK(zeros_dims[2] == n);
  PD_CHECK(tokens_per_expert_dims[0] == n_experts);

  PD_CHECK(x.dtype() == paddle::DataType::BFLOAT16 ||
           x.dtype() == paddle::DataType::FLOAT16);
  PD_CHECK(weight.dtype() == paddle::DataType::INT8);
  PD_CHECK(weight_scale.dtype() == x.dtype());
  PD_CHECK(weight_zeros.dtype() == x.dtype());
  PD_CHECK(tokens_per_expert.dtype() == paddle::DataType::INT32);

  PD_CHECK(x.is_contiguous());
  PD_CHECK(weight.is_contiguous());
  PD_CHECK(weight_scale.is_contiguous());
  PD_CHECK(weight_zeros.is_contiguous());
  PD_CHECK(tokens_per_expert.is_contiguous());

  auto output = GetEmptyTensor({m, n}, x.dtype(), x.place());

  cuinferHandle_t handle = iluvatar::getContextInstance()->getIxInferHandle();
  cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
  cuinferOperation_t transa = CUINFER_OP_T;
  cuinferOperation_t transb = CUINFER_OP_N;
  cudaDataType_t Atype = CUDA_R_4I;
  cudaDataType_t Btype;
  if (x.dtype() == paddle::DataType::FLOAT16) {
    Btype = CUDA_R_16F;
  } else if (x.dtype() == paddle::DataType::BFLOAT16) {
    Btype = CUDA_R_16BF;
  } else {
    PADDLE_THROW(common::errors::Unimplemented("Unsupported input dtype."));
  }
  cudaDataType_t Ctype = Btype;
  cudaDataType_t computeType = CUDA_R_32F;
  cudaDataType_t scaleType = CUDA_R_32F;
  cuinferGEMMCustomOption_t customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;

  cuinferQuantGEMMHostParam cust_host_param;
  cuinferCustomGemmHostParamInit(&cust_host_param);
  cust_host_param.size = sizeof(cuinferQuantGEMMHostParam);
  cust_host_param.persistent = 0;
  cust_host_param.groupSize = group_size;
  cust_host_param.expertCount = n_experts;
  cust_host_param.type = 2;

  cuinferQuantGEMMDeviceParam cust_device_param;
  cust_device_param.size = sizeof(cuinferQuantGEMMDeviceParam);
  cust_device_param.sortedId = nullptr;
  cust_device_param.bias = nullptr;
  cust_device_param.scale = weight_scale.data();
  cust_device_param.zero = weight_zeros.data();
  cust_device_param.nSize = tokens_per_expert.data<int32_t>();

  int lda = k;
  int ldb = k;
  int ldc = n;
  float beta = 0.f;
  float alpha = 1.f;
  int batch_count = 1;

  size_t workspace_size = 0;
  CUINFER_CHECK(cuinferGetCustomGemmExWorkspaceWithParam(n,
                                                         m,
                                                         k,
                                                         transa,
                                                         transb,
                                                         batch_count,
                                                         Atype,
                                                         Btype,
                                                         Ctype,
                                                         computeType,
                                                         scaleType,
                                                         &cust_host_param,
                                                         customOption,
                                                         &workspace_size));
  if (workspace_size > 0) {
    auto* allocator = paddle::GetAllocator(x.place());
    phi::Allocator::AllocationPtr tmp_workspace;
    tmp_workspace = allocator->Allocate(workspace_size);
    cust_device_param.workspace = tmp_workspace->ptr();
  } else {
    cust_device_param.workspace = nullptr;
  }

  CUINFER_CHECK(cuinferCustomGemmEx(handle,
                                    stream,
                                    cuinfer_ptr_mode,
                                    transa,
                                    transb,
                                    n,
                                    m,
                                    k,
                                    &alpha,
                                    weight.data(),
                                    Atype,
                                    lda,
                                    0,  // n * k
                                    x.data(),
                                    Btype,
                                    ldb,
                                    0,
                                    &beta,
                                    output.data(),
                                    Ctype,
                                    ldc,
                                    0,
                                    batch_count,
                                    computeType,
                                    scaleType,
                                    &cust_host_param,
                                    &cust_device_param,
                                    customOption,
                                    cust_device_param.workspace));
  return {output};
}

std::vector<std::vector<int64_t>> WI4A16GroupGemvInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& weight_shape) {
  return {{x_shape[0], weight_shape[1] * 2}};
}

std::vector<paddle::DataType> WI4A16GroupGemvInferDtype(
    const paddle::DataType& input_dtype) {
  return {input_dtype};
}

PD_BUILD_STATIC_OP(wi4a16_group_gemv)
    .Inputs(
        {"x", "weight", "weight_scale", "weight_zeros", "tokens_per_expert"})
    .Outputs({"output"})
    .Attrs({
        "group_size:int",
    })
    .SetKernelFn(PD_KERNEL(WI4A16GroupGemv))
    .SetInferShapeFn(PD_INFER_SHAPE(WI4A16GroupGemvInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WI4A16GroupGemvInferDtype));
