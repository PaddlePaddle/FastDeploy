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

std::vector<paddle::Tensor> GroupGemm(const paddle::Tensor& x,
                                      const paddle::Tensor& weight,
                                      const paddle::Tensor& weight_scale,
                                      const paddle::Tensor& prefix_sum,
                                      const int32_t group_size) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));
  auto stream = static_cast<const cudaStream_t>(dev_ctx->stream());
  const auto& x_dims = x.dims();
  const auto& w_dims = weight.dims();
  const auto& ws_dims = weight_scale.dims();
  const auto& prefix_sum_dims = prefix_sum.dims();
  // [m, k]
  PD_CHECK(x_dims.size() == 2, "x should be 2D");
  // [n_experts, n, k]
  PD_CHECK(w_dims.size() == 3, "weight should be 3D");
  // [n_experts, n]
  PD_CHECK(ws_dims.size() == 2, "weight_scale should be 2D");
  // [n_experts]
  PD_CHECK(prefix_sum_dims.size() == 1, "prefix_sum should be 1D");
  PD_CHECK(group_size == -1);
  auto m = x_dims[0];
  auto k = x_dims[1];
  auto n_experts = w_dims[0];
  auto n = w_dims[1];
  PD_CHECK(w_dims[2] == k);
  PD_CHECK(ws_dims[0] == n_experts);
  PD_CHECK(ws_dims[1] == n);
  PD_CHECK(prefix_sum_dims[0] == n_experts);

  PD_CHECK(prefix_sum.dtype() == paddle::DataType::INT64);
  PD_CHECK(prefix_sum.is_cpu());
  PD_CHECK(x.dtype() == paddle::DataType::BFLOAT16 ||
           x.dtype() == paddle::DataType::FLOAT16);
  PD_CHECK(weight.dtype() == paddle::DataType::INT8);
  PD_CHECK(weight_scale.dtype() == x.dtype());
  PD_CHECK(x.is_contiguous());
  PD_CHECK(weight.is_contiguous());
  PD_CHECK(weight_scale.is_contiguous());

  const int64_t* prefix_sum_ptr = prefix_sum.data<int64_t>();
  auto output = GetEmptyTensor({m, n}, x.dtype(), x.place());
  int16_t* out_data = static_cast<int16_t*>(output.data());
  const int16_t* x_data = static_cast<const int16_t*>(x.data());
  const int8_t* weight_data = weight.data<int8_t>();
  const int16_t* weight_scale_data =
      static_cast<const int16_t*>(weight_scale.data());

  cuinferHandle_t handle = iluvatar::getContextInstance()->getIxInferHandle();
  cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
  cuinferOperation_t transa = CUINFER_OP_T;
  cuinferOperation_t transb = CUINFER_OP_N;
  cudaDataType_t a_type = CUDA_R_8I;
  cudaDataType_t b_type;
  cudaDataType_t c_type;
  if (x.dtype() == paddle::DataType::FLOAT16) {
    b_type = CUDA_R_16F;
  } else if (x.dtype() == paddle::DataType::BFLOAT16) {
    b_type = CUDA_R_16BF;
  } else {
    PADDLE_THROW(common::errors::Unimplemented("Unsupported input dtype."));
  }
  c_type = b_type;
  cudaDataType_t Atype = a_type;
  cudaDataType_t Btype = b_type;
  cudaDataType_t Ctype = c_type;
  cudaDataType_t computeType = CUDA_R_32F;
  cudaDataType_t scaleType = CUDA_R_32F;
  cuinferGEMMCustomOption_t customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;

  cuinferQuantGEMMHostParam cust_host_param;
  cust_host_param.size = sizeof(cuinferQuantGEMMHostParam);
  cust_host_param.persistent = 0;
  cust_host_param.groupSize = group_size;
  cuinferQuantGEMMDeviceParam cust_device_param;
  cust_device_param.bias = nullptr;
  cust_device_param.workspace = nullptr;

  int lda = k;
  int ldb = k;
  int ldc = n;
  float beta = 0.f;
  float alpha = 1.f;
  int batch_count = 1;
  size_t pre = 0;

  auto* allocator = paddle::GetAllocator(x.place());
  phi::Allocator::AllocationPtr tmp_workspace;
  for (int i = 0; i < n_experts; i++) {
    size_t expert_i_end = prefix_sum_ptr[i];
    size_t cur_len = expert_i_end - pre;
    pre = expert_i_end;
    if (cur_len != 0) {
      cust_device_param.scale = weight_scale_data;

      if (k % 64 != 0) {
        size_t workspace_size;
        CUINFER_CHECK(cuinferGetCustomGemmWorkspace(transa,
                                                    transb,
                                                    n,
                                                    cur_len,
                                                    k,
                                                    Atype,
                                                    lda,
                                                    lda,
                                                    Btype,
                                                    ldb,
                                                    ldb,
                                                    Ctype,
                                                    ldc,
                                                    ldc,
                                                    batch_count,
                                                    computeType,
                                                    scaleType,
                                                    &workspace_size));
        tmp_workspace = allocator->Allocate(workspace_size);
        cust_device_param.workspace = tmp_workspace->ptr();
      } else {
        cust_device_param.workspace = nullptr;
      }

      CUINFER_CHECK(cuinferCustomGemm(handle,
                                      stream,
                                      cuinfer_ptr_mode,
                                      transa,
                                      transb,
                                      n,
                                      cur_len,
                                      k,
                                      &alpha,
                                      weight_data,
                                      Atype,
                                      lda,
                                      lda,
                                      x_data,
                                      Btype,
                                      ldb,
                                      ldb,
                                      &beta,
                                      out_data,
                                      Ctype,
                                      ldc,
                                      ldc,
                                      batch_count,
                                      computeType,
                                      scaleType,
                                      &cust_host_param,
                                      &cust_device_param,
                                      customOption));
    }
    x_data += cur_len * k;
    weight_data += k * n;
    weight_scale_data += n;
    out_data += cur_len * n;
  }
  return {output};
}

std::vector<std::vector<int64_t>> GroupGemmInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& weight_scale_shape,
    const std::vector<int64_t>& prefix_sum_shape) {
  return {{x_shape[0], weight_shape[1]}};
}
std::vector<paddle::DataType> GroupGemmInferDtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& weight_output_dtype,
    const paddle::DataType& weight_scale_dtype,
    const paddle::DataType& prefix_sum_dtype,
    const int moe_topk) {
  return {input_dtype};
}

PD_BUILD_STATIC_OP(w8a16_group_gemm)
    .Inputs({"x", "weight", "weight_scale", "prefix_sum"})
    .Outputs({"output"})
    .Attrs({
        "group_size:int",
    })
    .SetKernelFn(PD_KERNEL(GroupGemm))
    .SetInferShapeFn(PD_INFER_SHAPE(GroupGemmInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GroupGemmInferDtype));
