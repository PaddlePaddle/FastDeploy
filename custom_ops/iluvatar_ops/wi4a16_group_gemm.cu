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

std::vector<paddle::Tensor> WI4A16GroupGemm(const paddle::Tensor& x,
                                            const paddle::Tensor& weight,
                                            const paddle::Tensor& weight_scale,
                                            const paddle::Tensor& weight_zeros,
                                            const paddle::Tensor& prefix_sum,
                                            const int32_t group_size) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));
  auto stream = static_cast<const cudaStream_t>(dev_ctx->stream());
  auto prefix_sum_cpu = prefix_sum.copy_to(paddle::CPUPlace(), false);

  const auto& x_dims = x.dims();
  const auto& w_dims = weight.dims();
  const auto& ws_dims = weight_scale.dims();
  const auto& prefix_sum_dims = prefix_sum.dims();
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
  PD_CHECK(prefix_sum_dims.size() == 1, "prefix_sum should be 1D");
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
  PD_CHECK(prefix_sum_dims[0] == n_experts);

  PD_CHECK(x.dtype() == paddle::DataType::BFLOAT16 ||
           x.dtype() == paddle::DataType::FLOAT16);
  PD_CHECK(weight.dtype() == paddle::DataType::INT8);
  PD_CHECK(weight_scale.dtype() == x.dtype());
  PD_CHECK(weight_zeros.dtype() == x.dtype());
  PD_CHECK(prefix_sum.dtype() == paddle::DataType::INT64);

  PD_CHECK(x.is_contiguous());
  PD_CHECK(weight.is_contiguous());
  PD_CHECK(weight_scale.is_contiguous());
  PD_CHECK(weight_zeros.is_contiguous());
  PD_CHECK(prefix_sum.is_contiguous());

  const int64_t* prefix_sum_cpu_ptr = prefix_sum_cpu.data<int64_t>();
  auto output = GetEmptyTensor({m, n}, x.dtype(), x.place());
  int16_t* out_data = static_cast<int16_t*>(output.data());
  const int16_t* x_data = static_cast<const int16_t*>(x.data());
  const int8_t* weight_data = weight.data<int8_t>();
  const int16_t* weight_scale_data =
      static_cast<const int16_t*>(weight_scale.data());
  const int16_t* weight_zeros_data =
      static_cast<const int16_t*>(weight_zeros.data());

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

  cuinferQuantGEMMDeviceParam cust_device_param;
  cust_device_param.size = sizeof(cuinferQuantGEMMDeviceParam);
  cust_device_param.bias = nullptr;

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
    size_t expert_i_end = prefix_sum_cpu_ptr[i];
    size_t cur_len = expert_i_end - pre;
    pre = expert_i_end;
    if (cur_len != 0) {
      cust_device_param.scale = weight_scale_data;
      cust_device_param.zero = weight_zeros_data;

      size_t workspace_size = 0;
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
      if (workspace_size > 0) {
        tmp_workspace = allocator->Allocate(workspace_size);
        cust_device_param.workspace = tmp_workspace->ptr();
      } else {
        cust_device_param.workspace = nullptr;
      }

      if (cur_len <= 1) {
        CUINFER_CHECK(cuinferCustomGemmEx(handle,
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
                                          customOption,
                                          cust_device_param.workspace));
      } else {
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
    }
    x_data += cur_len * k;
    weight_data += k * n / 2;
    weight_scale_data += k * n / group_size;
    weight_zeros_data += k * n / group_size;
    out_data += cur_len * n;
  }
  return {output};
}

std::vector<std::vector<int64_t>> WI4A16GroupGemmInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& weight_shape) {
  return {{x_shape[0], weight_shape[1] * 2}};
}
std::vector<paddle::DataType> WI4A16GroupGemmInferDtype(
    const paddle::DataType& input_dtype) {
  return {input_dtype};
}

PD_BUILD_STATIC_OP(wi4a16_group_gemm)
    .Inputs({"x", "weight", "weight_scale", "weight_zeros", "prefix_sum"})
    .Outputs({"output"})
    .Attrs({
        "group_size:int",
    })
    .SetKernelFn(PD_KERNEL(WI4A16GroupGemm))
    .SetInferShapeFn(PD_INFER_SHAPE(WI4A16GroupGemmInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WI4A16GroupGemmInferDtype));
