// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// C++ wrapper for flashinfer trtllm_allreduce_fusion.
// Converts paddle::Tensor -> DLTensor -> tvm::ffi::TensorView, then calls
// the registered TVM FFI function from the pre-compiled trtllm_comm.so.

#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "paddle/extension.h"

// TVM FFI C++ API (umbrella header)
#include "tvm/ffi/tvm_ffi.h"

// ---------------------------------------------------------------------------
// Global state: dlopen handle for trtllm_comm.so
// ---------------------------------------------------------------------------

static void* g_trtllm_so_handle = nullptr;

static bool LoadTrtllmSo(const std::string& so_path) {
  if (g_trtllm_so_handle != nullptr) return true;
  g_trtllm_so_handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!g_trtllm_so_handle) {
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Paddle DataType -> DLDataType
// ---------------------------------------------------------------------------

static DLDataType PaddleDTypeToDL(paddle::DataType dtype) {
  switch (dtype) {
    case paddle::DataType::FLOAT16:
      return DLDataType{kDLFloat, 16, 1};
    case paddle::DataType::BFLOAT16:
      return DLDataType{kDLBfloat, 16, 1};
    case paddle::DataType::FLOAT32:
      return DLDataType{kDLFloat, 32, 1};
    case paddle::DataType::INT64:
      return DLDataType{kDLInt, 64, 1};
    case paddle::DataType::INT32:
      return DLDataType{kDLInt, 32, 1};
    case paddle::DataType::UINT8:
      return DLDataType{kDLUInt, 8, 1};
    default:
      throw std::runtime_error("PaddleDTypeToDL: unsupported dtype");
  }
}

// ---------------------------------------------------------------------------
// Helper: paddle::Tensor -> DLTensor (shape stored in out_shape vector)
//
// shape_buf must outlive the returned DLTensor (shape ptr points into it).
// Call MakeTensorView() to get a TensorView (TensorView copies DLTensor).
// ---------------------------------------------------------------------------

static DLTensor MakeDLTensor(const paddle::Tensor& t,
                             std::vector<int64_t>& shape_buf) {
  int ndim = static_cast<int>(t.dims().size());
  shape_buf.resize(ndim);
  for (int i = 0; i < ndim; ++i) shape_buf[i] = t.dims()[i];

  DLTensor dl;
  dl.data = const_cast<void*>(t.data());
  dl.device = DLDevice{kDLCUDA, t.place().GetDeviceId()};
  dl.ndim = ndim;
  dl.shape = shape_buf.data();
  dl.strides = nullptr;
  dl.byte_offset = 0;
  dl.dtype = PaddleDTypeToDL(t.dtype());
  return dl;
}

// TensorView copies DLTensor by value (shape ptr NOT copied — still points
// into the original DLTensor's shape array).  Keep the DLTensor and its
// shape_buf alive for the duration of the TVM FFI call.
static tvm::ffi::TensorView MakeTensorView(const DLTensor& dl) {
  return tvm::ffi::TensorView(&dl);
}

// ---------------------------------------------------------------------------
// Main kernel: paddle::Tensor in -> (norm_out, residual_out)
// ---------------------------------------------------------------------------
//
// AllReduceFusionPattern::kARResidualRMSNorm == 1
// (from flashinfer/comm/trtllm_allreduce_fusion.cuh)
//
static constexpr int64_t kARResidualRMSNorm = 1;

std::vector<paddle::Tensor> TrtllmAllreduceResidualRmsnorm(
    const paddle::Tensor& input_tensor,
    const paddle::Tensor& residual,
    const paddle::Tensor& weight,
    const paddle::Tensor& workspace_ptrs,
    int64_t world_size,
    int64_t world_rank,
    bool use_oneshot,
    bool trigger_completion_at_end,
    bool fp32_acc,
    double rms_eps) {
  if (g_trtllm_so_handle == nullptr) {
    throw std::runtime_error(
        "TrtllmAllreduceResidualRmsnorm: trtllm_comm.so not loaded. "
        "Call init_trtllm_so(path) first.");
  }

  auto norm_out = paddle::empty_like(input_tensor);
  auto residual_out = paddle::empty_like(residual);

  int64_t token_num = input_tensor.dims()[0];
  // Support empty tensor
  if (token_num == 0) {
    return {norm_out, residual_out};
  }

  // Build DLTensors + keep shape buffers alive for the duration of the call.
  std::vector<int64_t> sh_input, sh_residual, sh_weight, sh_workspace,
      sh_residual_out, sh_norm_out;
  DLTensor dl_input = MakeDLTensor(input_tensor, sh_input);
  DLTensor dl_residual = MakeDLTensor(residual, sh_residual);
  DLTensor dl_weight = MakeDLTensor(weight, sh_weight);
  DLTensor dl_workspace = MakeDLTensor(workspace_ptrs, sh_workspace);
  DLTensor dl_residual_out = MakeDLTensor(residual_out, sh_residual_out);
  DLTensor dl_norm_out = MakeDLTensor(norm_out, sh_norm_out);

  int64_t hidden_dim = input_tensor.dims()[1];

  // Look up the TVM FFI registered function (registered when .so was dlopen'd)
  auto fn_opt = tvm::ffi::Function::GetGlobal("trtllm_allreduce_fusion");
  if (!fn_opt.has_value()) {
    throw std::runtime_error(
        "TrtllmAllreduceResidualRmsnorm: 'trtllm_allreduce_fusion' not found "
        "in TVM FFI global registry.");
  }
  auto& fn = *fn_opt;

  using OptTV = tvm::ffi::Optional<tvm::ffi::TensorView>;

  fn(MakeTensorView(dl_input),
     world_size,
     world_rank,
     token_num,
     hidden_dim,
     MakeTensorView(dl_workspace),
     /*launch_with_pdl=*/true,
     use_oneshot,
     trigger_completion_at_end,
     fp32_acc,
     kARResidualRMSNorm,
     OptTV(std::nullopt),                     // allreduce_out  = None
     OptTV(MakeTensorView(dl_residual)),      // residual_in
     OptTV(MakeTensorView(dl_residual_out)),  // residual_out
     OptTV(MakeTensorView(dl_norm_out)),      // norm_out
     OptTV(std::nullopt),                     // quant_out      = None
     OptTV(std::nullopt),                     // scale_out      = None
     OptTV(MakeTensorView(dl_weight)),        // rms_gamma
     tvm::ffi::Optional<double>(rms_eps),
     OptTV(std::nullopt),                       // scale_factor   = None
     tvm::ffi::Optional<int64_t>(std::nullopt)  // layout_code    = None
  );

  return {norm_out, residual_out};
}

bool InitTrtllmSo(const std::string& so_path) { return LoadTrtllmSo(so_path); }
