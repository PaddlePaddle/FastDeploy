// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "fastdeploy/backends/backend.h"

#include "fastdeploy/backends/poros/common/compile.h"
#include "fastdeploy/backends/poros/common/poros_module.h"

namespace fastdeploy {

struct PorosBackendOption {
#ifdef WITH_GPU
  bool use_gpu = true;
#else
  bool use_gpu = false;
#endif
  int gpu_id = 0;
  bool long_to_int = true;
  // There is calculation precision in tf32 mode on A10, it can bring some
  // performance improvement, but there may be diff
  bool use_nvidia_tf32 = false;
  // Threshold for the number of non-const ops
  int32_t unconst_ops_thres = -1;
  std::string poros_file = "";
  std::vector<FDDataType> prewarm_datatypes = {FDDataType::FP32};
  // TRT options
  bool enable_fp16 = false;
  bool enable_int8 = false;
  bool is_dynamic = false;
  size_t max_batch_size = 32;
  size_t max_workspace_size = 1 << 30;
};

// Convert data type from fastdeploy to poros
at::ScalarType GetPorosDtype(const FDDataType& fd_dtype);

// Convert data type from poros to fastdeploy
FDDataType GetFdDtype(const at::ScalarType& dtype);

// at::ScalarType to std::string for FDERROR
std::string AtType2String(const at::ScalarType& dtype);

// Create at::Tensor
// is_backend_cuda specify if Poros use GPU Device
// While is_backend_cuda = true, and tensor.device = Device::GPU
at::Tensor CreatePorosValue(FDTensor& tensor, bool is_backend_cuda = false);

// Copy memory data from at::Tensor to fastdeploy::FDTensor
void CopyTensorToCpu(const at::Tensor& tensor, FDTensor* fd_tensor,
                     bool is_backend_cuda = false);

class PorosBackend : public BaseBackend {
 public:
  PorosBackend() {}
  virtual ~PorosBackend() = default;

  void BuildOption(const PorosBackendOption& option);

  bool InitFromTorchScript(
      const std::string& model_file,
      const PorosBackendOption& option = PorosBackendOption());

  bool InitFromPoros(const std::string& model_file,
                     const PorosBackendOption& option = PorosBackendOption());

  bool Compile(const std::string& model_file,
               std::vector<std::vector<FDTensor>>& prewarm_tensors,
               const PorosBackendOption& option = PorosBackendOption());

  bool Infer(std::vector<FDTensor>& inputs,
             std::vector<FDTensor>* outputs,
             bool copy_to_fd = true) override;

  int NumInputs() const { return _numinputs; }

  int NumOutputs() const { return _numoutputs; }

  TensorInfo GetInputInfo(int index) override;
  TensorInfo GetOutputInfo(int index) override;
  std::vector<TensorInfo> GetInputInfos() override;
  std::vector<TensorInfo> GetOutputInfos() override;

 private:
  baidu::mirana::poros::PorosOptions _options;
  std::unique_ptr<baidu::mirana::poros::PorosModule> _poros_module;
  std::vector<std::vector<c10::IValue>> _prewarm_datas;
  int _numinputs = 1;
  int _numoutputs = 1;
};

}  // namespace fastdeploy
