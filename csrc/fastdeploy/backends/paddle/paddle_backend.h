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
#include "paddle_inference_api.h"  // NOLINT

namespace fastdeploy {

struct PaddleBackendOption {
#ifdef WITH_GPU
  bool use_gpu = true;
#else
  bool use_gpu = false;
#endif
  bool enable_mkldnn = true;

  int mkldnn_cache_size = 1;
  int cpu_thread_num = 8;
  // initialize memory size(MB) for GPU
  int gpu_mem_init_size = 100;
  // gpu device id
  int gpu_id = 0;
};

// Share memory buffer with paddle_infer::Tensor from fastdeploy::FDTensor
void ShareTensorFromCpu(paddle_infer::Tensor* tensor, FDTensor& fd_tensor);

// Copy memory data from paddle_infer::Tensor to fastdeploy::FDTensor
void CopyTensorToCpu(std::unique_ptr<paddle_infer::Tensor>& tensor,
                     FDTensor* fd_tensor);

// Convert data type from paddle inference to fastdeploy
FDDataType PaddleDataTypeToFD(const paddle_infer::DataType& dtype);

class PaddleBackend : public BaseBackend {
 public:
  PaddleBackend() {}
  virtual ~PaddleBackend() = default;
  void BuildOption(const PaddleBackendOption& option);

  bool InitFromPaddle(
      const std::string& model_file, const std::string& params_file,
      const PaddleBackendOption& option = PaddleBackendOption());

  bool Infer(std::vector<FDTensor>& inputs, std::vector<FDTensor>* outputs);

  int NumInputs() const { return inputs_desc_.size(); }

  int NumOutputs() const { return outputs_desc_.size(); }

  TensorInfo GetInputInfo(int index);
  TensorInfo GetOutputInfo(int index);

 private:
  paddle_infer::Config config_;
  std::shared_ptr<paddle_infer::Predictor> predictor_;
  std::vector<TensorInfo> inputs_desc_;
  std::vector<TensorInfo> outputs_desc_;
};
}  // namespace fastdeploy
