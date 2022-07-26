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
#include "onnxruntime_cxx_api.h"  // NOLINT

namespace fastdeploy {

struct OrtValueInfo {
  std::string name;
  std::vector<int64_t> shape;
  ONNXTensorElementDataType dtype;
};

struct OrtBackendOption {
  // -1 means default
  // 0: ORT_DISABLE_ALL
  // 1: ORT_ENABLE_BASIC
  // 2: ORT_ENABLE_EXTENDED
  // 99: ORT_ENABLE_ALL (enable some custom optimizations e.g bert)
  int graph_optimization_level = -1;
  int intra_op_num_threads = -1;
  int inter_op_num_threads = -1;
  // 0: ORT_SEQUENTIAL
  // 1: ORT_PARALLEL
  int execution_mode = -1;
  bool use_gpu = false;
  int gpu_id = 0;

  // inside parameter, maybe remove next version
  bool remove_multiclass_nms_ = false;
  std::map<std::string, std::string> custom_op_info_;
};

class OrtBackend : public BaseBackend {
 public:
  OrtBackend() {}
  void BuildOption(const OrtBackendOption& option);

  bool InitFromPaddle(const std::string& model_file,
                      const std::string& params_file,
                      const OrtBackendOption& option = OrtBackendOption(),
                      bool verbose = false);

  bool InitFromOnnx(const std::string& model_file,
                    const OrtBackendOption& option = OrtBackendOption(),
                    bool from_memory_buffer = false);

  bool Infer(std::vector<FDTensor>& inputs, std::vector<FDTensor>* outputs);

  int NumInputs() const { return inputs_desc_.size(); }

  int NumOutputs() const { return outputs_desc_.size(); }

  TensorInfo GetInputInfo(int index);
  TensorInfo GetOutputInfo(int index);
  static std::vector<OrtCustomOp*> custom_operators_;
  void InitCustomOperators();

 private:
  Ort::Env env_;
  Ort::Session session_{nullptr};
  Ort::SessionOptions session_options_;
  std::shared_ptr<Ort::IoBinding> binding_;
  std::vector<OrtValueInfo> inputs_desc_;
  std::vector<OrtValueInfo> outputs_desc_;
  Ort::CustomOpDomain custom_op_domain_ = Ort::CustomOpDomain("Paddle");
  OrtBackendOption option_;
  void CopyToCpu(const Ort::Value& value, FDTensor* tensor);
};
}  // namespace fastdeploy
