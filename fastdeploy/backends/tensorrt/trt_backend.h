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
#include <map>
#include <string>
#include <vector>

#include "fastdeploy/backends/backend.h"

#include "fastdeploy/backends/tensorrt/common/argsParser.h"
#include "fastdeploy/backends/tensorrt/common/buffers.h"
#include "fastdeploy/backends/tensorrt/common/common.h"
#include "fastdeploy/backends/tensorrt/common/logger.h"
#include "fastdeploy/backends/tensorrt/common/parserOnnxConfig.h"
#include "fastdeploy/backends/tensorrt/common/sampleUtils.h"

#include <cuda_runtime_api.h>
#include "NvInfer.h"

namespace fastdeploy {
using namespace samplesCommon;

struct TrtValueInfo {
  std::string name;
  std::vector<int> shape;
  nvinfer1::DataType dtype;
};

struct TrtBackendOption {
  int gpu_id = 0;
  bool enable_fp16 = false;
  bool enable_int8 = false;
  size_t max_batch_size = 32;
  size_t max_workspace_size = 1 << 30;
  std::map<std::string, std::vector<int32_t>> fixed_shape;
  std::map<std::string, std::vector<int32_t>> max_shape;
  std::map<std::string, std::vector<int32_t>> min_shape;
  std::map<std::string, std::vector<int32_t>> opt_shape;
  std::string serialize_file = "";
};

std::vector<int> toVec(const nvinfer1::Dims& dim);
size_t TrtDataTypeSize(const nvinfer1::DataType& dtype);
FDDataType GetFDDataType(const nvinfer1::DataType& dtype);

class TrtBackend : public BaseBackend {
 public:
  TrtBackend() : engine_(nullptr), context_(nullptr) {}
  void BuildOption(const TrtBackendOption& option);

  bool InitFromPaddle(const std::string& model_file,
                      const std::string& params_file,
                      const TrtBackendOption& option = TrtBackendOption(),
                      bool verbose = false);
  bool InitFromOnnx(const std::string& model_file,
                    const TrtBackendOption& option = TrtBackendOption(),
                    bool from_memory_buffer = false);
  bool InitFromTrt(const std::string& trt_engine_file,
                   const TrtBackendOption& option = TrtBackendOption());

  bool Infer(std::vector<FDTensor>& inputs, std::vector<FDTensor>* outputs);

  int NumInputs() const { return inputs_desc_.size(); }
  int NumOutputs() const { return outputs_desc_.size(); }
  TensorInfo GetInputInfo(int index);
  TensorInfo GetOutputInfo(int index);

 private:
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_{};
  std::vector<void*> bindings_;
  std::vector<TrtValueInfo> inputs_desc_;
  std::vector<TrtValueInfo> outputs_desc_;
  std::map<std::string, DeviceBuffer> inputs_buffer_;
  std::map<std::string, DeviceBuffer> outputs_buffer_;

  // Sometimes while the number of outputs > 1
  // the output order of tensorrt may not be same
  // with the original onnx model
  // So this parameter will record to origin outputs
  // order, to help recover the rigt order
  std::map<std::string, int> outputs_order_;

  void GetInputOutputInfo();
  void AllocateBufferInDynamicShape(const std::vector<FDTensor>& inputs,
                                    std::vector<FDTensor>* outputs);
  bool CreateTrtEngine(const std::string& onnx_model,
                       const TrtBackendOption& option);
};

}  // namespace fastdeploy
