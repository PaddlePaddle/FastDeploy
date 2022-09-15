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

#include <cuda_runtime_api.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "fastdeploy/backends/backend.h"
#include "fastdeploy/backends/tensorrt/utils.h"

namespace fastdeploy {

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
  std::map<std::string, std::vector<int32_t>> max_shape;
  std::map<std::string, std::vector<int32_t>> min_shape;
  std::map<std::string, std::vector<int32_t>> opt_shape;
  std::string serialize_file = "";

  // inside parameter, maybe remove next version
  bool remove_multiclass_nms_ = false;
  std::map<std::string, std::string> custom_op_info_;
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
  bool Infer(std::vector<FDTensor>& inputs, std::vector<FDTensor>* outputs);

  int NumInputs() const { return inputs_desc_.size(); }
  int NumOutputs() const { return outputs_desc_.size(); }
  TensorInfo GetInputInfo(int index);
  TensorInfo GetOutputInfo(int index);
  std::vector<TensorInfo> GetInputInfos() override;
  std::vector<TensorInfo> GetOutputInfos() override;

  ~TrtBackend() {
    if (parser_) {
      parser_.reset();
    }
  }

 private:
  TrtBackendOption option_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  FDUniquePtr<nvonnxparser::IParser> parser_;
  FDUniquePtr<nvinfer1::IBuilder> builder_;
  FDUniquePtr<nvinfer1::INetworkDefinition> network_;
  cudaStream_t stream_{};
  std::vector<void*> bindings_;
  std::vector<TrtValueInfo> inputs_desc_;
  std::vector<TrtValueInfo> outputs_desc_;
  std::map<std::string, FDDeviceBuffer> inputs_buffer_;
  std::map<std::string, FDDeviceBuffer> outputs_buffer_;

  // Sometimes while the number of outputs > 1
  // the output order of tensorrt may not be same
  // with the original onnx model
  // So this parameter will record to origin outputs
  // order, to help recover the rigt order
  std::map<std::string, int> outputs_order_;

  // temporary store onnx model content
  // once it used to build trt egnine done
  // it will be released
  std::string onnx_model_buffer_;
  // Stores shape information of the loaded model
  // For dynmaic shape will record its range information
  // Also will update the range information while inferencing
  std::map<std::string, ShapeRangeInfo> shape_range_info_;

  void GetInputOutputInfo();
  bool CreateTrtEngineFromOnnx(const std::string& onnx_model_buffer);
  bool BuildTrtEngine();
  bool LoadTrtCache(const std::string& trt_engine_file);
  int ShapeRangeInfoUpdated(const std::vector<FDTensor>& inputs);
  void SetInputs(const std::vector<FDTensor>& inputs);
  void AllocateOutputsBuffer(std::vector<FDTensor>* outputs);
};

}  // namespace fastdeploy
