// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <vector>

#include "tnn/core/macro.h"
#include "tnn/core/tnn.h"
#include "tnn/core/mat.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/cpu_utils.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"

#include "fastdeploy/runtime/backends/backend.h"
#include "fastdeploy/runtime/runtime_option.h"
#include "fastdeploy/runtime/backends/tnn/option.h"

namespace fastdeploy {

class TNNBackend : public BaseBackend {
 public:
  TNNBackend() {}

  bool Init(const RuntimeOption& runtime_option) override;

  bool Infer(std::vector<FDTensor>& inputs,
             std::vector<FDTensor>* outputs,
             bool copy_to_fd = true) override;  // NOLINT

  int NumInputs() const override { return inputs_desc_.size(); }

  int NumOutputs() const override { return outputs_desc_.size(); }

  TensorInfo GetInputInfo(int index) override;
  TensorInfo GetOutputInfo(int index) override;
  std::vector<TensorInfo> GetInputInfos() override;
  std::vector<TensorInfo> GetOutputInfos() override;

 private:
  void BuildOption(const TNNBackendOption& option);
  bool UpdateInputShapeAndDesc(const std::vector<FDTensor>& inputs);
  std::string ContentBufferFromFile(const char *proto_or_model_path);
  bool SetTensorInfoByCustomOrder(
    const std::map<std::string, int>& custom_orders,
    const tnn::BlobMap& blobs, std::vector<TensorInfo>* desc,
    std::map<std::string, int>* order);
  bool SetTensorInfo(const tnn::BlobMap& blobs,
                     std::vector<TensorInfo>* desc,
                     std::map<std::string, int>* order);
  std::vector<int> GetTNNShape(const std::vector<int64_t>& shape);
  std::vector<int64_t> GetFDShape(const std::vector<int>& shape);
  std::string ShapeStr(const std::vector<int>& shape);

  std::vector<TensorInfo> inputs_desc_;
  std::vector<TensorInfo> outputs_desc_;
  std::map<std::string, int> inputs_order_;
  std::map<std::string, int> outputs_order_;

  TNNBackendOption option_;

  tnn::BlobMap inputs_blob_map_;
  tnn::BlobMap outputs_blob_map_;
  std::shared_ptr<tnn::TNN> net_;
  std::shared_ptr<tnn::Instance> instance_;
  tnn::ModelConfig model_config_;  // for net_
  tnn::NetworkConfig network_config_;  // for instance_
};

// Convert data/mat type from TNN to fastdeploy
FDDataType TNNDataTypeToFD(const tnn::DataType& dtype);
FDDataType TNNMatTypeToFD(const tnn::MatType& mtype);
}  // namespace fastdeploy
