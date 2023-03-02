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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ncnn/net.h"
#include "ncnn/layer.h"
#include "ncnn/cpu.h"

#include "fastdeploy/runtime/backends/backend.h"
#include "fastdeploy/runtime/runtime_option.h"
#include "fastdeploy/runtime/backends/ncnn/option.h"

namespace fastdeploy {

class NCNNBackend : public BaseBackend {
 public:
  NCNNBackend() {}

  bool Init(const RuntimeOption& runtime_option) override;

  bool Infer(std::vector<FDTensor>& inputs,
             std::vector<FDTensor>* outputs,
             bool copy_to_fd = true) override; // NOLINT

  int NumInputs() const override { return inputs_desc_.size(); }

  int NumOutputs() const override { return outputs_desc_.size(); }

  TensorInfo GetInputInfo(int index) override;
  TensorInfo GetOutputInfo(int index) override;
  std::vector<TensorInfo> GetInputInfos() override;
  std::vector<TensorInfo> GetOutputInfos() override;

 private:
  void BuildOption(const NCNNBackendOption& option);
  bool UpdateInputShapeAndDesc(const std::vector<FDTensor>& inputs);
  bool SetTensorInfoByCustomOrder(
    const std::map<std::string, int>& custom_orders,
    const std::vector<const char*>& tensor_names,
    const std::vector<int>& tensor_indexes,
    std::vector<TensorInfo>* desc,
    std::map<std::string, int>* order);
  bool SetTensorInfo(const std::vector<const char*>& tensor_names,
                     const std::vector<int>& tensor_indexes,
                     std::vector<TensorInfo>* desc,
                     std::map<std::string, int>* order);
  std::vector<int> GetNCNNShape(const std::vector<int64_t>& shape);
  std::vector<int64_t> GetFDShape(const std::vector<int>& shape);
  std::string ShapeStr(const std::vector<int>& shape);
  std::vector<int> GetMatShapeByBlob(int id, size_t* elemsize);
  std::vector<int> GetMatShape(const ncnn::Mat& mat, size_t* elemsize);

  std::vector<TensorInfo> inputs_desc_;
  std::vector<TensorInfo> outputs_desc_;
  std::map<std::string, int> inputs_order_;
  std::map<std::string, int> outputs_order_;
  NCNNBackendOption option_;

  std::shared_ptr<ncnn::Net> net_;
  ncnn::Option opt_;
  std::vector<int> input_indexes_;
  std::vector<int> output_indexes_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
};

/// Convert data type from NCNN to FastDeploy
/// element size in bytes
/// 4 = float32/int32
/// 2 = float16
/// 1 = int8/uint8
/// 0 = empty
/// size_t elemsize;
/// Only support float32/int32 in FastDeploy now.
FDDataType NCNNDataTypeToFD(size_t elemsize, bool integer = false);

}  // namespace fastdeploy
