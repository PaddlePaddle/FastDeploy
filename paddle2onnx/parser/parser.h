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
#include <algorithm>
#include <cassert>
#include <numeric>
#include <type_traits>

#include "paddle2onnx/proto/p2o_paddle.pb.h"
#include "paddle2onnx/utils/utils.h"

namespace paddle2onnx {

enum P2ODataType {
  BOOL,
  INT16,
  INT32,
  INT64,
  FP16,
  FP32,
  FP64,
  X7,
  X8,
  X9,
  X10,
  X11,
  X12,
  X13,
  X14,
  X15,
  X16,
  X17,
  X18,
  X19,
  UINT8,
  INT8,
};
int32_t PaddleDataTypeSize(int32_t paddle_dtype);

struct TensorInfo {
  std::string name;
  std::vector<int64_t> shape;
  int64_t Rank() const { return static_cast<int64_t>(shape.size()); }
  int32_t dtype;
  bool is_tensor_array = false;

  TensorInfo() {}
  TensorInfo(const std::string& _name, const std::vector<int64_t>& _shape,
             const int32_t& _dtype) {
    name = _name;
    shape.assign(_shape.begin(), _shape.end());
    dtype = _dtype;
  }

  TensorInfo(const TensorInfo& info) {
    name = info.name;
    shape.assign(info.shape.begin(), info.shape.end());
    dtype = info.dtype;
    is_tensor_array = info.is_tensor_array;
  }
};

struct Weight {
  std::vector<char> buffer;
  std::vector<int32_t> shape;
  int32_t dtype;

  template <typename T>
  void set(int32_t data_type, const std::vector<int64_t>& dims,
           const std::vector<T>& data) {
    buffer.clear();
    shape.clear();
    dtype = data_type;
    buffer.resize(data.size() * PaddleDataTypeSize(dtype));
    memcpy(buffer.data(), data.data(), data.size() * PaddleDataTypeSize(dtype));
    for (auto& d : dims) {
      shape.push_back(d);
    }
  }
  template <typename T>
  void get(std::vector<T>* data) const {
    int64_t nums = std::accumulate(std::begin(shape), std::end(shape), 1,
                                   std::multiplies<int64_t>());
    data->resize(nums);
    if (dtype == P2ODataType::INT64) {
      std::vector<int64_t> value(nums);
      memcpy(value.data(), buffer.data(), nums * sizeof(int64_t));
      data->assign(value.begin(), value.end());
    } else if (dtype == P2ODataType::INT32) {
      std::vector<int32_t> value(nums);
      memcpy(value.data(), buffer.data(), nums * sizeof(int32_t));
      data->assign(value.begin(), value.end());
    } else if (dtype == P2ODataType::INT8) {
      std::vector<int8_t> value(nums);
      memcpy(value.data(), buffer.data(), nums * sizeof(int8_t));
      data->assign(value.begin(), value.end());
    } else if (dtype == P2ODataType::FP32) {
      std::vector<float> value(nums);
      memcpy(value.data(), buffer.data(), nums * sizeof(float));
      data->assign(value.begin(), value.end());
    } else if (dtype == P2ODataType::FP64) {
      std::vector<double> value(nums);
      memcpy(value.data(), buffer.data(), nums * sizeof(double));
      data->assign(value.begin(), value.end());
    } else {
      Assert(false,
             "Weight::get() only support int64/int32/int8/float32/float64.");
    }
  }
};

class PaddleParser {
 public:
  // recording variable name:id for each block of a program
  std::vector<std::map<std::string, int32_t>> _blocks_var_name2id;
  // recoring set of operators for each block
  std::vector<std::vector<const paddle2onnx::framework::proto::OpDesc*>>
      _blocks_ops;
  std::shared_ptr<paddle2onnx::framework::proto::ProgramDesc> prog;
  std::map<std::string, Weight> params;
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
  bool is_quantized_model = false;  // If the Paddle model is a quantized model,
                                    // set is_quantized_model to be true

  bool Init(const std::string& _model, const std::string& _params = "");
  bool Init(const void* model_buffer, int64_t model_size,
            const void* params_buffer = nullptr, int64_t params_size = 0);
  void InitBlock();

  int NumOfBlocks() const;
  int NumOfOps(int block_idx) const;
  bool HasNms() const { return _has_nms; }
  const framework::proto::OpDesc& GetOpDesc(int32_t block_idx,
                                            int32_t op_idx) const;

  bool OpHasInput(int64_t block_id, int64_t op_id,
                  const std::string& name) const;
  bool OpHasOutput(int64_t block_id, int64_t op_id,
                   const std::string& name) const;

  std::vector<TensorInfo> GetOpInput(int64_t block_id, int64_t op_id,
                                     const std::string& name) const;
  std::vector<TensorInfo> GetOpOutput(int64_t block_id, int64_t op_id,
                                      const std::string& name) const;

  bool OpIsAttrVar(int64_t block_id, int64_t op_id,
                   const std::string& name) const;

  std::vector<TensorInfo> GetOpAttrVar(int64_t block_id, int64_t op_id,
                                       const std::string& name) const;

  bool OpHasAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name) const;

  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name, int64_t* res) const;
  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name, float* res) const;
  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name, bool* res) const;
  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name, std::string* res) const;
  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name, std::vector<int64_t>* res) const;
  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name, std::vector<float>* res) const;
  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name, std::vector<double>* res) const;

  bool IsConstantTensor(const int64_t& block_idx,
                        const std::string& tensor_name) const;
  template <typename T>
  bool TryGetTensorValue(const int64_t& block_id,
                         const std::string& tensor_name,
                         std::vector<T>* data) const;

 private:
  // If the model has same output name in difference operators
  // will fail to convert
  bool IsAttrVar(const paddle2onnx::framework::proto::OpDesc& op,
                 const int64_t& attr_id) const;
  bool ExistsDumplicateTensorName() const;
  void GetBlocksVarName2Id();
  void GetBlocksOps();
  TensorInfo GetTensorInfo(
      const std::string& name,
      const paddle2onnx::framework::proto::BlockDesc& block) const;
  void GetGlobalBlockInputOutputInfo();
  bool GetParamNames(std::vector<std::string>* var_names);
  bool LoadProgram(const std::string& model);
  bool LoadProgram(const void* model_buffer, int64_t model_size);
  bool LoadParams(const std::string& path);
  bool LoadParamsFromMemoryBuffer(const std::string& buffer);
  bool LoadParamsFromMemoryBuffer(const void* params_buffer,
                                  int64_t params_size);
  // This is a trick flag
  // While there's a nms operator in paddle model,
  // the shape inference of paddle is not correct
  bool _has_nms = false;
  std::vector<std::map<std::string, int64_t>> _constant_ops;
};

template <typename T>
bool PaddleParser::TryGetTensorValue(const int64_t& block_id,
                                     const std::string& tensor_name,
                                     std::vector<T>* data) const {
  {
    auto iter = params.find(tensor_name);
    if (iter != params.end()) {
      (iter->second).get(data);
      return true;
    }
  }
  Assert(block_id < _constant_ops.size(),
         "block_id is out of range while calling TryGetTensorValue.");
  auto iter = _constant_ops[block_id].find(tensor_name);
  if (iter == _constant_ops[block_id].end()) {
    return false;
  }
  Assert(iter->second < _blocks_ops[block_id].size(),
         "op_idx is out of range while calling TryGetTensorValue.");
  auto op = _blocks_ops[block_id][iter->second];
  int64_t dtype;
  GetOpAttr(*op, "dtype", &dtype);
  if (dtype == P2ODataType::INT64) {
    std::vector<int64_t> value;
    GetOpAttr(*op, "int64_values", &value);
    data->assign(value.begin(), value.end());
  } else if (dtype == P2ODataType::INT32) {
    std::vector<int64_t> value;
    GetOpAttr(*op, "int32_values", &value);
    data->assign(value.begin(), value.end());
  } else if (dtype == P2ODataType::FP32) {
    std::vector<float> value;
    GetOpAttr(*op, "fp32_values", &value);
    data->assign(value.begin(), value.end());
  } else {
    Assert(
        false,
        "Only support int32/int64/float32 data type in assign_value operator.");
  }
  return true;
}

}  // namespace paddle2onnx
