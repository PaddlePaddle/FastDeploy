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
#include <vector>

#include "paddle2onnx/mapper/data_helper.h"
#include "paddle2onnx/mapper/onnx_helper.h"
#include "paddle2onnx/mapper/register_mapper.h"
#include "paddle2onnx/parser/parser.h"

namespace paddle2onnx {

class Mapper {
 public:
  Mapper() {}
  Mapper(const PaddleParser& p, OnnxHelper* helper, int32_t block_id,
         int32_t op_id, std::string name = {})
      : parser_(&p) {
    block_idx_ = block_id;
    op_idx_ = op_id;
    helper_ = helper;
    name_ = name;
  }

  // The flag will control if the op is exported as a custom operator
  // if export_as_custom_op = true, will exported as description in
  // custom_op_info
  bool export_as_custom_op = false;
  // [exported_op_name, domain]
  std::string custom_op_name;
  std::string deploy_backend;

  P2OLogger Logger(const bool& verbose, const int32_t& opset_version = 100) {
    bool v = verbose;
    if (opset_version <= helper_->GetOpsetVersion()) {
      v = false;
    }
    auto& op = parser_->GetOpDesc(block_idx_, op_idx_);
    std::string output_name = "";
    if (op.outputs(0).arguments_size() > 0) {
      output_name = op.outputs(0).arguments(0);
    }
    std::string op_type = op.type();
    std::string prefix = "[Paddle2ONNX] [" + op_type + ": " + output_name + "]";
    return P2OLogger(v, prefix);
  }

  P2OLogger Error() {
    auto& op = parser_->GetOpDesc(block_idx_, op_idx_);
    std::string output_name = "";
    if (op.outputs(0).arguments_size() > 0) {
      output_name = op.outputs(0).arguments(0);
    }
    std::string op_type = op.type();
    std::string prefix =
        "[ERROR][Paddle2ONNX] [" + op_type + ": " + output_name + "]";
    return P2OLogger(true, prefix);
  }

  P2OLogger Warn() {
    auto& op = parser_->GetOpDesc(block_idx_, op_idx_);
    std::string output_name = "";
    if (op.outputs(0).arguments_size() > 0) {
      output_name = op.outputs(0).arguments(0);
    }
    std::string op_type = op.type();
    std::string prefix =
        "[WARN][Paddle2ONNX] [" + op_type + ": " + output_name + "]";
    return P2OLogger(true, prefix);
  }

  // Some operators is not implement very well, e.g the output may not be same
  // We mark these operators as experimental, these operators requires double
  // checking after model exported.
  virtual void MarkAsExperimentalOp() { is_experimental_op_ = true; }
  virtual bool IsExperimentalOp() const { return is_experimental_op_; }
  // the return value in [7, MAX_ONNX_OPSET_VERSION], represent the minimum
  // opset_version
  // if return value < 0, means the op is not supported.
  virtual int32_t GetMinOpset(bool verbose = false) { return 7; }

  virtual bool IsExportAsCustomOp() { return export_as_custom_op; }

  void Run() {
    int32_t opset_version = helper_->GetOpsetVersion();
    Assert(opset_version >= 7 && opset_version <= MAX_ONNX_OPSET_VERSION,
           "[Paddle2ONNX] Only support opset_version in range of [7, " +
               std::to_string(MAX_ONNX_OPSET_VERSION) + "].");
    if (IsExportAsCustomOp()) {
      return ExportAsCustomOp();
    }
    if (opset_version == 16) {
      Opset16();
    } else if (opset_version == 15) {
      Opset15();
    } else if (opset_version == 14) {
      Opset14();
    } else if (opset_version == 13) {
      Opset13();
    } else if (opset_version == 12) {
      Opset12();
    } else if (opset_version == 11) {
      Opset11();
    } else if (opset_version == 10) {
      Opset10();
    } else if (opset_version == 9) {
      Opset9();
    } else if (opset_version == 8) {
      Opset8();
    } else {
      Opset7();
    }
  }

  virtual void ExportAsCustomOp() {
    Assert(false,
           "Operator " + name_ + "doesn't support export as custom operator.");
  }

  virtual void Opset16() { Opset15(); }

  virtual void Opset15() { Opset14(); }

  virtual void Opset14() { Opset13(); }

  virtual void Opset13() { Opset12(); }

  virtual void Opset12() { Opset11(); }

  virtual void Opset11() { Opset10(); }

  virtual void Opset10() { Opset9(); }

  virtual void Opset9() { Opset8(); }

  virtual void Opset8() { Opset7(); }

  virtual void Opset7() {
    Assert(false,
           "This error shouldn't happend, please report to "
           "https://github.com/PaddlePaddle/Paddle2ONNX.git.");
  }

  virtual ~Mapper() = default;
  bool is_experimental_op_ = false;
  const PaddleParser* parser_;
  OnnxHelper* helper_;
  int32_t block_idx_;
  int32_t op_idx_;
  std::string name_;  // op transform name

  std::string OpType() const {
    auto& op = parser_->GetOpDesc(block_idx_, op_idx_);
    return op.type();
  }

  std::string Name() const { return name_; }

  bool HasInput(const std::string& name) const {
    return parser_->OpHasInput(block_idx_, op_idx_, name);
  }
  bool HasOutput(const std::string& name) const {
    return parser_->OpHasOutput(block_idx_, op_idx_, name);
  }
  std::vector<TensorInfo> GetInput(const std::string& name) const {
    return parser_->GetOpInput(block_idx_, op_idx_, name);
  }
  std::vector<TensorInfo> GetOutput(const std::string& name) const {
    return parser_->GetOpOutput(block_idx_, op_idx_, name);
  }
  // Judge whether Attribute(name)'s type is Var or Vars.
  bool IsAttrVar(const std::string& name) const {
    return parser_->OpIsAttrVar(block_idx_, op_idx_, name);
  }

  // Get TensorInfo(s) from Attribute Var or Vars.
  std::vector<TensorInfo> GetAttrVar(const std::string& name) const {
    return parser_->GetOpAttrVar(block_idx_, op_idx_, name);
  }

  bool HasAttr(const std::string& name) const {
    auto& op = parser_->GetOpDesc(block_idx_, op_idx_);
    return parser_->OpHasAttr(op, name);
  }
  void GetAttr(const std::string& name, int64_t* val) {
    auto& op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, name, val);
  }
  void GetAttr(const std::string& name, float* val) {
    auto& op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, name, val);
  }
  void GetAttr(const std::string& name, bool* val) {
    auto& op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, name, val);
  }
  void GetAttr(const std::string& name, std::string* val) {
    auto& op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, name, val);
  }
  void GetAttr(const std::string& name, std::vector<int64_t>* val) {
    auto& op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, name, val);
  }
  void GetAttr(const std::string& name, std::vector<float>* val) {
    auto& op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, name, val);
  }
  void GetAttr(const std::string& name, std::vector<double>* val) {
    auto& op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, name, val);
  }

  bool IsConstantInput(const std::string& input_key) const {
    auto input_info = GetInput(input_key);
    return parser_->IsConstantTensor(block_idx_, input_info[0].name);
  }

  bool IsConstant(const TensorInfo& info) const {
    return parser_->IsConstantTensor(block_idx_, info.name);
  }

  template <typename T>
  bool TryGetInputValue(const std::string& input_key, std::vector<T>* data) {
    auto input_info = GetInput(input_key);
    return parser_->TryGetTensorValue(block_idx_, input_info[0].name, data);
  }

  template <typename T>
  bool TryGetValue(const TensorInfo& info, std::vector<T>* data) {
    return parser_->TryGetTensorValue(block_idx_, info.name, data);
  }
};

}  // namespace paddle2onnx
