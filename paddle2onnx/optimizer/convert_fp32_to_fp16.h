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
#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>

#include <cmath>
#include <fstream>
#include <iomanip>

#include "paddle2onnx/mapper/mapper.h"
#include "paddle2onnx/parser/parser.h"
namespace paddle2onnx {

struct proto_node {
 public:
  std::string node_type;  // model, graph, node, arribute
  ONNX_NAMESPACE::ModelProto* model;
  ONNX_NAMESPACE::GraphProto* graph;
  ONNX_NAMESPACE::NodeProto* node;
  ONNX_NAMESPACE::AttributeProto* attr;

  explicit proto_node(ONNX_NAMESPACE::ModelProto new_model) {
    node_type = "model";
    model = &new_model;
  }

  explicit proto_node(ONNX_NAMESPACE::ModelProto* new_model) {
    node_type = "model";
    model = new_model;
  }

  explicit proto_node(ONNX_NAMESPACE::GraphProto new_graph) {
    node_type = "graph";
    graph = &new_graph;
  }

  explicit proto_node(ONNX_NAMESPACE::GraphProto* new_graph) {
    node_type = "graph";
    graph = new_graph;
  }

  explicit proto_node(ONNX_NAMESPACE::NodeProto new_node) {
    node_type = "node";
    node = &new_node;
  }

  explicit proto_node(ONNX_NAMESPACE::NodeProto* new_node) {
    node_type = "node";
    node = new_node;
  }

  explicit proto_node(ONNX_NAMESPACE::AttributeProto new_attribute) {
    node_type = "attribute";
    attr = &new_attribute;
  }

  explicit proto_node(ONNX_NAMESPACE::AttributeProto* new_attribute) {
    node_type = "attribute";
    attr = new_attribute;
  }
};

struct ConvertFp32ToFp16 {
 public:
  ConvertFp32ToFp16(float min_positive_val = 1e-7, float max_finite_val = 1e4,
                    bool keep_io_types = false,
                    bool disable_shape_infer = false,
                    const std::vector<std::string>& op_block_list = {},
                    const std::vector<std::string>& node_block_list = {}) {
    min_positive_val_ = min_positive_val;
    max_finite_val_ = max_finite_val;
    keep_io_types_ = keep_io_types;
    disable_shape_infer_ = disable_shape_infer;
    op_block_list_ = op_block_list;
    node_block_list_ = node_block_list;
  }

  void Convert(ONNX_NAMESPACE::ModelProto* model);

  ONNX_NAMESPACE::NodeProto* MakeCastNode(
      const std::string& op_name, const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs, int32_t to_dtype);

  ONNX_NAMESPACE::ValueInfoProto* MakeValueInfoFromTensor(
      const ONNX_NAMESPACE::TensorProto& tensor);

  void KeepIoType(ONNX_NAMESPACE::ModelProto* model);

  void ConvertAttribute(ONNX_NAMESPACE::ModelProto* model);

  void ConvertTensorFloatToFloat16(ONNX_NAMESPACE::TensorProto* tensor);

  // return if keep the type of node
  bool KeepNodeType(ONNX_NAMESPACE::NodeProto* node);

  bool GetTensorValue(const ONNX_NAMESPACE::TensorProto& tensor,
                      std::vector<float>* value);

  // topo sort
  void SortNodes(ONNX_NAMESPACE::ModelProto* model);

  void ConvertValToFloat16(float val, uint16_t* x);

  // return if the next node of name is Cast and its attr type is dtype.
  bool CastedTo(const std::string& name, ONNX_NAMESPACE::ModelProto& model,
                int64_t dtype);
  // return if the pre node of name is Cast and its attr type is dtype.
  bool CastedFrom(const std::string& name, ONNX_NAMESPACE::ModelProto& model,
                  int64_t dtype);
  // return if the name is the input of DEFAULT_OP_BLOCK_LIST
  bool IsInputOfOpBlock(const std::string& name,
                        ONNX_NAMESPACE::ModelProto& model);

  // return if the name is the input of DEFAULT_OP_BLOCK_LIST and
  // fp32_output_op_list
  bool IsOutputOfOpBlockAndFP32Out(const std::string& name,
                                   ONNX_NAMESPACE::ModelProto& model);

  void SetCustomOps(const std::map<std::string, std::string>& custom_ops) {
    if (custom_ops.size()) {
      custom_ops_.clear();
      for (auto op : custom_ops) {
        custom_ops_.push_back(op.second);
      }
    }
  }

  void AddDisabledOpTypes(const std::vector<std::string>& disable_fp16_ops) {
    op_block_list_.insert(op_block_list_.end(), disable_fp16_ops.begin(),
                          disable_fp16_ops.end());
  }
  // If the input ONNX model is a FP16 model, return True
  bool IsFP16Model(const ONNX_NAMESPACE::ModelProto& model);

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };
  static const int shift = 13;
  static const int shiftSign = 16;

  static const int32_t infN = 0x7F800000;
  static const int32_t maxN = 0x477FE000;  // max flt16 as flt32
  static const int32_t minN = 0x38800000;  // min flt16 normal as flt32
  static const int32_t sigN = 0x80000000;  // sign bit

  static constexpr int32_t infC = infN >> shift;
  static constexpr int32_t nanN = (infC + 1)
                                  << shift;  // minimum flt16 nan as float32
  static constexpr int32_t maxC = maxN >> shift;
  static constexpr int32_t minC = minN >> shift;
  static constexpr int32_t sigC = sigN >> shiftSign;

  static const int32_t mulN = 0x52000000;  // (1 << 23) / minN
  static const int32_t mulC = 0x33800000;  // minN / (1 << (23 - shift))
  static const int32_t subC = 0x003FF;     // max flt32 subnormal downshifted
  static const int32_t norC = 0x00400;     // min flt32 normal downshifted

  static constexpr int32_t maxD = infC - maxC - 1;
  static constexpr int32_t minD = minC - subC - 1;

  float min_positive_val_ = 1e-7;
  float max_finite_val_ = 1e4;
  bool keep_io_types_ = false;
  bool disable_shape_infer_ = false;
  std::vector<std::string> op_block_list_ = {};
  std::vector<std::string> node_block_list_ = {};

  std::vector<std::string> custom_ops_ = {"AdaptivePool2d", "MultiClassNMS"};

  int64_t converted_attr = 0;

  std::map<std::string, std::string> name_mapping;
  std::vector<std::string> graph_io_to_skip;
  std::vector<ONNX_NAMESPACE::ValueInfoProto*> value_info_list;
  std::vector<std::string> io_casts;

  std::vector<ONNX_NAMESPACE::NodeProto*> node_list;

  std::vector<proto_node> queue;
  std::vector<proto_node> next_level;

  std::map<std::string, int64_t> name_index_mapper;
  // int64_t name_index = 0;
  std::string GenName(const std::string& prefix);

  // save the tensor names that should keep data type
  std::vector<std::string> keep_type_tensors;

  // The input can be FP16, but the output can only be fp32
  std::vector<std::string> fp32_output_op_list = {"RandomNormalLike"};

  std::vector<std::string> DEFAULT_OP_BLOCK_LIST = {
      "ArrayFeatureExtractor",
      "ReduceMean",  // this op may cause wrong results on FP16
      "Binarizer",
      "CastMap",
      "CategoryMapper",
      "DictVectorizer",
      "FeatureVectorizer",
      "Imputer",
      "LabelEncoder",
      "LinearClassifier",
      "LinearRegressor",
      "Normalizer",
      "OneHotEncoder",
      "RandomUniformLike",
      "SVMClassifier",
      "SVMRegressor",
      "Scaler",
      "TreeEnsembleClassifier",
      "TreeEnsembleRegressor",
      "ZipMap",
      "NonMaxSuppression",
      "TopK",
      "RoiAlign",
      "Resize",
      "Range",
      "CumSum",
      "Min",
      "Max",
      "Upsample",  // The following OP is added by Paddle developer
      "EyeLike"};
};
}  // namespace paddle2onnx
