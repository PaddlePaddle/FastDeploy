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

#include "paddle2onnx/mapper/quantize_helper.h"

namespace paddle2onnx {

void QuantizeModelProcessor::RemoveNodeByName(const std::string& name,
                                              const bool& update_io) {
  if (name.empty()) {
    return;
  }
  for (auto iter = nodes_->begin(); iter != nodes_->end(); iter++) {
    if ((*iter)->name() == name) {
      std::string input_name = (*iter)->input(0);
      std::string output_name = (*iter)->output(0);
      nodes_->erase(iter);
      if (update_io) {
        ReplaceInputOfAllNodes(output_name, input_name);
      }
      return;
    }
  }
}

void QuantizeModelProcessor::ReplaceInputOfAllNodes(
    const std::string& old_name, const std::string& new_name,
    const std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>&
        except_nodes) {
  auto iter = name2node_dict_.find(old_name);
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> need_rename_nodes;
  // after replace all old_name to new_name, replace the quantize_info of new
  // name with the quantize_info of old name
  auto quantize_info_iter = helper_->quantize_info.find(old_name);
  if (quantize_info_iter != helper_->quantize_info.end()) {
    helper_->quantize_info[new_name] = helper_->quantize_info[old_name];
  }

  if (iter != name2node_dict_.end()) {
    need_rename_nodes = iter->second;
  }
  for (auto& node : need_rename_nodes) {
    auto iter = std::find(except_nodes.begin(), except_nodes.end(), node);
    if (iter != except_nodes.end()) {
      continue;
    }
    for (size_t i = 0; i < node->input_size(); ++i) {
      if (node->input(i) == old_name) {
        node->set_input(i, new_name);
      }
    }
  }
}

void QuantizeModelProcessor::UpdateInputNameToNodes() {
  name2node_dict_.clear();
  for (auto& node : *nodes_) {
    for (size_t i = 0; i < node->input_size(); ++i) {
      std::string node_input = node->input(i);
      if (name2node_dict_.find(node_input) != name2node_dict_.end()) {
        name2node_dict_[node_input].push_back(node);
      } else {
        name2node_dict_[node_input] = {node};
      }
    }
  }
}

void QuantizeModelProcessor::ProcessQuantizeModel(
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* parameters,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* inputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* outputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes,
    OnnxHelper* helper, const std::string& deploy_backend,
    const PaddleParser& parser, std::string* calibration_cache) {
  // Determine whether the model contains quantization related OPs, if not, exit
  // directly
  bool quantized_model = false;
  for (auto& node : *nodes) {
    if (node->op_type() == "QuantizeLinear" ||
        node->op_type() == "DequantizeLinear") {
      quantized_model = true;
      break;
    }
  }
  if (!quantized_model) {
    return;
  }
  parser_ = &parser;
  helper_ = helper;
  parameters_ = parameters;
  inputs_ = inputs;
  outputs_ = outputs;
  nodes_ = nodes;
  P2OLogger() << "[Info] Quantize model deploy backend is: " << deploy_backend
              << std::endl;
  // Determine the format of the exported ONNX quantization model according to
  // the deploy_backend
  if (deploy_backend == "others") {
    // If deploy_backend is others, the quantization model is exported as a
    // float model + quantization table.
    RemoveAllQuantizeOps();
    std::ofstream outfile;
    outfile.open("max_range.txt", std::ios::out);
    if (!outfile.is_open()) {
      P2OLogger() << "[WARNING] Quantize model processer failed to write range "
                     "information in current location."
                  << std::endl;
      return;
    }
    for (auto iter = helper_->quantize_info.begin();
         iter != helper_->quantize_info.end(); iter++) {
      std::string log = iter->first;
      auto scale = iter->second.scale_;
      if (scale.size() == 1) {
        log = log + ": " + std::to_string(scale[0] * 127);
        outfile << log << std::endl;
      }
    }
    outfile.close();
  } else if (deploy_backend == "onnxruntime") {
    // When deploy_backend is ONNXRuntime, use the follow four steps to process:
    // 1. broadcast quantize info
    // 2. remove all quantize ops
    // 3. merge conv and add
    // 4. merge conv and bn
    // 5. add Q and DQ according ONNXRuntime quantize OP fuse patten.
    // 6. use topo sort in nodes
    QuantizeInfoBroadcast();
    RemoveAllQuantizeOps();
    MergeConvAdd();
    MergeConvBN();
    AddQDQForORT();
    SortNodes();
  } else if (deploy_backend == "tensorrt") {
    // When deploy_backend is TensorRT, use the follow four steps to process:
    // For Explicit Quantization
    // 1. broadcast quantize info
    // 2. remove all quantize ops
    // 3. add Q and DQ before conv and matmul.
    // 4. use topo sort in nodes

    // For Implicit Quantization
    // 1. remove all quantize ops
    // 2. broadcast quantize info
    // 3. save float onnx model and alibration.cache
    QuantizeInfoBroadcast();
    RemoveAllQuantizeOps();
    // Add qdq for Explicit Quantization
    // AddTrtQDQ();
    // SortNodes();

    // Genarate calibration.cache for Implicit Quantization
    // convert float to hex
    GenerateCache(calibration_cache);
  } else if (deploy_backend == "rknn") {
    // When deploy_backend is RKNN, use the follow four steps to process:
    // 1. broadcast quantize info
    // 2. remove all quantize ops
    // 3. merge conv and add
    // 4. merge conv and bn
    // 5. add Q and DQ
    // 6. use topo sort in nodes
    QuantizeInfoBroadcast();
    RemoveAllQuantizeOps();
    MergeConvAdd();
    MergeConvBN();
    AddQDQForRKNN();
    SortNodes();
  } else {
    Assert(false,
           "[QuantizeModelProcessor] Only support 'onnxruntime'  / 'tensorrt' "
           "/ 'others' as "
           "backend now, but now the backend is: " +
               deploy_backend + ".");
  }
}

void QuantizeModelProcessor::AddQDQForRKNN() {
  UpdateInputNameToNodes();
  supported_quantize_type_ = {"Abs",
                              "Acos",
                              "Add",
                              "Asin",
                              "Atan",
                              "AveragePool",
                              "BatchNormalization",
                              "Ceil",
                              "Clip",
                              "Conv",
                              "ConvTranspose",
                              "Cos",
                              "Cosh",
                              "Concat",
                              "Elu",
                              "Erf",
                              "Exp",
                              "Floor",
                              "Gemm",
                              "HardSigmoid",
                              "InstanceNormalization",
                              "IsInf",
                              "IsNaN",
                              "Log",
                              "MaxPool",
                              "Mul",
                              "Neg",
                              "ReduceMean",
                              "Relu",
                              "Resize",
                              "Round",
                              "Sigmoid",
                              "Sin",
                              "Sinh",
                              "Split",
                              "Sqrt",
                              "Tan",
                              "Tanh"};
  for (auto iter = nodes_->begin(); iter < nodes_->end(); iter++) {
    auto node = *iter;
    auto type_iter = std::find(supported_quantize_type_.begin(),
                               supported_quantize_type_.end(), node->op_type());
    if (!supported_quantize_type_.empty() &&
        type_iter == supported_quantize_type_.end()) {
      continue;
    }
    // Add Miss scale for Mul and MatMul
    if (node->op_type() == "Mul" || node->op_type() == "MatMul" ||
        node->op_type() == "Add") {
      for (size_t i = 0; i < node->input_size(); ++i) {
        std::string node_input = node->input(i);
        if (helper_->quantize_info.find(node_input) !=
            helper_->quantize_info.end()) {
          continue;
        }
        std::vector<float> weight;
        if (!GetTensorByName(node_input, &weight)) {
          continue;
        }
        std::vector<int64_t> weight_shape;
        GetTensorShape(node_input, &weight_shape);
        int64_t quantize_axis = 1;
        if (node->op_type() == "Add") {
          quantize_axis = 0;
        }
        std::vector<float> scale;
        std::vector<int64_t> zeros;

        if (node->op_type() == "Add" || weight_shape.size() == 1 ||
            weight_shape.empty()) {
          GetTensorWiseQuantizeInfo(weight, &scale, &zeros);
        } else {
          GetChannelWiseQuantizeInfo(weight, weight_shape, quantize_axis,
                                     &scale, &zeros);
        }
        std::string weight_scale_node, weight_zero_node;
        if (scale.size() == 1) {
          weight_scale_node = helper_->Constant(
              {}, ONNX_NAMESPACE::TensorProto::FLOAT, scale[0]);
          weight_zero_node = helper_->Constant(
              {}, ONNX_NAMESPACE::TensorProto::INT8, zeros[0]);
        } else {
          weight_scale_node =
              helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT, scale);
          weight_zero_node =
              helper_->Constant(ONNX_NAMESPACE::TensorProto::INT8, zeros);
        }
        QuantizeInfo weight_quantize_info(scale, zeros, weight_scale_node,
                                          weight_zero_node, quantize_axis);
        helper_->quantize_info[node_input] = weight_quantize_info;
      }
    }
    std::vector<std::string> tensor_names;
    for (size_t i = 0; i < node->input_size(); ++i) {
      std::string node_input = node->input(i);
      tensor_names.push_back(node_input);
    }
    for (size_t i = 0; i < node->output_size(); ++i) {
      std::string node_output = node->output(i);
      tensor_names.push_back(node_output);
    }
    if (!CanBeQuantize(tensor_names)) {
      continue;
    }
    for (auto& name : tensor_names) {
      AppendQuantizeTensor(name);
    }
  }

  // update name2node_dict for the change of Relu op.
  UpdateInputNameToNodes();
  // Add QDQ in model
  AddQDQInModel(tensors_to_be_quantize);
}

void QuantizeModelProcessor::GenerateCache(std::string* calibration_cache) {
  union {
    float f;
    unsigned char farray[4];
  } un;
  *calibration_cache += "TRT-8XXX-EntropyCalibration2 \n";
  for (auto iter = helper_->quantize_info.rbegin();
       iter != helper_->quantize_info.rend(); iter++) {
    std::string tensor_name = iter->first;
    QuantizeInfo quantize_info = iter->second;
    if (quantize_info.scale_.size() == 1) {
      float val = quantize_info.scale_[0];
      un.f = val;
      *calibration_cache += (tensor_name + ": ");
      std::stringstream enc;
      for (int64_t i = 3; i >= 0; i--) {
        enc << std::hex << std::setw(2) << std::setfill('0')
            << (int)(un.farray[i]);
      }
      *calibration_cache = *calibration_cache + enc.str() + "\n";
    }
  }
}
// In TensorRT, all quantized op: Conv, ConvTranspose, liner(MatMul), MaxPool,
// AvgPool, AdaptiveAvgPool, rnn(not support now)
// https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization/pytorch_quantization/nn/modules
void QuantizeModelProcessor::AddTrtQDQ() {
  UpdateInputNameToNodes();
  std::vector<std::string>
      quantize_tensors;  // save the tensor names that need add quantize ops
  std::vector<std::string> pool_types = {"MaxPool", "AvgPool",
                                         "AdaptiveAvgPool"};
  for (auto iter = nodes_->begin(); iter < nodes_->end(); iter++) {
    quantize_tensors.clear();
    auto node = *iter;
    if (node->op_type() == "Conv" || node->op_type() == "ConvTranspose") {
      std::vector<std::string> tensor_names = {node->input(0), node->input(1)};
      if (!CanBeQuantize(tensor_names)) {
        continue;
      }
      quantize_tensors = tensor_names;
    }
    if (node->op_type() == "MatMul") {
      std::vector<std::string> tensor_names = {node->input(0), node->input(1)};
      for (auto& name : tensor_names) {
        if (helper_->quantize_info.find(name) != helper_->quantize_info.end()) {
          continue;
        }
        std::vector<float> matmul_weight;
        if (!GetTensorByName(name, &matmul_weight)) {
          continue;
        }
        std::vector<int64_t> matmul_weight_shape;
        if (!GetTensorShape(name, &matmul_weight_shape)) {
          continue;
        }
        int64_t quantize_axis = 1;
        std::vector<float> scale;
        std::vector<int64_t> zeros;
        GetChannelWiseQuantizeInfo(matmul_weight, matmul_weight_shape,
                                   quantize_axis, &scale, &zeros);
        auto scale_node =
            helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT, scale);
        auto zero_node =
            helper_->Constant(ONNX_NAMESPACE::TensorProto::INT8, zeros);
        QuantizeInfo matmul_weight_quantize_info(scale, zeros, scale_node,
                                                 zero_node, quantize_axis);
        helper_->quantize_info[name] = matmul_weight_quantize_info;
      }
      if (!CanBeQuantize(tensor_names)) {
        continue;
      }
      quantize_tensors = tensor_names;
    }
    auto type_iter =
        std::find(pool_types.begin(), pool_types.end(), node->op_type());
    if (type_iter != pool_types.end()) {
      std::vector<std::string> tensor_names = {node->input(0)};
      if (!CanBeQuantize(tensor_names)) {
        continue;
      }
      quantize_tensors = tensor_names;
    }

    std::string negative_scale_tensor = "";
    for (std::string& name : quantize_tensors) {
      Assert(
          helper_->quantize_info.find(name) != helper_->quantize_info.end(),
          "[QuantizeModelProcessor] Can not find quantize info for tensor: " +
              name);
      QuantizeInfo quantize_info = helper_->quantize_info[name];
      std::vector<float> scales = quantize_info.scale_;
      for (auto& i : scales) {
        if (i <= 1e-10) {
          negative_scale_tensor = negative_scale_tensor + " " + name;
        }
      }
    }
    if (negative_scale_tensor.size() > 0) {
      P2OLogger()
          << "[Warning] The scale of tensors: [ " + negative_scale_tensor +
                 " ] contains negative scale, so this OP will not be quantized."
          << std::endl;
      continue;
    }
    // An OP requires a separate quantize op
    for (std::string& name : quantize_tensors) {
      if (IsGraphOutput(name)) {
        continue;
      }
      QuantizeInfo quantize_info = helper_->quantize_info[name];
      std::string scale_node = quantize_info.scale_node_;
      std::string zeros_node = quantize_info.zeros_node_;
      int64_t quantize_axis = quantize_info.quantize_axis_;
      auto q_node =
          helper_->MakeNode("QuantizeLinear", {name, scale_node, zeros_node});
      if (helper_->GetOpsetVersion() >= 13) {
        AddAttribute(q_node, "axis", quantize_axis);
      }
      auto dq_node = helper_->MakeNode(
          "DequantizeLinear", {q_node->output(0), scale_node, zeros_node});
      if (helper_->GetOpsetVersion() >= 13) {
        AddAttribute(dq_node, "axis", quantize_axis);
      }
      for (size_t i = 0; i < node->input_size(); ++i) {
        if (node->input(i) == name) {
          node->set_input(i, dq_node->output(0));
        }
      }
    }
  }
}

// According to:
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.cc
void QuantizeModelProcessor::AddQDQForORT() {
  UpdateInputNameToNodes();
  supported_quantize_type_ = {"Conv", "MatMul",    "Mul", "Sigmoid",
                              "Add",  "LeakyRelu", "Relu"};
  for (auto iter = nodes_->begin(); iter < nodes_->end(); iter++) {
    auto node = *iter;
    auto type_iter = std::find(supported_quantize_type_.begin(),
                               supported_quantize_type_.end(), node->op_type());
    if (!supported_quantize_type_.empty() &&
        type_iter == supported_quantize_type_.end()) {
      continue;
    }
    // Here we only add Relu, Conv, mul and matmul, all tensors should add Q and
    // DQ will be saved in tensors_to_be_quantize
    if (node->op_type() == "Relu") {
      std::vector<std::string> tensor_names = {node->input(0), node->output(0)};
      if (!CanBeQuantize(tensor_names)) {
        continue;
      }
      node->set_op_type("LeakyRelu");
      AddAttribute(node, "alpha", static_cast<float>(0.0));
      for (auto& name : tensor_names) {
        AppendQuantizeTensor(name);
      }
    }
    if (node->op_type() == "LeakyRelu") {
      std::vector<std::string> tensor_names = {node->input(0), node->output(0)};
      if (!CanBeQuantize(tensor_names)) {
        continue;
      }
      for (auto& name : tensor_names) {
        AppendQuantizeTensor(name);
      }
    }
    if (node->op_type() == "Add") {
      std::vector<std::string> tensor_names = {node->input(0), node->input(1),
                                               node->output(0)};
      if (!CanBeQuantize(tensor_names)) {
        continue;
      }
      for (auto& name : tensor_names) {
        AppendQuantizeTensor(name);
      }
    }
    if (node->op_type() == "Sigmoid") {
      std::vector<std::string> tensor_names = {node->input(0), node->output(0)};
      if (!CanBeQuantize(tensor_names)) {
        continue;
      }
      for (auto& name : tensor_names) {
        AppendQuantizeTensor(name);
      }
    }
    if (node->op_type() == "Conv") {
      std::vector<std::string> tensor_names = {node->input(0), node->input(1),
                                               node->output(0)};
      if (node->input_size() == 3) {
        tensor_names.push_back(node->input(2));
      }
      if (!CanBeQuantize(tensor_names, {2})) {
        continue;
      }
      for (auto& name : tensor_names) {
        AppendQuantizeTensor(name);
      }
    }
    if (node->op_type() == "MatMul") {
      std::vector<std::string> tensor_names = {node->input(0), node->input(1),
                                               node->output(0)};
      for (auto& name : tensor_names) {
        if (helper_->quantize_info.find(name) != helper_->quantize_info.end()) {
          continue;
        }
        std::vector<float> matmul_weight;
        if (!GetTensorByName(name, &matmul_weight)) {
          continue;
        }
        std::vector<int64_t> matmul_weight_shape;
        if (!GetTensorShape(name, &matmul_weight_shape)) {
          continue;
        }
        int64_t quantize_axis = 1;
        std::vector<float> scale;
        std::vector<int64_t> zeros;
        GetChannelWiseQuantizeInfo(matmul_weight, matmul_weight_shape,
                                   quantize_axis, &scale, &zeros);
        auto scale_node =
            helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT, scale);
        auto zero_node =
            helper_->Constant(ONNX_NAMESPACE::TensorProto::INT8, zeros);
        QuantizeInfo matmul_weight_quantize_info(scale, zeros, scale_node,
                                                 zero_node, quantize_axis);
        helper_->quantize_info[name] = matmul_weight_quantize_info;
      }
      if (!CanBeQuantize(tensor_names)) {
        tensor_names.pop_back();
        if (!CanBeQuantize(tensor_names)) {
          continue;
        }
      }
      for (auto& name : tensor_names) {
        AppendQuantizeTensor(name);
      }
    }
    if (node->op_type() == "Mul") {
      std::vector<std::string> tensor_names = {node->input(0), node->input(1),
                                               node->output(0)};
      if (!CanBeQuantize(tensor_names)) {
        continue;
      }
      for (auto& name : tensor_names) {
        AppendQuantizeTensor(name);
      }
    }
  }
  // update name2node_dict for the change of Relu op.
  UpdateInputNameToNodes();
  // Add QDQ in model
  AddQDQInModel(tensors_to_be_quantize);
}

void QuantizeModelProcessor::AddQDQInModel(
    const std::vector<std::string>& tensors_to_be_quantize) {
  // add Q and DQ according to tensors_to_be_quantize
  for (auto& name : tensors_to_be_quantize) {
    if (IsGraphOutput(name)) {
      continue;
    }
    Assert(helper_->quantize_info.find(name) != helper_->quantize_info.end(),
           "[QuantizeModelProcessor] Can not find quantize info for tensor: " +
               name);
    QuantizeInfo quantize_info = helper_->quantize_info[name];
    std::string scale_node = quantize_info.scale_node_;
    std::string zeros_node = quantize_info.zeros_node_;
    int64_t quantize_axis = quantize_info.quantize_axis_;
    auto iter = std::find(only_dequantize_tensors.begin(),
                          only_dequantize_tensors.end(), name);
    if (iter != only_dequantize_tensors.end()) {
      // if only add DequantizeLinear
      std::vector<float> scale = quantize_info.scale_;
      std::vector<float> bias;
      Assert(GetTensorByName(name, &bias),
             "[QuantizeModelProcessor] Can not find bias value: " + name);
      std::vector<int32_t> new_bias(scale.size(), 0);
      for (int64_t i = 0; i < bias.size(); i++) {
        float scale_val = scale.size() == 1 ? scale[0] : scale[i];
        new_bias[i] = rint(bias[i] / scale_val);
      }

      Weight updated_bias;
      std::vector<int64_t> bias_shape = {static_cast<int64_t>(new_bias.size())};
      updated_bias.set(P2ODataType::INT32, bias_shape, new_bias);
      helper_->updated_params[name] = updated_bias;
      auto dq_node =
          helper_->MakeNode("DequantizeLinear", {name, scale_node, zeros_node});
      if (helper_->GetOpsetVersion() >= 13) {
        AddAttribute(dq_node, "axis", quantize_axis);
      }
      ReplaceInputOfAllNodes(name, dq_node->output(0));
    } else {
      // Handle the following situations
      //           conv                   conv
      //         /  |  \         ->     /    \ 
      //      conv conv scale         DQD   scale
      //                             /   \       
      //                           conv conv
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> except_nodes;
      auto next_nodes = name2node_dict_[name];
      if (next_nodes.size() > 1) {
        for (auto& node : next_nodes) {
          auto iter =
              std::find(supported_quantize_type_.begin(),
                        supported_quantize_type_.end(), node->op_type());
          if (iter == supported_quantize_type_.end()) {
            except_nodes.push_back(node);
          }
        }
      }
      // When all the outputs of this tensor cannot be renamed,
      // it means that the quantization OP will be merged
      if (next_nodes.size() == except_nodes.size()) {
        except_nodes.clear();
      }
      auto q_node =
          helper_->MakeNode("QuantizeLinear", {name, scale_node, zeros_node});
      if (helper_->GetOpsetVersion() >= 13) {
        AddAttribute(q_node, "axis", quantize_axis);
      }
      auto dq_node = helper_->MakeNode(
          "DequantizeLinear", {q_node->output(0), scale_node, zeros_node});
      if (helper_->GetOpsetVersion() >= 13) {
        AddAttribute(dq_node, "axis", quantize_axis);
      }
      ReplaceInputOfAllNodes(name, dq_node->output(0), except_nodes);
    }
  }
}

void QuantizeModelProcessor::MergeConvBN() {
  UpdateInputNameToNodes();
  for (auto iter = nodes_->begin(); iter < nodes_->end(); iter++) {
    auto conv_node = *iter;
    if (conv_node->op_type() != "Conv") {
      continue;
    }

    bool act_has_quantize_info =
        helper_->quantize_info.find(conv_node->input(0)) !=
        helper_->quantize_info.end();
    if (!act_has_quantize_info) {
      continue;
    }
    auto next_nodes = name2node_dict_[conv_node->output(0)];

    if (next_nodes.size() > 1 || IsGraphOutput(conv_node->output(0))) {
      continue;
    }

    auto bn_node = next_nodes[0];
    if (bn_node->op_type() != "BatchNormalization" ||
        IsGraphOutput(bn_node->output(0))) {
      continue;
    }

    std::vector<float> conv_weight;
    Assert(GetTensorByName(conv_node->input(1), &conv_weight),
           "Can not get " + conv_node->input(1) + " from Conv.");

    std::vector<float> bn_scale;
    Assert(GetTensorByName(bn_node->input(1), &bn_scale),
           "Can not get " + bn_node->input(1) + " from BN.");

    std::vector<float> bn_bias;
    Assert(GetTensorByName(bn_node->input(2), &bn_bias),
           "Can not get " + bn_node->input(2) + " from BN.");

    std::vector<float> bn_mean;
    Assert(GetTensorByName(bn_node->input(3), &bn_mean),
           "Can not get " + bn_node->input(3) + " from BN.");

    std::vector<float> bn_var;
    Assert(GetTensorByName(bn_node->input(4), &bn_var),
           "Can not get " + bn_node->input(4) + " from BN.");

    float epsilon = 1;
    for (auto i = 0; i < bn_node->attribute_size(); i++) {
      auto attr = bn_node->attribute(i);
      if (attr.name() == "epsilon") {
        epsilon = attr.f();
      }
    }

    std::vector<float> conv_bias(bn_bias.size(), 0);
    std::string conv_bias_node = conv_node->input(1) + ".merged.bias";
    if (conv_node->input_size() == 3) {
      conv_bias_node = conv_node->input(2);
      conv_bias.clear();
      Assert(GetTensorByName(conv_bias_node, &conv_bias),
             "Can not get " + conv_node->input(2) + " in Conv.");
    }

    // merge conv and bn
    std::vector<float> alpha(bn_scale.size());
    for (int64_t i = 0; i < bn_scale.size(); i++) {
      alpha[i] = bn_scale[i] / sqrt(bn_var[i] + epsilon);
    }

    std::vector<float> new_bias(bn_scale.size());
    for (int64_t i = 0; i < bn_scale.size(); i++) {
      new_bias[i] =
          conv_bias[i] * alpha[i] + (bn_bias[i] - bn_mean[i] * alpha[i]);
    }
    std::vector<float> new_weight(conv_weight.size());
    int64_t offset = conv_weight.size() / bn_bias.size();
    for (int64_t i = 0; i < bn_scale.size(); i++) {
      int64_t outter_offset = i * offset;
      for (int64_t j = 0; j < offset; j++) {
        int64_t index = outter_offset + j;
        new_weight[index] = conv_weight[index] * alpha[i];
      }
    }
    // update weight
    std::vector<int64_t> weight_shape;
    Assert(GetTensorShape(conv_node->input(1), &weight_shape),
           "Can not get the shape of " + conv_node->input(1) + " in Conv.");
    Weight updated_conv_weight;
    updated_conv_weight.set(P2ODataType::FP32, weight_shape, new_weight);
    helper_->updated_params[conv_node->input(1)] = updated_conv_weight;
    // update bias
    Weight updated_bias_weight;
    std::vector<int64_t> bias_shape = {static_cast<int64_t>(new_bias.size())};
    updated_bias_weight.set(P2ODataType::FP32, bias_shape, new_bias);
    helper_->updated_params[conv_bias_node] = updated_bias_weight;
    AppendQuantizeTensor(conv_bias_node, true);
    // update weight scale
    auto quantize_info = helper_->quantize_info[conv_node->input(1)];
    std::string scale_node = quantize_info.scale_node_;
    std::string zero_node = quantize_info.zeros_node_;
    int64_t quantize_axis = quantize_info.quantize_axis_;
    RemoveNodeByName(scale_node);
    RemoveNodeByName(zero_node);
    std::vector<float> scale = quantize_info.scale_;
    std::vector<float> new_scale;
    std::vector<int64_t> new_zeros;
    if (scale.size() == 1) {
      GetTensorWiseQuantizeInfo(new_weight, &new_scale, &new_zeros);
    } else {
      GetChannelWiseQuantizeInfo(new_weight, weight_shape, quantize_axis,
                                 &new_scale, &new_zeros);
    }
    auto weight_scale_node =
        helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT, new_scale);
    auto weight_zero_node =
        helper_->Constant(ONNX_NAMESPACE::TensorProto::INT8, new_zeros);
    QuantizeInfo updated_weight_quantize_info(new_scale, new_zeros,
                                              weight_scale_node,
                                              weight_zero_node, quantize_axis);
    helper_->quantize_info[conv_node->input(1)] = updated_weight_quantize_info;
    // add bias scale and update bias
    auto act_quantize_info = helper_->quantize_info[conv_node->input(0)];
    std::vector<float> act_scale = act_quantize_info.scale_;
    std::vector<float> bias_scale;
    for (int64_t i = 0; i < new_scale.size(); i++) {
      bias_scale.push_back(act_scale[0] * new_scale[i]);
    }
    std::vector<int64_t> bias_zeros(bias_scale.size(), 0);
    auto bias_scale_node =
        helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT, bias_scale);
    auto bias_zero_node =
        helper_->Constant(ONNX_NAMESPACE::TensorProto::INT32, bias_zeros);
    QuantizeInfo bias_quantize_info(bias_scale, bias_zeros, bias_scale_node,
                                    bias_zero_node, 0);
    helper_->quantize_info[conv_bias_node] = bias_quantize_info;
    if (conv_node->input_size() == 2) {
      conv_node->add_input(conv_bias_node);
    }
    // remove BN op
    RemoveNodeByName(bn_node->name());
  }
}

void QuantizeModelProcessor::MergeConvAdd() {
  UpdateInputNameToNodes();
  for (auto iter = nodes_->begin(); iter < nodes_->end(); iter++) {
    auto node = *iter;
    if (node->op_type() != "Conv") {
      continue;
    }
    // if act input of conv does not have quantize info, continue
    bool act_has_quantize_info = helper_->quantize_info.find(node->input(0)) !=
                                 helper_->quantize_info.end();
    if (!act_has_quantize_info) {
      continue;
    }

    // if weight of conv does not have quantize info, continue
    bool weight_has_quantize_info =
        helper_->quantize_info.find(node->input(1)) !=
        helper_->quantize_info.end();
    if (!weight_has_quantize_info) {
      continue;
    }
    auto next_nodes = name2node_dict_[node->output(0)];

    if (next_nodes.size() > 1 || IsGraphOutput(node->output(0))) {
      continue;
    }

    auto next_node = next_nodes[0];
    if (next_node->op_type() != "Add" || IsGraphOutput(next_node->output(0))) {
      continue;
    }
    std::string reshape_node = node->output(0) == next_node->input(0)
                                   ? next_node->input(1)
                                   : next_node->input(0);
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> before_nodes;
    for (auto& node : *nodes_) {
      for (size_t i = 0; i < node->output_size(); ++i) {
        std::string node_output = node->output(i);
        if (node_output == reshape_node) {
          before_nodes.push_back(node);
          break;
        }
      }
    }

    if (before_nodes.size() != 1 || before_nodes[0]->op_type() != "Reshape") {
      continue;
    }

    std::string bias_node = before_nodes[0]->input(0);
    // continue if bias is not a constant
    std::vector<float> bias_val;
    if (!GetTensorByName(bias_node, &bias_val)) {
      continue;
    }

    // continue if shape tensor of reshape op is not a constant
    std::vector<int64_t> shape_val;
    if (!GetTensorByName(before_nodes[0]->input(1), &shape_val)) {
      continue;
    }
    // continue if shape_val != [1, bias_val.size(), 1, 1]
    std::vector<int64_t> target = {1, static_cast<int64_t>(bias_val.size()), 1,
                                   1};
    if (target != shape_val) {
      continue;
    }
    // remove Reshape op
    RemoveNodeByName(before_nodes[0]->name());
    // add scale for bias
    std::vector<float> weight_scale =
        helper_->quantize_info[node->input(1)].scale_;
    std::vector<float> act_scale =
        helper_->quantize_info[node->input(0)].scale_;
    std::vector<float> bias_scale;
    for (int64_t i = 0; i < weight_scale.size(); i++) {
      bias_scale.push_back(weight_scale[i] * act_scale[0]);
    }
    std::vector<int64_t> onnx_zeros(bias_scale.size(), 0);
    auto scale_node =
        helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT, bias_scale);
    auto zero_node =
        helper_->Constant(ONNX_NAMESPACE::TensorProto::INT32, onnx_zeros);

    QuantizeInfo quantize_info(bias_scale, onnx_zeros, scale_node, zero_node,
                               0);

    helper_->quantize_info[bias_node] = quantize_info;
    AppendQuantizeTensor(bias_node, true);
    node->add_input(bias_node);
    RemoveNodeByName(next_node->name());
  }
}

void QuantizeModelProcessor::SortNodes() {
  // return the topo sort of nodes;
  // 1. Get i2o_mapper and  constant_nodes, i2o_mapper means the node map to its
  // all output nodes, constant_nodes save all constant nodes.
  // 2. Nodes without output nodes are first saved to new_nodes, and then
  // cyclically delete the records of the node in i2o_mapper items, and nodes
  // whose output nodes are empty are also saved to new_nodes in turn.
  // 3. Store constant nodes in new_nodes.
  // 4. Reverse new_nodes, then assign to nodes.
  std::map<std::string, std::vector<std::string>> i2o_mapper;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> constant_nodes;
  std::map<std::string, std::shared_ptr<ONNX_NAMESPACE::NodeProto>>
      name2node_mapper;
  for (int64_t i = 0; i < nodes_->size(); i++) {
    auto node = (*nodes_)[i];
    if (node->op_type() == "Constant") {
      constant_nodes.push_back(node);
      continue;
    }
    name2node_mapper[node->name()] = node;
    for (int64_t in_index = 0; in_index < node->input_size(); in_index++) {
      std::string input = node->input(in_index);
      for (int64_t j = 0; j < nodes_->size(); j++) {
        if (i == j) {
          continue;
        }
        auto input_node = (*nodes_)[j];
        if (input_node->op_type() == "Constant") {
          continue;
        }
        for (int64_t out_index = 0; out_index < input_node->output_size();
             out_index++) {
          if (input == input_node->output(out_index)) {
            if (i2o_mapper.find(input_node->name()) == i2o_mapper.end()) {
              i2o_mapper[input_node->name()] = {node->name()};
            } else {
              auto iter =
                  std::find(i2o_mapper[input_node->name()].begin(),
                            i2o_mapper[input_node->name()].end(), node->name());
              if (iter == i2o_mapper[input_node->name()].end()) {
                i2o_mapper[input_node->name()].push_back(node->name());
              }
            }
          }
        }
      }
    }
  }
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> new_nodes;

  for (int64_t i = 0; i < nodes_->size(); i++) {
    auto node_name = (*nodes_)[i]->name();
    auto node = (*nodes_)[i];
    if (node->op_type() == "Constant") {
      continue;
    }
    if (i2o_mapper.find(node_name) == i2o_mapper.end()) {
      new_nodes.push_back(node);
    }
  }
  int64_t index = 0;
  while (index < new_nodes.size()) {
    auto current_node = new_nodes[index];
    std::string current_node_name = current_node->name();
    for (auto iter = i2o_mapper.begin(); iter != i2o_mapper.end(); iter++) {
      std::string input_node_name = iter->first;
      std::vector<std::string>* output_nodes_name = &iter->second;
      if (output_nodes_name->empty()) {
        continue;
      }
      auto in_inter = std::find(output_nodes_name->begin(),
                                output_nodes_name->end(), current_node_name);
      if (in_inter != output_nodes_name->end()) {
        output_nodes_name->erase(in_inter);
      }
      if (output_nodes_name->empty()) {
        new_nodes.push_back(name2node_mapper[input_node_name]);
      }
    }
    index++;
  }

  for (auto& node : constant_nodes) {
    new_nodes.push_back(node);
  }
  std::reverse(new_nodes.begin(), new_nodes.end());
  Assert(nodes_->size() == new_nodes.size(),
         "The number of nodes after topological sorting is not equal to the "
         "number before sorting");
  *nodes_ = new_nodes;
}

void QuantizeModelProcessor::RemoveAllQuantizeOps() {
  UpdateInputNameToNodes();
  for (auto iter = nodes_->begin(); iter < nodes_->end(); iter++) {
    auto node = *iter;
    if (node->op_type() != "QuantizeLinear") {
      continue;
    }
    auto next_node_names = name2node_dict_[node->output(0)];

    if (next_node_names.empty() || !next_node_names[0]->has_op_type() ||
        next_node_names[0]->op_type() != "DequantizeLinear") {
      continue;
    }
    std::string input_name = node->input(0);
    RemoveNodeByName(node->name());
    std::string output_name = next_node_names[0]->output(0);
    RemoveNodeByName(next_node_names[0]->name());
    ReplaceInputOfAllNodes(output_name, input_name);
  }
}

// Broadcast quantize info between the input and output of the OPs that will not
// change quantize info
void QuantizeModelProcessor::QuantizeInfoBroadcast() {
  UpdateInputNameToNodes();
  for (auto iter = nodes_->begin(); iter < nodes_->end(); iter++) {
    auto node = *iter;
    if (node->op_type() != "Identity") {
      continue;
    }
    std::string input_name = node->input(0);
    std::string output_name = node->output(0);
    auto input_quantize_info_iter = helper_->quantize_info.find(input_name);
    auto output_quantize_info_iter = helper_->quantize_info.find(output_name);
    // The input and output of Identity do not have quantize info
    if (input_quantize_info_iter == helper_->quantize_info.end() &&
        output_quantize_info_iter == helper_->quantize_info.end()) {
      continue;
    }
    // The input and output of Identity have quantize info
    if (input_quantize_info_iter != helper_->quantize_info.end() &&
        output_quantize_info_iter != helper_->quantize_info.end()) {
      continue;
    }
    if (input_quantize_info_iter != helper_->quantize_info.end()) {
      helper_->quantize_info[output_name] = helper_->quantize_info[input_name];
    } else if (output_quantize_info_iter != helper_->quantize_info.end()) {
      helper_->quantize_info[input_name] = helper_->quantize_info[output_name];
    }
    if (ConnectToOutput(output_name)) {
      continue;
    }
    RemoveNodeByName(node->name());
    iter--;
  }
}

bool QuantizeModelProcessor::IsGraphOutput(const std::string& name) {
  for (auto& item : *outputs_) {
    auto out_node = (*item.get());
    if (name == out_node.name()) {
      return true;
    }
  }
  return false;
}

// Try get tensor shape value
bool QuantizeModelProcessor::GetTensorShape(const std::string& name,
                                            std::vector<int64_t>* shape) {
  for (auto& item : *parameters_) {
    auto node = *(item.get());
    if (node.output(0) != name) {
      continue;
    }
    for (auto i = 0; i < node.attribute_size(); i++) {
      auto attr = node.attribute(i);
      if (attr.name() == "value") {
        auto tensor = attr.mutable_t();
        for (int64_t i = 0; i < tensor->dims_size(); i++) {
          shape->push_back(tensor->dims(i));
        }
      }
    }
  }
  return !shape->empty();
}

void QuantizeModelProcessor::GetTensorWiseQuantizeInfo(
    const std::vector<float>& tensor, std::vector<float>* scale,
    std::vector<int64_t>* zero) {
  float max_val = -1;
  for (int64_t i = 0; i < tensor.size(); i++) {
    if (fabs(tensor[i]) > max_val) {
      max_val = fabs(tensor[i]);
    }
  }
  Assert(max_val >= 0,
         "[GetTensorWiseQuantizeInfo] Require the scale >= 0, but now it's " +
             std::to_string(max_val) + ".");
  scale->push_back(max_val / 127);
  zero->push_back(0);
}

void QuantizeModelProcessor::GetChannelWiseQuantizeInfo(
    const std::vector<float>& tensor, const std::vector<int64_t>& shape,
    const int64_t& quant_axis, std::vector<float>* scale,
    std::vector<int64_t>* zero) {
  int64_t channel_count = shape[quant_axis];

  for (int64_t i = 0; i < channel_count; i++) {
    if (quant_axis == 0) {
      float max_val = -1;
      int64_t inner_offset = 1;
      for (auto& j : shape) {
        inner_offset *= j;
      }
      inner_offset /= channel_count;
      int64_t index = i * inner_offset;
      for (int64_t j = 0; j < inner_offset; j++) {
        if (fabs(tensor[index + j]) > max_val) {
          max_val = fabs(tensor[index + j]);
        }
      }
      Assert(
          max_val >= 0,
          "[GetChannelWiseQuantizeInfo] Require the scale >= 0, but now it's " +
              std::to_string(max_val) + ".");
      scale->push_back(max_val / 127);
      zero->push_back(0);
    } else if (quant_axis == 1) {
      float max_val = -1;
      int64_t inner_offset = shape.size() == 4 ? shape[2] * shape[3] : 1;
      for (int64_t outter = 0; outter < shape[0]; outter++) {
        int64_t index = outter * channel_count * inner_offset;
        for (int64_t inner = 0; inner < inner_offset; inner++) {
          int64_t final_index = index + i * inner_offset + inner;
          if (fabs(tensor[final_index]) > max_val) {
            max_val = fabs(tensor[final_index]);
          }
        }
      }
      Assert(
          max_val >= 0,
          "[GetChannelWiseQuantizeInfo] Require the scale >= 0, but now it's " +
              std::to_string(max_val) + ".");
      scale->push_back(max_val / 127);
      zero->push_back(0);
    } else {
      Assert(false,
             "QuantizeModelProcessor::GetChannelWiseQuantizeInfo only supports "
             "quant_axis equals to 0 or 1, but now it's " +
                 std::to_string(quant_axis) + ".");
    }
  }
}

template <typename T>
bool QuantizeModelProcessor::GetTensorByName(const std::string& name,
                                             std::vector<T>* value) {
  // Find tensor values in the following order, if found, store the data in
  // value, and return true：
  // 1. updated_parameters, the weight of conv or matmul.
  // 2. parameters of original graph, the scale or bias of BN.
  // 3. constant node in nodes, other vals.
  auto updated_params_iter = helper_->updated_params.find(name);
  if (updated_params_iter != helper_->updated_params.end()) {
    (updated_params_iter->second).get(value);
    return true;
  }
  for (int64_t block_index = 0; block_index < parser_->NumOfBlocks();
       block_index++) {
    if (parser_->TryGetTensorValue(block_index, name, value)) {
      return true;
    }
  }
  return helper_->TryGetTensorValue(name, value);
}

bool QuantizeModelProcessor::ConnectToOutput(const std::string& output_name) {
  std::vector<std::string> names = {output_name};
  while (!names.empty()) {
    std::string name = names[names.size() - 1];
    names.pop_back();
    if (IsGraphOutput(name)) {
      return true;
    }
    auto next_nodes = name2node_dict_[name];
    for (auto& next : next_nodes) {
      if (next->op_type() == "Identity") {
        names.push_back(next->output(0));
      }
    }
  }
  return false;
}

bool QuantizeModelProcessor::CanBeQuantize(
    const std::vector<std::string>& tensor_names,
    const std::vector<int64_t>& output_index) {
  for (auto& tensor : tensor_names) {
    if (helper_->quantize_info.find(tensor) == helper_->quantize_info.end()) {
      return false;
    }
  }
  // If there is an OP linked to the output by identity, it needs to be skipped,
  // do not quantize the OP
  for (auto i = 0; i < output_index.size(); i++) {
    int64_t index = output_index[i];
    if (index == -1) {
      index = tensor_names.size() - 1;
    }

    std::string output_name = tensor_names[index];
    if (ConnectToOutput(output_name)) {
      return false;
    }
  }
  return true;
}

void QuantizeModelProcessor::AppendQuantizeTensor(const std::string& tensor,
                                                  const bool& only_dequantize) {
  if (only_dequantize) {
    if (std::find(only_dequantize_tensors.begin(),
                  only_dequantize_tensors.end(),
                  tensor) == only_dequantize_tensors.end()) {
      only_dequantize_tensors.push_back(tensor);
    }
  } else {
    if (std::find(tensors_to_be_quantize.begin(), tensors_to_be_quantize.end(),
                  tensor) == tensors_to_be_quantize.end()) {
      tensors_to_be_quantize.push_back(tensor);
    }
  }
}
}  // namespace paddle2onnx
