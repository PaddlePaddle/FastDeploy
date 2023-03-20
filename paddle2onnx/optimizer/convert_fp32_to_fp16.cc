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

#include "paddle2onnx/optimizer/convert_fp32_to_fp16.h"
#include "paddle2onnx/utils/utils.h"

namespace paddle2onnx {

void ConvertFp32ToFp16::ConvertValToFloat16(float val, uint16_t* x) {
  // Conversion routine adapted from
  // http://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
  Bits v, s;
  v.f = val;
  uint32_t sign = v.si & sigN;
  v.si ^= sign;
  sign >>= shiftSign;  // logical shift
  s.si = mulN;
  s.si = s.f * v.f;  // correct subnormals
  v.si ^= (s.si ^ v.si) & -(minN > v.si);
  v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
  v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
  v.ui >>= shift;  // logical shift
  v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
  v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
  *x = v.ui | sign;
}

void ConvertFp32ToFp16::SortNodes(ONNX_NAMESPACE::ModelProto* model) {
  // return the topo sort of nodes;
  // 1. Get i2o_mapper and  constant_nodes, i2o_mapper means the node map to its
  // all output nodes, constant_nodes save all constant nodes.
  // 2. Nodes without output nodes are first saved to new_nodes, and then
  // cyclically delete the records of the node in i2o_mapper items, and nodes
  // whose output nodes are empty are also saved to new_nodes in turn.
  // 3. Store constant nodes in new_nodes.
  // 4. Reverse new_nodes, then assign to nodes.
  auto graph = model->mutable_graph();

  // means the node map to its all output nodes
  std::map<std::string, std::vector<std::string>> i2o_mapper;
  // constant_nodes save all constant nodes.
  std::vector<ONNX_NAMESPACE::NodeProto> constant_nodes;
  // name map to its node
  std::map<std::string, ONNX_NAMESPACE::NodeProto> name2node_mapper;
  for (int64_t i = 0; i < graph->node_size(); i++) {
    auto node = graph->mutable_node(i);
    if (node->op_type() == "Constant") {
      constant_nodes.push_back(*node);
      continue;
    }
    name2node_mapper[node->name()] = *node;
    for (int64_t in_index = 0; in_index < node->input_size(); in_index++) {
      std::string input = node->input(in_index);
      for (int64_t j = 0; j < graph->node_size(); j++) {
        if (i == j) {
          continue;
        }
        auto input_node = graph->mutable_node(j);
        if (input_node->op_type() == "Constant") {
          continue;
        }
        for (int64_t out_index = 0; out_index < input_node->output_size();
             out_index++) {
          // find the pre node
          if (input == input_node->output(out_index)) {
            // does not find other input node before
            if (i2o_mapper.find(input_node->name()) == i2o_mapper.end()) {
              i2o_mapper[input_node->name()] = {node->name()};
            } else {
              auto iter =
                  std::find(i2o_mapper[input_node->name()].begin(),
                            i2o_mapper[input_node->name()].end(), node->name());
              // not been found before
              if (iter == i2o_mapper[input_node->name()].end()) {
                i2o_mapper[input_node->name()].push_back(node->name());
              }
            }
          }
        }
      }
    }
  }

  // Store topologically sorted nodes
  std::vector<ONNX_NAMESPACE::NodeProto> new_nodes;

  for (int64_t i = 0; i < graph->node_size(); i++) {
    auto node = graph->mutable_node(i);
    auto node_name = node->name();
    if (node->op_type() == "Constant") {
      continue;
    }
    // Store those nodes that have no output first.
    if (i2o_mapper.find(node_name) == i2o_mapper.end()) {
      new_nodes.push_back(*node);
    }
  }

  int64_t index = 0;
  while (index < new_nodes.size()) {
    auto current_node = new_nodes[index];
    std::string current_node_name = current_node.name();
    for (auto iter = i2o_mapper.begin(); iter != i2o_mapper.end(); iter++) {
      std::string input_node_name = iter->first;
      std::vector<std::string>* output_nodes_name = &iter->second;
      if (output_nodes_name->empty()) {
        continue;
      }
      auto in_inter = std::find(output_nodes_name->begin(),
                                output_nodes_name->end(), current_node_name);
      // if find the pre node, erase current node name in i2o_mapper
      if (in_inter != output_nodes_name->end()) {
        output_nodes_name->erase(in_inter);
      }
      // if find on node that have no output, store it
      if (output_nodes_name->empty()) {
        new_nodes.push_back(name2node_mapper[input_node_name]);
      }
    }
    index++;
  }

  // store all constant node finally
  for (auto& node : constant_nodes) {
    new_nodes.push_back(node);
  }

  // reverse the sorted nodes
  std::reverse(new_nodes.begin(), new_nodes.end());

  Assert(model->mutable_graph()->node_size() == new_nodes.size(),
         "The number of nodes after topological sorting is not equal to the "
         "number before sorting");
  // copy all new_nodes to graph
  for (int64_t i = 0; i < graph->node_size(); i++) {
    auto node = graph->mutable_node(i);
    node->CopyFrom(new_nodes[i]);
  }
}

std::string ConvertFp32ToFp16::GenName(const std::string& prefix) {
  int64_t name_index = 0;
  auto iter = name_index_mapper.find(prefix);
  if (iter != name_index_mapper.end()) {
    name_index = iter->second;
    name_index_mapper[prefix]++;
  } else {
    name_index_mapper[prefix] = 1;
  }
  return prefix + std::to_string(name_index);
}

ONNX_NAMESPACE::ValueInfoProto* ConvertFp32ToFp16::MakeValueInfoFromTensor(
    const ONNX_NAMESPACE::TensorProto& tensor) {
  ONNX_NAMESPACE::ValueInfoProto* value_info =
      new ONNX_NAMESPACE::ValueInfoProto();
  value_info->set_name(tensor.name());
  auto type_proto = value_info->mutable_type();
  auto tensor_type_proto = type_proto->mutable_tensor_type();
  tensor_type_proto->set_elem_type(tensor.data_type());  // TODO
  auto shape = tensor_type_proto->mutable_shape();
  for (auto i = 0; i < tensor.dims_size(); i++) {
    auto dim = tensor.dims(i);
    if (dim < 0) {
      auto dynamic_dim_name = GenName("DynamicDimension");
      shape->add_dim()->set_dim_param(dynamic_dim_name);
    } else {
      shape->add_dim()->set_dim_value(dim);
    }
  }
  return value_info;
}

ONNX_NAMESPACE::NodeProto* ConvertFp32ToFp16::MakeCastNode(
    const std::string& op_name, const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs, int32_t to_dtype) {
  ONNX_NAMESPACE::NodeProto* node = new ONNX_NAMESPACE::NodeProto();
  node->set_name(op_name);
  node->set_op_type("Cast");
  for (size_t i = 0; i < inputs.size(); ++i) {
    node->add_input(inputs[i]);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    node->add_output(outputs[i]);
  }
  auto attr = node->add_attribute();
  attr->set_name("to");
  attr->set_i(static_cast<int>(to_dtype));
  attr->set_type(ONNX_NAMESPACE::AttributeProto::INT);
  return node;
}

bool ConvertFp32ToFp16::GetTensorValue(
    const ONNX_NAMESPACE::TensorProto& tensor, std::vector<float>* value) {
  auto dtype = tensor.data_type();
  if (dtype != ONNX_NAMESPACE::TensorProto::FLOAT) {
    return false;
  }
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < tensor.dims_size(); i++) {
    shape.push_back(tensor.dims(i));
  }
  int64_t nums = 1;
  for (auto& i : shape) nums *= i;
  value->resize(nums);
  memcpy(value->data(), tensor.raw_data().data(), nums * sizeof(float));
  return value->size();
}

// When the value of a tensor is greater than 10000, it is reserved as FP32 and
// not converted.
bool ConvertFp32ToFp16::KeepNodeType(ONNX_NAMESPACE::NodeProto* node) {
  auto KeepType = [=](const ONNX_NAMESPACE::TensorProto& tensor) {
    std::vector<float> fp32_val;
    GetTensorValue(tensor, &fp32_val);
    for (auto i = 0; i < fp32_val.size(); i++) {
      if (fp32_val[i] > 10000) {
        return true;
      }
    }
    return false;
  };

  for (auto attr_index = 0; attr_index < node->attribute_size(); attr_index++) {
    auto attr = node->attribute(attr_index);
    if (attr.has_t() && KeepType(attr.t())) {
      return true;
    }
    for (auto t_index = 0; t_index < attr.tensors_size(); t_index++) {
      if (KeepType(attr.tensors(t_index))) {
        return true;
      }
    }
  }
  return false;
}

void ConvertFp32ToFp16::ConvertTensorFloatToFloat16(
    ONNX_NAMESPACE::TensorProto* tensor) {
  if (tensor->data_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
    if (tensor->float_data_size()) {
      Assert(false, "No implemented! Please raise an issue to us.");
    }
    if (tensor->has_raw_data()) {
      std::vector<float> fp32_val;
      GetTensorValue(*tensor, &fp32_val);
      if (fp32_val.empty()) {
        return;
      }
      converted_attr++;
      tensor->set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT16);

      std::vector<uint16_t> fp16_val(fp32_val.size(), 0);

      float pos_min_val = max_finite_val_;
      float pos_max_val = min_positive_val_;
      float neg_min_val = -1 * max_finite_val_;
      float neg_max_val = -1 * min_positive_val_;

      for (auto i = 0; i < fp32_val.size(); i++) {
        if (0 < fp32_val[i] && fp32_val[i] < min_positive_val_) {
          if (fp32_val[i] < pos_min_val) {
            pos_min_val = fp32_val[i];
          }
          fp32_val[i] = min_positive_val_;
        } else if (0 > fp32_val[i] && fp32_val[i] > -1 * min_positive_val_) {
          if (fp32_val[i] > neg_min_val) {
            neg_min_val = fp32_val[i];
          }
          fp32_val[i] = -1 * min_positive_val_;
        } else if (fp32_val[i] > max_finite_val_) {
          if (fp32_val[i] > pos_max_val) {
            pos_max_val = fp32_val[i];
          }
          fp32_val[i] = max_finite_val_;
        } else if (fp32_val[i] < -1 * max_finite_val_) {
          if (fp32_val[i] < neg_max_val) {
            neg_max_val = fp32_val[i];
          }
          fp32_val[i] = -1 * max_finite_val_;
        }
        ConvertValToFloat16(fp32_val[i], &fp16_val[i]);
      }
      if (pos_min_val < max_finite_val_ - 1) {
        P2OLogger() << "[Info] the float32 number: " << pos_min_val
                    << " will be truncated to: " << min_positive_val_
                    << std::endl;
      }
      if (pos_max_val > min_positive_val_ + 1) {
        P2OLogger() << "[Info] the float32 number: " << pos_max_val
                    << " will be truncated to: " << max_finite_val_
                    << std::endl;
      }
      if (neg_min_val > -1 * max_finite_val_ + 1) {
        P2OLogger() << "[Info] the float32 number: " << neg_min_val
                    << " will be truncated to: " << -1 * min_positive_val_
                    << std::endl;
      }
      if (neg_max_val < -1 * min_positive_val_ - 1) {
        P2OLogger() << "[Info] the float32 number: " << neg_max_val
                    << " will be truncated to: " << -1 * max_finite_val_
                    << std::endl;
      }
      tensor->set_raw_data(std::string((const char*)(fp16_val.data()),
                                       fp16_val.size() * sizeof(uint16_t)));
    }
  }
}

// return if the next node of name is Cast and its attr type is dtype.
bool ConvertFp32ToFp16::CastedTo(const std::string& name,
                                 ONNX_NAMESPACE::ModelProto& model,
                                 int64_t dtype) {
  auto graph = model.mutable_graph();
  std::vector<ONNX_NAMESPACE::NodeProto*> next_nodes;
  for (auto i = 0; i < graph->node_size(); i++) {
    auto n = graph->mutable_node(i);
    for (auto i_index = 0; i_index < n->input_size(); i_index++) {
      std::string input = n->input(i_index);
      if (name == input) {
        next_nodes.push_back(n);
      }
    }
  }
  bool casted = false;
  for (auto node : next_nodes) {
    if (node->op_type() == "Cast") {
      for (auto attr_index = 0; attr_index < node->attribute_size();
           attr_index++) {
        if (node->attribute(attr_index).has_i() &&
            node->attribute(attr_index).i() == dtype) {
          casted = true;
          break;
        }
      }
    }
  }
  return casted;
}

// return if the pre node of name is Cast and its attr type is dtype.
bool ConvertFp32ToFp16::CastedFrom(const std::string& name,
                                   ONNX_NAMESPACE::ModelProto& model,
                                   int64_t dtype) {
  auto graph = model.mutable_graph();
  std::vector<ONNX_NAMESPACE::NodeProto*> pre_nodes;
  for (auto i = 0; i < graph->node_size(); i++) {
    auto n = graph->mutable_node(i);
    for (auto o_index = 0; o_index < n->output_size(); o_index++) {
      std::string output = n->output(o_index);
      if (name == output) {
        pre_nodes.push_back(n);
      }
    }
  }
  bool casted = false;
  for (auto node : pre_nodes) {
    if (node->op_type() == "Cast") {
      for (auto attr_index = 0; attr_index < node->attribute_size();
           attr_index++) {
        if (node->attribute(attr_index).has_i() &&
            node->attribute(attr_index).i() == dtype) {
          casted = true;
          break;
        }
      }
    }
  }
  return casted;
}

// return if the name is the input of DEFAULT_OP_BLOCK_LIST
bool ConvertFp32ToFp16::IsInputOfOpBlock(const std::string& name,
                                         ONNX_NAMESPACE::ModelProto& model) {
  auto graph = model.mutable_graph();
  for (auto i = 0; i < graph->node_size(); i++) {
    auto n = graph->mutable_node(i);
    if (std::find(op_block_list_.begin(), op_block_list_.end(), n->op_type()) ==
        op_block_list_.end()) {
      continue;
    }

    for (auto i_index = 0; i_index < n->input_size(); i_index++) {
      std::string input = n->input(i_index);
      if (name == input) {
        return true;
      }
    }
  }
  return false;
}

bool ConvertFp32ToFp16::IsOutputOfOpBlockAndFP32Out(
    const std::string& name, ONNX_NAMESPACE::ModelProto& model) {
  auto graph = model.mutable_graph();
  for (auto i = 0; i < graph->node_size(); i++) {
    auto n = graph->mutable_node(i);
    if (std::find(op_block_list_.begin(), op_block_list_.end(), n->op_type()) ==
            op_block_list_.end() &&
        std::find(fp32_output_op_list.begin(), fp32_output_op_list.end(),
                  n->op_type()) == fp32_output_op_list.end()) {
      continue;
    }
    for (auto o_index = 0; o_index < n->output_size(); o_index++) {
      std::string output = n->output(o_index);
      if (name == output) {
        return true;
      }
    }
  }
  return false;
}

void ConvertFp32ToFp16::KeepIoType(ONNX_NAMESPACE::ModelProto* model) {
  auto graph = model->mutable_graph();
  for (auto i = 0; i < graph->input_size(); i++) {
    auto input = graph->input(i);
    if (input.type().tensor_type().elem_type() ==
        ONNX_NAMESPACE::TensorProto::FLOAT) {
      // if the pre node is cast, and it is cast to float16, we do not need add
      // Cast OP any more
      if (CastedTo(input.name(), *model, 10)) {
        graph_io_to_skip.push_back(input.name());
        continue;
      }
      std::string output_name = "graph_input_cast_" + std::to_string(i);
      name_mapping[input.name()] = output_name;
      graph_io_to_skip.push_back(input.name());
      std::string node_name = "graph_input_cast" + std::to_string(i);
      auto new_value_info = graph->add_value_info();
      new_value_info->CopyFrom(input);
      new_value_info->set_name(output_name);
      new_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
          ONNX_NAMESPACE::TensorProto::FLOAT16);
      auto new_node =
          MakeCastNode(node_name, {input.name()}, {output_name}, 10);
      *(graph->add_node()) = (*new_node);
      value_info_list.push_back(new_value_info);
      io_casts.push_back(node_name);
    }
  }
  for (auto i = 0; i < graph->output_size(); i++) {
    auto output = graph->output(i);
    if (output.type().tensor_type().elem_type() ==
        ONNX_NAMESPACE::TensorProto::FLOAT) {
      // if the next node is cast, and it is cast to float, we do not need add
      // Cast OP any more
      if (CastedFrom(output.name(), *model, 1)) {
        graph_io_to_skip.push_back(output.name());
        continue;
      }
      std::string output_name = "graph_output_cast_" + std::to_string(i);
      name_mapping[output.name()] = output_name;
      graph_io_to_skip.push_back(output.name());
      std::string node_name = "graph_output_cast" + std::to_string(i);
      auto new_value_info = graph->add_value_info();
      new_value_info->CopyFrom(output);
      new_value_info->set_name(output_name);
      new_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
          ONNX_NAMESPACE::TensorProto::FLOAT16);
      auto new_node =
          MakeCastNode(node_name, {output_name}, {output.name()}, 1);
      *(graph->add_node()) = (*new_node);
      value_info_list.push_back(new_value_info);
      io_casts.push_back(node_name);
    }
  }
}

void ConvertFp32ToFp16::ConvertAttribute(ONNX_NAMESPACE::ModelProto* model) {
  proto_node new_node(*model);
  queue.push_back(new_node);

  while (queue.size()) {
    next_level.clear();
    for (auto q : queue) {
      // process model proto
      if (q.node_type == "model" && model->has_graph()) {
        proto_node new_node(model->mutable_graph());
        next_level.push_back(new_node);
      }
      // process graph proto
      if (q.node_type == "graph") {
        for (auto i = 0; i < q.graph->node_size(); i++) {
          auto n = q.graph->mutable_node(i);
          if (std::find(io_casts.begin(), io_casts.end(), n->name()) !=
              io_casts.end()) {
            continue;
          }
          for (auto i_index = 0; i_index < n->input_size(); i_index++) {
            std::string* input = n->mutable_input(i_index);
            auto iter = name_mapping.find(*input);
            if (iter != name_mapping.end()) {
              *input = iter->second;
            }
          }
          for (auto o_index = 0; o_index < n->output_size(); o_index++) {
            std::string* output = n->mutable_output(o_index);
            auto iter = name_mapping.find(*output);
            if (iter != name_mapping.end()) {
              *output = iter->second;
            }
          }
          // If the op type is in op_block_list_ or fp32_output_op_list,
          // or needs to be kept without conversion, then store the node in
          // node_list,
          // which is convenient for adding cast op in the front or back
          if (KeepNodeType(n)) {
            if (n->op_type() != "Constant" &&
                std::find(node_list.begin(), node_list.end(), n) ==
                    node_list.end()) {
              node_list.push_back(n);
            } else {
              for (auto index = 0; index < q.graph->node_size(); index++) {
                auto keep_type_node = q.graph->mutable_node(index);
                bool is_pre_node =
                    std::find(keep_type_node->input().begin(),
                              keep_type_node->input().end(),
                              n->output()[0]) != keep_type_node->input().end();
                if (is_pre_node &&
                    std::find(node_list.begin(), node_list.end(), n) ==
                        node_list.end()) {
                  node_list.push_back(keep_type_node);
                  Assert(
                      n->op_type() == "Constant",
                      "The node type be Constant, but it is: " + n->op_type());
                  keep_type_tensors.push_back(n->output()[0]);
                }
              }
            }
          } else if ((std::find(op_block_list_.begin(), op_block_list_.end(),
                                n->op_type()) != op_block_list_.end() ||
                      std::find(fp32_output_op_list.begin(),
                                fp32_output_op_list.end(),
                                n->op_type()) != fp32_output_op_list.end()) &&
                     std::find(node_list.begin(), node_list.end(), n) ==
                         node_list.end()) {
            node_list.push_back(n);
          } else {
            std::string op_name = n->name();
            if (n->op_type() == "Cast") {
              for (auto attr_index = 0; attr_index < n->attribute_size();
                   attr_index++) {
                auto attr = n->mutable_attribute(attr_index);
                if (attr->name() == "to" && attr->i() == 1) {
                  attr->set_i(10);
                  break;
                }
              }
            }
            for (auto attr_index = 0; attr_index < n->attribute_size();
                 attr_index++) {
              proto_node new_node(n->mutable_attribute(attr_index));
              next_level.push_back(new_node);
            }
          }
        }
      }

      // process attribute proto
      if (q.node_type == "attribute") {
        if (q.attr->has_g()) {
          proto_node new_node(q.attr->mutable_g());
          next_level.push_back(new_node);
        }

        for (auto g_index = 0; g_index < q.attr->graphs_size(); g_index++) {
          proto_node new_node(q.attr->mutable_graphs(g_index));
          next_level.push_back(new_node);
        }
        if (q.attr->has_t()) {
          ConvertTensorFloatToFloat16(q.attr->mutable_t());
        }

        for (auto t_index = 0; t_index < q.attr->tensors_size(); t_index++) {
          ConvertTensorFloatToFloat16(q.attr->mutable_tensors(t_index));
        }
      }

      // process graph proto
      if (q.node_type == "graph") {
        for (auto init_index = 0; init_index < q.graph->initializer_size();
             init_index++) {
          auto init = q.graph->mutable_initializer(init_index);
          if (init->data_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
            ConvertTensorFloatToFloat16(init);
            auto new_value = MakeValueInfoFromTensor(*init);
            value_info_list.push_back(new_value);
          }
        }
        for (auto i_index = 0; i_index < q.graph->input_size(); i_index++) {
          auto input = q.graph->mutable_input(i_index);
          bool skip =
              std::find(graph_io_to_skip.begin(), graph_io_to_skip.end(),
                        input->name()) != graph_io_to_skip.end();
          if (!skip &&
              input->type().tensor_type().elem_type() ==
                  ONNX_NAMESPACE::TensorProto::FLOAT) {
            input->mutable_type()->mutable_tensor_type()->set_elem_type(
                ONNX_NAMESPACE::TensorProto::FLOAT16);
            value_info_list.push_back(input);
          }
        }
        for (auto o_index = 0; o_index < q.graph->output_size(); o_index++) {
          auto output = q.graph->mutable_output(o_index);
          bool skip =
              std::find(graph_io_to_skip.begin(), graph_io_to_skip.end(),
                        output->name()) != graph_io_to_skip.end();
          if (!skip &&
              output->type().tensor_type().elem_type() ==
                  ONNX_NAMESPACE::TensorProto::FLOAT) {
            output->mutable_type()->mutable_tensor_type()->set_elem_type(
                ONNX_NAMESPACE::TensorProto::FLOAT16);
            value_info_list.push_back(output);
          }
        }
        for (auto value_index = 0; value_index < q.graph->value_info_size();
             value_index++) {
          auto value = q.graph->mutable_value_info(value_index);
          bool skip =
              std::find(graph_io_to_skip.begin(), graph_io_to_skip.end(),
                        value->name()) != graph_io_to_skip.end();

          // in Resize op, when the dims of input tensor is zero.
          bool zero_shape_constant = false;
          for (auto i = 0; i < q.graph->node_size(); i++) {
            auto n = q.graph->mutable_node(i);
            if (n->op_type() != "Constant" || n->output(0) != value->name()) {
              continue;
            }
            for (auto attr_index = 0; attr_index < n->attribute_size();
                 attr_index++) {
              auto attr = n->mutable_attribute(attr_index);
              if (attr->has_t() && attr->t().dims_size() == 1 &&
                  attr->t().dims(0) == 0) {
                zero_shape_constant = true;
              }
            }
            if (zero_shape_constant) break;
          }

          // if it is a tensor that should keep type float
          bool keep_type_tensor =
              std::find(keep_type_tensors.begin(), keep_type_tensors.end(),
                        value->name()) != keep_type_tensors.end();
          if (!zero_shape_constant && !skip && !keep_type_tensor &&
              value->type().tensor_type().elem_type() ==
                  ONNX_NAMESPACE::TensorProto::FLOAT) {
            value->mutable_type()->mutable_tensor_type()->set_elem_type(
                ONNX_NAMESPACE::TensorProto::FLOAT16);
            value_info_list.push_back(value);
          }
        }
      }
    }
    queue.clear();
    queue = next_level;
  }
  // the model is a FP16 model
  if (!converted_attr) {
    return;
  }

  auto graph = model->mutable_graph();
  for (auto node : node_list) {
    // Handle the case of fp32_output OPs
    // Add cast op for node output
    if (std::find(fp32_output_op_list.begin(), fp32_output_op_list.end(),
                  node->op_type()) != fp32_output_op_list.end()) {
      for (auto o_index = 0; o_index < node->output_size(); o_index++) {
        std::string* output = node->mutable_output(o_index);
        for (auto v_index = 0; v_index < graph->value_info().size();
             v_index++) {
          auto value_info = graph->mutable_value_info(v_index);
          if (value_info->name() == *output) {
            if (value_info->type().tensor_type().elem_type() ==
                    ONNX_NAMESPACE::TensorProto::FLOAT16 &&
                !CastedTo(*output, *model, 10)) {
              std::string input_name = GenName(node->name() + "_output_cast_");
              std::string node_name = GenName(node->name() + "_output_cast");
              auto new_node =
                  MakeCastNode(node_name, {input_name}, {*output}, 10);
              *(graph->add_node()) = (*new_node);
              *(node->mutable_output(o_index)) = input_name;
            } else {
              break;
            }
          }
        }
      }
      continue;
    }

    // Handle the case of custom OPs
    if (std::find(custom_ops_.begin(), custom_ops_.end(), node->op_type()) !=
        custom_ops_.end()) {
      // add cast op for node input
      for (auto i_index = 0; i_index < node->input_size(); i_index++) {
        std::string* input = node->mutable_input(i_index);
        for (auto v_index = 0; v_index < graph->value_info().size();
             v_index++) {
          auto value_info = graph->mutable_value_info(v_index);
          if (value_info->name() == *input) {
            if (value_info->type().tensor_type().elem_type() ==
                    ONNX_NAMESPACE::TensorProto::FLOAT16 &&
                !CastedFrom(*input, *model, 1)) {
              std::string output_name = GenName(node->name() + "_input_cast_");
              std::string node_name = GenName(node->name() + "_input_cast");
              auto new_node =
                  MakeCastNode(node_name, {*input}, {output_name}, 1);
              *(graph->add_node()) = (*new_node);
              *(node->mutable_input(i_index)) = output_name;
            } else {
              break;
            }
          }
        }
      }
      // add cast op for node output
      for (auto o_index = 0; o_index < node->output_size(); o_index++) {
        std::string* output = node->mutable_output(o_index);
        for (auto v_index = 0; v_index < graph->value_info().size();
             v_index++) {
          auto value_info = graph->mutable_value_info(v_index);
          if (value_info->name() == *output) {
            if (value_info->type().tensor_type().elem_type() ==
                    ONNX_NAMESPACE::TensorProto::FLOAT16 &&
                !CastedTo(*output, *model, 10)) {
              std::string input_name = GenName(node->name() + "_output_cast_");
              std::string node_name = GenName(node->name() + "_output_cast");
              auto new_node =
                  MakeCastNode(node_name, {input_name}, {*output}, 10);
              *(graph->add_node()) = (*new_node);
              *(node->mutable_output(o_index)) = input_name;
            } else if (value_info->type().tensor_type().elem_type() ==
                       ONNX_NAMESPACE::TensorProto::FLOAT16) {
              value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
                  ONNX_NAMESPACE::TensorProto::FLOAT);
            } else {
              break;
            }
          }
        }
      }
    } else {
      // Handle the case of DEFAULT_OP_BLOCK_LIST OPs
      for (auto i_index = 0; i_index < node->input_size(); i_index++) {
        std::string* input = node->mutable_input(i_index);
        for (auto value_index = 0; value_index < value_info_list.size();
             value_index++) {
          auto value_info = value_info_list[value_index];
          if (value_info->has_name() && *input == value_info->name() &&
              !CastedFrom(*input, *model, 1)) {
            auto new_value_info = model->mutable_graph()->add_value_info();
            new_value_info->CopyFrom(*value_info);
            std::string output_name = GenName(node->name() + "_input_cast_");
            new_value_info->set_name(output_name);
            new_value_info->mutable_type()
                ->mutable_tensor_type()
                ->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
            std::string node_name = GenName(node->name() + "_input_cast");
            auto new_node = MakeCastNode(node_name, {*input}, {output_name}, 1);
            *(graph->add_node()) = (*new_node);
            *(node->mutable_input(i_index)) = output_name;
            break;
          }
        }
      }
      for (auto o_index = 0; o_index < node->output_size(); o_index++) {
        std::string* output = node->mutable_output(o_index);
        for (auto value_index = 0; value_index < value_info_list.size();
             value_index++) {
          if (*output == value_info_list[value_index]->name() &&
              !CastedTo(*output, *model, 10)) {
            auto new_value_info = model->mutable_graph()->add_value_info();
            new_value_info->CopyFrom(*value_info_list[value_index]);
            std::string input_name = GenName(node->name() + "_output_cast_");
            new_value_info->set_name(input_name);
            new_value_info->mutable_type()
                ->mutable_tensor_type()
                ->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
            std::string node_name = GenName(node->name() + "_output_cast");
            auto new_node =
                MakeCastNode(node_name, {input_name}, {*output}, 10);
            *(graph->add_node()) = (*new_node);
            *(node->mutable_output(o_index)) = input_name;
            break;
          }
        }
      }
    }
  }
}

bool ConvertFp32ToFp16::IsFP16Model(const ONNX_NAMESPACE::ModelProto& model) {
  for (auto node : model.graph().node()) {
    if (node.op_type() == "Cast") {
      auto name = node.name();
      if (name.find("_output_cast") != name.npos ||
          name.find("_input_cast") != name.npos ||
          name.find("graph_output_cast") != name.npos ||
          name.find("graph_input_cast") != name.npos) {
        return true;
      }
    }
  }
  return false;
}

void ConvertFp32ToFp16::Convert(ONNX_NAMESPACE::ModelProto* model) {
  if (op_block_list_.empty()) {
    op_block_list_ = DEFAULT_OP_BLOCK_LIST;
  }
  if (custom_ops_.size()) {
    op_block_list_.insert(op_block_list_.end(), custom_ops_.begin(),
                          custom_ops_.end());
  }
  shape_inference::InferShapes(*model);
  // 1 if it is a FP16 model, skip this
  if (IsFP16Model(*model)) {
    P2OLogger() << "[Info] The input ONNX Model is a FP16 model." << std::endl;
    return;
  }
  // 2 keep IO types
  KeepIoType(model);
  // 3 ConvertAttribute
  ConvertAttribute(model);
  // 4 sortnodes
  SortNodes(model);
}

}  // namespace paddle2onnx
