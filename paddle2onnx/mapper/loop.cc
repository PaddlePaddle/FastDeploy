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

#include "paddle2onnx/mapper/exporter.h"

namespace paddle2onnx {

bool ModelExporter::IsLoopSupported(const PaddleParser& parser,
                                    const int64_t& block_id,
                                    const int64_t& op_id) {
  auto x_info = parser.GetOpInput(block_id, op_id, "X");
  auto out_info = parser.GetOpOutput(block_id, op_id, "Out");
  auto cond_info = parser.GetOpInput(block_id, op_id, "Condition");
  std::set<std::string> input_names;
  for (size_t i = 0; i < x_info.size(); ++i) {
    input_names.insert(x_info[i].name);
  }
  input_names.insert(cond_info[0].name);

  for (size_t i = 0; i < out_info.size(); ++i) {
    auto iter = input_names.find(out_info[i].name);
    if (iter == input_names.end()) {
      P2OLogger() << "Cannot find output:" << out_info[i].name << " in input tensors while converting operator 'while', Paddle2ONNX doesn't support this situation now." << std::endl;
      return false;
    }
  }
  for (size_t i = 0; i < x_info.size(); ++i) {
    if (x_info[i].is_tensor_array) {
      P2OLogger() << "LodTensorArray is not supported." << std::endl;
      return false;
    }
  }
  return true;
}

void ModelExporter::ExportLoop(const PaddleParser& parser, OnnxHelper* helper,
                               int32_t opset_version, int64_t block_id,
                               int64_t op_id, bool verbose) {
  auto op = parser.GetOpDesc(block_id, op_id);
  int32_t sub_block_idx = -1;
  for (size_t i = 0; i < op.attrs_size(); ++i) {
    if (op.attrs(i).name() == "sub_block") {
      sub_block_idx = op.attrs(i).block_idx();
      break;
    }
  }
  Assert(sub_block_idx > 0, "Cannot find sub_block in while operator.");
  auto x_info = parser.GetOpInput(block_id, op_id, "X");
  auto cond_info = parser.GetOpInput(block_id, op_id, "Condition");

  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> inputs;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> outputs;

  // make loop iter
  auto iter_name = MapperHelper::Get()->GenName("loop.iter");
  TensorInfo iter_info(iter_name, std::vector<int64_t>(1, 1),
                       P2ODataType::INT64);
  inputs.push_back(std::move(MakeValueInfo(iter_info)));

  std::set<std::string> input_names;
  // make cond
  inputs.push_back(std::move(MakeValueInfo(cond_info[0])));
  input_names.insert(cond_info[0].name);
  // other inputs
  outputs.push_back(std::move(std::move(MakeValueInfo(cond_info[0]))));
  for (size_t i = 0; i < x_info.size(); ++i) {
    if (x_info[i].is_tensor_array) {
      continue;
    }
    if (input_names.find(x_info[i].name) != input_names.end()) {
      continue;
    }
    input_names.insert(x_info[i].name);
    inputs.push_back(std::move(MakeValueInfo(x_info[i])));
    outputs.push_back(std::move(MakeValueInfo(x_info[i])));
  }
  for (size_t i = 0; i < x_info.size(); ++i) {
    if (x_info[i].is_tensor_array) {
      if (input_names.find(x_info[i].name) != input_names.end()) {
        continue;
      }
      input_names.insert(x_info[i].name);
      outputs.push_back(std::move(MakeValueInfo(x_info[i])));
    }
  }

  // make op nodes
  OnnxHelper loop_helper;
  loop_helper.SetOpsetVersion(opset_version);

  for (auto i = 0; i < parser.NumOfOps(sub_block_idx); ++i) {
    auto op = parser.GetOpDesc(sub_block_idx, i);
    ExportOp(parser, &loop_helper, opset_version, sub_block_idx, i, verbose);
  }

  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> parameters;
  ProcessGraphDumplicateNames(&parameters, &inputs, &outputs,
                              &loop_helper.nodes);
  std::map<std::string, std::string> renamer;
  for (auto& item : inputs) {
    auto name = MapperHelper::Get()->GenName("loop.input");
    renamer[item->name()] = name;
    item->set_name(name);
  }
  for (auto& item : loop_helper.nodes) {
    for (size_t i = 0; i < item->input_size(); ++i) {
      if (renamer.find(item->input(i)) != renamer.end()) {
        auto updated_name = renamer[item->input(i)];
        while (renamer.find(updated_name) != renamer.end()) {
          updated_name = renamer[updated_name];
        }
        *(item->mutable_input(i)) = updated_name;
      }
    }
  }
  for (auto& item : outputs) {
    if (renamer.find(item->name()) != renamer.end()) {
      auto updated_name = renamer[item->name()];
      while (renamer.find(updated_name) != renamer.end()) {
        updated_name = renamer[updated_name];
      }
      item->set_name(updated_name);
    }
  }

  //  // construct a onnx model proto
  //  // consider to optimize the subgraph
  //  auto model = std::make_shared<ONNX_NAMESPACE::ModelProto>();
  //  model->set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  //  auto graph = model->mutable_graph();
  //  auto graph_name = MapperHelper::Get()->GenName("Model from
  //  PaddlePaddle(Loop).");
  //  graph->set_name(graph_name);
  //  auto opset_id = model->add_opset_import();
  //  opset_id->set_domain("");
  //  opset_id->set_version(loop_helper->GetOpsetVersion());

  auto graph_name = MapperHelper::Get()->GenName("paddle.loop");
  auto graph = std::make_shared<ONNX_NAMESPACE::GraphProto>();
  graph->set_name(graph_name);
  for (auto& item : inputs) {
    *(graph->add_input()) = *(item.get());
  }
  for (auto& item : loop_helper.nodes) {
    *(graph->add_node()) = (*item.get());
  }
  for (auto& item : outputs) {
    *(graph->add_output()) = (*item.get());
  }

  // fake iter
  auto fake_iter = helper->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                    std::vector<int64_t>(1, 1024));
  std::vector<std::string> x_names;
  x_names.push_back(fake_iter);
  x_names.push_back(cond_info[0].name);
  std::vector<std::string> out_names;
  for (size_t i = 0; i < x_info.size(); ++i) {
    if (x_info[i].is_tensor_array) {
      continue;
    }
    if (std::find(x_names.begin(), x_names.end(), x_info[i].name) != x_names.end()) {
      continue;
    }
    x_names.push_back(x_info[i].name);
    out_names.push_back(x_info[i].name);
  }
  for (size_t i = 0; i < x_info.size(); ++i) {
    if (x_info[i].is_tensor_array) {
      if (std::find(x_names.begin(), x_names.end(), x_info[i].name) != x_names.end()) {
        continue;
      }
      out_names.push_back(x_info[i].name);
    }
  }

  auto loop_node = helper->MakeNode("Loop", x_names, out_names);
  auto attr = loop_node->add_attribute();
  attr->set_name("body");
  attr->set_type(ONNX_NAMESPACE::AttributeProto::GRAPH);
  *(attr->mutable_g()) = *(graph.get());
}

}  // namespace paddle2onnx
