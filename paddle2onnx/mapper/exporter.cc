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

#include <google/protobuf/message.h>
#include <onnx/checker.h>

#include <array>

#include "onnxoptimizer/optimize.h"
#include "paddle2onnx/optimizer/convert_fp32_to_fp16.h"
#include "paddle2onnx/optimizer/eliminate_non_transpose.h"
#include "paddle2onnx/optimizer/fuse_constant_cast.h"
#include "paddle2onnx/optimizer/fuse_constant_reshape.h"
#include "paddle2onnx/optimizer/fuse_constant_unsqueeze.h"
#include "paddle2onnx/optimizer/fuse_paddle_conv_bias.h"
#include "paddle2onnx/optimizer/fuse_unsqueeze_conv2d_squeeze.h"

namespace paddle2onnx {
MapperHelper* MapperHelper::helper = nullptr;

void ModelExporter::ExportParameters(
    const std::map<std::string, Weight>& params, bool use_initializer) {
  for (auto& item : params) {
    // TODO(jiangjiajun) I'm not handling use_initializer now, but some day I
    // will
    auto node = MakeConstant(item.first, item.second);
    parameters.push_back(std::move(node));
  }
}

void ModelExporter::UpdateParameters(
    const std::map<std::string, Weight>& params) {
  for (auto& item : params) {
    auto node = MakeConstant(item.first, item.second);
    bool updated = false;
    for (int i = 0; i < parameters.size(); ++i) {
      auto old_node = parameters[i];
      if (old_node->output(0) == item.first) {
        parameters.erase(parameters.begin() + i);
        parameters.push_back(std::move(node));
        updated = true;
        break;
      }
    }
    if (!updated) {
      parameters.push_back(std::move(node));
    }
  }
}

void ModelExporter::ExportInputOutputs(
    const std::vector<TensorInfo>& input_infos,
    const std::vector<TensorInfo>& output_infos) {
  for (auto& item : input_infos) {
    auto value_info = MakeValueInfo(item);
    inputs.push_back(std::move(value_info));
  }
  for (auto& item : output_infos) {
    auto value_info = MakeValueInfo(item);
    outputs.push_back(std::move(value_info));
  }
}

void ModelExporter::ExportOp(const PaddleParser& parser, OnnxHelper* helper,
                             int32_t opset_version, int64_t block_id,
                             int64_t op_id, bool verbose) {
  _current_exported_num += 1;
  auto op = parser.GetOpDesc(block_id, op_id);
#ifdef PADDLE2ONNX_DEBUG
  P2OLogger(true) << "---Converting operator: " << op.type() << " ---"
                  << std::endl;
#endif
  if (op.type() == "while") {
    return ExportLoop(parser, helper, opset_version, block_id, op_id, verbose);
  }

  auto mapper = MapperHelper::Get()->CreateMapper(op.type(), parser, helper,
                                                  block_id, op_id);
  mapper->deploy_backend = _deploy_backend;
#ifdef PADDLE2ONNX_DEBUG
  P2OLogger(true) << "Mapper Name: " << mapper->Name() << std::endl;
#endif
  // Some operators will export as custom operator
  auto iter = custom_ops.find(op.type());
  if (iter != custom_ops.end()) {
    mapper->export_as_custom_op = true;
    mapper->custom_op_name = iter->second;
  }
  mapper->Run();
  delete mapper;

#ifdef PADDLE2ONNX_DEBUG
  P2OLogger(true) << "---Converting operator: " << op.type() << " done---"
                  << std::endl;
#endif
}

void ModelExporter::ProcessGraphDumplicateNames(
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* parameters,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* inputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* outputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes,
    std::map<std::string, QuantizeInfo>* quantize_info) {
  // process dumplicate tensor names
  std::map<std::string, std::string> renamer;
  std::set<std::string> tensor_names;
  for (auto& item : *parameters) {
    for (size_t i = 0; i < item->output_size(); ++i) {
      if (tensor_names.find(item->output(i)) != tensor_names.end()) {
        Assert(false, "There's dumplicate names in exported parameters.");
      }
      tensor_names.insert(item->output(i));
    }
  }
  for (auto& item : *inputs) {
    if (tensor_names.find(item->name()) != tensor_names.end()) {
      Assert(false, "There's dumplicate names:" + item->name() +
                        " in exported parameters and inputs.");
    }
    tensor_names.insert(item->name());
  }
  for (auto& item : *nodes) {
    // update node inputs
    for (size_t i = 0; i < item->input_size(); ++i) {
      if (renamer.find(item->input(i)) != renamer.end()) {
        auto updated_name = renamer[item->input(i)];
        while (renamer.find(updated_name) != renamer.end()) {
          updated_name = renamer[updated_name];
        }
        *(item->mutable_input(i)) = updated_name;
      }
    }
    // if there's dumplicate name
    // will generate new name and replace it
    for (size_t i = 0; i < item->output_size(); ++i) {
      if (tensor_names.find(item->output(i)) != tensor_names.end()) {
        std::string renamed_tensor_name = item->output(i);
        while (renamer.find(renamed_tensor_name) != renamer.end()) {
          renamed_tensor_name = renamer[renamed_tensor_name];
        }
        auto new_tensor_name =
            MapperHelper::Get()->GenName(renamed_tensor_name);
        P2OLogger() << "Find dumplicate output name '" << renamed_tensor_name
                    << "', it will rename to '" << new_tensor_name << "'."
                    << std::endl;
        if (quantize_info &&
            quantize_info->find(renamed_tensor_name) != quantize_info->end()) {
          (*quantize_info)[new_tensor_name] =
              (*quantize_info)[renamed_tensor_name];
        }
        *(item->mutable_output(i)) = new_tensor_name;
        renamer[renamed_tensor_name] = new_tensor_name;
      }
      tensor_names.insert(item->output(i));
    }
  }

  for (auto& item : *outputs) {
    if (renamer.find(item->name()) != renamer.end()) {
      auto updated_name = renamer[item->name()];
      while (renamer.find(updated_name) != renamer.end()) {
        updated_name = renamer[updated_name];
      }
      item->set_name(updated_name);
    }
  }
}

void ModelExporter::SaveExternalData(::paddle2onnx::GraphProto* graph,
                                     const std::string& external_file_path,
                                     bool* save_external) {
  P2OLogger() << "The exported ONNX model is bigger than 2G, external data "
                 "will save to file: "
              << external_file_path << std::endl;
  std::string file_name = GetFilenameFromPath(external_file_path);
  if (save_external) {
    *save_external = true;
  }
  std::fstream f(external_file_path, std::ios::out);
  Assert(f.is_open(), "Failed to open: " + external_file_path +
                          " file to save external data");
  for (auto index = 0; index < graph->node_size(); index++) {
    auto node = graph->mutable_node(index);
    if (node->op_type() != "Constant") {
      continue;
    }
    for (auto i = 0; i < node->attribute_size(); i++) {
      auto attr = node->mutable_attribute(i);
      if (attr->name() != "value") {
        continue;
      }
      auto tensor = attr->mutable_t();

      if (tensor->raw_data().size() <= 128) {
        continue;
      }

      tensor->set_data_location(TensorProto::EXTERNAL);
      auto external_data = tensor->add_external_data();
      external_data->set_key("location");
      external_data->set_value(file_name);

      external_data = tensor->add_external_data();
      external_data->set_key("offset");
      f.seekg(0, std::ios::end);
      int64_t offset = f.tellg();
      external_data->set_value(std::to_string(offset));
      auto raw_data = tensor->raw_data();
      f << raw_data;
      external_data = tensor->add_external_data();
      external_data->set_key("length");
      int64_t raw_datas_size = raw_data.size();
      external_data->set_value(std::to_string(raw_datas_size));
      tensor->clear_raw_data();
    }
  }
  f.close();
}
void ModelExporter::ONNXChecker(const ONNX_NAMESPACE::ModelProto& model,
                                const bool& verbose) {
  // TODO(jiangjiajun)
  // If we need to integrate with framework
  // this check will return a information
  // to let framework know the conversion is
  // pass or fail
  try {
    // ONNX_NAMESPACE::checker::check_model(*(model.get()));
    ONNX_NAMESPACE::checker::check_model(model);
  } catch (const std::exception& e) {
    P2OLogger(verbose) << "The exported ONNX model is invalid." << std::endl;
    P2OLogger(verbose) << "Model checker error log: " << e.what() << std::endl;
  }
  P2OLogger(verbose) << "PaddlePaddle model is exported as ONNX format now."
                     << std::endl;
}

std::string ModelExporter::Run(
    const PaddleParser& parser, int opset_version, bool auto_upgrade_opset,
    bool verbose, bool enable_onnx_checker, bool enable_experimental_op,
    bool enable_optimize, const std::string& deploy_backend,
    std::string* calibration_cache, const std::string& external_file,
    bool* save_external, bool export_fp16_model) {
  _deploy_backend = deploy_backend;
  _helper.SetOpsetVersion(opset_version);
  _total_ops_num = 0;
  _current_exported_num = 0;
  for (auto i = 0; i < parser.NumOfBlocks(); ++i) {
    _total_ops_num += parser.NumOfOps(i);
  }
  _helper.nodes.reserve(_total_ops_num * 3);
  Assert(opset_version <= MAX_ONNX_OPSET_VERSION && opset_version >= 7,
         "Paddle2ONNX now only support opset version in range of [7, " +
             std::to_string(MAX_ONNX_OPSET_VERSION) + "].");
  _helper.Clear();
  inputs.clear();
  outputs.clear();
  parameters.clear();

  // clear name_counter
  // this use to generate unique name
  // for intermdiate
  // while converting all the op
  MapperHelper::Get()->ClearNameCounter();

  std::set<std::string> unsupported_ops;
  if (!CheckIfOpSupported(parser, &unsupported_ops, enable_experimental_op)) {
    auto logger = P2OLogger();
    logger << "Oops, there are some operators not supported yet, including ";
    for (auto& item : unsupported_ops) {
      logger << item << ",";
    }
    logger << std::endl;
    Assert(1 == 0,
           "Due to the unsupported operators, the conversion is aborted.");
  }

  int32_t min_opset = GetMinOpset(parser, verbose);
  if (min_opset < 0) {
    Assert(false,
           "Model exporting failed, you can report this problem to "
           "https://github.com/PaddlePaddle/Paddle2ONNX.git.");
  }
  if (!auto_upgrade_opset) {
    if (min_opset > opset_version) {
      P2OLogger() << "This PaddlePaddle model is not able to export to ONNX "
                     "with opset_version="
                  << opset_version << ", please set the opset_version to "
                  << min_opset << " or higher for successfully conversion."
                  << std::endl;
      Assert(false,
             "Due to opset version, the model exporting is aborted, please set "
             "a higher opset_version or set auto_upgrade_opset=true.");
    }
  } else {
    if (min_opset > opset_version) {
      P2OLogger() << "Opset version will change to " << min_opset << " from "
                  << opset_version << std::endl;
      opset_version = min_opset;
    }
  }
  _helper.SetOpsetVersion(opset_version);
  P2OLogger(verbose) << "Use opset_version = " << _helper.GetOpsetVersion()
                     << " for ONNX export." << std::endl;
  ExportParameters(parser.params);
  ExportInputOutputs(parser.inputs, parser.outputs);

  // Only convert blocks 0 now
  // because control flow is not supported yet
  for (auto i = 0; i < parser.NumOfOps(0); ++i) {
    auto op = parser.GetOpDesc(0, i);
    if (op.type() == "feed") {
      continue;
    } else if (op.type() == "fetch") {
      continue;
    }
    ExportOp(parser, &_helper, opset_version, 0, i, verbose);
  }

  // construct a onnx model proto
  auto model = std::make_shared<ONNX_NAMESPACE::ModelProto>();
  // TODO(jiangjiajun) ir version is related to onnx version
  model->set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto graph = model->mutable_graph();
  graph->set_name("Model from PaddlePaddle.");
  auto opset_id = model->add_opset_import();
  opset_id->set_domain("");
  opset_id->set_version(opset_version);
  if (custom_ops.size()) {
    auto opset_paddle_id = model->add_opset_import();
    opset_paddle_id->set_domain("Paddle");
    opset_paddle_id->set_version(1);
  }

  ProcessGraphDumplicateNames(&parameters, &inputs, &outputs, &_helper.nodes,
                              &_helper.quantize_info);
  if (parser.is_quantized_model) {
    quantize_model_processer.ProcessQuantizeModel(
        &parameters, &inputs, &outputs, &_helper.nodes, &_helper,
        deploy_backend, parser, calibration_cache);
    // Update int8 weights in quantized OP to float32
    UpdateParameters(_helper.updated_params);
  }

  for (auto& item : parameters) {
    *(graph->add_node()) = *(item.get());
  }
  for (auto& item : inputs) {
    *(graph->add_input()) = *(item.get());
  }
  for (auto& item : _helper.nodes) {
    *(graph->add_node()) = (*item.get());
  }
  for (auto& item : outputs) {
    *(graph->add_output()) = (*item.get());
  }
  for (auto& item : _helper.value_infos) {
    *(graph->add_value_info()) = (*item.get());
  }

  ONNX_NAMESPACE::ModelProto onnx_model;
  std::string out;
  if (enable_optimize) {
    onnx_model = Optimize(*(model.get()));
  } else {
    onnx_model = *model.get();
  }

  // convert fp32 model to fp16
  if (export_fp16_model) {
    P2OLogger(verbose) << "Convert FP32 ONNX model to FP16." << std::endl;
    ConvertFp32ToFp16 convert;
    convert.SetCustomOps(custom_ops);
    convert.Convert(&onnx_model);
  }

  // save external data file for big model
  std::string external_data_file;
  if (onnx_model.ByteSizeLong() > INT_MAX) {
    if (external_file.empty()) {
      external_data_file = "external_data";
    } else {
      external_data_file = external_file;
    }
  }
  if (external_data_file.size()) {
    SaveExternalData(onnx_model.mutable_graph(), external_data_file,
                     save_external);
  }
  // check model
  if (enable_onnx_checker) {
    ONNXChecker(onnx_model, verbose);
  }

  if (!onnx_model.SerializeToString(&out)) {
    P2OLogger(verbose)
        << "Error happenedd while optimizing the exported ONNX model."
        << std::endl;
    return "";
  }
  return out;
}

bool ModelExporter::CheckIfOpSupported(const PaddleParser& parser,
                                       std::set<std::string>* unsupported_ops,
                                       bool enable_experimental_op) {
  unsupported_ops->clear();
  for (auto i = 0; i < parser.NumOfBlocks(); ++i) {
    for (auto j = 0; j < parser.NumOfOps(i); ++j) {
      auto op = parser.GetOpDesc(i, j);
      if (op.type() == "feed" || op.type() == "fetch") {
        continue;
      }
      if (op.type() == "while" && enable_experimental_op) {
        if (!IsLoopSupported(parser, i, j)) {
          unsupported_ops->insert("while");
        }
        continue;
      }
      if (!MapperHelper::Get()->IsRegistered(op.type())) {
        unsupported_ops->insert(op.type());
      } else if (!enable_experimental_op) {
        auto mapper = MapperHelper::Get()->CreateMapper(op.type(), parser,
                                                        &_helper, i, j);
        if (mapper->IsExperimentalOp()) {
          unsupported_ops->insert(op.type());
        }
        delete mapper;
      }
    }
  }
  return (unsupported_ops->size() == 0);
}

int32_t ModelExporter::GetMinOpset(const PaddleParser& parser, bool verbose) {
  int32_t opset_version = _helper.GetOpsetVersion();
  int32_t max_opset = 7;
  bool exportable = true;
  // Record the number of ops that need to be converted
  int converted_op_num = 0;
  std::set<std::string> verbose_log;
  for (auto i = 0; i < parser.NumOfBlocks(); ++i) {
    for (auto j = 0; j < parser.NumOfOps(i); ++j) {
      auto op = parser.GetOpDesc(i, j);
      if (custom_ops.find(op.type()) != custom_ops.end()) {
        continue;
      }
      if (op.type() == "feed" || op.type() == "fetch") {
        continue;
      }
      converted_op_num += 1;
      int current_min_opset = 7;
      if (op.type() == "while") {
        P2OLogger() << "Detected there's control flow 'while' op in your "
                       "model, this requires the minimal opset version of 13."
                    << std::endl;
        current_min_opset = 13;
      } else {
        auto mapper = MapperHelper::Get()->CreateMapper(op.type(), parser,
                                                        &_helper, i, j);
        auto iter = custom_ops.find(op.type());
        if (iter != custom_ops.end()) {
          mapper->export_as_custom_op = true;
        }
        current_min_opset = mapper->GetMinOpset(verbose);
        delete mapper;
      }
      if (current_min_opset < 0) {
        exportable = false;
        P2OLogger(verbose) << "Due to the operator: " << op.type()
                           << ", this model cannot be exported to ONNX."
                           << std::endl;
      } else if (current_min_opset > max_opset) {
        max_opset = current_min_opset;
        if (verbose && current_min_opset > opset_version) {
          verbose_log.insert("Due to the operator: " + op.type() +
                             ", requires opset_version >= " +
                             std::to_string(current_min_opset) + ".");
        }
      }
    }
  }
  if (verbose) {
    for (auto iter = verbose_log.begin(); iter != verbose_log.end(); ++iter) {
      P2OLogger() << *iter << std::endl;
    }
  }

  // Here we put some checks to make sure
  // paddle2onnx could compatible with
  // other version of onnx
  int32_t max_support_opset = MAX_ONNX_OPSET_VERSION;
  if (exportable && (max_opset > MAX_ONNX_OPSET_VERSION)) {
    exportable = false;
    P2OLogger() << "[ERROR] The compiled ONNX version only supports opset 7~"
                << MAX_ONNX_OPSET_VERSION
                << ", but now this model need as least opset " << max_opset
                << ", please compile with higher version of ONNX." << std::endl;
  }
  if (exportable) {
    return max_opset;
  }

  return -1;
}

ONNX_NAMESPACE::ModelProto ModelExporter::Optimize(
    const ONNX_NAMESPACE::ModelProto& model) {
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseConstantReshape>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseConstantUnsqueeze>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FusePaddleConvBias>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseUnsqueezeConv2dSqueeze>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::EliminateNonTranspose>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseConstantCast>();
  std::vector<std::string> passes = {"eliminate_identity",
                                     "eliminate_deadend",
                                     "eliminate_deadend",
                                     "fuse_constant_reshape",
                                     "fuse_constant_unsqueeze",
                                     "fuse_paddle_conv_bias",
                                     "fuse_consecutive_transposes",
                                     "eliminate_non_transpose",
                                     "fuse_matmul_add_bias_into_gemm",
                                     "eliminate_identity",
                                     "eliminate_deadend",
                                     "eliminate_unused_initializer"};
  return ONNX_NAMESPACE::optimization::Optimize(model, passes);
}

}  // namespace paddle2onnx
