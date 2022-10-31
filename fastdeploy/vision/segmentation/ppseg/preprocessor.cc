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
#include "fastdeploy/vision/segmentation/ppseg/preprocessor.h"
#include "yaml-cpp/yaml.h"

namespace fastdeploy {
namespace vision {
namespace segmentation {
PaddleSegPreprocessor::PaddleSegPreprocessor(const std::string& config_file) {
  if (!BuildPreprocessPipelineFromConfig(config_file)) {
    FDERROR << "Failed to build preprocess pipeline from configuration file."
            << std::endl;
  }
}

bool PaddleSegPreprocessor::BuildPreprocessPipelineFromConfig(const std::string& config_file) {
  processors_.clear();
  YAML::Node cfg;
  processors_.push_back(std::make_shared<BGR2RGB>());
  try {
    cfg = YAML::LoadFile(config_file);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file
            << ", maybe you should check this file." << std::endl;
    return false;
  }   
  bool yml_contain_resize_op = false;

  if (cfg["Deploy"]["transforms"]) {
    auto preprocess_cfg = cfg["Deploy"]["transforms"];
    for (const auto& op : preprocess_cfg) {
      FDASSERT(op.IsMap(),
               "Require the transform information in yaml be Map type.");
      if (op["type"].as<std::string>() == "Normalize") {
        std::vector<float> mean = {0.5, 0.5, 0.5};
        std::vector<float> std = {0.5, 0.5, 0.5};
        if (op["mean"]) {
          mean = op["mean"].as<std::vector<float>>();
        }
        if (op["std"]) {
          std = op["std"].as<std::vector<float>>();
        }
        processors_.push_back(std::make_shared<Normalize>(mean, std));

      } else if (op["type"].as<std::string>() == "Resize") {
        yml_contain_resize_op = true;
        const auto& target_size = op["target_size"];
        int resize_width = target_size[0].as<int>();
        int resize_height = target_size[1].as<int>();
        processors_.push_back(
            std::make_shared<Resize>(resize_width, resize_height));
      } else {
        std::string op_name = op["type"].as<std::string>();
        FDERROR << "Unexcepted preprocess operator: " << op_name << "."
                << std::endl;
        return false;
      }
    }
  }
  if (cfg["Deploy"]["input_shape"]) {
    auto input_shape = cfg["Deploy"]["input_shape"];
    int input_batch = input_shape[0].as<int>();
    int input_channel = input_shape[1].as<int>();
    int input_height = input_shape[2].as<int>();
    int input_width = input_shape[3].as<int>();
    if (input_height == -1 || input_width == -1) {
      FDWARNING << "The exported PaddleSeg model is with dynamic shape input, "
                << "which is not supported by ONNX Runtime and Tensorrt. "
                << "Only OpenVINO and Paddle Inference are available now. " 
                << "For using ONNX Runtime or Tensorrt, "
                << "Please refer to https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/model_export.md"
                << " to export model with fixed input shape."
                << std::endl;
      is_change_backends = true;
    }
    if (input_height != -1 && input_width != -1 && !yml_contain_resize_op) {
      processors_.push_back(
          std::make_shared<Resize>(input_width, input_height));
    }
  }
  if (cfg["Deploy"]["output_op"]) {
    std::string output_op = cfg["Deploy"]["output_op"].as<std::string>();
    if (output_op == "softmax") {
      is_with_softmax_ = true;
      is_with_argmax_ = false;
    } else if (output_op == "argmax") {
      is_with_softmax_ = false;
      is_with_argmax_ = true;
    } else if (output_op == "none") {
      is_with_softmax_ = false;
      is_with_argmax_ = false;
    } else {
      FDERROR << "Unexcepted output_op operator in deploy.yml: " << output_op
              << "." << std::endl;
    }
  }
  processors_.push_back(std::make_shared<HWC2CHW>());
  return true;
}

bool PaddleSegPreprocessor::Run(Mat* mat, FDTensor* output) {
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (processors_[i]->Name().compare("Resize") == 0) {
      auto processor = dynamic_cast<Resize*>(processors_[i].get());
      int resize_width = -1;
      int resize_height = -1;
      std::tie(resize_width, resize_height) = processor->GetWidthAndHeight();
      if (is_vertical_screen_ && (resize_width > resize_height)) {
        if (!(processor->SetWidthAndHeight(resize_height, resize_width))) {
          FDERROR << "Failed to set width and height of "
                  << processors_[i]->Name() << " processor." << std::endl;
        }
      }
    }
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
  }

  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);
  return true;
}
}  // namespace segmentation
}  // namespace vision
}  // namespace fastdeploy