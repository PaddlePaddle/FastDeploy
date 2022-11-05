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

#include "fastdeploy/vision/detection/rknpu2/rkpicodet.h"
#include "yaml-cpp/yaml.h"
namespace fastdeploy {
namespace vision {
namespace detection {

RKPicoDet::RKPicoDet(const std::string& model_file,
                     const std::string& params_file,
                     const std::string& config_file,
                     const RuntimeOption& custom_option,
                     const ModelFormat& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::ORT};
  valid_rknpu_backends = {Backend::RKNPU2};
  if ((model_format == ModelFormat::RKNN) ||
      (model_format == ModelFormat::ONNX)) {
    has_nms_ = false;
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;

  // NMS parameters come from RKPicoDet_s_nms
  background_label = -1;
  keep_top_k = 100;
  nms_eta = 1;
  nms_threshold = 0.5;
  nms_top_k = 1000;
  normalized = true;
  score_threshold = 0.3;
  initialized = Initialize();
}

bool RKPicoDet::Initialize() {
  if (!BuildPreprocessPipelineFromConfig()) {
    FDERROR << "Failed to build preprocess pipeline from configuration file."
            << std::endl;
    return false;
  }
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

bool RKPicoDet::Preprocess(Mat* mat, std::vector<FDTensor>* outputs) {
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
  }

  Cast::Run(mat, "float");

  scale_factor.resize(2);
  scale_factor[0] = mat->Height() * 1.0 / origin_h;
  scale_factor[1] = mat->Width() * 1.0 / origin_w;

  outputs->resize(1);
  (*outputs)[0].name = InputInfoOfRuntime(0).name;
  mat->ShareWithTensor(&((*outputs)[0]));
  // reshape to [1, c, h, w]
  (*outputs)[0].shape.insert((*outputs)[0].shape.begin(), 1);
  return true;
}

bool RKPicoDet::BuildPreprocessPipelineFromConfig() {
  processors_.clear();
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }

  processors_.push_back(std::make_shared<BGR2RGB>());

  for (const auto& op : cfg["Preprocess"]) {
    std::string op_name = op["type"].as<std::string>();
    if (op_name == "NormalizeImage") {
      continue;
    } else if (op_name == "Resize") {
      bool keep_ratio = op["keep_ratio"].as<bool>();
      auto target_size = op["target_size"].as<std::vector<int>>();
      int interp = op["interp"].as<int>();
      FDASSERT(target_size.size() == 2,
               "Require size of target_size be 2, but now it's %lu.",
               target_size.size());
      if (!keep_ratio) {
        int width = target_size[1];
        int height = target_size[0];
        processors_.push_back(
            std::make_shared<Resize>(width, height, -1.0, -1.0, interp, false));
      } else {
        int min_target_size = std::min(target_size[0], target_size[1]);
        int max_target_size = std::max(target_size[0], target_size[1]);
        std::vector<int> max_size;
        if (max_target_size > 0) {
          max_size.push_back(max_target_size);
          max_size.push_back(max_target_size);
        }
        processors_.push_back(std::make_shared<ResizeByShort>(
            min_target_size, interp, true, max_size));
      }
    } else if (op_name == "Permute") {
      continue;
    } else if (op_name == "Pad") {
      auto size = op["size"].as<std::vector<int>>();
      auto value = op["fill_value"].as<std::vector<float>>();
      processors_.push_back(std::make_shared<Cast>("float"));
      processors_.push_back(
          std::make_shared<PadToSize>(size[1], size[0], value));
    } else if (op_name == "PadStride") {
      auto stride = op["stride"].as<int>();
      processors_.push_back(
          std::make_shared<StridePad>(stride, std::vector<float>(3, 0)));
    } else {
      FDERROR << "Unexcepted preprocess operator: " << op_name << "."
              << std::endl;
      return false;
    }
  }
  return true;
}

bool RKPicoDet::Postprocess(std::vector<FDTensor>& infer_result,
                            DetectionResult* result) {
  FDASSERT(infer_result[1].shape[0] == 1,
           "Only support batch = 1 in FastDeploy now.");

  if (!has_nms_) {
    int boxes_index = 0;
    int scores_index = 1;
    if (infer_result[0].shape[1] == infer_result[1].shape[2]) {
      boxes_index = 0;
      scores_index = 1;
    } else if (infer_result[0].shape[2] == infer_result[1].shape[1]) {
      boxes_index = 1;
      scores_index = 0;
    } else {
      FDERROR << "The shape of boxes and scores should be [batch, boxes_num, "
                 "4], [batch, classes_num, boxes_num]"
              << std::endl;
      return false;
    }

    backend::MultiClassNMS nms;
    nms.background_label = background_label;
    nms.keep_top_k = keep_top_k;
    nms.nms_eta = nms_eta;
    nms.nms_threshold = nms_threshold;
    nms.score_threshold = score_threshold;
    nms.nms_top_k = nms_top_k;
    nms.normalized = normalized;
    nms.Compute(static_cast<float*>(infer_result[boxes_index].Data()),
                static_cast<float*>(infer_result[scores_index].Data()),
                infer_result[boxes_index].shape,
                infer_result[scores_index].shape);
    if (nms.out_num_rois_data[0] > 0) {
      result->Reserve(nms.out_num_rois_data[0]);
    }
    for (size_t i = 0; i < nms.out_num_rois_data[0]; ++i) {
      result->label_ids.push_back(nms.out_box_data[i * 6]);
      result->scores.push_back(nms.out_box_data[i * 6 + 1]);
      result->boxes.emplace_back(
          std::array<float, 4>{nms.out_box_data[i * 6 + 2] / scale_factor[1],
                               nms.out_box_data[i * 6 + 3] / scale_factor[0],
                               nms.out_box_data[i * 6 + 4] / scale_factor[1],
                               nms.out_box_data[i * 6 + 5] / scale_factor[0]});
    }
  } else {
    FDERROR << "Picodet in Backend::RKNPU2 don't support NMS" << std::endl;
  }
  return true;
}

} // namespace detection
} // namespace vision
} // namespace fastdeploy
