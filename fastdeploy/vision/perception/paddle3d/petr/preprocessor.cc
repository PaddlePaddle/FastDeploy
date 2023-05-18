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

#include "fastdeploy/vision/perception/paddle3d/petr/preprocessor.h"
#include <iostream>

#include "fastdeploy/function/concat.h"
#include "yaml-cpp/yaml.h"

namespace fastdeploy {
namespace vision {
namespace perception {

PetrPreprocessor::PetrPreprocessor(const std::string& config_file) {
  config_file_ = config_file;
  FDASSERT(BuildPreprocessPipelineFromConfig(),
           "Failed to create Paddle3DDetPreprocessor.");
  initialized_ = true;
}

bool PetrPreprocessor::BuildPreprocessPipelineFromConfig() {
  processors_.clear();
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }

  // read for preprocess
  bool has_permute = false;
  for (const auto& op : cfg["Preprocess"]) {
    std::string op_name = op["type"].as<std::string>();
    if (op_name == "NormalizeImage") {
      auto mean = op["mean"].as<std::vector<float>>();
      auto std = op["std"].as<std::vector<float>>();
      bool is_scale = true;
      if (op["is_scale"]) {
        is_scale = op["is_scale"].as<bool>();
      }
      std::string norm_type = "mean_std";
      if (op["norm_type"]) {
        norm_type = op["norm_type"].as<std::string>();
      }
      if (norm_type != "mean_std") {
        std::fill(mean.begin(), mean.end(), 0.0);
        std::fill(std.begin(), std.end(), 1.0);
      }
      mean_ = mean;
      std_ = std;
    } else if (op_name == "Resize") {
      bool keep_ratio = op["keep_ratio"].as<bool>();
      auto target_size = op["target_size"].as<std::vector<int>>();
      int interp = op["interp"].as<int>();
      FDASSERT(target_size.size() == 2,
               "Require size of target_size be 2, but now it's %lu.",
               target_size.size());
      if (!keep_ratio) {
        int width = target_size[0];
        int height = target_size[1];
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
      // Do nothing, do permute as the last operation
      has_permute = true;
      continue;
    } else {
      FDERROR << "Unexcepted preprocess operator: " << op_name << "."
              << std::endl;
      return false;
    }
  }
  if (!disable_permute_) {
    if (has_permute) {
      // permute = cast<float> + HWC2CHW
      processors_.push_back(std::make_shared<Cast>("float"));
      processors_.push_back(std::make_shared<HWC2CHW>());
    }
  }

  input_k_data_ = cfg["k_data"].as<std::vector<float>>();
  return true;
}

bool PetrPreprocessor::Apply(FDMatBatch* image_batch,
                             std::vector<FDTensor>* outputs) {
  if (image_batch->mats->empty()) {
    FDERROR << "The size of input images should be greater than 0."
            << std::endl;
    return false;
  }
  if (!initialized_) {
    FDERROR << "The preprocessor is not initialized." << std::endl;
    return false;
  }
  // There are 3 outputs, image, k_data, timestamp
  outputs->resize(3);
  int batch = static_cast<int>(image_batch->mats->size());

  // Allocate memory for k_data
  (*outputs)[1].Resize({1, batch, 4, 4}, FDDataType::FP32);

  // Allocate memory for image_data
  (*outputs)[0].Resize({1, batch, 3, 320, 800}, FDDataType::FP32);

  // Allocate memory for timestamp
  (*outputs)[2].Resize({1, batch}, FDDataType::FP32);

  auto* image_ptr = reinterpret_cast<float*>((*outputs)[0].MutableData());

  auto* k_data_ptr = reinterpret_cast<float*>((*outputs)[1].MutableData());

  auto* timestamp_ptr = reinterpret_cast<float*>((*outputs)[2].MutableData());

  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat* mat = &(image_batch->mats->at(i));
    for (size_t j = 0; j < processors_.size(); ++j) {
      if (!(*(processors_[j].get()))(mat)) {
        FDERROR << "Failed to processs image:" << i << " in "
                << processors_[j]->Name() << "." << std::endl;
        return false;
      }
      if (processors_[j]->Name() == "Resize") {
        // crop and normalize after Resize
        auto img = *(mat->GetOpenCVMat());
        cv::Mat crop_img = img(cv::Range(130, 450), cv::Range(0, 800));
        Normalize(&crop_img, mean_, std_, scale_);
        FDMat fd_mat = WrapMat(crop_img);
        image_batch->mats->at(i) = fd_mat;
      }
    }
  }

  for (int i = 0; i < batch / 2 * 4 * 4; ++i) {
    input_k_data_.emplace_back(input_k_data_[i]);
  }

  memcpy(k_data_ptr, input_k_data_.data(), batch * 16 * sizeof(float));

  std::vector<float> timestamp(batch, 0.0f);
  for (int i = batch / 2; i < batch; ++i) {
    timestamp[i] = 1.0f;
  }
  memcpy(timestamp_ptr, timestamp.data(), batch * sizeof(float));

  FDTensor* tensor = image_batch->Tensor();
  (*outputs)[0].SetExternalData(tensor->Shape(), tensor->Dtype(),
                                tensor->Data(), tensor->device,
                                tensor->device_id);
  return true;
}

void PetrPreprocessor::Normalize(cv::Mat* im, const std::vector<float>& mean,
                                 const std::vector<float>& std, float& scale) {
  if (scale) {
    (*im).convertTo(*im, CV_32FC3, scale);
  }
  for (int h = 0; h < im->rows; h++) {
    for (int w = 0; w < im->cols; w++) {
      im->at<cv::Vec3f>(h, w)[0] =
          (im->at<cv::Vec3f>(h, w)[0] - mean[0]) / std[0];
      im->at<cv::Vec3f>(h, w)[1] =
          (im->at<cv::Vec3f>(h, w)[1] - mean[1]) / std[1];
      im->at<cv::Vec3f>(h, w)[2] =
          (im->at<cv::Vec3f>(h, w)[2] - mean[2]) / std[2];
    }
  }
}

}  // namespace perception
}  // namespace vision
}  // namespace fastdeploy
