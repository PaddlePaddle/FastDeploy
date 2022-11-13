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

#include "fastdeploy_jni/assets_loader_jni.h"

namespace fastdeploy {
namespace jni {

/// Assets loader
bool AssetsLoader::detection_labels_loaded_ = false;
bool AssetsLoader::classification_labels_loaded_ = false;
std::vector<std::string> AssetsLoader::detection_labels_ = {};
std::vector<std::string> AssetsLoader::classification_labels_ = {};

bool AssetsLoader::IsDetectionLabelsLoaded() {
  return detection_labels_loaded_;
}

bool AssetsLoader::IsClassificationLabelsLoaded() {
  return classification_labels_loaded_;
}

const std::vector<std::string>& AssetsLoader::GetDetectionLabels() {
  return detection_labels_;
}

const std::vector<std::string>& AssetsLoader::GetClassificationLabels() {
  return classification_labels_;
}

void AssetsLoader::LoadClassificationLabels(const std::string& path,
                                            bool force_reload) {
  if (force_reload || (!classification_labels_loaded_)) {
    classification_labels_loaded_ =
        LoadLabelsFromTxt(path, &classification_labels_);
  }
}

void AssetsLoader::LoadDetectionLabels(const std::string& path,
                                       bool force_reload) {
  if (force_reload || (!detection_labels_loaded_)) {
    detection_labels_loaded_ = LoadLabelsFromTxt(path, &detection_labels_);
  }
}

bool AssetsLoader::LoadLabelsFromTxt(const std::string& txt_path,
                                     std::vector<std::string>* labels) {
  labels->clear();
  std::ifstream file;
  file.open(txt_path);
  if (!file.is_open()) {
    return false;
  }
  while (file) {
    std::string line;
    std::getline(file, line);
    if (!line.empty() && line != "\n") {
      labels->push_back(line);
    }
  }
  file.clear();
  file.close();
  return !labels->empty();
}

}  // namespace jni
}  // namespace fastdeploy