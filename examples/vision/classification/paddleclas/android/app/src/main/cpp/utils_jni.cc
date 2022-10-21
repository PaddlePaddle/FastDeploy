//
// Created by qiuyanjun on 2022/10/19.
//

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

#include "utils_jni.h"

namespace fastdeploy {
namespace jni {

// Assets Loader Utils.
bool AssetsLoaderUtils::detection_labels_loaded_ = false;
bool AssetsLoaderUtils::classification_labels_loaded_ = false;
std::vector<std::string> AssetsLoaderUtils::detection_labels_ = {};
std::vector<std::string> AssetsLoaderUtils::classification_labels_ = {};

bool AssetsLoaderUtils::IsDetectionLabelsLoaded() {
  return detection_labels_loaded_;
}

bool AssetsLoaderUtils::IsClassificationLabelsLoaded() {
  return classification_labels_loaded_;
}

const std::vector<std::string>& AssetsLoaderUtils::GetDetectionLabels() {
  return detection_labels_;
}

const std::vector<std::string>& AssetsLoaderUtils::GetClassificationLabels() {
  return classification_labels_;
}

void AssetsLoaderUtils::LoadClassificationLabels(const std::string& path,
                                                 bool force_reload) {
  if (force_reload || (!classification_labels_loaded_)) {
    classification_labels_loaded_ =
        LoadLabelsFromTxt(path, &classification_labels_);
  }
}

void AssetsLoaderUtils::LoadDetectionLabels(const std::string& path,
                                            bool force_reload) {
  if (force_reload || (!detection_labels_loaded_)) {
    detection_labels_loaded_ = LoadLabelsFromTxt(path, &detection_labels_);
  }
}

bool AssetsLoaderUtils::LoadLabelsFromTxt(const std::string& txt_path,
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
  return labels->size() > 0;
}

}  // namespace jni
}  // namespace fastdeploy