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
#include <fstream>  // NOLINT
#include <string>   // NOLINT
#include <vector>   // NOLINT

namespace fastdeploy {
namespace jni {

/// Assets loader
class AssetsLoader {
public:
  static bool detection_labels_loaded_;
  static bool classification_labels_loaded_;
  static std::vector<std::string> detection_labels_;
  static std::vector<std::string> classification_labels_;

public:
  static bool IsDetectionLabelsLoaded();
  static bool IsClassificationLabelsLoaded();
  static const std::vector<std::string>& GetDetectionLabels();
  static const std::vector<std::string>& GetClassificationLabels();
  static void LoadClassificationLabels(const std::string& path,
                                       bool force_reload = false);
  static void LoadDetectionLabels(const std::string& path,
                                  bool force_reload = false);

private:
  static bool LoadLabelsFromTxt(const std::string& txt_path,
                                std::vector<std::string>* labels);
};

}  // namespace jni
}  // namespace fastdeploy
