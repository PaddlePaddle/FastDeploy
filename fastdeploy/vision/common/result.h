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
#include "fastdeploy/fastdeploy_model.h"
#include "opencv2/core/core.hpp"

namespace fastdeploy {
namespace vision {
enum ResultType { UNKNOWN, CLASSIFY, DETECTION, SEGMENTATION };

struct FASTDEPLOY_DECL BaseResult {
  ResultType type = ResultType::UNKNOWN;
};

struct FASTDEPLOY_DECL ClassifyResult : public BaseResult {
  std::vector<int32_t> label_ids;
  std::vector<float> scores;
  ResultType type = ResultType::CLASSIFY;

  void Clear();
  std::string Str();
};

struct FASTDEPLOY_DECL DetectionResult : public BaseResult {
  // box: xmin, ymin, xmax, ymax
  std::vector<std::array<float, 4>> boxes;
  std::vector<float> scores;
  std::vector<int32_t> label_ids;
  ResultType type = ResultType::DETECTION;

  DetectionResult() {}
  DetectionResult(const DetectionResult& res);

  void Clear();

  void Reserve(int size);

  void Resize(int size);

  void Sort();

  std::string Str();
};

} // namespace vision
} // namespace fastdeploy
