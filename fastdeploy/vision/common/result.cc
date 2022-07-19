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
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {

void ClassifyResult::Clear() {
  std::vector<int32_t>().swap(label_ids);
  std::vector<float>().swap(scores);
}

std::string ClassifyResult::Str() {
  std::string out;
  out = "ClassifyResult(\nlabel_ids: ";
  for (size_t i = 0; i < label_ids.size(); ++i) {
    out = out + std::to_string(label_ids[i]) + ", ";
  }
  out += "\nscores: ";
  for (size_t i = 0; i < label_ids.size(); ++i) {
    out = out + std::to_string(scores[i]) + ", ";
  }
  out += "\n)";
  return out;
}

DetectionResult::DetectionResult(const DetectionResult& res) {
  boxes.assign(res.boxes.begin(), res.boxes.end());
  scores.assign(res.scores.begin(), res.scores.end());
  label_ids.assign(res.label_ids.begin(), res.label_ids.end());
}

void DetectionResult::Clear() {
  std::vector<std::array<float, 4>>().swap(boxes);
  std::vector<float>().swap(scores);
  std::vector<int32_t>().swap(label_ids);
}

void DetectionResult::Reserve(int size) {
  boxes.reserve(size);
  scores.reserve(size);
  label_ids.reserve(size);
}

void DetectionResult::Resize(int size) {
  boxes.resize(size);
  scores.resize(size);
  label_ids.resize(size);
}

std::string DetectionResult::Str() {
  std::string out;
  out = "DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]\n";
  for (size_t i = 0; i < boxes.size(); ++i) {
    out = out + std::to_string(boxes[i][0]) + "," +
          std::to_string(boxes[i][1]) + ", " + std::to_string(boxes[i][2]) +
          ", " + std::to_string(boxes[i][3]) + ", " +
          std::to_string(scores[i]) + ", " + std::to_string(label_ids[i]) +
          "\n";
  }
  return out;
}

void SegmentationResult::Clear() {
  std::vector<std::vector<int64_t>>().swap(masks);
}

void SegmentationResult::Resize(int64_t height, int64_t width) {
  masks.resize(height, std::vector<int64_t>(width));
}

std::string SegmentationResult::Str() {
  std::string out;
  out = "SegmentationResult(\nImage masks: ";
  for (size_t i = 0; i < masks.size(); ++i) {
    for (size_t j = 0; j < masks[0].size(); ++j) {
      out = out + std::to_string(masks[i][j]) + ", ";
    }
  }
  out += "\n)";
  return out;
}

}  // namespace vision
}  // namespace fastdeploy
